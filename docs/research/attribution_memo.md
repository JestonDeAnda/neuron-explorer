# 神经元级 attribution graph：推导、验证与集成方案（Qwen3.5-4B）

2026-10-26。对应代码：`src/neuron_explorer/attribution/graph.py`（线性化与边计算），`scripts/check_attribution.py`（正确性检查）。

## 结论

1. **数学上可行，而且比 transcoder 版本更“干净”**：在 MLP 神经元基底下，每层 MLP 输出严格等于 $\sum_i a_i\,d_i$（$d_i$ 是 `down_proj` 的第 $i$ 列），Qwen3.5 又没有任何 bias，所以冻结非线性之后，**没有误差节点（error node），completeness 恒等于 1**。Anthropic 原文也把“MLP neurons”列为可替换 CLT 的选项（“alternatives like per-layer transcoders or even MLP neurons can be used instead (keeping the same steps 3–8)”），并用“thresholded neurons”作为基线。
2. **代价是稀疏性和可解释性**：神经元是稠密激活、多义的。原文在所有指标上都发现 thresholded neurons 明显劣于 CLT/PLT 的 Pareto 前沿，路径也更长。所以剪枝是必需的，而不是可选项。
3. **Qwen3.5 特有的难点在线性注意力（Gated DeltaNet）层**：32 层里 24 层是 DeltaNet。冻结 $q,k,\beta$、衰减率和门控之后，$v\mapsto o$ 是严格线性的，相当于一个“冻结的注意力矩阵”，可以和普通注意力用同样方式处理（推导见 §2.3）。
4. **SwiGLU 必须选一个分账规则**。半分规则（half rule，即 RelP/Shapley）满足 completeness。但它的边 **不等于** 单个源的消融效应，在一阶近似下约差 2 倍；而在 $g<-1.278$ 区间，冻结 sigmoid 给出的 gate 路径符号与局部导数相反。在 Qwen3.5-4B 激活最强的 1% 神经元里，有 15.9% 落在这个区间（§5）。
5. **现有库无法直接装进来用**。没有任何库支持 Qwen3.5 的混合架构。最接近的 Transluce ADAG 用的是同一套数学（我核对过它的代码），但仓库没有 LICENSE，并且锁死了旧版依赖。建议：线性化部分用我们自己的实现（约 200 行，已验证）；剪枝和影响力计算以及前端可以 vendor 进 circuit-tracer 的代码（MIT 许可）。见 §7。

---

## 1. 图的定义

对一个 prompt（长度 $T$）：

- **神经元节点** $(\ell,t,i)$：第 $\ell$ 层、位置 $t$、神经元 $i$，激活 $a = \mathrm{silu}(g)\,u$，其中 $g = W_g[i]\cdot\hat x_{\ell,t}$，$u = W_u[i]\cdot\hat x_{\ell,t}$，$\hat x$ 是 `post_attention_layernorm` 之后的 MLP 输入。共有 $32\times9216\times T$ 个。
- **embedding 节点** $(t)$：token embedding $e_t$。
- **logit 节点** $(t,v)$：去均值的 logit $\ell_v - \frac1{|V|}\sum_w \ell_w$（与 circuit-tracer 一致）。
- **边** $A_{s\to t}$：在“局部替换模型”里 $s$ 对 $t$ 的 **直接** 贡献，形如 $A_{s\to t}=a_s\,w_{s\to t}$，$w$ 是虚拟权重（virtual weight）。

## 2. 线性化：冻结什么

原文的做法是：冻结注意力模式和归一化分母，让 transcoder 特征“跨过”MLP 非线性，于是特征对特征的直接作用是线性的。在神经元基底下，“跨过 MLP”这一步由神经元本身完成：每个神经元的激活是一个节点，它的输出 $a_i d_i$ 线性地写回残差流；输入一侧的非线性只在计算 **目标** 的分账时出现（§3）。

### 2.1 归一化

所有 RMSNorm（`input_layernorm`、`post_attention_layernorm`、最终 `norm`、注意力的 `q_norm`/`k_norm`）都冻结分母：
$$\mathrm{RMSNorm}(x) = (1+w)\odot x\cdot \underbrace{\mathrm{rsqrt}(\overline{x^2}+\epsilon)}_{\text{冻结}}.$$
注意 Qwen3.5 的 RMSNorm 是零中心化的（缩放系数是 $1+w$）。ADAG 的 `StraightThroughRMSNorm` 直接乘 `norm.weight`，用在 Qwen3.5 上会算错。

### 2.2 全注意力层（每 4 层一层：L3, L7, …）

冻结 softmax 模式 $P$ 和 Qwen3.5 特有的输出门 $\sigma(\text{gate})$：
$$o_t = \sigma(\text{gate}_t)\odot\sum_{\tau\le t} P_{t\tau}\,W_V\hat x_\tau \quad(\text{只剩 OV 路径，对 }\hat x\text{ 线性}).$$

### 2.3 Gated DeltaNet 层（其余 24 层）

状态递推（每个 value head）：
$$S_t = S_{t-1}\,e^{\gamma_t}\big(I-\beta_t k_t k_t^\top\big) + \beta_t\,k_t v_t^\top,\qquad o_t = S_t^\top q_t .$$
展开后
$$o_t=\sum_{\tau\le t} \underbrace{\beta_\tau\, q_t^\top\Big[\textstyle\prod_{r=\tau+1}^{t} e^{\gamma_r}(I-\beta_r k_r k_r^\top)\Big]^{\!\top}\! k_\tau}_{M_{t\tau}\ (\text{只依赖 } q,k,\beta,\gamma)}\; v_\tau .$$
所以只要冻结 $q,k,\beta,\gamma$，$o$ 就是 $v$ 的线性函数，$M$ 的角色等同于注意力模式。其余非线性：

- 短卷积后的 SiLU：$\mathrm{silu}(c)=c\,\sigma(c)$，冻结 $\sigma(c)$。由于卷积本身是线性的，$v$ 对输入是线性的。
- 输出的 gated RMSNorm：冻结分母和 $\mathrm{silu}(z)$ 门。

我检查过：Qwen3.5 的配置里 `attention_bias=False`、`conv1d` 的 `bias=False`、MLP 没有 bias，`dt_bias` 只出现在已冻结的 $\gamma$ 里。因此残差流里 **没有任何常数项**。

### 2.4 结论：精确分解

记 $J_{s\to(\ell,t)}$ 为“从源写入残差流的位置 / 层，到目标 MLP 输入 $\hat x_{\ell,t}$”的冻结线性映射。它是经过残差连接以及各层冻结注意力 / DeltaNet 的所有线性路径之和，不经过任何 MLP。于是
$$\hat x_{\ell,t} = \sum_{s:\ \ell_s<\ell,\ t_s\le t} a_s\,J_{s\to(\ell,t)}\,d_s \;+\; \sum_{\tau\le t} J_{\tau\to(\ell,t)}\,e_\tau \qquad(\text{严格等式}).$$
同层 MLP 不能作为源，因为它写在 $\hat x_{\ell}$ 之后；注意力是“透明”的输运，本身不是节点。

## 3. SwiGLU 目标的分账规则

$g = \sum_s g^{(s)}$，$u=\sum_s u^{(s)}$，其中 $g^{(s)} = a_s\,W_g[i]^\top J_{s\to t}d_s$，$u^{(s)}$ 同理。目标激活 $a=\sigma(g)\,g\,u$ 对源是 **双线性** 的，需要一个规则把它拆成对每个源线性的份额。冻结 $\sigma(g)$ 后有一个单参数族，每一个都满足 completeness：

| 规则 | 边 $A_{s\to t}$ | $\sum_s A_{s\to t}$ |
|---|---|---|
| up（整个 gate 冻结，ADAG 的 `StopGradGateMLP`） | $\mathrm{silu}(g)\,u^{(s)}$ | $\mathrm{silu}(g)\,u = a$ |
| gate（up 冻结） | $\sigma(g)\,u\,g^{(s)}$ | $\sigma(g)\,u\,g = a$ |
| **half（默认，= RelP = ADAG 的 `RelPGradMLP`）** | $\tfrac12\sigma(g)\big(g^{(s)}u + g\,u^{(s)}\big)$ | $\tfrac12\sigma(g)(gu+gu)=a$ |

half 规则就是双线性项 $g^{(s)}u^{(s')}$ 的 Shapley 分配：交叉项两个源各分一半，对角项 $g^{(s)}u^{(s)}$ 整个归 $s$。它也等于从零基线出发的 integrated gradients。half 是 up 和 gate 两条规则的平均，所以 **建议把 gate 份额和 up 份额分开存**，界面上可以显示“这条边是走 gate 还是走 up”，这对本仓库关心的 SwiGLU 符号象限正好有用。

虚拟权重（half 规则）：
$$w_{s\to t} = \tfrac12\sigma(g_t)\Big(u_t\,W_g[i]^\top J_{s\to t}d_s + g_t\,W_u[i]^\top J_{s\to t}d_s\Big),\qquad A_{s\to t}=a_s w_{s\to t}.$$
logit 目标：$A_{s\to(t,v)} = a_s\,(W_U[v]-\bar W_U)^\top N_{\text{final}}\,J_{s\to t}d_s$，其中 $N_{\text{final}}$ 是冻结的最终 norm。

**实现**：不显式构造 $J$。前向时把上面所有冻结量 `detach()`，把每层 `down_proj` 的输入换成一个叶子张量 $a$（梯度不再穿过 MLP 往回走，所以只剩直接效应），然后对目标做一次反向，就得到 $w_{s\to t}$；$a_s\cdot\partial(\text{目标})/\partial a_s$ 即边。把同一 prompt 复制 $B$ 份、每份放一个目标，一次反向就能算出 $B$ 行邻接矩阵。

**“total”模式**（RelP）：MLP 不截断，SiLU 用冻结的 $\sigma(g)$ 乘 $g$，$g\cdot u$ 用 half 规则反传。对 logit 只做一次反向，就得到每个神经元经所有路径的线性化总贡献，它等于线性图里所有路径的带符号乘积之和。这一步用来选候选节点（§6）。

## 4. 正确性检查

`scripts/check_attribution.py` 在一个随机初始化的 4 层 Qwen3.5（3 层 DeltaNet + 1 层全注意力，norm 权重非零）和真实 Qwen3.5-4B（float32，MPS，8 个 token 的 prompt）上运行，两者全部通过：

| 检查 | 含义 | tiny | 4B |
|---|---|---|---|
| 前向不变 | 打补丁的模型 logits 与原模型一致（相对误差） | 2.6e-6 | 3.1e-6 |
| 神经元 completeness（half / up / gate） | $\sum_s A_{s\to t}=a_t$，6 个跨层目标 | ≤5.2e-7 | ≤2.5e-6 |
| 因果性 | 目标同层及更深层的边恒为 0 | 0 | 0 |
| logit completeness | $\sum_s A_{s\to\text{logit}}$ = 去均值 logit | 1.8e-7 | 5.5e-7 |
| 线性（冻结模型回放） | 回放所有冻结量和其余神经元激活，把最强源置 0，目标线性化值的变化 = −边 | 6.4e-7 | 8.4e-6 |
| total 模式守恒 | embedding 的总贡献 = logit | 1.8e-7 | 1.9e-6 |

completeness 检查对漏冻结的非线性很敏感：只要有一个非线性忘了冻结（比如 conv 后的 SiLU），图就不再对源齐次一次，求和就会偏离目标。回放检查则直接验证“边 = 冻结模型里的因果效应”。

## 5. 解读时必须记住的偏差

**(a) 边不等于真实消融效应。** 冻结模型里，把源 $s$ 置 0 使目标变化 $-\sigma(g)\,(g^{(s)}u+g\,u^{(s)}-g^{(s)}u^{(s)})$，一阶近似约为 $-2A_{s\to t}$；half 规则的边只记一半，这是满足 completeness 的代价（双线性函数的欧拉齐次性：梯度 × 输入之和等于 $2a$）。在真实模型（不冻结）里把最强源置 0 的结果：

| 目标 ← 源 | $a_t$ | 边 | 真实 $\Delta a_t$ | 比值 $-\Delta/\text{边}$ |
|---|---|---|---|---|
| L1 N348 ← L0 N353（t0，sink 位置） | −0.68 | −0.058 | +0.124 | 2.1 |
| L7 N8187 ← L5 N8630 | −1.20 | −0.254 | +0.257 | 1.0 |
| L13 N2554 ← L10 N8507 | 1.66 | 0.208 | −0.291 | 1.4 |
| L19 N3520 ← L17 N1930 | −3.79 | −0.301 | +0.488 | 1.6 |
| L25 N545 ← L24 N121 | −6.67 | −0.516 | +0.936 | 1.8 |
| L31 N837 ← L22 N1373 | −22.5 | −0.724 | −0.206 | **−0.3（符号相反）** |

近距离的边，比值落在预期的 1–2 之间；跨 9 层的边（L22→L31）被中间层的连锁反应和冻结掉的注意力变化压过去，符号都反了。这与原文的说法一致：图里的边是假设，需要干预实验来验证；直接边在冻结模型里几乎必然成立，真正要检验的是“连锁效应”（knock-on effects）。

**(b) 冻结 sigmoid 在 $g<-1.278$ 时 gate 路径符号错误。** $\mathrm{silu}'(g)=\sigma(g)\,(1+g(1-\sigma(g)))$ 在 $g\approx-1.278$ 处过零，再往负方向就小于 0，但冻结后的系数 $\sigma(g)$ 始终为正。在 4B 这个 prompt 上，$|a|$ 最大的 1% 神经元中有 **15.9%** 处于这个区间，这正是 [source notes](notes.md) 里说的 $(-,-)$“双负回声”象限。对这些目标，经 gate 进入的边的符号与“把源加一点点、目标怎么变”相反；经 up 进入的边不受影响。建议：边分 gate / up 两部分存储，并给 $g<-1.278$ 的目标打标记。

**(c) 看不见的部分。** 冻结掉的东西都不在图里：注意力的 QK（谁看谁），以及 DeltaNet 中由数据决定的 $k$、$\beta$、衰减率和门（写入强度、遗忘速度）。DeltaNet 层占 3/4，这一块可能比纯注意力模型更大。

**(d) 稠密。** 每个神经元都非零，$32\times9216\times T$ 个节点，每个目标的入边分散在大量小贡献上。剪掉的源之和应当显式作为每个目标的“其余神经元”节点保留。它在剪枝后起到原文中误差节点的作用，用来报告 completeness / replacement 分数。

**(e) 与现有 `connections.npz` 的关系。** 现有的权重连接 $P[B,A]=(W_g[B]\cdot v_A)(W_u[B]\cdot v_A)$ 是与上下文无关的二次“自作用”项（假如 $A$ 是 $B$ 唯一的输入，$B$ 会被激活多少）。attribution 边则是在具体上下文里、对源线性的那一项。同位置、只走残差的那部分，恰好是 $\tfrac12\sigma(g_B)\big(u_B\,W_g[B]+g_B\,W_u[B]\big)\cdot N\,d_A$ 乘以 $a_A$（$N$ 是冻结的 norm）。所以 attribution 图可以看作现有静态连接的“上下文版本”，两者可以并排显示。

## 6. 建图算法（建议）

1. **前向**：线性化模式，一个 prompt。
2. **选候选节点**：total 模式对目标 logit 做一次反向，得到全部 $L\times T\times N$ 个神经元的线性化总贡献；按 $|\cdot|$ 累计质量取前 $K$ 个（例如 $K$=2k–8k，或累计 90%），加上全部 embedding。
3. **算边**：直接模式，把候选按 $B$ 个一批作为目标（每批复制 $B$ 份 prompt），每批一次反向，只收集候选源的边，其余归入“其余神经元”。显存只需要 $K\times K$ 的矩阵。目标侧同时存 gate 和 up 两份。
4. **剪枝**：沿用原文 / circuit-tracer 的方法：$|A|$ 按行归一化，$B=(I-A)^{-1}-I$，按 logit 影响力做节点剪枝（默认 0.8）和边剪枝（0.98）。
5. **显示**：导出成 circuit-tracer 前端的 JSON，或者直接画进我们的 3D 地图（`/api/web` 已经是“源层、源神经元、目标层、目标神经元、带符号强度、跳数”的二进制格式，只要再加一个位置维度）。
6. **验证**：对子图里的关键边做真实干预（现有的 `/api/agent/activations` 已支持推神经元 / 消融层或 head），区分直接效应和连锁效应。

成本粗估：每批反向的代价约等于一次带梯度的前向加反向；在 Mac MPS 上，上面整套检查（十几次前向 / 反向，$B\le6$，$T=8$）连同加载权重共 30 秒。$K$=4000、$B$=16 时需要 250 批，在 4090 上应在分钟量级；更长的 prompt 代价线性增长。

实测（M 系列 Mac，MPS，bf16，Dallas prompt，$T\approx12$，$B$=16）：$K$=256 约 18 秒，$K$=512 约 28 秒（经 `/api/attribution`），$K$=1024 约 60 秒。bf16 与 fp32 建出的图前 12 个节点相同，影响力相差约 2%。

**已实现**（`src/neuron_explorer/attribution/graph.py` 的 `build_graph`，`src/neuron_explorer/explorer/runtime.py` 的 `attribution`，`POST /api/attribution`）：map3d 的 run 标签页里点“trace next token”（可在旁边的框里指定目标 token，默认取模型的 top-1），列表显示按影响力排序的神经元，地图上用 `setWeb(..., {flat: true})` 画出神经元→神经元的边（不同位置折叠到同一对神经元上）。剪枝沿用 §6 的第 4 步，另加 `max_edges`（默认 3000）。gate 小于 −1.278 的目标在悬停提示里标出。“Fact: the capital of the state containing Dallas is” → “ Austin”（p 0.78）：512 个候选直接贡献了 72% 的 logit，最强的节点是 L5 N8109（位于 “Dallas”，steer lens 指向 Texas），L22 / L28 / L31 在 “is” 位置推城市名。

### 6.1 干预验证：把 Texas 换成 California（`scripts/swap_state.py`、`scripts/swap_followup.py`）

只在 “ Dallas” 这一个位置上改 MLP 激活，后面的前向真实重算。

- **L5 N8109 是 Texas 城市神经元，但不是开关。** 负激活对应 Texas 城市：Dallas −2.84、Houston −2.28、San Antonio −5.59，Santa Fe −1.68；其他州的城市接近 0。把它置 0，Austin 只从 0.78 降到 0.69；把它的符号翻成 +2.84，答案变成 Atlanta 0.29、Austin 0.20，而不是 Sacramento。Texas 的信息分散在许多神经元上（Texas 组前 40 个全部置 0，Austin 仍有 0.41）。
- **California 这一侧的主杠杆是同层的 L5 N6203。** 它的负激活对应湾区城市：San Francisco −7.44，Oakland、Berkeley、San Jose 约 −2.3，LA、San Diego、Fresno 只有 −0.5 左右，其他州的城市接近 0。它的 steer lens 没有可读的词，单看 lens 是找不到它的，要靠 San Francisco prompt 的 attribution 图。
- **两个神经元就能换州。** N8109 置 0、N6203 设为 −14.9（San Francisco 值的 2 倍）后，Austin 0.010，Sacramento 0.53，San 0.29。把 San Francisco 图里 “ Francisco” 位置前 5 个神经元设为 2 倍的 San Francisco 值时，Austin 0.003、Sacramento 0.66。这 5 个里，除 N6203 外的 4 个单独改都几乎没有作用，去掉 N6203 后整组也没有作用（Austin 0.735）。把 L0–L3 的全部神经元从 San Francisco prompt 修补过来，Sacramento 0.72，这是上限。
- **改动的是“州”这个中间概念，而不是直接写入 “Sacramento”。** 同样的改动下：“Dallas is a city in the state of” 的回答从 Texas 0.99 变为 California 0.975；州长从 Greg（Abbott）变为 Jerry（Brown）0.27 和 Gavin 0.11；最大城市从 Houston 变为 San 0.63（贪心续写 “San Francisco”）。“Dallas Cowboys” 的语境不受影响；问答格式的 prompt 受影响较小（Texas 0.40→0.15，California 0.09，贪心续写仍是 Texas）。把这一改动放到 Chicago、Miami、Seattle、Boston 的城市位置上，Sacramento 都升到 0.53–0.71。
- 每个干预都画了干预后模型的 attribution 图：`scripts/plot_interventions.py` 输出到 `runs/attribution/interventions/`（69 张图，有 `index.html` 汇总表）。`build_graph(..., clamp={(layer, pos, n): value})` 把被干预的神经元作为没有输入的源节点（图中以橙色圈标出）。置 0 的神经元贡献为 0，所以不会出现在图里。在 fp32 下，线性化前向与真实的干预后前向在概率上一致到小数点后 4 位；bf16 下相差至多约 0.04，是舍入误差，真实 bf16 模型与 fp32 模型之间也差这么多。
- 换上去的更像是 “San Francisco 湾区” 而不是整个 California：候选里 “ San” 一直有 0.2–0.3，最大城市给出的是 San Francisco 而不是 Los Angeles。

### 6.2 从发现电路到改写答案：成功经验与方法论

这组实验的目标不是直接提高 “ Sacramento” 的 logit，而是找到模型在 “Dallas” 位置写入的 **Texas 状态**，再用一个参考 prompt 的 **California 状态**替换它，观察后续计算是否自然地产生 Sacramento。实践表明，可靠的工作流应当是“对照 prompt → attribution 定位 → activation patching → 最小化 → 语义验证”，而不是从可读的 lens 词或单条大边直接猜一个神经元。

#### 6.2.1 先定义反事实，而不是先找神经元

1. 选一个基线 prompt，使原答案稳定且概率高：
   `Fact: the capital of the state containing Dallas is` → Austin 0.78。
2. 选一个结构和 token 位置尽量相似、但中间概念不同的参考 prompt：
   `Fact: the capital of the state containing San Francisco is` → Sacramento 0.70。
3. 把任务定义为：在城市 token 处，把基线运行的内部状态改成参考运行的内部状态，而保持其余 token 和计算不变。这样得到的是局部反事实，避免同时改变句法、长度和问题类型。
4. “San Francisco” 有两个 token，而 “Dallas” 只有一个；实验把参考状态取在最后一个城市 token “ Francisco”，写到 “ Dallas” 的位置。以后应优先选择 tokenization 对齐的实体，或明确记录采用首 token、末 token还是跨 token 聚合。

这种 framing 很重要：若直接优化 Sacramento，可能只找到输出端的“答案写入”神经元；通过 Texas→California 的状态替换，才有机会找到上游表示，并检验它是否对州长、最大城市等不同下游问题复用。

#### 6.2.2 用两张 attribution 图做差分候选选择

分别对基线和参考 prompt 建图，并只看实体位置、较早层（本实验主要是 L0–L12）的高影响力节点：

- Dallas 图给出“原状态”候选，其中 L5 N8109 最突出；
- San Francisco 图给出“目标状态”候选，其中 L5 N6203 最突出；
- 候选排序使用节点对各自目标 logit 的全路径 influence，而不是仅使用激活绝对值；
- 再比较候选在两个 prompt 同一语义位置的实际激活值，得到要写入的方向和尺度。

不能只依赖 steer/logit lens。L5 N8109 的负向 steer lens 明确含 Texas，因此容易发现；真正关键的 L5 N6203 的 lens 却是“作为、正如”等不可解释词。如果只搜索 “California” lens，会选中 L4 N5319；把它推到参考值的 3 倍，Austin 反而从 0.78 升到 0.82。**可读性是命名证据，不是因果筛选标准；上下文 attribution 和反事实激活差才是。**

#### 6.2.3 干预采用“定值修补”，而不是无标度 steering

每个候选使用 reference activation patching：

\[
a^{\mathrm{Dallas}}_{\ell,t,n}
\leftarrow s\,a^{\mathrm{SF}}_{\ell,t',n},
\]

其中 $s\in\{1,2,4\}$，而不是笼统地“加一个正剂量”。这样有三个优点：

1. 符号由参考运行决定；SwiGLU 神经元的语义极性不一定是正向；
2. 剂量有自然单位，$s=1$ 表示真实参考状态；
3. 可以比较单神经元、候选组和整层 patch，形成由稀疏到稠密的干预阶梯。

干预只发生在城市 token 的一个位置。对每次干预都完整重算后续前向，因此测到的是包括 knock-on effects 在内的真实因果结果，而不是线性化图里的预测。

#### 6.2.4 用消融和写入分开回答两个问题

“去掉 Texas 是否足够”和“写入 California 是否足够”是不同问题，应分开测：

- **必要性 / 消融：** N8109 置 0，Austin 0.78→0.69；Texas 组约 36 个候选全部置 0，Austin 仍为 0.41。说明 Texas 表示是分布式的，单个显著节点并非必要开关。
- **充分性 / 写入：** N6203 单独写为 $2\times$ 参考值后，Austin 与 Sacramento 都约 0.24；California 前 5 个一起写入后，Sacramento 达 0.66。说明目标状态的写入比原状态的删除更有效。
- **组合替换：** N8109 置 0，加 N6203 写为 $2\times$ 参考值，Sacramento 达 0.53。一个节点削弱旧状态，一个节点写入新状态，是最小且易解释的成功方案。

因此，大影响力节点应被看作“杠杆”而不是“概念的唯一载体”。消融失败不等于节点无因果作用；可能只是同义通路和冗余节点仍保留旧状态。

#### 6.2.5 用剂量网格和逐步稀疏化找最小电路

实验按以下顺序搜索：

1. 单节点置 0、设成参考值、符号翻转；
2. 目标组 top-$m$ 写入，$m\in\{5,10,20,40\}$；
3. 对每组扫描 $s\in\{1,2,4\}$；
4. 原组 top-$m$ 置 0，同时目标组 top-$m$ 写入；
5. 将 L0–L3、L0–L5、L0–L8、L0–L12 整体 patch，估计可达到的上限；
6. 从成功的 top-5 组逐个移除或单独测试节点，找真正承担效果的节点。

这里最关键的是第 6 步。California top-5 在 $2\times$ 时使 Sacramento 达 0.66，但其中四个节点单独几乎无效；移除 N6203 后，Austin 回到 0.735、Sacramento 仅 0.007。这说明“组干预成功”不能证明组内每个成员都重要，必须做 leave-one-out 和 single-node 对照。剂量也不是越大越好：从 $2\times$ 增至 $4\times$ 往往使 Sacramento 概率下降并增加 “ San”，表明过强写入会离开参考分布。

#### 6.2.6 成功判据必须超出目标答案

仅把 Sacramento 概率推高，不能区分“改了州表示”和“直接推了答案 token”。本实验使用四级验证：

1. **主任务：** Austin 显著下降，Sacramento 成为 top-1；
2. **中间概念：** “Dallas is a city in the state of” 从 Texas 0.99 变为 California 0.975；
3. **多后果一致性：** 州长由 Texas 人物转向 California 人物，最大城市由 Houston 转向 San Francisco；
4. **边界 / 负对照：** Dallas Cowboys 的回答基本不变；改变 prompt 格式后效果减弱，说明干预不是无条件的字符串替换。

还应把同一干预迁移到其他实体位置。Chicago、Miami、Seattle、Boston 都被推向 Sacramento，说明这组激活具有“把当前城市当成 California 城市”的通用作用；但输出同时偏向 San Francisco，说明它更精确地编码了“湾区 / San Francisco 型 California”，不能把实验结论夸大为纯粹、完整的 California 概念。

#### 6.2.7 每次干预后重新建图

原始 attribution 图只解释未干预模型。干预会改变后续神经元、注意力和目标 token，因此不能拿原图当作干预后的机制解释。`build_graph(..., clamp=...)` 在干预状态下重新冻结非线性并建图：

- clamped 神经元是外部注入的源节点，没有输入边；
- 图中用橙圈和 `:= value` 标记；
- 置 0 节点没有输出贡献，通常不会通过剪枝，因此必须在标题和元数据中记录；
- 图的目标是干预后真实模型的 top-1，而不是预先强制为 Sacramento；
- 图旁同时保存真实模型概率，避免把线性化概率当作干预结果。

这一步既用于解释成功案例，也用于发现失败模式。例如 N8109 符号翻转后的 top-1 是 Atlanta，重建的图应解释 Atlanta 电路，而不是仍然画 Austin 或 Sacramento。

#### 6.2.8 可复用的实验协议

对新的“答案 A→B”任务，可以复用以下协议：

1. 找到高置信的基线 prompt 和结构匹配的参考 prompt；
2. 运行两张 attribution 图，在关键实体位置取早中层 top-$K$ 节点；
3. 用激活差、influence 和跨样本选择性筛选候选；lens 只辅助命名；
4. 先测原状态候选的消融，再测目标状态候选的 reference patch；
5. 扫描组大小和剂量，保留真实模型的完整词表概率，不只看两个目标 token；
6. 对最佳组做 single-node、leave-one-out 和原状态关闭 + 目标状态写入；
7. 用中间概念、多种下游后果、不同模板和无关任务验证语义与边界；
8. 对成功和关键失败干预重新建 attribution 图；
9. 保存 prompt、token 位置、具体 clamp 值、dtype、完整 top-$k$、图 JSON 和 SVG/HTML，使实验可复现。

方法论上的核心结论是：**先用 attribution 找“在哪里计算”，再用参考状态 patch 确定“写什么值”，最后通过跨问题行为判断“改的是什么概念”。** 单个可解释神经元、单条大边或一次答案翻转，都不足以独立支持机制结论。

### 6.3 同一算式的三种写法：阿拉伯数字、英文、中文（`scripts/arith_forms.py`）

主算式 8 − 3 = 5，对照算式 2 × 4 = 8，各用三种写法，共 6 张图（$K$=1024），报告在 `runs/attribution/arith_forms.html`（单文件，内嵌 SVG），数据在 `runs/attribution/arith/`。

| 写法 | prompt | 答案 |
|---|---|---|
| 数字 | `Fact: 8-3=` | “5” 0.77 |
| 英文 | `Fact: eight minus three equals` | “ five” 0.96 |
| 中文 | `事实：八减三等于` | “五” 0.78 |

（数字写法若带空格 `8 - 3 =`，模型先输出一个空格 token，所以用了无空格形式。）

**对齐方式。** 三种写法的最后四个 token 依次是第一个操作数、运算符、第二个操作数、等号（`等于` / ` equals` / `=` 各是一个 token），之前的算作上下文。神经元按（层，编号，角色）匹配，节点影响力在每张图内归一化。对照算式用来区分“这道题特有的”和“任何算术 prompt 都有的”。

**结果。**

- **共享的规模。** 108 个神经元在 8 − 3 的三张图里都出现，占各图影响力的 36–43%；其中 106 个在三种写法下激活符号相同。108 个中 41 个也出现在 2 × 4 的三张图里（通用部分，多在上下文位置和 L0–L7），67 个只属于 8 − 3。
- **相似度结构。** 按影响力加权的重合度，同一写法、不同算式（数字 0.53，英文 0.54，中文 0.47）不低于同一算式、不同写法（英文–中文 0.53，数字–英文 0.41，数字–中文 0.34）。也就是说，图里约一半是由写法和 prompt 格式决定的；英文与中文彼此更接近，数字写法最独特。
- **共享在哪里。** 按层看，L0–L7 的共享比例最高（47–59%，主要是上下文位置的通用神经元），L8–L15 最低（9–20%，这一段本身影响力也很小），L16–L31 约 30%。按角色看，影响力 51–63% 集中在等号位置，其中约三分之一共享。**运算符位置没有一个共享神经元**：“-”、“minus”、“减”由不同的神经元读取（不过这个位置在各图里只占 1–5% 的影响力）。
- **答案以与写法无关的编码保存。** 等号位置共享的晚层神经元，steer lens 混合了答案的各种写法：L30 N1485（_five、 five、五是、五、 Five），L28 N1070（ five、五天、5），L29 N4856（5、５、 Five、五是），L29 N3642（5、五百、５、 five），L25 N1351、L24 N2499（ five、五、 пять）。L29 N154 的 lens 是一串数字词（Nine、Eight、Four、five），也出现在 2 × 4 的图里，像是通用的“答案是个数字词”神经元。操作数一侧也有跨写法的神经元：L0 N5795 在第一个操作数位置，lens 是 8 / eight / 八，第 0 层就已经把三种写法映射到同一个神经元。
- **消融（必要性）。** 把 78 个共享的非上下文神经元在各自位置置 0：数字 0.77→0.24，英文 0.96→0.32，中文 0.78→0.15。同样数量、同层同位置的随机神经元几乎没有影响（变化 <0.005）。同样数量的“本写法独有的最强神经元”置 0：数字 0.62，英文 0.78，中文 0.010——中文写法另有一条很强的专属通路。
- **写入（充分性）。** 把 8 − 3 中等号位置的 49 个共享晚层神经元（L20 以上、不在对照组里）设为 8 − 3 的激活值（2 倍）写进 2 × 4 的运行，答案变为 5：数字 P(5) 最高 0.87，英文 0.64，中文 0.84，原答案 8 都降到 ≤0.02。**取值的来源写法基本不重要**：英文的取值写进数字和中文分别得 0.87、0.84，比同写法自身（0.69、0.82）还好；唯一例外是用数字写法的取值写进英文，只得 0.07。1 倍剂量只能让 5 和 8 势均力敌。
- **反方向不对称。** 把这 67 个共享等号神经元设成 2 × 4 的值写进 8 − 3，只能把 P(5) 压到 0.28–0.67，P(8) 最多 0.085；把等号位置的全部 MLP 神经元都换掉，P(8) 才到 0.67–0.76。“写入一个答案”比“擦掉一个答案再换成另一个”容易，与 §6.1 中 California 写入比 Texas 删除更有效的结论一致。

**共同点（结论）。** 三种写法在入口（操作数 token）和出口（答案 token）处各有专属的神经元，但在等号位置的晚层（L20–L30）汇合到同一组“答案数值”神经元上。这些神经元的激活符号一致，lens 同时指向 5 / five / 五，并且其取值可以跨写法移植。运算符的读取则不共享。换句话说，算式在模型里是“各自读入、共享答案、各自拼写”。

**局限。** 只有一道主算式和一道对照算式，结论需要在更多算式（不同运算、不同答案）上重复。共享的判定依赖剪枝后的节点集合（node_threshold 0.8），阈值会影响共享数量。消融用的是置 0 而不是均值消融。数字写法因分词需要去掉空格，所以它的 token 结构与另两种不完全一致。

## 7. 现有库调研

| 库 | 许可 | 内容 | 能否直接用 |
|---|---|---|---|
| [safety-research/circuit-tracer](https://github.com/safety-research/circuit-tracer)（decoderesearch） | MIT | 原文的开源实现：transcoder（PLT/CLT）、TransformerLens / nnsight 后端、剪枝、前端 | 不能直接用：以 transcoder 为核心，锁定 `transformers<=4.57.3`（我们的环境是 5.20.1）。**可以 vendor**：`graph.py` 里的 `normalize_matrix` / `compute_influence` / `prune_graph`，以及前端 |
| [TransluceAI/circuits](https://github.com/TransluceAI/circuits)（ADAG；论文 arXiv 2601.22594 “Language Model Circuits Are Sparse in the Neuron Basis”，2604.07615） | **仓库里没有 LICENSE 文件** | 正是神经元级 attribution graph，并改造了 circuit-tracer 前端。规则与本文相同：`StraightThroughRMSNorm`、`NoQKGradAttention`、`RelPGradMLP`（half 规则） | 不能直接用：只支持 Llama / Qwen3 模块（没有 DeltaNet、没有注意力输出门、norm 没有 $1+w$），依赖 torch 2.5.1、nnsight<0.4 和若干内部库。没有许可证，代码默认保留所有权利，只能作参考 |
| [NousResearch/neural-steering](https://github.com/NousResearch/neural-steering) | MIT | RelP 神经元归因、神经元之间的边、steering | 方法相同，不支持 Qwen3.5，可作参考 |
| [RollingRo11/relp-circuits](https://github.com/RollingRo11/relp-circuits) | — | ADAG 的复现（Llama / OLMo） | 同上 |

**建议**：线性化必须针对 Qwen3.5 自己写（大部分工作量在这里，现在的 `src/neuron_explorer/attribution/graph.py` 约 200 行，已通过 §4 的检查）。剪枝和影响力计算从 circuit-tracer vendor（保留 MIT 版权头）。前端两条路：一是 circuit-tracer 前端（成熟，有 subgraph / supernode 交互），二是接进我们自己的 3D 地图（与 steering、lens、连接锥体联动）。最后选了第二条。剪枝没有 vendor，而是按同样的公式自己写的（`_influence`，用级数代替求逆）。

## 8. 下一步

- 画 completeness / replacement 曲线（候选数 $K$ 对“其余神经元”质量的影响）。
- gate / up 分开存储（目前只标记了 $g<-1.278$ 的目标）。
- 可选：导出 circuit-tracer 前端的 JSON。
- 拿 N2742（外星人神经元）在 “Who made you?” 上的 attribution 子图，与 steering 和现有的权重连接锥体对照。
