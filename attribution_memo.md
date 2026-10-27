# 神经元级 attribution graph：推导、验证与集成方案（Qwen3.5-4B）

2026-10-26。对应代码：`explorer/attribution.py`（线性化与边计算），`sweep/check_attribution.py`（正确性检查）。

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

`sweep/check_attribution.py` 在一个随机初始化的 4 层 Qwen3.5（3 层 DeltaNet + 1 层全注意力，norm 权重非零）和真实 Qwen3.5-4B（float32，MPS，8 个 token 的 prompt）上运行，两者全部通过：

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

**(b) 冻结 sigmoid 在 $g<-1.278$ 时 gate 路径符号错误。** $\mathrm{silu}'(g)=\sigma(g)\,(1+g(1-\sigma(g)))$ 在 $g\approx-1.278$ 处过零，再往负方向就小于 0，但冻结后的系数 $\sigma(g)$ 始终为正。在 4B 这个 prompt 上，$|a|$ 最大的 1% 神经元中有 **15.9%** 处于这个区间，这正是 NOTES.md 里说的 $(-,-)$“双负回声”象限。对这些目标，经 gate 进入的边的符号与“把源加一点点、目标怎么变”相反；经 up 进入的边不受影响。建议：边分 gate / up 两部分存储，并给 $g<-1.278$ 的目标打标记。

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

**已实现**（`explorer/attribution.py` 的 `build_graph`，`explorer/runtime.py` 的 `attribution`，`POST /api/attribution`）：map3d 的 run 标签页里点“trace next token”（可在旁边的框里指定目标 token，默认取模型的 top-1），列表显示按影响力排序的神经元，地图上用 `setWeb(..., {flat: true})` 画出神经元→神经元的边（不同位置折叠到同一对神经元上）。剪枝沿用 §6 的第 4 步，另加 `max_edges`（默认 3000）。gate 小于 −1.278 的目标在悬停提示里标出。“Fact: the capital of the state containing Dallas is” → “ Austin”（p 0.78）：512 个候选直接贡献了 72% 的 logit，最强的节点是 L5 N8109（位于 “Dallas”，steer lens 指向 Texas），L22 / L28 / L31 在 “is” 位置推城市名。

### 6.1 干预验证：把 Texas 换成 California（`sweep/swap_state.py`、`sweep/swap_followup.py`）

只在 “ Dallas” 这一个位置上改 MLP 激活，后面的前向真实重算。

- **L5 N8109 是 Texas 城市神经元，但不是开关。** 负激活对应 Texas 城市：Dallas −2.84、Houston −2.28、San Antonio −5.59，Santa Fe −1.68；其他州的城市接近 0。把它置 0，Austin 只从 0.78 降到 0.69；把它的符号翻成 +2.84，答案变成 Atlanta 0.29、Austin 0.20，而不是 Sacramento。Texas 的信息分散在许多神经元上（Texas 组前 40 个全部置 0，Austin 仍有 0.41）。
- **California 这一侧的主杠杆是同层的 L5 N6203。** 它的负激活对应湾区城市：San Francisco −7.44，Oakland、Berkeley、San Jose 约 −2.3，LA、San Diego、Fresno 只有 −0.5 左右，其他州的城市接近 0。它的 steer lens 没有可读的词，单看 lens 是找不到它的，要靠 San Francisco prompt 的 attribution 图。
- **两个神经元就能换州。** N8109 置 0、N6203 设为 −14.9（San Francisco 值的 2 倍）后，Austin 0.010，Sacramento 0.53，San 0.29。把 San Francisco 图里 “ Francisco” 位置前 5 个神经元设为 2 倍的 San Francisco 值时，Austin 0.003、Sacramento 0.66。这 5 个里，除 N6203 外的 4 个单独改都几乎没有作用，去掉 N6203 后整组也没有作用（Austin 0.735）。把 L0–L3 的全部神经元从 San Francisco prompt 修补过来，Sacramento 0.72，这是上限。
- **改动的是“州”这个中间概念，而不是直接写入 “Sacramento”。** 同样的改动下：“Dallas is a city in the state of” 的回答从 Texas 0.99 变为 California 0.975；州长从 Greg（Abbott）变为 Jerry（Brown）0.27 和 Gavin 0.11；最大城市从 Houston 变为 San 0.63（贪心续写 “San Francisco”）。“Dallas Cowboys” 的语境不受影响；问答格式的 prompt 受影响较小（Texas 0.40→0.15，California 0.09，贪心续写仍是 Texas）。把这一改动放到 Chicago、Miami、Seattle、Boston 的城市位置上，Sacramento 都升到 0.53–0.71。
- 每个干预都画了干预后模型的 attribution 图：`sweep/plot_interventions.py` 输出到 `runs/attribution/interventions/`（69 张图，有 `index.html` 汇总表）。`build_graph(..., clamp={(layer, pos, n): value})` 把被干预的神经元作为没有输入的源节点（图中以橙色圈标出）。置 0 的神经元贡献为 0，所以不会出现在图里。在 fp32 下，线性化前向与真实的干预后前向在概率上一致到小数点后 4 位；bf16 下相差至多约 0.04，是舍入误差，真实 bf16 模型与 fp32 模型之间也差这么多。
- 换上去的更像是 “San Francisco 湾区” 而不是整个 California：候选里 “ San” 一直有 0.2–0.3，最大城市给出的是 San Francisco 而不是 Los Angeles。

## 7. 现有库调研

| 库 | 许可 | 内容 | 能否直接用 |
|---|---|---|---|
| [safety-research/circuit-tracer](https://github.com/safety-research/circuit-tracer)（decoderesearch） | MIT | 原文的开源实现：transcoder（PLT/CLT）、TransformerLens / nnsight 后端、剪枝、前端 | 不能直接用：以 transcoder 为核心，锁定 `transformers<=4.57.3`（我们的环境是 5.20.1）。**可以 vendor**：`graph.py` 里的 `normalize_matrix` / `compute_influence` / `prune_graph`，以及前端 |
| [TransluceAI/circuits](https://github.com/TransluceAI/circuits)（ADAG；论文 arXiv 2601.22594 “Language Model Circuits Are Sparse in the Neuron Basis”，2604.07615） | **仓库里没有 LICENSE 文件** | 正是神经元级 attribution graph，并改造了 circuit-tracer 前端。规则与本文相同：`StraightThroughRMSNorm`、`NoQKGradAttention`、`RelPGradMLP`（half 规则） | 不能直接用：只支持 Llama / Qwen3 模块（没有 DeltaNet、没有注意力输出门、norm 没有 $1+w$），依赖 torch 2.5.1、nnsight<0.4 和若干内部库。没有许可证，代码默认保留所有权利，只能作参考 |
| [NousResearch/neural-steering](https://github.com/NousResearch/neural-steering) | MIT | RelP 神经元归因、神经元之间的边、steering | 方法相同，不支持 Qwen3.5，可作参考 |
| [RollingRo11/relp-circuits](https://github.com/RollingRo11/relp-circuits) | — | ADAG 的复现（Llama / OLMo） | 同上 |

**建议**：线性化必须针对 Qwen3.5 自己写（大部分工作量在这里，现在的 `explorer/attribution.py` 约 200 行，已通过 §4 的检查）。剪枝和影响力计算从 circuit-tracer vendor（保留 MIT 版权头）。前端两条路：一是 circuit-tracer 前端（成熟，有 subgraph / supernode 交互），二是接进我们自己的 3D 地图（与 steering、lens、连接锥体联动）。最后选了第二条。剪枝没有 vendor，而是按同样的公式自己写的（`_influence`，用级数代替求逆）。

## 8. 下一步

- 画 completeness / replacement 曲线（候选数 $K$ 对“其余神经元”质量的影响）。
- gate / up 分开存储（目前只标记了 $g<-1.278$ 的目标）。
- 可选：导出 circuit-tracer 前端的 JSON。
- 拿 N2742（外星人神经元）在 “Who made you?” 上的 attribution 子图，与 steering 和现有的权重连接锥体对照。
