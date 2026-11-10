"""Neuron-level attribution for Qwen3.5: the model with every nonlinearity except the MLP neurons frozen, so that each
neuron's gate and up pre-activations, and each logit, are exactly linear in the activations of earlier neurons and the
token embeddings (Ameisen et al. 2025, "Circuit Tracing", with MLP neurons in place of transcoder features).

Frozen (the forward value is kept, the gradient is stopped):
- RMSNorm denominators (input, post-attention, final, q/k norms);
- full attention: softmax patterns and the sigmoid output gate; gradient flows through V and O only;
- Gated DeltaNet: q, k, beta and decay (so the token mix is a fixed per-head matrix acting on v), the sigmoid factor
  of the SiLU after the short conv, and the gated norm's denominator and silu(z) gate;
- each MLP neuron a = silu(g) * u is a leaf: its output reaches the residual stream through down_proj, nothing flows
  back through its own inputs (direct effects only), except in mode "total" (RelP: SiLU as sigmoid(g) * g with
  sigmoid frozen, half rule on g * u), where the whole linearized network is differentiated.

A target neuron's activation is split over sources with the half rule, a = 1/2 sigmoid(g) (g u + g u), linear in the
sources; summed over every source (earlier neurons at every earlier or equal position, plus embeddings) the edges
equal the target's activation exactly, and the same holds for a logit.

    with Linearized(lm, lm_head) as L:
        logits = L.forward(ids, copies=B)          # B copies of one prompt, one target per copy
        tgt = L.neuron_target([(layer, pos, n), ...])
        con = L.contributions(tgt)                 # {"mlp": [L, B, T, N], "embed": [B, T]}
"""
import types

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as q35


class _Shapley(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.saved_tensors
        return 0.5 * grad * y, 0.5 * grad * x


class Linearized:
    def __init__(self, lm, lm_head, mode="direct"):
        """lm: the Qwen3.5 text model (decoder layers, embed_tokens, norm); lm_head: the unembedding."""
        self.lm, self.lm_head, self.mode = lm, lm_head, mode
        self.cache, self.replay, self.freeze_mlp = {}, False, False
        self.inject = {}  # layer -> list of (pos, neuron, delta) added to that neuron's activation (all copies)
        self.clamp = {}  # layer -> (positions, neurons, values): activations set by an intervention; leaves, no inputs
        self.mlp = {}  # layer -> {"x", "g", "u", "a"}
        self._saved = []

    # ---------- patching ----------
    def __enter__(self):
        for m in self.lm.modules():
            if isinstance(m, (q35.Qwen3_5RMSNorm,)):
                self._patch(m, _norm_forward)
            elif isinstance(m, q35.Qwen3_5Attention):
                self._patch(m, _attn_forward)
            elif isinstance(m, q35.Qwen3_5GatedDeltaNet):
                self._patch(m, _deltanet_forward)
        for i, layer in enumerate(self.lm.layers):
            self._patch(layer.mlp, _mlp_forward, layer=i)
        return self

    def __exit__(self, *exc):
        for m, had, old in self._saved:
            if had:
                m.forward = old
            else:
                del m.forward
            del m._lin
        self._saved.clear()

    def _patch(self, m, fn, layer=None):
        self._saved.append((m, "forward" in m.__dict__, m.__dict__.get("forward")))
        m._lin, m._layer = self, layer
        m.forward = types.MethodType(fn, m)

    def frozen(self, m, key, value):
        k = (id(m), key)
        if self.replay:
            return self.cache[k]
        v = value.detach()
        self.cache[k] = v
        return v

    # ---------- forward ----------
    def forward(self, ids, copies=1, replay=False, freeze_mlp=False):
        """Runs the prompt (a list of token ids) as `copies` identical batch rows; returns logits [B, T, V] with their
        graph. replay=True reuses every frozen quantity from the last non-replay pass (and, with freeze_mlp, every
        neuron's activation too, plus `inject`), which is the frozen linear model itself."""
        self.replay, self.freeze_mlp = replay, freeze_mlp
        dev = self.lm.embed_tokens.weight.device
        ids_t = torch.tensor([ids] * copies, device=dev)
        emb = self.lm.embed_tokens(ids_t).detach().requires_grad_(True)
        self.embed = emb
        T = len(ids)
        pos = torch.arange(T, device=dev)[None].expand(copies, -1)
        cos_sin = self.lm.rotary_emb(emb, pos[None].expand(3, -1, -1))
        h = emb
        for layer in self.lm.layers:
            h = layer(h, position_embeddings=cos_sin, attention_mask=None, position_ids=pos)
        h = self.lm.norm(h)
        self.logits = self.lm_head(h)
        return self.logits

    # ---------- targets ----------
    def neuron_target(self, specs, rule="half"):
        """One target per batch copy: specs[b] = (layer, pos, neuron). rule "half" (default), "up" (gate frozen
        whole: silu(g) * u) or "gate" (up frozen: sigmoid(g) u * g); each sums to the activation over all sources."""
        out = []
        for b, (l, t, n) in enumerate(specs):
            st = self.mlp[l]
            x = st["x"][b, t]
            gl = x @ self.lm.layers[l].mlp.gate_proj.weight[n]
            ul = x @ self.lm.layers[l].mlp.up_proj.weight[n]
            g, u = st["g"][b, t, n].float(), st["u"][b, t, n].float()
            s = torch.sigmoid(g)
            if rule == "half":
                out.append(0.5 * s * (gl.float() * u + g * ul.float()))
            elif rule == "up":
                out.append(s * g * ul.float())
            else:
                out.append(s * u * gl.float())
        return torch.stack(out)

    def logit_target(self, specs, center=True):
        """specs[b] = (pos, token id); the logit, minus the mean logit over the vocabulary when center."""
        out = []
        for b, (t, v) in enumerate(specs):
            row = self.logits[b, t].float()
            out.append(row[v] - row.mean() if center else row[v])
        return torch.stack(out)

    def contributions(self, targets, keep=True):
        """Backward from one target per copy. Returns activation x gradient for every neuron ("mlp": [L, B, T, N])
        and every token embedding ("embed": [B, T]); in direct mode these are the edges into each target. keep=False
        frees the graph (no further backward passes on this forward)."""
        leaves = [self.mlp[l]["a"] for l in range(len(self.lm.layers))]
        grads = torch.autograd.grad(targets.sum(), leaves + [self.embed], allow_unused=True, retain_graph=keep)
        mlp = torch.stack([(g if g is not None else torch.zeros_like(a)).float() * a.detach().float()
                           for g, a in zip(grads[:-1], leaves)])
        embed = (grads[-1].float() * self.embed.detach().float()).sum(-1)
        return {"mlp": mlp, "embed": embed}


# ---------- graph ----------
def build_graph(lm, head, ids, token=None, k=1024, batch=16, node_threshold=0.8, edge_threshold=0.98, max_edges=None,
                clamp=None):
    """Attribution graph for the next token after `ids` (the top prediction, or token id `token`).

    1. One backward in total (RelP) mode ranks every neuron at every position by its linearized total effect on the
       centered logit; the k largest |effect| become the candidate nodes, with every token embedding.
    2. Direct mode, `batch` candidates per backward: each candidate's incoming edges from the other candidates and the
       embeddings (half rule); what the other neurons send it is kept as its `other` input.
    3. Pruning as in Ameisen et al.: |A| normalized over each node's inputs (other included), influence on the logit
       summed over all paths, nodes kept up to `node_threshold` of the total influence, then edges up to
       `edge_threshold` of the edge influence, at most `max_edges` (by edge influence) when given.

    Returns {"token", "prob", "nodes": [{kind, layer, pos, n, act, influence}], "edges": [[src, dst, weight]], stats}.
    kind is "neuron", "embed" (layer -1) or "logit" (layer = number of layers); edge weights are raw attributions.

    clamp: {(layer, pos, neuron): value}, activations set by an intervention for the whole run (the graph is of the
    intervened model). Clamped neurons are sources only, marked "clamped" in the nodes."""
    params = [p for p in list(lm.parameters()) + list(head.parameters()) if p.requires_grad]
    for p in params:
        p.requires_grad_(False)
    try:
        return _build_graph(lm, head, ids, token, k, batch, node_threshold, edge_threshold, max_edges, clamp or {})
    finally:
        for p in params:
            p.requires_grad_(True)


def _build_graph(lm, head, ids, token, k, batch, node_threshold, edge_threshold, max_edges, clamp):
    T, Lyr = len(ids), len(lm.layers)
    with Linearized(lm, head, mode="total") as L:
        for (l, t, n), v in clamp.items():
            c = L.clamp.setdefault(l, ([], [], []))
            c[0].append(t), c[1].append(n), c[2].append(float(v))
        logits = L.forward(ids)
        last = logits[0, T - 1].detach().float()
        tok = int(last.argmax()) if token is None else int(token)
        prob = float(torch.softmax(last, -1)[tok])
        tot = L.contributions(L.logit_target([(T - 1, tok)]), keep=False)
        score = tot["mlp"][:, 0]  # [L, T, N]
        N = score.shape[-1]
        flat = score.abs().flatten()
        k = min(k, flat.numel())
        top = flat.topk(k).indices
        cl, rest = top // (T * N), top % (T * N)
        ct, cn = rest // N, rest % N
        captured = float(flat[top].sum() / flat.sum())
        del tot, score, flat
        fixed = torch.tensor([(int(l), int(t), int(n)) in clamp for l, t, n in zip(cl, ct, cn)], device=cl.device)

        L.mode, L.cache = "direct", {}
        L.forward(ids, copies=batch)
        acts = torch.stack([L.mlp[l]["a"].detach()[0] for l in range(Lyr)]).float()[cl, ct, cn]
        gates = torch.stack([L.mlp[l]["g"][0] for l in range(Lyr)]).float()[cl, ct, cn]
        n_nodes = k + T + 1
        dev = acts.device
        A = torch.zeros(n_nodes, n_nodes, device=dev)
        other = torch.zeros(n_nodes, device=dev)
        targets = list(range(k)) + [k + T]
        for i in range(0, len(targets), batch):
            chunk = targets[i:i + batch]
            neurons = [j for j in chunk if j < k]
            tg = []
            if neurons:
                tg.append(L.neuron_target([(int(cl[j]), int(ct[j]), int(cn[j])) for j in neurons]))
            if chunk[-1] == k + T:  # the logit target sits in the copy after the neuron targets
                row = L.logits[len(neurons), T - 1].float()
                tg.append((row[tok] - row.mean())[None])
            tgt = torch.cat(tg)
            con = L.contributions(tgt, keep=i + batch < len(targets))
            B = len(chunk)
            rows = con["mlp"][cl[:, None], torch.arange(B, device=dev)[None], ct[:, None], cn[:, None]].T  # [B, k]
            idx = torch.tensor(chunk, device=dev)
            A[idx, :k] = rows
            A[idx, k:k + T] = con["embed"][:B]
            other[idx] = tgt.detach().float() - A[idx].sum(1)
            del con
        A[:k][fixed], other[:k][fixed] = 0, 0
        logit_val = float(tgt[-1].detach())

    # pruning
    absA = A.abs()
    norm = absA / (absA.sum(1, keepdim=True) + other.abs()[:, None]).clamp(min=1e-10)
    w = torch.zeros(n_nodes, device=dev)
    w[k + T] = 1.0
    infl = _influence(norm, w)
    order = infl[:k].sort(descending=True)
    cum = order.values.cumsum(0) / order.values.sum().clamp(min=1e-10)
    keep = torch.zeros(n_nodes, dtype=torch.bool, device=dev)
    keep[order.indices[: int((cum < node_threshold).sum()) + 1]] = True
    keep[k:] = True
    pruned = norm * keep[:, None] * keep[None, :]
    infl_p = _influence(pruned, w) + w
    escore = pruned * infl_p[:, None]
    flat_e = escore.flatten()
    srt = flat_e.sort(descending=True).values
    cut = srt[min(int(((srt.cumsum(0) / srt.sum().clamp(min=1e-10)) < edge_threshold).sum()), len(srt) - 1)]
    emask = (escore >= cut) & (escore > 0)
    if max_edges and int(emask.sum()) > max_edges:
        emask &= escore >= flat_e.topk(max_edges).values[-1]
    used = emask.any(0) | emask.any(1)
    used[k + T] = True

    stats = {"candidates": k, "candidate_share": round(captured, 4),
             "logit_from_candidates": round(float(A[k + T].sum()) / logit_val, 4) if logit_val else None,
             "logit_other": round(float(other[k + T]) / logit_val, 4) if logit_val else None,
             "kept_neurons": int(used[:k].sum()), "kept_edges": int(emask.sum())}
    ids_kept = used.nonzero().flatten().tolist()
    remap = {old: new for new, old in enumerate(ids_kept)}
    nodes = []
    for j in ids_kept:
        if j < k:
            nodes.append({"kind": "neuron", "layer": int(cl[j]), "pos": int(ct[j]), "n": int(cn[j]),
                          "act": round(float(acts[j]), 4), "gate": round(float(gates[j]), 3),
                          "influence": round(float(infl[j]), 5), "other": round(float(other[j]), 4),
                          **({"clamped": True} if fixed[j] else {})})
        elif j < k + T:
            nodes.append({"kind": "embed", "layer": -1, "pos": j - k, "n": 0, "act": None,
                          "influence": round(float(infl[j]), 5)})
        else:
            nodes.append({"kind": "logit", "layer": Lyr, "pos": T - 1, "n": tok, "act": round(logit_val, 4),
                          "influence": 1.0, "other": round(float(other[j]), 4)})
    dst, src = emask.nonzero(as_tuple=True)
    edges = [[remap[int(s)], remap[int(d)], round(float(A[d, s]), 5)]
             for s, d in zip(src.tolist(), dst.tolist()) if int(s) in remap and int(d) in remap]
    return {"token": tok, "prob": round(prob, 4), "nodes": nodes, "edges": edges, "stats": stats}


def _influence(A, w):
    """w @ (A + A^2 + ...): A is strictly causal (rows = targets, columns = sources), so the series ends."""
    cur, out = w @ A, torch.zeros_like(w)
    for _ in range(A.shape[0]):
        if not cur.any():
            break
        out += cur
        cur = cur @ A
    return out


# ---------- patched forwards (self is the module) ----------
def _norm_forward(self, x):
    xf = x.float()
    inv = self._lin.frozen(self, "inv", torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps))
    return (xf * inv * (1.0 + self.weight.float())).type_as(x)


def _attn_forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kw):
    lin = self._lin
    shape = hidden_states.shape[:-1]
    hshape = (*shape, -1, self.head_dim)
    q, gate = torch.chunk(self.q_proj(hidden_states).view(*shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*shape, -1)
    q = self.q_norm(q.view(hshape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hshape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hshape).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = q35.apply_rotary_pos_emb(q, k, cos, sin)
    k, v = q35.repeat_kv(k, self.num_key_value_groups), q35.repeat_kv(v, self.num_key_value_groups)
    T = q.shape[-2]
    scores = (q @ k.transpose(2, 3)).float() * self.scaling
    scores = scores.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    w = lin.frozen(self, "pattern", torch.softmax(scores, dim=-1).to(q.dtype))
    out = (w @ v).transpose(1, 2).reshape(*shape, -1)
    out = out * lin.frozen(self, "gate", torch.sigmoid(gate))
    return self.o_proj(out), None


def _deltanet_forward(self, hidden_states, cache_params=None, attention_mask=None, **kw):
    lin = self._lin
    B, T, _ = hidden_states.shape
    mixed = self.in_proj_qkv(hidden_states).transpose(1, 2)
    conv = F.conv1d(mixed.to(self.conv1d.weight.dtype), self.conv1d.weight, None, padding=self.conv_kernel_size - 1,
                    groups=self.conv_dim)[..., :T]
    mixed = (conv * lin.frozen(self, "conv_sig", torch.sigmoid(conv))).to(hidden_states.dtype).transpose(1, 2)
    q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    q = lin.frozen(self, "q", q.reshape(B, T, -1, self.head_k_dim))
    k = lin.frozen(self, "k", k.reshape(B, T, -1, self.head_k_dim))
    v = v.reshape(B, T, -1, self.head_v_dim)
    beta = lin.frozen(self, "beta", self.in_proj_b(hidden_states).sigmoid())
    g = lin.frozen(self, "decay", -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias))
    if self.num_v_heads // self.num_k_heads > 1:
        r = self.num_v_heads // self.num_k_heads
        q, k = q.repeat_interleave(r, dim=2), k.repeat_interleave(r, dim=2)
    core, _ = q35.torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
                                               use_qk_l2norm_in_kernel=True)
    core = core.reshape(-1, self.head_v_dim)
    z = self.in_proj_z(hidden_states).reshape(-1, self.head_v_dim)
    nm = self.norm
    cf = core.float()
    inv = lin.frozen(self, "gn_inv", torch.rsqrt(cf.pow(2).mean(-1, keepdim=True) + nm.variance_epsilon))
    out = nm.weight * (cf * inv).to(core.dtype)
    out = (out * lin.frozen(self, "gn_gate", F.silu(z.float()))).to(core.dtype)
    return self.out_proj(out.reshape(B, T, -1))


def _mlp_forward(self, x):
    lin, l = self._lin, self._layer
    g, u = self.gate_proj(x), self.up_proj(x)
    key = (id(self), "act")
    if lin.replay and lin.freeze_mlp:
        a = lin.cache[key].clone()
    else:
        a = (F.silu(g) * u).detach()
        if not lin.replay:
            lin.cache[key] = a.clone()
    for t, n, d in lin.inject.get(l, []):
        a[:, t, n] += d
    cl = lin.clamp.get(l)
    if cl:
        ts, ns, vs = (torch.as_tensor(c, device=a.device) for c in cl)
        a[:, ts, ns] = vs.to(a.dtype)
    if lin.mode == "total":
        sg = torch.sigmoid(g).detach()
        a_lin = _Shapley.apply(g * sg, u)
        leaf = a_lin + (a - a_lin).detach()  # same value as the (possibly injected) activation
        if cl:
            mask = torch.zeros(a.shape[1:], dtype=torch.bool, device=a.device)
            mask[ts, ns] = True
            leaf = torch.where(mask, a, leaf)
    else:
        leaf = a.requires_grad_(True)
    lin.mlp[l] = {"x": x, "g": g.detach(), "u": u.detach(), "a": leaf}
    return self.down_proj(leaf)
