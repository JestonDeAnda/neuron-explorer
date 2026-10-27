"""Checks explorer/attribution.py: the linearized Qwen3.5 reproduces the forward pass, its edges sum to each target
(completeness), and each edge equals the change a source's removal causes in the frozen model (linearity).

    uv run python sweep/check_attribution.py --tiny            # random 4-layer Qwen3.5, CPU, seconds
    uv run python sweep/check_attribution.py                   # Qwen3.5-4B, float32
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explorer"))
from attribution import Linearized  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tiny", action="store_true")
ap.add_argument("--device", default="cpu")
ap.add_argument("--prompt", default="The capital of the state containing Dallas is")
ap.add_argument("--targets", type=int, default=6)
a = ap.parse_args()
torch.manual_seed(0)

if a.tiny:
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    cfg = Qwen3_5TextConfig(vocab_size=200, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
                            layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=4,
                            num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                            linear_key_head_dim=16, linear_value_head_dim=16, initializer_range=0.25)
    model = Qwen3_5ForCausalLM(cfg).float().eval()
    with torch.no_grad():  # non-trivial norm weights, so the (1 + w) scales are exercised
        for m in model.modules():
            if hasattr(m, "weight") and m.weight is not None and m.weight.ndim == 1:
                m.weight.normal_(0, 0.3)
    lm, head = model.model, model.lm_head
    ids = torch.randint(0, cfg.vocab_size, (12,)).tolist()
else:
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    model = Qwen3_5ForConditionalGeneration.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.float32).eval()
    lm, head = model.model.language_model, model.lm_head
    ids = tok(a.prompt).input_ids
model.to(a.device)
for p in model.parameters():
    p.requires_grad_(False)
T, Lyr = len(ids), len(lm.layers)

with torch.no_grad():
    ref = head(lm(input_ids=torch.tensor([ids], device=a.device)).last_hidden_state)[0].float()

ok = True


def report(name, err, tol):
    global ok
    ok &= err <= tol
    print(f"{'PASS' if err <= tol else 'FAIL'}  {name}: {err:.2e} (tol {tol:.0e})")


with Linearized(lm, head) as L:
    B = a.targets
    logits = L.forward(ids, copies=B)
    report("forward unchanged (max |logit diff| / max |logit|)",
           float((logits[0].detach().float() - ref).abs().max() / ref.abs().max()), 1e-4)

    # targets: the strongest-activated neurons at the last position, one per layer from the middle on
    acts = {l: L.mlp[l]["a"].detach()[0] for l in range(Lyr)}
    layers = torch.linspace(1, Lyr - 1, B).round().long().tolist()
    specs = []
    for l in layers:
        n = int(acts[l][T - 1].abs().argmax())
        specs.append((l, T - 1, n))
    val = torch.tensor([float(acts[l][t, n]) for l, t, n in specs], device=a.device)
    clean_g = {l: L.mlp[l]["g"].clone() for l in range(Lyr)}
    clean_u = {l: L.mlp[l]["u"].clone() for l in range(Lyr)}

    for rule in ("half", "up", "gate"):
        tgt = L.neuron_target(specs, rule)
        con = L.contributions(tgt)
        total = con["mlp"].sum((0, 2, 3)) + con["embed"].sum(-1)
        report(f"neuron completeness, rule {rule} (max |sum edges - a| / |a|)",
               float(((total - val).abs() / val.abs()).max()), 1e-3)
        if rule == "half":
            half = con
            later = sum(float(con["mlp"][l, b].abs().sum()) for b, (lt, _, _) in enumerate(specs) for l in range(lt, Lyr))
            report("no edges from the target's own or later layers", later, 0.0)

    L.logits = logits
    top = logits[:, T - 1].detach().float().argmax(-1)
    ltgt = L.logit_target([(T - 1, int(top[0]))] * B)
    lcon = L.contributions(ltgt)
    ltotal = lcon["mlp"].sum((0, 2, 3)) + lcon["embed"].sum(-1)
    report("logit completeness (centered logit)", float(((ltotal - ltgt.detach()).abs() / ltgt.detach().abs()).max()), 1e-3)

    # linearity: removing the strongest source of each target in the frozen model changes the target by -edge
    errs, rel_true = [], []
    for b, (l, t, n) in enumerate(specs):
        e = half["mlp"][:l, b]
        flat = int(e.abs().flatten().argmax())
        sl, rest = divmod(flat, e.shape[1] * e.shape[2])
        st, sn = divmod(rest, e.shape[2])
        edge = float(e[sl, st, sn])
        L.inject = {sl: [(st, sn, -float(acts[sl][st, sn]))]}
        with torch.no_grad():
            L.forward(ids, copies=1, replay=True, freeze_mlp=True)
        x = L.mlp[l]["x"][0, t]
        gl = float(x @ lm.layers[l].mlp.gate_proj.weight[n])
        ul = float(x @ lm.layers[l].mlp.up_proj.weight[n])
        g, u = float(clean_g[l][0, t, n]), float(clean_u[l][0, t, n])
        s = float(torch.sigmoid(torch.tensor(g)))
        changed = 0.5 * s * (gl * u + g * ul) - float(val[b])
        errs.append(abs(changed + edge) / max(abs(edge), 1e-12))
        rel_true.append((l, t, n, sl, st, sn, float(val[b]), edge, g))
    L.inject = {}
    report("frozen-model removal of top source = -edge (max rel err)", max(errs), 1e-3)

    with torch.no_grad():
        L.forward(ids, copies=1)  # refresh the cache after the replays
    L.mode = "total"
    logits = L.forward(ids, copies=1)
    lt = L.logit_target([(T - 1, int(top[0]))])
    tcon = L.contributions(lt)
    report("total (RelP) mode: embeddings carry the whole logit", float((tcon["embed"].sum() - lt.detach()).abs() / lt.detach().abs()), 1e-3)

all_g = torch.cat([clean_g[l][0].flatten() for l in range(Lyr)])
all_a = torch.cat([(F.silu(clean_g[l][0]) * clean_u[l][0]).flatten() for l in range(Lyr)])


def removed(sl, st, sn, l, t, n):
    """The target's activation in the real model (nothing frozen) with one source neuron set to 0."""
    seen = {}

    def zero(m, i):
        h = i[0].clone()
        h[0, st, sn] = 0
        return (h,)
    hk = lm.layers[sl].mlp.down_proj.register_forward_pre_hook(zero)
    hr = lm.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, i: seen.__setitem__("a", float(i[0][0, t, n])))
    with torch.no_grad():
        lm(input_ids=torch.tensor([ids], device=a.device))
    hk.remove(), hr.remove()
    return seen["a"]


print("\nreal model, top source of each target set to 0 (nothing frozen); the edge predicts Δa_t = -edge in the frozen"
      " model, and about -2 x edge to first order when the source feeds g or u alone")
print("  target           source            a_t      edge   true Δa_t   gate g_t")
for l, t, n, sl, st, sn, v, e, g in rel_true:
    d = removed(sl, st, sn, l, t, n) - v
    print(f"  L{l:<2} t{t:<2} N{n:<5}  L{sl:<2} t{st:<2} N{sn:<5}  {v:8.3f} {e:8.3f} {d:9.3f}  {g:8.3f}")

big = all_a.abs() > all_a.abs().quantile(0.99)
print(f"\nneurons in the top 1% |a| with g < -1.278 (silu' < 0, where the frozen sigmoid has the wrong sign): "
      f"{float((all_g[big] < -1.278).float().mean()):.1%}")
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
