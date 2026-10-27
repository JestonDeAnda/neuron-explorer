"""Swap the state in a two-hop prompt: at the city token, turn the Texas neurons down and California neurons up, and
see whether the capital the model names follows (Austin -> Sacramento). Interventions are real (the forward pass after
the edited layer is recomputed), at one position only.

    uv run python sweep/swap_state.py explore      # baselines, California-lens neurons, activation contrasts
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "explorer", "cache")

DALLAS = "Fact: the capital of the state containing Dallas is"
OTHERS = {"San Francisco": "Fact: the capital of the state containing San Francisco is",
          "Los Angeles": "Fact: the capital of the state containing Los Angeles is",
          "Miami": "Fact: the capital of the state containing Miami is",
          "Chicago": "Fact: the capital of the state containing Chicago is"}
CAPITALS = [" Austin", " Sacramento", " Tallahassee", " Springfield", " Houston", " Dallas", " Los", " San"]


class Model:
    def __init__(self, dtype=torch.bfloat16):
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        self.dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
        m = Qwen3_5ForConditionalGeneration.from_pretrained("Qwen/Qwen3.5-4B", dtype=dtype).to(self.dev).eval()
        self.model, self.lm = m, m.model.language_model
        self.edits = {}  # layer -> pos -> (neurons, values): activations set at one position
        self.capture = None
        for i, layer in enumerate(self.lm.layers):
            layer.mlp.down_proj.register_forward_pre_hook(self._hook(i))

    def _hook(self, i):
        def hook(mod, inp):
            h = inp[0]
            if self.capture is not None:
                self.capture[i] = h[0].float().cpu()
            mine = self.edits.get(i)
            if not mine:
                return None
            h = h.clone()
            for pos, (ns, vs) in mine.items():
                h[0, pos, torch.tensor(ns, device=h.device)] = torch.tensor(vs, device=h.device, dtype=h.dtype)
            return (h,)
        return hook

    def ids(self, text):
        return self.tok(text).input_ids

    @torch.inference_mode()
    def run(self, text, edits=(), capture=False):
        """Next-token log-probs after text, with edits; optionally every layer's activations [L][T, N]."""
        self.edits, self.capture = {}, ({} if capture else None)
        for layer, pos, n, v in edits:
            ns, vs = self.edits.setdefault(layer, {}).setdefault(pos, ([], []))
            ns.append(n)
            vs.append(v)
        try:
            out = self.model(input_ids=torch.tensor([self.ids(text)], device=self.dev), use_cache=False)
            lp = out.logits[0, -1].float().log_softmax(-1).cpu()
            return lp, self.capture
        finally:
            self.edits, self.capture = {}, None

    def top(self, lp, k=8):
        v, i = lp.topk(k)
        return [(self.tok.decode([int(t)]), round(float(x.exp()), 3)) for x, t in zip(v, i)]

    def prob(self, lp, word):
        return float(lp[self.tok(word, add_special_tokens=False).input_ids[0]].exp())


def vocab():
    return json.load(open(os.path.join(CACHE, "vocab.json")))


def explore(M):
    v, st = vocab(), np.load(os.path.join(CACHE, "steer_index.npz"))
    print("== baselines")
    for name, text in [("Dallas", DALLAS)] + list(OTHERS.items()):
        lp, _ = M.run(text)
        print(f"{name:14s}", M.top(lp, 6))

    print("\n== neurons whose steer lens (top 5) names California / Texas")
    for word in ("California", "Texas"):
        hits = []
        for pole in ("up", "down"):
            top = st[pole][:, :, :5]
            ids = [i for i, t in enumerate(v) if word.lower() in t.lower()]
            for l, n in zip(*np.nonzero(np.isin(top, ids).any(-1))):
                hits.append((int(l), int(n), pole, [v[t] for t in st[pole][l, n, :5]]))
        print(f"{word}: {len(hits)}")
        for h in sorted(hits)[:60]:
            print("  ", h)

    print("\n== activations at the city token, Dallas vs San Francisco, layers 0-12")
    _, A = M.run(DALLAS, capture=True)
    _, B = M.run(OTHERS["San Francisco"], capture=True)
    td, ts = M.ids(DALLAS), M.ids(OTHERS["San Francisco"])
    pd = len(td) - 2
    ps = len(ts) - 2
    print("dallas token", repr(M.tok.decode([td[pd]])), "sf token", repr(M.tok.decode([ts[ps]])))
    rows = []
    for l in range(13):
        a, b = A[l][pd], B[l][ps]
        d = b - a
        for n in torch.topk(d.abs(), 6).indices.tolist():
            rows.append((l, n, round(float(a[n]), 3), round(float(b[n]), 3),
                         [v[t] for t in st["up" if d[n] > 0 else "down"][l, n, :5]]))
    for r in rows:
        print("  L%d N%d dallas %+.3f sf %+.3f" % r[:4], r[4])
    print("L5 N8109: dallas %+.3f sf %+.3f" % (A[5][pd, 8109], B[5][ps, 8109]),
          "steer up", [v[t] for t in st["up"][5, 8109, :5]], "down", [v[t] for t in st["down"][5, 8109, :5]])


def group(M, text, pos, max_layer, k):
    """Neurons of text's attribution graph at pos, layers <= max_layer, by influence: [(layer, n, act, influence)]."""
    sys.path.insert(0, os.path.join(ROOT, "explorer"))
    from attribution import build_graph
    g = build_graph(M.lm, M.model.lm_head, M.ids(text), k=k)
    rows = [(x["layer"], x["n"], x["act"], x["influence"]) for x in g["nodes"]
            if x["kind"] == "neuron" and x["pos"] == pos and x["layer"] <= max_layer]
    return sorted(rows, key=lambda r: -r[3])


def swap(M, max_layer, k, out):
    v, st = vocab(), np.load(os.path.join(CACHE, "steer_index.npz"))
    td, ts = M.ids(DALLAS), M.ids(OTHERS["San Francisco"])
    pd, ps = len(td) - 2, len(ts) - 2
    texas = group(M, DALLAS, pd, max_layer, k)
    cal = group(M, OTHERS["San Francisco"], ps, max_layer, k)
    lens = lambda l, n, a: [v[t] for t in st["up" if a > 0 else "down"][l, n, :4]]
    for name, rows in (("Texas group (Dallas graph, at ' Dallas')", texas),
                       ("California group (San Francisco graph, at ' Francisco')", cal)):
        print(f"\n== {name}: {len(rows)}")
        for l, n, a, inf in rows[:20]:
            print(f"  L{l} N{n} act {a:+.2f} infl {inf:.3f}", lens(l, n, a))

    _, A = M.run(DALLAS, capture=True)
    _, B = M.run(OTHERS["San Francisco"], capture=True)
    results = []

    def trial(label, edits):
        lp, _ = M.run(DALLAS, edits)
        r = {"label": label, "edits": len(edits), "Austin": M.prob(lp, " Austin"),
             "Sacramento": M.prob(lp, " Sacramento"), "top": M.top(lp, 5)}
        results.append(r)
        print(f"  {label:44s} Austin {r['Austin']:.3f}  Sacramento {r['Sacramento']:.3f}  ", r["top"][:4])

    print("\n== interventions at ' Dallas' (position %d); other positions untouched" % pd)
    trial("baseline", [])
    sf8109 = float(B[5][ps, 8109])
    trial("L5 N8109 -> 0", [(5, pd, 8109, 0.0)])
    trial(f"L5 N8109 -> its SF value {sf8109:+.2f}", [(5, pd, 8109, sf8109)])
    trial("L5 N8109 -> +2.84 (sign flipped)", [(5, pd, 8109, 2.84)])
    trial("L4 N5319 -> 3 x SF value", [(4, pd, 5319, 3 * float(B[4][ps, 5319]))])
    for m in (5, 10, 20, 40):
        T = texas[:m]
        trial(f"Texas top {m} -> 0", [(l, pd, n, 0.0) for l, n, _, _ in T])
    for m in (5, 10, 20, 40):
        for s in (1, 2, 4):
            C = cal[:m]
            trial(f"California top {m} -> {s} x SF value", [(l, pd, n, s * float(B[l][ps, n])) for l, n, _, _ in C])
    for m in (5, 10, 20, 40):
        for s in (1, 2, 4):
            T, C = texas[:m], cal[:m]
            e = {(l, n): 0.0 for l, n, _, _ in T}
            e.update({(l, n): s * float(B[l][ps, n]) for l, n, _, _ in C})
            trial(f"Texas top {m} -> 0, California top {m} -> {s} x", [(l, pd, n, x) for (l, n), x in e.items()])
    for L in (3, 5, 8, 12):
        trial(f"every MLP neuron L0-L{L} patched from SF", [(l, pd, n, float(B[l][ps, n]))
                                                          for l in range(L + 1) for n in range(B[l].shape[1])])
    if out:
        json.dump({"texas": texas, "california": cal, "results": results}, open(out, "w"), ensure_ascii=False, indent=1)
        print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["explore", "swap"])
    ap.add_argument("--max-layer", type=int, default=12, help="deepest layer the state groups are taken from")
    ap.add_argument("--k", type=int, default=512, help="attribution candidates per graph")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "attribution", "swap_state.json"))
    a = ap.parse_args()
    M = Model()
    if a.cmd == "explore":
        explore(M)
    else:
        swap(M, a.max_layer, a.k, a.out)


if __name__ == "__main__":
    main()
