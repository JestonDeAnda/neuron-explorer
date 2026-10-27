"""Controls for swap_state.py: single neurons of the California group, whether the swapped Dallas reads as being in
California on other questions, and the same edit on other cities. Needs runs/attribution/swap_state.json.

    uv run python sweep/swap_followup.py
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swap_state import DALLAS, OTHERS, Model  # noqa: E402

M = Model()
SF = OTHERS["San Francisco"]
ps = len(M.ids(SF)) - 2
_, B = M.run(SF, capture=True)
cal = [tuple(r[:2]) for r in json.load(open("runs/attribution/swap_state.json"))["california"]]


def city_pos(text, city):
    ids, last = M.ids(text), M.tok(" " + city, add_special_tokens=False).input_ids[-1]
    return max(i for i, t in enumerate(ids) if t == last)


def edits(pos, group, s):
    return [(l, pos, n, s * float(B[l][ps, n])) for l, n in group]


def show(text, e, words, label):
    lp, _ = M.run(text, e)
    print(f"  {label:38s}", " ".join(f"{w.strip()} {M.prob(lp, w):.3f}" for w in words), "|", M.top(lp, 4))


@torch.inference_mode()
def greedy(text, e, n=12):
    ids = M.ids(text)
    for _ in range(n):
        M.edits = {}
        for layer, pos, nn, v in e:
            ns, vs = M.edits.setdefault(layer, {}).setdefault(pos, ([], []))
            ns.append(nn), vs.append(v)
        logits = M.model(input_ids=torch.tensor([ids], device=M.dev), use_cache=False).logits[0, -1]
        ids.append(int(logits.argmax()))
    M.edits = {}
    return M.tok.decode(ids[len(M.ids(text)):])


pd = city_pos(DALLAS, "Dallas")
print("== two L5 neurons at ' Dallas'")
show(DALLAS, [(5, pd, 8109, 0.0), (5, pd, 6203, 2 * float(B[5][ps, 6203]))], [" Austin", " Sacramento"],
     "N8109 -> 0, N6203 -> 2 x SF value")
print("\n== single California-group neurons at ' Dallas', 2 x SF value")
for l, n in cal[:8]:
    show(DALLAS, edits(pd, [(l, n)], 2), [" Austin", " Sacramento"], f"L{l} N{n} ({float(B[l][ps, n]):+.2f})")
show(DALLAS, edits(pd, cal[:5], 2), [" Austin", " Sacramento"], "top 5 together")
show(DALLAS, edits(pd, cal[1:5], 2), [" Austin", " Sacramento"], "top 5 without L5 N6203")

print("\n== does Dallas now sit in California? (top-5 California group, 2 x, at ' Dallas')")
for text, words in [("Fact: Dallas is a city in the state of", [" Texas", " California"]),
                    ("Q: Which state is Dallas in?\nA: Dallas is in", [" Texas", " California"]),
                    ("Fact: the governor of the state containing Dallas is", [" Greg", " Gavin"]),
                    ("Fact: the largest city in the state containing Dallas is", [" Houston", " Los"]),
                    ("Fact: the sports team the Dallas Cowboys plays in the city of", [" Dallas", " Arlington", " San"])]:
    p = city_pos(text, "Dallas")
    show(text, [], words, "baseline: " + text[:30])
    show(text, edits(p, cal[:5], 2), words, "swapped")
    print("    greedy, swapped:", repr(greedy(text, edits(p, cal[:5], 2))))

print("\n== other cities, the same edit at the city token")
for city, text in [("Chicago", OTHERS["Chicago"]), ("Miami", OTHERS["Miami"]),
                   ("Seattle", "Fact: the capital of the state containing Seattle is"),
                   ("Boston", "Fact: the capital of the state containing Boston is")]:
    p = city_pos(text, city)
    show(text, [], [" Sacramento"], f"{city} baseline")
    show(text, edits(p, cal[:5], 2), [" Sacramento"], f"{city} swapped")
