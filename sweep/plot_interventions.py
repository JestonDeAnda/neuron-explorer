"""Attribution graphs of the intervened model for every swap_state / swap_followup experiment, one SVG each, and an
index page: runs/attribution/interventions/{index.html, NN_<slug>.json, NN_<slug>.svg}. Clamped neurons are drawn as
sources with an orange ring. Needs runs/attribution/swap_state.json (from `swap_state.py swap`).

    uv run python sweep/plot_interventions.py            # all, skipping graphs already saved
    uv run python sweep/plot_interventions.py --redo     # rebuild every graph
"""
import argparse
import html
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "explorer"))
from attribution import build_graph  # noqa: E402
from plot_attribution import svg  # noqa: E402
from swap_state import CACHE, DALLAS, OTHERS, ROOT, Model, vocab  # noqa: E402

OUT = os.path.join(ROOT, "runs", "attribution", "interventions")
SF = OTHERS["San Francisco"]
WATCH = [" Austin", " Sacramento"]


def experiments(M, B, ps, texas, cal):
    """[(group, title, text, edits [(layer, pos, n, value)], words to report)]"""
    def city_pos(text, city):
        last = M.tok(" " + city, add_special_tokens=False).input_ids[-1]
        return max(i for i, t in enumerate(M.ids(text)) if t == last)

    sf = lambda l, n: float(B[l][ps, n])
    to_cal = lambda pos, group, s: [(l, pos, n, s * sf(l, n)) for l, n in group]
    pd = city_pos(DALLAS, "Dallas")
    X = [("Baselines", "Dallas prompt, no intervention", DALLAS, [], WATCH),
         ("Baselines", "San Francisco prompt, no intervention (source of the California group)", SF, [], WATCH)]
    X += [("L5 N8109 alone", f"L5 N8109 at ' Dallas' -> {v:+.2f} ({why})", DALLAS, [(5, pd, 8109, v)], WATCH)
          for v, why in ((0.0, "ablated"), (sf(5, 8109), "its San Francisco value"), (2.84, "sign flipped"))]
    X.append(("California neuron picked by its lens", f"L4 N5319 at ' Dallas' -> 3 x its SF value ({3 * sf(4, 5319):+.2f})",
              DALLAS, to_cal(pd, [(4, 5319)], 3), WATCH))
    X += [("Texas group ablated", f"Texas group top {m} at ' Dallas' -> 0", DALLAS,
           [(l, pd, n, 0.0) for l, n in texas[:m]], WATCH) for m in (5, 10, 20, 40)]
    X += [("California group written in", f"California group top {m} at ' Dallas' -> {s} x SF value", DALLAS,
           to_cal(pd, cal[:m], s), WATCH) for m in (5, 10, 20, 40) for s in (1, 2, 4)]
    for m in (5, 10, 20, 40):
        for s in (1, 2, 4):
            e = {(l, n): 0.0 for l, n in texas[:m]}
            e.update({(l, n): s * sf(l, n) for l, n in cal[:m]})
            X.append(("Texas off and California on", f"Texas top {m} -> 0, California top {m} -> {s} x SF value",
                      DALLAS, [(l, pd, n, v) for (l, n), v in e.items()], WATCH))
    X += [("Whole MLP patched from San Francisco", f"every MLP neuron L0-L{L} at ' Dallas' set to its SF value", DALLAS,
           [(l, pd, n, sf(l, n)) for l in range(L + 1) for n in range(B[l].shape[1])], WATCH) for L in (3, 5, 8, 12)]
    X += [("Single California-group neurons", f"L{l} N{n} at ' Dallas' -> 2 x SF value ({2 * sf(l, n):+.2f})", DALLAS,
           to_cal(pd, [(l, n)], 2), WATCH) for l, n in cal[:8]]
    X.append(("Single California-group neurons", "California top 5 without L5 N6203 -> 2 x SF value", DALLAS,
              to_cal(pd, cal[1:5], 2), WATCH))
    v6 = sf(5, 6203)
    X += [("Two L5 neurons", t, DALLAS, e, WATCH) for t, e in [
        (f"N8109 -> 0, N6203 -> {v6:+.2f} (1 x SF)", [(5, pd, 8109, 0.0), (5, pd, 6203, v6)]),
        (f"N8109 -> 0, N6203 -> {2 * v6:+.2f} (2 x SF)", [(5, pd, 8109, 0.0), (5, pd, 6203, 2 * v6)]),
        (f"N8109 -> +2.84, N6203 -> {2 * v6:+.2f}", [(5, pd, 8109, 2.84), (5, pd, 6203, 2 * v6)]),
        (f"N6203 -> {3 * v6:+.2f} (3 x SF) alone", [(5, pd, 6203, 3 * v6)])]]
    for text, words in [("Fact: Dallas is a city in the state of", [" Texas", " California"]),
                        ("Q: Which state is Dallas in?\nA: Dallas is in", [" Texas", " California"]),
                        ("Fact: the governor of the state containing Dallas is", [" Greg", " Gavin", " Jerry"]),
                        ("Fact: the largest city in the state containing Dallas is", [" Houston", " San", " Los"]),
                        ("Fact: the sports team the Dallas Cowboys plays in the city of", [" Dallas", " San"])]:
        p = city_pos(text, "Dallas")
        short = text.replace("\n", " ")
        X.append(("Does Dallas now sit in California?", f"'{short}', no intervention", text, [], words))
        X.append(("Does Dallas now sit in California?", f"'{short}', California top 5 -> 2 x SF at ' Dallas'",
                  text, to_cal(p, cal[:5], 2), words))
    for city in ("Chicago", "Miami", "Seattle", "Boston"):
        text = f"Fact: the capital of the state containing {city} is"
        p = city_pos(text, city)
        X.append(("Other cities", f"{city}, no intervention", text, [], [" Sacramento"]))
        X.append(("Other cities", f"{city}, California top 5 -> 2 x SF at ' {city}'", text, to_cal(p, cal[:5], 2),
                  [" Sacramento"]))
    return X


def annotate(M, g, ids, v, st):
    g["tokens"] = [M.tok.decode([i]) for i in ids]
    g["target"] = M.tok.decode([g["token"]])
    for x in g["nodes"]:
        x["tok"] = g["tokens"][x["pos"]]
        if x["kind"] == "neuron":
            x["lens"] = [v[t] for t in st["up" if x["act"] > 0 else "down"][x["layer"], x["n"], :5]]
    return g


def index(rows):
    out = ['<!doctype html><meta charset="utf-8"><title>attribution graphs of the state-swap interventions</title>',
           '<style>body{font:14px -apple-system,Helvetica,sans-serif;margin:24px;color:#1a202c}td,th{padding:3px 10px;'
           'text-align:left;border-bottom:1px solid #eee}th{background:#f7fafc}img{max-width:100%;border:1px solid #ddd;'
           'margin:6px 0 28px}h2{margin-top:36px}.n{text-align:right;font-variant-numeric:tabular-nums}</style>',
           '<h1>Attribution graphs of the state-swap interventions</h1>',
           '<p>Prompt edits are made at the city token only and the rest of the forward pass is recomputed. Each graph '
           'explains the intervened model\'s top token; clamped neurons have an orange ring and no inputs. '
           'Probabilities are from the real (unlinearized) intervened model.</p>',
           '<table><tr><th>#</th><th>group</th><th>intervention</th><th>clamped</th><th>watched tokens</th>'
           '<th>top 3</th></tr>']
    for r in rows:
        w = ", ".join(f"{k.strip()} {p:.3f}" for k, p in r["watch"].items())
        top = ", ".join(f"{t!r} {p:.2f}" for t, p in r["top"][:3])
        out.append(f'<tr><td><a href="#g{r["i"]}">{r["i"]:02d}</a></td><td>{html.escape(r["group"])}</td>'
                   f'<td>{html.escape(r["title"])}</td><td class="n">{r["clamped"]}</td><td>{html.escape(w)}</td>'
                   f'<td>{html.escape(top)}</td></tr>')
    out.append("</table>")
    group = None
    for r in rows:
        if r["group"] != group:
            group = r["group"]
            out.append(f"<h2>{html.escape(group)}</h2>")
        out.append(f'<h3 id="g{r["i"]}">{r["i"]:02d}. {html.escape(r["title"])}</h3>'
                   f'<a href="{r["svg"]}"><img src="{r["svg"]}"></a>')
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write("\n".join(out))


def bundle(rows):
    """One self-contained HTML file, the graphs inlined as SVG, so it can be sent on its own."""
    out = ['<!doctype html><meta charset="utf-8"><title>attribution graphs of the state-swap interventions</title>',
           '<style>body{font:14px -apple-system,Helvetica,sans-serif;margin:24px auto;max-width:1400px;color:#1a202c}'
           'td,th{padding:3px 10px;text-align:left;border-bottom:1px solid #eee;vertical-align:top}th{background:#f7fafc}'
           'svg{width:100%;height:auto;border:1px solid #ddd;margin:6px 0 28px}h2{margin-top:36px}'
           '.n{text-align:right;font-variant-numeric:tabular-nums}</style>',
           '<h1>Attribution graphs of the state-swap interventions</h1>',
           '<p>One file, graphs inlined. Prompt edits are made at the city token only and the rest of the forward pass '
           'is recomputed. Each graph explains the intervened model\'s top token; clamped neurons have an orange ring '
           'and no inputs. Hover a node or edge for its activation, steer lens and edge weight. Probabilities in the '
           'table are from the real (unlinearized) intervened model.</p>',
           '<table><tr><th>#</th><th>group</th><th>intervention</th><th>clamped</th><th>watched tokens</th>'
           '<th>top 3</th></tr>']
    body = []
    group = None
    for r in rows:
        w = ", ".join(f"{k.strip()} {p:.3f}" for k, p in r["watch"].items())
        top = ", ".join(f"{t!r} {p:.2f}" for t, p in r["top"][:3])
        out.append(f'<tr><td><a href="#g{r["i"]}">{r["i"]:02d}</a></td><td>{html.escape(r["group"])}</td>'
                   f'<td>{html.escape(r["title"])}</td><td class="n">{r["clamped"]}</td><td>{html.escape(w)}</td>'
                   f'<td>{html.escape(top)}</td></tr>')
        if r["group"] != group:
            group = r["group"]
            body.append(f"<h2>{html.escape(group)}</h2>")
        svg = open(os.path.join(OUT, r["svg"])).read()
        body.append(f'<h3 id="g{r["i"]}">{r["i"]:02d}. {html.escape(r["title"])}</h3>\n{svg}')
    out.append("</table>")
    path = os.path.join(ROOT, "runs", "attribution", "state_swap.html")
    with open(path, "w") as f:
        f.write("\n".join(out + body))
    print(path, os.path.getsize(path) // (1 << 20), "MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    saved = json.load(open(os.path.join(ROOT, "runs", "attribution", "swap_state.json")))
    texas = [tuple(r[:2]) for r in saved["texas"]]
    cal = [tuple(r[:2]) for r in saved["california"]]
    M = Model()
    v, st = vocab(), np.load(os.path.join(CACHE, "steer_index.npz"))
    ps = len(M.ids(SF)) - 2
    _, B = M.run(SF, capture=True)
    rows = []
    for i, (group, title, text, edits, words) in enumerate(experiments(M, B, ps, texas, cal)):
        slug = re.sub(r"\W+", "_", title).strip("_")[:70]
        base = os.path.join(OUT, f"{i:02d}_{slug}")
        lp, _ = M.run(text, edits)
        row = {"i": i, "group": group, "title": title, "clamped": len(edits),
               "watch": {w: M.prob(lp, w) for w in words}, "top": M.top(lp, 5), "svg": os.path.basename(base) + ".svg"}
        if a.redo or not os.path.exists(base + ".json"):
            ids = M.ids(text)
            g = build_graph(M.lm, M.model.lm_head, ids, k=a.k, max_edges=3000,
                            clamp={(l, p, n): x for l, p, n, x in edits})
            g = annotate(M, g, ids, v, st)
            g.update(title=f"{i:02d}. {title}", group=group, edits=edits if len(edits) <= 200 else len(edits),
                     watch=row["watch"], top=row["top"])
            if abs(g["prob"] - row["top"][0][1]) > 0.02:
                print(f"  warning: linearized forward p={g['prob']:.3f}, real p={row['top'][0][1]:.3f}")
            json.dump(g, open(base + ".json", "w"), ensure_ascii=False)
            with open(base + ".svg", "w") as f:
                f.write(svg(g))
        rows.append(row)
        index(rows)
        print(f"{i:02d} {title[:70]:70s}", {k.strip(): round(p, 3) for k, p in row["watch"].items()}, row["top"][:2],
              flush=True)
    bundle(rows)
    print(os.path.join(OUT, "index.html"))


if __name__ == "__main__":
    main()
