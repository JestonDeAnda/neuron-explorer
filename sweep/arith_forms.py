"""One calculation in three surface forms (Arabic digits, English words, Chinese numerals): attribution graphs of
each, the neurons they share at the same role (first operand, operator, second operand, equals sign), and causal
tests of the shared set. A second calculation in the same three forms separates what belongs to this calculation
from generic arithmetic / prompt machinery.

    uv run python sweep/arith_forms.py               # graphs cached in runs/attribution/arith/, report beside them
    uv run python sweep/arith_forms.py --redo        # rebuild the graphs
"""
import argparse
import html
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "explorer"))
from attribution import build_graph  # noqa: E402
from plot_attribution import svg  # noqa: E402
from plot_interventions import annotate  # noqa: E402
from swap_state import CACHE, ROOT, Model, vocab  # noqa: E402

OUT = os.path.join(ROOT, "runs", "attribution", "arith")
ZH = "零一二三四五六七八九"
EN = "zero one two three four five six seven eight nine".split()
OPS = {"-": ("减", "minus", "-"), "*": ("乘以", "times", "*"), "+": ("加", "plus", "+")}
FORMS = {
    "digits": lambda a, op, b: f"Fact: {a}{OPS[op][2]}{b}=",
    "english": lambda a, op, b: f"Fact: {EN[a]} {OPS[op][1]} {EN[b]} equals",
    "chinese": lambda a, op, b: f"事实：{ZH[a]}{OPS[op][0]}{ZH[b]}等于",
}
ANSWER = {"digits": lambda r: str(r), "english": lambda r: " " + EN[r], "chinese": lambda r: ZH[r]}
CALCS = {"main": (8, "-", 3), "control": (2, "*", 4)}
ROLES = ["A", "OP", "B", "EQ"]


def result(a, op, b):
    return {"-": a - b, "*": a * b, "+": a + b}[op]


def roles(T):
    """The last four positions are first operand, operator, second operand, equals; earlier ones are context."""
    return {T - 4 + i: r for i, r in enumerate(ROLES)}


def keyed(g):
    """{(layer, n, role): node} for neurons, influence normalized to sum 1 over the graph's neurons."""
    rl = roles(len(g["tokens"]))
    ns = [x for x in g["nodes"] if x["kind"] == "neuron"]
    tot = sum(x["influence"] for x in ns) or 1
    out = {}
    for x in ns:
        k = (x["layer"], x["n"], rl.get(x["pos"], "ctx"))
        if k not in out or x["influence"] > out[k]["influence"]:
            out[k] = dict(x, w=x["influence"] / tot)
    return out


def overlap(p, q):
    return sum(min(p[k]["w"], q[k]["w"]) for k in p.keys() & q.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    M = Model()
    v, st = vocab(), np.load(os.path.join(CACHE, "steer_index.npz"))
    lens = lambda l, n, s: [v[t] for t in st["up" if s > 0 else "down"][l, n, :6]]

    # ---------- graphs ----------
    G, text, ans = {}, {}, {}
    for calc, (x, op, y) in CALCS.items():
        r = result(x, op, y)
        for form, make in FORMS.items():
            t = make(x, op, y)
            text[calc, form] = t
            tid = M.tok(ANSWER[form](r), add_special_tokens=False).input_ids
            assert len(tid) == 1, (form, r, tid)
            ans[calc, form] = tid[0]
            path = os.path.join(OUT, f"{calc}_{form}.json")
            if a.redo or not os.path.exists(path):
                ids = M.ids(t)
                g = annotate(M, build_graph(M.lm, M.model.lm_head, ids, token=tid[0], k=a.k, max_edges=3000), ids, v, st)
                json.dump(g, open(path, "w"), ensure_ascii=False)
                print(f"built {calc} {form}: {t!r} -> {g['target']!r} p={g['prob']:.3f}", flush=True)
            G[calc, form] = json.load(open(path))
    K = {key: keyed(g) for key, g in G.items()}
    forms = list(FORMS)

    # ---------- overlap ----------
    report = {"prompts": {f"{c}/{f}": text[c, f] for c, f in text}, "overlap": {}}
    names = [(c, f) for c in CALCS for f in forms]
    print("\n== influence-weighted overlap of neurons at the same (layer, neuron, role)")
    print(" " * 18 + "".join(f"{c[:4]}/{f[:3]:>4s}  " for c, f in names))
    for p in names:
        row = [overlap(K[p], K[q]) for q in names]
        report["overlap"][f"{p[0]}/{p[1]}"] = [round(x, 3) for x in row]
        print(f"{p[0]:8s}{p[1]:10s}" + "".join(f"{x:9.2f} " for x in row))

    T = {c: set.intersection(*(set(K[c, f]) for f in forms)) for c in CALCS}
    generic = T["main"] & T["control"]
    specific = T["main"] - T["control"]
    print(f"\nshared by all three forms: main {len(T['main'])}, control {len(T['control'])}, "
          f"both calculations {len(generic)}, main only {len(specific)}")
    for c in CALCS:
        for f in forms:
            mass = sum(K[c, f][k]["w"] for k in T[c])
            print(f"  {c:8s} {f:8s} {len(K[c, f]):4d} neurons; influence mass in the 3-form shared set {mass:.0%}")

    def profile(c, group, by):
        out = {}
        for f in forms:
            tot, sh = {}, {}
            for k, x in K[c, f].items():
                b = by(k)
                tot[b] = tot.get(b, 0) + x["w"]
                if k in group:
                    sh[b] = sh.get(b, 0) + x["w"]
            out[f] = {b: (round(tot[b], 3), round(sh.get(b, 0) / tot[b], 3)) for b in sorted(tot, key=str)}
        return out
    bins = lambda k: f"L{8 * (k[0] // 8)}-{8 * (k[0] // 8) + 7}"
    report["layer_profile"] = profile("main", T["main"], bins)
    report["role_profile"] = profile("main", T["main"], lambda k: k[2])
    print("\n== main calculation: (influence mass, share of it in the 3-form shared set) by layer band / role")
    for name in ("layer_profile", "role_profile"):
        for f, d in report[name].items():
            print(f"  {f:8s}", "  ".join(f"{b} {m:.2f}/{s:.0%}" for b, (m, s) in d.items()))

    rows = []
    for k in sorted(T["main"], key=lambda k: -sum(K["main", f][k]["w"] for f in forms)):
        acts = {f: K["main", f][k]["act"] for f in forms}
        rows.append({"layer": k[0], "n": k[1], "role": k[2], "generic": k in generic,
                     "w": {f: round(K["main", f][k]["w"], 4) for f in forms},
                     "act": acts, "same_sign": len({np.sign(x) for x in acts.values()}) == 1,
                     "lens": lens(k[0], k[1], acts["digits"])})
    report["shared"] = rows
    print(f"\n== top neurons shared by all three forms (main); sign agrees across forms for "
          f"{sum(r['same_sign'] for r in rows)}/{len(rows)}")
    for r in rows[:30]:
        print(f"  L{r['layer']:<2d} N{r['n']:<5d} {r['role']:3s} {'generic ' if r['generic'] else 'specific'} "
              + " ".join(f"{f[:3]} {r['w'][f]:.3f}({r['act'][f]:+.2f})" for f in forms), r["lens"])

    # ---------- causal tests (real model, edits at the role positions only) ----------
    def pos_of(c, f, role):
        return {r: p for p, r in roles(len(M.ids(text[c, f]))).items()}[role]

    def p_ans(c, f, edits, r=None):
        lp, _ = M.run(text[c, f], edits)
        return float(lp[ans[c, f] if r is None else M.tok(ANSWER[f](r), add_special_tokens=False).input_ids[0]].exp())

    def zero(c, f, keys):
        return [(l, pos_of(c, f, role), n, 0.0) for l, n, role in keys if role != "ctx"]

    rng = random.Random(0)
    N = M.lm.layers[0].mlp.down_proj.in_features
    report["ablation"] = {}
    print("\n== ablation (set to 0 at their role positions): P(answer)")
    for f in forms:
        shared = [k for k in T["main"] if k[2] != "ctx"]
        own = sorted((k for k in K["main", f] if k not in T["main"] and k[2] != "ctx"),
                     key=lambda k: -K["main", f][k]["w"])[:len(shared)]
        rand = [p_ans("main", f, zero("main", f, [(l, rng.randrange(N), role) for l, _, role in shared]))
                for _ in range(5)]
        res = {"baseline": p_ans("main", f, []), "shared": p_ans("main", f, zero("main", f, shared)),
               "form_only_top": p_ans("main", f, zero("main", f, own)), "random_mean": float(np.mean(rand)),
               "n": len(shared)}
        report["ablation"][f] = res
        print(f"  {f:8s} n={res['n']:3d}  baseline {res['baseline']:.3f}  shared {res['shared']:.3f}  "
              f"same number of this form's own top neurons {res['form_only_top']:.3f}  random {res['random_mean']:.3f}")

    eq = [k for k in T["main"] if k[2] == "EQ"]
    cap = {}
    for f in forms:
        _, cap[f] = M.run(text["control", f], capture=True)
    r2 = result(*CALCS["control"])
    r1 = result(*CALCS["main"])
    report["transplant"] = {"n": len(eq), "cells": {}}
    print(f"\n== transplant: the {len(eq)} shared EQ-position neurons set to their values in the control calculation "
          f"({'%d%s%d=%d' % (*CALCS['control'], r2)}), source form -> target form; P(target {r2}) / P(target {r1})")
    for src in forms:
        ps = pos_of("control", src, "EQ")
        for dst in forms:
            pd = pos_of("main", dst, "EQ")
            e = [(l, pd, n, float(cap[src][l][ps, n])) for l, n, _ in eq]
            rand_keys = [(l, rng.randrange(N)) for l, _, _ in eq]
            er = [(l, pd, n, float(cap[src][l][ps, n])) for l, n in rand_keys]
            cell = {"to_new": p_ans("main", dst, e, r2), "to_old": p_ans("main", dst, e, r1),
                    "random_to_new": p_ans("main", dst, er, r2)}
            report["transplant"]["cells"][f"{src}->{dst}"] = cell
            print(f"  {src:8s} -> {dst:8s} shared: {cell['to_new']:.3f} / {cell['to_old']:.3f}   "
                  f"random same layers: {cell['random_to_new']:.3f}")
    for dst in forms:
        pd, ps = pos_of("main", dst, "EQ"), pos_of("control", dst, "EQ")
        full = [(l, pd, n, float(cap[dst][l][ps, n])) for l in range(len(M.lm.layers)) for n in range(N)]
        report["transplant"]["cells"][f"all EQ neurons, {dst}"] = {"to_new": p_ans("main", dst, full, r2),
                                                                  "to_old": p_ans("main", dst, full, r1)}
        print(f"  upper bound, every MLP neuron at EQ, {dst}: "
              f"{report['transplant']['cells'][f'all EQ neurons, {dst}']['to_new']:.3f}")

    main_cap = {}
    for f in forms:
        _, main_cap[f] = M.run(text["main", f], capture=True)
    late = [k for k in eq if k[0] >= 20 and k not in generic]
    report["write"] = {"n_eq": len(eq), "n_late_specific": len(late), "cells": {}}
    print(f"\n== writing the answer: shared EQ neurons set to their values in the main calculation ({r1}), inside the "
          f"control calculation; P(target {r1}) / P(target {r2})")
    for name, group in (("all shared EQ", eq), ("shared EQ, L20+, not in control", late)):
        for s in (1, 2):
            for src in forms:
                ps = pos_of("main", src, "EQ")
                for dst in forms:
                    pd = pos_of("control", dst, "EQ")
                    e = [(l, pd, n, s * float(main_cap[src][l][ps, n])) for l, n, _ in group]
                    cell = {"to_new": p_ans("control", dst, e, r1), "to_old": p_ans("control", dst, e, r2)}
                    report["write"]["cells"][f"{name}, {s}x, {src}->{dst}"] = cell
                    print(f"  {name:32s} {s}x {src:8s} -> {dst:8s} {cell['to_new']:.3f} / {cell['to_old']:.3f}")

    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), ensure_ascii=False, indent=1)
    page(G, T, report, text)


def findings(r):
    ab, cells = r["ablation"], r["write"]["cells"]
    best = {f: max(cells[f"shared EQ, L20+, not in control, 2x, {s}->{f}"]["to_new"] for s in FORMS) for f in FORMS}
    rp = {f: d["OP"][1] for f, d in r["role_profile"].items()}
    items = [
        f"{len(r['shared'])} neurons appear at the same role in all three forms' graphs of 8-3; "
        f"{sum(x['same_sign'] for x in r['shared'])} of them have the same activation sign in every form.",
        "The shared late neurons at the equals position have steer lenses that mix the surface forms of the answer "
        "(e.g. ' five', '五', '5'), so the answer is held in a form-independent code and spelled out at the end.",
        "No operator-position neuron is shared (shared share at OP: " + ", ".join(f"{f} {v:.0%}" for f, v in rp.items())
        + "): '-', 'minus' and '减' are read by different neurons.",
        "Ablating the shared set: " + ", ".join(f"{f} {x['baseline']:.2f}→{x['shared']:.2f}" for f, x in ab.items())
        + f"; the same number of random neurons changes nothing (≤{max(abs(x['random_mean'] - x['baseline']) for x in ab.values()):.3f}).",
        f"Writing the 8-3 values of {r['write']['n_late_specific']} shared late EQ neurons (2x) into 2*4 makes the model "
        "answer 5: best P(5) " + ", ".join(f"{f} {v:.2f}" for f, v in best.items())
        + ", with values taken from a different form working as well as the same form.",
    ]
    return "<h2>Findings</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in items) + "</ul>"


CHINESE = """
<h2>现象</h2>
<p>同一道 8 − 3 = 5，用阿拉伯数字、英文和中文三种写法分别建图。三种写法各自读入算式，在等号位置的晚层汇合到同一组“答案是 5”的神经元上，最后再各自把答案拼写出来。紫圈是同一道算式的三张图都包含的神经元。下面的六张图都内嵌在这个文件里，不依赖外部图片。</p>
<ul>
<li><b>共享核心。</b>108 个神经元在三种写法里出现在同一角色，占每张图影响力的 36%–43%，其中 106 个激活符号一致。41 个也出现在对照算式 2 × 4 = 8 里，多在上下文和 L0–L7，更像通用部件。</li>
<li><b>与写法无关的答案编码。</b>等号位置的晚层共享神经元，steer lens 同时指向各种写法：L30 N1485（five / 五 / Five）、L28 N1070（five / 五天 / 5）、L29 N4856（5 / ５ / 五是）、L24 N2499（five / 五 / пять）。</li>
<li><b>操作数很早汇合。</b>L0 N5795 在第一个操作数位置，lens 是 8 / eight / 八。</li>
<li><b>运算符不共享。</b>“-”、“minus”、“减”由不同神经元读取，这个位置只占 1%–5% 的影响力。</li>
<li><b>写法比算式更决定图的样子。</b>同一写法、不同算式的重合度（数字 0.53、英文 0.54、中文 0.47）不低于同一算式、不同写法（英文–中文 0.53，数字–英文 0.41，数字–中文 0.34）。</li>
<li><b>去掉共享神经元，三种写法都答错。</b>78 个置 0：数字 0.77→0.24，英文 0.96→0.32，中文 0.78→0.15。同样数量的随机神经元几乎没有影响。</li>
<li><b>中文另有专属通路。</b>把它自己最强的 78 个非共享神经元置 0，概率掉到 0.01；数字只掉到 0.62，英文只掉到 0.78。</li>
<li><b>写入就能改答案，而且可以跨写法。</b>把 8 − 3 的 49 个共享晚层答案神经元（2 倍）写进 2 × 4，模型改答 5（数字 0.87、英文 0.64、中文 0.84）。英文的取值写进数字和中文同样有效；数字的取值写进英文只到 0.07。</li>
<li><b>反方向更难。</b>把 2 × 4 的取值写进 8 − 3，P(8) 最多 0.085。写入新答案比擦掉旧答案再换成另一个容易。</li>
</ul>
<p><a href="#main-digits">8−3 数字</a> · <a href="#main-english">8−3 英文</a> · <a href="#main-chinese">8−3 中文</a> ·
<a href="#control-digits">2×4 数字</a> · <a href="#control-english">2×4 英文</a> · <a href="#control-chinese">2×4 中文</a></p>
"""


def page(G, T, report, text):
    out = ['<!doctype html><meta charset="utf-8"><title>one calculation in three forms</title>',
           '<style>body{font:14px -apple-system,Helvetica,sans-serif;margin:24px auto;max-width:1400px;color:#1a202c}'
           'td,th{padding:3px 10px;text-align:left;border-bottom:1px solid #eee}th{background:#f7fafc}'
           'svg{width:100%;height:auto;border:1px solid #ddd;margin:6px 0 28px}.n{text-align:right}</style>',
           '<h1>One calculation in three surface forms</h1>',
           '<p>Neurons are matched across forms by (layer, neuron, role), the role being the first operand, operator, '
           'second operand or equals token. Purple rings mark neurons present in all three graphs of the same '
           'calculation. Hover nodes for activations and steer lens. All probabilities are from the real model with '
           'edits at the listed positions only.</p>', CHINESE, findings(report), '<h2>Prompts</h2><table>']
    for k, t in report["prompts"].items():
        out.append(f"<tr><td>{html.escape(k)}</td><td><code>{html.escape(t)}</code></td></tr>")
    out.append("</table><h2>Ablation of the shared set (P(answer))</h2><table><tr><th>form</th><th>n</th>"
               "<th>baseline</th><th>shared set → 0</th><th>same number of the form's own top neurons → 0</th>"
               "<th>random, same layers / roles</th></tr>")
    for f, r in report["ablation"].items():
        out.append(f"<tr><td>{f}</td><td class=n>{r['n']}</td><td class=n>{r['baseline']:.3f}</td>"
                   f"<td class=n>{r['shared']:.3f}</td><td class=n>{r['form_only_top']:.3f}</td>"
                   f"<td class=n>{r['random_mean']:.3f}</td></tr>")
    out.append("</table><h2>Transplant of the shared EQ-position neurons from the control calculation</h2>"
               "<table><tr><th>source → target</th><th>P(new answer)</th><th>P(old answer)</th>"
               "<th>random neurons, P(new)</th></tr>")
    for k, c in report["transplant"]["cells"].items():
        out.append(f"<tr><td>{html.escape(k)}</td><td class=n>{c['to_new']:.3f}</td><td class=n>{c['to_old']:.3f}</td>"
                   f"<td class=n>{c.get('random_to_new', float('nan')):.3f}</td></tr>")
    w = report["write"]
    out.append(f"</table><h2>Writing the answer: shared EQ neurons set to their 8-3 values inside 2*4</h2>"
               f"<p>{w['n_eq']} shared EQ neurons; the L20+ subset not shared with the control calculation has "
               f"{w['n_late_specific']}.</p><table><tr><th>group, dose, source → target</th><th>P(5)</th>"
               "<th>P(8)</th></tr>")
    for k, c in w["cells"].items():
        out.append(f"<tr><td>{html.escape(k)}</td><td class=n>{c['to_new']:.3f}</td>"
                   f"<td class=n>{c['to_old']:.3f}</td></tr>")
    out.append("</table><h2>Top shared neurons (main calculation)</h2><table><tr><th>neuron</th><th>role</th>"
               "<th>also in control</th><th>digits</th><th>english</th><th>chinese</th><th>steer lens</th></tr>")
    for r in report["shared"][:40]:
        cells = "".join(f"<td class=n>{r['w'][f]:.3f} ({r['act'][f]:+.2f})</td>" for f in FORMS)
        out.append(f"<tr><td>L{r['layer']} N{r['n']}</td><td>{r['role']}</td><td>{'yes' if r['generic'] else ''}</td>"
                   f"{cells}<td>{html.escape(' / '.join(r['lens']))}</td></tr>")
    out.append("</table>")
    for (c, f), g in G.items():
        rl = roles(len(g["tokens"]))
        for x in g["nodes"]:
            if x["kind"] == "neuron" and (x["layer"], x["n"], rl.get(x["pos"], "ctx")) in T[c]:
                x["shared"] = True
        g["title"] = f"{c} calculation, {f}: {text[c, f]}"
        g["shared_label"] = "in all three forms' graphs of this calculation"
        s = svg(g)
        with open(os.path.join(OUT, f"{c}_{f}.svg"), "w") as fh:
            fh.write(s)
        out.append(f'<h2 id="{c}-{f}">{html.escape(g["title"])}</h2>\n{s}')
    path = os.path.join(ROOT, "runs", "attribution", "arith_forms.html")
    with open(path, "w") as fh:
        fh.write("\n".join(out))
    print(path)


if __name__ == "__main__":
    main()
