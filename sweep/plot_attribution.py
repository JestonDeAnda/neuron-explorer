"""Draw a neuron attribution graph as a standalone SVG in the circuit-tracer layout: token positions left to
right, layers bottom to top (embeddings at the bottom, the target logit on top), blue edges excite, red inhibit.

    python sweep/plot_attribution.py --text "Fact: the capital of the state containing Dallas is"   # via the explorer server
    python sweep/plot_attribution.py --json graph.json                                            # a saved /api/attribution reply

Hovering a node or edge in a browser shows its details."""

import argparse
import html
import json
import math
import os
import re
import urllib.request

ROW = 22
CELL = 11
PAD_L, PAD_R, PAD_T, PAD_B = 56, 40, 70, 110


def esc(s):
    return html.escape(re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(s)))


def fetch(url, text, token, k, chat):
    body = {"text": text, "k": k, "chat": chat}
    if token:
        body["token"] = token
    req = urllib.request.Request(url.rstrip("/") + "/api/attribution", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        g = json.load(r)
    if "error" in g:
        raise SystemExit(g["error"])
    return g


def select(g, max_edges, labels):
    nodes, edges = g["nodes"], g["edges"]
    incoming = [0.0] * len(nodes)
    for s, d, w in edges:
        incoming[d] += abs(w)
    for i, n in enumerate(nodes):
        if n["kind"] == "neuron":
            incoming[i] += abs(n.get("other") or 0)

    def weight(i):
        return 1.0 if nodes[i]["kind"] == "logit" else nodes[i]["influence"]

    score = [abs(w) / (incoming[d] or 1) * weight(d) for s, d, w in edges]
    order = sorted(range(len(edges)), key=lambda e: -score[e])[:max_edges]
    keep = {i for e in order for i in edges[e][:2]}
    shown = [(edges[e], score[e]) for e in order]
    named = sorted((i for i in keep if nodes[i]["kind"] == "neuron"), key=lambda i: -nodes[i]["influence"])[:labels]
    named += [i for i in keep if nodes[i].get("clamped")][:labels]
    return keep, shown, set(named)


def layout(g, keep):
    nodes, toks = g["nodes"], g["tokens"]
    top = max(n["layer"] for n in nodes)
    cells = {}
    for i in sorted(keep, key=lambda i: -nodes[i]["influence"]):
        n = nodes[i]
        cells.setdefault((n["layer"], n["pos"]), []).append(i)
    widths = [max(64, 14 + CELL * max([len(v) for (l, p), v in cells.items() if p == t] or [0]))
              for t in range(len(toks))]
    x0 = [PAD_L]
    for w in widths:
        x0.append(x0[-1] + w)
    xy = {}
    for (layer, pos), ids in cells.items():
        y = PAD_T + (top - layer) * ROW
        mid = x0[pos] + widths[pos] / 2
        for j, i in enumerate(ids):
            xy[i] = (mid + (j - (len(ids) - 1) / 2) * CELL, y)
    return xy, x0, widths, top


def svg(g, max_edges=300, labels=40):
    nodes, toks = g["nodes"], g["tokens"]
    keep, shown, named = select(g, max_edges, labels)
    xy, x0, widths, top = layout(g, keep)
    W = x0[-1] + PAD_R + 170
    title = g.get("title")
    PAD = 22 if title else 0
    H = PAD_T + (top + 1) * ROW + PAD_B + PAD
    peak = max(s for _, s in shown) or 1
    imax = max(nodes[i]["influence"] for i in keep if nodes[i]["kind"] != "logit") or 1
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
           f'font-family="-apple-system,Helvetica,Arial,sans-serif" font-size="11">',
           '<style>.e{fill:none}.e:hover{stroke-opacity:1;stroke-width:3}.n:hover{stroke:#000;stroke-width:2}</style>',
           f'<rect width="100%" height="100%" fill="#fff"/>', f'<g transform="translate(0,{PAD})">']
    if title:
        out.append(f'<text x="{PAD_L}" y="2" font-size="14" fill="#c05621" font-weight="600">{esc(title)}</text>')

    st = g["stats"]
    tgt = esc(repr(g.get("target", g["token"])))
    out.append(f'<text x="{PAD_L}" y="24" font-size="15" font-weight="600">→ {tgt}  p={g["prob"]:.2f}</text>')
    out.append(f'<text x="{PAD_L}" y="44" fill="#555">{st["kept_neurons"]} neurons from {st["candidates"]} '
               f'candidates · candidates explain {st["logit_from_candidates"]:.0%} of the logit directly · '
               f'top {len(shown)} of {st["kept_edges"]} edges shown</text>')

    for layer in range(-1, top + 1):
        y = PAD_T + (top - layer) * ROW
        if layer % 4 == 3 or layer in (-1, top):
            name = "Emb" if layer < 0 else "Logit" if layer == top else f"L{layer}"
            out.append(f'<text x="{PAD_L - 8}" y="{y + 4}" text-anchor="end" fill="#888">{name}</text>')
            out.append(f'<line x1="{PAD_L}" x2="{x0[-1]}" y1="{y}" y2="{y}" stroke="#f0f0f0"/>')
    base = PAD_T + (top + 1) * ROW
    for t, tok in enumerate(toks):
        cx = x0[t] + widths[t] / 2
        if t:
            out.append(f'<line x1="{x0[t]}" x2="{x0[t]}" y1="{PAD_T - 10}" y2="{base}" stroke="#f4f4f4"/>')
        label = esc(tok.replace("\n", "⏎"))
        out.append(f'<text transform="translate({cx + 4},{base + 14}) rotate(-45)" text-anchor="end" '
                   f'font-size="12" fill="#333">{label}</text>')

    for (s, d, w), sc in sorted(shown, key=lambda e: e[1]):
        (xs, ys), (xd, yd) = xy[s], xy[d]
        a = sc / peak
        col = "#2b6cb0" if w > 0 else "#c53030"
        bend = (yd - ys) * 0.45
        tip = f"{node_name(nodes[s])} → {node_name(nodes[d])}: {w:+.3g}"
        out.append(f'<path class="e" d="M{xs:.1f},{ys:.1f} C{xs:.1f},{ys + bend:.1f} {xd:.1f},{yd - bend:.1f} '
                   f'{xd:.1f},{yd:.1f}" stroke="{col}" stroke-opacity="{0.15 + 0.75 * a:.2f}" '
                   f'stroke-width="{0.4 + 3.6 * a:.2f}"><title>{esc(tip)}</title></path>')

    labeled = []
    for i in sorted(keep, key=lambda i: nodes[i]["influence"] if nodes[i]["kind"] != "logit" else 9):
        n, (x, y) = nodes[i], xy[i]
        tip = esc(node_tip(n))
        if n["kind"] == "logit":
            pts = " ".join(f"{x + 9 * math.sin(a):.1f},{y - 9 * math.cos(a):.1f}"
                           for a in (2 * math.pi * j / 5 for j in range(5)))
            out.append(f'<polygon class="n" points="{pts}" fill="#d69e2e" stroke="#744210"><title>{tip}</title></polygon>')
            out.append(f'<text x="{x + 14}" y="{y + 4}" font-weight="600">{tgt}</text>')
            continue
        r = 3 + 5 * math.sqrt(n["influence"] / imax)
        if n["kind"] == "embed":
            out.append(f'<rect class="n" x="{x - r}" y="{y - r}" width="{2 * r}" height="{2 * r}" fill="#a0aec0" '
                       f'stroke="#4a5568"><title>{tip}</title></rect>')
            continue
        fill = "#9ae6b4" if (n.get("act") or 0) > 0 else "#fbb6ce"
        if n.get("gate") is not None and n["gate"] < -1.278:
            fill = "#e2e8f0"
        ring = ('stroke="#dd6b20" stroke-width="2.6"' if n.get("clamped") else 'stroke="#2d3748" stroke-width="0.8"')
        out.append(f'<circle class="n" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" {ring}>'
                   f'<title>{tip}</title></circle>')
        if i in named:
            labeled.append((n["influence"], x, y, r, n))

    taken = [(x - 4, y - 4, 8, 8) for x, y in xy.values()]
    for _, x, y, r, n in sorted(labeled, key=lambda t: (not t[4].get("clamped"), -t[0])):
        if n.get("clamped"):
            text = f'L{n["layer"]}.{n["n"]} := {n["act"]:+.2f}'
        else:
            text = f'L{n["layer"]}.{n["n"]} ' + (n.get("name") or "/".join(n.get("lens", [])[:2]))[:18]
        w = 5.6 * len(text)
        for bx, anchor in ((x + r + 2, "start"), (x - r - 2 - w, "end")):
            box = (bx, y - r - 10, w, 11)
            if not any(overlap(box, t) for t in taken if t[2] > 8 or not near(t, x, y)):
                taken.append(box)
                tx = bx if anchor == "start" else bx + w
                out.append(f'<text x="{tx:.1f}" y="{y - r:.1f}" font-size="9.5" text-anchor="{anchor}" '
                           f'fill="#1a202c" paint-order="stroke" stroke="#fff" stroke-width="2.5">{esc(text)}</text>')
                break

    ly = H - 22 - PAD
    items = [("#9ae6b4", "circle", "neuron, act > 0"), ("#fbb6ce", "circle", "neuron, act < 0"),
             ("#e2e8f0", "circle", "gate < −1.278 (gate path sign unreliable)"), ("#a0aec0", "rect", "token embedding")]
    if any(n.get("clamped") for n in nodes):
        items.append(("#fff", "ring", "clamped by the intervention (no inputs)"))
    lx = PAD_L
    for col, shape, label in items:
        if shape == "ring":
            out.append(f'<circle cx="{lx + 5}" cy="{ly}" r="5" fill="{col}" stroke="#dd6b20" stroke-width="2.6"/>')
        elif shape == "circle":
            out.append(f'<circle cx="{lx + 5}" cy="{ly}" r="5" fill="{col}" stroke="#2d3748" stroke-width="0.8"/>')
        else:
            out.append(f'<rect x="{lx}" y="{ly - 5}" width="10" height="10" fill="{col}" stroke="#4a5568"/>')
        out.append(f'<text x="{lx + 14}" y="{ly + 4}" fill="#555">{esc(label)}</text>')
        lx += 24 + 6.2 * len(label)
    for col, label in (("#2b6cb0", "excites"), ("#c53030", "inhibits")):
        out.append(f'<line x1="{lx}" x2="{lx + 22}" y1="{ly}" y2="{ly}" stroke="{col}" stroke-width="2.5"/>')
        out.append(f'<text x="{lx + 28}" y="{ly + 4}" fill="#555">{label}</text>')
        lx += 90
    out.append("</g></svg>")
    return "\n".join(out)


def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def near(box, x, y):
    return abs(box[0] + 4 - x) < 0.5 and abs(box[1] + 4 - y) < 0.5


def node_name(n):
    if n["kind"] == "embed":
        return f"emb {n['tok']!r}"
    if n["kind"] == "logit":
        return "logit"
    return f"L{n['layer']} N{n['n']}@{n['pos']}"


def node_tip(n):
    if n["kind"] == "logit":
        return f"target logit"
    head = f"{node_name(n)}  token {n['tok']!r}  influence {n['influence']:.3f}"
    if n["kind"] == "embed":
        return head
    extra = f"\nact {n['act']:+.3f}" + ("  (clamped)" if n.get("clamped") else f"  gate {n['gate']:+.2f}")
    if n.get("name"):
        extra += f"\nname: {n['name']}"
    if n.get("lens"):
        extra += "\nsteer lens: " + ", ".join(n["lens"])
    return head + extra


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text")
    ap.add_argument("--json", help="read a saved graph instead of calling the server")
    ap.add_argument("--token", help="target token (default: the model's top-1)")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--url", default="http://127.0.0.1:8791")
    ap.add_argument("--edges", type=int, default=300, help="edges drawn, ranked by share of the target's influence")
    ap.add_argument("--labels", type=int, default=40, help="most influential neurons to label, where there is room")
    ap.add_argument("--out", help="output .svg (default runs/attribution/<prompt>.svg)")
    a = ap.parse_args()
    if not (a.text or a.json):
        ap.error("give --text or --json")
    g = json.load(open(a.json)) if a.json else fetch(a.url, a.text, a.token, a.k, a.chat)
    out = a.out
    if not out:
        slug = re.sub(r"\W+", "_", "".join(g["tokens"])).strip("_")[:60] or "graph"
        out = os.path.join("runs", "attribution", slug + ".svg")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if not a.json:
        with open(out[:-4] + ".json", "w") as f:
            json.dump(g, f, ensure_ascii=False)
    with open(out, "w") as f:
        f.write(svg(g, a.edges, a.labels))
    print(out)


if __name__ == "__main__":
    main()
