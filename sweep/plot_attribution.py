"""Draw a neuron attribution graph as a standalone SVG in the circuit-tracer layout: token positions left to
right, layers bottom to top (embeddings at the bottom, the target logit on top), blue edges excite, red inhibit.

    python sweep/plot_attribution.py --text "Fact: the capital of the state containing Dallas is"   # via the explorer server
    python sweep/plot_attribution.py --json graph.json                                            # a saved /api/attribution reply

Hovering a node or edge in a browser shows its details."""

import argparse
import json
import os
import re
import urllib.request

from neuron_explorer.attribution import render_svg

svg = render_svg


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
