"""Word-shift score per unit, measured against the random-direction controls of the same run. Every word counts, including
words the unsteered baselines use, so units that move common discourse words ("however", "might", "specifically")
score alongside units that introduce new vocabulary.

For each unit and word: presence count over the unit's rollouts with push <= max_rel, z = (count - mu) / sqrt(mu + 0.5),
mu = mean count over the controls (leave-one-out when the unit is itself a control). Words that fall count as well:
shift = summed |z| of the unit's 5 largest rises + 5 largest falls.

    from shift import shift_scores; scores = shift_scores(rows)   # {(kind, neuron): {"shift", "up", "down"}}
"""
import collections
import re

import numpy as np

TOKEN = re.compile(r"[^\W\d_]{3,}")


def shift_scores(rows, max_rel=1.6, k=5):
    units = collections.defaultdict(collections.Counter)
    n_roll = collections.Counter()
    for r in rows:
        if r["kind"] == "baseline" or r["rel_push"] > max_rel:
            continue
        u = (r["kind"], r["neuron"])
        units[u].update(set(TOKEN.findall(r["text"].lower())))
        n_roll[u] += 1
    ctl = [u for u in units if u[0] == "random"]
    tot = collections.Counter()
    for u in ctl:
        tot.update(units[u])
    nc = len(ctl)
    out = {}
    for u, c in units.items():
        loo = u[0] == "random"
        denom = nc - 1 if loo else nc
        zs = []
        for w in set(c) | set(tot):
            mu = (tot[w] - (c[w] if loo else 0)) / denom
            if mu == 0 and c[w] < 2:
                continue
            zs.append(((c[w] - mu) / np.sqrt(mu + 0.5), w))
        zs.sort()
        up = [(w, round(z, 2)) for z, w in zs[::-1][:k] if z > 0]
        down = [(w, round(z, 2)) for z, w in zs[:k] if z < 0]
        out[u] = {"shift": round(sum(z for _, z in up) - sum(z for _, z in down), 3), "up": up, "down": down}
    return out
