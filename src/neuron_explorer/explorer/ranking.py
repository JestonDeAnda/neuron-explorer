"""Explorer-owned rollout ranking used while loading sweep results."""

import collections
import re

import numpy as np

TOKEN = re.compile(r"[^\W\d_]{3,}")


def words(text):
    return set(TOKEN.findall(text.lower()))


def distinct4(ids):
    grams = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
    return len(set(grams)) / len(grams) if grams else 1.0


def compute_units(rows, max_rel=1.6, top_words=12):
    """Return one ranking record per steered unit."""
    base_words = set().union(
        *(words(row["text"]) for row in rows if row["kind"] == "baseline")
    )
    units = collections.defaultdict(list)
    for row in rows:
        if row["kind"] != "baseline":
            units[(row["kind"], row["neuron"])].append(row)

    counts = {}
    for unit, records in units.items():
        count = collections.Counter()
        for row in records:
            if row["rel_push"] <= max_rel:
                count.update(words(row["text"]) - base_words)
        counts[unit] = count
    total = collections.Counter()
    for count in counts.values():
        total.update(count)
    mean = {word: value / len(counts) for word, value in total.items()}

    ranked_records = []
    for unit, records in units.items():
        count = counts[unit]
        ranked = sorted(
            (
                (
                    (frequency - mean[word]) / np.sqrt(mean[word] + 0.5),
                    word,
                    frequency,
                )
                for word, frequency in count.items()
                if frequency >= 2
            ),
            reverse=True,
        )[:top_words]
        per_prompt = {
            word: sorted(
                {row["prompt"] for row in records if word in words(row["text"])}
            )
            for _, word, _ in ranked[:5]
        }
        dose = {}
        for value in sorted({row["dose"] for row in records}):
            matching = [row for row in records if row["dose"] == value]
            dose[f"{value:+g}"] = {
                "rel_push": round(
                    float(np.mean([row["rel_push"] for row in matching])), 3
                ),
                "distinct4": round(
                    float(
                        np.mean([distinct4(row["token_ids"]) for row in matching])
                    ),
                    3,
                ),
                "capped": round(
                    float(
                        np.mean(
                            [row["finish"] == "length" for row in matching]
                        )
                    ),
                    3,
                ),
                "logprob": round(
                    float(
                        np.mean(
                            [
                                np.mean(row["logprob"])
                                if row["logprob"]
                                else 0.0
                                for row in matching
                            ]
                        )
                    ),
                    3,
                ),
            }
        ranked_records.append(
            {
                "kind": unit[0],
                "neuron": unit[1],
                "layer": records[0]["layer"],
                "ref_neuron": records[0]["ref_neuron"],
                "z": round(float(ranked[0][0]), 3) if ranked else 0.0,
                "mass": round(
                    float(sum(max(score, 0.0) for score, _, _ in ranked[:5])), 3
                ),
                "words": [
                    [word, frequency, round(float(score), 2)]
                    for score, word, frequency in ranked
                ],
                "per_prompt": per_prompt,
                "dose": dose,
            }
        )

    null = np.sort(
        [record["mass"] for record in ranked_records if record["kind"] == "random"]
    )
    for record in ranked_records:
        record["null_pct"] = (
            round(float(np.searchsorted(null, record["mass"]) / len(null)), 4)
            if len(null)
            else None
        )
    ranked_records.sort(key=lambda record: -record["mass"])
    return ranked_records
