"""Depth curve from per-layer random-subset sweeps: how many neurons steer the model's text further from baseline
than random directions do, per layer, plus coherence numbers per dose and each layer's strongest neurons.

    uv run python sweep/depth_report.py --out runs/depth_report.json

Ranks runs/q35-4b-L{l}-r384.jsonl, writing <run>.units.jsonl. Layers with only a full run (-full.jsonl) are subsampled
to the same pool size (384 neurons, 64 controls) so word-cluster mass is comparable across layers.
"""
import argparse
import json
import os
import random

import numpy as np

from neuron_explorer.sweep.rank import compute_units


def build_depth_report(layers, neurons=384, controls=64, runs_dir="runs"):
    """Compute the per-layer sweep depth report."""
    lo, _, hi = layers.partition("-")
    report = []
    for layer in range(int(lo), int(hi or lo) + 1):
        path = os.path.join(runs_dir, f"q35-4b-L{layer}-r{neurons}.jsonl")
        subsampled = False
        if not os.path.exists(path):
            path = os.path.join(runs_dir, f"q35-4b-L{layer}-full.jsonl")
            subsampled = True
            if not os.path.exists(path):
                continue
        rows = [json.loads(line) for line in open(path)]
        if subsampled:
            rng = random.Random(layer)
            neuron_ids = sorted(
                {row["neuron"] for row in rows if row["kind"] == "neuron"}
            )
            control_ids = sorted(
                {row["neuron"] for row in rows if row["kind"] == "random"}
            )
            keep = {
                ("neuron", value)
                for value in rng.sample(neuron_ids, neurons)
            } | {
                ("random", value)
                for value in rng.sample(control_ids, controls)
            }
            rows = [
                row
                for row in rows
                if row["kind"] == "baseline"
                or (row["kind"], row["neuron"]) in keep
            ]
        records = compute_units(rows)
        if not subsampled:
            with open(path.replace(".jsonl", ".units.jsonl"), "w") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
        null = np.array(
            [record["mass"] for record in records if record["kind"] == "random"]
        )
        neuron_records = [
            record for record in records if record["kind"] == "neuron"
        ]
        masses = np.array([record["mass"] for record in neuron_records])
        p95 = np.percentile(null, 95)

        def coherence(kind, dose):
            values = [
                record["dose"].get(dose)
                for record in records
                if record["kind"] == kind
            ]
            values = [value for value in values if value]
            return {
                key: round(float(np.mean([value[key] for value in values])), 3)
                for key in ("distinct4", "capped", "logprob")
            }

        report.append(
            {
                "layer": layer,
                "subsampled": subsampled,
                "n_neurons": len(neuron_records),
                "n_controls": len(null),
                "null_p50": round(float(np.median(null)), 2),
                "null_p95": round(float(p95), 2),
                "null_max": round(float(null.max()), 2),
                "neuron_p50": round(float(np.median(masses)), 2),
                "neuron_p90": round(float(np.percentile(masses, 90)), 2),
                "above_p95": round(float((masses > p95).mean()), 3),
                "above_max": round(float((masses > null.max()).mean()), 3),
                "coh_neuron_+0.8": coherence("neuron", "+0.8"),
                "coh_random_+0.8": coherence("random", "+0.8"),
                "coh_neuron_-0.7": coherence("neuron", "-0.7"),
                "top": [
                    {
                        "n": record["neuron"],
                        "mass": record["mass"],
                        "words": [
                            word for word, _, _ in record["words"][:6]
                        ],
                    }
                    for record in neuron_records[:5]
                ],
            }
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="0-31")
    parser.add_argument("--neurons", type=int, default=384)
    parser.add_argument("--controls", type=int, default=64)
    parser.add_argument("--out", default="runs/depth_report.json")
    args = parser.parse_args(argv)
    report = build_depth_report(args.layers, args.neurons, args.controls)
    for row in report:
        print(
            f"L{row['layer']:<2}{'*' if row['subsampled'] else ' '} "
            f"null p50/p95 {row['null_p50']:5.1f}/{row['null_p95']:5.1f}  "
            f"neuron p50/p90 {row['neuron_p50']:5.1f}/{row['neuron_p90']:5.1f}  "
            f">p95 {row['above_p95']:.0%}  >max {row['above_max']:.0%}  "
            f"d4(+.8) n {row['coh_neuron_+0.8']['distinct4']:.2f} "
            f"r {row['coh_random_+0.8']['distinct4']:.2f}  | "
            + " ; ".join(
                f"{top['n']}:{' '.join(top['words'][:4])}"
                for top in row["top"][:3]
            ),
            flush=True,
        )
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
