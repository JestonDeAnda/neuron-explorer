#!/usr/bin/env python3
"""Attribute the xLLM LoopedLM MoE Huginn checkpoint on the cognition23 tasks.

Routes, RMS denominators, and explicit attention patterns are frozen. Writes
version-3 graphs under ``<output-root>/output/{json,svg}/<task>/<sample>__<condition>``.
A current JSON file is never recomputed; a missing SVG is redrawn from it.
``--smoke`` gates one short condition at k=8, max_edges=40 and writes
``<output-root>/smoke.json`` instead of running the batch.
"""

from __future__ import annotations

import argparse


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local xLLM Huginn checkpoint directory")
    parser.add_argument("--xllm-source", required=True, help="pinned xLLM-Loop source checkout")
    parser.add_argument("--dataset", required=True, help="directory of per-task jsonl files")
    parser.add_argument("--output-root", required=True, help="durable root for output/, logs/, and status")
    parser.add_argument("--commit", default="unknown", help="project git commit recorded in every graph")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k", type=int, default=512, help="total-effect candidate expert neurons")
    parser.add_argument("--max-edges", type=int, default=3000, help="maximum retained direct edges")
    parser.add_argument("--benchmark-first", action="store_true", help="measure one median-length graph and choose edge copies")
    parser.add_argument("--limit", type=int, default=0, help="optional per-task row cap")
    parser.add_argument("--smoke", action="store_true", help="gate one short condition and write smoke.json; no batch")
    args = parser.parse_args(argv)
    from neuron_explorer.attribution.moe.xllm_huginn.runner import execute

    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
