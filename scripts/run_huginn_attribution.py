#!/usr/bin/env python3
"""Attribute Huginn-0125 on the cognition23 original-text tasks.

Writes one graph per prompt/completion condition under
``<output-root>/output/{json,svg}/<task>/<sample>__<condition>``.
JSON is validated before SVG. A complete JSON file is not recomputed.
"""

from __future__ import annotations

import argparse


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local Huginn checkpoint directory")
    parser.add_argument("--dataset", required=True, help="directory of per-task jsonl files")
    parser.add_argument("--output-root", required=True, help="durable root for output/, logs/, and status")
    parser.add_argument("--commit", default="unknown", help="git commit recorded in every graph")
    parser.add_argument("--k", type=int, default=512, help="total-effect candidate neurons")
    parser.add_argument("--max-edges", type=int, default=3000, help="maximum retained direct edges")
    parser.add_argument("--benchmark-first", action="store_true", help="time one median-length graph before the batch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="optional per-task row cap for smoke runs")
    args = parser.parse_args(argv)
    from neuron_explorer.attribution.huginn.runner import execute

    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
