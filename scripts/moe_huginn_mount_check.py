#!/usr/bin/env python3
"""Validate xLLM Huginn dataset, checkpoint, source, stage, and output mounts."""

from __future__ import annotations

import argparse
import json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--xllm-source", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-tasks", type=int, default=23)
    parser.add_argument("--expected-rows", type=int, default=1461)
    args = parser.parse_args(argv)

    from neuron_explorer.attribution.moe.xllm_huginn.mount import check_moe_mounts

    report = check_moe_mounts(
        dataset=args.dataset,
        model=args.model,
        xllm_source=args.xllm_source,
        stage=args.stage,
        commit=args.commit,
        output_root=args.output_root,
        expected_tasks=args.expected_tasks,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
