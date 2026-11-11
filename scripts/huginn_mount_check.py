#!/usr/bin/env python3
"""Verify dataset, Huginn checkpoint, staged code, and DLC mounts before GPU work."""

from __future__ import annotations

import argparse


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage", default=None, help="directory containing the staged commit")
    parser.add_argument("--commit", default=None, help="expected COMMIT file contents")
    parser.add_argument("--expected-tasks", type=int, default=23)
    parser.add_argument("--expected-rows", type=int, default=1461)
    parser.add_argument("--mounts", action="store_true", help="also check the four standard DLC mounts")
    args = parser.parse_args(argv)
    from neuron_explorer.attribution.huginn.mount import main_check

    return main_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
