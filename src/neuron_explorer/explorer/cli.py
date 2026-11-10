"""Serve the neuron explorer web application."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from neuron_explorer.paths import ProjectPaths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--root", type=Path, help="project data root")
    parser.add_argument("--runs", type=Path, help="sweep run directory")
    parser.add_argument("--stats", type=Path, help="neuron statistics directory")
    parser.add_argument("--captures", type=Path, help="activation capture directory")
    parser.add_argument("--cache", type=Path, help="explorer cache directory")
    parser.add_argument("--output", type=Path, help="generated output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.discover(args.root)
    overrides = {
        name: getattr(args, name)
        for name in ("runs", "stats", "captures", "cache", "output")
        if getattr(args, name) is not None
    }
    if overrides:
        paths = replace(paths, **overrides)

    import uvicorn

    from neuron_explorer.explorer.server import create_app

    uvicorn.run(create_app(paths), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
