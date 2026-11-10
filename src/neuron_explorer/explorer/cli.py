"""Serve the neuron explorer web application."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from neuron_explorer.paths import ProjectPaths


def run_server(args) -> int:
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

    uvicorn.run(create_app(paths, port=args.port), host=args.host, port=args.port)
    return 0
