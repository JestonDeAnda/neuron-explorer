#!/usr/bin/env python3
"""Render a saved or remotely generated attribution graph."""

import argparse
from pathlib import Path


class CatalogAction(argparse.Action):
    def __init__(self, *args, catalog_kind=None, **kwargs):
        self.catalog_kind = catalog_kind
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        entries = list(getattr(namespace, self.dest, None) or [])
        value = str(values)
        if self.catalog_kind is None:
            try:
                kind, path = value.split(":", 1)
            except ValueError:
                parser.error(f"{option_string} requires TYPE:PATH")
            if kind not in {"dataset", "cache"}:
                parser.error(f"unsupported catalog type: {kind!r}")
        else:
            kind, path = self.catalog_kind, value
        entries.append((kind, Path(path)))
        setattr(namespace, self.dest, entries)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--json", type=Path)
    parser.add_argument("--token")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--url", default="http://127.0.0.1:8791")
    parser.add_argument("--edges", type=int, default=300)
    parser.add_argument("--labels", type=int, default=40)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--catalog", dest="catalogs", action=CatalogAction)
    parser.add_argument(
        "--dataset-catalog",
        dest="catalogs",
        action=CatalogAction,
        catalog_kind="dataset",
    )
    parser.add_argument(
        "--cache-catalog",
        dest="catalogs",
        action=CatalogAction,
        catalog_kind="cache",
    )
    parser.add_argument("--no-annotations", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.no_annotations and args.catalogs:
        parser.error("--no-annotations cannot be combined with annotation catalogs")
    from neuron_explorer.paths import ProjectPaths
    from neuron_explorer.attribution.cli import run_plot_attribution

    paths = ProjectPaths.discover(args.root)
    try:
        result = run_plot_attribution(
            text=args.text,
            json_path=args.json,
            token=args.token,
            chat=args.chat,
            k=args.k,
            url=args.url,
            edges=args.edges,
            labels=args.labels,
            output=args.out,
            runs_dir=paths.runs,
            catalogs=args.catalogs,
            no_annotations=args.no_annotations,
        )
    except ValueError as error:
        parser.error(str(error))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
