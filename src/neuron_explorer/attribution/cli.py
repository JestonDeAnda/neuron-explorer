"""Command-line orchestration for attribution graph rendering."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from neuron_explorer.attribution.annotations import annotate_graph
from neuron_explorer.attribution.catalogs import (
    CacheCatalog,
    CompositeCatalog,
    DatasetCatalog,
    EmptyCatalog,
)
from neuron_explorer.attribution.render_svg import render_svg
from neuron_explorer.attribution.schema import load_graph, save_graph, validate_graph


class _CatalogAction(argparse.Action):
    def __init__(self, *args, catalog_kind: str | None = None, **kwargs):
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a saved attribution graph or fetch one from an explorer server."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="prompt to send to the explorer server")
    source.add_argument("--json", type=Path, help="saved attribution graph")
    parser.add_argument("--token", help="target token (default: model top-1)")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--url", default="http://127.0.0.1:8791")
    parser.add_argument("--edges", type=int, default=300)
    parser.add_argument("--labels", type=int, default=40)
    parser.add_argument("--out", type=Path, help="output SVG path")
    parser.add_argument(
        "--catalog",
        dest="catalogs",
        action=_CatalogAction,
        metavar="TYPE:PATH",
        help="ordered annotation source; TYPE is dataset or cache",
    )
    parser.add_argument(
        "--dataset-catalog",
        dest="catalogs",
        action=_CatalogAction,
        catalog_kind="dataset",
        metavar="PATH",
    )
    parser.add_argument(
        "--cache-catalog",
        dest="catalogs",
        action=_CatalogAction,
        catalog_kind="cache",
        metavar="DIR",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="render without looking up neuron metadata",
    )
    return parser


def _fetch(args: argparse.Namespace) -> dict[str, Any]:
    body: dict[str, Any] = {"text": args.text, "k": args.k, "chat": args.chat}
    if args.token:
        body["token"] = args.token
    request = urllib.request.Request(
        args.url.rstrip("/") + "/api/attribution",
        json.dumps(body).encode(),
        {"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        graph = json.load(response)
    if "error" in graph:
        raise SystemExit(graph["error"])
    validate_graph(graph)
    return graph


def _catalog(entries: list[tuple[str, Path]]):
    catalogs = []
    for kind, path in entries:
        if kind == "dataset":
            catalogs.append(DatasetCatalog.from_path(path))
        else:
            catalogs.append(
                CacheCatalog(
                    path / "names.json",
                    path / "steer_index.npz",
                    path / "vocab.json",
                    path / "lens_index.npz",
                )
            )
    if not catalogs:
        return EmptyCatalog()
    return catalogs[0] if len(catalogs) == 1 else CompositeCatalog(*catalogs)


def _default_output(graph: dict[str, Any]) -> Path:
    slug = re.sub(r"\W+", "_", "".join(graph.get("tokens", []))).strip("_")[:60]
    return Path("runs") / "attribution" / f"{slug or 'graph'}.svg"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.no_annotations and args.catalogs:
        parser.error("--no-annotations cannot be combined with annotation catalogs")

    graph = load_graph(args.json) if args.json else _fetch(args)
    enriched = graph
    if args.catalogs and not args.no_annotations:
        if "tokens" not in graph or "target" not in graph:
            parser.error("annotation requires graph tokens and target fields")
        enriched = annotate_graph(
            graph,
            tokens=graph["tokens"],
            target=graph["target"],
            catalog=_catalog(args.catalogs),
        )
        if args.json and enriched != graph:
            save_graph(enriched, args.json)

    output = args.out or _default_output(enriched)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_svg(enriched, args.edges, args.labels),
        encoding="utf-8",
    )
    if not args.json:
        save_graph(enriched, output.with_suffix(".json"))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
