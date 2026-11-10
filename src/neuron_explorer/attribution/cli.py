"""Command-line orchestration for attribution graph rendering."""

from __future__ import annotations

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
from neuron_explorer.paths import ProjectPaths


def fetch_graph(
    *,
    text: str,
    token: str | None = None,
    chat: bool = False,
    k: int = 512,
    url: str = "http://127.0.0.1:8791",
) -> dict[str, Any]:
    body: dict[str, Any] = {"text": text, "k": k, "chat": chat}
    if token:
        body["token"] = token
    request = urllib.request.Request(
        url.rstrip("/") + "/api/attribution",
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


def _default_output(graph: dict[str, Any], runs_dir: Path) -> Path:
    slug = re.sub(r"\W+", "_", "".join(graph.get("tokens", []))).strip("_")[:60]
    return runs_dir / "attribution" / f"{slug or 'graph'}.svg"


def run_plot_attribution(
    *,
    text: str | None = None,
    json_path: Path | None = None,
    token: str | None = None,
    chat: bool = False,
    k: int = 512,
    url: str = "http://127.0.0.1:8791",
    edges: int = 300,
    labels: int = 40,
    output: Path | None = None,
    runs_dir: Path | None = None,
    catalogs: list[tuple[str, Path]] | None = None,
    no_annotations: bool = False,
) -> Path:
    graph = (
        load_graph(json_path)
        if json_path
        else fetch_graph(text=text or "", token=token, chat=chat, k=k, url=url)
    )
    enriched = graph
    if catalogs and not no_annotations:
        if "tokens" not in graph or "target" not in graph:
            raise ValueError("annotation requires graph tokens and target fields")
        enriched = annotate_graph(
            graph,
            tokens=graph["tokens"],
            target=graph["target"],
            catalog=_catalog(catalogs),
        )
        if json_path and enriched != graph:
            save_graph(enriched, json_path)

    destination = output or _default_output(
        enriched, runs_dir or ProjectPaths.discover().runs
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not json_path:
        save_graph(enriched, destination.with_suffix(".json"))
    destination.write_text(
        render_svg(enriched, edges, labels),
        encoding="utf-8",
    )
    return destination
