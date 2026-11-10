"""Validation and JSON persistence for attribution graphs."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
NODE_KINDS = frozenset({"neuron", "embed", "logit"})
NODE_FIELDS = frozenset({"kind", "layer", "pos", "n", "act", "influence"})
GRAPH_FIELDS = frozenset({"token", "prob", "nodes", "edges", "stats"})


def validate_graph(graph: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when *graph* is not a supported attribution graph."""
    if not isinstance(graph, Mapping):
        raise ValueError("graph must be a mapping")

    version = graph.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {version!r}")

    missing = GRAPH_FIELDS - graph.keys()
    if missing:
        raise ValueError(f"graph is missing required fields: {sorted(missing)}")

    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list):
        raise ValueError("graph nodes must be a list")
    if not isinstance(edges, list):
        raise ValueError("graph edges must be a list")

    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            raise ValueError(f"node {index} must be a mapping")
        missing = NODE_FIELDS - node.keys()
        if missing:
            raise ValueError(f"node {index} is missing fields: {sorted(missing)}")
        if node["kind"] not in NODE_KINDS:
            raise ValueError(f"node {index} has invalid kind: {node['kind']!r}")

    for index, edge in enumerate(edges):
        if not isinstance(edge, (list, tuple)) or len(edge) != 3:
            raise ValueError(f"edge {index} must contain source, destination, and weight")
        source, destination, weight = edge
        if (
            not isinstance(source, int)
            or isinstance(source, bool)
            or not isinstance(destination, int)
            or isinstance(destination, bool)
            or not 0 <= source < len(nodes)
            or not 0 <= destination < len(nodes)
        ):
            raise ValueError(f"edge {index} node index is out of bounds")
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(weight)
        ):
            raise ValueError(f"edge {index} weight must be a finite number")


def load_graph(path: Path) -> dict[str, Any]:
    """Load and validate a graph, treating unversioned current graphs as version 1."""
    with Path(path).open(encoding="utf-8") as file:
        graph = json.load(file)
    if not isinstance(graph, dict):
        raise ValueError("graph must be a JSON object")
    graph.setdefault("schema_version", SCHEMA_VERSION)
    validate_graph(graph)
    return graph


def save_graph(graph: Mapping[str, Any], path: Path) -> Path:
    """Validate and atomically save *graph* as UTF-8 JSON."""
    saved_graph = dict(graph)
    saved_graph["schema_version"] = SCHEMA_VERSION
    validate_graph(saved_graph)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(saved_graph, file, ensure_ascii=False)
            file.write("\n")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination
