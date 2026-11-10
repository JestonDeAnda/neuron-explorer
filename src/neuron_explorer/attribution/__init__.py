"""Attribution graph persistence and annotation tools."""

from typing import Any

from neuron_explorer.attribution.annotations import (
    NeuronAnnotation,
    NeuronCatalog,
    annotate_graph,
)
from neuron_explorer.attribution.catalogs import (
    CacheCatalog,
    CompositeCatalog,
    DatasetCatalog,
    EmptyCatalog,
)
from neuron_explorer.attribution.render_svg import (
    layout_graph,
    render_svg,
    select_graph,
)
from neuron_explorer.attribution.schema import load_graph, save_graph, validate_graph


def __getattr__(name: str) -> Any:
    """Load the torch-backed graph builder only when it is requested."""
    if name in {"Linearized", "build_graph"}:
        from neuron_explorer.attribution.graph import Linearized, build_graph

        return {"Linearized": Linearized, "build_graph": build_graph}[name]
    raise AttributeError(name)

__all__ = [
    "CacheCatalog",
    "CompositeCatalog",
    "DatasetCatalog",
    "EmptyCatalog",
    "Linearized",
    "NeuronAnnotation",
    "NeuronCatalog",
    "annotate_graph",
    "build_graph",
    "layout_graph",
    "load_graph",
    "render_svg",
    "save_graph",
    "select_graph",
    "validate_graph",
]
