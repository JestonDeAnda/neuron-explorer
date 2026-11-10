"""Catalog values and graph annotation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class NeuronAnnotation:
    """Interpretability metadata for one neuron."""

    explanation: str | None = None
    steer_up: tuple[str, ...] = ()
    steer_down: tuple[str, ...] = ()
    kl_up: float | None = None
    kl_down: float | None = None
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "steer_up", tuple(str(token) for token in self.steer_up))
        object.__setattr__(
            self, "steer_down", tuple(str(token) for token in self.steer_down)
        )
        if self.provenance is not None:
            object.__setattr__(self, "provenance", _freeze(self.provenance))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible copy of all annotation metadata."""
        return {
            "explanation": self.explanation,
            "steer_up": list(self.steer_up),
            "steer_down": list(self.steer_down),
            "kl_up": self.kl_up,
            "kl_down": self.kl_down,
            "provenance": _thaw(self.provenance),
        }


class NeuronCatalog(Protocol):
    """Lookup interface for neuron interpretation metadata."""

    def lookup(self, layer: int, neuron: int) -> NeuronAnnotation | None:
        """Return metadata for ``(layer, neuron)`` when available."""


def annotate_graph(
    graph: Mapping[str, Any],
    *,
    tokens: Sequence[str],
    target: str,
    catalog: NeuronCatalog | None,
    lens_tokens: int = 5,
) -> dict[str, Any]:
    """Return an annotated deep copy of *graph*."""
    if lens_tokens < 0:
        raise ValueError("lens_tokens must be non-negative")

    annotated = deepcopy(dict(graph))
    token_strings = [str(token) for token in tokens]
    annotated["tokens"] = token_strings
    annotated["target"] = str(target)

    for node in annotated.get("nodes", []):
        position = node.get("pos")
        if isinstance(position, int) and 0 <= position < len(token_strings):
            node["tok"] = token_strings[position]
        if node.get("kind") != "neuron" or catalog is None:
            continue

        annotation = catalog.lookup(int(node["layer"]), int(node["n"]))
        if annotation is None:
            continue
        if annotation.explanation:
            node["name"] = annotation.explanation
        pole = annotation.steer_up if node.get("act", 0) > 0 else annotation.steer_down
        if pole:
            node["lens"] = list(pole[:lens_tokens])
        node["annotation"] = annotation.as_dict()
    return annotated
