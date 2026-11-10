"""Annotation catalogs backed by datasets and existing cache files."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from neuron_explorer.attribution.annotations import NeuronAnnotation, NeuronCatalog

DuplicatePolicy = Literal["error", "last"]
_ANNOTATION_FIELDS = (
    "explanation",
    "steer_up",
    "steer_down",
    "kl_up",
    "kl_down",
    "provenance",
)


def _tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Sequence):
        raise ValueError("token sequences must be strings or sequences")
    return [str(token) for token in value]


def _annotation(record: Mapping[str, Any]) -> NeuronAnnotation:
    explanation = record.get("explanation", record.get("name"))
    provenance = record.get("provenance")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")
    known = {
        "layer",
        "neuron",
        "n",
        "explanation",
        "name",
        "steer_up",
        "steer_down",
        "kl_up",
        "kl_down",
        "provenance",
    }
    extras = {key: value for key, value in record.items() if key not in known}
    source_metadata = dict(provenance or {})
    source_metadata.update(extras)
    return NeuronAnnotation(
        explanation=str(explanation) if explanation is not None else None,
        steer_up=_tokens(record.get("steer_up")),
        steer_down=_tokens(record.get("steer_down")),
        kl_up=float(record["kl_up"]) if record.get("kl_up") is not None else None,
        kl_down=(
            float(record["kl_down"]) if record.get("kl_down") is not None else None
        ),
        provenance=source_metadata or None,
    )


class DatasetCatalog:
    """In-memory catalog loaded from JSON or JSONL records."""

    def __init__(self, records: Mapping[tuple[int, int], NeuronAnnotation]):
        self._records = dict(records)

    @classmethod
    def from_path(
        cls, path: Path, duplicate: DuplicatePolicy = "error"
    ) -> DatasetCatalog:
        if duplicate not in {"error", "last"}:
            raise ValueError(f"unsupported duplicate policy: {duplicate!r}")
        source = Path(path)
        if source.suffix.lower() == ".jsonl":
            records = [
                json.loads(line)
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            records = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("annotation dataset must be a JSON array or JSONL")

        indexed: dict[tuple[int, int], NeuronAnnotation] = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("annotation records must be mappings")
            if "layer" not in record or (
                "neuron" not in record and "n" not in record
            ):
                raise ValueError("annotation records require layer and neuron")
            key = (int(record["layer"]), int(record.get("neuron", record.get("n"))))
            if key in indexed and duplicate == "error":
                raise ValueError(
                    f"duplicate annotation for layer {key[0]}, neuron {key[1]}"
                )
            indexed[key] = _annotation(record)
        return cls(indexed)

    def lookup(self, layer: int, neuron: int) -> NeuronAnnotation | None:
        return self._records.get((layer, neuron))


class CacheCatalog:
    """Catalog backed by names, vocabulary, and steer/logit-lens caches."""

    def __init__(
        self,
        names_path: Path,
        steer_path: Path,
        vocab_path: Path,
        logit_lens_path: Path | None = None,
    ):
        names_file = Path(names_path)
        self._names = (
            json.loads(names_file.read_text(encoding="utf-8"))
            if names_file.is_file()
            else {}
        )
        vocab_file = Path(vocab_path)
        self._vocab = (
            json.loads(vocab_file.read_text(encoding="utf-8"))
            if vocab_file.is_file()
            else []
        )
        self._steer = self._load_npz(Path(steer_path))
        self._lens = (
            self._load_npz(Path(logit_lens_path))
            if logit_lens_path is not None
            else {}
        )

    @staticmethod
    def _load_npz(path: Path) -> dict[str, np.ndarray]:
        if not path.is_file():
            return {}
        with np.load(path) as archive:
            return {name: archive[name] for name in archive.files}

    def _indexed_tokens(
        self, index: Mapping[str, np.ndarray], pole: str, layer: int, neuron: int
    ) -> list[str]:
        values = index.get(pole)
        if values is None:
            return []
        try:
            token_ids: Iterable[Any] = values[layer, neuron]
            return [str(self._vocab[int(token)]) for token in token_ids]
        except (IndexError, TypeError):
            return []

    @staticmethod
    def _score(
        index: Mapping[str, np.ndarray], pole: str, layer: int, neuron: int
    ) -> float | None:
        values = index.get(f"{pole}_kl")
        if values is None:
            return None
        try:
            return float(values[layer, neuron])
        except (IndexError, TypeError):
            return None

    def lookup(self, layer: int, neuron: int) -> NeuronAnnotation | None:
        name = self._names.get(f"L{layer}/N{neuron}")
        up = self._indexed_tokens(self._steer, "up", layer, neuron)
        down = self._indexed_tokens(self._steer, "down", layer, neuron)
        if not up:
            up = self._indexed_tokens(self._lens, "up", layer, neuron)
        if not down:
            down = self._indexed_tokens(self._lens, "down", layer, neuron)
        kl_up = self._score(self._steer, "up", layer, neuron)
        kl_down = self._score(self._steer, "down", layer, neuron)
        if not any((name, up, down, kl_up is not None, kl_down is not None)):
            return None
        return NeuronAnnotation(
            explanation=str(name) if name else None,
            steer_up=up,
            steer_down=down,
            kl_up=kl_up,
            kl_down=kl_down,
        )


class CompositeCatalog:
    """Combine fields from ordered catalogs, preferring earlier values."""

    def __init__(self, *catalogs: NeuronCatalog):
        self._catalogs = catalogs

    def lookup(self, layer: int, neuron: int) -> NeuronAnnotation | None:
        annotations = [
            annotation
            for catalog in self._catalogs
            if (annotation := catalog.lookup(layer, neuron)) is not None
        ]
        if not annotations:
            return None

        values: dict[str, Any] = {}
        for field_name in _ANNOTATION_FIELDS:
            fallback = () if field_name in {"steer_up", "steer_down"} else None
            values[field_name] = next(
                (
                    value
                    for annotation in annotations
                    if (value := getattr(annotation, field_name))
                    not in (None, "", (), {})
                ),
                fallback,
            )
        return NeuronAnnotation(**values)


class EmptyCatalog:
    """Catalog that never returns interpretability metadata."""

    def lookup(self, layer: int, neuron: int) -> None:
        return None
