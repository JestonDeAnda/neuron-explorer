import json

import numpy as np
import pytest

from neuron_explorer.attribution.annotations import NeuronAnnotation
from neuron_explorer.attribution.catalogs import (
    CacheCatalog,
    CompositeCatalog,
    DatasetCatalog,
    EmptyCatalog,
)


RECORD = {
    "layer": 5,
    "neuron": 8109,
    "explanation": "Texas state",
    "steer_up": [" Texas"],
    "steer_down": [" California"],
}


def test_dataset_catalog_loads_json_array(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps([RECORD]), encoding="utf-8")

    annotation = DatasetCatalog.from_path(path).lookup(5, 8109)

    assert annotation == NeuronAnnotation(
        explanation="Texas state",
        steer_up=[" Texas"],
        steer_down=[" California"],
    )


def test_dataset_catalog_loads_jsonl_and_normalizes_aliases(tmp_path):
    path = tmp_path / "annotations.jsonl"
    path.write_text(
        json.dumps(
            {
                "layer": 5,
                "n": 8109,
                "name": "Texas state",
                "steer_up": (" Texas",),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    annotation = DatasetCatalog.from_path(path).lookup(5, 8109)

    assert annotation.explanation == "Texas state"
    assert annotation.steer_up == [" Texas"]


def test_dataset_catalog_rejects_duplicate_keys_by_default(tmp_path):
    path = tmp_path / "duplicates.json"
    path.write_text(json.dumps([RECORD, {**RECORD, "name": "duplicate"}]))

    with pytest.raises(ValueError, match="duplicate.*5.*8109"):
        DatasetCatalog.from_path(path)


def test_dataset_catalog_can_keep_last_duplicate(tmp_path):
    path = tmp_path / "duplicates.json"
    path.write_text(
        json.dumps([RECORD, {**RECORD, "explanation": "last value"}]),
        encoding="utf-8",
    )

    catalog = DatasetCatalog.from_path(path, duplicate="last")

    assert catalog.lookup(5, 8109).explanation == "last value"


def test_cache_catalog_combines_names_and_steer_index(tmp_path):
    names_path = tmp_path / "names.json"
    steer_path = tmp_path / "steer.npz"
    vocab_path = tmp_path / "vocab.json"
    names_path.write_text(json.dumps({"L0/N1": "Texas state"}), encoding="utf-8")
    vocab_path.write_text(
        json.dumps([" unused", " Texas", " California"]), encoding="utf-8"
    )
    np.savez(
        steer_path,
        up=np.array([[[0, 0], [1, 0]]]),
        down=np.array([[[0, 0], [2, 0]]]),
        up_kl=np.array([[0.0, 1.25]]),
        down_kl=np.array([[0.0, 0.75]]),
    )

    annotation = CacheCatalog(names_path, steer_path, vocab_path).lookup(0, 1)

    assert annotation.explanation == "Texas state"
    assert annotation.steer_up == [" Texas", " unused"]
    assert annotation.steer_down == [" California", " unused"]
    assert annotation.kl_up == pytest.approx(1.25)
    assert annotation.kl_down == pytest.approx(0.75)


def test_cache_catalog_uses_logit_lens_when_steer_index_is_missing(tmp_path):
    names_path = tmp_path / "missing-names.json"
    steer_path = tmp_path / "missing-steer.npz"
    vocab_path = tmp_path / "vocab.json"
    lens_path = tmp_path / "lens.npz"
    vocab_path.write_text(json.dumps([" zero", " up", " down"]), encoding="utf-8")
    np.savez(
        lens_path,
        up=np.array([[[1]]]),
        down=np.array([[[2]]]),
    )

    annotation = CacheCatalog(
        names_path, steer_path, vocab_path, logit_lens_path=lens_path
    ).lookup(0, 0)

    assert annotation.steer_up == [" up"]
    assert annotation.steer_down == [" down"]


def test_empty_catalog_lookup_returns_none():
    assert EmptyCatalog().lookup(5, 8109) is None


def test_composite_catalog_fills_fields_in_priority_order(tmp_path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(
        json.dumps(
            [
                {
                    "layer": 5,
                    "neuron": 8109,
                    "explanation": "preferred",
                    "steer_up": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    second_path.write_text(
        json.dumps(
            [
                {
                    "layer": 5,
                    "neuron": 8109,
                    "explanation": "fallback",
                    "steer_up": [" up"],
                    "steer_down": [" down"],
                    "kl_up": 0.4,
                }
            ]
        ),
        encoding="utf-8",
    )

    annotation = CompositeCatalog(
        DatasetCatalog.from_path(first_path),
        DatasetCatalog.from_path(second_path),
    ).lookup(5, 8109)

    assert annotation.explanation == "preferred"
    assert annotation.steer_up == [" up"]
    assert annotation.steer_down == [" down"]
    assert annotation.kl_up == pytest.approx(0.4)
