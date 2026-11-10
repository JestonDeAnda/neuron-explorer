import json
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from neuron_explorer.attribution.annotations import NeuronAnnotation, annotate_graph
from neuron_explorer.attribution.schema import validate_graph


class StaticCatalog:
    def __init__(self, records):
        self.records = records

    def lookup(self, layer, neuron):
        return self.records.get((layer, neuron))


def raw_graph():
    return {
        "token": 42,
        "prob": 0.75,
        "nodes": [
            {
                "kind": "neuron",
                "layer": 5,
                "pos": 0,
                "n": 8109,
                "act": 1.25,
                "influence": 0.6,
            },
            {
                "kind": "neuron",
                "layer": 5,
                "pos": 1,
                "n": 8109,
                "act": -0.5,
                "influence": 0.4,
            },
            {
                "kind": "logit",
                "layer": 32,
                "pos": 1,
                "n": 42,
                "act": 3.0,
                "influence": 1.0,
            },
        ],
        "edges": [[0, 2, 0.5], [1, 2, -0.25]],
        "stats": {"kept_edges": 2},
    }


def test_annotate_graph_selects_lens_for_activation_sign():
    annotation = NeuronAnnotation(
        explanation="state feature",
        steer_up=[" Texas", " Austin"],
        steer_down=[" California", " Sacramento"],
    )

    annotated = annotate_graph(
        raw_graph(),
        tokens=[" Dallas", " is"],
        target=" Austin",
        catalog=StaticCatalog({(5, 8109): annotation}),
        lens_tokens=1,
    )

    assert annotated["nodes"][0]["lens"] == [" Texas"]
    assert annotated["nodes"][1]["lens"] == [" California"]
    assert annotated["nodes"][0]["name"] == "state feature"


def test_annotate_graph_leaves_missing_records_as_valid_raw_nodes():
    annotated = annotate_graph(
        raw_graph(),
        tokens=[" Dallas", " is"],
        target=" Austin",
        catalog=StaticCatalog({}),
    )

    node = annotated["nodes"][0]
    assert node["tok"] == " Dallas"
    assert "name" not in node
    assert "lens" not in node
    assert "annotation" not in node
    validate_graph(annotated)


def test_annotate_graph_does_not_mutate_input_graph():
    graph = raw_graph()
    original = deepcopy(graph)

    annotate_graph(
        graph,
        tokens=[" Dallas", " is"],
        target=" Austin",
        catalog=StaticCatalog({}),
    )

    assert graph == original


def test_annotate_graph_retains_complete_source_metadata():
    annotation = NeuronAnnotation(
        explanation="state feature",
        steer_up=[" Texas", " Austin"],
        steer_down=[" California"],
        kl_up=1.2,
        kl_down=0.7,
        provenance={"dataset": "manual-v2", "reviewed": True},
    )

    annotated = annotate_graph(
        raw_graph(),
        tokens=[" Dallas", " is"],
        target=" Austin",
        catalog=StaticCatalog({(5, 8109): annotation}),
        lens_tokens=1,
    )

    assert annotated["tokens"] == [" Dallas", " is"]
    assert annotated["target"] == " Austin"
    assert annotated["nodes"][0]["annotation"] == {
        "explanation": "state feature",
        "steer_up": [" Texas", " Austin"],
        "steer_down": [" California"],
        "kl_up": 1.2,
        "kl_down": 0.7,
        "provenance": {"dataset": "manual-v2", "reviewed": True},
    }
    json.dumps(annotated)


def test_neuron_annotation_detaches_and_freezes_constructor_values():
    steer_up = [" Texas"]
    provenance = {"review": {"labels": ["manual"]}}

    annotation = NeuronAnnotation(steer_up=steer_up, provenance=provenance)
    steer_up.append(" changed")
    provenance["review"]["labels"].append("changed")

    assert annotation.steer_up == (" Texas",)
    assert annotation.provenance["review"]["labels"] == ("manual",)
    with pytest.raises(FrozenInstanceError):
        annotation.steer_up = (" changed",)
