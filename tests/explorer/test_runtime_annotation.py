from neuron_explorer.attribution import NeuronAnnotation
from neuron_explorer.explorer.runtime import annotate_attribution_graph


class StaticCatalog:
    def lookup(self, layer, neuron):
        if (layer, neuron) == (3, 7):
            return NeuronAnnotation(
                explanation="named feature",
                steer_up=[" positive"],
                steer_down=[" negative"],
            )
        return None


def test_annotation_helper_decodes_and_catalogs_without_runtime():
    graph = {
        "token": 19,
        "nodes": [
            {
                "kind": "neuron",
                "layer": 3,
                "pos": 1,
                "n": 7,
                "act": 2.0,
            }
        ],
        "edges": [],
    }
    decoded = {11: " first", 13: " second", 19: " target"}

    annotated = annotate_attribution_graph(
        graph,
        input_ids=[11, 13],
        decode=lambda token_ids: decoded[token_ids[0]],
        catalog=StaticCatalog(),
    )

    assert annotated["tokens"] == [" first", " second"]
    assert annotated["target"] == " target"
    assert annotated["nodes"][0]["tok"] == " second"
    assert annotated["nodes"][0]["name"] == "named feature"
    assert annotated["nodes"][0]["lens"] == [" positive"]
