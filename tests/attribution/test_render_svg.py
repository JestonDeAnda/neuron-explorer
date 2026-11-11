from copy import deepcopy

from neuron_explorer.attribution import render_svg


def graph_fixture():
    return {
        "token": 42,
        "prob": 0.75,
        "nodes": [
            {
                "kind": "embed",
                "layer": -1,
                "pos": 0,
                "n": 0,
                "act": None,
                "influence": 0.2,
            },
            {
                "kind": "neuron",
                "layer": 1,
                "pos": 0,
                "n": 7,
                "act": 1.25,
                "gate": 0.5,
                "influence": 0.6,
            },
            {
                "kind": "logit",
                "layer": 2,
                "pos": 0,
                "n": 42,
                "act": 3.0,
                "influence": 1.0,
            },
        ],
        "edges": [[0, 1, 0.5], [1, 2, -0.25]],
        "stats": {
            "candidates": 1,
            "logit_from_candidates": 0.8,
            "kept_neurons": 1,
            "kept_edges": 2,
        },
    }


def test_raw_graph_renders_as_standalone_svg():
    rendered = render_svg(graph_fixture())

    assert rendered.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert rendered.endswith("</svg>")
    assert "L1 N7@0" in rendered


def test_annotated_graph_prefers_explanation_over_lens_for_label():
    graph = deepcopy(graph_fixture())
    graph["tokens"] = [" Dallas"]
    graph["target"] = " Austin"
    graph["nodes"][1].update(
        {"tok": " Dallas", "name": "state capital", "lens": [" Texas"]}
    )

    rendered = render_svg(graph, labels=1)

    assert "L1.7 state capital" in rendered
    assert "L1.7 Texas" not in rendered


def test_signed_edges_use_distinct_colors():
    rendered = render_svg(graph_fixture())

    assert 'stroke="#2b6cb0"' in rendered
    assert 'stroke="#c53030"' in rendered


def test_neuron_tooltip_contains_layer_neuron_and_position_identifier():
    rendered = render_svg(graph_fixture())

    assert "<title>L1 N7@0" in rendered


def test_completion_logprob_target_is_not_mislabeled_as_one_logit():
    graph = deepcopy(graph_fixture())
    graph["target"] = {
        "kind": "completion_logprob",
        "token_ids": [41, 42],
        "prediction_positions": [0, 1],
    }
    graph["completion_logprob"] = -2.5
    graph["nodes"][-1].update(
        {
            "target_kind": "completion_logprob",
            "target_token_ids": [41, 42],
            "prediction_positions": [0, 1],
        }
    )

    rendered = render_svg(graph)

    assert "completion log-prob (2 tokens)" in rendered
    assert "log p=" in rendered
    assert "target scalar" in rendered
    assert "target logit" not in rendered
