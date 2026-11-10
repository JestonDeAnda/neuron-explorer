import ast
import json
from pathlib import Path

import pytest

import neuron_explorer.attribution as attribution
import neuron_explorer.attribution.catalogs as catalogs
from neuron_explorer.attribution.schema import load_graph, save_graph, validate_graph


@pytest.fixture
def current_graph():
    return {
        "token": 42,
        "prob": 0.75,
        "nodes": [
            {
                "kind": "neuron",
                "layer": 5,
                "pos": 1,
                "n": 8109,
                "act": 1.25,
                "influence": 0.6,
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
        "edges": [[0, 1, 0.5]],
        "stats": {"kept_edges": 1},
    }


def test_save_graph_adds_schema_version(tmp_path, current_graph):
    path = save_graph(current_graph, tmp_path / "graph.json")

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert "schema_version" not in current_graph


def test_load_graph_accepts_unversioned_current_graph(tmp_path, current_graph):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(current_graph), encoding="utf-8")

    loaded = load_graph(path)

    assert loaded["schema_version"] == 1
    assert loaded["nodes"] == current_graph["nodes"]


def test_validate_graph_rejects_out_of_bounds_edge(current_graph):
    current_graph["edges"] = [[0, len(current_graph["nodes"]), 0.5]]

    with pytest.raises(ValueError, match="edge.*index.*bounds"):
        validate_graph(current_graph)


def test_validate_graph_rejects_missing_required_graph_field(current_graph):
    del current_graph["stats"]

    with pytest.raises(ValueError, match="missing.*stats"):
        validate_graph(current_graph)


def test_package_exports_structured_attribution_interfaces():
    assert attribution.load_graph is load_graph
    assert attribution.save_graph is save_graph
    assert attribution.validate_graph is validate_graph
    assert callable(attribution.annotate_graph)
    assert attribution.NeuronAnnotation.__name__ == "NeuronAnnotation"


@pytest.mark.parametrize("module", [attribution, catalogs])
def test_package_modules_use_absolute_imports(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))

    relative = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level
    ]

    assert relative == []
