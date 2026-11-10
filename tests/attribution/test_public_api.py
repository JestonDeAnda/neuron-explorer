import ast
from pathlib import Path

import neuron_explorer.attribution as attribution


def test_package_exports_graph_computation_and_rendering_interfaces():
    assert callable(attribution.build_graph)
    assert attribution.Linearized.__name__ == "Linearized"
    assert callable(attribution.select_graph)
    assert callable(attribution.layout_graph)
    assert callable(attribution.render_svg)


def test_new_attribution_modules_use_absolute_imports():
    modules = [
        attribution,
        __import__(
            "neuron_explorer.attribution.graph", fromlist=["graph"]
        ),
        __import__(
            "neuron_explorer.attribution.render_svg", fromlist=["render_svg"]
        ),
    ]

    for module in modules:
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        relative = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level
        ]
        assert relative == [], module.__name__
