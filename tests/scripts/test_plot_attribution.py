from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from neuron_explorer.attribution import cli


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "plot_attribution.py"


def _graph() -> dict:
    return {
        "token": 7,
        "prob": 0.75,
        "tokens": ["Fact", ":"],
        "target": " answer",
        "nodes": [
            {
                "kind": "embed",
                "layer": -1,
                "pos": 0,
                "n": 0,
                "act": 1.0,
                "influence": 0.2,
            },
            {
                "kind": "neuron",
                "layer": 0,
                "pos": 1,
                "n": 3,
                "act": 2.0,
                "gate": 1.0,
                "influence": 0.8,
            },
            {
                "kind": "logit",
                "layer": 1,
                "pos": 1,
                "n": 7,
                "act": 1.0,
                "influence": 1.0,
            },
        ],
        "edges": [[0, 1, 0.5], [1, 2, 1.0]],
        "stats": {
            "kept_neurons": 1,
            "candidates": 1,
            "logit_from_candidates": 1.0,
            "kept_edges": 2,
        },
    }


def _run(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(_graph()), encoding="utf-8")
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        "import urllib.request\n"
        "def blocked(*args, **kwargs):\n"
        "    raise AssertionError('network access attempted')\n"
        "urllib.request.urlopen = blocked\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    return subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT),
            "python",
            str(SCRIPT),
            "--json",
            str(graph_path),
            "--out",
            str(tmp_path / "graph.svg"),
            *extra,
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_json_to_svg_is_model_and_network_free(tmp_path: Path) -> None:
    result = _run(tmp_path, "--no-annotations")

    assert result.returncode == 0, result.stderr
    svg = (tmp_path / "graph.svg").read_text(encoding="utf-8")
    assert svg.startswith("<svg")
    assert "answer" in svg


@pytest.mark.parametrize("catalog_option", ["--dataset-catalog", "--catalog"])
def test_dataset_catalog_enriches_and_canonicalizes_json(
    tmp_path: Path, catalog_option: str
) -> None:
    dataset = tmp_path / "annotations.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "layer": 0,
                    "neuron": 3,
                    "explanation": "arithmetic result",
                    "steer_up": [" five"],
                }
            ]
        ),
        encoding="utf-8",
    )
    value = str(dataset) if catalog_option == "--dataset-catalog" else f"dataset:{dataset}"

    result = _run(tmp_path, catalog_option, value)

    assert result.returncode == 0, result.stderr
    saved = json.loads((tmp_path / "graph.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert saved["nodes"][1]["name"] == "arithmetic result"
    assert saved["nodes"][1]["lens"] == [" five"]


def test_cache_and_dataset_catalogs_compose_in_command_order(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "names.json").write_text(
        json.dumps({"L0/N3": "cache name"}), encoding="utf-8"
    )
    (cache / "vocab.json").write_text("[]", encoding="utf-8")
    dataset = tmp_path / "annotations.json"
    dataset.write_text(
        json.dumps([{"layer": 0, "neuron": 3, "explanation": "dataset name"}]),
        encoding="utf-8",
    )

    result = _run(
        tmp_path,
        "--cache-catalog",
        str(cache),
        "--dataset-catalog",
        str(dataset),
    )

    assert result.returncode == 0, result.stderr
    saved = json.loads((tmp_path / "graph.json").read_text(encoding="utf-8"))
    assert saved["nodes"][1]["name"] == "cache name"


def test_remote_graph_is_saved_before_rendering(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "remote.svg"
    monkeypatch.setattr(cli, "fetch_graph", lambda **kwargs: _graph())

    def fail_render(*args, **kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(cli, "render_svg", fail_render)

    with pytest.raises(RuntimeError, match="render failed"):
        cli.run_plot_attribution(
            text="remote prompt",
            output=output,
            no_annotations=True,
        )

    saved = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
