import json

from fastapi.testclient import TestClient

from neuron_explorer.explorer.server import create_app
from neuron_explorer.paths import ProjectPaths
from neuron_explorer.sweep.rank import compute_units


def _paths(root):
    return ProjectPaths.discover(root)


def _write_run(paths, name, layer):
    paths.runs.mkdir(parents=True)
    run = paths.runs / f"{name}.jsonl"
    run.write_text("{}\n", encoding="utf-8")
    run.with_suffix(".meta.json").write_text(
        json.dumps({"layer": layer, "design_name": "main_rel"}),
        encoding="utf-8",
    )


def test_rank_is_importable_from_installed_package_namespace():
    assert callable(compute_units)


def test_create_app_keeps_project_state_isolated(tmp_path):
    first_paths = _paths(tmp_path / "first")
    second_paths = _paths(tmp_path / "second")
    _write_run(first_paths, "first-run", 3)
    _write_run(second_paths, "second-run", 7)

    first = create_app(first_paths)
    second = create_app(second_paths)

    assert first is not second
    assert first.state.runtime is not second.state.runtime
    assert first.state.runtime.paths == first_paths
    assert second.state.runtime.paths == second_paths
    assert first.state.runtime._db is not second.state.runtime._db
    assert first.state.paths.stats != second.state.paths.stats
    assert first.state.paths.cache != second.state.paths.cache

    first.state.explorer.layer_cache["sentinel"] = "first"
    assert "sentinel" not in second.state.explorer.layer_cache
    first.state.runtime._db.execute(
        "INSERT INTO con (key, val) VALUES (?, ?)", ("sentinel", "first")
    )
    first.state.runtime._db.commit()
    assert second.state.runtime._db.execute(
        "SELECT val FROM con WHERE key = ?", ("sentinel",)
    ).fetchone() is None

    with TestClient(first, base_url="http://localhost:8791") as client:
        first_index = client.get("/api/index").json()
    with TestClient(second, base_url="http://localhost:8791") as client:
        second_index = client.get("/api/index").json()

    assert first_index["layers"] == {"3": "first-run.jsonl"}
    assert second_index["layers"] == {"7": "second-run.jsonl"}


def test_layout_rejects_path_traversal(tmp_path):
    app = create_app(_paths(tmp_path))

    with TestClient(app, base_url="http://localhost:8791") as client:
        response = client.get("/api/layout", params={"kind": "../names"})

    assert response.status_code == 422
