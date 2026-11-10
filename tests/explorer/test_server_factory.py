import json
import sqlite3
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from neuron_explorer.explorer.ranking import compute_units
from neuron_explorer.explorer.server import (
    _active_state,
    agent_prompt_file,
    create_app,
)
from neuron_explorer.paths import ProjectPaths


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


def test_ranking_is_importable_from_explorer_package():
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


def test_agent_prompt_uses_active_apps_cache_override(tmp_path):
    paths = replace(
        _paths(tmp_path / "project"),
        cache=tmp_path / "machine-cache",
    )
    app = create_app(paths)
    (paths.cache / "agent").mkdir(parents=True)
    token = _active_state.set(app.state.explorer)
    try:
        prompt_path = agent_prompt_file()
    finally:
        _active_state.reset(token)

    prompt = prompt_path.read_text(encoding="utf-8")
    assert f"{paths.cache}/agent/work/" in prompt
    assert "{CACHE}" not in prompt
    app.state.runtime.close()


def test_app_shutdown_closes_only_its_runtime(tmp_path):
    first = create_app(_paths(tmp_path / "first"))
    second = create_app(_paths(tmp_path / "second"))

    with TestClient(first, base_url="http://localhost:8791") as client:
        assert client.get("/api/status").status_code == 200

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        first.state.runtime._db.execute("SELECT 1")
    assert second.state.runtime._db.execute("SELECT 1").fetchone() == (1,)

    first.state.runtime.close()
    second.state.runtime.close()
    second.state.runtime.close()
