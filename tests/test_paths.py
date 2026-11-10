import json
import os
import subprocess
from pathlib import Path

from neuron_explorer.paths import ProjectPaths

ROOT = Path(__file__).parents[1]


def test_explicit_root_does_not_depend_on_cwd(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    elsewhere = tmp_path / "elsewhere"
    root.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    paths = ProjectPaths.discover(root)
    assert paths.runs == root / "runs"
    assert paths.cache == root / "cache"


def test_environment_overrides_one_data_directory(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    custom = tmp_path / "custom-runs"
    root.mkdir()
    monkeypatch.setenv("NEURON_EXPLORER_RUNS", str(custom))
    assert ProjectPaths.discover(root).runs == custom


def test_all_environment_data_roots_apply_outside_repository(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    expected = {}
    for name in ("RUNS", "STATS", "CAPTURES", "CACHE", "OUTPUT"):
        path = tmp_path / name.lower()
        monkeypatch.setenv(f"NEURON_EXPLORER_{name}", str(path))
        expected[name.lower()] = path

    paths = ProjectPaths.discover()

    for name, path in expected.items():
        assert getattr(paths, name) == path


def test_depth_report_uses_environment_runs_outside_repository(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    runs = tmp_path / "custom-runs"
    elsewhere.mkdir()
    runs.mkdir()
    env = os.environ.copy()
    env["NEURON_EXPLORER_RUNS"] = str(runs)

    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT),
            "python",
            str(ROOT / "scripts" / "depth_report.py"),
            "--layers",
            "0",
        ],
        cwd=elsewhere,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads((runs / "depth_report.json").read_text()) == []
