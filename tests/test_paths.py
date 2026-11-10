from neuron_explorer.paths import ProjectPaths


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
