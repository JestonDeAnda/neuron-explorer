"""xLLM Huginn schedule, runtime identity, and mount contracts."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from neuron_explorer.attribution.moe.xllm_huginn.mount import check_moe_mounts
from neuron_explorer.attribution.moe.xllm_huginn.runtime import (
    XLLM_SOURCE_COMMIT,
    checkpoint_fingerprint,
    load_xllm_huginn,
    verify_xllm_source,
)
from neuron_explorer.attribution.moe.xllm_huginn.schedule import execution_schedule


def _config(**overrides: object) -> dict[str, object]:
    model: dict[str, object] = {
        "arch": "huginn",
        "num_layers": 28,
        "huginn_prelude_layers": 8,
        "huginn_recurrent_layers": 12,
        "huginn_coda_layers": 8,
        "loop_times": 8,
        "huginn_hierarchical_state": "none",
        "huginn_hierarchical_h_cycles": 2,
        "huginn_hierarchical_l_cycles": 3,
        "num_experts": 25,
        "num_activated_experts": 2,
        "expert_inter_dim": 2432,
        "vocab_size": 250624,
    }
    model.update(overrides)
    return {"model": model}


def test_execution_schedule_unrolls_physical_huginn_layers() -> None:
    slots = execution_schedule(_config())

    assert len(slots) == 112
    assert [slot.virtual_depth for slot in slots] == list(range(112))
    assert [slot.physical_layer for slot in slots[:8]] == list(range(8))
    assert [slot.physical_layer for slot in slots[8:20]] == list(range(8, 20))
    assert slots[20].recurrence_iteration == 1
    assert slots[104].stage == "coda"
    assert [slot.physical_layer for slot in slots[104:]] == list(range(20, 28))


def test_schedule_ignores_h_l_counters_for_nonhierarchical_huginn() -> None:
    baseline = execution_schedule(_config())
    changed = execution_schedule(
        _config(
            huginn_hierarchical_h_cycles=97,
            huginn_hierarchical_l_cycles=101,
        )
    )

    assert changed == baseline


def test_schedule_rejects_hierarchical_huginn() -> None:
    with pytest.raises(ValueError, match="hierarchical"):
        execution_schedule(_config(huginn_hierarchical_state="h_cycles"))


@pytest.mark.parametrize(
    "model",
    [
        {"huginn_hierarchical_state": "none"},
        {
            **_config()["model"],
            "huginn_prelude_layers": "eight",
        },
    ],
)
def test_schedule_rejects_missing_or_noninteger_values(model: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="invalid xLLM Huginn schedule"):
        execution_schedule({"model": model})


def _checkpoint(root: Path) -> None:
    (root / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 7},
                "weight_map": {"layer.weight": "model-00001-of-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    (root / "native_model.py").write_text("MODEL = 'huginn'\n", encoding="utf-8")
    (root / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (root / "tokenizer").mkdir()
    (root / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    names = [
        "config.json",
        "model.safetensors.index.json",
        "native_model.py",
        "model-00001-of-00001.safetensors",
        "tokenizer/tokenizer.json",
    ]
    manifest = {
        "schema_version": 1,
        "files": [
            {
                "path": name,
                "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
                "size_bytes": (root / name).stat().st_size,
            }
            for name in names
        ],
    }
    (root / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    ("relative", "replacement"),
    [
        ("config.json", b'{"changed": true}'),
        ("artifact_manifest.json", b'{"schema_version": 2}'),
        ("native_model.py", b"MODEL = 'changed'\n"),
        ("model-00001-of-00001.safetensors", b"changed weights"),
    ],
)
def test_checkpoint_fingerprint_changes_for_runtime_bytes(
    tmp_path: Path, relative: str, replacement: bytes
) -> None:
    _checkpoint(tmp_path)
    before = checkpoint_fingerprint(tmp_path)

    (tmp_path / relative).write_bytes(replacement)

    assert checkpoint_fingerprint(tmp_path) != before


def test_checkpoint_fingerprint_requires_weights(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="safetensors"):
        checkpoint_fingerprint(tmp_path)


@pytest.mark.parametrize(
    "relative",
    [
        "model.safetensors.index.json",
        "tokenizer/tokenizer.json",
    ],
)
def test_checkpoint_fingerprint_covers_index_and_tokenizer(
    tmp_path: Path, relative: str
) -> None:
    _checkpoint(tmp_path)
    before = checkpoint_fingerprint(tmp_path)

    with (tmp_path / relative).open("ab") as handle:
        handle.write(b"changed")

    assert checkpoint_fingerprint(tmp_path) != before


def test_checkpoint_fingerprint_changes_when_source_file_is_renamed(tmp_path: Path) -> None:
    _checkpoint(tmp_path)
    before = checkpoint_fingerprint(tmp_path)

    (tmp_path / "native_model.py").rename(tmp_path / "renamed_model.py")

    assert checkpoint_fingerprint(tmp_path) != before


def test_checkpoint_fingerprint_rejects_symlinks(tmp_path: Path) -> None:
    _checkpoint(tmp_path)
    (tmp_path / "linked.py").symlink_to(tmp_path / "native_model.py")

    with pytest.raises(ValueError, match="symlink"):
        checkpoint_fingerprint(tmp_path)


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def test_source_commit_accepts_head_and_rejects_wrong_commit(tmp_path: Path) -> None:
    _git("init", "-q", cwd=tmp_path)
    _git("config", "user.name", "Test", cwd=tmp_path)
    _git("config", "user.email", "test@example.com", cwd=tmp_path)
    (tmp_path / "xllm.py").write_text("PINNED = True\n", encoding="utf-8")
    _git("add", "xllm.py", cwd=tmp_path)
    _git("commit", "-qm", "fixture", cwd=tmp_path)
    head = _git("rev-parse", "HEAD", cwd=tmp_path)

    assert verify_xllm_source(tmp_path, head) == head
    with pytest.raises(RuntimeError, match="xLLM source commit"):
        verify_xllm_source(tmp_path, "0" * 40)


def test_source_commit_allows_untracked_extension_build_artifacts(tmp_path: Path) -> None:
    _git("init", "-q", cwd=tmp_path)
    _git("config", "user.name", "Test", cwd=tmp_path)
    _git("config", "user.email", "test@example.com", cwd=tmp_path)
    (tmp_path / "xllm.py").write_text("PINNED = True\n", encoding="utf-8")
    _git("add", "xllm.py", cwd=tmp_path)
    _git("commit", "-qm", "fixture", cwd=tmp_path)
    head = _git("rev-parse", "HEAD", cwd=tmp_path)
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "xllm_extension.so").write_bytes(b"compiled")

    assert verify_xllm_source(tmp_path, head) == head


def test_source_commit_rejects_tracked_modifications(tmp_path: Path) -> None:
    _git("init", "-q", cwd=tmp_path)
    _git("config", "user.name", "Test", cwd=tmp_path)
    _git("config", "user.email", "test@example.com", cwd=tmp_path)
    (tmp_path / "xllm.py").write_text("PINNED = True\n", encoding="utf-8")
    _git("add", "xllm.py", cwd=tmp_path)
    _git("commit", "-qm", "fixture", cwd=tmp_path)
    head = _git("rev-parse", "HEAD", cwd=tmp_path)
    (tmp_path / "xllm.py").write_text("PINNED = False\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="tracked modifications"):
        verify_xllm_source(tmp_path, head)


def test_source_commit_rejects_subdirectory_of_checkout(tmp_path: Path) -> None:
    _git("init", "-q", cwd=tmp_path)
    _git("config", "user.name", "Test", cwd=tmp_path)
    _git("config", "user.email", "test@example.com", cwd=tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "xllm.py").write_text("PINNED = True\n", encoding="utf-8")
    _git("add", ".", cwd=tmp_path)
    _git("commit", "-qm", "fixture", cwd=tmp_path)
    head = _git("rev-parse", "HEAD", cwd=tmp_path)

    with pytest.raises(RuntimeError, match="top-level"):
        verify_xllm_source(source, head)


def test_pinned_source_commit_is_exact_release() -> None:
    assert XLLM_SOURCE_COMMIT == "755c7b3a91081fe0f83500eabd53bd2490b8b745"


class _FakeTokenizer:
    def __init__(self, length: int = 250624, bos_id: int = 0) -> None:
        self.length = length
        self.bos_id = bos_id
        self.vocab_size = length

    def __len__(self) -> int:
        return self.length


def _runtime_config(**overrides: object) -> SimpleNamespace:
    values = {
        "num_layers": 28,
        "huginn_prelude_layers": 8,
        "huginn_recurrent_layers": 12,
        "huginn_coda_layers": 8,
        "loop_times": 8,
        "huginn_hierarchical_state": "none",
        "num_experts": 25,
        "num_activated_experts": 2,
        "expert_inter_dim": 2432,
        "init_mode": "none",
        "fused_block": False,
        "fused_output_layer": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _runtime_model(class_name: str = "Huginn", **overrides: object):
    model = type(class_name, (), {})()
    values = {
        "num_layers": 28,
        "huginn_prelude_layers": 8,
        "huginn_recurrent_layers": 12,
        "huginn_coda_layers": 8,
        "layers": [object()] * 28,
    }
    values.update(overrides)
    for name, value in values.items():
        setattr(model, name, value)
    model.state_dict = lambda: {"layer.weight": object()}
    return model


def _install_fake_xllm(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    *,
    model=None,
    tokenizer=None,
    config=None,
) -> list[tuple[Path, str]]:
    package = source / "xllm"
    package.mkdir(parents=True, exist_ok=True)
    package_file = package / "__init__.py"
    package_file.write_text("", encoding="utf-8")
    paper = package / "paper_part1"
    paper.mkdir(exist_ok=True)
    native_file = paper / "native_inference.py"
    native_file.write_text("", encoding="utf-8")
    calls: list[tuple[Path, str]] = []

    def fake_load(path: Path, *, device: str):
        calls.append((path, device))
        return (
            model if model is not None else _runtime_model(),
            tokenizer if tokenizer is not None else _FakeTokenizer(),
            config if config is not None else _runtime_config(),
        )

    xllm = types.ModuleType("xllm")
    xllm.__file__ = str(package_file)
    paper_part1 = types.ModuleType("xllm.paper_part1")
    paper_part1.__file__ = str(paper / "__init__.py")
    native = types.ModuleType("xllm.paper_part1.native_inference")
    native.__file__ = str(native_file)
    native.load_native_model = fake_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xllm", xllm)
    monkeypatch.setitem(sys.modules, "xllm.paper_part1", paper_part1)
    monkeypatch.setitem(sys.modules, "xllm.paper_part1.native_inference", native)
    return calls


def test_native_loader_prioritizes_pinned_source_and_returns_verified_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "xllm-source"
    source.mkdir()
    model_path = tmp_path / "model"
    model_path.mkdir()
    _checkpoint(model_path)
    tokenizer = _FakeTokenizer()
    config = _runtime_config()
    calls = _install_fake_xllm(
        monkeypatch,
        source,
        tokenizer=tokenizer,
        config=config,
    )
    monkeypatch.setattr(
        "neuron_explorer.attribution.moe.xllm_huginn.runtime.verify_xllm_source",
        lambda path, expected: XLLM_SOURCE_COMMIT,
    )
    monkeypatch.setattr(
        "neuron_explorer.attribution.moe.xllm_huginn.runtime.checkpoint_fingerprint",
        lambda path: "checkpoint-sha256:test",
    )
    monkeypatch.setattr(sys, "path", [*sys.path, str(source)])

    loaded = load_xllm_huginn(model_path, source, "cuda:0")

    assert sys.path[0] == str(source)
    assert calls == [(model_path, "cuda:0")]
    assert loaded == (
        tokenizer,
        loaded[1],
        config,
        "checkpoint-sha256:test",
        XLLM_SOURCE_COMMIT,
    )
    assert loaded[1].__class__.__name__ == "Huginn"


def test_native_loader_verifies_source_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git("init", "-q", cwd=source)
    _git("config", "user.name", "Test", cwd=source)
    _git("config", "user.email", "test@example.com", cwd=source)
    (source / "tracked").write_text("source\n", encoding="utf-8")
    _git("add", ".", cwd=source)
    _git("commit", "-qm", "wrong source", cwd=source)
    imported = False
    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        nonlocal imported
        if name == "xllm.paper_part1.native_inference":
            imported = True
            raise AssertionError("xllm imported before source verification")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)

    with pytest.raises(RuntimeError, match="xLLM source commit"):
        load_xllm_huginn(tmp_path / "model", source, "cpu")
    assert imported is False


def test_native_loader_rejects_preimported_foreign_xllm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    model_path = tmp_path / "model"
    model_path.mkdir()
    _checkpoint(model_path)
    _install_fake_xllm(monkeypatch, source)
    sys.modules["xllm"].__file__ = str(tmp_path / "foreign" / "xllm" / "__init__.py")
    monkeypatch.setattr(
        "neuron_explorer.attribution.moe.xllm_huginn.runtime.verify_xllm_source",
        lambda path, expected: XLLM_SOURCE_COMMIT,
    )

    with pytest.raises(RuntimeError, match="outside pinned source"):
        load_xllm_huginn(model_path, source, "cpu")


@pytest.mark.parametrize(
    ("model", "tokenizer", "config", "message"),
    [
        (_runtime_model("OtherModel"), _FakeTokenizer(), _runtime_config(), "expected Huginn"),
        (
            _runtime_model(),
            _FakeTokenizer(),
            _runtime_config(num_experts=24),
            "num_experts",
        ),
        (_runtime_model(), _FakeTokenizer(length=250623), _runtime_config(), "length"),
        (_runtime_model(), _FakeTokenizer(bos_id=1), _runtime_config(), "BOS"),
        (
            _runtime_model(layers=[object()] * 27),
            _FakeTokenizer(),
            _runtime_config(),
            "physical blocks",
        ),
        (
            _runtime_model(huginn_coda_layers=7),
            _FakeTokenizer(),
            _runtime_config(),
            "model huginn_coda_layers",
        ),
        (
            _runtime_model(),
            _FakeTokenizer(),
            _runtime_config(fused_block=True),
            "fused_block",
        ),
    ],
)
def test_native_loader_rejects_runtime_identity_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model,
    tokenizer,
    config,
    message: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    model_path = tmp_path / "model"
    model_path.mkdir()
    _checkpoint(model_path)
    _install_fake_xllm(
        monkeypatch,
        source,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
    monkeypatch.setattr(
        "neuron_explorer.attribution.moe.xllm_huginn.runtime.verify_xllm_source",
        lambda path, expected: XLLM_SOURCE_COMMIT,
    )

    with pytest.raises(RuntimeError, match=message):
        load_xllm_huginn(model_path, source, "cpu")


def test_native_loader_rejects_non_strict_state_dict_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    model_path = tmp_path / "model"
    model_path.mkdir()
    _checkpoint(model_path)
    model = _runtime_model()
    model.state_dict = lambda: {"different.weight": object()}
    _install_fake_xllm(monkeypatch, source, model=model)
    monkeypatch.setattr(
        "neuron_explorer.attribution.moe.xllm_huginn.runtime.verify_xllm_source",
        lambda path, expected: XLLM_SOURCE_COMMIT,
    )

    with pytest.raises(RuntimeError, match="strict state-dict"):
        load_xllm_huginn(model_path, source, "cpu")


def test_native_loader_restores_sys_path_after_identity_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    model_path = tmp_path / "model"
    model_path.mkdir()
    _checkpoint(model_path)
    _install_fake_xllm(monkeypatch, source, model=_runtime_model("Wrong"))
    monkeypatch.setattr(
        "neuron_explorer.attribution.moe.xllm_huginn.runtime.verify_xllm_source",
        lambda path, expected: XLLM_SOURCE_COMMIT,
    )
    original_path = [*sys.path]
    monkeypatch.setattr(sys, "path", original_path.copy())

    with pytest.raises(RuntimeError, match="expected Huginn"):
        load_xllm_huginn(model_path, source, "cpu")

    assert sys.path == original_path


def _valid_mount_fixture(tmp_path: Path) -> dict[str, object]:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for task in range(23):
        rows = 64 if task < 22 else 53
        (dataset / f"{task:02}.jsonl").write_text("{}\n" * rows, encoding="utf-8")

    model = tmp_path / "model"
    model.mkdir()
    _checkpoint(model)

    source = tmp_path / "xllm"
    source.mkdir()
    _git("init", "-q", cwd=source)
    _git("config", "user.name", "Test", cwd=source)
    _git("config", "user.email", "test@example.com", cwd=source)
    grouped_mm = source / "xllm" / "modules" / "fused_ops" / "mgmm"
    grouped_mm.mkdir(parents=True)
    (grouped_mm / "multi_group_matmul.py").write_text("mgmm = object()\n", encoding="utf-8")
    _git("add", ".", cwd=source)
    _git("commit", "-qm", "fixture", cwd=source)
    source_commit = _git("rev-parse", "HEAD", cwd=source)
    extension_file = source / "build" / "xllm_extension.so"
    extension_file.parent.mkdir()
    extension_file.write_bytes(b"compiled")

    stage = tmp_path / "stage"
    (stage / "src" / "neuron_explorer" / "attribution" / "moe" / "xllm_huginn").mkdir(
        parents=True
    )
    (stage / "COMMIT").write_text("project-commit\n", encoding="utf-8")
    output_root = tmp_path / "output"
    extension = type(
        "Extension",
        (),
        {
            "__file__": str(extension_file),
            "te": type("TE", (), {"general_grouped_gemm": object()})(),
        },
    )()
    return {
        "dataset": dataset,
        "model": model,
        "xllm_source": source,
        "expected_xllm_commit": source_commit,
        "stage": stage,
        "commit": "project-commit",
        "output_root": output_root,
        "extension": extension,
    }


def _run_mount_fixture(
    fixture: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    extension = fixture["extension"]
    monkeypatch.setattr(
        "neuron_explorer.attribution.moe.xllm_huginn.mount.import_module",
        lambda name: extension if name == "xllm_extension" else None,
    )
    arguments = {key: value for key, value in fixture.items() if key != "extension"}
    return check_moe_mounts(**arguments)


def _report_check(report: dict[str, object], name: str) -> dict[str, object]:
    return next(check for check in report["checks"] if check["name"] == name)


def test_mount_report_has_individual_passed_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _run_mount_fixture(_valid_mount_fixture(tmp_path), monkeypatch)

    assert report["ok"] is True
    assert report["checks"]
    assert all(check["ok"] for check in report["checks"])
    assert {
        "dataset_tasks",
        "dataset_rows",
        "project_commit",
        "xllm_source_commit",
        "checkpoint_files",
        "artifact_manifest",
        "xllm_extension",
        "grouped_mm",
        "output_root",
    } <= {check["name"] for check in report["checks"]}


def test_mount_rejects_wrong_dataset_file_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _valid_mount_fixture(tmp_path)
    next(Path(fixture["dataset"]).glob("*.jsonl")).unlink()

    report = _run_mount_fixture(fixture, monkeypatch)

    assert _report_check(report, "dataset_tasks")["ok"] is False


def test_mount_rejects_wrong_dataset_row_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _valid_mount_fixture(tmp_path)
    dataset_file = next(Path(fixture["dataset"]).glob("*.jsonl"))
    with dataset_file.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    report = _run_mount_fixture(fixture, monkeypatch)

    assert _report_check(report, "dataset_rows")["ok"] is False


def test_mount_rejects_project_commit_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _valid_mount_fixture(tmp_path)
    (Path(fixture["stage"]) / "COMMIT").write_text("wrong\n", encoding="utf-8")

    report = _run_mount_fixture(fixture, monkeypatch)

    assert _report_check(report, "project_commit")["ok"] is False


def test_mount_rejects_wrong_source_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _valid_mount_fixture(tmp_path)
    fixture["expected_xllm_commit"] = "0" * 40

    report = _run_mount_fixture(fixture, monkeypatch)

    assert _report_check(report, "xllm_source_commit")["ok"] is False


def test_mount_rejects_dirty_tracked_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _valid_mount_fixture(tmp_path)
    grouped_mm = (
        Path(fixture["xllm_source"])
        / "xllm"
        / "modules"
        / "fused_ops"
        / "mgmm"
        / "multi_group_matmul.py"
    )
    grouped_mm.write_text("changed = True\n", encoding="utf-8")

    report = _run_mount_fixture(fixture, monkeypatch)

    assert _report_check(report, "xllm_source_commit")["ok"] is False


def test_mount_rejects_missing_grouped_mm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _valid_mount_fixture(tmp_path)
    source = Path(fixture["xllm_source"])
    grouped_mm = (
        source
        / "xllm"
        / "modules"
        / "fused_ops"
        / "mgmm"
        / "multi_group_matmul.py"
    )
    grouped_mm.unlink()
    _git("add", "-u", cwd=source)
    _git("commit", "-qm", "remove grouped mm", cwd=source)
    fixture["expected_xllm_commit"] = _git("rev-parse", "HEAD", cwd=source)

    report = _run_mount_fixture(fixture, monkeypatch)

    assert _report_check(report, "grouped_mm")["ok"] is False


def test_mount_rejects_missing_checkpoint_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _valid_mount_fixture(tmp_path)
    (Path(fixture["model"]) / "model.safetensors.index.json").unlink()

    report = _run_mount_fixture(fixture, monkeypatch)

    assert _report_check(report, "checkpoint_files")["ok"] is False


def test_mount_rejects_ambient_xllm_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _valid_mount_fixture(tmp_path)
    fixture["extension"].__file__ = str(tmp_path / "site-packages" / "xllm_extension.so")

    report = _run_mount_fixture(fixture, monkeypatch)

    assert _report_check(report, "xllm_extension")["ok"] is False


def test_mount_report_fails_when_a_required_check_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "neuron_explorer.attribution.moe.xllm_huginn.mount.import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("not built")),
    )

    report = check_moe_mounts(
        dataset=tmp_path / "missing-dataset",
        model=tmp_path / "missing-model",
        xllm_source=tmp_path / "missing-source",
        expected_xllm_commit=XLLM_SOURCE_COMMIT,
        stage=tmp_path / "missing-stage",
        commit="missing-commit",
        output_root=tmp_path / "output",
    )

    assert report["ok"] is False
    assert any(not check["ok"] for check in report["checks"])


def test_mount_manifest_rejects_paths_outside_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"not a checkpoint artifact")
    manifest = {
        "files": [
            {
                "path": "../outside",
                "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                "size_bytes": outside.stat().st_size,
            }
        ]
    }
    (model / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "neuron_explorer.attribution.moe.xllm_huginn.mount.import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("not built")),
    )

    report = check_moe_mounts(
        dataset=tmp_path / "dataset",
        model=model,
        xllm_source=tmp_path / "source",
        stage=tmp_path / "stage",
        commit="project-commit",
        output_root=tmp_path / "output",
    )

    manifest_check = next(
        check for check in report["checks"] if check["name"] == "artifact_manifest"
    )
    assert manifest_check["ok"] is False
    assert "escapes" in manifest_check["detail"]


@pytest.mark.parametrize("extra", ["unlisted.py", "unlisted.safetensors"])
def test_mount_manifest_rejects_unlisted_fingerprint_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: str,
) -> None:
    fixture = _valid_mount_fixture(tmp_path)
    (Path(fixture["model"]) / extra).write_bytes(b"unlisted")

    report = _run_mount_fixture(fixture, monkeypatch)

    check = _report_check(report, "artifact_manifest")
    assert check["ok"] is False
    assert "unlisted fingerprint inputs" in check["detail"]


def test_mount_manifest_rejects_missing_weight_map_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _valid_mount_fixture(tmp_path)
    model = Path(fixture["model"])
    index_path = model / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 7},
                "weight_map": {"layer.weight": "missing.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    manifest_path = model / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_record = next(
        entry
        for entry in manifest["files"]
        if entry["path"] == "model.safetensors.index.json"
    )
    index_record["sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    index_record["size_bytes"] = index_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = _run_mount_fixture(fixture, monkeypatch)

    check = _report_check(report, "artifact_manifest")
    assert check["ok"] is False
    assert "weight-map shard missing" in check["detail"]


def test_mount_reports_dataset_decode_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _valid_mount_fixture(tmp_path)
    next(Path(fixture["dataset"]).glob("*.jsonl")).write_bytes(b"\xff")

    report = _run_mount_fixture(fixture, monkeypatch)

    assert report["ok"] is False
    assert _report_check(report, "dataset_rows")["ok"] is False


def test_mount_reports_extension_runtime_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _valid_mount_fixture(tmp_path)
    monkeypatch.setattr(
        "neuron_explorer.attribution.moe.xllm_huginn.mount.import_module",
        lambda _name: (_ for _ in ()).throw(RuntimeError("undefined symbol")),
    )
    arguments = {key: value for key, value in fixture.items() if key != "extension"}

    report = check_moe_mounts(**arguments)

    assert report["ok"] is False
    assert _report_check(report, "xllm_extension")["ok"] is False


def test_importing_package_does_not_import_xllm() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import neuron_explorer.attribution.moe.xllm_huginn; "
                "assert not any(name == 'xllm' or name.startswith('xllm.') "
                "for name in sys.modules)"
            ),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
