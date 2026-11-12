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
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

from neuron_explorer.attribution.moe import (
    MoELinearizationBackend,
    MoESourceLeaf,
    build_moe_graph,
)
from neuron_explorer.attribution.moe.xllm_huginn.linearized import (
    XLLMHuginnLinearized,
    _attention_forward,
)
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
    save_file(
        {"layer.weight": _checkpoint_tensor()},
        root / "model-00001-of-00001.safetensors",
    )
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


class _TokenizerWithoutLength:
    bos_id = 0

    def __init__(self, backend=None) -> None:
        if backend is not None:
            self._tokenizer = backend


def _checkpoint_tensor() -> torch.Tensor:
    return torch.arange(6, dtype=torch.float32).reshape(2, 3)


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


def _runtime_model(
    class_name: str = "Huginn",
    *,
    state_tensor: torch.Tensor | None = None,
    **overrides: object,
):
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
    tensor = _checkpoint_tensor() if state_tensor is None else state_tensor
    model.state_dict = lambda: {"layer.weight": tensor}
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


@pytest.mark.parametrize(
    ("state_tensor", "message"),
    [
        (torch.zeros(3, 2, dtype=torch.float32), "shape"),
        (torch.zeros(2, 3, dtype=torch.float64), "dtype"),
        (_checkpoint_tensor() + 1, "value"),
    ],
)
def test_native_loader_rejects_tensor_content_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state_tensor: torch.Tensor,
    message: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    model_path = tmp_path / "model"
    model_path.mkdir()
    _checkpoint(model_path)
    model = _runtime_model(state_tensor=state_tensor)
    _install_fake_xllm(monkeypatch, source, model=model)
    monkeypatch.setattr(
        "neuron_explorer.attribution.moe.xllm_huginn.runtime.verify_xllm_source",
        lambda path, expected: XLLM_SOURCE_COMMIT,
    )

    with pytest.raises(RuntimeError, match=message):
        load_xllm_huginn(model_path, source, "cpu")


def test_native_loader_uses_backing_tokenizer_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    model_path = tmp_path / "model"
    model_path.mkdir()
    _checkpoint(model_path)
    tokenizer = _TokenizerWithoutLength(_FakeTokenizer())
    _install_fake_xllm(monkeypatch, source, tokenizer=tokenizer)
    monkeypatch.setattr(
        "neuron_explorer.attribution.moe.xllm_huginn.runtime.verify_xllm_source",
        lambda path, expected: XLLM_SOURCE_COMMIT,
    )

    loaded = load_xllm_huginn(model_path, source, "cpu")

    assert loaded[0] is tokenizer


@pytest.mark.parametrize(
    "tokenizer",
    [
        _TokenizerWithoutLength(),
        _TokenizerWithoutLength(object()),
    ],
)
def test_native_loader_rejects_tokenizer_without_any_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: _TokenizerWithoutLength,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    model_path = tmp_path / "model"
    model_path.mkdir()
    _checkpoint(model_path)
    _install_fake_xllm(
        monkeypatch,
        source,
        tokenizer=tokenizer,
    )
    monkeypatch.setattr(
        "neuron_explorer.attribution.moe.xllm_huginn.runtime.verify_xllm_source",
        lambda path, expected: XLLM_SOURCE_COMMIT,
    )

    with pytest.raises(RuntimeError, match="actual length"):
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


# Tiny modules with the public names and attributes of the pinned xLLM classes.
# The linearized backend discovers these classes from the model object and looks
# up ``apply_rotary_embeddings`` and ``mgmm`` in each class's defining module.


def apply_rotary_embeddings(xq, xk, freqs_cis, backward):
    if backward:
        freqs_cis = freqs_cis.conj()
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def mgmm(x, weight, group_sizes, transpose, backend):
    del backend
    chunks = torch.split(x, [int(size) for size in group_sizes])
    return torch.cat(
        [
            chunk @ (expert.transpose(0, 1) if transpose else expert)
            for chunk, expert in zip(chunks, weight, strict=True)
        ]
    )


def _repeat_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, length, heads, dim = x.shape
    return x.unsqueeze(3).expand(batch, length, heads, n_rep, dim).reshape(
        batch, length, heads * n_rep, dim
    )


class GroupRMSNorm(nn.Module):
    def __init__(self, num_features: int, num_groups: int = 1, eps: float = 1e-6):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.groups_per_partition = num_groups
        self.gather_input = False
        self.eps = eps
        self.elementwise_affine = True
        self.weight = nn.Parameter(0.2 * torch.randn(num_features))

    def forward(self, x):
        features = x.float().reshape(*x.shape[:-1], self.groups_per_partition, -1)
        rstd = torch.rsqrt(features.pow(2).mean(-1, keepdim=True) + self.eps)
        return (features * rstd).reshape(x.shape).to(x.dtype) * (self.weight + 1)


class CausalSoftmaxAttention(nn.Module):
    def __init__(self, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rope_head_dim = head_dim
        self.dropout = 0.0
        self.backend = "flash"
        self.scale = 1.0 / head_dim**0.5

    def forward(
        self, xq, xk, xv, g, freqs_cis, segments,
        fp32_attn_output=False, deterministic=True, cache=None,
    ):
        del g, segments, fp32_attn_output, deterministic, cache
        batch, length, _ = xq.shape
        xq = xq.reshape(batch, length, self.n_heads, self.head_dim)
        xk = xk.reshape(batch, length, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(batch, length, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_embeddings(xq, xk, freqs_cis=freqs_cis, backward=False)
        n_rep = self.n_heads // self.n_kv_heads
        xk = _repeat_heads(xk, n_rep).permute(0, 2, 3, 1)
        xv = _repeat_heads(xv, n_rep).transpose(1, 2)
        xq = xq.transpose(1, 2)
        mask = torch.full((batch, length, length), float("-inf")).triu(1).type_as(xq)
        scores = torch.matmul(xq, xk) * self.scale + mask.unsqueeze(1)
        scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(xq)
        output = torch.matmul(scores, xv).transpose(1, 2)
        return output.reshape(batch, length, -1), None


class BaseResidual(nn.Module):
    def forward(self, h, residual, *args, **kwargs):
        del args, kwargs
        return h + residual


class OrthoResidual(BaseResidual):
    def forward(self, h, residual, *args, **kwargs):
        del args, kwargs
        coefficient = (h * residual).sum(-1, keepdim=True) / (
            residual.square().sum(-1, keepdim=True) + 1e-6
        )
        return h - coefficient * residual + residual


class MultiheadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.norm = GroupRMSNorm(dim)
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.causal_attention = CausalSoftmaxAttention(n_heads, n_kv_heads, head_dim)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.apply_attn_gate = False
        self.wr = None
        self.residual = BaseResidual()

    def forward(
        self, x, freqs_cis, segments=None, fp32_attn_output=False,
        deterministic=True, cache=None,
    ):
        hidden = self.norm(x)
        attention, new_cache = self.causal_attention(
            self.wq(hidden), self.wk(hidden), self.wv(hidden), None, freqs_cis,
            segments, fp32_attn_output, deterministic, cache,
        )
        return self.residual(self.wo(attention), x, c=attention), new_cache


class TopKRouter(nn.Module):
    def __init__(self, dim: int, num_experts: int, topk: int):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.score_func = "sigmoid"
        self.scaling_factor = None
        self.weight = nn.Parameter(torch.randn(num_experts, dim))
        self.register_buffer("bias", None)
        self.native_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, x, load_balancing_type=None):
        del load_balancing_type
        scores = torch.sigmoid(F.linear(x, self.weight).float())
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        selected = torch.gather(scores, -1, indices)
        selected = selected / selected.sum(-1, keepdim=True)
        self.native_calls.append((indices.detach().clone(), selected.detach().clone()))
        tokens_per_expert = torch.bincount(indices.flatten(), minlength=self.num_experts)
        return selected.to(x), indices, tokens_per_expert, None


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, num_experts: int):
        super().__init__()
        self.model_dim = dim
        self.inter_dim = inter_dim
        self.n_experts = num_experts
        self.n_local_experts = num_experts
        self.backend = "torch"
        self.hidden_dropout = 0.0
        self.weight1 = nn.Parameter(0.6 * torch.randn(num_experts * inter_dim, dim))
        self.weight2 = nn.Parameter(0.6 * torch.randn(num_experts * dim, inter_dim))
        self.weight3 = nn.Parameter(0.6 * torch.randn(num_experts * inter_dim, dim))

    def forward(self, x, group_sizes):
        w1 = self.weight1.reshape(self.n_local_experts, self.inter_dim, self.model_dim)
        w2 = self.weight2.reshape(self.n_local_experts, self.model_dim, self.inter_dim)
        w3 = self.weight3.reshape(self.n_local_experts, self.inter_dim, self.model_dim)
        h1 = F.silu(mgmm(x, w1, group_sizes, True, self.backend))
        h3 = mgmm(x, w3, group_sizes, True, self.backend)
        return mgmm(h1 * h3, w2, group_sizes, True, self.backend)


class NormalizedMoE(nn.Module):
    def __init__(self, dim: int, inter_dim: int, num_experts: int, topk: int):
        super().__init__()
        self.n_experts = num_experts
        self.n_local_experts = num_experts
        self.inter_dim = inter_dim
        self.topk = topk
        self.n_shared_experts = 0
        self.norm = GroupRMSNorm(dim)
        self.router = TopKRouter(dim, num_experts, topk)
        self.experts = Expert(dim, inter_dim, num_experts)
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None
        self.residual = BaseResidual()

    def forward(self, x, load_balancing_type=None):
        batch, length, _ = x.shape
        hidden = self.norm(x)
        scores, indices, tokens_per_expert, aux = self.router(hidden, load_balancing_type)
        flat = hidden.reshape(batch * length, -1)
        sorted_indices = torch.argsort(indices.flatten(), stable=True)
        permute_indices = sorted_indices // self.topk
        routed = self.experts(
            torch.index_select(flat, 0, permute_indices), tokens_per_expert.tolist()
        )
        weights = scores.flatten()[sorted_indices].unsqueeze(-1)
        combined = torch.zeros_like(flat).index_add(0, permute_indices, routed * weights)
        y = combined.reshape(batch, length, -1)
        return self.residual(y, x, c=y), aux


class TransformerMoEBlock(nn.Module):
    def __init__(self, layer_id: int, **sizes: int):
        super().__init__()
        self.layer_id = layer_id
        self.attention = MultiheadAttention(
            sizes["dim"], sizes["n_heads"], sizes["n_kv_heads"], sizes["head_dim"]
        )
        self.moe = NormalizedMoE(
            sizes["dim"], sizes["inter_dim"], sizes["num_experts"], sizes["topk"]
        )

    def forward(
        self, x, freqs_cis, segments=None, moe_router_load_balancing_type=None,
        fp32_attn_output=False, deterministic=True, cache=None,
    ):
        y, cache = self.attention(
            x, freqs_cis, segments, fp32_attn_output, deterministic, cache
        )
        out, aux = self.moe(y, moe_router_load_balancing_type)
        return out, aux, cache


class InputInjection(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.input_adapter = nn.Linear(dim, dim, bias=False)
        self.A_log = nn.Parameter(torch.randn(dim) * 0.1)
        self.dt_bias = nn.Parameter(torch.randn(dim) * 0.1)

    def forward(self, state, input_embeds):
        dt = F.softplus(self.dt_bias)
        decay = torch.exp(-dt * torch.exp(self.A_log))
        return state * decay + dt * self.input_adapter(input_embeds)


class TransformerOutputLayer(nn.Module):
    def __init__(self, dim: int, vocab: int):
        super().__init__()
        self.final_norm = GroupRMSNorm(dim)
        self.output = nn.Linear(dim, vocab, bias=False)

    def forward(self, x, y, mask=None):
        del y, mask
        return self.output(self.final_norm(x)).float()


class _Rope:
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        self.freqs = 1.0 / torch.tensor(
            [base ** (j / dim) for j in range(0, dim, 2)], dtype=torch.float32
        )

    def get_freqs_cis(self, start: int, end: int, device) -> torch.Tensor:
        positions = torch.arange(start, end, dtype=torch.float32, device=device)
        angles = torch.outer(positions, self.freqs.to(device))
        return torch.polar(torch.ones_like(angles), angles)


class Huginn(nn.Module):
    def __init__(
        self, *, prelude: int = 1, recurrent: int = 2, coda: int = 1,
        loops: int = 2, num_experts: int = 6, seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        sizes = {
            "dim": 8, "n_heads": 4, "n_kv_heads": 2, "head_dim": 4,
            "inter_dim": 3, "num_experts": num_experts, "topk": 2,
        }
        self.num_layers = prelude + recurrent + coda
        self.huginn_prelude_layers = prelude
        self.huginn_recurrent_layers = recurrent
        self.huginn_coda_layers = coda
        self.loop_times = loops
        self.loop_start_layer = prelude
        self.loop_end_layers = prelude + recurrent
        self.huginn_state_init = "normal"
        self.huginn_hierarchical_state = "none"
        self.embed = nn.Embedding(17, sizes["dim"])
        self.rope = _Rope(sizes["head_dim"])
        self.layers = nn.ModuleList(
            TransformerMoEBlock(index, **sizes) for index in range(self.num_layers)
        )
        self.input_injection = InputInjection(sizes["dim"])
        self.output = TransformerOutputLayer(sizes["dim"], 17)

    def initialize_recurrent_state(self, input_embeds, cache_len=0):
        batch, length, dim = input_embeds.shape
        row = torch.arange(batch, dtype=torch.float32).view(-1, 1, 1) * 0.37
        position = torch.arange(cache_len, cache_len + length).view(1, -1, 1) * 0.21
        channel = torch.arange(dim, dtype=torch.float32).view(1, 1, -1) * 0.13
        return (0.5 * torch.sin(row + position + channel + 0.1)).to(input_embeds.dtype)

    def _run_layers(self, layer_ids, hidden, freqs):
        for layer_id in layer_ids:
            hidden, _aux, _cache = self.layers[layer_id](
                hidden, freqs, None, moe_router_load_balancing_type=None,
                fp32_attn_output=False, deterministic=True, cache=None,
            )
        return hidden

    def forward(
        self, tokens, multi_segments, targets=None, token_mask=None,
        moe_router_load_balancing_type=None, fp32_attn_output=False,
        deterministic=True, cache=None,
    ):
        del multi_segments, moe_router_load_balancing_type, fp32_attn_output
        del deterministic, cache
        hidden = self.embed(tokens)
        freqs = self.rope.get_freqs_cis(0, tokens.shape[1], hidden.device)
        hidden = self._run_layers(range(self.loop_start_layer), hidden, freqs)
        input_embeds = hidden
        hidden = self.initialize_recurrent_state(input_embeds)
        for _ in range(self.loop_times):
            hidden = self.input_injection(hidden, input_embeds)
            hidden = self._run_layers(
                range(self.loop_start_layer, self.loop_end_layers), hidden, freqs
            )
        hidden = self._run_layers(
            range(self.loop_end_layers, self.num_layers), hidden, freqs
        )
        return self.output(hidden, targets, token_mask), None, None


def _tiny_config(model: Huginn) -> dict[str, object]:
    return {
        "model": {
            "num_layers": model.num_layers,
            "huginn_prelude_layers": model.huginn_prelude_layers,
            "huginn_recurrent_layers": model.huginn_recurrent_layers,
            "huginn_coda_layers": model.huginn_coda_layers,
            "loop_times": model.loop_times,
            "huginn_hierarchical_state": "none",
        }
    }


def _tiny_backend(**kwargs: int) -> tuple[Huginn, XLLMHuginnLinearized]:
    model = Huginn(**kwargs).eval()
    return model, XLLMHuginnLinearized(model, _tiny_config(model))


_TOKENS = [1, 5, 9, 3, 12]


def _native_logits(model: Huginn, tokens=_TOKENS, copies: int = 1) -> torch.Tensor:
    ids = torch.tensor([list(tokens)] * copies)
    with torch.no_grad():
        return model(ids, multi_segments=False)[0]


def _patched_forwards(model: nn.Module) -> list[str]:
    return [name for name, module in model.named_modules() if "forward" in module.__dict__]


def test_linearized_backend_satisfies_protocol_and_identity() -> None:
    _model, backend = _tiny_backend()

    assert isinstance(backend, MoELinearizationBackend)
    assert len(backend.schedule) == 6
    assert backend.identity.num_experts == 6
    assert backend.identity.expert_width == 3
    assert backend.identity.top_k == 2
    assert backend.identity.router_activation == "sigmoid"
    assert backend.identity.router_normalization == "selected_renormalized"
    assert backend.identity.has_shared_experts is False


def test_linearized_backend_accepts_attribute_config() -> None:
    model = Huginn().eval()
    config = SimpleNamespace(**_tiny_config(model)["model"])

    backend = XLLMHuginnLinearized(model, config)

    assert [slot.virtual_depth for slot in backend.schedule] == list(range(6))


def test_frozen_rms_uses_cached_rstd_and_preserves_forward_values() -> None:
    model, backend = _tiny_backend()
    norm = model.output.final_norm
    first = torch.randn(1, 3, 8, requires_grad=True)
    second = 3.0 * torch.randn(1, 3, 8)
    native_first = norm(first).detach()
    native_rstd = torch.rsqrt(first.detach().pow(2).mean(-1, keepdim=True) + norm.eps)

    with backend:
        backend._reset(replay=False)
        patched_first = norm(first)
        gradient = torch.autograd.grad(patched_first[0, 1, 2], first)[0]
        backend._reset(replay=True)
        patched_second = norm(second)

    torch.testing.assert_close(patched_first.detach(), native_first)
    expected_gradient = torch.zeros_like(first)
    expected_gradient[0, 1, 2] = native_rstd[0, 1, 0] * (norm.weight[2].detach() + 1)
    torch.testing.assert_close(gradient, expected_gradient)
    torch.testing.assert_close(
        patched_second, second * native_rstd * (norm.weight.detach() + 1)
    )


def _attention_inputs(module: CausalSoftmaxAttention, length: int = 5):
    generator = torch.Generator().manual_seed(11)
    xq = torch.randn(1, length, module.n_heads * module.head_dim, generator=generator)
    xk = torch.randn(1, length, module.n_kv_heads * module.head_dim, generator=generator)
    xv = torch.randn(1, length, module.n_kv_heads * module.head_dim, generator=generator)
    freqs = _Rope(module.head_dim).get_freqs_cis(0, length, xq.device)
    return xq, xk, xv, freqs


def test_explicit_attention_matches_native_and_freezes_pattern() -> None:
    model, backend = _tiny_backend()
    attention = model.layers[0].attention.causal_attention
    xq, xk, xv, freqs = _attention_inputs(attention)
    native, _cache = attention(xq, xk, xv, None, freqs, None)
    xq = xq.clone().requires_grad_(True)
    xv = xv.clone().requires_grad_(True)

    with backend:
        backend._reset(replay=False)
        patched, cache = attention(xq, xk, xv, None, freqs, None)
        query_gradient, value_gradient = torch.autograd.grad(
            patched.sum(), (xq, xv), allow_unused=True
        )

    assert cache is None
    assert patched.shape == native.shape
    torch.testing.assert_close(patched.detach(), native)
    assert query_gradient is None or torch.count_nonzero(query_gradient) == 0
    assert value_gradient is not None and torch.count_nonzero(value_gradient) > 0


def test_explicit_attention_repeats_six_kv_heads_to_24_query_heads() -> None:
    attention = CausalSoftmaxAttention(n_heads=24, n_kv_heads=6, head_dim=2)
    _model, backend = _tiny_backend()
    xq, xk, xv, freqs = _attention_inputs(attention)
    native, _cache = attention(xq, xk, xv, None, freqs, None)
    changed_value = xv.clone().reshape(1, 5, 6, 2)
    changed_value[:, :, 1] += 10.0
    changed_value = changed_value.reshape(1, 5, 12)

    with backend:
        backend._patch(attention, _attention_forward)
        backend._reset(replay=False)
        patched, _cache = attention(xq, xk, xv, None, freqs, None)
        backend._reset(replay=True)
        changed, _cache = attention(xq, xk, changed_value, None, freqs, None)

    assert "forward" not in attention.__dict__
    torch.testing.assert_close(patched, native)
    difference = (changed - patched).reshape(1, 5, 24, 2).abs().amax(dim=(0, 1, 3))
    assert torch.all(difference[4:8] > 0)
    assert torch.count_nonzero(difference[:4]) == 0
    assert torch.count_nonzero(difference[8:]) == 0


def test_explicit_attention_masks_future_positions() -> None:
    model, backend = _tiny_backend()
    attention = model.layers[0].attention.causal_attention
    xq, xk, xv, freqs = _attention_inputs(attention)
    changed_key = xk.clone()
    changed_value = xv.clone()
    changed_key[:, -1] += 5.0
    changed_value[:, -1] += 5.0

    with backend:
        backend._reset(replay=False)
        baseline, _cache = attention(xq, xk, xv, None, freqs, None)
        backend._reset(replay=False)
        changed, _cache = attention(xq, changed_key, changed_value, None, freqs, None)

    torch.testing.assert_close(changed[:, :-1], baseline[:, :-1])
    assert not torch.allclose(changed[:, -1], baseline[:, -1])


def test_patched_forward_matches_native_logits_in_both_modes() -> None:
    model, backend = _tiny_backend()
    native = _native_logits(model)

    with backend:
        direct = backend.forward(_TOKENS, mode="direct")
        total = backend.forward(_TOKENS, mode="total")

    torch.testing.assert_close(direct.detach(), native)
    torch.testing.assert_close(total.detach(), native)


def test_route_snapshots_preserve_ids_rank_row_and_weight() -> None:
    model, backend = _tiny_backend()
    routers = [layer.moe.router for layer in model.layers]
    for router in routers:
        router.native_calls.clear()

    with backend:
        backend.forward(_TOKENS, mode="total")
        snapshot = backend.routing_snapshot
        neurons = list(backend.routed_neurons())

    calls = {
        slot.virtual_depth: routers[slot.physical_layer].native_calls[
            slot.recurrence_iteration or 0
        ]
        for slot in backend.schedule
    }
    width = backend.identity.expert_width
    assert snapshot.selected_experts.shape == (6, len(_TOKENS), 2)
    assert torch.all(snapshot.selected_expert_counts == 2)
    for depth, (indices, weights) in calls.items():
        torch.testing.assert_close(snapshot.selected_experts[depth], indices[0])
        torch.testing.assert_close(snapshot.routing_weights[depth], weights[0])
    assert len(neurons) == 6 * len(_TOKENS) * 2 * width
    for neuron in neurons[:: width]:
        indices, weights = calls[neuron.slot.virtual_depth]
        assert neuron.expert_id == int(indices[0, neuron.position, neuron.router_rank])
        assert neuron.routing_weight == pytest.approx(
            float(weights[0, neuron.position, neuron.router_rank])
        )


def test_direct_expert_leaves_map_grouped_rows_to_position_and_expert() -> None:
    model, backend = _tiny_backend()
    captured: dict[int, list[torch.Tensor]] = {}
    hooks = [
        layer.moe.norm.register_forward_hook(
            lambda module, args, output, index=index: captured.setdefault(
                index, []
            ).append(output.detach())
        )
        for index, layer in enumerate(model.layers)
    ]

    try:
        with backend:
            backend.forward(_TOKENS, copies=2, mode="direct")
            neurons = list(backend.routed_neurons())
            values = backend.activations(neurons)
    finally:
        for hook in hooks:
            hook.remove()

    experts = model.layers[0].moe.experts
    width = experts.inter_dim
    for neuron, value in zip(neurons, values, strict=True):
        slot = neuron.slot
        layer = model.layers[slot.physical_layer].moe.experts
        hidden = captured[slot.physical_layer][slot.recurrence_iteration or 0]
        row = hidden[0, neuron.position]
        start = neuron.expert_id * width
        gate = row @ layer.weight1[start + neuron.expert_neuron].detach()
        up = row @ layer.weight3[start + neuron.expert_neuron].detach()
        torch.testing.assert_close(value, F.silu(gate) * up)


def test_unselected_expert_cannot_change_frozen_route_output() -> None:
    model, backend = _tiny_backend(num_experts=8)

    with backend:
        baseline = backend.forward(_TOKENS[:3], mode="total").detach()
        selected = set(backend.routing_snapshot.selected_experts[0].flatten().tolist())
        unselected = next(expert for expert in range(8) if expert not in selected)
        experts = model.layers[0].moe.experts
        width = experts.inter_dim
        with torch.no_grad():
            rows = slice(unselected * width, (unselected + 1) * width)
            experts.weight1[rows] += 50.0
            experts.weight3[rows] -= 50.0
            experts.weight2[unselected * 8 : (unselected + 1) * 8] += 50.0
        replayed = backend.forward(_TOKENS[:3], copies=2, replay=True, mode="direct")

    torch.testing.assert_close(replayed[0].detach(), baseline[0])
    torch.testing.assert_close(replayed[1].detach(), baseline[0])


def test_half_rule_forward_equals_native_swiglu() -> None:
    _model, backend = _tiny_backend()

    with backend:
        backend.forward(_TOKENS, mode="total")
        total_values = backend.activations(list(backend.routed_neurons()))
        backend.forward(_TOKENS, copies=3, mode="direct")
        neurons = list(backend.routed_neurons())
        direct_values = backend.activations(neurons)
        targets = backend.direct_targets([0, 7, 31], 3, _TOKENS)

    torch.testing.assert_close(total_values, direct_values)
    torch.testing.assert_close(targets.targets, direct_values[[0, 7, 31]])


def test_replay_copies_row_zero_recurrent_state_across_edge_rows() -> None:
    model, backend = _tiny_backend()
    native_row_zero = model.initialize_recurrent_state(torch.zeros(1, len(_TOKENS), 8))

    with backend:
        backend.forward(_TOKENS, copies=1, mode="total")
        logits = backend.forward(_TOKENS, copies=3, replay=True, mode="direct")
        state = backend.state_leaf
        fresh = backend.forward(_TOKENS, copies=3, mode="direct")
        fresh_state = backend.state_leaf

    assert state.shape == (3, len(_TOKENS), 8)
    for row in range(3):
        torch.testing.assert_close(state[row].detach(), native_row_zero[0])
        torch.testing.assert_close(fresh_state[row].detach(), native_row_zero[0])
        torch.testing.assert_close(logits[row].detach(), fresh[0].detach())
    assert state.requires_grad


def test_invocation_counts_map_8_96_8_calls_to_112_virtual_depths() -> None:
    model, backend = _tiny_backend(prelude=8, recurrent=12, coda=8, loops=8)

    with backend:
        backend.forward(_TOKENS[:2], mode="direct")
        neurons = backend.routed_neurons()
        depths = sorted({neuron.slot.virtual_depth for neuron in neurons})
        counts = dict(backend.invocation_counts)

    assert depths == list(range(112))
    experts = [id(layer.moe.experts) for layer in model.layers]
    assert [counts[key] for key in experts] == [1] * 8 + [8] * 12 + [1] * 8
    recurrent = [n.slot for n in neurons if n.slot.stage == "recurrent"]
    assert {slot.recurrence_iteration for slot in recurrent} == set(range(8))
    assert {
        (slot.virtual_depth, slot.physical_layer)
        for slot in recurrent
        if slot.recurrence_iteration == 7
    } == {(92 + block, 8 + block) for block in range(12)}


def test_unexpected_execution_count_fails_closed() -> None:
    model, backend = _tiny_backend()
    model.loop_times = 3

    with backend, pytest.raises(RuntimeError, match="ran 3 times"):
        backend.forward(_TOKENS, mode="direct")


def test_replay_rejects_inconsistent_routes() -> None:
    model, backend = _tiny_backend()

    with backend:
        backend.forward(_TOKENS, mode="total")
        with torch.no_grad():
            model.layers[0].moe.router.weight.mul_(-1.0)
        with pytest.raises(RuntimeError, match="replay route"):
            backend.forward(_TOKENS, copies=2, replay=True, mode="direct")


def test_replay_requires_cached_forward_for_same_tokens() -> None:
    _model, backend = _tiny_backend()

    with backend:
        with pytest.raises(RuntimeError, match="replay"):
            backend.forward(_TOKENS, replay=True)
        backend.forward(_TOKENS, mode="total")
        with pytest.raises(RuntimeError, match="replay"):
            backend.forward(_TOKENS[:-1], replay=True)


def test_direct_targets_have_contribution_shapes_and_no_self_edges() -> None:
    _model, backend = _tiny_backend()
    completion_start = 3

    with backend:
        backend.forward(_TOKENS, mode="total")
        axis = list(backend.routed_neurons())
        backend.forward(_TOKENS, copies=3, replay=True, mode="direct")
        late = next(
            index for index, neuron in enumerate(axis)
            if neuron.slot.virtual_depth == 5 and neuron.position == 2
        )
        contribution = backend.direct_targets([late, 2, -1], completion_start, _TOKENS)
        completion = backend.completion_logprob(0, completion_start, _TOKENS)

    assert tuple(contribution.routed_neurons) == tuple(axis)
    assert contribution.routed.shape == (3, len(axis))
    assert contribution.token_embeddings.shape == (3, len(_TOKENS))
    assert contribution.source_leaves == (
        MoESourceLeaf(
            "recurrent_state",
            kind="huginn_state",
            metadata=(("state_init", "normal"),),
        ),
    )
    assert contribution.source_contributions.shape == (3, 1)
    assert contribution.routed[0, late] == 0
    assert contribution.routed[1, 2] == 0
    later = [
        index for index, neuron in enumerate(axis)
        if neuron.slot.virtual_depth >= 5 or neuron.position > 2
    ]
    assert torch.count_nonzero(contribution.routed[0, later]) == 0
    assert torch.count_nonzero(contribution.routed[0]) > 0
    assert torch.count_nonzero(contribution.token_embeddings[0]) > 0
    assert torch.count_nonzero(contribution.routed[2]) > 0
    torch.testing.assert_close(contribution.targets[2], completion.detach())
    assert contribution.source_contributions[0, 0] != 0


def test_direct_targets_reject_invalid_requests() -> None:
    _model, backend = _tiny_backend()

    with backend:
        backend.forward(_TOKENS, copies=2, mode="direct")
        with pytest.raises(ValueError, match="copies"):
            backend.direct_targets([0, 1, -1], 3, _TOKENS)
        backend.forward(_TOKENS, copies=2, mode="total")
        with pytest.raises(RuntimeError, match="direct"):
            backend.direct_targets([0, -1], 3, _TOKENS)
        with pytest.raises(ValueError, match="routed axis"):
            backend.forward(_TOKENS, copies=2, mode="direct")
            backend.direct_targets([10**6], 3, _TOKENS)


def test_candidate_effects_align_with_routed_axis() -> None:
    _model, backend = _tiny_backend()

    with backend:
        backend.forward(_TOKENS, mode="total")
        effects = backend.candidate_effects(3, _TOKENS)
        axis = backend.routed_neurons()

    assert effects.shape == (len(axis),)
    assert torch.count_nonzero(effects) > 0
    assert not effects.requires_grad


def test_tiny_xllm_backend_builds_generic_graph() -> None:
    _model, backend = _tiny_backend()

    with backend:
        graph = build_moe_graph(backend, _TOKENS, 3, k=12, batch=4, max_edges=200)

    neurons = [node for node in graph["nodes"] if node["kind"] == "neuron"]
    assert neurons
    assert all(0 <= node["expert_id"] < 6 for node in neurons)
    assert all(node["router_rank"] in {0, 1} for node in neurons)
    assert graph["metadata"]["model_identity"]["model_family"] == "xllm-huginn"


def test_context_exit_restores_forwards_and_requires_grad_after_success() -> None:
    model, backend = _tiny_backend()
    model.layers[0].moe.experts.weight1.requires_grad_(False)
    before = {name: p.requires_grad for name, p in model.named_parameters()}
    original_init = model.__dict__.get("initialize_recurrent_state")

    with backend:
        assert _patched_forwards(model)
        assert not any(p.requires_grad for p in model.parameters())
        backend.forward(_TOKENS, mode="total")

    assert _patched_forwards(model) == []
    assert model.__dict__.get("initialize_recurrent_state") is original_init
    assert {name: p.requires_grad for name, p in model.named_parameters()} == before
    assert not any(hasattr(m, "_xllm_linearized") for m in model.modules())
    torch.testing.assert_close(_native_logits(model), _native_logits(Huginn().eval()))


def test_context_exit_restores_forwards_after_error() -> None:
    model, backend = _tiny_backend()
    before = {name: p.requires_grad for name, p in model.named_parameters()}

    with pytest.raises(RuntimeError, match="ran 3 times"), backend:
        model.loop_times = 3
        backend.forward(_TOKENS, mode="direct")
    model.loop_times = 2
    with pytest.raises(ValueError, match="boom"), backend:
        raise ValueError("boom")

    assert _patched_forwards(model) == []
    assert "initialize_recurrent_state" not in model.__dict__
    assert {name: p.requires_grad for name, p in model.named_parameters()} == before


def _gate_flag(model: Huginn) -> None:
    model.layers[1].attention.apply_attn_gate = True


def _gate_module(model: Huginn) -> None:
    model.layers[0].attention.wr = nn.Linear(8, 16, bias=False)


def _ortho_attention(model: Huginn) -> None:
    model.layers[1].attention.residual = OrthoResidual()


def _ortho_moe(model: Huginn) -> None:
    model.layers[2].moe.residual = OrthoResidual()


def _linear_injection(model: Huginn) -> None:
    model.input_injection = type("LinearInputInjection", (InputInjection,), {})(8)


def _no_injection(model: Huginn) -> None:
    model.input_injection = None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_gate_flag, "attention output gate"),
        (_gate_module, "attention output gate"),
        (_ortho_attention, "attention residual"),
        (_ortho_moe, "MoE residual"),
        (_linear_injection, "InputInjection"),
        (_no_injection, "InputInjection"),
    ],
)
def test_backend_rejects_nonlinear_paths_it_does_not_linearize(mutate, message) -> None:
    model = Huginn().eval()
    mutate(model)

    with pytest.raises(RuntimeError, match=message):
        XLLMHuginnLinearized(model, _tiny_config(model))


def test_backend_rejects_unexpected_runtime_classes() -> None:
    model = Huginn().eval()
    model.layers[0].moe.fc1 = nn.Linear(8, 3)

    with pytest.raises(RuntimeError, match="shared experts"):
        XLLMHuginnLinearized(model, _tiny_config(model))

    other = Huginn().eval()
    other.huginn_state_init = "input"
    with pytest.raises(RuntimeError, match="state init"):
        XLLMHuginnLinearized(other, _tiny_config(other))


def test_forward_requires_entered_context() -> None:
    _model, backend = _tiny_backend()

    with pytest.raises(RuntimeError, match="context"):
        backend.forward(_TOKENS)


def test_importing_linearized_does_not_import_xllm() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import neuron_explorer.attribution.moe.xllm_huginn.linearized; "
                "assert not any(name == 'xllm' or name.startswith('xllm.') "
                "for name in sys.modules)"
            ),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
