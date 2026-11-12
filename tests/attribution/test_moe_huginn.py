"""xLLM Huginn schedule, runtime identity, and mount contracts."""

from __future__ import annotations

import json
import hashlib
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


def _checkpoint(root: Path) -> None:
    (root / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    (root / "model.safetensors.index.json").write_text('{"weight_map": {}}', encoding="utf-8")
    (root / "native_model.py").write_text("MODEL = 'huginn'\n", encoding="utf-8")
    (root / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    names = [
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00001.safetensors",
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


def test_pinned_source_commit_is_exact_release() -> None:
    assert XLLM_SOURCE_COMMIT == "755c7b3a91081fe0f83500eabd53bd2490b8b745"


def test_native_loader_prioritizes_pinned_source_and_returns_verified_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "xllm-source"
    source.mkdir()
    model_path = tmp_path / "model"
    model_path.mkdir()
    calls: list[tuple[Path, str]] = []

    class Huginn:
        pass

    tokenizer = SimpleNamespace(vocab_size=250624, bos_id=0)
    config = SimpleNamespace(
        num_layers=28,
        huginn_prelude_layers=8,
        huginn_recurrent_layers=12,
        huginn_coda_layers=8,
        loop_times=8,
        huginn_hierarchical_state="none",
        num_experts=25,
        num_activated_experts=2,
        expert_inter_dim=2432,
    )

    def fake_load(path: Path, *, device: str):
        calls.append((path, device))
        return Huginn(), tokenizer, config

    native = types.ModuleType("xllm.paper_part1.native_inference")
    native.load_native_model = fake_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xllm", types.ModuleType("xllm"))
    monkeypatch.setitem(sys.modules, "xllm.paper_part1", types.ModuleType("xllm.paper_part1"))
    monkeypatch.setitem(sys.modules, "xllm.paper_part1.native_inference", native)
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


def test_mount_report_has_individual_passed_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    stage = tmp_path / "stage"
    (stage / "src" / "neuron_explorer" / "attribution" / "moe" / "xllm_huginn").mkdir(
        parents=True
    )
    (stage / "COMMIT").write_text("project-commit\n", encoding="utf-8")
    output_root = tmp_path / "output"
    extension = type(
        "Extension",
        (),
        {"te": type("TE", (), {"general_grouped_gemm": object()})()},
    )()
    monkeypatch.setattr(
        "neuron_explorer.attribution.moe.xllm_huginn.mount.import_module",
        lambda name: extension if name == "xllm_extension" else None,
    )

    report = check_moe_mounts(
        dataset=dataset,
        model=model,
        xllm_source=source,
        expected_xllm_commit=source_commit,
        stage=stage,
        commit="project-commit",
        output_root=output_root,
    )

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
