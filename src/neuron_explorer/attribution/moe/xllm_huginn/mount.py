"""Validate every mounted artifact required by xLLM Huginn attribution."""

from __future__ import annotations

import hashlib
import json
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any

from neuron_explorer.attribution.moe.xllm_huginn.runtime import (
    XLLM_SOURCE_COMMIT,
    _fingerprinted_files,
    verify_xllm_source,
)


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _dataset_checks(dataset: Path, expected_tasks: int, expected_rows: int):
    files = sorted(
        path
        for path in dataset.glob("*.jsonl")
        if path.is_file() and not path.name.startswith("._")
    )
    yield _check(
        "dataset_tasks",
        len(files) == expected_tasks,
        f"expected {expected_tasks}, found {len(files)} in {dataset}",
    )
    try:
        rows = sum(
            1
            for path in files
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        yield _check(
            "dataset_rows",
            rows == expected_rows,
            f"expected {expected_rows}, found {rows}",
        )
    except (OSError, UnicodeError) as exc:
        yield _check("dataset_rows", False, f"cannot read dataset: {exc}")


def _project_check(stage: Path, commit: str) -> dict[str, Any]:
    package = (
        stage
        / "src"
        / "neuron_explorer"
        / "attribution"
        / "moe"
        / "xllm_huginn"
    )
    commit_file = stage / "COMMIT"
    try:
        recorded = commit_file.read_text(encoding="utf-8").strip()
    except OSError:
        recorded = ""
    ok = package.is_dir() and recorded == commit
    return _check(
        "project_commit",
        ok,
        f"package_present={package.is_dir()} recorded={recorded!r} expected={commit!r}",
    )


def _source_check(source: Path, expected_commit: str) -> dict[str, Any]:
    try:
        actual = verify_xllm_source(source, expected_commit)
    except RuntimeError as exc:
        return _check("xllm_source_commit", False, str(exc))
    return _check("xllm_source_commit", True, actual)


def _checkpoint_files_check(model: Path) -> dict[str, Any]:
    required = [
        model / "config.json",
        model / "model.safetensors.index.json",
    ]
    shards = sorted(model.glob("*.safetensors"))
    missing = [str(path) for path in required if not path.is_file()]
    if not shards:
        missing.append("*.safetensors")
    return _check(
        "checkpoint_files",
        not missing,
        "all required checkpoint files present"
        if not missing
        else f"missing: {', '.join(missing)}",
    )


def _manifest_check(model: Path) -> dict[str, Any]:
    manifest_path = model / "artifact_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest["files"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("files must be a non-empty list")
        listed: set[str] = set()
        for entry in entries:
            relative_text = entry["path"]
            if not isinstance(relative_text, str) or not relative_text:
                raise ValueError("manifest contains an invalid artifact path")
            relative = Path(relative_text)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() != relative_text
            ):
                raise ValueError(f"artifact path escapes checkpoint: {relative_text}")
            if relative_text in listed:
                raise ValueError(f"duplicate artifact path: {relative_text}")
            listed.add(relative_text)
            artifact = model / relative
            if artifact.is_symlink():
                raise ValueError(f"artifact path is a symlink: {relative_text}")
            expected_size = int(entry["size_bytes"])
            if artifact.stat().st_size != expected_size:
                raise ValueError(f"size mismatch: {entry['path']}")
            digest = hashlib.sha256()
            with artifact.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {entry['path']}")
        fingerprint_inputs = {
            path.relative_to(model).as_posix()
            for path in _fingerprinted_files(model)
            if path.name != "artifact_manifest.json"
        }
        unlisted = sorted(fingerprint_inputs - listed)
        if unlisted:
            raise ValueError(f"unlisted fingerprint inputs: {unlisted}")
        index = json.loads(
            (model / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        weight_map = index["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("checkpoint weight_map must be a non-empty object")
        for shard in set(weight_map.values()):
            if not isinstance(shard, str) or not (model / shard).is_file():
                raise ValueError(f"weight-map shard missing: {shard!r}")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _check("artifact_manifest", False, f"manifest verification failed: {exc}")
    return _check(
        "artifact_manifest",
        True,
        f"verified {len(entries)} manifest files",
    )


def _runtime_checks(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        extension = import_module("xllm_extension")
    except Exception as exc:
        extension_check = _check("xllm_extension", False, f"import failed: {exc}")
        extension = None
    else:
        extension_file = getattr(extension, "__file__", None)
        try:
            Path(extension_file).resolve().relative_to(source.resolve())
        except (OSError, TypeError, ValueError):
            extension_check = _check(
                "xllm_extension",
                False,
                f"extension outside pinned source: {extension_file!r}",
            )
            extension = None
        else:
            extension_check = _check(
                "xllm_extension",
                True,
                f"imported pinned extension: {extension_file}",
            )

    source_api = (
        source
        / "xllm"
        / "modules"
        / "fused_ops"
        / "mgmm"
        / "multi_group_matmul.py"
    ).is_file()
    extension_api = (
        extension is not None
        and hasattr(extension, "te")
        and hasattr(extension.te, "general_grouped_gemm")
    )
    grouped_check = _check(
        "grouped_mm",
        source_api and extension_api,
        f"source_api={source_api} extension_api={extension_api}",
    )
    return extension_check, grouped_check


def _output_check(output_root: Path) -> dict[str, Any]:
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=output_root,
            prefix=".moe-huginn-write-",
            delete=True,
        ):
            pass
    except OSError as exc:
        return _check("output_root", False, f"not writable: {exc}")
    return _check("output_root", True, f"writable: {output_root}")


def check_moe_mounts(
    *,
    dataset: str | Path,
    model: str | Path,
    xllm_source: str | Path,
    stage: str | Path,
    commit: str,
    output_root: str | Path,
    expected_xllm_commit: str = XLLM_SOURCE_COMMIT,
    expected_tasks: int = 23,
    expected_rows: int = 1461,
) -> dict[str, Any]:
    """Return one JSON-serializable result per required mount/runtime check."""
    dataset_path = Path(dataset)
    model_path = Path(model)
    source_path = Path(xllm_source)
    checks = [
        *_dataset_checks(dataset_path, expected_tasks, expected_rows),
        _project_check(Path(stage), commit),
        _source_check(source_path, expected_xllm_commit),
        _checkpoint_files_check(model_path),
        _manifest_check(model_path),
        *_runtime_checks(source_path),
        _output_check(Path(output_root)),
    ]
    return {"ok": all(check["ok"] for check in checks), "checks": checks}
