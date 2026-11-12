"""Pinned xLLM source verification and strict Huginn native loading."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

XLLM_SOURCE_COMMIT = "755c7b3a91081fe0f83500eabd53bd2490b8b745"

_MODEL_CONTRACT = {
    "num_layers": 28,
    "huginn_prelude_layers": 8,
    "huginn_recurrent_layers": 12,
    "huginn_coda_layers": 8,
    "loop_times": 8,
    "huginn_hierarchical_state": "none",
    "num_experts": 25,
    "num_activated_experts": 2,
    "expert_inter_dim": 2432,
}

_MODEL_PHYSICAL_CONTRACT = {
    "num_layers": 28,
    "huginn_prelude_layers": 8,
    "huginn_recurrent_layers": 12,
    "huginn_coda_layers": 8,
}

_LOADER_CONFIG_CONTRACT = {
    "init_mode": "none",
    "fused_block": False,
    "fused_output_layer": False,
}


def _fingerprinted_files(root: Path) -> list[Path]:
    if root.is_symlink():
        raise ValueError(f"checkpoint root must not be a symlink: {root}")
    candidates = {
        root / "config.json",
        root / "artifact_manifest.json",
        root / "model.safetensors.index.json",
        *root.rglob("*.py"),
        *root.rglob("*.safetensors"),
    }
    tokenizer = root / "tokenizer"
    if tokenizer.is_dir():
        candidates.update(path for path in tokenizer.rglob("*") if path.is_file())
    symlinks = sorted(path for path in candidates if path.is_symlink())
    if symlinks:
        raise ValueError(f"checkpoint fingerprint input is a symlink: {symlinks[0]}")
    files = sorted(path for path in candidates if path.is_file())
    if not any(path.suffix == ".safetensors" for path in files):
        raise FileNotFoundError(f"no safetensors checkpoint files found in {root}")
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"checkpoint config missing: {root / 'config.json'}")
    return files


def checkpoint_fingerprint(path: Path) -> str:
    """Hash model config, manifest/index, tokenizer, source, and weight bytes."""
    root = Path(path)
    digest = hashlib.sha256()
    for file_path in _fingerprinted_files(root):
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(file_path.stat().st_size.to_bytes(8, "big"))
        with file_path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return f"checkpoint-sha256:{digest.hexdigest()}"


def _git(path: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(f"cannot inspect xLLM source checkout {path}: {detail}") from exc
    return result.stdout.strip()


def verify_xllm_source(path: Path, expected_commit: str) -> str:
    """Require the exact top-level checkout with no tracked modifications."""
    source = Path(path).resolve()
    top_level = Path(_git(source, "rev-parse", "--show-toplevel")).resolve()
    if top_level != source:
        raise RuntimeError(
            f"xLLM source must be the checkout top-level: {source} != {top_level}"
        )
    head = _git(source, "rev-parse", "HEAD")
    if head != expected_commit:
        raise RuntimeError(
            f"xLLM source commit {head!r} != expected {expected_commit!r}"
        )
    tracked_changes = _git(
        source,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_changes:
        raise RuntimeError("xLLM source has tracked modifications")
    return head


def _config_value(config: object, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _verify_runtime_identity(model: object, tokenizer: object, config: object) -> None:
    if model.__class__.__name__ != "Huginn":
        raise RuntimeError(
            f"xLLM native loader returned {model.__class__.__name__}, expected Huginn"
        )
    for name, expected in _MODEL_CONTRACT.items():
        actual = _config_value(config, name)
        if actual != expected:
            raise RuntimeError(f"xLLM model config {name}={actual!r}, expected {expected!r}")
    for name, expected in _LOADER_CONFIG_CONTRACT.items():
        actual = _config_value(config, name)
        if actual != expected:
            raise RuntimeError(
                f"xLLM loader config {name}={actual!r}, expected {expected!r}"
            )
    for name, expected in _MODEL_PHYSICAL_CONTRACT.items():
        actual = getattr(model, name, None)
        if actual != expected:
            raise RuntimeError(
                f"xLLM model {name}={actual!r}, expected {expected!r}"
            )
    try:
        physical_blocks = len(getattr(model, "layers"))
    except (AttributeError, TypeError) as exc:
        raise RuntimeError("xLLM model does not expose physical blocks") from exc
    if physical_blocks != 28:
        raise RuntimeError(
            f"xLLM model has {physical_blocks} physical blocks, expected 28"
        )
    try:
        tokenizer_length = len(tokenizer)  # type: ignore[arg-type]
    except TypeError:
        backend = getattr(tokenizer, "_tokenizer", None)
        if backend is None:
            raise RuntimeError("xLLM tokenizer does not expose its actual length")
        tokenizer_length = len(backend)
    if tokenizer_length != 250624:
        raise RuntimeError(
            f"xLLM tokenizer length {tokenizer_length!r}, expected 250624"
        )
    bos_id = getattr(tokenizer, "bos_id", getattr(tokenizer, "bos_token_id", None))
    if bos_id != 0:
        raise RuntimeError(f"xLLM tokenizer BOS {bos_id!r}, expected 0")


def _verify_strict_state_dict(model: object, model_root: Path) -> None:
    index_path = model_root / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        expected = set(index["weight_map"])
        actual = set(model.state_dict())  # type: ignore[attr-defined]
    except (AttributeError, KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot verify strict state-dict load: {exc}") from exc
    if not expected or actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            "strict state-dict verification failed: "
            f"missing={missing} unexpected={unexpected}"
        )


def _module_inside_source(module: object, source_root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        Path(module_file).resolve().relative_to(source_root.resolve())
    except (OSError, ValueError):
        return False
    return True


def load_xllm_huginn(
    model_path: str | Path,
    source_path: str | Path,
    device: str,
) -> tuple[object, object, object, str, str]:
    """Load the pinned native model and verify its complete runtime identity."""
    model_root = Path(model_path)
    source_root = Path(source_path).resolve()
    source_commit = verify_xllm_source(source_root, XLLM_SOURCE_COMMIT)
    source_text = str(source_root)
    original_path = list(sys.path)
    original_modules = set(sys.modules)
    try:
        preimported = sys.modules.get("xllm")
        if preimported is not None and not _module_inside_source(
            preimported, source_root
        ):
            raise RuntimeError("pre-imported xllm is outside pinned source")
        sys.path[:] = [entry for entry in sys.path if entry != source_text]
        sys.path.insert(0, source_text)

        # Keep these imports local: importing xLLM loads its CUDA extension.
        import xllm
        from xllm.paper_part1.native_inference import load_native_model

        if not _module_inside_source(xllm, source_root):
            raise RuntimeError("imported xllm is outside pinned source")
        model, tokenizer, model_config = load_native_model(
            model_root,
            device=device,
        )
        _verify_runtime_identity(model, tokenizer, model_config)
        _verify_strict_state_dict(model, model_root)
        return (
            tokenizer,
            model,
            model_config,
            checkpoint_fingerprint(model_root),
            source_commit,
        )
    except Exception:
        sys.path[:] = original_path
        for name in set(sys.modules) - original_modules:
            if name == "xllm" or name.startswith("xllm."):
                sys.modules.pop(name, None)
        raise
