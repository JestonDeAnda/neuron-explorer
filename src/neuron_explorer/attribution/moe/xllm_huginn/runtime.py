"""Pinned xLLM source verification and strict Huginn native loading."""

from __future__ import annotations

import hashlib
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


def _fingerprinted_files(root: Path) -> list[Path]:
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
    """Require a Git checkout at the exact pinned xLLM commit."""
    source = Path(path)
    head = _git(source, "rev-parse", "HEAD")
    if head != expected_commit:
        raise RuntimeError(
            f"xLLM source commit {head!r} != expected {expected_commit!r}"
        )
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
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size != 250624:
        raise RuntimeError(
            f"xLLM tokenizer vocabulary length {vocab_size!r}, expected 250624"
        )
    bos_id = getattr(tokenizer, "bos_id", getattr(tokenizer, "bos_token_id", None))
    if bos_id != 0:
        raise RuntimeError(f"xLLM tokenizer BOS {bos_id!r}, expected 0")


def load_xllm_huginn(
    model_path: str | Path,
    source_path: str | Path,
    device: str,
) -> tuple[object, object, object, str, str]:
    """Load the pinned native model and verify its complete runtime identity."""
    model_root = Path(model_path)
    source_root = Path(source_path)
    source_commit = verify_xllm_source(source_root, XLLM_SOURCE_COMMIT)
    source_text = str(source_root)
    sys.path[:] = [entry for entry in sys.path if entry != source_text]
    sys.path.insert(0, source_text)

    # Keep this import local: importing xLLM loads its CUDA extension.
    from xllm.paper_part1.native_inference import load_native_model

    model, tokenizer, model_config = load_native_model(
        model_root,
        device=device,
    )
    _verify_runtime_identity(model, tokenizer, model_config)
    return (
        tokenizer,
        model,
        model_config,
        checkpoint_fingerprint(model_root),
        source_commit,
    )
