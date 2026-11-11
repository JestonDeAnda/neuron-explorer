"""Check the files a Huginn attribution job must see through DLC mounts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def _writable(path: Path) -> bool:
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".huginn-write-", delete=True):
            return True
    except OSError:
        return False


def check_paths(
    *,
    dataset: Path,
    model: Path,
    output_root: Path,
    stage: Path | None = None,
    commit: str | None = None,
    expected_tasks: int = 23,
    expected_rows: int = 1461,
) -> list[str]:
    """Return human-readable errors. An empty list means the mount check passed."""
    errors: list[str] = []
    dataset = Path(dataset)
    model = Path(model)
    output_root = Path(output_root)
    files = sorted(
        path for path in dataset.glob("*.jsonl") if path.is_file() and not path.name.startswith("._")
    )
    if len(files) != expected_tasks:
        errors.append(f"dataset tasks: expected {expected_tasks}, found {len(files)} in {dataset}")
    rows = 0
    for path in files:
        rows += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if files and rows != expected_rows:
        errors.append(f"dataset rows: expected {expected_rows}, found {rows}")
    for name in (
        "config.json",
        "raven_modeling_minimal.py",
        "raven_config_minimal.py",
        "tokenizer.json",
        "model.safetensors.index.json",
    ):
        if not (model / name).is_file():
            errors.append(f"model file missing: {model / name}")
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)
    if not _writable(output_root):
        errors.append(f"output root is not writable: {output_root}")
    if stage is not None:
        stage = Path(stage)
        if not (stage / "src" / "neuron_explorer" / "attribution" / "huginn").is_dir():
            errors.append(f"staged Huginn package missing: {stage}")
        if commit is not None:
            recorded = (stage / "COMMIT").read_text(encoding="utf-8").strip() if (stage / "COMMIT").is_file() else ""
            if recorded != commit:
                errors.append(f"staged commit {recorded!r} != {commit!r}")
    return errors


def check_standard_mounts() -> list[str]:
    """Verify the four DLC mounts from the launching-dlc-jobs skill."""
    errors = []
    expected = (
        ("/mnt/nas-save", True),
        ("/mnt/oss-save", True),
        ("/mnt/oss-modelscope", False),
        ("/mnt/oss-dataset", False),
    )
    for path, should_write in expected:
        directory = Path(path)
        if not directory.is_dir():
            errors.append(f"mount missing: {path}")
            continue
        writable = _writable(directory)
        failed = False
        if should_write and not writable:
            errors.append(f"mount not writable: {path}")
            failed = True
        if not should_write and writable:
            errors.append(f"read-only mount accepted a write: {path}")
            failed = True
        if failed:
            continue
        try:
            count = len(os.listdir(directory))
        except OSError as exc:
            errors.append(f"mount unreadable: {path}: {exc}")
            continue
        print(f"OK   {path}  writable={writable}  entries={count}")
    return errors


def main_check(namespace) -> int:
    errors = []
    if namespace.mounts:
        errors.extend(check_standard_mounts())
    errors.extend(
        check_paths(
            dataset=Path(namespace.dataset),
            model=Path(namespace.model),
            output_root=Path(namespace.output_root),
            stage=Path(namespace.stage) if namespace.stage else None,
            commit=namespace.commit,
            expected_tasks=namespace.expected_tasks,
            expected_rows=namespace.expected_rows,
        )
    )
    report = {"ok": not errors, "errors": errors}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("RESULT:", "PASS" if not errors else "FAIL")
    return 0 if not errors else 1
