"""Resumable Huginn attribution batch layout, planning, and writing."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuron_explorer.attribution.huginn.protocol import encode_condition
from neuron_explorer.attribution.huginn.seed import row_seed
from neuron_explorer.attribution.schema import load_graph, save_graph

CONDITIONS = (
    ("clean_correct", "clean", "correct"),
    ("clean_incorrect", "clean", "incorrect"),
    ("corrupted_correct", "corrupted", "corrupted_correct"),
    ("corrupted_incorrect", "corrupted", "corrupted_incorrect"),
)

_NEURON_FIELDS = (
    "virtual_depth",
    "stage",
    "recurrence_iteration",
    "physical_layer",
    "physical_neuron",
    "shared_weight_id",
)
_METADATA_FIELDS = (
    "seed",
    "model_path",
    "model_revision",
    "recurrence_count",
    "prompt_protocol",
    "code_commit",
    "graph_config",
)


@dataclass(frozen=True)
class GraphJob:
    """One of the four teacher-forced graphs for a dataset row."""

    row_id: str
    task_slug: str
    condition: str
    prompt_variant: str
    prompt_text: str
    completion_text: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    token_ids: list[int]
    completion_start: int
    prompt_protocol: dict[str, Any]
    seed: int


def task_slug_from_filename(filename: str) -> str:
    return Path(filename).stem


def safe_sample_id(row_id: str) -> str:
    """Make a row id a single path segment without changing its colon."""
    return row_id.replace("/", "__")


def graph_paths(
    output_root: Path, task_slug: str, row_id: str, condition: str
) -> tuple[Path, Path]:
    """Return ``output/{json,svg}/<task>/<sample>__<condition>`` paths."""
    name = f"{safe_sample_id(row_id)}__{condition}"
    root = Path(output_root) / "output"
    return (
        root / "json" / task_slug / f"{name}.json",
        root / "svg" / task_slug / f"{name}.svg",
    )


def plan_row(row: Mapping[str, Any], task_slug: str, tokenizer: Any) -> list[GraphJob]:
    """Plan four conditions that share one row seed."""
    seed = row_seed(str(row["id"]))
    variant = str(row["prompt_variant"])
    jobs: list[GraphJob] = []
    for condition, prompt_key, completion_key in CONDITIONS:
        encoded = encode_condition(
            tokenizer, str(row[prompt_key]), str(row[completion_key]), variant
        )
        jobs.append(
            GraphJob(
                row_id=str(row["id"]),
                task_slug=task_slug,
                condition=condition,
                prompt_variant=variant,
                prompt_text=str(row[prompt_key]),
                completion_text=str(row[completion_key]),
                prompt_token_ids=list(encoded["prompt_token_ids"]),
                completion_token_ids=list(encoded["completion_token_ids"]),
                token_ids=list(encoded["token_ids"]),
                completion_start=int(encoded["completion_start"]),
                prompt_protocol=dict(encoded["prompt_protocol"]),
                seed=seed,
            )
        )
    return jobs


def json_is_complete(path: Path) -> bool:
    """True when ``path`` is a validated Huginn graph with execution metadata."""
    try:
        graph = load_graph(path)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeError):
        return False
    completion = graph.get("completion_token_ids")
    if not isinstance(completion, list) or not completion:
        return False
    if "completion_logprob" not in graph:
        return False
    metadata = graph.get("metadata")
    if not isinstance(metadata, Mapping) or any(key not in metadata for key in _METADATA_FIELDS):
        return False
    for node in graph["nodes"]:
        if node.get("kind") != "neuron":
            continue
        if any(key not in node for key in _NEURON_FIELDS):
            return False
        if node["virtual_depth"] != node["layer"] or node["physical_neuron"] != node["n"]:
            return False
    return True


def iter_task_rows(dataset_dir: Path) -> list[tuple[str, list[dict[str, Any]]]]:
    """Read the 23 task jsonl files, ignoring AppleDouble sidecars."""
    tasks: list[tuple[str, list[dict[str, Any]]]] = []
    for path in sorted(Path(dataset_dir).glob("*.jsonl")):
        if path.name.startswith("._"):
            continue
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        tasks.append((path.stem, rows))
    return tasks


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _svg_ready(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 10:
        return False
    return "<svg" in path.read_text(encoding="utf-8", errors="ignore")[:200]


def _write_svg(path: Path, svg: str) -> None:
    _atomic_text(path, svg)


def _manifest_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(record["task_slug"]), str(record["row_id"]), str(record["condition"]))


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _append_manifest(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _attach_metadata(
    graph: Mapping[str, Any],
    job: GraphJob,
    *,
    code_commit: str,
    model_path: str,
    model_revision: str,
    recurrence_count: int,
    graph_config: Mapping[str, Any],
) -> dict[str, Any]:
    saved = dict(graph)
    saved["completion_token_ids"] = list(job.completion_token_ids)
    saved["prompt_token_ids"] = list(job.prompt_token_ids)
    metadata = dict(saved.get("metadata") or {})
    metadata.update(
        {
            "seed": job.seed,
            "model_path": model_path,
            "model_revision": model_revision,
            "recurrence_count": recurrence_count,
            "prompt_protocol": job.prompt_protocol,
            "code_commit": code_commit,
            "graph_config": dict(graph_config),
            "condition": job.condition,
            "row_id": job.row_id,
            "task_slug": job.task_slug,
        }
    )
    saved["metadata"] = metadata
    return saved


def run_batch(
    rows: Sequence[Mapping[str, Any]],
    task_slug: str,
    output_root: Path,
    tokenizer: Any,
    build: Callable[[GraphJob], Mapping[str, Any]],
    *,
    code_commit: str,
    model_path: str,
    model_revision: str,
    recurrence_count: int,
    graph_config: Mapping[str, Any],
    render: Callable[[Mapping[str, Any]], str] | None = None,
    order: str = "given",
) -> dict[str, Any]:
    """Build missing graphs. Valid JSON is never replaced; missing SVG is redrawn."""
    if render is None:
        from neuron_explorer.attribution.render_svg import render_svg

        render = render_svg
    jobs = [job for row in rows for job in plan_row(row, task_slug, tokenizer)]
    if order == "length":
        jobs.sort(key=lambda job: (len(job.token_ids), job.row_id, job.condition))
    root = Path(output_root)
    manifest_path = root / "output" / "manifest.jsonl"
    seen = {_manifest_key(record) for record in _load_manifest(manifest_path)}
    complete = 0
    failed: list[dict[str, str]] = []
    pending: list[Future[None]] = []

    def wait_one() -> None:
        if pending:
            pending.pop(0).result()

    def write_progress() -> None:
        path = root / "output" / "progress.json"
        existing: dict[str, Any] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except json.JSONDecodeError:
                existing = {}
        tasks = dict(existing.get("tasks") or {})
        tasks[task_slug] = {
            "rows": len(rows),
            "expected": len(jobs),
            "complete": complete,
            "failed": len(failed),
        }
        _atomic_text(path, json.dumps({"tasks": tasks}, ensure_ascii=False, indent=2) + "\n")

    with ThreadPoolExecutor(max_workers=2) as pool:
        for job in jobs:
            json_path, svg_path = graph_paths(root, job.task_slug, job.row_id, job.condition)
            key = (job.task_slug, job.row_id, job.condition)
            if json_is_complete(json_path):
                if not _svg_ready(svg_path):
                    if len(pending) >= 2:
                        wait_one()
                    graph = load_graph(json_path)
                    pending.append(pool.submit(_write_svg, svg_path, render(graph)))
                if key not in seen:
                    record = {
                        "task_slug": job.task_slug,
                        "row_id": job.row_id,
                        "condition": job.condition,
                        "json": str(json_path.relative_to(root)),
                        "svg": str(svg_path.relative_to(root)),
                        "seed": job.seed,
                        "completion_logprob": load_graph(json_path).get("completion_logprob"),
                    }
                    _append_manifest(manifest_path, record)
                    seen.add(key)
                complete += 1
                write_progress()
                continue
            try:
                built = build(job)
                graph = _attach_metadata(
                    built,
                    job,
                    code_commit=code_commit,
                    model_path=model_path,
                    model_revision=model_revision,
                    recurrence_count=recurrence_count,
                    graph_config=graph_config,
                )
                if json_is_complete(json_path):
                    complete += 1
                    write_progress()
                    continue
                save_graph(graph, json_path)
                if not json_is_complete(json_path):
                    raise RuntimeError(f"saved graph failed Huginn validation: {json_path}")
            except Exception as exc:  # noqa: BLE001 - one row must not abort the batch
                failed.append({"row_id": job.row_id, "condition": job.condition, "error": str(exc)})
                write_progress()
                continue
            if len(pending) >= 2:
                wait_one()
            saved_graph = load_graph(json_path)
            pending.append(pool.submit(_write_svg, svg_path, render(saved_graph)))
            if key not in seen:
                _append_manifest(
                    manifest_path,
                    {
                        "task_slug": job.task_slug,
                        "row_id": job.row_id,
                        "condition": job.condition,
                        "json": str(json_path.relative_to(root)),
                        "svg": str(svg_path.relative_to(root)),
                        "seed": job.seed,
                        "completion_logprob": saved_graph.get("completion_logprob"),
                    },
                )
                seen.add(key)
            complete += 1
            write_progress()
        while pending:
            wait_one()

    summary_path = root / "output" / "summary.json"
    previous: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
        except json.JSONDecodeError:
            previous = {}
    task_summaries = dict(previous.get("tasks") or {})
    task_summaries[task_slug] = {
        "rows": len(rows),
        "expected": len(jobs),
        "complete": complete,
    }
    prior_failed = [
        item
        for item in list(previous.get("failed") or [])
        if isinstance(item, dict) and item.get("task") != task_slug
    ]
    summary = {
        "rows": sum(int(item["rows"]) for item in task_summaries.values()),
        "expected": sum(int(item["expected"]) for item in task_summaries.values()),
        "complete": sum(int(item["complete"]) for item in task_summaries.values()),
        "failed": prior_failed + failed,
        "tasks": task_summaries,
    }
    _atomic_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    summary["complete"] = complete
    summary["expected"] = len(jobs)
    summary["failed"] = failed
    return summary
