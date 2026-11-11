"""GPU batch entry point: benchmark one graph, then resume the full set."""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from neuron_explorer.attribution.huginn.batch import (
    ATTRIBUTION_IMPLEMENTATION_VERSION,
    graph_paths,
    iter_task_rows,
    json_is_complete,
    run_batch,
)
from neuron_explorer.attribution.huginn.graph import build_huginn_graph
from neuron_explorer.attribution.huginn.model import load_huginn, model_revision
from neuron_explorer.attribution.huginn.settings import choose_edge_batch

DEFAULT_K = 512
DEFAULT_MAX_EDGES = 3000
DEFAULT_NODE_THRESHOLD = 0.8
DEFAULT_EDGE_THRESHOLD = 0.98


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = message.rstrip() + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def _graph_config(args, *, edge_batch: int, fidelity_note: str | None) -> dict[str, Any]:
    config = {
        "k": int(args.k),
        "max_edges": int(args.max_edges),
        "batch": int(edge_batch),
        "node_threshold": DEFAULT_NODE_THRESHOLD,
        "edge_threshold": DEFAULT_EDGE_THRESHOLD,
        "target": "teacher_forced_completion_logprob",
        "edge_semantics": "local_linear_half_rule_direct",
        "target_transform": "differentiated_log_softmax",
        "output_aggregation": "sum_over_completion_tokens",
        "fidelity_reduced": fidelity_note is not None,
    }
    if fidelity_note:
        config["fidelity_note"] = fidelity_note
    return config


def _median_job(jobs):
    ordered = sorted(jobs, key=lambda job: (len(job.token_ids), job.row_id, job.condition))
    return ordered[len(ordered) // 2]


def _device_memory(device: str) -> int:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        return int(torch.cuda.get_device_properties(0).total_memory)
    return 0


def _peak_memory(device: str) -> int:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())
    return 0


def execute(args) -> int:
    """Load Huginn, benchmark one real row, then attribute every remaining graph."""
    import torch

    dataset = Path(args.dataset)
    output_root = Path(args.output_root)
    model_path = Path(args.model)
    log_path = output_root / "logs" / "batch.log"
    tasks = iter_task_rows(dataset)
    if args.limit:
        tasks = [(slug, rows[: args.limit]) for slug, rows in tasks]
    revision = model_revision(model_path)
    _log(log_path, f"loading {model_path} on {args.device}")
    tokenizer, model = load_huginn(model_path, args.device)
    recurrence = int(model.config.mean_recurrence)
    fidelity_note = None
    edge_batch = 1
    settings = {"edge_batch": 1, "graph_concurrency": 1, "model_copies": 1}
    benchmark = None

    from neuron_explorer.attribution.huginn.batch import plan_row

    planned = [
        (slug, job)
        for slug, rows in tasks
        for row in rows
        for job in plan_row(row, slug, tokenizer)
    ]
    if args.benchmark_first and planned:
        sample = _median_job([job for _slug, job in planned])
        _log(
            log_path,
            f"benchmark {sample.task_slug} {sample.row_id} {sample.condition} "
            f"tokens={len(sample.token_ids)} k={args.k} max_edges={args.max_edges}",
        )
        while True:
            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                graph = build_huginn_graph(
                    model,
                    sample.token_ids,
                    sample.completion_start,
                    k=int(args.k),
                    batch=1,
                    max_edges=int(args.max_edges),
                    metadata=_sample_metadata(sample, args, revision, recurrence, edge_batch=1, fidelity_note=fidelity_note),
                )
            except Exception as exc:
                oom = isinstance(exc, getattr(torch, "OutOfMemoryError", ())) or (
                    "out of memory" in str(exc).lower()
                )
                if not oom:
                    raise
                if args.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if int(args.k) <= 64:
                    raise
                args.k = max(64, int(args.k) // 2)
                args.max_edges = max(256, int(args.max_edges) // 2)
                fidelity_note = (
                    f"benchmark OOM reduced fidelity to k={args.k} max_edges={args.max_edges}"
                )
                _log(log_path, fidelity_note)
                continue
            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak = _peak_memory(args.device)
            total = _device_memory(args.device)
            settings = choose_edge_batch(peak, total) if total else settings
            # One resident model. Spend the measured slots on direct-edge copies.
            edge_batch = min(4, settings["edge_batch"] * settings["graph_concurrency"])
            benchmark = {
                "task_slug": sample.task_slug,
                "row_id": sample.row_id,
                "condition": sample.condition,
                "tokens": len(sample.token_ids),
                "seconds_per_graph": round(elapsed, 4),
                "peak_bytes": peak,
                "device_bytes": total,
                "k": int(args.k),
                "max_edges": int(args.max_edges),
                "measured_edge_batch": 1,
                "chosen": settings,
                "applied_edge_batch": edge_batch,
                "applied_graph_concurrency": 1,
                "model_copies": 1,
                "io_overlap": True,
                "fidelity_note": fidelity_note,
                "completion_logprob": graph.get("completion_logprob"),
            }
            remaining = sum(
                1
                for slug, job in planned
                if not json_is_complete(graph_paths(output_root, slug, job.row_id, job.condition)[0])
            )
            benchmark["remaining_graphs_before_save"] = remaining
            benchmark["estimated_seconds"] = round(elapsed * max(remaining, 1), 1)
            benchmark["estimated_hours"] = round(benchmark["estimated_seconds"] / 3600, 2)
            benchmark["note"] = (
                "Direct-edge copies use the full memory-slot budget, capped at 4. "
                "Graphs run one at a time on the single loaded model; JSON validation "
                "and SVG rendering overlap on CPU. A second model copy is not started."
            )
            _atomic_json(output_root / "output" / "benchmark.json", benchmark)
            _update_launch_status(output_root, benchmark)
            _log(
                log_path,
                f"benchmark {elapsed:.2f}s peak={peak} edge_batch={edge_batch} "
                f"estimate_hours={benchmark['estimated_hours']}",
            )
            json_path, _svg = graph_paths(output_root, sample.task_slug, sample.row_id, sample.condition)
            from neuron_explorer.attribution.schema import save_graph

            if not json_is_complete(json_path):
                save_graph(graph, json_path)
            break

    config = _graph_config(args, edge_batch=edge_batch, fidelity_note=fidelity_note)
    failures = []
    for slug, rows in tasks:
        _log(log_path, f"task {slug} rows={len(rows)}")
        try:
            summary = run_batch(
                rows,
                slug,
                output_root,
                tokenizer,
                lambda job, cfg=config: _build_job(model, job, cfg),
                code_commit=args.commit,
                model_path=str(model_path),
                model_revision=revision,
                recurrence_count=recurrence,
                graph_config=config,
                order="length",
            )
        except Exception:
            failures.append(traceback.format_exc())
            _log(log_path, failures[-1])
            continue
        _log(log_path, f"task {slug} complete={summary['complete']} failed={len(summary['failed'])}")
    if failures:
        return 1
    return 0


def _sample_metadata(sample, args, revision, recurrence, *, edge_batch, fidelity_note):
    return {
        "attribution_implementation_version": ATTRIBUTION_IMPLEMENTATION_VERSION,
        "seed": sample.seed,
        "model_path": str(args.model),
        "model_revision": revision,
        "recurrence_count": recurrence,
        "prompt_protocol": sample.prompt_protocol,
        "code_commit": args.commit,
        "graph_config": _graph_config(args, edge_batch=edge_batch, fidelity_note=fidelity_note),
        "condition": sample.condition,
        "row_id": sample.row_id,
        "task_slug": sample.task_slug,
    }


def _build_job(model, job, graph_config: dict) -> dict:
    import torch

    torch.manual_seed(job.seed)
    return build_huginn_graph(
        model,
        job.token_ids,
        job.completion_start,
        k=int(graph_config["k"]),
        batch=int(graph_config["batch"]),
        node_threshold=float(graph_config["node_threshold"]),
        edge_threshold=float(graph_config["edge_threshold"]),
        max_edges=graph_config.get("max_edges"),
        metadata={
            "attribution_implementation_version": ATTRIBUTION_IMPLEMENTATION_VERSION,
            "seed": job.seed,
            "graph_config": graph_config,
            "condition": job.condition,
            "row_id": job.row_id,
            "task_slug": job.task_slug,
            "prompt_protocol": job.prompt_protocol,
        },
    )


def _update_launch_status(output_root: Path, benchmark: dict) -> None:
    path = Path(output_root) / "launch-status.json"
    if not path.is_file():
        return
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    status["benchmark"] = benchmark
    _atomic_json(path, status)
