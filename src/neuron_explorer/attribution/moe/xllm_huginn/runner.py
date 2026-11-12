"""Resumable xLLM Huginn MoE attribution: graph wrapper, benchmark, smoke, batch.

Graphs reuse the dense Huginn four-condition planner, output layout, and resume
writer with implementation version 3 and the xLLM runtime identity. One model
copy stays resident; the benchmark spends measured memory on direct-edge
copies only. Importing this module does not import xLLM.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from neuron_explorer.attribution.huginn.batch import (
    GraphJob,
    graph_paths,
    iter_task_rows,
    json_is_complete,
    plan_row,
    run_batch,
)
from neuron_explorer.attribution.huginn.settings import choose_edge_batch
from neuron_explorer.attribution.moe.graph import build_moe_graph
from neuron_explorer.attribution.moe.types import MoEModelIdentity, RoutedNeuron
from neuron_explorer.attribution.moe.xllm_huginn.linearized import XLLMHuginnLinearized
from neuron_explorer.attribution.moe.xllm_huginn.runtime import (
    XLLM_SOURCE_COMMIT,
    load_xllm_huginn,
)
from neuron_explorer.attribution.schema import load_graph, save_graph

ATTRIBUTION_IMPLEMENTATION_VERSION = 3
DEFAULT_K = 512
DEFAULT_MAX_EDGES = 3000
NODE_THRESHOLD = 0.8
EDGE_THRESHOLD = 0.98
MAX_EDGE_COPIES = 4
MIN_K = 64
MIN_MAX_EDGES = 256
SMOKE_K = 8
SMOKE_MAX_EDGES = 40
NATIVE_ATTENTION_RELATIVE_TOLERANCE = 2.0**-7
PATCHED_RELATIVE_TOLERANCE = 2.0**-9
ATTENTION = "explicit_causal_fp32_frozen_pattern"
ROUTING_METADATA = {
    "frozen": True,
    "score": "sigmoid",
    "normalization": "selected_sum",
    "top_k": 2,
    "experts": 25,
    "expert_width": 2432,
}


def build_xllm_huginn_graph(
    backend: XLLMHuginnLinearized,
    token_ids: Sequence[int],
    completion_start: int,
    *,
    k: int = DEFAULT_K,
    batch: int = 1,
    node_threshold: float = NODE_THRESHOLD,
    edge_threshold: float = EDGE_THRESHOLD,
    max_edges: int | None = DEFAULT_MAX_EDGES,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one frozen-route graph and restore the native model afterwards."""
    try:
        with backend:
            return build_moe_graph(
                backend,
                token_ids,
                completion_start,
                k=k,
                batch=batch,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
                max_edges=max_edges,
                metadata=metadata,
            )
    finally:
        backend.release()


class PromptTokenizer:
    """Planner view of a native tokenizer.

    Exposes ``bos_token_id`` (from ``bos_id`` when needed) and an ``encode``
    that never lets the native encoder add BOS/EOS; the prompt protocol alone
    decides whether one BOS is prepended.
    """

    def __init__(self, native: Any) -> None:
        bos = getattr(native, "bos_token_id", None)
        if bos is None:
            bos = getattr(native, "bos_id", None)
        if callable(bos):
            bos = bos()
        if bos is None:
            raise ValueError("native tokenizer exposes neither bos_token_id nor bos_id")
        self.native = native
        self.bos_token_id = int(bos)
        self._options = _no_special_token_options(native.encode)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise ValueError("PromptTokenizer never adds special tokens")
        return [int(token) for token in self.native.encode(text, **self._options)]


def _no_special_token_options(encode: Any) -> dict[str, bool]:
    try:
        parameters = inspect.signature(encode).parameters
    except (TypeError, ValueError):
        return {}
    if "add_special_tokens" in parameters:
        return {"add_special_tokens": False}
    return {
        name: False for name in ("bos", "eos", "add_bos", "add_eos") if name in parameters
    }


@dataclass(frozen=True)
class _Run:
    model_path: str
    model_revision: str
    code_commit: str
    recurrence_count: int
    extra_metadata: Mapping[str, Any]


def _routing_contract(identity: MoEModelIdentity) -> dict[str, Any]:
    normalization = identity.router_normalization
    if normalization == "selected_renormalized":
        normalization = "selected_sum"
    observed = {
        "frozen": True,
        "score": identity.router_activation,
        "normalization": normalization,
        "top_k": identity.top_k,
        "experts": identity.num_experts,
        "expert_width": identity.expert_width,
    }
    if observed != ROUTING_METADATA:
        raise RuntimeError(
            f"xLLM routing {observed} does not match the pinned contract {ROUTING_METADATA}"
        )
    return dict(ROUTING_METADATA)


def _graph_config(
    k: int, max_edges: int, batch: int, fidelity_note: str | None
) -> dict[str, Any]:
    """Full graph configuration and resume identity, including edge copies."""
    config: dict[str, Any] = {
        "k": int(k),
        "max_edges": int(max_edges),
        "batch": int(batch),
        "node_threshold": NODE_THRESHOLD,
        "edge_threshold": EDGE_THRESHOLD,
        "target": "teacher_forced_completion_logprob",
        "edge_semantics": "local_linear_half_rule_direct",
        "target_transform": "differentiated_log_softmax",
        "output_aggregation": "sum_over_completion_tokens",
        "attention": ATTENTION,
        "fidelity_reduced": fidelity_note is not None,
    }
    if fidelity_note:
        config["fidelity_note"] = fidelity_note
    return config


def _job_metadata(job: GraphJob, run: _Run, graph_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "attribution_implementation_version": ATTRIBUTION_IMPLEMENTATION_VERSION,
        "seed": job.seed,
        "model_path": run.model_path,
        "model_revision": run.model_revision,
        "recurrence_count": run.recurrence_count,
        "prompt_protocol": job.prompt_protocol,
        "code_commit": run.code_commit,
        "graph_config": dict(graph_config),
        "condition": job.condition,
        "row_id": job.row_id,
        "task_slug": job.task_slug,
        **run.extra_metadata,
    }


def _build_job(
    backend: XLLMHuginnLinearized,
    job: GraphJob,
    run: _Run,
    graph_config: Mapping[str, Any],
) -> dict[str, Any]:
    torch.manual_seed(job.seed)
    return build_xllm_huginn_graph(
        backend,
        job.token_ids,
        job.completion_start,
        k=int(graph_config["k"]),
        batch=int(graph_config["batch"]),
        node_threshold=float(graph_config["node_threshold"]),
        edge_threshold=float(graph_config["edge_threshold"]),
        max_edges=int(graph_config["max_edges"]),
        metadata=_job_metadata(job, run, graph_config),
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = message.rstrip() + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def _cuda(device: str) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()


def _device_memory(device: str) -> int:
    if _cuda(device):
        return int(torch.cuda.get_device_properties(torch.device(device)).total_memory)
    return 0


def _peak_memory(device: str) -> int:
    if _cuda(device):
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())
    return 0


def _reset_memory(device: str) -> None:
    if _cuda(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _median_job(jobs: Sequence[GraphJob]) -> GraphJob:
    ordered = sorted(jobs, key=lambda job: (len(job.token_ids), job.row_id, job.condition))
    return ordered[len(ordered) // 2]


def _benchmark(
    backend: XLLMHuginnLinearized,
    planned: Sequence[GraphJob],
    run: _Run,
    args,
    output_root: Path,
    log_path: Path,
) -> tuple[int, int, int, str | None]:
    """Measure one graph at one copy, then find the largest safe copy count.

    OOM halves direct-edge copies first. Only an OOM at one copy halves ``k``
    and ``max_edges``, and every such reduction is recorded.
    """
    sample = _median_job(planned)
    requested_k, requested_edges = int(args.k), int(args.max_edges)
    k, max_edges, batch = requested_k, requested_edges, 1
    copy_reductions: list[dict[str, int]] = []
    fidelity_reductions: list[dict[str, int]] = []
    fidelity_note: str | None = None
    measured: dict[str, Any] | None = None
    _log(
        log_path,
        f"benchmark {sample.task_slug} {sample.row_id} {sample.condition} "
        f"tokens={len(sample.token_ids)} k={k} max_edges={max_edges}",
    )
    while True:
        _reset_memory(args.device)
        started = time.perf_counter()
        try:
            graph = _build_job(
                backend, sample, run, _graph_config(k, max_edges, batch, fidelity_note)
            )
        except Exception as exc:
            if not _is_oom(exc):
                raise
            _reset_memory(args.device)
            if batch > 1:
                copy_reductions.append({"from": batch, "to": batch // 2})
                batch //= 2
                _log(log_path, f"benchmark OOM reduced edge copies to {batch}")
                continue
            if k <= MIN_K:
                raise
            reduced_k = max(MIN_K, k // 2)
            reduced_edges = max(min(max_edges, MIN_MAX_EDGES), max_edges // 2)
            fidelity_reductions.append(
                {
                    "from_k": k,
                    "to_k": reduced_k,
                    "from_max_edges": max_edges,
                    "to_max_edges": reduced_edges,
                }
            )
            k, max_edges, measured = reduced_k, reduced_edges, None
            fidelity_note = f"benchmark OOM reduced fidelity to k={k} max_edges={max_edges}"
            _log(log_path, fidelity_note)
            continue
        elapsed = time.perf_counter() - started
        if measured is None:
            peak, total = _peak_memory(args.device), _device_memory(args.device)
            settings = (
                choose_edge_batch(peak, total)
                if total
                else {"edge_batch": 1, "graph_concurrency": 1, "model_copies": 1}
            )
            measured = {"peak": peak, "total": total, "settings": settings}
            candidate = min(
                MAX_EDGE_COPIES, settings["edge_batch"] * settings["graph_concurrency"]
            )
            if candidate > 1:
                batch = candidate
                continue
        break

    remaining = sum(
        1
        for job in planned
        if not json_is_complete(
            graph_paths(output_root, job.task_slug, job.row_id, job.condition)[0]
        )
    )
    benchmark = {
        "task_slug": sample.task_slug,
        "row_id": sample.row_id,
        "condition": sample.condition,
        "tokens": len(sample.token_ids),
        "seconds_per_graph": round(elapsed, 4),
        "peak_bytes": measured["peak"],
        "device_bytes": measured["total"],
        "requested_k": requested_k,
        "requested_max_edges": requested_edges,
        "k": k,
        "max_edges": max_edges,
        "measured_edge_batch": 1,
        "chosen": measured["settings"],
        "applied_edge_batch": batch,
        "applied_graph_concurrency": 1,
        "model_copies": 1,
        "max_edge_copies": MAX_EDGE_COPIES,
        "copy_reductions": copy_reductions,
        "fidelity_reductions": fidelity_reductions,
        "fidelity_note": fidelity_note,
        "frozen_routes": True,
        "routing": dict(ROUTING_METADATA),
        "attention": ATTENTION,
        "io_overlap": True,
        "completion_logprob": graph.get("completion_logprob"),
        "remaining_graphs": remaining,
        "estimated_seconds": round(elapsed * max(remaining, 1), 1),
        "estimated_hours": round(elapsed * max(remaining, 1) / 3600, 2),
        "note": (
            "Direct-edge copies use the measured memory slots, capped at 4, and "
            "are reduced before k on OOM. Graphs run one at a time on the single "
            "loaded model; JSON validation and SVG rendering overlap on CPU."
        ),
    }
    _atomic_json(output_root / "output" / "benchmark.json", benchmark)
    _update_launch_status(output_root, benchmark)
    _log(
        log_path,
        f"benchmark {elapsed:.2f}s peak={measured['peak']} edge_batch={batch} "
        f"k={k} max_edges={max_edges} estimate_hours={benchmark['estimated_hours']}",
    )
    return k, max_edges, batch, fidelity_note


def _update_launch_status(output_root: Path, benchmark: Mapping[str, Any]) -> None:
    path = Path(output_root) / "launch-status.json"
    if not path.is_file():
        return
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    status["benchmark"] = dict(benchmark)
    _atomic_json(path, status)


def _gate(name: str, ok: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), **details}


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def _native_logits(model: Any, ids: Sequence[int]) -> torch.Tensor:
    device = next(model.parameters()).device
    tokens = torch.tensor([list(ids)], device=device)
    with torch.no_grad():
        output = model(tokens, multi_segments=False)
    return (output[0] if isinstance(output, tuple) else output)[0].detach()


def _logit_gates(
    native: torch.Tensor, explicit: torch.Tensor, patched: torch.Tensor
) -> list[dict[str, Any]]:
    """Gate native attention and the patched forward against the explicit control.

    Both bounds are fixed fractions of the control's logit scale, so a large
    native kernel gap can never loosen the patched-forward bound.
    """
    differences = {
        "patched_vs_explicit_max_abs": _max_abs(patched, explicit),
        "native_vs_explicit_max_abs": _max_abs(native, explicit),
        "patched_vs_native_max_abs": _max_abs(patched, native),
    }
    scale = float(explicit.float().abs().max())
    native_tolerance = NATIVE_ATTENTION_RELATIVE_TOLERANCE * scale
    patched_tolerance = PATCHED_RELATIVE_TOLERANCE * scale

    def within(value: float, tolerance: float) -> bool:
        return math.isfinite(value) and value <= tolerance

    return [
        _gate(
            "native_attention_control",
            within(differences["native_vs_explicit_max_abs"], native_tolerance),
            tolerance=native_tolerance,
            tolerance_rule="2**-7 * max|explicit logits|",
            dtype=str(explicit.dtype),
            **differences,
        ),
        _gate(
            "logit_equivalence",
            within(differences["patched_vs_explicit_max_abs"], patched_tolerance),
            tolerance=patched_tolerance,
            tolerance_rule="2**-9 * max|explicit logits|",
            dtype=str(explicit.dtype),
            **differences,
        ),
    ]


def _leading_bos_gate(job: GraphJob, bos: int) -> dict[str, Any]:
    ids = list(job.token_ids)
    leading = 0
    while leading < len(ids) and ids[leading] == bos:
        leading += 1
    expected = 1 if job.prompt_protocol.get("native_bos") else 0
    inner = ids[leading:].count(bos)
    return _gate(
        "single_leading_bos",
        leading == expected and inner == 0,
        leading_bos=leading,
        expected_leading_bos=expected,
        inner_bos=inner,
        bos_token_id=bos,
    )


def _tokenizer_length(tokenizer: Any) -> int | None:
    for candidate in (tokenizer, getattr(tokenizer, "_tokenizer", None)):
        try:
            return len(candidate)  # type: ignore[arg-type]
        except TypeError:
            continue
    return None


def _edge_causality(graph: Mapping[str, Any]) -> list[list[Any]]:
    nodes = graph["nodes"]
    last_prediction = len(graph["prompt_token_ids"]) + len(graph["completion_token_ids"]) - 2
    violations = []
    for source_index, target_index, weight in graph["edges"]:
        source, target = nodes[source_index], nodes[target_index]
        if source["kind"] == "logit" or target["kind"] == "embed":
            violations.append([source_index, target_index, weight])
        elif target["kind"] == "logit":
            if source["pos"] > last_prediction:
                violations.append([source_index, target_index, weight])
        elif source["pos"] > target["pos"] or (
            source["kind"] == "neuron" and source["virtual_depth"] >= target["virtual_depth"]
        ):
            violations.append([source_index, target_index, weight])
    return violations


def _smoke(
    backend: XLLMHuginnLinearized,
    model: Any,
    tokenizer: PromptTokenizer,
    planned: Sequence[GraphJob],
    run: _Run,
    args,
    output_root: Path,
    log_path: Path,
) -> int:
    """Gate one short real condition before any batch is allowed."""
    job = min(planned, key=lambda item: (len(item.token_ids), item.row_id, item.condition))
    ids = job.token_ids
    schedule = backend.schedule
    identity = backend.identity
    gates: list[dict[str, Any]] = []
    gates.append(
        _gate(
            "runtime_identity",
            run.model_revision.startswith("checkpoint-sha256:")
            and run.extra_metadata["xllm_source_commit"] == XLLM_SOURCE_COMMIT,
            checkpoint_fingerprint=run.model_revision,
            xllm_source_commit=run.extra_metadata["xllm_source_commit"],
        )
    )

    native = _native_logits(model, ids)
    explicit = backend.explicit_attention_logits(ids)[0].detach()

    injections = 0

    def count_injection(*_args) -> None:
        nonlocal injections
        injections += 1

    hook = model.input_injection.register_forward_hook(count_injection)
    try:
        with backend:
            patched = backend.forward(ids, mode="total")[0].detach()
            executions = len(backend._records)
            snapshot = backend.routing_snapshot
    finally:
        hook.remove()
        backend.release()
    recurrence = run.recurrence_count
    gates.append(
        _gate(
            "moe_executions",
            executions == len(schedule),
            observed=executions,
            expected=len(schedule),
        )
    )
    gates.append(
        _gate(
            "recurrent_injections",
            injections == recurrence,
            observed=injections,
            expected=recurrence,
        )
    )

    experts = snapshot.selected_experts
    weight_sums = snapshot.routing_weights.float().sum(-1)
    distinct = bool((experts[..., :1] != experts[..., 1:]).all()) if experts.shape[-1] > 1 else True
    gates.append(
        _gate(
            "routes",
            bool((snapshot.selected_expert_counts == identity.top_k).all())
            and bool(((experts >= 0) & (experts < identity.num_experts)).all())
            and distinct
            and float((weight_sums - 1).abs().max()) <= 1e-2,
            shape=list(experts.shape),
            max_weight_sum_error=float((weight_sums - 1).abs().max()),
        )
    )

    recurrent: dict[int, list[Any]] = {}
    for slot in schedule:
        if slot.stage == "recurrent":
            recurrent.setdefault(slot.physical_layer, []).append(slot)
    shared_ok = bool(recurrent)
    for slots in recurrent.values():
        identities = {
            RoutedNeuron(slot, 0, 0, 0, identity.expert_width, 0, 1.0).shared_weight_id
            for slot in slots
        }
        iterations = [slot.recurrence_iteration for slot in slots]
        shared_ok &= len(identities) == 1 and iterations == list(range(recurrence))
    gates.append(_gate("recurrent_shared_ids", shared_ok, recurrent_layers=len(recurrent)))

    logit_gates = _logit_gates(native, explicit, patched)
    gates.extend(logit_gates)
    gates.append(_leading_bos_gate(job, int(tokenizer.bos_token_id)))

    graph_config = _graph_config(SMOKE_K, SMOKE_MAX_EDGES, 1, None)
    _reset_memory(args.device)
    started = time.perf_counter()
    graph = _build_job(backend, job, run, graph_config)
    elapsed = time.perf_counter() - started
    peak = _peak_memory(args.device)
    graph["completion_token_ids"] = list(job.completion_token_ids)
    graph["prompt_token_ids"] = list(job.prompt_token_ids)
    graph["metadata"].update(_job_metadata(job, run, graph_config))
    json_path, svg_path = graph_paths(
        output_root / "smoke", job.task_slug, job.row_id, job.condition
    )
    save_graph(graph, json_path)
    from neuron_explorer.attribution.render_svg import render_svg

    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(render_svg(load_graph(json_path)), encoding="utf-8")

    logit = next(node for node in graph["nodes"] if node["kind"] == "logit")
    completion = list(ids[job.completion_start :])
    positions = list(range(job.completion_start - 1, len(ids) - 1))
    gates.append(
        _gate(
            "completion_coverage",
            logit["target_token_ids"] == completion
            and logit["prediction_positions"] == positions
            and math.isfinite(float(graph["completion_logprob"])),
            tokens=len(completion),
            completion_logprob=graph["completion_logprob"],
        )
    )
    expected = {
        "prompt_token_ids": list(job.prompt_token_ids),
        "completion_token_ids": completion,
        "metadata": _job_metadata(job, run, graph_config),
    }
    gates.append(_gate("graph_version_3", json_is_complete(json_path, expected)))
    violations = _edge_causality(graph)
    gates.append(
        _gate(
            "edge_causality",
            not violations,
            edges=len(graph["edges"]),
            violations=violations[:20],
        )
    )

    report = {
        "ok": all(gate["ok"] for gate in gates),
        "task_slug": job.task_slug,
        "row_id": job.row_id,
        "condition": job.condition,
        "tokens": len(ids),
        "checkpoint_fingerprint": run.model_revision,
        "xllm_source_commit": run.extra_metadata["xllm_source_commit"],
        "code_commit": run.code_commit,
        "tokenizer_length": _tokenizer_length(tokenizer.native),
        "bos_id": tokenizer.bos_token_id,
        "model_identity": identity.as_dict(),
        "graph_config": graph_config,
        "logit_differences": {
            key: value
            for key, value in logit_gates[0].items()
            if key.endswith("_max_abs")
        },
        "seconds_per_graph": round(elapsed, 4),
        "peak_bytes": peak,
        "device_bytes": _device_memory(args.device),
        "graph_json": str(json_path),
        "graph_svg": str(svg_path),
        "gates": gates,
    }
    _atomic_json(output_root / "smoke.json", report)
    for gate in gates:
        _log(log_path, f"smoke gate {gate['name']} ok={gate['ok']}")
    return 0 if report["ok"] else 1


def execute(args) -> int:
    """Load the pinned xLLM model, optionally benchmark or smoke, then resume all graphs."""
    output_root = Path(args.output_root)
    log_path = output_root / "logs" / "batch.log"
    tasks = iter_task_rows(Path(args.dataset))
    if args.limit:
        tasks = [(slug, rows[: args.limit]) for slug, rows in tasks]
    _log(log_path, f"loading {args.model} with xLLM {args.xllm_source} on {args.device}")
    native_tokenizer, model, config, fingerprint, source_commit = load_xllm_huginn(
        args.model, args.xllm_source, args.device
    )
    tokenizer = PromptTokenizer(native_tokenizer)
    if source_commit != XLLM_SOURCE_COMMIT:
        raise RuntimeError(f"xLLM source {source_commit!r} != {XLLM_SOURCE_COMMIT!r}")
    model.eval()
    backend = XLLMHuginnLinearized(model, config)
    routing = _routing_contract(backend.identity)
    schedule_metadata = [slot.as_dict() for slot in backend.schedule]
    run = _Run(
        model_path=str(args.model),
        model_revision=fingerprint,
        code_commit=args.commit,
        recurrence_count=len(
            {slot.recurrence_iteration for slot in backend.schedule if slot.stage == "recurrent"}
        ),
        extra_metadata={
            "runtime": "xllm",
            "xllm_source_commit": XLLM_SOURCE_COMMIT,
            "schedule": schedule_metadata,
            "routing": routing,
        },
    )
    planned = [
        job for slug, rows in tasks for row in rows for job in plan_row(row, slug, tokenizer)
    ]
    if args.smoke:
        if not planned:
            raise RuntimeError("smoke requires at least one dataset row")
        return _smoke(backend, model, tokenizer, planned, run, args, output_root, log_path)

    k, max_edges, edge_batch, fidelity_note = int(args.k), int(args.max_edges), 1, None
    if args.benchmark_first and planned:
        k, max_edges, edge_batch, fidelity_note = _benchmark(
            backend, planned, run, args, output_root, log_path
        )
    graph_config = _graph_config(k, max_edges, edge_batch, fidelity_note)
    failures = 0
    for slug, rows in tasks:
        _log(log_path, f"task {slug} rows={len(rows)} edge_batch={edge_batch}")
        try:
            summary = run_batch(
                rows,
                slug,
                output_root,
                tokenizer,
                lambda job: _build_job(backend, job, run, graph_config),
                code_commit=run.code_commit,
                model_path=run.model_path,
                model_revision=run.model_revision,
                recurrence_count=run.recurrence_count,
                graph_config=graph_config,
                order="length",
                attribution_implementation_version=ATTRIBUTION_IMPLEMENTATION_VERSION,
                extra_metadata=run.extra_metadata,
            )
        except Exception:  # noqa: BLE001 - one task must not abort the others
            failures += 1
            _log(log_path, traceback.format_exc())
            continue
        failures += len(summary["failed"])
        for failure in summary["failed"]:
            _log(
                log_path,
                f"failed {slug} {failure['row_id']} {failure['condition']}: "
                f"{failure['error']}",
            )
        _log(
            log_path,
            f"task {slug} complete={summary['complete']} failed={len(summary['failed'])}",
        )
    return 1 if failures else 0
