"""Recurrence-aware steering sweep for Ouro, Nanbeige, and Huginn.

One unsteered baseline per prompt, then ``main_rel``: {who, snake, orange} x
{-0.7, +0.5, +0.8} and {free} x {+3}. A real neuron's activation add at a
selected firing is ``dose * residual_p50 / column_norm``. Random controls
borrow that column's norm on a deterministic Gaussian direction and add the
scaled vector at the projection output. Records keep the fields ``rank.py``
reads, plus schema 2 site metadata. Rerunning the same command skips jsonl
rows already written.

This runner does not capture activations, call vLLM, or apply the absmax /
calib dose designs.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sweep.looped_models import BatchEdits, ExecutionSite, adapt
from sweep.looped_neuron_stats import load_compact_arrays

PROMPTS = {
    "who": "Who are you?",
    "snake": "Write a snake game in Python.",
    "orange": "Describe the color orange.",
    "free": "Write whatever you like.",
}
MAIN_REL = [(prompt, dose) for prompt in ("who", "snake", "orange") for dose in (-0.7, 0.5, 0.8)] + [("free", 3.0)]


@dataclass(frozen=True, slots=True)
class CompactStats:
    """Site-aligned residual scales from a looped_neuron_stats npz."""

    sites: tuple[ExecutionSite, ...]
    resid_p50: np.ndarray
    abs_max: np.ndarray
    config: dict

    def p50(self, site: ExecutionSite) -> float:
        try:
            index = self.sites.index(site)
        except ValueError as exc:
            raise KeyError(site) from exc
        return float(self.resid_p50[index])


def load_compact_stats(path: str) -> CompactStats:
    """Read regime quantiles and site metadata. Does not read a checkpoint."""
    data = load_compact_arrays(path)
    meta = json.loads(data["sites_json"].item())
    config = json.loads(data["config_json"].item())
    sites = tuple(
        ExecutionSite(
            str(item["kind"]),
            int(item["physical_index"]),
            int(item["recurrence"]),
            int(item["virtual_index"]),
        )
        for item in meta
    )
    resid = np.array(data["resid_norm_q10_50_90"], dtype=np.float64, copy=True)
    abs_max = np.array(data["abs_max"], dtype=np.float64, copy=True)
    return CompactStats(sites=sites, resid_p50=resid[:, 1], abs_max=abs_max, config=config)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse a steering invocation. ``--site`` or ``--all-recurrences`` is required."""
    parser = argparse.ArgumentParser(description="Recurrence-aware main_rel steering sweep")
    parser.add_argument("--model", required=True)
    parser.add_argument("--family", required=True, choices=("ouro", "huginn", "nanbeige"))
    parser.add_argument("--stats", required=True)
    parser.add_argument("--site", type=int, help="virtual index of the firing to steer")
    parser.add_argument("--all-recurrences", help="kind:index, every firing of that physical block")
    parser.add_argument("--smoke-neurons", required=True, help='"all", "0,1", "0-3", or a newline-separated file')
    parser.add_argument("--run", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--random-controls", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recurrences", type=int)
    args = parser.parse_args(argv)
    if args.site is None and not args.all_recurrences:
        parser.error("provide --site or --all-recurrences")
    return args


def record_key(row: dict) -> tuple:
    """Identity of one jsonl row. Resume skips rows whose key is already present."""
    virtual = row.get("virtual_indices") or ()
    return (
        row["kind"],
        int(row["neuron"]),
        row["prompt"],
        float(row["dose"]),
        row.get("selector"),
        tuple(int(index) for index in virtual),
    )


def edits_for_rows(rows, sites, stats: CompactStats, column, matched, hidden: int) -> dict[ExecutionSite, BatchEdits]:
    """Row-aligned activation deltas and residual vectors for one shared prompt."""
    column = torch.as_tensor(column, dtype=torch.float32).flatten()
    matched = torch.as_tensor(matched, dtype=torch.float32)
    if int(column.shape[0]) != int(hidden) or matched.ndim != 2 or int(matched.shape[-1]) != int(hidden):
        raise ValueError(f"column and matched directions must have hidden size {hidden}")
    batch = len(rows)
    edits = {}
    for site in sites:
        neurons = torch.full((batch,), -1, dtype=torch.long)
        deltas = torch.zeros(batch, dtype=torch.float32)
        residuals = torch.zeros(batch, int(hidden), dtype=torch.float32)
        used_residual = False
        p50 = float(stats.p50(site))
        for index, row in enumerate(rows):
            if row["kind"] == "baseline":
                continue
            if row["kind"] == "random":
                ref = int(row["ref_neuron"])
                scale = _push(row["dose"], p50, float(column[ref]))
                residuals[index] = scale * matched[int(row["neuron"])]
                used_residual = True
                continue
            if row["kind"] != "neuron":
                raise ValueError(f"unknown rollout kind {row['kind']!r}")
            neuron = int(row["neuron"])
            neurons[index] = neuron
            deltas[index] = _push(row["dose"], p50, float(column[neuron]))
        edits[site] = BatchEdits(neurons, deltas, residuals if used_residual else None)
    return edits


def run(args, model, tokenizer) -> str:
    """Steer ``model`` and append new rows to ``<out-dir>/<run>.jsonl``."""
    model.eval()
    family = args.family.lower()
    if family == "huginn":
        if args.recurrences is None:
            raise ValueError("Huginn recurrences must be set")
        adapter = adapt(model, family=family, recurrences=args.recurrences, seed=args.seed, test_time_noise=0.0)
    else:
        if args.recurrences is not None:
            raise ValueError("recurrences is a Huginn control")
        adapter = adapt(model, family=family)
    stats = load_compact_stats(args.stats)
    _check_stats(stats, family, args.recurrences)
    sites, selector = _resolve_sites(adapter, args)
    column = torch.linalg.vector_norm(adapter.write_matrix(sites[0]).detach().float().cpu(), dim=0)
    hidden = int(column.shape[0])
    neurons = _resolve_neurons(args.smoke_neurons, hidden)
    matched, refs = _random_directions(args.random_controls, args.seed, column)
    meta = _selector_meta(sites, selector)
    jobs = _jobs(neurons, args.random_controls, refs, meta)
    path = Path(args.out_dir) / f"{args.run}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(path)
    pending = [job for job in jobs if record_key(job) not in done]
    if not pending:
        return str(path)

    prompt_ids = _encode(tokenizer)
    device = _device(model)
    eos = getattr(tokenizer, "eos_token_id", None)
    with path.open("a", encoding="utf-8") as handle:
        for batch in _batches(pending, args.batch_size):
            generated = _generate(adapter, batch, prompt_ids, sites, stats, column, matched, hidden, args, eos, device)
            for job, rollout in zip(batch, generated):
                text = tokenizer.decode(rollout["token_ids"], skip_special_tokens=True)
                handle.write(json.dumps(_record(job, rollout, text, sites, stats, column, family)) + "\n")
            handle.flush()
    return str(path)


def load_causal_lm(model_path: str):
    """Load a local causal LM. Callers that already hold a module skip this."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return model, tokenizer


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model, tokenizer = load_causal_lm(args.model)
    run(args, model, tokenizer)


def _push(dose, p50: float, norm: float) -> float:
    return float(dose) * float(p50) / float(norm)


def _check_stats(stats: CompactStats, family: str, recurrences: int | None) -> None:
    config_family = stats.config.get("family")
    if config_family is not None and config_family != family:
        raise ValueError(f"stats family {config_family} does not match {family}")
    config_recurrences = stats.config.get("recurrences")
    if recurrences is not None and config_recurrences is not None and int(config_recurrences) != int(recurrences):
        raise ValueError(f"stats recurrences {config_recurrences} does not match {recurrences}")


def _resolve_sites(adapter, args) -> tuple[tuple[ExecutionSite, ...], str]:
    if args.all_recurrences:
        kind, _sep, index = str(args.all_recurrences).partition(":")
        if not _sep or not kind:
            raise ValueError(f"all-recurrences must be kind:index, got {args.all_recurrences!r}")
        physical = int(index)
        sites = tuple(site for site in adapter.sites if site.kind == kind and site.physical_index == physical)
        if not sites:
            raise ValueError(f"no execution sites for {kind}[{physical}]")
        return sites, "all_recurrences"
    matches = tuple(site for site in adapter.sites if site.virtual_index == args.site)
    if len(matches) != 1:
        raise ValueError(f"virtual site {args.site} matched {len(matches)} firings")
    return matches, "site"


def _resolve_neurons(spec: str, hidden: int) -> list[int]:
    if spec == "all":
        return list(range(hidden))
    path = Path(spec)
    text = path.read_text(encoding="utf-8") if path.is_file() else spec
    neurons = []
    for part in text.replace("\n", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            start, end = part.split("-", 1)
            neurons.extend(range(int(start), int(end) + 1))
        else:
            neurons.append(int(part))
    if any(neuron < 0 or neuron >= hidden for neuron in neurons):
        raise ValueError(f"smoke neurons must be in 0..{hidden - 1}")
    return neurons


def _random_directions(n_controls: int, seed: int, column: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    hidden = int(column.shape[0])
    if n_controls <= 0:
        return torch.zeros(0, hidden, dtype=torch.float32), np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    refs = np.asarray(rng.integers(0, hidden, size=n_controls), dtype=np.int64)
    directions = rng.standard_normal((n_controls, hidden))
    norms = np.linalg.norm(directions, axis=1)
    borrowed = column.numpy()[refs]
    directions = directions * (borrowed / norms)[:, None]
    return torch.tensor(directions, dtype=torch.float32), refs


def _selector_meta(sites: tuple[ExecutionSite, ...], selector: str) -> dict:
    first = sites[0]
    single = selector == "site"
    return {
        "selector": selector,
        "virtual_index": first.virtual_index if single else None,
        "physical_index": first.physical_index,
        "recurrence": first.recurrence if single else None,
        "block_kind": first.kind,
        "virtual_indices": [site.virtual_index for site in sites],
        "recurrences": [site.recurrence for site in sites],
        "layer": first.virtual_index if single else first.physical_index,
        "sites": [
            {
                "kind": site.kind,
                "physical_index": site.physical_index,
                "recurrence": site.recurrence,
                "virtual_index": site.virtual_index,
            }
            for site in sites
        ],
    }


def _units(neurons: Sequence[int], n_controls: int) -> list[tuple[str, int]]:
    units = [("neuron", int(neuron)) for neuron in neurons]
    for control in range(n_controls):
        units.insert(round((control + 1) * len(units) / (n_controls + 1)), ("random", control))
    return units


def _jobs(neurons, n_controls: int, refs: np.ndarray, meta: dict) -> list[dict]:
    jobs = []
    for prompt in PROMPTS:
        jobs.append(_job("baseline", -1, -1, prompt, 0.0, meta))
    for prompt in PROMPTS:
        doses = [dose for name, dose in MAIN_REL if name == prompt]
        for kind, index in _units(neurons, n_controls):
            ref = int(index if kind == "neuron" else refs[index])
            for dose in doses:
                jobs.append(_job(kind, index, ref, prompt, dose, meta))
    return jobs


def _job(kind: str, neuron: int, ref: int, prompt: str, dose: float, meta: dict) -> dict:
    return {
        "kind": kind,
        "neuron": int(neuron),
        "ref_neuron": int(ref),
        "prompt": prompt,
        "dose": float(dose),
        **meta,
    }


def _done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            keys.add(record_key(json.loads(line)))
    return keys


def _batches(jobs: Sequence[dict], batch_size: int) -> Iterator[list[dict]]:
    chunk: list[dict] = []
    for job in jobs:
        if chunk and (job["prompt"] != chunk[0]["prompt"] or len(chunk) == batch_size):
            yield chunk
            chunk = []
        chunk.append(job)
    if chunk:
        yield chunk


def _encode(tokenizer) -> dict[str, list[int]]:
    encoded = {}
    for key, text in PROMPTS.items():
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
        )
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        if torch.is_tensor(ids):
            ids = ids.tolist()
        encoded[key] = [int(token) for token in ids]
    return encoded


def _device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _generate(adapter, batch, prompt_ids, sites, stats, column, matched, hidden, args, eos, device):
    ids = torch.tensor([prompt_ids[batch[0]["prompt"]]] * len(batch), dtype=torch.long, device=device)
    rollouts = [{"token_ids": [], "logprob": [], "top_ids": [], "top_lps": []} for _ in batch]
    alive = [True] * len(batch)
    edits = None if all(job["kind"] == "baseline" for job in batch) else edits_for_rows(
        batch, sites, stats, column, matched, hidden
    )
    context = nullcontext() if edits is None else adapter.intervene(edits)
    with context, torch.no_grad():
        for _step in range(args.max_tokens):
            if not any(alive):
                break
            logits = _logits(adapter.forward(ids, use_cache=False))[:, -1, :]
            logp = torch.log_softmax(logits.float(), dim=-1)
            chosen = torch.argmax(logits, dim=-1)
            k = min(int(args.top_k), int(logp.shape[-1]))
            top_vals, top_idx = torch.topk(logp, k=k, dim=-1)
            for row, job_alive in enumerate(alive):
                if not job_alive:
                    continue
                token = int(chosen[row])
                rollouts[row]["token_ids"].append(token)
                rollouts[row]["logprob"].append(round(float(logp[row, token]), 4))
                rollouts[row]["top_ids"].append([int(item) for item in top_idx[row].tolist()])
                rollouts[row]["top_lps"].append([round(float(item), 4) for item in top_vals[row].tolist()])
                if eos is not None and token == int(eos):
                    alive[row] = False
            ids = torch.cat([ids, chosen.to(dtype=ids.dtype).view(-1, 1)], dim=1)
    for rollout in rollouts:
        last = rollout["token_ids"][-1] if rollout["token_ids"] else None
        rollout["finish"] = "stop" if eos is not None and last == int(eos) else "length"
        rollout["n_gen"] = len(rollout["token_ids"])
    return rollouts


def _logits(output):
    logits = output.logits if hasattr(output, "logits") else output
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits


def _record(job, rollout, text, sites, stats, column, family) -> dict:
    scales = _site_adds(job, sites, stats, column)
    if job["kind"] == "baseline":
        add, rel = 0.0, 0.0
    else:
        ref = int(job["ref_neuron"] if job["kind"] == "random" else job["neuron"])
        p50 = float(stats.p50(sites[0]))
        norm = float(column[ref])
        add = scales[0]
        rel = abs(scales[0]) * norm / p50 if p50 else 0.0
    return {
        "schema_version": 2,
        "family": family,
        "selector": job["selector"],
        "virtual_index": job["virtual_index"],
        "physical_index": job["physical_index"],
        "recurrence": job["recurrence"],
        "block_kind": job["block_kind"],
        "virtual_indices": list(job["virtual_indices"]),
        "recurrences": list(job["recurrences"]),
        "sites": job["sites"],
        "layer": job["layer"],
        "kind": job["kind"],
        "neuron": job["neuron"],
        "ref_neuron": job["ref_neuron"],
        "prompt": job["prompt"],
        "dose": job["dose"],
        "add": add,
        "site_add": scales,
        "rel_push": rel,
        "text": text,
        "finish": rollout["finish"],
        "n_gen": rollout["n_gen"],
        "token_ids": rollout["token_ids"],
        "logprob": rollout["logprob"],
        "top_ids": rollout["top_ids"],
        "top_lps": rollout["top_lps"],
    }


def _site_adds(job, sites, stats, column) -> list[float]:
    if job["kind"] == "baseline":
        return [0.0 for _ in sites]
    ref = int(job["ref_neuron"] if job["kind"] == "random" else job["neuron"])
    norm = float(column[ref])
    return [_push(job["dose"], float(stats.p50(site)), norm) for site in sites]


if __name__ == "__main__":
    main()
