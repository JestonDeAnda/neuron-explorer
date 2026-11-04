"""Recurrence-aware activation statistics for looped models.

A neuron activation is ``silu(gate) * up``, the tensor read by the down
projection (``mlp.proj`` on Huginn). Positive values come from the (+, +) and
(−, −) sign regimes; negative values come from (+, −) and (−, +). Each
execution site keeps the extreme of each regime. Position 0 of every window
is an attention sink and is left out of both the regime extremes and the
residual-norm quantiles.

This module does not load a checkpoint. Pair it with :func:`sweep.looped_models.adapt`.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class Run:
    """Resolved controls for one statistics pass. ``test_time_noise`` stays 0."""

    model: str
    family: str
    out: str
    texts: str | None
    text_field: str
    corpus_parquet: str | None
    tokens: int | None
    seq_len: int | None
    batch: int | None
    recurrences: int | None
    seed: int
    test_time_noise: float = 0.0


@dataclass
class SiteStats:
    """Extremes for one forward schedule, row-aligned with ``sites``."""

    sites: tuple
    pp_max: np.ndarray
    nn_max: np.ndarray
    pn_min: np.ndarray
    np_min: np.ndarray
    abs_max: np.ndarray
    resid_norm_q10_50_90: np.ndarray


class _Regime:
    """Online sign-regime extremes and residual norms for one execution site."""

    def __init__(self, neurons: int, device: torch.device) -> None:
        zeros = torch.zeros(neurons, device=device, dtype=torch.float32)
        self.pp_max = zeros.clone()
        self.nn_max = zeros.clone()
        self.pn_min = zeros.clone()
        self.np_min = zeros.clone()
        self.resid_norms: list[np.ndarray] = []

    def update(self, gate: torch.Tensor, up: torch.Tensor, act: torch.Tensor) -> None:
        # Position 0 is an attention sink; it would dominate abs_max.
        gate, up, act = (_body(gate), _body(up), _body(act))
        positive_gate = gate > 0
        positive_up = up > 0
        ninf = torch.tensor(float("-inf"), device=act.device)
        pinf = torch.tensor(float("inf"), device=act.device)
        self.pp_max = torch.maximum(self.pp_max, torch.where(positive_gate & positive_up, act, ninf).amax(0))
        self.nn_max = torch.maximum(self.nn_max, torch.where(~positive_gate & ~positive_up, act, ninf).amax(0))
        self.pn_min = torch.minimum(self.pn_min, torch.where(positive_gate & ~positive_up, act, pinf).amin(0))
        self.np_min = torch.minimum(self.np_min, torch.where(~positive_gate & positive_up, act, pinf).amin(0))

    def add_residual(self, residual: torch.Tensor) -> None:
        norms = _body(residual).norm(dim=-1).flatten().detach().cpu().numpy()
        self.resid_norms.append(norms)


def collect_site_stats(adapter, batches: Iterable[torch.Tensor], sites: Sequence | None = None) -> SiteStats:
    """Run ``batches`` through ``adapter`` and reduce per-site activation extremes.

    ``sites`` selects firings. Omitting it records every site on the adapter.
    Neuron width is checked before any forward, and capture hooks are removed
    on the way out, including when a forward fires the wrong number of times.
    """
    chosen = tuple(adapter.sites if sites is None else sites)
    if not chosen:
        raise ValueError("no execution sites selected")
    width = _check_neuron_width(adapter, chosen)
    device = _model_device(adapter.model)
    regimes = {site: _Regime(width, device) for site in chosen}
    plan = _capture_plan(adapter, chosen)
    chosen_set = set(chosen)
    predicate = None if sites is None else (lambda site: site in chosen_set)

    def on_activation(site, activation: torch.Tensor) -> None:
        if site not in regimes:
            return
        gate, up = _read_gate_up(plan, latest, site)
        regimes[site].update(gate, up, activation)

    def on_residual(site, residual: torch.Tensor) -> None:
        if site in regimes:
            regimes[site].add_residual(residual)

    with _gate_up_hooks(plan) as latest:
        with adapter.observe(on_activation, predicate, on_block_output=on_residual):
            for batch in batches:
                adapter.forward(batch, use_cache=False)

    return _pack(chosen, regimes)


def save_stats(path: str, stats: SiteStats, config: dict) -> None:
    """Write regime extremes, residual quantiles, and site metadata.

    Down-projection matrices stay out of the file. They are shared across
    recurrences and are not needed to read the activation scale.
    """
    meta = [
        {
            "kind": site.kind,
            "physical_index": int(site.physical_index),
            "recurrence": int(site.recurrence),
            "virtual_index": int(site.virtual_index),
        }
        for site in stats.sites
    ]
    np.savez(
        path,
        pp_max=np.asarray(stats.pp_max),
        nn_max=np.asarray(stats.nn_max),
        pn_min=np.asarray(stats.pn_min),
        np_min=np.asarray(stats.np_min),
        abs_max=np.asarray(stats.abs_max),
        resid_norm_q10_50_90=np.asarray(stats.resid_norm_q10_50_90),
        sites_json=np.array(json.dumps(meta)),
        config_json=np.array(json.dumps(config)),
    )


def iter_text_batches(tokenizer, path, text_field: str):
    """Yield one ``[1, seq]`` batch per JSONL row."""
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if text_field not in row:
                raise ValueError(f"line {line_number} is missing text field {text_field!r}")
            ids = _input_ids(tokenizer, row[text_field])
            yield torch.tensor([ids], dtype=torch.long)


def iter_parquet_batches(tokenizer, path, tokens: int, seq_len: int, batch: int, text_field: str):
    """Pack a parquet text column into ``[batch, seq_len]`` windows.

    Documents are tokenized without special tokens and separated by EOS, the
    same packing the flat sweep uses. The iterator stops after enough full
    windows to cover ``tokens`` and raises if the column runs out first.
    """
    if tokens < 1 or seq_len < 1 or batch < 1:
        raise ValueError("tokens, seq_len, and batch must be positive")
    window = seq_len * batch
    n_batches = (tokens + window - 1) // window
    needed = n_batches * window
    table = pq.read_table(path, columns=[text_field])
    eos = tokenizer.eos_token_id
    buf: list[int] = []
    produced = 0
    for text in table.column(text_field).to_pylist():
        buf.extend(_input_ids(tokenizer, text))
        buf.append(eos)
        while len(buf) >= window and produced < n_batches:
            chunk = buf[:window]
            del buf[:window]
            produced += 1
            yield torch.tensor(chunk, dtype=torch.long).view(batch, seq_len)
    if produced < n_batches:
        have = produced * window + len(buf)
        raise ValueError(f"not enough tokens in {path}: have {have}, need {needed}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse a stats invocation. Test-time noise is not a flag; it stays 0."""
    parser = argparse.ArgumentParser(description="Recurrence-aware neuron activation statistics")
    parser.add_argument("--model", required=True)
    parser.add_argument("--family", required=True, choices=("ouro", "huginn", "nanbeige"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--texts")
    parser.add_argument("--text-field", default="passage")
    parser.add_argument("--corpus-parquet")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--recurrences", type=int)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def resolve_run(args: argparse.Namespace) -> Run:
    """Check family controls and that exactly one corpus source is set."""
    family = args.family.lower()
    if family not in ("ouro", "huginn", "nanbeige"):
        raise ValueError("family must be one of: ouro, huginn, nanbeige")
    texts = args.texts
    parquet = args.corpus_parquet
    if (texts is None) == (parquet is None):
        raise ValueError("provide exactly one of --texts or --corpus-parquet")
    if family == "huginn":
        if args.recurrences is None:
            raise ValueError("Huginn recurrences must be set")
        if isinstance(args.recurrences, bool) or not isinstance(args.recurrences, int) or args.recurrences < 1:
            raise ValueError("recurrences must be a positive integer")
        recurrences = args.recurrences
    else:
        if args.recurrences is not None or args.seed != 0:
            raise ValueError("recurrences and seed are Huginn controls")
        recurrences = None
    if parquet is not None and (args.tokens is None or args.seq_len is None or args.batch is None):
        raise ValueError("corpus-parquet requires --tokens, --seq-len, and --batch")
    return Run(
        model=args.model,
        family=family,
        out=args.out,
        texts=texts,
        text_field=args.text_field,
        corpus_parquet=parquet,
        tokens=args.tokens,
        seq_len=args.seq_len,
        batch=args.batch,
        recurrences=recurrences,
        seed=args.seed,
        test_time_noise=0.0,
    )


def _check_neuron_width(adapter, sites: Sequence) -> int:
    found: dict[int, tuple] = {}
    for site in sites:
        width = _neuron_width(adapter.activation_module(site))
        found.setdefault(width, site)
    if len(found) != 1:
        detail = ", ".join(f"{width} (e.g. {site.kind}[{site.physical_index}])" for width, site in sorted(found.items()))
        raise ValueError(f"neuron width mismatch across sites: {detail}")
    return next(iter(found))


def _neuron_width(module: nn.Module) -> int:
    width = getattr(module, "in_features", None)
    if isinstance(width, int) and not isinstance(width, bool):
        return width
    weight = getattr(module, "weight", None)
    if torch.is_tensor(weight) and weight.ndim == 2:
        return int(weight.shape[1])
    raise ValueError(f"neuron width is unavailable for {type(module).__name__}")


def _capture_plan(adapter, sites: Sequence) -> dict[tuple[str, int], tuple]:
    plan = {}
    for site in sites:
        key = (site.kind, site.physical_index)
        if key in plan:
            continue
        mlp = adapter.physical_block(site).mlp
        if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
            plan[key] = ("split", mlp.gate_proj, mlp.up_proj)
        elif hasattr(mlp, "fc"):
            plan[key] = ("chunk", mlp.fc, None)
        else:
            raise ValueError(f"{site.kind}[{site.physical_index}] has no SwiGLU gate/up projection")
    return plan


@contextmanager
def _gate_up_hooks(plan: dict[tuple[str, int], tuple]):
    latest: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []
    hooked: set[int] = set()
    try:
        for spec in plan.values():
            if spec[0] == "split":
                _, gate_mod, up_mod = spec
                _hook_split(gate_mod, up_mod, latest, handles, hooked)
            else:
                _hook_chunk(spec[1], latest, handles, hooked)
        yield latest
    finally:
        for handle in handles:
            handle.remove()


def _hook_split(gate_mod, up_mod, latest, handles, hooked) -> None:
    def gate_hook(_module, _inputs, output, module_id=id(gate_mod)) -> None:
        gate = _as_tensor(output, "gate")
        up = latest.get(module_id, (None, None))[1]
        latest[module_id] = (gate, up)

    def up_hook(_module, _inputs, output, module_id=id(gate_mod), up_id=id(up_mod)) -> None:
        up = _as_tensor(output, "up")
        gate = latest.get(module_id, (None, None))[0]
        latest[module_id] = (gate, up)
        latest[up_id] = (gate, up)

    if id(gate_mod) not in hooked:
        handles.append(gate_mod.register_forward_hook(gate_hook))
        hooked.add(id(gate_mod))
    if id(up_mod) not in hooked:
        handles.append(up_mod.register_forward_hook(up_hook))
        hooked.add(id(up_mod))


def _hook_chunk(fc_mod, latest, handles, hooked) -> None:
    if id(fc_mod) in hooked:
        return

    def fc_hook(_module, _inputs, output, module_id=id(fc_mod)) -> None:
        hidden = _as_tensor(output, "fc output")
        gate, up = hidden.chunk(2, dim=-1)
        latest[module_id] = (gate, up)

    handles.append(fc_mod.register_forward_hook(fc_hook))
    hooked.add(id(fc_mod))


def _read_gate_up(plan, latest, site) -> tuple[torch.Tensor, torch.Tensor]:
    spec = plan[(site.kind, site.physical_index)]
    module_id = id(spec[1])
    gate, up = latest[module_id]
    if gate is None or up is None:
        kind, index = site.kind, site.physical_index
        raise RuntimeError(f"{kind}[{index}] activation fired before gate and up were captured")
    return gate, up


def _pack(sites: tuple, regimes: dict) -> SiteStats:
    pp, nn, pn, np_, resid = [], [], [], [], []
    for site in sites:
        regime = regimes[site]
        pp.append(regime.pp_max.detach().cpu().numpy())
        nn.append(regime.nn_max.detach().cpu().numpy())
        pn.append(regime.pn_min.detach().cpu().numpy())
        np_.append(regime.np_min.detach().cpu().numpy())
        if not regime.resid_norms:
            raise RuntimeError(f"{site.kind}[{site.physical_index}] recurrence {site.recurrence} produced no residual")
        values = np.concatenate(regime.resid_norms)
        resid.append(np.quantile(values, [0.1, 0.5, 0.9], method="linear"))
    pp_max = np.stack(pp)
    nn_max = np.stack(nn)
    pn_min = np.stack(pn)
    np_min = np.stack(np_)
    abs_max = np.maximum.reduce([pp_max, nn_max, -pn_min, -np_min])
    return SiteStats(
        sites=sites,
        pp_max=pp_max,
        nn_max=nn_max,
        pn_min=pn_min,
        np_min=np_min,
        abs_max=abs_max,
        resid_norm_q10_50_90=np.stack(resid),
    )


def _body(value: torch.Tensor) -> torch.Tensor:
    tensor = _as_tensor(value, "activation")
    return tensor[:, 1:].detach().flatten(0, 1).float()


def _as_tensor(value, what: str) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f"{what} has unsupported type {type(value)!r}")


def _input_ids(tokenizer, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if torch.is_tensor(ids):
        ids = ids.tolist()
    return [int(token) for token in ids]


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
