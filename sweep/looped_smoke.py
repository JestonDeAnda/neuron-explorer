"""Production smoke validation for recurrence-aware model sweeps."""
from __future__ import annotations

import argparse
import json
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F

from sweep.looped_models import BatchEdits, adapt
from sweep.looped_neuron_stats import collect_site_stats, save_stats
from sweep.looped_steer_sweep import run as run_steering

CALIBRATION_PROMPTS = (
    "Explain why the sky appears blue.",
    "Write two short sentences about careful experiments.",
)
PROBE_PROMPT = "Smoke test"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one looped model on CUDA")
    parser.add_argument("--model", required=True)
    parser.add_argument("--family", required=True, choices=("ouro", "nanbeige", "huginn"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--recurrences", type=int)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_tokens < 1:
        parser.error("--batch-size and --max-tokens must be positive")
    if args.family == "huginn" and (args.recurrences is None or args.recurrences < 2):
        parser.error("Huginn requires --recurrences >= 2")
    if args.family != "huginn" and args.recurrences is not None:
        parser.error("--recurrences is a Huginn-only control")
    return args


def prepare_checkpoint_config(config, tokenizer, *, rope_functions=None):
    """Fill fields required by older custom model code without overriding them."""
    if getattr(config, "pad_token_id", None) is None:
        candidates = (
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(config, "eos_token_id", None),
        )
        pad = next((value for value in candidates if value is not None), 0)
        if isinstance(pad, (tuple, list)):
            pad = pad[0]
        config.pad_token_id = int(pad)
    rope_scaling = getattr(config, "rope_scaling", None)
    if (
        rope_functions is not None
        and getattr(config, "model_type", None) == "ouro"
        and (
            rope_scaling is None
            or (
                isinstance(rope_scaling, dict)
                and rope_scaling.get("rope_type", rope_scaling.get("type")) == "default"
            )
        )
        and "default" not in rope_functions
        and "proportional" in rope_functions
    ):
        # Transformers 5's proportional RoPE with factor 1 is the removed
        # standard/default formula. Naming it avoids the new base initializer
        # trying to call a method absent from this older remote model class.
        replacement = {"rope_type": "proportional", "factor": 1.0}
        if getattr(config, "rope_theta", None) is not None:
            replacement["rope_theta"] = config.rope_theta
        config.rope_scaling = replacement
    return config


def _legacy_tied_weight_mapping(model) -> dict[str, str]:
    """Convert the pre-Transformers-5 tied-key list using real parameter aliases."""
    keys = getattr(model, "_tied_weights_keys", None)
    if not isinstance(keys, (tuple, list)):
        return keys
    parameters = dict(model.named_parameters(remove_duplicate=False))
    input_embedding = None
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        input_embedding = getattr(get_input_embeddings(), "weight", None)
    input_sources = [
        name for name, candidate in parameters.items()
        if candidate is input_embedding and name not in keys
    ]
    mapping = {}
    for target in keys:
        parameter = parameters.get(target)
        aliases = [
            name for name, candidate in parameters.items()
            if name != target and candidate is parameter
        ]
        sources = input_sources or aliases
        if parameter is None or not sources:
            raise ValueError(
                f"cannot resolve legacy tied weight {target!r} to a source parameter"
            )
        mapping[target] = sources[0]
    return mapping


@contextmanager
def _legacy_tied_weights_compat(pretrained_model_class) -> Iterator[None]:
    """Temporarily adapt old remote-code lists to the Transformers 5 mapping API."""
    original = pretrained_model_class.get_expanded_tied_weights_keys

    def compatible(model, *args, **kwargs):
        if (
            getattr(getattr(model, "config", None), "tie_word_embeddings", False)
            and isinstance(getattr(model, "_tied_weights_keys", None), (tuple, list))
        ):
            model._tied_weights_keys = _legacy_tied_weight_mapping(model)
        return original(model, *args, **kwargs)

    pretrained_model_class.get_expanded_tied_weights_keys = compatible
    try:
        yield
    finally:
        pretrained_model_class.get_expanded_tied_weights_keys = original


@contextmanager
def _legacy_rope_compat(rope_functions) -> Iterator[None]:
    """Restore the removed standard RoPE key expected by older remote code."""
    added = "default" not in rope_functions
    if added:
        if "proportional" not in rope_functions:
            raise RuntimeError("Transformers exposes neither default nor proportional RoPE")
        rope_functions["default"] = rope_functions["proportional"]
    try:
        yield
    finally:
        if added:
            del rope_functions["default"]


def stabilize_low_precision_attention(model):
    """Keep bf16/fp16 causal masks finite.

    Nanbeige builds the disallowed mask with ``finfo(dtype).min``. In bf16 that
    value is large enough that the attention softmax overflows to NaN. A floor
    of -1e4 still hides those positions and stays finite.
    """
    owners = [model]
    inner = getattr(model, "model", None)
    if isinstance(inner, torch.nn.Module):
        owners.append(inner)
    for owner in owners:
        original = getattr(owner, "_update_causal_mask", None)
        if original is None or getattr(original, "_low_precision_safe", False):
            continue

        def wrapped(*args, _original=original, **kwargs):
            mask = _original(*args, **kwargs)
            if torch.is_tensor(mask) and mask.is_floating_point() and mask.dtype.itemsize <= 2:
                floor = torch.tensor(-1.0e4, dtype=mask.dtype, device=mask.device)
                mask = torch.maximum(mask, floor)
            return mask

        wrapped._low_precision_safe = True
        owner._update_causal_mask = wrapped
    return model


def load_local_model(model_path: str):
    """Load a local checkpoint in the exact production configuration."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for production smoke validation")
    path = Path(model_path)
    if not path.is_dir():
        raise ValueError(f"--model must be a staged local directory: {model_path}")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    tokenizer = AutoTokenizer.from_pretrained(
        str(path), trust_remote_code=True, local_files_only=True
    )
    config = AutoConfig.from_pretrained(
        str(path), trust_remote_code=True, local_files_only=True
    )
    prepare_checkpoint_config(config, tokenizer, rope_functions=ROPE_INIT_FUNCTIONS)
    with (
        _legacy_rope_compat(ROPE_INIT_FUNCTIONS),
        _legacy_tied_weights_compat(PreTrainedModel),
    ):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(path), config=config, trust_remote_code=True, local_files_only=True,
                dtype=torch.bfloat16,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                str(path), config=config, trust_remote_code=True, local_files_only=True,
                torch_dtype=torch.bfloat16,
            )
    model.to("cuda")
    model.eval()
    stabilize_low_precision_attention(model)
    return model, tokenizer


def _adapter(model, args):
    if args.family == "huginn":
        return adapt(model, family=args.family, recurrences=args.recurrences,
                     seed=args.seed, test_time_noise=0.0)
    return adapt(model, family=args.family)


def representative_sites(adapter):
    """Return first and last recurrence of one recurrent physical block."""
    groups: dict[tuple[str, int], list] = {}
    for site in adapter.sites:
        groups.setdefault((site.kind, site.physical_index), []).append(site)
    candidates = [sites for sites in groups.values()
                  if len({site.recurrence for site in sites}) >= 2]
    if not candidates:
        raise RuntimeError("model exposes no physical block with multiple recurrences")
    sites = sorted(max(candidates, key=len), key=lambda site: site.recurrence)
    return sites[0], sites[-1]


def _ids(tokenizer, text: str, device: torch.device, length: int | None = None):
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if torch.is_tensor(ids):
        ids = ids.tolist()
    ids = [int(item) for item in ids]
    fallback = getattr(tokenizer, "eos_token_id", None)
    fallback = 0 if fallback is None else int(fallback)
    if length is not None:
        ids = (ids + [fallback] * length)[:length]
    if not ids:
        ids = [fallback]
    return torch.tensor([ids], dtype=torch.long, device=device)


def _logits(output):
    value = output.logits if hasattr(output, "logits") else output
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not torch.is_tensor(value):
        raise TypeError(f"unsupported logits output {type(value)!r}")
    return value


def validate_observer_equivalence(adapter, input_ids: torch.Tensor) -> dict:
    with torch.inference_mode():
        plain = _logits(adapter.forward(input_ids, use_cache=False)).detach()
        with adapter.observe(lambda _site, _activation: None):
            observed = _logits(adapter.forward(input_ids, use_cache=False)).detach()
    plain_finite = torch.isfinite(plain)
    observed_finite = torch.isfinite(observed)
    finite_mask_equal = bool(torch.equal(plain_finite, observed_finite))
    nonfinite_kind_equal = all(
        bool(torch.equal(check(plain), check(observed)))
        for check in (torch.isnan, torch.isposinf, torch.isneginf)
    )
    finite_logits = int(plain_finite.sum())
    if finite_logits == 0:
        raise AssertionError("no-op observer comparison has no finite logits")
    finite_equal = (
        finite_mask_equal
        and bool(torch.equal(plain[plain_finite], observed[observed_finite]))
    )
    logits_equal = finite_equal and nonfinite_kind_equal
    greedy_equal = bool(torch.equal(plain.argmax(-1), observed.argmax(-1)))
    if not logits_equal or not greedy_equal:
        if finite_mask_equal and finite_logits:
            difference = float(
                (plain[plain_finite].float() - observed[observed_finite].float())
                .abs().max()
            )
        else:
            difference = float("nan")
        raise AssertionError(
            f"no-op observer changed output: logits_equal={logits_equal}, "
            f"greedy_equal={greedy_equal}, finite_mask_equal={finite_mask_equal}, "
            f"nonfinite_mask_equal={nonfinite_kind_equal}, max_abs_diff={difference}"
        )
    return {"logits_equal": logits_equal, "greedy_equal": greedy_equal,
            "finite_logits": finite_logits,
            "nonfinite_logits": int(plain.numel() - finite_logits),
            "shape": list(plain.shape)}


def _synchronize(cuda) -> None:
    if cuda is not None and cuda.is_available():
        cuda.synchronize()


def validate_full_occupancy(adapter, input_ids, selected, unselected,
                            cuda=torch.cuda) -> dict:
    """Validate every requested row and time one synchronized two-token pass."""
    batch = int(input_ids.shape[0])
    if int(input_ids.shape[1]) != 2:
        raise ValueError("occupancy probe must have exactly two tokens")
    module = adapter.activation_module(selected)
    if module is not adapter.activation_module(unselected):
        raise AssertionError("representative sites must share one projection module")
    width = int(adapter.write_matrix(selected).shape[1])
    neurons = torch.tensor([row % min(3, width) for row in range(batch)], dtype=torch.long)
    deltas = torch.tensor([0.125 * (1 + row % 4) for row in range(batch)],
                          dtype=torch.float32)
    before, after, outputs = [], [], []

    def capture_before(_module, inputs):
        before.append(inputs[0].detach().clone())

    def capture_after(_module, inputs):
        after.append(inputs[0].detach().clone())

    def capture_output(_module, _inputs, output):
        outputs.append(output.detach().clone())

    h_before = module.register_forward_pre_hook(capture_before)
    h_output = module.register_forward_hook(capture_output)
    try:
        with adapter.intervene({selected: BatchEdits(neurons, deltas)}):
            h_after = module.register_forward_pre_hook(capture_after)
            try:
                if cuda is not None and cuda.is_available():
                    cuda.reset_peak_memory_stats()
                _synchronize(cuda)
                started = time.perf_counter()
                with torch.inference_mode():
                    adapter.forward(input_ids, use_cache=False)
                _synchronize(cuda)
                seconds = time.perf_counter() - started
            finally:
                h_after.remove()
    finally:
        h_before.remove()
        h_output.remove()

    early_index, late_index = selected.recurrence, unselected.recurrence
    if len(before) <= late_index or len(after) <= late_index:
        raise AssertionError("projection did not fire at representative recurrences")
    weight, bias = module.weight.detach(), getattr(module, "bias", None)
    activation_delta = after[early_index] - before[early_index]
    expected_delta = torch.zeros_like(activation_delta)
    edited = deltas.to(device=activation_delta.device, dtype=activation_delta.dtype)
    for row in range(int(activation_delta.shape[0])):
        neuron = int(neurons[row])
        if neuron >= 0:
            expected_delta[row, ..., neuron] = edited[row]
    # Adding the dose rounds to the activation dtype, so the edited column can
    # differ by a few units in the last place. Every other column stays zero.
    # bf16 also cannot reconstruct delta * W by subtracting two GEMMs.
    edited_positions = expected_delta != 0
    rounding = float(4 * torch.finfo(activation_delta.dtype).eps)
    torch.testing.assert_close(
        activation_delta[edited_positions], expected_delta[edited_positions],
        rtol=0, atol=rounding,
    )
    torch.testing.assert_close(
        activation_delta[~edited_positions], expected_delta[~edited_positions],
        rtol=0, atol=0,
    )
    torch.testing.assert_close(outputs[early_index],
                               F.linear(after[early_index], weight, bias), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(after[late_index], before[late_index], rtol=0, atol=0)
    torch.testing.assert_close(outputs[late_index],
                               F.linear(before[late_index], weight, bias), rtol=1e-2, atol=1e-2)
    allocated = reserved = 0
    if cuda is not None and cuda.is_available():
        allocated = int(cuda.max_memory_allocated())
        reserved = int(cuda.max_memory_reserved())
    tokens = batch * 2
    return {
        "requested_batch": batch, "actual_batch": int(input_ids.shape[0]),
        "sequence_length": 2, "rows_verified": batch,
        "selected_site": _site_dict(selected), "unselected_site": _site_dict(unselected),
        "selected_delta_matches_write_column": True,
        "unselected_recurrence_unchanged": True, "seconds": seconds, "tokens": tokens,
        "throughput_tokens_per_second": tokens / seconds,
        "cuda_peak_allocated_bytes": allocated, "cuda_peak_reserved_bytes": reserved,
    }


def _site_dict(site) -> dict:
    return {"kind": site.kind, "physical_index": int(site.physical_index),
            "recurrence": int(site.recurrence), "virtual_index": int(site.virtual_index)}


def run_smoke(args, model, tokenizer, cuda=torch.cuda) -> dict:
    torch.manual_seed(args.seed)
    if cuda is not None and cuda.is_available():
        cuda.manual_seed_all(args.seed)
    model.eval()
    adapter = _adapter(model, args)
    selected, unselected = representative_sites(adapter)
    device = next(model.parameters()).device
    equivalence = validate_observer_equivalence(
        adapter, _ids(tokenizer, CALIBRATION_PROMPTS[0], device))
    probe = _ids(tokenizer, PROBE_PROMPT, device, length=2).expand(
        args.batch_size, -1).clone()
    occupancy = validate_full_occupancy(adapter, probe, selected, unselected, cuda)
    calibration_batches = [_ids(tokenizer, prompt, device)
                           for prompt in CALIBRATION_PROMPTS]
    with torch.inference_mode():
        stats = collect_site_stats(adapter, calibration_batches,
                                   sites=(selected, unselected))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats_path = out / "tiny_stats.npz"
    save_stats(str(stats_path), stats, {
        "model": args.model, "family": args.family,
        "recurrences": args.recurrences, "seed": args.seed,
        "test_time_noise": 0.0, "prompts": list(CALIBRATION_PROMPTS),
    })
    steering_args = SimpleNamespace(
        family=args.family, recurrences=args.recurrences, seed=args.seed,
        stats=str(stats_path), site=selected.virtual_index, all_recurrences=None,
        smoke_neurons="0,1,2", run="steering", out_dir=str(out),
        max_tokens=args.max_tokens, top_k=5, batch_size=min(args.batch_size, 9),
        random_controls=0,
    )
    steering_path = run_steering(steering_args, model, tokenizer)
    rows = sum(bool(line.strip()) for line in
               Path(steering_path).read_text(encoding="utf-8").splitlines())
    return {
        "schema_version": 1, "status": "passed", "model": args.model,
        "family": args.family, "batch_size": args.batch_size,
        "recurrences": args.recurrences, "seed": args.seed,
        "max_tokens": args.max_tokens,
        "huginn": {"fixed_recurrences": args.recurrences if args.family == "huginn" else None,
                   "seed": args.seed if args.family == "huginn" else None,
                   "test_time_noise": 0.0 if args.family == "huginn" else None},
        "representative_sites": [_site_dict(selected), _site_dict(unselected)],
        "observer_equivalence": equivalence, "occupancy_probe": occupancy,
        "stats": {"path": str(stats_path), "prompts": len(CALIBRATION_PROMPTS),
                  "sites": len(stats.sites)},
        "steering": {"path": steering_path, "neurons": [0, 1, 2], "rows": rows},
    }


def aggregate_nvidia_csv(path: str | Path) -> dict:
    """Aggregate nvidia-smi timestamp,index,memory.used,utilization.gpu CSV."""
    import csv

    memory, utilization = [], []
    path = Path(path)
    if not path.exists():
        return {"samples": 0, "peak_memory_mib": 0.0,
                "utilization_percent": {"mean": 0.0, "p50": 0.0,
                                        "p95": 0.0, "max": 0.0}}
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 4:
                continue
            try:
                memory.append(float(row[-2].strip()))
                utilization.append(float(row[-1].strip()))
            except ValueError:
                continue
    if not utilization:
        return {"samples": 0, "peak_memory_mib": 0.0,
                "utilization_percent": {"mean": 0.0, "p50": 0.0,
                                        "p95": 0.0, "max": 0.0}}
    import numpy as np
    values = np.asarray(utilization, dtype=np.float64)
    return {
        "samples": len(utilization),
        "peak_memory_mib": max(memory),
        "utilization_percent": {
            "mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max()),
        },
    }


def build_summary(records: list[dict]) -> dict:
    """Build the stable cross-model driver summary schema."""
    passed = sum(record.get("status") == "passed" for record in records)
    return {
        "schema_version": 1,
        "status": "passed" if passed == len(records) else "failed",
        "models_total": len(records), "models_passed": passed,
        "models": records,
    }


def _write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    smoke_path = Path(args.out) / "smoke.json"
    try:
        model, tokenizer = load_local_model(args.model)
        _write_result(smoke_path, run_smoke(args, model, tokenizer))
    except BaseException as exc:
        _write_result(smoke_path, {
            "schema_version": 1, "status": "failed", "model": args.model,
            "family": args.family, "batch_size": args.batch_size,
            "recurrences": args.recurrences, "seed": args.seed,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
