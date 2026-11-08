"""Compare Nanbeige attention implementations and record the first non-finite stage."""
from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from sweep.looped_smoke import (
    CALIBRATION_PROMPTS,
    _ids,
    load_local_model,
    run_smoke,
    trace_nonfinite,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Locate Nanbeige non-finite logits")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _release(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def diagnose(args) -> dict:
    """Try the loader default, then eager attention. Smoke only a finite result."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    modes = []
    for requested in (None, "eager"):
        label = requested or "default"
        mode = {"requested_attn": label, "status": "failed"}
        model = None
        try:
            model, tokenizer = load_local_model(args.model, attn_implementation=requested)
            device = next(model.parameters()).device
            ids = _ids(tokenizer, CALIBRATION_PROMPTS[0], device)
            mode["trace"] = trace_nonfinite(model, ids)
            finite = next(
                (row for row in mode["trace"]["stages"] if row["name"] == "logits"),
                {},
            ).get("finite", 0)
            if finite:
                smoke_args = SimpleNamespace(
                    model=args.model, family="nanbeige", out=str(out / label),
                    batch_size=args.batch_size, recurrences=None,
                    max_tokens=args.max_tokens, seed=args.seed,
                )
                mode["smoke"] = run_smoke(smoke_args, model, tokenizer)
                mode["status"] = "passed"
            else:
                mode["status"] = "nonfinite"
        except Exception as exc:
            mode["error"] = f"{type(exc).__name__}: {exc}"
            mode["traceback"] = traceback.format_exc()
        finally:
            if model is not None:
                _release(model)
        modes.append(mode)
        if mode["status"] == "passed":
            break
    passed = any(mode["status"] == "passed" for mode in modes)
    return {"schema_version": 1, "status": "passed" if passed else "failed", "modes": modes}


def main(argv=None) -> None:
    args = parse_args(argv)
    report = diagnose(args)
    path = Path(args.out) / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
