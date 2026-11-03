"""Per-neuron activation ranges for MLP layers of a Qwen3.5 model, plus their down-projection columns.

A neuron's activation is silu(gate) * up, the input to down_proj. Positive activations come from two sign regimes
(gate>0, up>0) and (gate<0, up<0); negative ones from (gate>0, up<0) and (gate<0, up>0). This records the extreme of
each regime per neuron over a mixed web + chat corpus, and saves W_down[:, n] for every neuron, which is the direction
a steering push on neuron n adds to the residual stream.

    uv run python sweep/neuron_stats.py --model Qwen/Qwen3.5-4B --layers 22 --tokens 2000000 --out stats/q35-4b

Pass --corpus-parquet to stream a local parquet `text` column instead of FineWeb + UltraChat. A local --model
directory is loaded with local_files_only.

Writes stats/q35-4b_L22.npz with
    pp_max, nn_max   largest positive activation in the (+,+) and (-,-) regimes          [I]
    pn_min, np_min   most negative activation in the (gate+, up-) and (gate-, up+) regimes [I]
    abs_max          max |activation| over everything (the unit for steering doses)      [I]
    w_down           down_proj columns, one row per neuron                                [I, H] float16
    resid_norm_q10_50_90   quantiles of the per-token residual norm leaving the layer (scale for push sizes)
"""
import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoTokenizer


def _is_string_type(dtype):
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype) or pa.types.is_string_view(dtype):
        return True
    if pa.types.is_dictionary(dtype):
        return _is_string_type(dtype.value_type)
    return False


def _model_arg(value):
    """Accept a hub id, or an on-disk model directory with config and tokenizer."""
    path = Path(value)
    if not path.is_absolute() and not path.exists():
        return value
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{value}: path not found")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"{value} is not a directory")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise argparse.ArgumentTypeError(f"{value} is missing config.json")
    if not (path / "tokenizer_config.json").is_file():
        raise argparse.ArgumentTypeError(f"{value} is missing tokenizer files")
    try:
        with config_path.open(encoding="utf-8") as handle:
            json.load(handle)
    except (OSError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{value}: invalid config.json ({exc})") from exc
    return value


def _parquet_arg(value):
    """Accept a parquet file whose `text` column is a string (or dictionary-encoded string)."""
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{value}: path not found")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"{value} is not a parquet file")
    try:
        schema = pq.read_schema(path)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"{value}: unreadable parquet ({exc})") from exc
    if "text" not in schema.names:
        found = ", ".join(schema.names) or "(none)"
        raise argparse.ArgumentTypeError(f"{value}: missing text column; columns: {found}")
    dtype = schema.field("text").type
    if not _is_string_type(dtype):
        raise argparse.ArgumentTypeError(f"{value}: text column must be string, got {dtype}")
    return value


def iter_parquet_text(path, batch_size=65536):
    """Yield non-null `text` values in file order, without reading the whole table."""
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["text"]):
        for value in batch.column("text").to_pylist():
            if value is not None:
                yield value


def text_stream(tok, seed, corpus_parquet=None):
    if corpus_parquet is not None:
        yield from iter_parquet_text(corpus_parquet)
        return
    web = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=seed)
    chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).shuffle(seed=seed)
    for w, c in zip(web, chat):
        yield w["text"]
        yield tok.apply_chat_template(c["messages"], tokenize=False, enable_thinking=False)


def windows(tok, seq_len, batch, seed, corpus_parquet=None):
    """Packs documents (EOS-separated) into [batch, seq_len] token windows."""
    buf = []
    eos = tok.eos_token_id
    for text in text_stream(tok, seed, corpus_parquet=corpus_parquet):
        buf.extend(tok(text, add_special_tokens=False)["input_ids"] + [eos])
        while len(buf) >= seq_len * batch:
            chunk, buf = buf[: seq_len * batch], buf[seq_len * batch :]
            yield torch.tensor(chunk).view(batch, seq_len)


class RegimeStats:
    def __init__(self, n, device):
        f = lambda v: torch.full((n,), v, device=device, dtype=torch.float32)
        self.pp_max, self.nn_max = f(0.0), f(0.0)
        self.pn_min, self.np_min = f(0.0), f(0.0)
        self.gate = self.up = None
        self.resid_norms = []  # per-token residual norm leaving the layer, positions >= 1 (position 0 is a sink)

    def update_resid(self, out):
        h = out[0] if isinstance(out, tuple) else out
        self.resid_norms.append(h[:, 1:].float().norm(dim=-1).flatten().cpu())

    def update(self, act):
        # position 0 of each window is an attention sink with outsized activations; it would inflate abs_max
        g, u, a = (t[:, 1:].flatten(0, 1).float() for t in (self.gate, self.up, act))
        gp, up_ = g > 0, u > 0
        ninf = torch.tensor(float("-inf"), device=a.device)
        pinf = torch.tensor(float("inf"), device=a.device)
        self.pp_max = torch.maximum(self.pp_max, torch.where(gp & up_, a, ninf).amax(0))
        self.nn_max = torch.maximum(self.nn_max, torch.where(~gp & ~up_, a, ninf).amax(0))
        self.pn_min = torch.minimum(self.pn_min, torch.where(gp & ~up_, a, pinf).amin(0))
        self.np_min = torch.minimum(self.np_min, torch.where(~gp & up_, a, pinf).amin(0))


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", type=_model_arg)
    ap.add_argument("--layers", default="22", help="comma-separated decoder layer indices")
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="stats/q35-4b")
    ap.add_argument("--corpus-parquet", default=None, type=_parquet_arg,
                    help="local parquet file with a string text column; replaces FineWeb + UltraChat")
    return ap


def main():
    a = build_parser().parse_args()
    layers = [int(x) for x in a.layers.split(",")]

    local_files_only = os.path.isdir(a.model)
    model_ref = os.path.abspath(a.model) if local_files_only else a.model
    tok = AutoTokenizer.from_pretrained(model_ref, local_files_only=local_files_only)
    model = AutoModelForImageTextToText.from_pretrained(
        model_ref, dtype=torch.bfloat16, device_map="cuda", local_files_only=local_files_only).eval()
    lm = model.model.language_model
    lm.layers = lm.layers[: max(layers) + 1]  # nothing past the deepest target layer is needed

    stats = {}
    for l in layers:
        mlp = lm.layers[l].mlp
        s = stats[l] = RegimeStats(mlp.down_proj.in_features, "cuda")
        mlp.gate_proj.register_forward_hook(lambda m, i, o, s=s: setattr(s, "gate", o))
        mlp.up_proj.register_forward_hook(lambda m, i, o, s=s: setattr(s, "up", o))
        mlp.down_proj.register_forward_pre_hook(lambda m, i, s=s: s.update(i[0]))
        lm.layers[l].register_forward_hook(lambda m, i, o, s=s: s.update_resid(o))

    n_steps = -(-a.tokens // (a.seq_len * a.batch))
    t0 = time.time()
    with torch.inference_mode():
        for step, ids in enumerate(itertools.islice(
                windows(tok, a.seq_len, a.batch, a.seed, corpus_parquet=a.corpus_parquet), n_steps)):
            lm(input_ids=ids.cuda(), use_cache=False)
            if step % 10 == 0 or step == n_steps - 1:
                done = (step + 1) * a.seq_len * a.batch
                print(f"{step + 1}/{n_steps} steps, {done:,} tokens, {done / (time.time() - t0):,.0f} tok/s", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for l, s in stats.items():
        pp, nn, pn, np_ = (x.cpu().numpy() for x in (s.pp_max, s.nn_max, s.pn_min, s.np_min))
        abs_max = np.maximum.reduce([pp, nn, -pn, -np_])
        w_down = lm.layers[l].mlp.down_proj.weight.detach().float().T.cpu().numpy().astype(np.float16)
        resid_q = np.quantile(torch.cat(s.resid_norms).numpy(), [0.1, 0.5, 0.9])
        path = f"{a.out}_L{l}.npz"
        np.savez(path, pp_max=pp, nn_max=nn, pn_min=pn, np_min=np_, abs_max=abs_max, w_down=w_down,
                 resid_norm_q10_50_90=resid_q, model=a.model, layer=l, tokens=n_steps * a.seq_len * a.batch)
        print(f"L{l}: residual norm leaving the layer p10/p50/p90 = {resid_q.round(1)}")
        dn_top = (nn > pp).mean()
        print(f"L{l}: abs_max median {np.median(abs_max):.3f} [{abs_max.min():.3f}, {abs_max.max():.3f}]; "
              f"(-,-) regime holds the positive max for {dn_top:.1%} of neurons -> {path}")


if __name__ == "__main__":
    main()
