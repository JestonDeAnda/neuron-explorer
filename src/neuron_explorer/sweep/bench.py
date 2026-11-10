"""Throughput of the steering fork on this GPU under three conditions, same prompt and fixed-length rollouts:
unsteered, a distinct steering config per request, and steering plus a last_prompt mlp_in capture.

    uv run python scripts/bench.py --stats stats/q35-4b_L22.npz --n 1280 --max-num-seqs 320
"""
import argparse
import time

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--stats", required=True)
    ap.add_argument("--n", type=int, default=1280)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--max-num-seqs", type=int, default=320)
    ap.add_argument("--max-configs", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--only", default="", help="comma-separated condition names")
    a = ap.parse_args()

    st = np.load(a.stats)
    layer, abs_max, w_down = int(st["layer"]), st["abs_max"], st["w_down"]
    tok = AutoTokenizer.from_pretrained(a.model)
    enc = tok.apply_chat_template([{"role": "user", "content": "Describe the color orange."}],
                                  add_generation_prompt=True, enable_thinking=False, tokenize=True)
    ids = list(enc["input_ids"] if hasattr(enc, "keys") else enc)
    llm = LLM(model=a.model, dtype="bfloat16", enable_steering=True, max_steering_configs=a.max_configs,
              max_num_seqs=a.max_num_seqs, max_model_len=len(ids) + a.tokens + 16, gpu_memory_utilization=a.gpu_mem,
              limit_mm_per_prompt={"image": 0, "video": 0},
              capture_consumers=[{"name": "filesystem", "params": {"root": "/tmp/bench-captures"}}])
    prompts = [TokensPrompt(prompt_token_ids=ids)] * a.n
    base = {"temperature": 0.0, "max_tokens": a.tokens, "ignore_eos": True}

    def steer(i):
        n = i % len(abs_max)
        return {"steering_vectors": {"mlp_out": {layer: {"vector": w_down[n].astype(np.float32).tolist(),
                                                         "scale": float(0.5 * abs_max[n])}}}}

    def cap(i):
        return {"capture": {"filesystem": {"request_id": f"b{i}", "tag": "bench", "layout": "packed",
                                           "hooks": {"mlp_in": [layer, layer + 1, layer + 4, layer + 8]}, "positions": "last_prompt"}}}

    conditions = {
        "unsteered": lambda i: SamplingParams(**base),
        "steered": lambda i: SamplingParams(**base, **steer(i)),
        "steered+capture": lambda i: SamplingParams(**base, **steer(i), **cap(i)),
        "steered+logprobs5": lambda i: SamplingParams(**base, **steer(i), logprobs=5),
        # the sweep's real configuration, with natural stopping vs fixed length
        "sweep-varlen": lambda i: SamplingParams(**{**base, "ignore_eos": False}, **steer(i), **cap(i), logprobs=5),
        "sweep-fixed": lambda i: SamplingParams(**base, **steer(i), **cap(i), logprobs=5),
    }
    llm.generate(prompts[:32], [SamplingParams(**base)] * 32, use_tqdm=False)  # warm-up
    for name, make in conditions.items():
        if a.only and name not in a.only.split(","):
            continue
        t0 = time.time()
        params = [make(i) for i in range(a.n)]
        t1 = time.time()
        outs = llm.generate(prompts, params, use_tqdm=False)
        dt = time.time() - t1
        n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
        eos = tok.convert_tokens_to_ids("<|im_end|>")
        kept = sum((list(o.outputs[0].token_ids) + [eos]).index(eos) for o in outs)
        print(f"{name:18s} {n_tok / dt:8,.0f} gen tok/s  {kept / dt:8,.0f} kept tok/s  {a.n / dt:6.1f} rollouts/s  "
              f"(building params {t1 - t0:.1f}s)",
              flush=True)
    llm.llm_engine.engine_core.shutdown(timeout=60)


if __name__ == "__main__":
    main()
