"""Agreement between capture_prefill.py output and the vLLM fork's own captures for the same run.

    python scripts/check_capture_prefill.py runs/mainrel-e2e.jsonl captures/mainrel-e2e
"""
import json
import os

import numpy as np


def run_capture_check(run, fork_dir, captures):
    from vllm.v1.capture.consumers.filesystem.reader import read_request

    name = os.path.basename(run).replace(".jsonl", "")
    ours = np.load(os.path.join(captures, f"{name}.mlp_in.npy")).astype(np.float32)
    layers = json.load(open(os.path.join(captures, f"{name}.mlp_in.json")))["capture_layers"]
    rows = [json.loads(line) for line in open(run)]
    errs, n = {layer: [] for layer in layers}, 0
    for i, row in enumerate(rows):
        directory = os.path.join(fork_dir, row["capture_id"])
        if not os.path.isdir(directory):
            continue
        entries = read_request(directory)
        for j, layer in enumerate(layers):
            arr = entries[(layer, "mlp_in")].array
            if arr.dtype == np.uint16:
                arr = (arr.astype(np.uint32) << 16).view(np.float32)
            ref = arr.reshape(-1, arr.shape[-1])[-1].astype(np.float32)
            errs[layer].append(
                np.linalg.norm(ours[i, j] - ref) / np.linalg.norm(ref)
            )
        n += 1
    print(f"{n} rollouts compared")
    for layer in layers:
        error = np.array(errs[layer])
        print(
            f"L{layer}: relative error median {np.median(error):.4f}  "
            f"p95 {np.percentile(error, 95):.4f}  max {error.max():.4f}"
        )
