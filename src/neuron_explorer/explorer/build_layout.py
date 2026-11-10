"""Seriated sheet layouts: places each layer's neurons on its 96x96 grid so that neurons with similar weights sit
together. Embeds each neuron by its down_proj column ("writes": the direction it adds to the residual stream) or its
gate and up rows ("reads"), cosine-normalised, PCA to 50 dims, UMAP to 2D, then snaps to the grid: sort by y into
rows of 96, then by x within each row.

    uv run python -m neuron_explorer.explorer.build_layout --kind writes

Writes cache/layout_<kind>.npz: cell[L, N] (int16, row * 96 + col for neuron n of layer L).
"""
import json
import os

import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open

def run_build_layout(options):
    import umap

    root = snapshot_download(options.model, allow_patterns=["*.safetensors", "*.json"])
    index = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(root, "config.json")))["text_config"]
    L, N = cfg["num_hidden_layers"], cfg["intermediate_size"]
    side = int(round(N ** 0.5))
    assert side * side == N


    def tensor(name):
        with safe_open(os.path.join(root, index[name]), "pt") as f:
            return f.get_tensor(name).float().numpy()


    cells = np.zeros((L, N), np.int16)
    for l in range(L):
        pre = f"model.language_model.layers.{l}.mlp."
        if options.kind == "writes":
            X = tensor(pre + "down_proj.weight").T  # [N, hidden]
        else:
            X = np.concatenate([tensor(pre + "gate_proj.weight"), tensor(pre + "up_proj.weight")], 1)  # [N, 2*hidden]
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        X = X - X.mean(0)
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        Z = U[:, : options.pca] * S[: options.pca]
        xy = umap.UMAP(n_neighbors=options.neighbors, min_dist=0.05, metric="cosine", random_state=0).fit_transform(Z)
        by_y = np.argsort(xy[:, 1], kind="stable")
        for r in range(side):
            band = by_y[r * side:(r + 1) * side]
            band = band[np.argsort(xy[band, 0], kind="stable")]
            cells[l, band] = r * side + np.arange(side)
        print(f"L{l} done", flush=True)

    out = options.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, cell=cells)
    print("->", out)
