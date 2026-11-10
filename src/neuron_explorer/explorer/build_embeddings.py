"""Embeds every rollout of a sweep run and lays them out in 2D with UMAP, for the explorer's embeddings view.

    uv run python -m neuron_explorer.explorer.build_embeddings runs/mainrel-e2e.jsonl

Writes <run>.emb.npy (normalized sentence embeddings, one row per jsonl line) and <run>.umap.json with parallel arrays:
x, y, row (jsonl line index), kind, neuron, prompt, dose.
"""
import json

import numpy as np


def run_build_embeddings(a):
    import umap
    from sentence_transformers import SentenceTransformer

    rows = [json.loads(l) for l in open(a.jsonl)]
    import torch
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    enc = SentenceTransformer(a.model, device=dev)
    emb = enc.encode([r["text"][: a.chars] for r in rows], batch_size=256, normalize_embeddings=True,
                     show_progress_bar=True, convert_to_numpy=True)
    np.save(a.jsonl.replace(".jsonl", ".emb.npy"), emb.astype(np.float16))
    xy = umap.UMAP(n_neighbors=a.neighbors, min_dist=0.08, metric="cosine", random_state=0).fit_transform(emb)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    out = {
        "x": np.round(xy[:, 0], 4).tolist(), "y": np.round(xy[:, 1], 4).tolist(), "row": list(range(len(rows))),
        "kind": [r["kind"] for r in rows], "neuron": [r["neuron"] for r in rows],
        "prompt": [r["prompt"] for r in rows], "dose": [r["dose"] for r in rows],
    }
    path = a.jsonl.replace(".jsonl", ".umap.json")
    json.dump(out, open(path, "w"))
    print(f"{len(rows)} rollouts -> {path}")
