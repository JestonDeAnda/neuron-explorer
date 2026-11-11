"""Load Raven classes and construct a tiny random model for tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def model_revision(model_path: Path) -> str:
    """Short identity of the checkpoint config. Weights stay at ``model_path``."""
    digest = hashlib.sha256((Path(model_path) / "config.json").read_bytes()).hexdigest()
    return f"config-sha256:{digest[:16]}"


def _repair_tied_weights(model_cls) -> None:
    """Current Transformers expects ``_tied_weights_keys`` to be a mapping.

    The Huginn checkpoint still publishes the older list of tied parameter
    names. Patch the loaded class in memory; the checkpoint files stay intact.
    """
    mapping = {"lm_head.weight": "transformer.wte.weight"}
    for cls in model_cls.mro():
        current = cls.__dict__.get("_tied_weights_keys")
        if isinstance(current, list):
            cls._tied_weights_keys = mapping


def load_raven_classes(model_path: Path):
    """Import RavenConfig and RavenForCausalLM from a local trust_remote_code dir."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    probe = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_cls = get_class_from_dynamic_module(
        probe.auto_map["AutoModelForCausalLM"], str(model_path)
    )
    _repair_tied_weights(model_cls)
    return type(probe), model_cls


def tiny_raven(model_path: Path, *, recurrence: int = 2, seed: int = 0):
    """Random Raven with Huginn's block counts and a short recurrence."""
    config_cls, model_cls = load_raven_classes(model_path)
    config = config_cls(
        n_embd=32,
        n_heads=4,
        n_layers=8,
        vocab_size=128,
        intermediate_size=16,
        n_layers_in_prelude=2,
        n_layers_in_recurrent_block=4,
        n_layers_in_coda=2,
        mean_recurrence=recurrence,
        block_size=64,
        tie_embeddings=True,
        qk_bias=True,
        torch_dtype="float32",
        rope_base=10000.0,
    )
    torch.manual_seed(seed)
    model = model_cls(config).float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_huginn(model_path: Path, device: str):
    """Load the float32 checkpoint and tokenizer onto ``device``.

    The checkpoint class still publishes ``_tied_weights_keys`` as a list.
    Repair that class before ``from_pretrained`` so a fresh process can tie
    ``lm_head`` without editing the checkpoint files.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    _config_cls, model_cls = load_raven_classes(model_path)
    model = model_cls.from_pretrained(str(model_path), torch_dtype=torch.float32)
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model
