"""Prompt and completion encoding for the cognition23 Huginn protocol."""

from __future__ import annotations

from typing import Any, Protocol

ANSWER_CUE = "\nAnswer:"


class TokenEncoder(Protocol):
    bos_token_id: int

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode ``text`` without inserting extra special tokens."""


def prompt_uses_native_bos(prompt_variant: str) -> bool:
    return "native_bos" in prompt_variant


def prompt_uses_answer_cue(prompt_variant: str) -> bool:
    return "answer_cue" in prompt_variant


def apply_prompt_text(prompt: str, prompt_variant: str) -> str:
    """Apply the documented text cue. Leading completion spaces are not touched."""
    if prompt_uses_answer_cue(prompt_variant):
        return f"{prompt}{ANSWER_CUE}"
    return prompt


def encode_condition(
    tokenizer: TokenEncoder,
    prompt: str,
    completion: str,
    prompt_variant: str,
) -> dict[str, Any]:
    """Tokenize one prompt/completion pair.

    ``answer_cue`` appends a newline and ``Answer:`` before tokenization.
    ``native_bos`` prepends the tokenizer BOS id after tokenization. Completion
    text, including a leading space, is encoded exactly as stored.
    """
    prompt_text = apply_prompt_text(prompt, prompt_variant)
    prompt_ids = [int(token) for token in tokenizer.encode(prompt_text, add_special_tokens=False)]
    completion_ids = [
        int(token)
        for token in tokenizer.encode(completion, add_special_tokens=False)
    ]
    native_bos = prompt_uses_native_bos(prompt_variant)
    raw_bos_token_id = tokenizer.bos_token_id if native_bos else None
    bos_token_id = int(raw_bos_token_id) if raw_bos_token_id is not None else None
    if native_bos:
        if bos_token_id is None:
            raise ValueError("tokenizer has no bos_token_id for native_bos")
        prompt_ids = [bos_token_id, *prompt_ids]
    if not prompt_ids:
        raise ValueError("prompt tokenization is empty")
    if not completion_ids:
        raise ValueError("completion tokenization is empty")
    return {
        "prompt_text": prompt_text,
        "completion_text": completion,
        "prompt_token_ids": prompt_ids,
        "completion_token_ids": completion_ids,
        "token_ids": [*prompt_ids, *completion_ids],
        "completion_start": len(prompt_ids),
        "prompt_protocol": {
            "prompt_variant": prompt_variant,
            "native_bos": native_bos,
            "answer_cue": prompt_uses_answer_cue(prompt_variant),
            "answer_cue_text": ANSWER_CUE if prompt_uses_answer_cue(prompt_variant) else None,
            "bos_token_id": bos_token_id,
            "tokenization": "separate_context_continuation",
        },
    }
