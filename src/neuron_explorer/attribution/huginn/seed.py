"""Deterministic recurrent-state seeds derived from dataset row ids."""

from __future__ import annotations

import hashlib


def row_seed(row_id: str) -> int:
    """Return a stable 31-bit seed for ``row_id``.

    The same id always yields the same seed. All four condition graphs of a
    row use this seed so their recurrent states are drawn from one stream.
    """
    digest = hashlib.sha256(row_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF
