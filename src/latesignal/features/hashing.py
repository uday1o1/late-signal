"""Deterministic field-specific categorical hashing and click identity."""

from __future__ import annotations

import xxhash
from blake3 import blake3


def click_id(raw_file_sha256: str, raw_row_index: int) -> str:
    """Derive an immutable ID from source identity and zero-based raw row index."""

    return blake3(f"{raw_file_sha256}:{raw_row_index}".encode()).hexdigest()


def categorical_bucket(field_name: str, raw_value: str, seed: int, bucket_count: int) -> int:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    value = f"{field_name}:{raw_value}".encode()
    return xxhash.xxh64_intdigest(value, seed=seed) % bucket_count
