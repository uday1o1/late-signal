"""Atomic JSON checkpoints with configuration and fixture identity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from latesignal.data.manifests import read_json, write_json_atomic
from latesignal.errors import ConsistencyError

CHECKPOINT_VERSION = 1


def write_checkpoint(path: Path, payload: dict[str, object]) -> None:
    document = {"checkpoint_version": CHECKPOINT_VERSION, **payload}
    write_json_atomic(path, document)


def read_checkpoint(path: Path) -> dict[str, Any]:
    document = read_json(path)
    if document.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ConsistencyError("Unsupported checkpoint version")
    return document
