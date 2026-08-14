"""Reproducibility controls for the supported software stack."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from latesignal.data.manifests import sha256_file
from latesignal.errors import ConsistencyError

RUNTIME_PACKAGES = (
    "numpy",
    "polars",
    "pyarrow",
    "pydantic",
    "scikit-learn",
    "torch",
    "typer",
)


def capture_runtime_identity(repository: Path | None = None) -> dict[str, Any]:
    """Hash source, dependencies, Git state, and the supported runtime stack."""

    root = repository.resolve() if repository is not None else Path(__file__).resolve().parents[3]
    source_root = root / "src" / "latesignal"
    digest = hashlib.sha256()
    source_files = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and (path.suffix == ".py" or path.name == "py.typed")
    )
    if not source_files:
        raise ConsistencyError("Runtime identity found no LateSignal source files")
    for path in source_files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    dependency_sha256, dependency_bytes = sha256_file(root / "uv.lock")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_paths = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ConsistencyError("Could not capture Git identity for reproducibility") from error
    identity: dict[str, Any] = {
        "source_tree_sha256": digest.hexdigest(),
        "source_file_count": len(source_files),
        "dependency_lock_sha256": dependency_sha256,
        "dependency_lock_bytes": dependency_bytes,
        "git_commit": commit,
        "git_dirty": bool(dirty_paths),
        "git_dirty_paths": dirty_paths,
        "python": sys.version.split()[0],
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES},
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    identity["runtime_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return identity


def configure_determinism(seed: int) -> None:
    """Configure deterministic behavior where PyTorch supports it."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():  # type: ignore[no-untyped-call]
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
