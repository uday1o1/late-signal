"""Reproducibility controls for the supported software stack."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


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
