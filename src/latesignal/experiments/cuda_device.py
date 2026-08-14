"""Fail-closed binding to one launcher-selected CUDA device."""

from __future__ import annotations

import os

import torch

from latesignal.errors import ConsistencyError


def require_selected_cuda_device(device_uuid: str) -> None:
    """Require a stable UUID to expose exactly one usable CUDA device."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        not device_uuid.startswith("GPU-")
        or visible != device_uuid
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ConsistencyError(
            "GPU execution requires exactly one CUDA device selected by its stable GPU UUID",
            details={"expected_cuda_visible_devices": device_uuid, "actual": visible},
        )
