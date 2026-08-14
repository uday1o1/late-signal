from __future__ import annotations

import pytest

from latesignal.errors import ConsistencyError
from latesignal.experiments.cuda_device import require_selected_cuda_device


def test_selected_cuda_device_requires_the_exact_stable_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-exact")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)

    require_selected_cuda_device("GPU-exact")

    with pytest.raises(ConsistencyError, match="stable GPU UUID"):
        require_selected_cuda_device("GPU-other")


@pytest.mark.parametrize(
    ("available", "count"),
    [(False, 0), (True, 0), (True, 2)],
)
def test_selected_cuda_device_rejects_an_unusable_or_ambiguous_runtime(
    monkeypatch: pytest.MonkeyPatch,
    available: bool,
    count: int,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-exact")
    monkeypatch.setattr("torch.cuda.is_available", lambda: available)
    monkeypatch.setattr("torch.cuda.device_count", lambda: count)

    with pytest.raises(ConsistencyError, match="exactly one CUDA device"):
        require_selected_cuda_device("GPU-exact")
