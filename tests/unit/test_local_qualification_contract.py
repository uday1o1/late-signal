from __future__ import annotations

from pathlib import Path

from latesignal.contracts.protocol import load_final_protocol


def test_local_gpu_smoke_preserves_the_locked_matrix() -> None:
    final, protocol, _ = load_final_protocol(Path("configs/experiments/gpu_smoke.yaml"))

    assert final.target_device == "cuda"
    assert final.require_real_pilot is False
    assert final.pilot.max_click_days == 2
    assert final.caps.max_runs == 89
    assert protocol.final_training.seeds == [17, 41, 73]


def test_final_qualification_preserves_resource_and_data_gates() -> None:
    final, protocol, _ = load_final_protocol(Path("configs/experiments/final.yaml"))

    assert final.target_device == "cuda"
    assert final.require_real_pilot is True
    assert final.pilot.max_click_days == 2
    assert final.caps.max_runs == 89
    assert final.caps.max_gpu_hours == 25.0
    assert final.caps.max_working_disk_gb == 26.0
    assert final.caps.max_retained_disk_gb == 2.0
    assert protocol.final_training.seeds == [17, 41, 73]
