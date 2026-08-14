"""Contract tests for the bounded remote GPU feasibility helper."""

import subprocess
from pathlib import Path

SCRIPT = Path("tools/run-gpu-feasibility.sh").resolve()


def test_remote_gpu_script_help_is_safe_and_bounded() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "SSH_HOST [GPU_INDEX]" in result.stdout
    assert "prepared dataset" in result.stdout
    assert "does not start" in result.stdout
    assert "source archive" in result.stdout
