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


def test_remote_gpu_script_uses_existing_https_credentials_without_reading_them() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "credential.helper=store" in source
    assert "GIT_TERMINAL_PROMPT=0" in source
    assert "cat ~/.git-credentials" not in source
    assert "git@github.com:${origin_url#https://github.com/}" not in source


def test_remote_gpu_script_verifies_the_preparation_manifest() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'Path("data/processed/manifests/preparation.json")' in source
    assert '_verify_prepared_data(Path("data/processed"))' not in source


def test_remote_gpu_script_preserves_immutable_results_per_revision() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "gpu${GPU_INDEX}-${LOCAL_HEAD:0:12}.json" in source
    assert 'RESULT_RELATIVE="runs/feasibility/gpu${GPU_INDEX}.json"' not in source
