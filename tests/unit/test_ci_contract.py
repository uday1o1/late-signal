from __future__ import annotations

import re
from pathlib import Path

from latesignal.contracts.protocol import load_final_protocol


def test_external_actions_are_commit_pinned_and_cpu_ci_never_fetches_data() -> None:
    workflow_root = Path(".github/workflows")
    workflows = [path.read_text(encoding="utf-8") for path in sorted(workflow_root.glob("*.yml"))]
    uses = [match for text in workflows for match in re.findall(r"uses:\s+[^@\s]+@([^\s#]+)", text)]

    assert uses
    assert all(re.fullmatch(r"[0-9a-f]{40}", reference) for reference in uses)
    cpu = (workflow_root / "ci.yml").read_text(encoding="utf-8")
    assert "latesignal data fetch" not in cpu
    assert "uv sync --frozen --all-groups" in cpu
    assert "uv run ruff format --check ." in cpu
    assert "uv run mypy" in cpu
    assert "uv run pytest" in cpu


def test_gpu_workflow_is_manual_bounded_and_uses_an_exact_cleanup_target() -> None:
    workflow = Path(".github/workflows/gpu-smoke.yml").read_text(encoding="utf-8")

    trigger = workflow.split("permissions:", maxsplit=1)[0]
    assert "workflow_dispatch:" in trigger
    assert "pull_request:" not in trigger
    assert "nvidia-gpu" in workflow
    assert "protocol validate configs/experiments/gpu_smoke.yaml" in workflow
    assert 'test "$RUN_ROOT" = "$expected"' in workflow
    assert 'rm -rf -- "$RUN_ROOT"' in workflow

    final, protocol, _ = load_final_protocol(Path("configs/experiments/gpu_smoke.yaml"))
    assert final.target_device == "cuda"
    assert final.require_real_pilot is False
    assert final.caps.max_runs == 89
    assert protocol.final_training.seeds == [17, 41, 73]
