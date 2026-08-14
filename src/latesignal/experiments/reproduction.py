"""Clean-checkout reproduction through the public synthetic experiment implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from latesignal.contracts.config import load_synthetic_config
from latesignal.contracts.reproduction import ReproductionManifest
from latesignal.data.manifests import write_json_atomic
from latesignal.errors import ConfigurationError, ConsistencyError
from latesignal.experiments.runner import run_synthetic_experiment
from latesignal.training.reproducibility import capture_runtime_identity


def _repository_path(repository: Path, relative: str) -> Path:
    candidate = (repository / relative).resolve()
    try:
        candidate.relative_to(repository.resolve())
    except ValueError as error:
        raise ConfigurationError("Reproduction path escapes the repository") from error
    if not candidate.is_file():
        raise ConfigurationError(f"Reproduction input does not exist: {relative}")
    return candidate


def reproduce_synthetic(
    manifest: ReproductionManifest,
    output_root: Path,
    *,
    repository: Path,
) -> dict[str, Any]:
    runtime = capture_runtime_identity(repository)
    identity_mismatches: dict[str, dict[str, str]] = {}
    for name, expected_identity, actual_identity in (
        (
            "source_tree_sha256",
            manifest.source_tree_sha256,
            str(runtime["source_tree_sha256"]),
        ),
        (
            "dependency_lock_sha256",
            manifest.dependency_lock_sha256,
            str(runtime["dependency_lock_sha256"]),
        ),
    ):
        if expected_identity != actual_identity:
            identity_mismatches[name] = {
                "expected": expected_identity,
                "actual": actual_identity,
            }
    if identity_mismatches:
        raise ConsistencyError(
            "Reproduction identity does not match the manifest",
            details={"mismatches": identity_mismatches},
        )
    config_path = _repository_path(repository, manifest.config)
    config = load_synthetic_config(config_path)
    if config.canonical_sha256 != manifest.config_sha256:
        raise ConsistencyError("Reproduction configuration does not match the manifest")
    run = run_synthetic_experiment(config, output_root)
    observed = {
        "ledger_sha256": run["ledger_sha256"],
        "counts": run["counts"],
        "metrics": run["metrics"],
    }
    expected = manifest.expected.model_dump(mode="json")
    status = "complete" if observed == expected else "mismatch"
    result: dict[str, Any] = {
        "manifest_version": 1,
        "status": status,
        "kind": manifest.kind,
        "config_sha256": config.canonical_sha256,
        "source_tree_sha256": runtime["source_tree_sha256"],
        "dependency_lock_sha256": runtime["dependency_lock_sha256"],
        "expected": expected,
        "observed": observed,
    }
    write_json_atomic(output_root / "reproduction.json", result)
    if status != "complete":
        raise ConsistencyError(
            "Reproduced output does not match the locked expected result",
            details={"result": str(output_root / "reproduction.json")},
        )
    return result
