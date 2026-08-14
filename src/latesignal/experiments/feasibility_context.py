"""Bind measured feasibility evidence to its exact data and GPU runtime."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from latesignal.data.manifests import (
    canonical_json_bytes,
    read_json,
    sha256_file,
    write_json_atomic,
)
from latesignal.data.prepared import verify_prepared_inventory
from latesignal.errors import ConsistencyError
from latesignal.training.reproducibility import capture_runtime_identity


def _current_context(
    repository: Path,
    data_manifest_path: Path,
    final_config_path: Path,
    device_uuid: str,
) -> dict[str, object]:
    runtime = capture_runtime_identity(repository)
    if runtime.get("git_dirty") is not False or runtime.get("git_commit") is None:
        raise ConsistencyError("Feasibility context requires a clean Git runtime")
    inventory = verify_prepared_inventory(data_manifest_path)
    config_sha256, _ = sha256_file(final_config_path)
    try:
        device_identity = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_uuid}",
                "--query-gpu=uuid,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ConsistencyError("Could not bind feasibility to the selected GPU") from error
    if not device_identity.startswith(f"{device_uuid},") or "\n" in device_identity:
        raise ConsistencyError("Feasibility GPU identity is ambiguous")
    return {
        "version": 2,
        "git_commit": runtime["git_commit"],
        "source_tree_sha256": runtime["source_tree_sha256"],
        "dependency_lock_sha256": runtime["dependency_lock_sha256"],
        "runtime_sha256": runtime["runtime_sha256"],
        "prepared_manifest_sha256": inventory.manifest_sha256,
        "final_config_sha256": config_sha256,
        "device_uuid": device_uuid,
        "device_identity": device_identity,
    }


def bind_feasibility_context(
    measured_path: Path,
    output_path: Path,
    *,
    repository: Path,
    data_manifest_path: Path,
    final_config_path: Path,
    device_uuid: str,
) -> dict[str, Any]:
    """Publish measured feasibility only after adding exact execution identities."""

    measured = read_json(measured_path)
    if measured.get("execution_context") is not None:
        raise ConsistencyError("Measured feasibility already has an execution context")
    context = _current_context(
        repository,
        data_manifest_path,
        final_config_path,
        device_uuid,
    )
    context["measured_payload_sha256"] = hashlib.sha256(canonical_json_bytes(measured)).hexdigest()
    context["context_sha256"] = hashlib.sha256(canonical_json_bytes(context)).hexdigest()
    payload = {**measured, "execution_context": context}
    write_json_atomic(output_path, payload)
    return payload


def verify_feasibility_context(
    path: Path,
    *,
    repository: Path,
    data_manifest_path: Path,
    final_config_path: Path,
    device_uuid: str,
) -> dict[str, Any]:
    """Require reusable feasibility evidence to match the current exact runtime."""

    value = read_json(path)
    stored = value.get("execution_context")
    if not isinstance(stored, dict):
        raise ConsistencyError("Stored feasibility has no execution context")
    expected_sha256 = stored.get("context_sha256")
    unsigned = {key: item for key, item in stored.items() if key != "context_sha256"}
    measured_sha256 = unsigned.pop("measured_payload_sha256", None)
    measured = {key: item for key, item in value.items() if key != "execution_context"}
    current = _current_context(
        repository,
        data_manifest_path,
        final_config_path,
        device_uuid,
    )
    if (
        not isinstance(expected_sha256, str)
        or hashlib.sha256(
            canonical_json_bytes({**unsigned, "measured_payload_sha256": measured_sha256})
        ).hexdigest()
        != expected_sha256
        or unsigned != current
    ):
        raise ConsistencyError("Stored feasibility belongs to a different execution context")
    if (
        not isinstance(measured_sha256, str)
        or hashlib.sha256(canonical_json_bytes(measured)).hexdigest() != measured_sha256
    ):
        raise ConsistencyError("Stored feasibility payload failed its content seal")
    return value
