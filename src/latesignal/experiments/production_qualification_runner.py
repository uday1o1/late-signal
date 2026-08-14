"""Public entry point for the locked hard pre-scoring quality gate."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from latesignal.contracts.protocol import load_final_protocol
from latesignal.errors import ConsistencyError
from latesignal.experiments.production_final import FinalPlanInputs, final_online_plans
from latesignal.experiments.production_final_runner import _require_selected_cuda_device
from latesignal.experiments.production_qualification import run_production_qualification
from latesignal.experiments.protocol_lock import (
    verify_locked_final_runtime,
    verify_protocol_lock,
)
from latesignal.features.cache import FeaturePolicyName, build_feature_cache, runtime_feature_policy
from latesignal.features.policy import load_feature_policy
from latesignal.features.store import RuntimeFeatureStore


def run_final_qualification(
    config_path: Path,
    *,
    protocol_lock_path: Path,
    data_manifest_path: Path,
    feature_config_path: Path,
    cache_root: Path,
    output_path: Path,
    device_uuid: str,
    repository: Path,
) -> dict[str, Any]:
    """Run the complete locked gate and real-schema CUDA resume rehearsal."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _require_selected_cuda_device(device_uuid)
    final, protocol, protocol_sha256 = load_final_protocol(config_path)
    lock = verify_protocol_lock(protocol_lock_path)
    verify_locked_final_runtime(
        lock,
        final_config_path=config_path,
        data_manifest_path=data_manifest_path,
        repository=repository,
    )
    if final.target_device != "cuda" or lock.get("protocol_sha256") != protocol_sha256:
        raise ConsistencyError("Final qualification does not match the authored CUDA protocol")
    authored_policy = load_feature_policy(feature_config_path)
    policy_names: tuple[FeaturePolicyName, ...] = ("compact", "large")
    feature_hashes: dict[FeaturePolicyName, str] = {
        name: runtime_feature_policy(authored_policy, name).canonical_sha256
        for name in policy_names
    }
    inputs = FinalPlanInputs(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        protocol_lock=lock,
        feature_policy_sha256=feature_hashes,
    )
    selected_policy = final_online_plans(inputs)[0].feature_policy
    cache = build_feature_cache(
        data_manifest_path,
        authored_policy=authored_policy,
        policy_name=selected_policy,
        storage_root=cache_root,
    )
    features = RuntimeFeatureStore(cache, build_id_lookup=False)
    return run_production_qualification(
        features,
        protocol_sha256=protocol_sha256,
        protocol_lock=lock,
        output_path=output_path,
        repository=repository,
        device_uuid=device_uuid,
        feature_policy=selected_policy,
    )
