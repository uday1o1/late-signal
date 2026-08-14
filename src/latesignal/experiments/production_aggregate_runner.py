"""Public production entry point for locked aggregate-only final analysis."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from latesignal.contracts.protocol import load_final_protocol
from latesignal.errors import ConsistencyError
from latesignal.experiments.cuda_device import require_selected_cuda_device
from latesignal.experiments.production_aggregate import aggregate_production_final
from latesignal.experiments.production_final import FinalPlanInputs, final_online_plans
from latesignal.experiments.protocol_lock import (
    verify_locked_final_runtime,
    verify_protocol_lock,
)
from latesignal.features.cache import FeaturePolicyName, build_feature_cache, runtime_feature_policy
from latesignal.features.policy import load_feature_policy
from latesignal.features.store import RuntimeFeatureStore
from latesignal.simulator.production_oracle import load_production_truth
from latesignal.training.reproducibility import configure_determinism


def run_production_aggregate(
    config_path: Path,
    *,
    protocol_lock_path: Path,
    data_manifest_path: Path,
    feature_config_path: Path,
    cache_root: Path,
    output_root: Path,
    quality_gate_path: Path,
    device_uuid: str,
    repository: Path,
) -> dict[str, Any]:
    """Verify final evidence and produce paired aggregate-only results."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require_selected_cuda_device(device_uuid)
    configure_determinism(20260813)
    final, protocol, protocol_sha256 = load_final_protocol(config_path)
    lock = verify_protocol_lock(protocol_lock_path)
    verify_locked_final_runtime(
        lock,
        final_config_path=config_path,
        data_manifest_path=data_manifest_path,
        repository=repository,
    )
    if final.target_device != "cuda" or lock.get("protocol_sha256") != protocol_sha256:
        raise ConsistencyError("Final aggregate inputs do not match the authored CUDA protocol")
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
    features = RuntimeFeatureStore(cache)
    truth = load_production_truth(data_manifest_path, features)
    return aggregate_production_final(
        inputs,
        output_root,
        features=features,
        truth=truth,
        prepared_manifest_path=data_manifest_path,
        quality_gate_path=quality_gate_path,
        device="cuda",
    )
