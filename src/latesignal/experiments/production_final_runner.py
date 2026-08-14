"""Public production entry point for the locked final online matrix."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from latesignal.contracts.protocol import load_final_protocol
from latesignal.data.manifests import canonical_json_bytes, read_json, write_json_atomic
from latesignal.errors import ConsistencyError
from latesignal.experiments.cuda_device import require_selected_cuda_device
from latesignal.experiments.final_coordinator import run_final_online_coordinator
from latesignal.experiments.final_executor import ProductionFinalExecutor
from latesignal.experiments.production_final import FinalPlanInputs, final_online_plans
from latesignal.experiments.production_offline import (
    ProductionOfflineExecutor,
    run_offline_references,
)
from latesignal.experiments.protocol_lock import (
    verify_locked_final_runtime,
    verify_protocol_lock,
)
from latesignal.features.cache import FeaturePolicyName, build_feature_cache, runtime_feature_policy
from latesignal.features.policy import load_feature_policy
from latesignal.features.store import RuntimeFeatureStore
from latesignal.scheduling.monitoring import monitoring_membership_mask
from latesignal.simulator.production_oracle import load_production_truth
from latesignal.training.reproducibility import capture_runtime_identity, configure_determinism


def run_production_final(
    config_path: Path,
    *,
    protocol_lock_path: Path,
    data_manifest_path: Path,
    feature_config_path: Path,
    cache_root: Path,
    output_root: Path,
    device_uuid: str,
    repository: Path,
) -> dict[str, Any]:
    """Execute or resume all 33 locked final online runs."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require_selected_cuda_device(device_uuid)
    configure_determinism(17)
    final, protocol, protocol_sha256 = load_final_protocol(config_path)
    lock = verify_protocol_lock(protocol_lock_path)
    verify_locked_final_runtime(
        lock,
        final_config_path=config_path,
        data_manifest_path=data_manifest_path,
        repository=repository,
    )
    if final.target_device != "cuda" or lock.get("protocol_sha256") != protocol_sha256:
        raise ConsistencyError("Final runner inputs do not match the authored CUDA protocol")
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
    plans = final_online_plans(inputs)
    selected_policy = plans[0].feature_policy
    if any(plan.feature_policy != selected_policy for plan in plans):
        raise ConsistencyError("Final online runs do not share one selected feature policy")
    cache = build_feature_cache(
        data_manifest_path,
        authored_policy=authored_policy,
        policy_name=selected_policy,
        storage_root=cache_root,
    )
    features = RuntimeFeatureStore(cache)
    if features.feature_policy_sha256 != feature_hashes[selected_policy]:
        raise ConsistencyError("Final runtime feature cache changed after plan expansion")
    truth = load_production_truth(data_manifest_path, features)
    monitoring_mask = monitoring_membership_mask(features.click_ids, 20260813)
    runtime_identity = capture_runtime_identity(repository)
    if (
        runtime_identity.get("git_dirty") is not False
        or runtime_identity.get("git_commit") != lock["git"]["commit"]
    ):
        raise ConsistencyError("Final runtime identity changed after protocol verification")
    executor = ProductionFinalExecutor(
        output_root=output_root,
        features=features,
        truth=truth,
        monitoring_mask=monitoring_mask,
        runtime_identity=runtime_identity,
        device_uuid=device_uuid,
    )
    online = run_final_online_coordinator(inputs, output_root, executor=executor)
    if online.get("status") != "complete" or online.get("completed_count") != 33:
        raise ConsistencyError("Production final online coordinator did not complete")
    offline = run_offline_references(
        inputs,
        output_root,
        executor=ProductionOfflineExecutor(
            output_root=output_root,
            features=features,
            truth=truth,
            monitoring_mask=monitoring_mask,
            runtime_identity=runtime_identity,
        ),
    )
    if offline.get("status") != "complete" or offline.get("completed_count") != 6:
        raise ConsistencyError("Production offline reference coordinator did not complete")
    payload: dict[str, object] = {
        "version": 1,
        "status": "complete",
        "online_runs": 33,
        "offline_runs": 6,
        "completed_count": 39,
        "online_manifest_sha256": online["manifest_sha256"],
        "offline_manifest_sha256": offline["manifest_sha256"],
        "protocol_lock_sha256": lock["lock_sha256"],
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    path = output_root / "final-manifest.json"
    if path.exists():
        stored = read_json(path)
        if stored != payload:
            raise ConsistencyError("Immutable combined final manifest changed")
    else:
        write_json_atomic(path, payload)
    return read_json(path)
