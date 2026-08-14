"""Public production entry point for the complete staged selection study."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from latesignal.contracts.protocol import load_final_protocol
from latesignal.data.manifests import read_json
from latesignal.errors import ConsistencyError
from latesignal.experiments.cuda_device import require_selected_cuda_device
from latesignal.experiments.selection_coordinator import run_selection_coordinator
from latesignal.experiments.selection_dag import SelectionPlanInputs
from latesignal.experiments.selection_executor import ProductionSelectionExecutor
from latesignal.features.cache import FeaturePolicyName, build_feature_cache
from latesignal.features.policy import load_feature_policy
from latesignal.features.store import RuntimeFeatureStore
from latesignal.scheduling.monitoring import monitoring_membership_mask
from latesignal.simulator.production_oracle import load_production_truth
from latesignal.training.reproducibility import capture_runtime_identity, configure_determinism


def run_production_selection(
    config_path: Path,
    *,
    data_manifest_path: Path,
    feature_config_path: Path,
    cache_root: Path,
    output_root: Path,
    steps_per_credit: int,
    device_uuid: str,
    repository: Path,
) -> dict[str, Any]:
    """Execute all 50 frozen selection candidates through the real-data path."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require_selected_cuda_device(device_uuid)
    configure_determinism(17)
    final, protocol, protocol_sha256 = load_final_protocol(config_path)
    if (
        final.target_device != "cuda"
        or steps_per_credit not in protocol.final_training.steps_per_credit_candidates
    ):
        raise ConsistencyError("Production selection requires an authorized CUDA training budget")
    authored_policy = load_feature_policy(feature_config_path)
    policy_names: tuple[FeaturePolicyName, ...] = ("compact", "large")
    caches = {
        name: build_feature_cache(
            data_manifest_path,
            authored_policy=authored_policy,
            policy_name=name,
            storage_root=cache_root,
        )
        for name in policy_names
    }
    stores: dict[FeaturePolicyName, RuntimeFeatureStore] = {
        name: RuntimeFeatureStore(cache) for name, cache in caches.items()
    }
    compact = stores["compact"]
    truth = load_production_truth(data_manifest_path, compact)
    monitoring_mask = monitoring_membership_mask(compact.click_ids, 20260813)
    runtime_identity = capture_runtime_identity(repository)
    inputs = SelectionPlanInputs(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        data_manifest_sha256=compact.prepared_manifest_sha256,
        feature_policy_sha256={name: store.feature_policy_sha256 for name, store in stores.items()},
        steps_per_credit=steps_per_credit,
        device="cuda",
    )
    executor = ProductionSelectionExecutor(
        output_root=output_root,
        features=stores,
        truth=truth,
        monitoring_mask=monitoring_mask,
        runtime_identity=runtime_identity,
        device_uuid=device_uuid,
    )
    result = run_selection_coordinator(inputs, output_root, executor=executor)
    manifest = read_json(output_root / "manifest.json")
    if manifest.get("status") != "complete" or result.protocol_sha256 != protocol_sha256:
        raise ConsistencyError("Production selection coordinator did not complete")
    return manifest
