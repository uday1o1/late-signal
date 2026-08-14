"""Public sealed-run evaluation and paired comparison orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from latesignal.contracts.results import EvaluationDataset
from latesignal.data.manifests import read_json, write_json_atomic
from latesignal.errors import ConfigurationError, ConsistencyError
from latesignal.evaluation.compare import compare_methods
from latesignal.evaluation.metrics import classification_metrics
from latesignal.evaluation.slices import evaluate_slices


def read_evaluation_dataset(run_dir: Path) -> EvaluationDataset:
    path = run_dir / "evaluation-input.json"
    if not path.exists():
        raise ConfigurationError(f"Run has no sealed evaluation input: {path}")
    return EvaluationDataset.from_dict(read_json(path))


def evaluate_run_dir(run_dir: Path) -> dict[str, Any]:
    dataset = read_evaluation_dataset(run_dir)
    output = run_dir / "evaluation.json"
    if output.exists():
        raise ConsistencyError(f"Evaluation output already exists: {output}")
    overall = classification_metrics(
        [row.final_label for row in dataset.examples],
        [row.probability for row in dataset.examples],
    )
    slices = evaluate_slices(dataset.examples)
    result: dict[str, Any] = {
        "manifest_version": 1,
        "status": "complete",
        "method": dataset.method,
        "seed": dataset.seed,
        "sealed": dataset.sealed,
        "ranking_eligible": dataset.ranking_eligible,
        "period": [dataset.period_first_day, dataset.period_last_day],
        "overall": overall,
        "slices": [item.as_dict() for item in slices],
    }
    write_json_atomic(output, result)
    return result


def compare_run_dirs(
    run_dirs: tuple[Path, ...],
    *,
    control_method: str,
    candidate_method: str,
    replicates: int,
) -> dict[str, Any]:
    datasets = tuple(read_evaluation_dataset(path) for path in run_dirs)
    control = tuple(dataset for dataset in datasets if dataset.method == control_method)
    candidate = tuple(dataset for dataset in datasets if dataset.method == candidate_method)
    unexpected = sorted(
        {dataset.method for dataset in datasets} - {control_method, candidate_method}
    )
    if unexpected:
        raise ConsistencyError(
            "Comparison received unexpected methods", details={"methods": unexpected}
        )
    return compare_methods(control, candidate, replicates=replicates)
