from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from typer.testing import CliRunner

from latesignal.cli import app
from latesignal.data.manifests import read_json
from latesignal.features.hashing import click_id

runner = CliRunner()
FEATURE_POLICY = Path("configs/features.yaml").resolve()


def test_public_prepare_partitions_features_and_truth_without_leakage(
    tmp_path: Path, trusted_config: Path
) -> None:
    data_root = tmp_path / "raw"
    processed = tmp_path / "processed"
    inspection = processed / "manifests" / "inspection.json"
    quarantine = processed / "quarantine" / "rejected.jsonl"
    fetch = runner.invoke(
        app,
        [
            "data",
            "fetch",
            "--accept-license",
            "--config",
            str(trusted_config),
            "--data-root",
            str(data_root),
            "--json",
        ],
    )
    assert fetch.exit_code == 0, fetch.stdout
    inspect_result = runner.invoke(
        app,
        [
            "data",
            "inspect",
            "--config",
            str(trusted_config),
            "--data-root",
            str(data_root),
            "--out",
            str(inspection),
            "--quarantine",
            str(quarantine),
            "--json",
        ],
    )
    assert inspect_result.exit_code == 0, inspect_result.stdout

    prepared = runner.invoke(
        app,
        [
            "data",
            "prepare",
            "--config",
            str(trusted_config),
            "--features",
            str(FEATURE_POLICY),
            "--data-root",
            str(data_root),
            "--inspection",
            str(inspection),
            "--out",
            str(processed),
            "--batch-rows",
            "2",
            "--json",
        ],
    )

    assert prepared.exit_code == 0, prepared.stdout
    payload = json.loads(prepared.stdout)
    assert payload["rows"] == {
        "features": 3,
        "inspection_accepted": 3,
        "inspection_quarantined": 0,
        "quarantine": 0,
        "reconciled": True,
        "truth": 3,
    }
    assert payload["streaming"]["source_materialized_in_memory"] is False
    feature_files = sorted((processed / "features").rglob("*.parquet"))
    reveal_files = sorted((processed / "truth" / "reveal").rglob("*.parquet"))
    maturity_files = sorted((processed / "truth" / "maturity").rglob("*.parquet"))
    assert feature_files and reveal_files and maturity_files

    features = pl.read_parquet(feature_files)
    truth = pl.read_parquet([*reveal_files, *maturity_files])
    assert features.height == 3
    assert truth.height == 3
    assert {
        "Sale",
        "SalesAmountInEuro",
        "time_delay_for_conversion",
        "available_at_seconds",
        "final_label",
    }.isdisjoint(features.columns)
    assert set(truth.columns) == {
        "click_id",
        "final_label",
        "click_time_seconds",
        "available_at_seconds",
    }
    assert set(features["click_id"]) == set(truth["click_id"])

    inspection_manifest = read_json(inspection)
    expected_first = click_id(inspection_manifest["extracted_data_member"]["sha256"], 0)
    assert features.sort("click_time_seconds")["click_id"][0] == expected_first
    assert features.sort("click_time_seconds")["click_day"].to_list() == [0, 44, 89]
    preparation = read_json(processed / "manifests" / "preparation.json")
    assert preparation["numeric_statistics"]["fit_click_days"] == [0, 14]
    assert (
        len(preparation["files"])
        == len(feature_files) + len(reveal_files) + len(maturity_files) + 1
    )
    feature_metadata = pq.read_schema(feature_files[0]).metadata
    assert feature_metadata is not None
    assert feature_metadata[b"latesignal_store"] == b"click_time_features"


def test_prepare_refuses_to_overwrite_published_stores(
    tmp_path: Path, trusted_config: Path
) -> None:
    data_root = tmp_path / "raw"
    processed = tmp_path / "processed"
    inspection = processed / "manifests" / "inspection.json"
    quarantine = processed / "quarantine" / "rejected.jsonl"
    assert (
        runner.invoke(
            app,
            [
                "data",
                "fetch",
                "--accept-license",
                "--config",
                str(trusted_config),
                "--data-root",
                str(data_root),
            ],
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            app,
            [
                "data",
                "inspect",
                "--config",
                str(trusted_config),
                "--data-root",
                str(data_root),
                "--out",
                str(inspection),
                "--quarantine",
                str(quarantine),
            ],
        ).exit_code
        == 0
    )
    command = [
        "data",
        "prepare",
        "--config",
        str(trusted_config),
        "--features",
        str(FEATURE_POLICY),
        "--data-root",
        str(data_root),
        "--inspection",
        str(inspection),
        "--out",
        str(processed),
        "--json",
    ]
    assert runner.invoke(app, command).exit_code == 0

    repeated = runner.invoke(app, command)

    assert repeated.exit_code == 5
    assert json.loads(repeated.stdout)["error"] == "INTERNAL_CONSISTENCY_FAILURE"


def test_prepare_external_sorts_an_unsorted_source(
    tmp_path: Path, trusted_unsorted_config: Path
) -> None:
    data_root = tmp_path / "raw"
    processed = tmp_path / "processed"
    inspection = processed / "manifests" / "inspection.json"
    quarantine = processed / "quarantine" / "rejected.jsonl"
    assert (
        runner.invoke(
            app,
            [
                "data",
                "fetch",
                "--accept-license",
                "--config",
                str(trusted_unsorted_config),
                "--data-root",
                str(data_root),
            ],
        ).exit_code
        == 0
    )
    inspected = runner.invoke(
        app,
        [
            "data",
            "inspect",
            "--config",
            str(trusted_unsorted_config),
            "--data-root",
            str(data_root),
            "--out",
            str(inspection),
            "--quarantine",
            str(quarantine),
            "--json",
        ],
    )
    assert inspected.exit_code == 0, inspected.stdout
    assert read_json(inspection)["click_time"]["monotonic"] is False

    prepared = runner.invoke(
        app,
        [
            "data",
            "prepare",
            "--config",
            str(trusted_unsorted_config),
            "--features",
            str(FEATURE_POLICY),
            "--data-root",
            str(data_root),
            "--inspection",
            str(inspection),
            "--out",
            str(processed),
            "--batch-rows",
            "2",
            "--json",
        ],
    )

    assert prepared.exit_code == 0, prepared.stdout
    features = pl.read_parquet(sorted((processed / "features").rglob("*.parquet"))).sort(
        "click_time_seconds"
    )
    assert features["click_time_seconds"].to_list() == [
        float(100),
        float(100 + 44 * 86_400),
        float(100 + 89 * 86_400),
    ]
    preparation = read_json(processed / "manifests" / "preparation.json")
    streaming = preparation["streaming"]
    assert streaming["source_click_time_monotonic"] is False
    assert streaming["chronological_sort"] == {
        "applied": True,
        "engine": "streaming",
        "keys": ["click_timestamp", "raw_row_index"],
    }
    inventory_paths = [item["path"] for item in preparation["files"]]
    assert "accepted-chronological.parquet" not in inventory_paths
    assert all((processed / path).is_file() for path in inventory_paths)
