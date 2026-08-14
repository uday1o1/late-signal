from __future__ import annotations

from pathlib import Path

import pytest

from latesignal.errors import ConsistencyError
from latesignal.experiments.collection import (
    build_collection_manifest,
    verify_collection_manifest,
)


def _job(root: Path) -> Path:
    for relative in (
        "feasibility.json",
        "selection/selection-results.json",
        "protocol-lock.json",
        "quality-gate.json",
        "final/final-manifest.json",
        "final/aggregate/manifest.json",
        "final/aggregate/report/report.html",
        "final/aggregate/report/tables/compute.csv",
        "final/aggregate/bootstrap/run/block-3/replicates.npz",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"aggregate fixture: {relative}\n".encode())
    return root


def test_collection_manifest_seals_exact_aggregate_only_files(tmp_path: Path) -> None:
    root = _job(tmp_path / "job")
    (root / "selection-provenance.json").write_text("{}\n", encoding="utf-8")
    manifest = build_collection_manifest(root)

    assert manifest["status"] == "verified_aggregate_only"
    assert "selection-provenance.json" in {item["path"] for item in manifest["files"]}
    assert verify_collection_manifest(root, root / "collection-manifest.json") == manifest

    (root / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ConsistencyError, match="exact manifest"):
        verify_collection_manifest(root, root / "collection-manifest.json")


def test_collection_manifest_rejects_row_level_or_redirected_artifacts(tmp_path: Path) -> None:
    row_level = _job(tmp_path / "row-level")
    (row_level / "final" / "aggregate" / "primary-probabilities.npy").write_bytes(b"rows")
    with pytest.raises(ConsistencyError, match="prohibited artifact"):
        build_collection_manifest(row_level)

    redirected = _job(tmp_path / "redirected")
    target = tmp_path / "outside.json"
    target.write_text("{}\n", encoding="utf-8")
    (redirected / "final" / "aggregate" / "redirect.json").symlink_to(target)
    with pytest.raises(ConsistencyError, match="non-regular artifact"):
        build_collection_manifest(redirected)
