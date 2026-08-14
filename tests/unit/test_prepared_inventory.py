from __future__ import annotations

import os
from pathlib import Path

import pytest

from latesignal.data.manifests import sha256_file, write_json_atomic
from latesignal.data.prepared import verify_prepared_inventory
from latesignal.errors import ConsistencyError


def _inventory(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "processed"
    paths = {
        "feature-0": root / "features" / "click_day=000" / "part-0.parquet",
        "feature-1": root / "features" / "click_day=001" / "part-1.parquet",
        "reveal": root / "truth" / "reveal" / "reveal_day=001" / "part.parquet",
        "maturity": root / "truth" / "maturity" / "maturity_day=030" / "part.parquet",
    }
    files: list[dict[str, object]] = []
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        sha256, size = sha256_file(path)
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256,
                "bytes": size,
            }
        )
    manifest_path = root / "manifests" / "preparation.json"
    write_json_atomic(
        manifest_path,
        {
            "manifest_version": 1,
            "rows": {"reconciled": True},
            "numeric_statistics": {"fit_click_days": [0, 14]},
            "files": files,
        },
    )
    return manifest_path, paths


def test_inventory_exposes_only_manifest_selected_day_files(tmp_path: Path) -> None:
    manifest_path, paths = _inventory(tmp_path)

    inventory = verify_prepared_inventory(manifest_path)

    assert inventory.feature_files(first_day=1, last_day=1) == (paths["feature-1"],)
    assert inventory.truth_files("reveal", first_day=1, last_day=1) == (paths["reveal"],)
    assert inventory.truth_files("maturity", first_day=30, last_day=30) == (paths["maturity"],)
    assert inventory.content_addressed_root(tmp_path / "store") == (
        tmp_path / "store" / "sha256" / inventory.manifest_sha256
    )


def test_inventory_rejects_modified_and_unlisted_files(tmp_path: Path) -> None:
    manifest_path, paths = _inventory(tmp_path)
    paths["feature-0"].write_bytes(b"modified")

    with pytest.raises(ConsistencyError, match="identity does not match"):
        verify_prepared_inventory(manifest_path)

    manifest_path, _ = _inventory(tmp_path / "second")
    extra = manifest_path.parent.parent / "features" / "unexpected.parquet"
    extra.write_bytes(b"not-authored")
    with pytest.raises(ConsistencyError, match="unlisted files"):
        verify_prepared_inventory(manifest_path)


def test_inventory_rejects_escape_and_symlink_inputs(tmp_path: Path) -> None:
    manifest_path, _ = _inventory(tmp_path)
    root = manifest_path.parent.parent
    escape_manifest = root / "manifests" / "escape.json"
    write_json_atomic(
        escape_manifest,
        {
            "manifest_version": 1,
            "rows": {"reconciled": True},
            "numeric_statistics": {"fit_click_days": [0, 14]},
            "files": [{"path": "../outside", "sha256": "0" * 64, "bytes": 0}],
        },
    )
    with pytest.raises(ConsistencyError, match="escapes"):
        verify_prepared_inventory(escape_manifest, reject_unlisted=False)

    if not hasattr(os, "symlink"):
        pytest.skip("symlinks are unavailable")
    symlink = root / "features" / "linked.parquet"
    symlink.symlink_to(root / "features" / "click_day=000" / "part-0.parquet")
    with pytest.raises(ConsistencyError, match="symlink"):
        verify_prepared_inventory(manifest_path)
