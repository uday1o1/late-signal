from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from latesignal.cli import app
from latesignal.security.repository import audit_repository

runner = CliRunner()


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "audit@example.invalid")
    _git(root, "config", "user.name", "Audit Fixture")
    source = root / "src" / "package" / "main.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    readme = root / "README.md"
    readme.write_text("# Safe fixture\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "--quiet", "-m", "test: initialize fixture")
    return root


def test_repository_audit_passes_a_clean_source_tree(tmp_path: Path) -> None:
    root = _repository(tmp_path)

    result = audit_repository(root / "src")

    assert result["status"] == "passed"
    assert result["findings"] == []


def test_repository_audit_reports_restricted_secret_placeholder_and_untracked_source(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    restricted = root / "data" / "raw.tsv"
    restricted.parent.mkdir()
    restricted.write_text("restricted\n", encoding="utf-8")
    source = root / "src" / "package" / "main.py"
    source.write_text(
        "# TO" + "DO remove\nTOKEN = 'ghp_" + "abcdefghijklmnopqrstuvwxyz1234567890'\n",
        encoding="utf-8",
    )
    new_source = root / "tests" / "test_missing.py"
    new_source.parent.mkdir()
    new_source.write_text("def test_missing(): pass\n", encoding="utf-8")
    _git(root, "add", "data/raw.tsv", "src/package/main.py")

    result = audit_repository(root)

    assert result["status"] == "failed"
    assert {item["code"] for item in result["findings"]} == {
        "RESTRICTED_ROOT_TRACKED",
        "SECRET_SIGNATURE",
        "SOURCE_PLACEHOLDER",
        "UNTRACKED_REQUIRED_SOURCE",
    }


def test_repository_audit_rejects_tracked_symlinks_and_extensionless_secrets(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    target = root / "README.md"
    link = root / "linked-readme"
    link.symlink_to(target)
    environment = root / ".env"
    environment.write_text(
        "AWS_ACCESS_KEY_ID=AK" + "IAABCDEFGHIJKLMNOP\n",
        encoding="utf-8",
    )
    _git(root, "add", "linked-readme", ".env")

    result = audit_repository(root)

    assert {item["code"] for item in result["findings"]} == {
        "SECRET_SIGNATURE",
        "TRACKED_SYMLINK",
    }


def test_audit_cli_renders_findings_without_crashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository(tmp_path)
    untracked = root / "tests" / "test_untracked.py"
    untracked.parent.mkdir()
    untracked.write_text("def test_untracked(): pass\n", encoding="utf-8")
    monkeypatch.chdir(root)

    result = runner.invoke(app, ["audit"])

    assert result.exit_code == 5
    assert "UNTRACKED_REQUIRED_SOURCE" in result.stdout
    assert "Traceback" not in result.stdout
