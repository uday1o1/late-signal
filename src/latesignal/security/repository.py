"""Fail-closed audit for tracked secrets, restricted data, and release hygiene."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from latesignal.errors import ConfigurationError

RESTRICTED_ROOTS = frozenset({"artifacts", "checkpoints", "data", "reports", "runs"})
GENERATED_SUFFIXES = frozenset(
    {".ckpt", ".ncu-rep", ".onnx", ".pt", ".pth", ".tar", ".tgz", ".zip"}
)
ALLOWED_RESULT_SUFFIXES = frozenset({".csv", ".json", ".md", ".parquet"})
MAX_RESULT_BYTES = 5 * 1024 * 1024
SECRET_PATTERNS = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
)
PLACEHOLDER_PATTERN = re.compile(r"\b(?:" + "FIX" + "ME|NotImplemented" + r"Error|TO[D]O)\b")


def _git(repository: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ConfigurationError("Repository audit could not read Git state") from error


def audit_repository(repository: Path) -> dict[str, Any]:
    requested_root = repository.resolve()
    root = Path(_git(requested_root, "rev-parse", "--show-toplevel").strip()).resolve()
    tracked = [Path(value) for value in _git(root, "ls-files").splitlines() if value]
    findings: list[dict[str, str]] = []

    def finding(code: str, path: Path, message: str) -> None:
        findings.append({"code": code, "path": str(path), "message": message})

    for relative in tracked:
        unresolved_path = root / relative
        if unresolved_path.is_symlink():
            finding("TRACKED_SYMLINK", relative, "Tracked symlinks are not release artifacts")
            continue
        path = unresolved_path.resolve()
        try:
            path.relative_to(root)
        except ValueError:
            finding("TRACKED_PATH_ESCAPE", relative, "Tracked path escapes the repository")
            continue
        if relative.parts and relative.parts[0] in RESTRICTED_ROOTS:
            finding("RESTRICTED_ROOT_TRACKED", relative, "Restricted or generated root is tracked")
        suffixes = set(relative.suffixes)
        if suffixes & GENERATED_SUFFIXES or relative.name.endswith(".tar.gz"):
            finding("GENERATED_ARTIFACT_TRACKED", relative, "Generated model or archive is tracked")
        if relative.parts[:1] == ("results",) and relative.name != "README.md":
            if relative.suffix not in ALLOWED_RESULT_SUFFIXES:
                finding("UNSAFE_RESULT_TYPE", relative, "Published result type is not allowed")
            elif path.stat().st_size > MAX_RESULT_BYTES:
                finding("OVERSIZED_RESULT", relative, "Published aggregate result exceeds 5 MiB")
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                finding("SECRET_SIGNATURE", relative, f"Matched high-confidence {name} signature")
        if (
            relative.parts
            and relative.parts[0] in {"src", "configs"}
            and PLACEHOLDER_PATTERN.search(text)
        ):
            finding("SOURCE_PLACEHOLDER", relative, "Product source contains a placeholder")

    status = _git(root, "status", "--porcelain", "--untracked-files=all").splitlines()
    required_roots = ("configs/", "docs/", "src/", "tests/")
    for line in status:
        if not line.startswith("?? "):
            continue
        untracked = line[3:]
        if untracked.startswith(required_roots):
            finding(
                "UNTRACKED_REQUIRED_SOURCE",
                Path(untracked),
                "Required source or documentation is untracked",
            )
    return {
        "manifest_version": 1,
        "status": "passed" if not findings else "failed",
        "tracked_files": len(tracked),
        "findings": sorted(findings, key=lambda item: (item["code"], item["path"])),
    }
