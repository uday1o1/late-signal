"""Stable application errors and public exit codes."""

from __future__ import annotations

from enum import IntEnum
from typing import Any


class ExitCode(IntEnum):
    """Process exit codes promised by the public CLI."""

    SUCCESS = 0
    GATE_NOT_MET = 1
    INVALID_CONFIGURATION = 2
    INVALID_DATA = 3
    INFRASTRUCTURE_FAILURE = 4
    CONSISTENCY_FAILURE = 5


class LateSignalError(Exception):
    """Base class for expected failures that have a stable public code."""

    exit_code = ExitCode.INFRASTRUCTURE_FAILURE
    error_code = "LATESIGNAL_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(LateSignalError):
    exit_code = ExitCode.INVALID_CONFIGURATION
    error_code = "INVALID_CONFIGURATION"


class DataArtifactError(LateSignalError):
    exit_code = ExitCode.INVALID_DATA
    error_code = "INVALID_DATA_ARTIFACT"


class LicenseNotAcceptedError(DataArtifactError):
    error_code = "LICENSE_NOT_ACCEPTED"


class FirstDownloadReviewRequired(DataArtifactError):
    error_code = "FIRST_DOWNLOAD_REVIEW_REQUIRED"


class AmbiguousTimeUnitError(DataArtifactError):
    error_code = "AMBIGUOUS_TIME_UNIT"


class ConsistencyError(LateSignalError):
    exit_code = ExitCode.CONSISTENCY_FAILURE
    error_code = "INTERNAL_CONSISTENCY_FAILURE"
