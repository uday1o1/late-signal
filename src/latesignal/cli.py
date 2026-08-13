"""Public LateSignal command-line interface."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from latesignal.contracts.config import load_synthetic_config
from latesignal.data.config import load_data_config
from latesignal.data.download import FetchNotice, fetch_dataset
from latesignal.data.inspect import inspect_locked_archive
from latesignal.data.prepare import prepare_data
from latesignal.errors import ExitCode, LateSignalError
from latesignal.experiments.runner import resume_synthetic_experiment, run_synthetic_experiment
from latesignal.features.policy import load_feature_policy

app = typer.Typer(
    name="latesignal",
    help="Leakage-audited event-time delayed-conversion benchmark.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Acquire and audit the licensed source dataset.", no_args_is_help=True)
app.add_typer(data_app, name="data")

ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Authored data configuration.",
    ),
]
DataRootOption = Annotated[
    Path,
    typer.Option(
        "--data-root",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Ignored local root for licensed artifacts and acknowledgements.",
    ),
]
JsonOption = Annotated[
    bool,
    typer.Option("--json", help="Emit newline-delimited machine-readable JSON."),
]


@app.command("run")
def run_experiment(
    config: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Authored experiment configuration.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="New run output directory.",
        ),
    ],
    stop_after_checkpoints: Annotated[
        int | None,
        typer.Option(
            "--stop-after-checkpoints",
            hidden=True,
            help="Testing hook that simulates interruption after N checkpoints.",
        ),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """Run an authored experiment through the public event-time path."""

    try:
        authored = load_synthetic_config(config)
        manifest = run_synthetic_experiment(
            authored,
            out,
            stop_after_checkpoints=stop_after_checkpoints,
        )
    except LateSignalError as error:
        _fail(error, json_output)
    _emit(
        {
            "ok": manifest["status"] == "complete",
            "status": manifest["status"],
            "out": str(out),
            "manifest": str(out / "manifest.json"),
            "counts": manifest["counts"],
            "metrics": manifest["metrics"],
            "ledger_sha256": manifest["ledger_sha256"],
        },
        json_output,
    )
    if manifest["status"] != "complete":
        raise typer.Exit(code=int(ExitCode.INFRASTRUCTURE_FAILURE))


@app.command("resume")
def resume_experiment(
    checkpoint: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Synthetic checkpoint to resume.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="New resume output directory.",
        ),
    ],
    json_output: JsonOption = False,
) -> None:
    """Resume a synthetic run after validating its immutable identities."""

    try:
        manifest = resume_synthetic_experiment(checkpoint, out)
    except LateSignalError as error:
        _fail(error, json_output)
    _emit(
        {
            "ok": True,
            "status": manifest["status"],
            "out": str(out),
            "manifest": str(out / "manifest.json"),
            "counts": manifest["counts"],
            "metrics": manifest["metrics"],
            "ledger_sha256": manifest["ledger_sha256"],
        },
        json_output,
    )


def _emit(value: dict[str, Any], json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(value, sort_keys=True))
        return
    if value.get("ok") is False:
        typer.echo(f"ERROR {value['error']}: {value['message']}", err=True)
        details = value.get("details")
        if details:
            typer.echo(json.dumps(details, indent=2, sort_keys=True), err=True)
        return
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


def _fail(error: LateSignalError, json_output: bool) -> None:
    _emit(error.as_dict(), json_output)
    raise typer.Exit(code=int(error.exit_code))


@data_app.command("fetch")
def data_fetch(
    accept_license: Annotated[
        bool,
        typer.Option(
            "--accept-license",
            help="Affirm that you reviewed and accept the configured dataset license.",
        ),
    ] = False,
    review_sha256: Annotated[
        str | None,
        typer.Option(
            "--review-sha256",
            help="Explicitly trust the displayed first-download SHA-256 after review.",
        ),
    ] = None,
    config: ConfigOption = Path("configs/data.yaml"),
    data_root: DataRootOption = Path("data/raw"),
    json_output: JsonOption = False,
) -> None:
    """Download the reviewed archive through the explicit license boundary."""

    def notice_handler(notice: FetchNotice) -> None:
        payload = {"ok": True, "status": "license_notice", **asdict(notice)}
        if json_output:
            _emit(payload, True)
            return
        typer.echo(f"Dataset: {notice.dataset}")
        typer.echo(f"License: {notice.license_id}")
        typer.echo(f"Official page: {notice.official_page}")
        typer.echo(f"Archive URL: {notice.archive_url}")
        typer.echo(f"Destination: {notice.destination}")
        typer.echo(f"Restriction: {notice.restriction}")

    try:
        authored = load_data_config(config)
        result = fetch_dataset(
            authored,
            data_root,
            accept_license=accept_license,
            reviewed_sha256=review_sha256,
            notice_handler=notice_handler,
        )
    except LateSignalError as error:
        _fail(error, json_output)
    _emit({"ok": True, "status": "verified", **asdict(result)}, json_output)


@data_app.command("inspect")
def data_inspect(
    config: ConfigOption = Path("configs/data.yaml"),
    data_root: DataRootOption = Path("data/raw"),
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="Immutable inspection manifest destination.",
        ),
    ] = Path("data/processed/manifests/inspection.json"),
    quarantine: Annotated[
        Path,
        typer.Option(
            "--quarantine",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="Immutable row-index and reason-code report destination.",
        ),
    ] = Path("data/processed/quarantine/rejected.jsonl"),
    json_output: JsonOption = False,
) -> None:
    """Inspect the trusted archive and lock a streaming data-audit manifest."""

    try:
        authored = load_data_config(config)
        manifest = inspect_locked_archive(
            authored,
            data_root,
            manifest_path=out,
            quarantine_path=quarantine,
        )
    except LateSignalError as error:
        _fail(error, json_output)
    _emit(
        {
            "ok": True,
            "status": "inspected",
            "manifest": str(out),
            "quarantine": str(quarantine),
            "rows": manifest["rows"],
            "time_unit": manifest["time_unit"],
        },
        json_output,
    )


@data_app.command("prepare")
def data_prepare(
    config: ConfigOption = Path("configs/data.yaml"),
    features: Annotated[
        Path,
        typer.Option(
            "--features",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Authored feature policy.",
        ),
    ] = Path("configs/features.yaml"),
    data_root: DataRootOption = Path("data/raw"),
    inspection: Annotated[
        Path,
        typer.Option(
            "--inspection",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Immutable inspection manifest.",
        ),
    ] = Path("data/processed/manifests/inspection.json"),
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Processed data root.",
        ),
    ] = Path("data/processed"),
    batch_rows: Annotated[
        int,
        typer.Option("--batch-rows", min=1, help="Maximum accepted rows per Polars batch."),
    ] = 65_536,
    json_output: JsonOption = False,
) -> None:
    """Prepare reviewed rows into isolated feature and truth Parquet stores."""

    try:
        authored = load_data_config(config)
        policy = load_feature_policy(features)
        manifest = prepare_data(
            authored,
            policy,
            data_root=data_root,
            inspection_path=inspection,
            output_root=out,
            batch_rows=batch_rows,
        )
    except LateSignalError as error:
        _fail(error, json_output)
    _emit(
        {
            "ok": True,
            "status": "prepared",
            "out": str(out),
            "manifest": str(out / "manifests" / "preparation.json"),
            "rows": manifest["rows"],
            "streaming": manifest["streaming"],
        },
        json_output,
    )


if __name__ == "__main__":
    app()
