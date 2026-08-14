"""Public LateSignal command-line interface."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
import yaml

from latesignal.contracts.config import load_synthetic_config
from latesignal.contracts.protocol import load_final_protocol
from latesignal.contracts.reproduction import load_reproduction_manifest
from latesignal.contracts.selection import load_selection_results
from latesignal.contracts.study_a import load_study_a_config
from latesignal.contracts.study_b import load_study_b_config
from latesignal.data.config import load_data_config
from latesignal.data.download import FetchNotice, fetch_dataset
from latesignal.data.inspect import inspect_locked_archive
from latesignal.data.manifests import read_json, write_json_atomic
from latesignal.data.prepare import prepare_data
from latesignal.errors import ExitCode, LateSignalError
from latesignal.evaluation.runner import compare_run_dirs, evaluate_run_dir
from latesignal.experiments.estimate import estimate_protocol
from latesignal.experiments.production_final_runner import run_production_final
from latesignal.experiments.production_selection_runner import run_production_selection
from latesignal.experiments.protocol_lock import create_protocol_lock
from latesignal.experiments.reproduction import reproduce_synthetic
from latesignal.experiments.runner import resume_synthetic_experiment, run_synthetic_experiment
from latesignal.experiments.study_a import run_study_a
from latesignal.experiments.study_b import run_study_b
from latesignal.features.policy import load_feature_policy
from latesignal.reporting.model import load_report_input
from latesignal.reporting.render import render_report
from latesignal.security.repository import audit_repository

app = typer.Typer(
    name="latesignal",
    help="Leakage-audited event-time delayed-conversion benchmark.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Acquire and audit the licensed source dataset.", no_args_is_help=True)
protocol_app = typer.Typer(
    help="Validate and estimate authored experiment matrices.", no_args_is_help=True
)
selection_app = typer.Typer(
    help="Run the frozen chronological selection study.", no_args_is_help=True
)
final_app = typer.Typer(help="Run the locked final experiment matrix.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(protocol_app, name="protocol")
app.add_typer(selection_app, name="selection")
app.add_typer(final_app, name="final")

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


@final_app.command("run")
def final_run_command(
    config: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Authored final experiment configuration.",
        ),
    ],
    protocol_lock: Annotated[
        Path,
        typer.Option(
            "--protocol-lock",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Verified pre-scoring protocol lock.",
        ),
    ],
    data_manifest: Annotated[
        Path,
        typer.Option(
            "--data-manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Verified prepared-data manifest.",
        ),
    ],
    feature_config: Annotated[
        Path,
        typer.Option(
            "--feature-config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Authored click-time feature policy.",
        ),
    ],
    cache_root: Annotated[
        Path,
        typer.Option(
            "--cache-root",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Ignored content-addressed runtime feature-cache root.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Ignored durable final-study output root.",
        ),
    ],
    device_uuid: Annotated[
        str,
        typer.Option(
            "--device-uuid",
            min=1,
            help="Stable UUID of the sole CUDA device selected by the launcher.",
        ),
    ],
    json_output: JsonOption = False,
) -> None:
    """Run or resume the exact 21 plus 12 final online experiments."""

    try:
        manifest = run_production_final(
            config,
            protocol_lock_path=protocol_lock,
            data_manifest_path=data_manifest,
            feature_config_path=feature_config,
            cache_root=cache_root,
            output_root=out,
            device_uuid=device_uuid,
            repository=Path.cwd(),
        )
    except LateSignalError as error:
        _fail(error, json_output)
    _emit(
        {
            "ok": True,
            "status": manifest["status"],
            "out": str(out),
            "completed_count": manifest["completed_count"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        json_output,
    )


@selection_app.command("run")
def selection_run_command(
    config: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Authored final experiment configuration.",
        ),
    ],
    data_manifest: Annotated[
        Path,
        typer.Option(
            "--data-manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Verified prepared-data manifest.",
        ),
    ],
    feature_config: Annotated[
        Path,
        typer.Option(
            "--feature-config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Authored click-time feature policy.",
        ),
    ],
    cache_root: Annotated[
        Path,
        typer.Option(
            "--cache-root",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Ignored content-addressed runtime feature-cache root.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Ignored durable selection output root.",
        ),
    ],
    steps_per_credit: Annotated[
        int,
        typer.Option(
            "--steps-per-credit",
            min=1,
            help="Feasibility-selected authored optimizer-step budget.",
        ),
    ],
    device_uuid: Annotated[
        str,
        typer.Option(
            "--device-uuid",
            min=1,
            help="Stable UUID of the CUDA device selected by the launcher.",
        ),
    ],
    json_output: JsonOption = False,
) -> None:
    """Run or resume all 50 staged production-selection candidates."""

    try:
        manifest = run_production_selection(
            config,
            data_manifest_path=data_manifest,
            feature_config_path=feature_config,
            cache_root=cache_root,
            output_root=out,
            steps_per_credit=steps_per_credit,
            device_uuid=device_uuid,
            repository=Path.cwd(),
        )
    except LateSignalError as error:
        _fail(error, json_output)
    _emit(
        {
            "ok": True,
            "status": manifest["status"],
            "out": str(out),
            "candidate_counts": manifest["candidate_counts"],
            "selection_results_sha256": manifest["selection_results_sha256"],
        },
        json_output,
    )


@app.command("audit")
def audit_command(json_output: JsonOption = False) -> None:
    """Audit the repository for restricted artifacts, secrets, and release hygiene."""

    try:
        result = audit_repository(Path.cwd())
    except LateSignalError as error:
        _fail(error, json_output)
    _emit({"ok": result["status"] == "passed", **result}, json_output)
    if result["status"] != "passed":
        raise typer.Exit(code=int(ExitCode.CONSISTENCY_FAILURE))


def _protocol_result(config: Path) -> dict[str, Any]:
    final, protocol, protocol_sha256 = load_final_protocol(config)
    return estimate_protocol(
        final,
        protocol,
        config_path=config,
        protocol_sha256=protocol_sha256,
    )


@protocol_app.command("estimate")
def protocol_estimate(
    config: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Authored final experiment configuration.",
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="Optional immutable feasibility-result destination.",
        ),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """Benchmark the local stack and project the exact authored matrix."""

    try:
        result = _protocol_result(config)
        if out is not None:
            write_json_atomic(out, result)
    except LateSignalError as error:
        _fail(error, json_output)
    _emit({"ok": result["status"] == "passed", **result}, json_output)
    if result["status"] != "passed":
        raise typer.Exit(code=int(ExitCode.GATE_NOT_MET))


@protocol_app.command("validate")
def protocol_validate(
    config: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Authored final experiment configuration.",
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="Optional immutable feasibility-result destination.",
        ),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """Require the matrix, pilot, accelerator, and authored caps to pass."""

    protocol_estimate(config, out, json_output)


@protocol_app.command("lock")
def protocol_lock_command(
    config: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Authored final experiment configuration.",
        ),
    ],
    selection: Annotated[
        Path,
        typer.Option(
            "--selection",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Exhaustive selection-period result document.",
        ),
    ],
    feasibility: Annotated[
        Path,
        typer.Option(
            "--feasibility",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Passing result from protocol estimate.",
        ),
    ],
    data_manifest: Annotated[
        Path,
        typer.Option(
            "--data-manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Prepared-data manifest whose files will be verified.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="New immutable protocol-lock destination.",
        ),
    ],
    allow_dirty: Annotated[
        bool,
        typer.Option(
            "--allow-dirty",
            help="Permit a non-publication lock that records the dirty Git override.",
        ),
    ] = False,
    json_output: JsonOption = False,
) -> None:
    """Freeze selections and all pre-scoring identities into a hashed lock."""

    try:
        final, protocol, protocol_sha256 = load_final_protocol(config)
        selection_results = load_selection_results(selection)
        lock = create_protocol_lock(
            final,
            protocol,
            selection_results,
            read_json(feasibility),
            protocol_sha256=protocol_sha256,
            final_config_path=config,
            selection_path=selection,
            data_manifest_path=data_manifest,
            output_path=out,
            repository=Path.cwd(),
            allow_dirty=allow_dirty,
        )
    except LateSignalError as error:
        _fail(error, json_output)
    _emit(
        {
            "ok": True,
            "status": "locked",
            "out": str(out),
            "lock_sha256": lock["lock_sha256"],
            "publication_eligible": lock["publication_eligible"],
        },
        json_output,
    )


@app.command("evaluate")
def evaluate_command(
    run_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Run directory containing a sealed evaluation-input.json.",
        ),
    ],
    json_output: JsonOption = False,
) -> None:
    """Evaluate one sealed final-period prediction ledger."""

    try:
        result = evaluate_run_dir(run_dir)
    except LateSignalError as error:
        _fail(error, json_output)
    _emit(result, json_output)


@app.command("compare")
def compare_command(
    run_dirs: Annotated[
        list[Path],
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Matched sealed run directories for both methods and every seed.",
        ),
    ],
    control: Annotated[str, typer.Option("--control", help="Primary control method name.")],
    candidate: Annotated[str, typer.Option("--candidate", help="Candidate method name.")],
    replicates: Annotated[
        int,
        typer.Option("--replicates", min=2_000, help="Paired bootstrap replicates."),
    ] = 2_000,
    json_output: JsonOption = False,
) -> None:
    """Compare two methods with matched final clicks and training seeds."""

    try:
        result = compare_run_dirs(
            tuple(run_dirs),
            control_method=control,
            candidate_method=candidate,
            replicates=replicates,
        )
    except LateSignalError as error:
        _fail(error, json_output)
    _emit({"ok": True, "status": "complete", **result}, json_output)


@app.command("report")
def report_command(
    run_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Run directory containing aggregate report-input.json.",
        ),
    ],
    report_format: Annotated[
        Literal["text", "json", "html"],
        typer.Option("--format", help="Static report format."),
    ] = "html",
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="New report directory, defaulting to RUN_DIR/report.",
        ),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """Render a static report from strictly aggregate inputs."""

    report_input = run_dir / "report-input.json"
    output_root = out if out is not None else run_dir / "report"
    try:
        report = load_report_input(report_input)
        manifest = render_report(
            report,
            output_root,
            report_format=report_format,
            input_path=report_input,
        )
    except LateSignalError as error:
        _fail(error, json_output)
    _emit(
        {
            "ok": True,
            "status": "complete",
            "aggregate_only": True,
            "out": str(output_root),
            "format": report_format,
            "manifest_sha256": manifest["manifest_sha256"],
        },
        json_output,
    )


@app.command("reproduce")
def reproduce_command(
    manifest: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Locked reproduction manifest.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="New reproduction run directory.",
        ),
    ],
    json_output: JsonOption = False,
) -> None:
    """Reproduce a locked synthetic result and compare every public output."""

    try:
        result = reproduce_synthetic(
            load_reproduction_manifest(manifest),
            out,
            repository=Path.cwd(),
        )
    except LateSignalError as error:
        _fail(error, json_output)
    _emit(
        {
            "ok": True,
            "status": result["status"],
            "out": str(out),
            "reproduction": str(out / "reproduction.json"),
        },
        json_output,
    )


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
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        mode = raw.get("mode") if isinstance(raw, dict) else None
        if mode == "synthetic":
            authored = load_synthetic_config(config)
            manifest = run_synthetic_experiment(
                authored,
                out,
                stop_after_checkpoints=stop_after_checkpoints,
            )
        elif mode == "synthetic-study-a":
            if stop_after_checkpoints is not None:
                raise typer.BadParameter(
                    "--stop-after-checkpoints is available only for the synthetic vertical slice"
                )
            manifest = run_study_a(load_study_a_config(config), out)
        elif mode == "synthetic-study-b":
            if stop_after_checkpoints is not None:
                raise typer.BadParameter(
                    "--stop-after-checkpoints is available only for the synthetic vertical slice"
                )
            manifest = run_study_b(load_study_b_config(config), out)
        else:
            raise typer.BadParameter(f"Unsupported experiment mode: {mode!r}")
    except LateSignalError as error:
        _fail(error, json_output)
    _emit(
        {
            "ok": manifest["status"] == "complete",
            "status": manifest["status"],
            "out": str(out),
            "manifest": str(out / "manifest.json"),
            "counts": manifest.get("counts", _experiment_counts(manifest)),
            "metrics": manifest.get("metrics"),
            "ledger_sha256": manifest.get("ledger_sha256"),
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
    error = value.get("error")
    message = value.get("message")
    if value.get("ok") is False and isinstance(error, str) and isinstance(message, str):
        typer.echo(f"ERROR {error}: {message}", err=True)
        details = value.get("details")
        if details:
            typer.echo(json.dumps(details, indent=2, sort_keys=True), err=True)
        return
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


def _experiment_counts(manifest: dict[str, Any]) -> dict[str, int]:
    method_count = manifest.get("method_count")
    if isinstance(method_count, int):
        return {"methods": method_count}
    policy_count = manifest.get("policy_count")
    if isinstance(policy_count, int):
        return {"policies": policy_count}
    return {}


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
