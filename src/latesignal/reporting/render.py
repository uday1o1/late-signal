"""Deterministic text, JSON, HTML, and CSV aggregate report rendering."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from latesignal.contracts.protocol import StrictModel
from latesignal.data.manifests import canonical_json_bytes, sha256_file, write_json_atomic
from latesignal.errors import ConfigurationError
from latesignal.reporting.model import ReportInput


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ConfigurationError(f"Report output already exists: {path}") from error
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def _rows(values: Iterable[StrictModel]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in values]


def _evaluation_tables(report: ReportInput) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluations: list[dict[str, Any]] = []
    reliability: list[dict[str, Any]] = []
    for item in report.evaluations:
        metrics = item.metrics.model_dump(mode="json")
        bins = metrics.pop("reliability")
        evaluations.append(
            {
                "method": item.method,
                "seed": item.seed,
                "ranking_eligible": item.ranking_eligible,
                **metrics,
            }
        )
        reliability.extend(
            {"method": item.method, "seed": item.seed, **bin_value} for bin_value in bins
        )
    return evaluations, reliability


def _paired_tables(report: ReportInput) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    intervals: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []
    for item in report.paired_intervals:
        value = item.model_dump(mode="json")
        seed_values = value.pop("seed_differences")
        intervals.append(value)
        seeds.extend(
            {
                "control": item.control,
                "candidate": item.candidate,
                "metric": item.metric,
                "block_days": item.block_days,
                **seed_value,
            }
            for seed_value in seed_values
        )
    return intervals, seeds


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        _atomic_text(path, "status\nnot_available\n")
        return
    fields = list(rows[0])
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: ";".join(str(item) for item in value) if isinstance(value, list) else value
                for key, value in row.items()
            }
        )
    _atomic_text(path, output.getvalue())


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>Not available.</p>"
    fields = list(rows[0])
    header = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _text(report: ReportInput) -> str:
    lines = [
        f"# {report.title}",
        "",
        f"Result kind: {report.result_kind}",
        f"Dataset: {report.dataset.name} ({report.dataset.license_id})",
        f"Protocol lock: {report.protocol.lock_sha256}",
        f"Code commit: {report.protocol.code_commit}",
        f"Seeds: {', '.join(str(seed) for seed in report.protocol.seeds)}",
        "",
        "## Chronology and information availability",
        "",
        "Burn-in days 0-14 -> selection outcomes days 25-34 -> maturity embargo days 35-64 "
        "-> sealed final predictions days 65-89.",
        "Predictions are persisted before same-time truth reveal and monitoring uses only fully "
        "mature reserved examples.",
        "",
        "## Result claim",
        "",
        report.claim.statement,
        f"Scheduler outcome: {report.claim.scheduler_outcome}",
        "",
        "## Evidence counts",
        "",
        f"Method rows: {len(report.methods)}",
        f"Scheduler rows: {len(report.schedulers)}",
        f"Seed-level evaluations: {len(report.evaluations)}",
        f"Paired intervals: {len(report.paired_intervals)}",
        f"Slice rows: {len(report.slices)}",
        "",
        "## Limitations",
        "",
        *[f"- {value}" for value in report.limitations],
        "",
        "## Reproduction",
        "",
        *[f"- `{value}`" for value in report.reproduction_commands],
        "",
        "See the adjacent CSV tables for all aggregate values and support counts.",
    ]
    return "\n".join(lines) + "\n"


def _html(report: ReportInput, tables: dict[str, list[dict[str, Any]]]) -> str:
    escaped_title = html.escape(report.title)
    limitations = "".join(f"<li>{html.escape(value)}</li>" for value in report.limitations)
    commands = "".join(
        f"<li><code>{html.escape(command)}</code></li>" for command in report.reproduction_commands
    )
    sections = [
        ("Study A equal-budget methods", "methods"),
        ("Study B credit ledger summary", "schedulers"),
        ("Final overall metrics by seed", "evaluations"),
        ("Paired uncertainty", "paired-intervals"),
        ("Seed-level paired differences", "paired-seeds"),
        ("Fixed-bin reliability evidence", "reliability"),
        ("Calibration and required slices", "slices"),
        ("Quality at intermediate budgets", "intermediate-budget"),
        ("Compute and Pareto accounting", "compute"),
        ("Leakage audit", "leakage-audit"),
    ]
    rendered_sections = "".join(
        f"<section><h2>{html.escape(title)}</h2>{_table(tables[key])}</section>"
        for title, key in sections
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped_title}</title>
<style>
body {{ color: #172033; font-family: ui-sans-serif, system-ui, sans-serif;
line-height: 1.5; margin: 0 auto; max-width: 1200px; padding: 2rem; }}
h1, h2 {{ line-height: 1.2; }}
.notice {{ background: #eef4ff; border-left: 4px solid #315efb; padding: 1rem; }}
table {{ border-collapse: collapse; display: block; font-size: 0.85rem;
overflow-x: auto; width: 100%; }}
th, td {{ border: 1px solid #ccd3df; padding: 0.4rem 0.55rem;
text-align: left; vertical-align: top; }}
th {{ background: #f3f5f8; }}
code {{ background: #f3f5f8; padding: 0.1rem 0.25rem; }}
</style>
</head>
<body>
<h1>{escaped_title}</h1>
<p class="notice">{html.escape(report.claim.statement)}</p>
<p><strong>Dataset:</strong> {html.escape(report.dataset.name)}
({html.escape(report.dataset.license_id)})<br>
<strong>Protocol lock:</strong> <code>{html.escape(report.protocol.lock_sha256)}</code><br>
<strong>Code commit:</strong> <code>{html.escape(report.protocol.code_commit)}</code><br>
<strong>Seeds:</strong> {html.escape(", ".join(str(seed) for seed in report.protocol.seeds))}</p>
<section><h2>Chronology and information availability</h2>
<p>Burn-in days 0-14 -> selection outcomes days 25-34 -> maturity embargo days 35-64
-> sealed final predictions days 65-89.</p>
<p>Predictions precede same-time reveals, features are click-time safe, and monitoring examples
never train the learner.</p></section>
{rendered_sections}
<section><h2>Limitations and threats to validity</h2><ul>{limitations}</ul></section>
<section><h2>Reproduction</h2><ul>{commands}</ul></section>
</body>
</html>
"""


def render_report(
    report: ReportInput,
    output_root: Path,
    *,
    report_format: str,
    input_path: Path,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise ConfigurationError(f"Report output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    evaluations, reliability = _evaluation_tables(report)
    paired_intervals, paired_seeds = _paired_tables(report)
    tables: dict[str, list[dict[str, Any]]] = {
        "methods": _rows(report.methods),
        "schedulers": _rows(report.schedulers),
        "evaluations": evaluations,
        "reliability": reliability,
        "slices": _rows(report.slices),
        "paired-intervals": paired_intervals,
        "paired-seeds": paired_seeds,
        "intermediate-budget": _rows(report.intermediate_budget),
        "compute": _rows(report.compute),
        "leakage-audit": _rows(report.leakage_audit),
    }
    table_root = output_root / "tables"
    for name, rows in tables.items():
        _write_csv(table_root / f"{name}.csv", rows)
    if report_format == "json":
        report_path = output_root / "report.json"
        write_json_atomic(report_path, report.model_dump(mode="json"))
    elif report_format == "text":
        report_path = output_root / "report.md"
        _atomic_text(report_path, _text(report))
    elif report_format == "html":
        report_path = output_root / "report.html"
        _atomic_text(report_path, _html(report, tables))
    else:
        raise ConfigurationError(f"Unsupported report format: {report_format}")
    input_sha256, _ = sha256_file(input_path)
    outputs = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            sha256, size = sha256_file(path)
            outputs.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "sha256": sha256,
                    "bytes": size,
                }
            )
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "status": "complete",
        "aggregate_only": True,
        "input_sha256": input_sha256,
        "report_format": report_format,
        "outputs": outputs,
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    write_json_atomic(output_root / "manifest.json", manifest)
    return manifest
