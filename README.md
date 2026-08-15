# LateSignal

LateSignal is a leakage-audited event-time benchmark for ML engineers and researchers studying conversion learning when positive labels arrive late and negatives become known only after a maturity window.
It compares delayed-label methods and asks when a fixed training budget should be spent using calibration evidence from legally mature cohorts.

The repository includes a deterministic CPU workflow that runs without licensed data, guarded Criteo acquisition and preparation, matched method and scheduler studies, sealed evaluation, paired uncertainty, protocol locking, feasibility estimation, and aggregate-only static reporting.
No Criteo row or measured real-data result is included.

## What this demonstrates

- Prediction is persisted before a truth reveal at the same timestamp.
- Training records cannot enter the learner before their legal availability time.
- Monitoring examples are permanently excluded from training.
- Study A gives every delayed-label method the same schedule and core optimizer budget.
- Study B gives every scheduler the same credits, steps, learner, sampler, and loss.
- Final comparisons require matched click IDs, truth, click days, seeds, and paired block-bootstrap uncertainty.
- A hashed protocol lock binds configuration, data, code, environment, selections, and compute before final scoring.
- Public reports accept aggregate inputs only and expose support counts, uncertainty, and compute.

## Requirements

The CPU workflow requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).
The licensed workflow additionally requires accepted noncommercial access to the Criteo Sponsored Search Conversion Log and at least 30 GiB of local disk.
The locked final experiment targets an NVIDIA CUDA GPU with at least 8 GiB of memory and approximately 16 GiB of host memory.
The GPU is not required for setup, tests, or the synthetic studies.

## Install

```console
uv sync --frozen --all-groups
uv run latesignal --help
```

## First run

Run the complete deterministic event-time path without licensed data:

```console
uv run latesignal run configs/experiments/synthetic.yaml --out runs/synthetic --json
```

A successful run ends with one JSON object like this:

```json
{
  "ok": true,
  "status": "complete",
  "counts": {
    "predictions": 6,
    "available_records": 6,
    "credits": 2,
    "optimizer_steps": 8,
    "optimizer_examples": 20,
    "checkpoints": 2
  },
  "manifest": "runs/synthetic/manifest.json"
}
```

The command writes prediction, availability, credit, exposure, and event ledgers plus checkpoints, metrics, and a reproducibility manifest.
Resume an interrupted run into a new directory with:

```console
uv run latesignal resume runs/synthetic/checkpoints/CHECKPOINT.json --out runs/resumed --json
```

Reproduce the exact checked-in synthetic evidence and verify every ledger hash, count, and metric with:

```console
uv run latesignal reproduce results/published/synthetic-reproduction.json \
  --out runs/reproduced \
  --json
```

The command also refuses a source-tree, configuration, or dependency-lock mismatch.

## Compare delayed-label methods

Run every Study A method through one shared initialization checkpoint, fixed schedule, and exactly reconciled core budget:

```console
uv run latesignal run configs/experiments/study_a.synthetic.yaml --out runs/study-a --json
```

The bounded path covers complete wait, immediate fake negative, fixed wait, DFM, FNW, the ES-DFM constant-wait transfer, and the separate unattainable oracle.
It reports method-specific auxiliary compute separately and does not claim to reproduce a published number.
See [Delayed-label methods](docs/delayed-methods.md) for equations, citations, and transfer boundaries.

## Compare credit schedulers

Run the three fixed timing policies and the calibration-drift scheduler over the complete 59-day adaptive horizon:

```console
uv run latesignal run configs/experiments/study_b.synthetic.yaml --out runs/study-b --json
```

The path retains the final partial five-day window, spends exactly 12 credits per policy, performs identical mature-monitoring inference, and rejects monitoring IDs from training.
Its authored synthetic shift must trigger the calibration scheduler before the fixed deadline.
See [Credit scheduling](docs/scheduler.md) for the residual equation and audit fields.

## Use the licensed dataset

Read [DATA_LICENSE.md](DATA_LICENSE.md) and the linked source terms before downloading.
The normal acquisition path is unavailable without an affirmative acknowledgement:

```console
uv run latesignal data fetch --accept-license --json
```

When no authoritative SHA-256 is configured, the first complete download is retained by content hash but remains untrusted.
The command displays the observed SHA-256 and an exact second command for explicit first-download review.
Inspection and preparation refuse an untrusted artifact.

After the artifact is reviewed and locked, run:

```console
uv run latesignal data inspect --json
uv run latesignal data prepare --json
```

Preparation uses bounded Polars batches and explicit Arrow schemas to publish physically separate click-day feature partitions and reveal-day or maturity-day truth partitions.
Raw data and every derived row-level artifact remain ignored by Git.

## Lock a final protocol

The authored matrix cannot silently omit a required candidate, method, scheduler, offline reference, or seed.
Estimate it with:

```console
uv run latesignal protocol estimate configs/experiments/final.yaml --json
```

The checked-in final configuration records hard authorization limits of 89 runs, 25 GPU-hours, 26 GiB working disk, and 2 GiB retained artifacts.
The estimator includes production-equivalent durable checkpoint and snapshot costs plus a machine-specific floor measured from completed checkpoint generations.
Strict validation intentionally blocks if its conservative upper range exceeds the checked-in limits.
Strict validation can pass only on that CUDA-class machine with the verified prepared data available and every authored cap satisfied; a CPU-only environment remains blocked.
See [Experimental protocol](docs/experimental-protocol.md) for feasibility, selection, protocol-lock, and uncertainty rules.

## Run the final study on CUDA

Clone the repository directly into the selected CUDA environment and install the frozen dependencies:

```console
uv sync --frozen --all-groups
make check
```

After accepting the dataset terms and preparing the licensed data, run the strict machine-bound feasibility gate:

```console
uv run latesignal protocol validate configs/experiments/final.yaml \
  --out runs/feasibility/final.json \
  --json
```

Selection must not start unless validation passes every authored cap and selects the largest eligible training budget without consulting a quality metric.
The public CLI then runs selection, freezes the protocol, qualifies CUDA checkpoint resume, executes the final matrix, and aggregates the report.
Every stage writes immutable resumable evidence under ignored run directories and refuses mismatched code, environment, data, or GPU identities.
See [Reproducibility](docs/reproducibility.md) for the complete resource-neutral command sequence and recovery rules.

## Render an aggregate report

Given a validated `RUN_DIR/report-input.json`, render static HTML and flat evidence tables with:

```console
uv run latesignal report RUN_DIR --format html --json
```

The report input schema rejects unknown row-level fields.
The output contains `report.html`, aggregate CSV tables, and a content-hashed manifest.

## Documentation

- [Architecture](docs/architecture.md)
- [Leakage model](docs/leakage-model.md)
- [Experimental protocol](docs/experimental-protocol.md)
- [Reproducibility](docs/reproducibility.md)
- [Limitations and threats to validity](docs/limitations.md)
- [Dataset and license](docs/dataset-and-license.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the local verification workflow and contribution rules.
Run `make check` locally before preserving a change.
The check does not download the licensed dataset or require a GPU.
When a compatible CUDA device is available, use `configs/experiments/gpu_smoke.yaml` for the bounded local qualification path.
