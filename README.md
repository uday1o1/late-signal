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

The checked-in final configuration records the authorized 89-run, 4 GPU-hour, 25 GiB working-disk, and 2 GiB retained-artifact caps measured for the qualified workstation.
Strict validation passes only on that CUDA-class machine with the verified prepared data available; a CPU-only environment remains blocked.
See [Experimental protocol](docs/experimental-protocol.md) for feasibility, selection, protocol-lock, and uncertainty rules.

## Run the one-shot GPU study

The supported final workflow submits the exact clean `origin/main` revision into a detached tmux session on the trusted GPU host:

```console
bash tools/gpu-study.sh submit cuda-pm 1
```

The submit command checks the remote toolchain, GPU availability, stable GPU UUID, memory, disk, and local and remote Git identities, then transfers the prepared dataset before it starts tmux in a detached commit worktree.
Exact prepared-manifest and file verification is the first gate inside the detached job.
The remote job installs the frozen environment, builds truth-free feature caches, runs the complete software gate, reruns feasibility, executes all 50 selection candidates, freezes the protocol, passes the CUDA checkpoint-resume qualification, runs all 39 final candidates, aggregates the report, and prunes rebuildable caches.
It stops instead of scoring when any prerequisite or gate fails.
After the command confirms the started receipt, the Mac may disconnect, sleep, or shut down without affecting the job.

Inspect or collect it later with:

```console
bash tools/gpu-study.sh status cuda-pm
bash tools/gpu-study.sh logs cuda-pm
bash tools/gpu-study.sh collect cuda-pm
```

Collection refuses a mixed destination and copies only feasibility, selection decisions, the protocol lock, the quality receipt, and files sealed by the aggregate-only collection manifest.
It does not copy row-level predictions, checkpoints, model weights, prepared rows, or the licensed source archive.
See [Reproducibility](docs/reproducibility.md) for recovery and resource-limit behavior.

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
Hosted CI is CPU-only and never downloads the licensed dataset.
A manual trusted-runner workflow provides the bounded CUDA qualification path.
