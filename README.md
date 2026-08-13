# LateSignal

LateSignal is a leakage-audited event-time benchmark for learning conversion probability when positive labels arrive late and negatives become known only after a maturity window.
It also evaluates when a fixed compute budget should be spent using calibration evidence from legally mature cohorts.

The repository is under active implementation against [BUILD_PLAN.md](BUILD_PLAN.md).
The current implemented surface covers the licensing and data-audit foundation, data preparation, synthetic event-time vertical slice, shared conversion model, offline sanity references, synthetic Study A qualification, and compute-matched Study B scheduler qualification.
No Criteo data or measured model result is included.

## Development setup

Install `uv`, then run:

```console
uv sync --all-groups
uv run latesignal --help
make check
```

Python 3.12 is the locked runtime.
Hosted verification is CPU-only and never downloads the licensed dataset.

## Dataset acknowledgement

Read [DATA_LICENSE.md](DATA_LICENSE.md) and the linked source terms first.
The official download is deliberately unavailable without an affirmative flag:

```console
uv run latesignal data fetch --accept-license
```

When no authoritative SHA-256 is configured, the first authorized download is retained by content hash but remains untrusted.
The command prints the observed SHA-256 and an exact second command for explicit first-download review.
Preparation and inspection refuse an untrusted artifact.

Raw data and every derived row-level artifact are ignored by Git.

After a reviewed archive has an immutable inspection manifest, prepare it with:

```console
uv run latesignal data prepare
```

Preparation uses bounded Polars batches and explicit Arrow schemas to publish physically separate click-day feature partitions and reveal-day or maturity-day truth partitions.
The command refuses to overwrite an existing prepared store.

## Synthetic vertical slice

Run the complete deterministic CPU path without licensed data:

```console
uv run latesignal run configs/experiments/synthetic.yaml --out runs/synthetic
```

The run writes predictions before same-time reveals, drains final truth after the last click, and records metrics, compute ledgers, checkpoints, and a reproducibility manifest.
Resume an interrupted run into a new directory with:

```console
uv run latesignal resume runs/synthetic/checkpoints/CHECKPOINT.json --out runs/resumed
```

## Synthetic Study A qualification

Run every delayed-label method through one shared initialization checkpoint, fixed schedule, and exactly reconciled core budget without licensed data:

```console
uv run latesignal run configs/experiments/study_a.synthetic.yaml --out runs/study-a
```

This bounded path qualifies complete wait, immediate fake negative, fixed wait, DFM, FNW, the ES-DFM constant-wait transfer, and the separate unattainable oracle.
It reports method-specific auxiliary compute separately and does not claim to reproduce any published result.
See [docs/delayed-methods.md](docs/delayed-methods.md) for equations, citations, and transfer boundaries.

## Synthetic Study B qualification

Run the three fixed timing policies and the calibration-drift scheduler over the complete 59-day adaptive horizon without licensed data:

```console
uv run latesignal run configs/experiments/study_b.synthetic.yaml --out runs/study-b
```

The path retains the final partial five-day window, spends exactly 12 credits per policy, performs identical daily mature-monitoring inference, and rejects monitoring IDs from training.
Its authored synthetic shift must trigger the calibration scheduler before the fixed deadline.
See [docs/scheduler.md](docs/scheduler.md) for the residual equation and audit fields.
