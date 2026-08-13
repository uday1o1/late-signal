# LateSignal

LateSignal is a leakage-audited event-time benchmark for learning conversion probability when positive labels arrive late and negatives become known only after a maturity window.
It also evaluates when a fixed compute budget should be spent using calibration evidence from legally mature cohorts.

The repository is under active implementation against [BUILD_PLAN.md](BUILD_PLAN.md).
The current implemented surface covers the licensing and data-audit foundation plus the synthetic event-time vertical slice.
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
