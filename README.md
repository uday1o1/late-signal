# LateSignal

LateSignal is a leakage-audited event-time benchmark for learning conversion probability when positive labels arrive late and negatives become known only after a maturity window.
It also evaluates when a fixed compute budget should be spent using calibration evidence from legally mature cohorts.

The repository is under active implementation against [BUILD_PLAN.md](BUILD_PLAN.md).
The current implemented surface covers the licensing and data-audit foundation.
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
