# Reproducibility

LateSignal distinguishes deterministic software qualification from statistical reproduction of the full GPU experiment.
The same software stack and seed must reproduce synthetic ledgers exactly.
The final GPU study is evaluated across fixed seeds and paired uncertainty rather than promising bitwise identity across different hardware.

## Clean CPU qualification

From a clean checkout with Python 3.12 and `uv` available, run:

```bash
uv sync --frozen --all-groups
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv run latesignal run configs/experiments/synthetic.yaml --out runs/synthetic
uv run latesignal run configs/experiments/study_a.synthetic.yaml --out runs/study-a
uv run latesignal run configs/experiments/study_b.synthetic.yaml --out runs/study-b
uv run latesignal reproduce results/published/synthetic-reproduction.json \
  --out runs/reproduced
```

Hosted CI performs the same formatting, lint, typing, and test checks without downloading the licensed dataset or requiring a GPU.
The checked-in synthetic manifest binds the source tree, dependency lock, configuration, five public ledgers, execution counts, and final metrics.

## Licensed data workflow

Review [DATA_LICENSE.md](../DATA_LICENSE.md) and the source terms before downloading.
Then run:

```bash
uv run latesignal data fetch --accept-license --json
```

If the configuration has no authoritative digest, the first complete download exits with `FIRST_DOWNLOAD_REVIEW_REQUIRED` and displays its SHA-256.
Review that exact digest and rerun with the supplied command:

```bash
uv run latesignal data fetch --accept-license --review-sha256 SHA256 --json
```

The second command creates the artifact lock only when the retained bytes still match the reviewed digest.
Continue with:

```bash
uv run latesignal data inspect --json
uv run latesignal data prepare --json
```

Raw data, prepared rows, acknowledgements, quarantine rows, checkpoints, and ordinary runs remain under ignored roots.

## Feasibility and selection

The actual final-run machine owner must supply all four caps in `configs/experiments/final.yaml`.
Run the bounded estimator on the intended CUDA machine with prepared data available:

```bash
uv run latesignal protocol validate configs/experiments/final.yaml \
  --out results/feasibility.json \
  --json
```

The command must exit successfully and choose the largest allowed steps-per-credit value that fits every cap.
It does not use a quality metric for that choice.

Selection runs use outcomes only for click days 25 through 34 and record every attempted candidate.
After the complete selection evidence exists, create the pre-scoring lock:

```bash
uv run latesignal protocol lock configs/experiments/final.yaml \
  --selection results/selection.json \
  --feasibility results/feasibility.json \
  --data-manifest data/processed/manifests/preparation.json \
  --out results/protocol-lock.json \
  --json
```

Do not use `--allow-dirty` for a publication run.
Do not inspect a final-period metric before the lock exists.

## Final evaluation and report

Every final method comparison must use identical persisted click IDs and the fixed seeds 17, 41, and 73.
At least 2,000 paired block-bootstrap replicates are required.
The primary block size is three days, with one-day and seven-day sensitivities.

After final aggregate evidence is assembled as `report-input.json`, render it with:

```bash
uv run latesignal report RUN_DIR --format html --json
```

The report directory contains one static report, flat aggregate CSV tables, and a content-hashed manifest.
The strict schema rejects unknown row-level fields and incomplete final evidence.

## Recorded identities

Publication evidence records the code commit, dirty-tree state, authored configuration hash, protocol hash, selection hash, feasibility hash, preparation manifest and file hashes, `uv.lock` hash, installed dependency versions, Python compiler, operating system, CUDA runtime, NVIDIA driver, GPU identity, seeds, budgets, and output hashes.
Any changed identity requires a new lock and a complete replay from day 0.

## Nondeterminism boundary

PyTorch deterministic algorithms are enabled where supported, cuDNN benchmarking is disabled, and random generators are seeded and checkpointed.
Resume equality is required on the same supported stack.
Across different GPUs, drivers, or library builds, claims rely on seed-level estimates and paired intervals rather than bitwise equivalence.
