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

Run the final workflow from one clean clone in the selected CUDA environment.
The prepared dataset may be rebuilt there or mounted from permitted storage, but the source archive and expanded raw file are not required during training.
The exact preparation manifest and every prepared file are rehashed before the protocol can be locked.

The checked-in final configuration provides all four resource caps.
Run the strict bounded estimator with the prepared data and intended CUDA device available:

```bash
uv run latesignal protocol validate configs/experiments/final.yaml \
  --out runs/feasibility/final.json \
  --json
```

The command must exit successfully and choose the largest allowed steps-per-credit value that fits every cap.
It does not use a quality metric for that choice.
The projection includes shared initialization, method-specific core training, worst-case ES-DFM auxiliary training, all primary and intermediate prediction passes, production state cloning, durable rolling checkpoint writes and reload verification, immutable snapshot writes, every checkpoint-time and terminal snapshot verification, pending stage ledgers, and aggregate retention.
The estimator benchmarks checkpoint artifacts in a fixed ignored work root on the result filesystem and removes that root after a normal completion.
Its ownership marker and exclusive lock make a retry recoverable without permitting deletion of foreign content.
The machine-specific checkpoint-generation floor is part of the hashed final configuration, and the estimator uses it whenever it is more conservative than the component benchmark.
The training environment requires only the verified prepared partitions after preparation completes.
The licensed archive and expanded source file remain ignored and must not be committed or copied to an unauthorized location.

Selection runs use outcomes only for click days 25 through 34 and record every attempted candidate.
Run or resume the frozen 36 + 8 + 6 selection graph with the feasibility-selected budget:

```bash
uv run latesignal selection run configs/experiments/final.yaml \
  --data-manifest data/processed/manifests/preparation.json \
  --feature-config configs/features.yaml \
  --cache-root data/runtime-features \
  --out runs/selection \
  --steps-per-credit SELECTED_STEPS \
  --device-uuid GPU_UUID \
  --json
```

After the complete selection evidence exists, create the pre-scoring lock:

```bash
uv run latesignal protocol lock configs/experiments/final.yaml \
  --selection runs/selection/selection-results.json \
  --feasibility runs/feasibility/final.json \
  --data-manifest data/processed/manifests/preparation.json \
  --out runs/protocol-lock.json \
  --json
```

Do not use `--allow-dirty` for a publication run.
Do not inspect a final-period metric before the lock exists.

Replace `SELECTED_STEPS` and `GPU_UUID` with the exact feasibility result and stable NVIDIA device UUID used throughout the run.
Qualify checkpoint resume before final scoring:

```bash
uv run latesignal final qualify configs/experiments/final.yaml \
  --protocol-lock runs/protocol-lock.json \
  --data-manifest data/processed/manifests/preparation.json \
  --feature-config configs/features.yaml \
  --cache-root data/runtime-features \
  --out runs/quality-gate.json \
  --device-uuid GPU_UUID \
  --json
```

Run or resume the complete final matrix, then aggregate only after all 39 runs complete:

```bash
uv run latesignal final run configs/experiments/final.yaml \
  --protocol-lock runs/protocol-lock.json \
  --data-manifest data/processed/manifests/preparation.json \
  --feature-config configs/features.yaml \
  --cache-root data/runtime-features \
  --out runs/final \
  --device-uuid GPU_UUID \
  --json

uv run latesignal final aggregate configs/experiments/final.yaml \
  --protocol-lock runs/protocol-lock.json \
  --data-manifest data/processed/manifests/preparation.json \
  --feature-config configs/features.yaml \
  --cache-root data/runtime-features \
  --out runs/final \
  --quality-gate runs/quality-gate.json \
  --device-uuid GPU_UUID \
  --json
```

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
