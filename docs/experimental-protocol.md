# Experimental protocol

LateSignal separates protocol authoring, selection, final replay, and final-period evaluation.
The separation is enforced so outcomes from the final period cannot influence configuration choices.

## Locked chronology

Click days 0 through 14 form burn-in.
Click days 25 through 34 are the only days whose outcomes may select model, delayed-method, and sampler parameters.
Click days 35 through 64 form a maturity embargo.
Their eventual outcomes cannot influence any selection or threshold.
Click days 65 through 89 form the sealed final evaluation period.

The final replay starts again at day 0 after the protocol is locked.
Predictions for days 65 through 89 must be persisted and sealed before eventual final truth is evaluated.

## Authored candidate sets

[`configs/protocol.yaml`](../configs/protocol.yaml) contains the complete finite selection spaces required by the build plan.
Strict validation rejects unknown keys, duplicate values, missing candidates, changed final seeds, and a narrowed final matrix.

Selection is staged in this order:

1. Select learning rate, weight decay, dropout, and feature-hash policy under complete-cohort training.
2. Select delayed-method parameters under the selected model.
3. Select sampler parameters under the selected deployable method.

The nuisance sampler for the first two stages is authored in the protocol before outcomes are read.
All three stages train only on click days 0 through 24 using ten credits at daily boundaries D55 through D64.
Every candidate uses a 500-step shared initialization trained only on the fully mature non-monitoring day-0 cohort.
Candidate predictions for held-out click days 25 through 34 are sealed before their truth is joined at D65.
Selection code rejects every click ID from embargo days 35 through 64 and final days 65 through 89.
The winning delayed pair supplies one shared wait duration to fixed-wait and ES-DFM in final Study A, and supplies the fixed Study B learner.

Each stage minimizes selection-period mean log loss under its fixed compute cap.
An exact metric tie within `1e-6` prefers lower measured compute, then fewer parameters, then the lexicographically smaller canonical configuration hash.
The complete attempted candidate table must be retained, including failed and incomplete runs.
Infrastructure failure, timeout, interruption, out-of-memory, disk exhaustion, corruption, or inconclusive evidence blocks protocol locking after bounded recovery.
Only a predeclared insufficient legal main or auxiliary pool may classify a candidate as scientifically protocol-invalid.

## Feasibility gate

Run the estimator before any full selection or final matrix:

```bash
uv run latesignal protocol estimate configs/experiments/final.yaml --json
```

The estimator expands the authored matrix rather than accepting a hand-entered run count.
The current extended V1 matrix contains 89 runs, of which 83 are online runs, and 1,883 online update credits.
It projects every allowed steps-per-credit candidate and chooses the largest candidate that fits every cap without consulting a quality metric.

The configured target is CUDA.
If CUDA is unavailable, the command may run a small CPU diagnostic to confirm the software path, but it does not extrapolate that rate to CUDA and the gate remains blocked.
On the requested device, the bounded model benchmark uses the locked large feature-hash candidate because it is the most resource-intensive permitted architecture.
The final feasibility benchmark uses the locked training batch size and measured warm-up steps, so its compute projection does not extrapolate throughput across batch sizes.
Checkpoint timing uses the production CPU-clone state materialization path followed by three generations through the durable rolling checkpoint store.
The benchmark also writes and verifies a production immutable model snapshot on the execution filesystem.
The projection separately counts 132 final snapshot writes, 4,875 checkpoint-time snapshot verifications, and 132 terminal exact-set verifications.
For a qualified machine, the authored minimum checkpoint-generation duration is a conservative floor over the 2,580 actual generations, while the component benchmark is applied to the 4,620 equivalent single-model writes.
The larger checkpoint estimate is used before snapshot costs are added.
The configured real-data pilot may inspect no more than two prepared click-day partitions.

The durable benchmark uses a fixed ignored work root beside its feasibility result.
An in-repository `--out` destination must be under the ignored `runs/` root so an interrupted benchmark cannot strand large artifacts in tracked project paths.
It holds an exclusive filesystem lock, checks that the work root and result are on the same device, preflights the full rolling-checkpoint peak plus a snapshot and safety margin, and deletes only a marker-verified owned root.
An interrupted owned benchmark can be retried safely.
Foreign, redirected, or concurrently owned content fails closed.

Runs execute sequentially under the feasibility storage model.
An active run reserves three full checkpoint copies for the current checkpoint, previous recoverable checkpoint, and atomic temporary write.
After a completed run's sealed predictions and aggregate evidence are verified, its restricted row-level checkpoints and intermediate ledgers are pruned rather than treated as publishable retained artifacts.
Only aggregate tables, manifests, and hashes contribute to retained artifact storage.

The final configuration records the resource caps authorized after qualification on the intended workstation:

- 89 total runs.
- 4 GPU-hours.
- 25 GiB working disk.
- 2 GiB retained artifact storage.

These values are hard authorization limits rather than predictions or unlimited resources.
Validation remains blocked when the conservative measured upper range exceeds any checked-in limit.
Changing the machine or authorized budget requires new feasibility evidence and a new protocol hash.
The final gate also fails when the requested accelerator is unavailable, the required real-data pilot is unavailable, or no steps-per-credit candidate fits.

Run strict validation after supplying caps and making the prepared data and accelerator available:

```bash
uv run latesignal protocol validate configs/experiments/final.yaml --json
```

Exit code `0` means every feasibility requirement passed.
Exit code `1` means the experimental gate was not met and the JSON `blockers` list identifies the prerequisites.
Configuration errors use exit code `2`.

## Selection evidence and protocol lock

Selection evidence must enumerate all 36 model candidates, all 8 delayed-method candidates, and all 6 sampler candidates.
Each candidate records its canonical configuration hash, status, selection-period log loss, measured compute, parameter count, and failure reason when applicable.
The selection contract rejects any result that accessed an embargo outcome or a final-period metric.
It also rejects a missing, duplicated, or silently narrowed candidate grid.

The lock command requires passing feasibility evidence and a prepared-data manifest:

```bash
uv run latesignal protocol lock configs/experiments/final.yaml \
  --selection results/selection.json \
  --feasibility runs/feasibility/final.json \
  --data-manifest data/processed/manifests/preparation.json \
  --out results/protocol-lock.json \
  --json
```

Before writing the immutable lock, the command verifies every prepared file against its recorded byte count and SHA-256 digest.
The lock captures the final configuration, selection evidence, feasibility result, prepared data, Git commit, dependency lock, installed environment, selected steps per credit, and final seeds.
Its own canonical SHA-256 digest detects later modification.

A dirty Git tree is refused by default.
`--allow-dirty` is an explicit non-publication override that records the dirty paths and makes the lock ineligible for publication.
The override exists for bounded development qualification and must not be used for the final public result.

## Uncertainty and scheduler claim

Final comparisons require the same persisted click IDs, truth labels, click days, and three training seeds for both methods.
The primary paired bootstrap uses three-day contiguous blocks and at least 2,000 replicates.
One-day and seven-day block results are locked sensitivity analyses.

The calibration scheduler is supported only when every predeclared condition passes against fixed deadline.
Those conditions cover paired log loss, bounded Brier and expected-calibration-error degradation, identical core compute, consistent seed signs, and no reversal in either block-size sensitivity.
Failure of any condition produces a negative or inconclusive result rather than a relaxed claim.

## External prerequisites

The authored final configuration cannot pass until the official prepared dataset, an NVIDIA CUDA machine, and actual resource caps are available together.
The public CLI reports this blocked state explicitly.
No final-period result may be inspected to remove or reduce those requirements.
