# LateSignal Build Plan

## 1. Document purpose and authority

This file is the implementation authority for LateSignal V1.
An implementation agent should read the complete file before changing the repository.
The project is currently an empty Git repository, so every path below is proposed rather than existing.
The agent should implement milestones in order and should not change the final evaluation protocol after viewing final-period results.
Every reported improvement must be supported by the locked protocol, paired uncertainty estimates, and equal-budget accounting described here.

## 2. Product definition

LateSignal is a leakage-audited event-time benchmark for learning conversion probability when positive labels arrive late and negative labels become known only after a maturity window.
It also evaluates a compute-matched scheduler that decides when to spend fixed update credits using calibration residuals measured only on legally available mature cohorts.

The defensible one-sentence contribution is:

> LateSignal provides a reproducible event-time simulator and equal-compute experimental framework for delayed conversion learning, including a mature-cohort calibration scheduler that chooses when to spend fixed update credits.

LateSignal is primarily a machine learning portfolio project.
Its strongest interview evidence should be scientific protocol design, temporal leakage prevention, baseline fidelity, calibrated evaluation, reproducibility, and honest negative results.

## 3. Problem being solved

An online conversion learner must predict when a click occurs, but the outcome may remain unknown for up to 30 days.
Treating every not-yet-converted click as negative introduces fake negatives.
Waiting for every label produces stale models.
Updating more frequently can improve freshness but changes compute consumption, which makes many scheduler comparisons unfair.

LateSignal separates three questions that are often mixed together:

1. Which delayed-label training method performs best under a fixed update schedule and compute budget?
2. Given one fixed training method, when should a system spend a fixed number of update credits?
3. How do quality, calibration, freshness, and compute trade off over chronological deployment time?

The project does not claim causal bidding lift because the source data does not contain randomized propensities or counterfactual outcomes for unshown ads.

## 4. Scope contract

### 4.1 Required V1 capabilities

- Download the official Criteo Sponsored Search Conversion Log only after explicit license acknowledgement.
- Verify the archive hash, byte count, member list, and extracted schema.
- Infer and lock timestamp units only when the official 90-day and 30-day constraints identify a unique scale.
- Produce an immutable inspection manifest and a quarantine report with zero silent row drops.
- Partition prepared data by click day and reveal day.
- Enforce a strict click-time feature allowlist.
- Isolate final truth behind a narrow oracle and evaluator interface.
- Simulate click arrival, positive conversion reveal, negative maturity, scheduling, and training in event-time order.
- Compare delayed-label methods under one fixed schedule and matched core optimizer budgets.
- Compare scheduling policies under one fixed learner, loss, sampler, and matched credit budget.
- Evaluate chronological final-period predictions with paired time-block bootstrap intervals.
- Report quality, calibration, slices, update cost, total system cost, and results at intermediate budget fractions.
- Provide checkpoint and resume behavior with deterministic seed and ledger restoration.
- Publish only aggregate results, manifests, configurations, code, and small synthetic fixtures.

### 4.2 Explicit non-goals

- V1 will not train a bidding policy or claim revenue lift, policy regret, or causal impact.
- V1 will not use the Criteo Attribution dataset as its primary dataset.
- V1 will not merge CriteoPrivateAd into the core experiment.
- V1 will not claim reproduction of published numbers from datasets with different distributions.
- V1 will not make a custom survival or hazard architecture its novelty claim.
- V1 will not combine scheduler and label-strategy changes in one comparison.
- V1 will not include revenue as a feature, objective, or primary outcome.
- V1 will not use post-click outcomes or future aggregates as model features.
- V1 will not tune on the final chronological period.
- V1 will not use a live dashboard or API as a substitute for scientific evidence.
- V1 will not claim bitwise reproducibility across different hardware and software stacks.
- V1 will not commit raw or prepared Criteo rows.

## 5. Dataset and legal boundary

The core dataset is the official Criteo Sponsored Search Conversion Log.
The official page describes one row per click, 90 days of click data, a conversion-within-30-days label, conversion delay, revenue, and hashed click-time characteristics.
The dataset license is Creative Commons Attribution-NonCommercial-ShareAlike 4.0.
The code should use a permissive repository license, but the data remains governed by its own license.

The fetch command must require:

```text
latesignal data fetch --accept-license
```

The command must print the dataset name, license identifier, official page, archive URL, destination, and noncommercial restriction before downloading.
The acknowledgement is recorded locally with timestamp and code version but is not committed.
The downloader must not mirror the archive to GitHub releases or another repository-controlled host.
Trained checkpoints are not redistributed unless the dataset-license implications have been reviewed and documented.
The archive streams into a newly created temporary file, is closed and hashed, passes member and expansion-safety inspection, and is atomically promoted to its content-addressed local path.

The official archive researched on 2026-08-13 is:

```text
https://criteostorage.blob.core.windows.net/criteo-research-datasets/Criteo_Conversion_Search.tar.gz
```

The server reported `Content-Length: 2002864638` and `Last-Modified: 2020-04-08` during planning.
The official page's older compressed-size statement is not a validation source.
The repository lock records the observed byte count and a locally computed cryptographic hash after the first authorized download.
If either changes, preparation stops until the new artifact is explicitly reviewed.

## 6. Raw schema and feature policy

The raw file has 23 documented fields.

| Category | Fields | V1 use |
| --- | --- | --- |
| Outcome | `Sale` | Final binary truth only. |
| Outcome value | `SalesAmountInEuro` | Excluded from V1. |
| Reveal timing | `time_delay_for_conversion` | Truth store and reveal scheduler only. |
| Event timing | `click_timestamp` | Chronology and event identity only, not a V1 model feature. |
| Numeric click context | `nb_clicks_1week`, `product_price` | Allowed after burn-in-fitted preprocessing. |
| Categorical click context | `product_age_group`, `device_type`, `audience_id`, `product_gender`, `product_brand`, `category1` through `category7`, `country`, `product_id`, `product_title`, `partner_id`, `user_id` | Allowed through deterministic field-specific hashing. |

The exact raw spelling must be captured from the official file during Phase 0 and mapped once in a checked-in schema contract.
No code should rely on informal field spelling from this planning document.

The V1 model has 17 categorical fields and two numeric fields with missing indicators.
`click_timestamp` is deliberately excluded from the model so the first study measures delayed feedback rather than direct calendar memorization.

Forbidden model inputs include:

- `Sale`
- `SalesAmountInEuro`
- `time_delay_for_conversion`
- Reveal time.
- Maturity time.
- Any future click, conversion, or frequency aggregate.
- A vocabulary, normalization statistic, or bucket boundary fit using validation or final-period rows.
- Any conversion identifier or post-outcome field from a different Criteo dataset.

The feature-policy validator must fail closed when an unapproved field enters a training batch.

## 7. Phase 0 data inspection

Data inspection occurs before model implementation because timestamp units are not explicitly documented in the public field description.

The inspector must record:

- Archive SHA-256 and byte count.
- Archive member names, sizes, and compression ratios.
- Extracted file SHA-256 and byte count.
- Row and column counts.
- Exact header or field order.
- Parse failures by field and raw-row identity.
- Click timestamp monotonicity and range.
- Candidate timestamp-unit interpretations.
- Conversion delay distribution and candidate units.
- Whether the inferred click span matches approximately 90 days.
- Whether the maximum valid conversion window matches 30 days.
- Sale and delay consistency counts.
- Sentinel, missing, invalid, negative, and nonfinite values by field.
- Duplicate raw rows without deleting them.
- Conversion rate and row count by inferred click day.
- Last click time, last positive reveal time, and last negative maturity time.

The inspector tests one shared multiplier for click timestamps and conversion delays from `{1, 1e-3, 1e-6, 1e-9}` seconds per raw unit.
A candidate passes only when the normalized click span is between 89 and 91 days, every accepted positive delay is between zero and 30 days inclusive, every positive reveal is at or after its click, and documented negative sentinels remain sentinels rather than timestamps.
The timestamp scale is accepted only if exactly one multiplier satisfies all conditions.
If zero or multiple scales remain plausible, the inspector must stop with `AMBIGUOUS_TIME_UNIT` and require a checked-in decision note supported by new evidence.
The preparation pipeline must not silently coerce an ambiguous unit.

Rows that violate the locked schema are written to a quarantine manifest with reason codes and raw-row indexes.
The pipeline must assert:

```text
accepted rows + quarantined rows = parsed raw rows
```

No deduplication occurs because identical click rows can be legitimate separate events.

## 8. Prepared data contract

Each row receives a stable `click_id` computed from the raw file hash and zero-based raw row index using BLAKE3.
The ID must not depend on mutable prepared values.

Prepared data is split into physically separate stores:

```text
data/processed/features/click_day=DDD/part-*.parquet
data/processed/truth/reveal_day=DDD/part-*.parquet
data/processed/truth/maturity_day=DDD/part-*.parquet
data/processed/manifests/inspection.json
data/processed/manifests/preparation.json
data/processed/quarantine/rejected.parquet
```

The feature store contains `click_id`, click time, allowed raw features, and prepared feature values.
The truth store contains `click_id`, final label, conversion reveal time for positives, and negative maturity time for negatives.
Only `LabelOracle` and final `Evaluator` code may open the truth store.
Learner, scheduler, sampler, feature pipeline, and report code must receive truth through typed records that enforce availability time.

For a positive click:

```text
available_at = click_time + conversion_delay
```

For a negative click:

```text
available_at = click_time + 30 days
```

The simulator may create pseudo-negative training observations before `available_at` for methods that explicitly require them.
Those observations must be typed as provisional and must never be confused with ground truth.

## 9. Feature engineering contract

Categorical features use deterministic field-specific hashing:

```text
bucket = xxhash64(field_name || ":" || raw_value, field_seed) mod bucket_count
```

Every field has a separate embedding table so the same hash bucket across fields does not imply shared meaning.
The field seed and bucket count are locked in the feature configuration.
High-cardinality fields start with 16-dimensional embeddings.
Low-cardinality fields start with 8-dimensional embeddings.
Final bucket counts and field grouping are selected only on the selection period under the declared memory budget.

`nb_clicks_1week` and `product_price` use `log1p` where values are nonnegative.
Each numeric field has a missing indicator.
Clipping thresholds, means, and scales are fit on burn-in clicks only.
Invalid negative values are quarantined or mapped only under an explicit inspected sentinel rule.

Cold-user and cold-product status are computed online from click history strictly before the current click.
No global frequency table is allowed.
Past-frequency slices are also based only on prior clicks.

## 10. Chronological protocol

The locked click-day periods include a 30-day maturity embargo before final evaluation:

| Click days | Purpose |
| --- | --- |
| 0 through 14 | Burn-in and preprocessing fit. |
| 15 through 24 | Method development and debugging. |
| 25 through 34 | Chronological configuration selection. |
| 35 through 64 | Maturity embargo whose outcomes cannot select configuration. |
| 65 through 89 | Locked final prequential evaluation. |

Burn-in transformations use clicks from days 0 through 14 only.
Development metrics use eventual outcomes only for clicks from days 15 through 24.
Selection metrics use eventual outcomes only for clicks from days 25 through 34.
All selection labels have reached the 30-day maturity boundary by the start of day 65.
Clicks from days 35 through 64 may participate legally when the final replay reaches them, but their eventual outcomes cannot influence architecture, hyperparameters, method choice, scheduler choice, threshold, metrics, or protocol.
After the protocol is frozen at the start of day 65, the final run starts again from day 0 and scores predictions made for clicks on days 65 through 89.

Final clicks are predicted at click time.
The simulation clock then continues until 30 days after the last click so all final labels mature for evaluation.
No model update after a final click may change the already persisted prediction for that click.
The update horizon ends at the last click timestamp, and the remaining 30-day interval is evaluation-only truth draining.
Any update credit not legally spent by the last click makes the run incomplete rather than being spent after it can no longer affect a scored prediction.
The final evaluator may read eventual truth only after the prediction ledger is sealed.

## 11. Event-time simulator

The simulator advances through hourly event boundaries and makes scheduling decisions once per UTC day.
At a boundary time `t`, the operation order is fixed:

1. Predict every click with `previous_boundary < click_time <= t` using the current model.
2. Persist predictions atomically before processing any label with the same timestamp.
3. Deliver those clicks to the label-strategy state machine.
4. Reveal positive conversions with `reveal_time <= t` and negatives with `maturity_time <= t` that were not previously revealed.
5. Append all legal training observations to the availability ledger.
6. Update mature monitoring statistics using only reserved monitoring examples.
7. Ask the scheduler whether to spend an update credit when `t` is a daily decision boundary.
8. If approved, sample only legally available training records and perform exactly the configured steps.
9. Persist counters, random-generator states, data cursors, optimizer state, scheduler state, and hashes atomically.

Same-timestamp prediction-before-reveal ordering is mandatory.
Boundary behavior must have golden tests at one unit before, exactly at, and one unit after reveal and maturity times.

## 12. Shared model and training contract

The online model is shared across the equal-compute studies.

Architecture:

```text
17 field-specific categorical embeddings
2 transformed numeric values plus 2 missing indicators
concatenate
Linear(256) -> SiLU -> LayerNorm -> Dropout(0.1)
Linear(128) -> SiLU -> LayerNorm -> Dropout(0.1)
Linear(64) -> SiLU
Linear(1) -> binary logit
```

The initial optimizer is AdamW.
The initial batch size is 2048.
The initial gradient norm clip is 5.0.
Learning rate, weight decay, embedding bucket counts, and one permitted dropout choice are selected on the selection period only.
Every method in a matched study receives the same selected backbone, optimizer family, batch size, sampler, random-seed set, and optimizer-example budget unless the method mathematically requires an explicitly reported auxiliary term.

PyTorch deterministic algorithms should be enabled where supported.
cuDNN benchmarking should be disabled for reproducibility runs.
The manifest records PyTorch, CUDA, driver, GPU, operating system, Python, compiler, dependency lock, commit, configuration, and random seeds.
Cross-hardware results are statistically reproducible rather than promised bitwise identical.

### 12.1 Locked finite selection spaces

All candidate sets are written to `configs/protocol.yaml` before any selection-period outcome is read.
The initial finite model grid is learning rate `{1e-4, 3e-4, 1e-3}`, weight decay `{0, 1e-5, 1e-4}`, dropout `{0.0, 0.1}`, and feature-hash policy `{compact, large}`.
The compact policy uses `2^18` buckets for high-cardinality identifier fields and `2^12` for other categorical fields.
The large policy uses `2^20` buckets for high-cardinality identifier fields and `2^14` for other categorical fields.
High-cardinality fields are user, product, product title, audience, partner, and brand.
The delayed-method wait grid is `{1, 3, 7, 14}` days.
The V1 scheduler allocation window is fixed at five days.
The V1 scheduler trigger is fixed at `3.0` standardized residual units.
The sampler recent-window grid is `{1, 3, 7}` days and its reservoir-capacity grid is `{1000000, 5000000}` records.
The probability bins are ten fixed equal-width bins on `[0, 1]` and are not tuned.
The primary bootstrap block is three days, with one-day and seven-day sensitivity analyses.
The slice reporting minimum is 10,000 examples and 100 positives.
The final training seeds are `{17, 41, 73}` and the monitoring-membership seed is `20260813`.
Steps per credit must be one of `{100, 250, 500}` and the feasibility gate chooses the largest value that fits every authored resource cap without using quality metrics.

Selection is staged rather than a full Cartesian search.
The order is model and feature policy under complete-cohort training, delayed-method parameters under the selected model, then sampler parameters under the selected deployable method.
Each stage minimizes mean selection-period log loss under its fixed compute cap.
Exact metric ties within `1e-6` prefer lower total measured compute, then fewer parameters, then the lexicographically smaller canonical configuration hash.
The complete candidate table, attempted runs, failed runs, and selection decision are retained.

## 13. Training-record and method interface

Each method receives events rather than direct truth-table access.

```python
class DelayedMethod(Protocol):
    def on_click(self, click: ClickEvent) -> list[TrainingRecord]: ...
    def on_positive_reveal(self, label: PositiveReveal) -> list[TrainingRecord]: ...
    def on_negative_maturity(self, label: NegativeMaturity) -> list[TrainingRecord]: ...
    def state_dict(self) -> dict[str, object]: ...
    def load_state_dict(self, state: dict[str, object]) -> None: ...
```

A `TrainingRecord` includes `click_id`, `available_at`, provisional or final status, target, nonnegative weight where applicable, correction group, source method, and feature reference.
The trainer rejects any record whose `available_at` exceeds simulator time.
Correction records remain distinct ledger entries and are not silently collapsed with their original pseudo-negative record.

Published methods must be reimplemented from their equations and verified against small hand-calculated cases.
Code must not be copied from a repository that lacks a compatible license.

## 14. Study A - Delayed-label strategy comparison

Study A asks which delayed-label method works best under one fixed daily update schedule and a matched core optimizer budget.

Required methods are:

### 14.1 Common notation and initialization

Let `x` be click-time features, `c` the click time, `D` the conversion delay in days, `W = 30` days, `f(x)` the conversion logit, `p = sigmoid(f(x))`, `softplus(-f)` the positive BCE term, and `softplus(f)` the negative BCE term.
At `D(31)`, all methods start from one shared initialization checkpoint trained with ordinary BCE on the fully mature day-0 click cohort under a fixed initialization budget.
Clicks and legal reveals before `D(31)` have already entered each method's state machine, but matched experimental update credits begin at `D(31)`.
Study A allocates one fixed daily credit at `D(31)` through `D(89)`, for 59 credits per method.
Every credit performs the same core optimizer steps and batch size.

### 14.2 Complete-cohort wait

For every click, emit exactly one final BCE record at `c + W` with its eventual binary label.
No provisional or correction record is emitted.
This is a low-bias but stale reference.

### 14.3 Immediate fake-negative BCE

Emit one provisional negative BCE record at click time.
If the click converts, emit one separate positive BCE correction record at its legal reveal time.
If the click never converts, emit no additional record at maturity.
Both records remain independently sampleable and linked through `correction_group = click_id`.
This intentionally simple baseline exposes fake-negative bias.

### 14.4 Fixed-wait correction

Choose `w` from `{1, 3, 7, 14}` days on selection.
If conversion reveals at or before `c + w`, emit one positive BCE record at reveal and no provisional negative.
Otherwise emit one provisional negative at `c + w` and emit a separate positive correction if conversion reveals later.
A final negative receives only the provisional negative.

### 14.5 Delayed Feedback Model transfer

DFM uses the shared trunk with conversion logit `f(x)` and a separate rate logit `g(x)`.
The exponential rate per day is `lambda = clamp(softplus(g(x)) + 1e-6, 1e-6, 100)`.
For a revealed positive with delay `d`, the loss is `-log(p) - log(lambda) + lambda * d`.
For an unresolved or mature-negative click observed at elapsed time `e = min(t - c, W)`, the right-censored loss is `-log((1 - p) + p * exp(-lambda * e))` evaluated with a stable log-sum-exp form.
At each update, the deterministic sampler selects click IDs and DFM materializes exactly one current record per selected click using only status legal at time `t`.
Mature negatives remain censored at `W` because the data establishes nonconversion within the attribution window rather than lifetime nonconversion.
DFM's delay head and its forward and backward work are counted as method-specific total compute while the number of selected core examples and steps remains matched.
It must be described as a method transfer because the original paper's dataset is not this Sponsored Search corpus.

### 14.6 Fake Negative Weighted transfer

FNW uses the same duplicate record stream as immediate fake-negative BCE.
For each sampled optimizer batch, compute the current pre-update logits once and set `p_stop = sigmoid(f).detach()` before zeroing gradients or changing parameters.
The probability is recomputed on every later exposure rather than stored when the record is emitted.
For model probability `p_stop`, positive record label `z = 1` has weight `1 + p_stop` on `softplus(-f)`.
Negative record label `z = 0` has weight `1 - p_stop^2` on `softplus(f)`.
The batch loss is the mean of `z * (1 + p_stop) * softplus(-f) + (1 - z) * (1 - p_stop^2) * softplus(f)`.
Probability values are clamped to `[1e-6, 1 - 1e-6]` only for numerical evaluation of weights.
The shared conversion backbone, sampler, and core budget remain unchanged.

### 14.7 ES-DFM constant-wait variant

ES-DFM uses the fixed-wait event stream from section 14.4 with selection-locked `w`.
It trains `q_tn(x)` on fully mature, non-monitoring clicks unresolved at `w`, with target one for an eventual negative and zero for a delayed positive.
It trains `q_dp(x)` on fully mature, non-monitoring clicks, with target one only for a positive whose delay exceeds `w`.
The auxiliary models use the same feature encoder shape but separate parameters and are updated only from legally mature records at the same daily credit boundaries.
Each auxiliary model uses AdamW with learning rate `3e-4`, weight decay `1e-4`, batch size 2048, gradient clipping 5.0, and ordinary BCE.
At `D(31)`, train `q_tn` for 500 steps and then `q_dp` for 500 steps on their legal day-0 mature pools using replacement when nonempty.
At every later ES-DFM credit, first freeze the legal mature pool at the credit time, update `q_tn` for 100 steps, update `q_dp` for 100 steps, and then place both auxiliary models in evaluation mode.
For every main-model batch in that credit, recompute `q_tn` and `q_dp` from this frozen auxiliary checkpoint, detach both probabilities, compute weights, and update only the main conversion model.
Auxiliary seeds are `training_seed + 1000` for `q_tn` and `training_seed + 2000` for `q_dp`.
An empty required auxiliary pool makes the ES-DFM run invalid rather than silently skipping its update.
For detached, clamped auxiliary probabilities, a positive main-model record has weight `1 + q_dp` and a negative record has weight `(1 + q_dp) * q_tn`.
Weights are clipped to `[1e-4, 2.0]` and the main loss is weighted BCE.
Auxiliary optimizer examples, steps, memory, and wall time are reported outside the matched core conversion-model budget.
Use the official BSD-3-Clause repository only as a behavioral reference.
The repository snapshot inspected during planning was commit `7f66101916db08d926b721153a874fc19eac21d3` at `https://github.com/ThyrixYang/es_dfm`.
The implementation must preserve required attribution and citation conditions if any code is adapted.

### 14.8 Oracle upper reference

An oracle may train on eventual labels at click time only as an unattainable upper reference.
It is not a deployable competitor and must be styled separately in every result table.

Logistic regression and LightGBM trained on mature offline labels are separate sanity references.
They are not part of the equal-compute online ranking because their optimization and representation differ.

## 15. Study B - Compute-matched scheduling

Study B fixes one validated delayed-label method, the shared MLP, loss, optimizer, sampler, training steps per credit, and total credits.
Only the timing policy changes.

The candidate fixed policies are:

- Spend each credit at the earliest legal time in its allocation window.
- Spend each credit at the midpoint of its allocation window.
- Spend each credit at the deadline of its allocation window.
- Spend a credit at the first mature-cohort calibration trigger, otherwise spend it at the deadline.

The V1 allocation-window size is fixed at five days before selection outcomes are read.
Each window owns exactly one credit.
All policies therefore spend the same number of credits and optimizer steps.

Let `T0` be the normalized start of click day 0 and `D(d) = T0 + d * 24 hours`.
Study B's adaptive horizon is the half-open interval `[D(31), D(90))` after the common initialization checkpoint.
For locked window width `w`, window `j` is `[D(31 + j*w), min(D(31 + (j+1)*w), D(90)))` and `M = ceil(59 / w)`.
The final partial window is retained and receives one credit.
The early policy spends at the first daily boundary in the window.
The midpoint policy spends at the first daily boundary at or after the temporal midpoint.
The deadline policy spends at the final daily boundary strictly before the window end.
Credits cannot roll over, be borrowed, or be spent twice.
If the eligible pool is nonempty but smaller than one batch, every policy samples with replacement under the same deterministic sampler.
If the pool is empty at a required spend, the run stops as `INSUFFICIENT_LEGAL_POOL` and is not included in equal-credit comparison.
Window width is selected once and then held identical across every final Study B policy.

### 15.1 Calibration Drift Credit Scheduler

The proposed scheduler is named `CalibrationDriftCreditScheduler`.
At each daily decision, it evaluates only a deterministic 10 percent monitoring split of fully mature click cohorts.
Monitoring examples are permanently excluded from training.
The monitoring window initially covers the most recent seven fully mature click days and is locked on validation.
Monitoring membership is `xxhash64(click_id, monitor_seed) mod 10 == 0` with a locked seed.
At each daily boundary, `p_i` is a fresh prediction from that policy's current checkpoint over the frozen click-time features of the mature monitoring examples.
The model checkpoint hash is recorded with the statistic.
Every fixed and adaptive policy performs the same daily monitoring inference so monitoring compute does not differ by scheduler.

Predictions are grouped into fixed probability bins defined before the final run.
For each nonempty bin `b`, compute:

```text
residual_b = abs(sum_i(y_i - p_i)) / sqrt(sum_i(p_i * (1 - p_i)) + epsilon)
D(t) = max_b residual_b
```

Use `epsilon = 1e-8`.
A bin is eligible only when it contains at least 1,000 examples and `sum_i(p_i * (1 - p_i)) >= 25`.
Undersupported bins are ignored rather than merged after seeing their labels.
If no bin is eligible, the day cannot trigger and the deadline rule remains active.

The scheduler spends the current window's credit at the first decision where `D(t)` exceeds the predeclared threshold `3.0`.
If no trigger occurs, it spends the credit at the window deadline.
The scheduler does not claim instantaneous drift detection because its newest labels are at least 30 days mature.
Its claim is narrower: mature calibration evidence can decide when a fixed retraining credit is most useful.

The report must include trigger day, window, threshold, contributing bin, monitoring cohort range, residual components, and whether the spend was triggered or forced at deadline.

## 16. Equal-compute accounting

Define:

```text
K = optimizer steps per update credit
M = total update credits
B = training batch size
core example budget = K * M * B
```

Every matched Study B policy receives identical `K`, `M`, `B`, model, optimizer, sampler, loss, and seed set.
Study A methods receive the same core optimizer-example budget and fixed schedule, while method-specific auxiliary work is measured separately.

The report includes:

- Core optimizer steps.
- Core optimizer examples.
- Credits allocated and spent.
- Auxiliary forward and backward examples.
- Feature-processing rows.
- Monitoring forward examples.
- Wall-clock training and evaluation time.
- Peak host and accelerator memory where measurable.
- Checkpoint bytes and I/O time.

Quality is reported at 25, 50, 75, and 100 percent of the core budget.
The portfolio claim must not reduce compute accounting to wall-clock time alone.

## 17. Sampling policy

Each update samples 50 percent from the most recent legally available training window and 50 percent from a uniform reservoir over older legal records.
The exact recent-window duration and reservoir capacity are selected on the selection period and locked.
Monitoring examples and future records are excluded.
Correction records remain available according to their own reveal time.

The exposure ledger records every sampled training-record ID, update credit, step, sample weight, and correction relationship.
The sampler must be deterministic for a given seed and restored state.
If the legal pool is smaller than the requested batch, the configuration's replacement policy must be explicit and reported.
The locked V1 policy is sampling with replacement when at least one legal record exists and failing the scheduled update when none exists.

## 18. Evaluation protocol

The primary metric is final-label binary log loss over predictions made for click days 65 through 89.
Secondary metrics are PR-AUC, Brier score, ROC-AUC, calibration intercept, calibration slope, fixed-bin expected calibration error, and reliability tables.
PR-AUC is important because conversion is expected to be imbalanced.
ROC-AUC must not be presented alone.

Required slices are:

- Cold user based on no prior click by that user.
- Cold product based on no prior click for that product.
- Past-only user and product frequency bands.
- Product-price bins fit on burn-in.
- Device type.
- Positive examples by conversion-delay band.
- Chronological click-day blocks.

Every slice definition must be computed from information available at prediction time.
Small slices must display count and positive count and must be suppressed from ranking when below a locked minimum support.

### 18.1 Paired uncertainty

Methods are compared on the same final clicks.
Use paired contiguous-day block bootstrap with at least 2,000 replicates.
The primary analysis uses three-day blocks fixed before final evaluation.
Sensitivity results use one-day and seven-day blocks.
At least three fixed training seeds are required for final claims.

For each replicate, sample the same contiguous day blocks jointly for both methods and every matched training seed, compute the candidate-minus-control metric within each seed, and average those paired differences across seeds.
The primary interval is the percentile interval over those seed-averaged paired replicates.
For log loss and Brier score, report this paired difference and 95 percent interval.
For PR-AUC and ROC-AUC, recompute the metric on each paired bootstrap resample.
Report every seed-level point estimate in addition to the primary seed-averaged interval.
“Consistent across seeds” means every required seed-level log-loss point difference has the same favorable sign.

### 18.2 Scheduler success criterion

The calibration scheduler is considered supported only if all conditions hold against the predeclared fixed-deadline primary control:

- The paired final-period log-loss difference is below zero with its locked 95 percent interval below zero.
- The upper paired 95 percent bound for Brier-score degradation is no greater than `0.0005` and the absolute ECE point degradation is no greater than `0.002`.
- The core optimizer-example budget is identical.
- The conclusion is consistent across the required seeds.
- The conclusion is not reversed by the locked block-size sensitivity analysis.

If these conditions do not hold, the project reports a negative or inconclusive result.
The framework remains portfolio-worthy because the reproducible protocol and leakage controls are the main engineering contribution.

## 19. Leakage-control architecture

The repository must make leakage difficult by construction rather than by convention.

Required mechanisms are:

- Separate feature and truth data roots.
- A truth-access module not imported by learner, scheduler, features, sampler, or trainer packages.
- Runtime `available_at <= simulator_time` assertions for every training record.
- Prediction persistence before same-time reveal processing.
- Burn-in-only preprocessing statistics.
- Past-only cold and frequency features.
- Monitoring split exclusion from training.
- Immutable final-period configuration hash.
- A run manifest that refuses a dirty Git tree for final mode unless `--allow-dirty` is explicit.
- Mutation tests that deliberately shift reveal times, inject a forbidden field, leak a global frequency, reuse monitoring examples, and process reveals before predictions.

The final report includes a leakage-audit table with every control and pass status.

## 20. Checkpoint and resume contract

A checkpoint is written atomically at every daily decision and configurable intra-day intervals.
It contains model and optimizer state, all random-generator states, simulator time, click cursor, truth-reveal cursor, method state, scheduler state, sampler state, monitoring state, credit ledger, exposure ledger position, prediction ledger position, configuration hash, data manifest hash, code commit, and environment manifest hash.

`latesignal resume` refuses a changed configuration, code commit, prepared-data hash, feature policy, or dependency lock unless an explicit non-publication override is supplied.
A resumed run must produce the same prediction and credit ledgers as an uninterrupted run on the same stack.
Integration tests should terminate a synthetic run at every checkpoint boundary and compare resumed output hashes.

## 21. CLI contract

```text
latesignal data fetch --accept-license
latesignal data inspect
latesignal data prepare
latesignal protocol validate CONFIG
latesignal protocol estimate CONFIG
latesignal run CONFIG --out DIR
latesignal resume CHECKPOINT --out DIR
latesignal evaluate RUN_DIR
latesignal compare RUN_DIR...
latesignal report RUN_DIR --format text|json|html
latesignal reproduce MANIFEST --out DIR
```

Every command supports `--json` for machine-readable status.
`data fetch` is the only command that accesses the public internet in the normal workflow.
Experiment commands operate from locked local inputs.
No command silently overwrites a completed final run.
Archive inspection rejects absolute paths, parent traversal, symlinks, device files, duplicate members, excessive member counts, and an excessive expanded-size ratio before extraction.

Exit codes are:

| Code | Meaning |
| --- | --- |
| 0 | Command completed and all requested validations passed. |
| 1 | An experimental acceptance gate was not met. |
| 2 | Configuration, protocol, or feature policy was invalid. |
| 3 | Data artifact, license acknowledgement, or hash was invalid. |
| 4 | Runtime infrastructure failed or results are incomplete. |
| 5 | Internal consistency or reproducibility check failed. |

## 22. Repository layout

```text
late-signal/
  BUILD_PLAN.md
  README.md
  LICENSE
  DATA_LICENSE.md
  SECURITY.md
  CONTRIBUTING.md
  Makefile
  pyproject.toml
  uv.lock
  configs/
    data.yaml
    features.yaml
    protocol.yaml
    models/
      conversion_mlp.yaml
    methods/
      complete_wait.yaml
      immediate_fake_negative.yaml
      fixed_wait.yaml
      dfm.yaml
      fnw.yaml
      es_dfm.yaml
    schedulers/
      fixed_early.yaml
      fixed_midpoint.yaml
      fixed_deadline.yaml
      calibration_drift.yaml
    experiments/
      study_a.yaml
      study_b.yaml
      final.yaml
  src/latesignal/
    __init__.py
    cli.py
    errors.py
    contracts/
      config.py
      events.py
      records.py
      manifests.py
      results.py
    data/
      download.py
      inspect.py
      schema.py
      prepare.py
      quarantine.py
      manifests.py
    simulator/
      engine.py
      clock.py
      boundary.py
      oracle.py
      ledger.py
      checkpoint.py
    features/
      policy.py
      hashing.py
      numeric.py
      online_history.py
      batch.py
    models/
      conversion_mlp.py
      logistic.py
      lightgbm.py
    methods/
      base.py
      complete_wait.py
      immediate_fake_negative.py
      fixed_wait.py
      dfm.py
      fnw.py
      es_dfm.py
      oracle_reference.py
    scheduling/
      base.py
      credit.py
      fixed.py
      calibration_drift.py
      monitoring.py
    training/
      trainer.py
      sampler.py
      budget.py
      reproducibility.py
    evaluation/
      metrics.py
      calibration.py
      slices.py
      bootstrap.py
      compare.py
    experiments/
      runner.py
      study_a.py
      study_b.py
      final.py
      estimate.py
    reporting/
      model.py
      tables.py
      figures.py
      html.py
      manifest.py
  tests/
    unit/
    property/
    integration/
    leakage/
    fixtures/
      synthetic/
      equations/
      manifests/
  docs/
    architecture.md
    dataset-and-license.md
    timestamp-unit-audit.md
    event-time-semantics.md
    leakage-model.md
    delayed-methods.md
    scheduler.md
    experimental-protocol.md
    reproducibility.md
    limitations.md
  results/
    README.md
    published/
  .github/workflows/
    ci.yml
    gpu-smoke.yml
    final-reproduction.yml
```

Raw data, prepared rows, checkpoints, ordinary runs, model weights trained on restricted data, and large figures are ignored by Git.
Only selected aggregate result tables and small synthetic artifacts enter the repository.

## 23. Implementation stack

The researched starting stack is Python 3.12, PyTorch 2.11, Polars, Apache Arrow and Parquet, NumPy, scikit-learn, LightGBM, Pydantic 2, PyYAML, Typer, xxhash, blake3, pytest, Hypothesis, Ruff, mypy, and uv.
The implementation agent must verify current stable versions and compatibility before locking `pyproject.toml` and `uv.lock`.

Polars should perform streaming scans and preparation so the complete source file does not need to reside in memory.
PyArrow defines the explicit Parquet schema and metadata.
Pydantic validates authored experiment configuration and forbids unknown keys.
Typer exposes the CLI.
Matplotlib or Altair may generate static figures, but an interactive web application is not a V1 dependency.

The full experiment should target an NVIDIA GPU with at least 8 GB VRAM and approximately 16 GB host memory.
A CPU path must run the complete synthetic fixture and small real-data smoke subset.
Apple MPS may be used for development but does not replace the final locked accelerator run.
Reserve at least 30 GB local disk for archive, extracted data, prepared partitions, checkpoints, and temporary results.

### 23.1 Feasibility gate

Before a full selection or final matrix, `latesignal protocol estimate` runs a bounded synthetic benchmark and a configured real-data pilot over at most two click days.
It measures preparation throughput, examples per training second, prediction throughput, checkpoint bytes, and report growth.
It expands the authored method, policy, seed, and configuration matrix into exact run and optimizer-step counts.
The estimate reports expected GPU-hours as a measured range, peak working disk, retained artifact disk, host-memory requirement, and the assumptions behind each projection.
`configs/experiments/final.yaml` must declare `max_runs`, `max_gpu_hours`, `max_working_disk_gb`, and `max_retained_disk_gb` supplied by the user for the actual machine.
Protocol validation refuses a final matrix whose upper estimate exceeds a cap.
Reducing the matrix requires a new protocol hash before final-period access rather than silently dropping failed or slow runs.

## 24. Verification strategy

### 24.1 Data tests

- Verify archive and extracted hashes.
- Verify unsafe or oversized archive members are rejected before extraction and promotion.
- Verify exact field count and locked raw schema.
- Verify accepted plus quarantined equals parsed rows.
- Verify monotonic click order or record every violation.
- Verify the timestamp-scale inference is unique.
- Verify positive reveal times and negative maturity times.
- Verify inconsistent label and delay combinations are quarantined.
- Verify prepared partitions preserve row identity and count.
- Verify no raw or prepared path is tracked by Git.

### 24.2 Simulator tests

- Predict before a reveal at the same timestamp.
- Confirm that a click at `D(1) - epsilon` does not enter the fully mature day-0 initialization pool until `D(31) - epsilon`.
- Reveal a positive exactly at its conversion time.
- Mature a negative exactly at 30 days.
- Never reveal a negative earlier.
- Never reveal a positive later than its stored reveal time.
- Process each click and final truth once.
- Resume from every checkpoint boundary.
- Extend the clock beyond the last click until final labels mature.

### 24.3 Property tests

- Every training record satisfies `available_at <= current_time`.
- Monitoring IDs never occur in the exposure ledger.
- The final prediction ledger is immutable.
- All Study B schedulers spend exactly `M` credits and `K * M` steps.
- No allocation window spends more than one credit.
- A credit is always spent by its deadline.
- Cold status depends only on prior clicks.
- Identical seed, environment, and checkpoint produce identical ledgers.

### 24.4 Method tests

- Verify DFM, FNW, and ES-DFM loss components against hand-calculated golden examples.
- Verify each method's click, early conversion, late conversion, and final-negative event-to-record sequence.
- Verify DFM's stable censored likelihood against direct high-precision calculations.
- Verify FNW and ES-DFM detached weights and clipping at boundary probabilities.
- Verify method state survives checkpoint round trips.
- Verify correction records preserve their relationship to provisional records.
- Verify methods never open the truth store directly.
- Verify the oracle reference is visually and programmatically excluded from deployable ranking.

### 24.5 Leakage mutation tests

- Add `Sale` to a batch and require failure.
- Add conversion delay to a batch and require failure.
- Fit a normalizer on final rows and require manifest mismatch.
- Compute cold-user status from global counts and require a golden divergence.
- Move reveal processing before prediction and require a boundary test failure.
- Add monitoring IDs to the sampler and require failure.
- Shift a truth record earlier than legal and require an availability assertion.

### 24.6 End-to-end tests

- Run fetch against a small local fake archive without internet.
- Inspect, prepare, train, checkpoint, resume, evaluate, compare, and report a synthetic stream through the public CLI.
- Run a CPU smoke subset of the real prepared schema without committing rows.
- Reproduce a published aggregate result from a locked manifest and locally available data.
- Verify a dirty final run is refused unless explicitly overridden.

## 25. Ordered implementation milestones

### Milestone 0 - Licensing and data audit

Create repository policies, the license acknowledgement flow, downloader, archive verifier, raw schema contract, inspector, timestamp-unit inference, quarantine reporting, and data documentation.

Gate:

- The official archive is downloaded only after explicit acknowledgement.
- Its actual size and hashes are persisted locally.
- Timestamp and delay units resolve uniquely or preparation refuses to proceed.
- Every parsed row is accepted or quarantined with a reason.

### Milestone 1 - Synthetic event-time vertical slice

Create a small generated click stream with known delayed positives and mature negatives.
Implement contracts, clock, prediction ledger, truth oracle, availability ledger, fixed daily scheduler, a tiny model, checkpointing, and the CLI run path.

Gate:

- A complete CPU run produces predictions, legal reveals, updates, metrics, and a manifest.
- Same-time boundaries and final maturity extension pass.
- Interrupted and resumed runs have identical ledgers.

### Milestone 2 - Real-data preparation and feature isolation

Implement Polars scans, click IDs, day partitioning, field hashing, burn-in statistics, online history, feature-policy validation, and truth-store separation.

Gate:

- Prepared counts reconcile with inspection counts.
- Feature batches contain only allowlisted click-time data.
- Cold and frequency features pass past-only tests.
- The raw data is not required in memory at once.

### Milestone 3 - Shared model and offline sanity references

Implement the conversion MLP, trainer, deterministic sampler, budget counters, mature-label logistic regression, mature-label LightGBM, and calibration evaluation.

Gate:

- The model overfits a tiny deterministic fixture.
- A label-shuffled fixture performs at chance.
- Offline references run on chronological mature labels.
- Budget and exposure ledgers reconcile exactly.

### Milestone 4 - Delayed-label method suite

Implement complete wait, immediate fake negative, fixed wait, DFM, FNW, ES-DFM constant-wait, and the separate oracle reference.

Gate:

- Every published-method equation has a golden test and citation.
- Study A runs all methods with the same fixed schedule and core budget.
- Auxiliary compute is reported separately.
- No result is described as a published-number reproduction.

### Milestone 5 - Credit schedulers

Implement credit windows, fixed early, midpoint, deadline, mature monitoring split, calibration residuals, threshold selection, and `CalibrationDriftCreditScheduler`.

Gate:

- Every scheduler spends exactly the same credits and steps.
- Monitoring rows never train the model.
- Triggers are reproducible and fully auditable.
- A synthetic calibration shift produces the expected earlier trigger.

### Milestone 6 - Locked evaluation and uncertainty

Implement final-period sealing, metrics, slices, calibration tables, paired block bootstrap, seed aggregation, intermediate-budget analysis, and compute Pareto tables.

Gate:

- Bootstrap code passes analytical and simulation sanity cases.
- Empty and low-support slices are handled without fabricated metrics.
- All methods are evaluated on the same persisted final predictions and truth IDs.
- The report exposes seed-level and paired-difference results.
- The cost estimator enumerates every planned run and the final matrix fits the authored compute and disk caps.

### Milestone 7 - Protocol freeze and final run

Select permitted hyperparameters using only outcomes for click days 25 through 34, preserve days 35 through 64 as a maturity embargo, generate a signed or hashed protocol lock before day-65 scoring, delete exploratory final outputs, and rerun every final experiment from day 0.

Gate:

- No final-period metric influenced configuration selection.
- No outcome from embargo click days 35 through 64 influenced configuration selection.
- The final configuration, data, code, and environment hashes are fixed before scoring.
- At least three seeds complete.
- Every run spends its required compute budget or is marked incomplete.

### Milestone 8 - Portfolio publication

Publish aggregate results, methodology, failure cases, leakage audit, compute accounting, limitations, and a short demonstration.

Gate:

- No raw or prepared Criteo row is committed.
- No claim exceeds the locked statistical result.
- A negative scheduler result is reported honestly if the success criterion is unmet.
- A new user with accepted data access can reproduce the pipeline from the manifest.

## 26. Portfolio-ready core and extended V1

The portfolio-ready core includes the audited data pipeline, event-time simulator, shared model, complete-cohort wait, immediate fake-negative BCE, FNW, fixed-deadline scheduling, `CalibrationDriftCreditScheduler`, three locked final seeds, paired uncertainty, compute accounting, and a static aggregate report.
It may be released after Milestone 7 when those components meet every applicable leakage and final-protocol gate.
Fixed-wait, DFM, ES-DFM, the two additional fixed scheduler timings, offline model references, and the broadest slice set form the extended V1 research matrix.
The core report must identify deferred extended components and cannot imply they ran.
This cut preserves the central ML contribution if the full literature-transfer matrix exceeds the authored feasibility caps.

## 27. CI design

Hosted CPU CI runs Ruff format and lint, mypy, pytest, Hypothesis profiles, schema validation, synthetic end-to-end tests, report generation, license-flow tests using a fake archive, and Git-ignore guards.
CI must not download the Criteo archive.
CI must not require a GPU.

A trusted self-hosted GPU workflow runs a bounded synthetic smoke and optionally a user-provisioned real-data experiment.
Public fork pull requests must never run code on the self-hosted GPU runner.
The workflow must use a hardware-specific concurrency group and clean only its exact run directory.
Final experiments should be manual, preserve manifests and aggregate outputs, and not upload restricted raw data or checkpoints by default.

## 28. Reporting deliverables

The final static report must contain:

- Dataset provenance and license boundary.
- Exact chronology and information-availability diagram.
- Feature allowlist and leakage audit.
- Study A method definitions and equal-budget table.
- Study B scheduler definitions and credit ledger summary.
- Final overall metrics with paired intervals.
- Calibration and reliability evidence.
- Cold-user, cold-product, device, price, frequency, and delay slices.
- Quality at 25, 50, 75, and 100 percent of the compute budget.
- Core and total cost tables.
- Trigger timeline for each scheduler seed.
- At least one deliberate leakage failure demonstration.
- Limitations and threats to validity.
- Exact reproduction manifest.

Figures must include underlying small aggregate CSV or Parquet tables.
Charts should never hide sample counts, uncertainty, or the unattainable status of the oracle reference.

## 29. Risks and mitigations

### Ambiguous source units

The public field description does not fully specify timestamp encoding.
The unique-scale inference and fail-closed audit prevent a silent calendar error.

### Dataset shift from cited papers

DFM and related papers may use a different Criteo dataset.
The project describes equation-level method transfer and avoids published-number replication claims.

### Scheduler signal staleness

Mature monitoring evidence is at least 30 days delayed.
The contribution is framed as compute allocation from mature calibration evidence rather than real-time drift detection.

### False novelty claim

Joint conversion and delay modeling is established literature.
The novelty claim is the reproducible simulator, leakage audit, and equal-credit scheduler experiment.

### Unequal hidden compute

Method-specific auxiliary models can consume substantial work.
Core optimizer budgets and total system cost are reported separately.

### Final-period overfitting

The 30-day embargo ensures every selection click is mature before final scoring, the protocol is hashed before day-65 scoring, and the final run restarts from day 0.
Exploratory final metrics must not be kept or consulted during development.

### Hardware nondeterminism

Seeds, deterministic settings, locks, and manifests reduce variance.
Multiple seeds and paired uncertainty replace unsupported bitwise guarantees.

### Data licensing

The fetch acknowledgement, ignored data roots, aggregate-only publication, and separate data-license document keep restricted artifacts out of the repository.

## 30. Definition of done

LateSignal V1 is complete only when all conditions below are true.

- The official source artifact has a verified local manifest and explicit license acknowledgement.
- Timestamp and reveal semantics have a documented, tested, unique interpretation.
- Every row is accepted or quarantined without silent loss.
- Truth cannot reach a learner before its legal availability time.
- Same-time prediction occurs before reveal.
- Study A changes only delayed-label strategy under its matched protocol.
- Study B changes only scheduling under identical credits, steps, learner, sampler, and loss.
- Final predictions for days 65 through 89 are sealed before eventual truth evaluation.
- Every configuration-selection label comes from click days 34 or earlier and is mature before the final period begins.
- Metrics include calibration, imbalance-aware quality, slices, compute, seeds, and paired intervals.
- Checkpoint resume reproduces uninterrupted ledgers on the same stack.
- Deliberate leakage mutations are detected.
- Published claims follow the predeclared success criteria.
- No restricted row-level data enters Git history.

## 31. Implementation-agent rules

The implementation agent should begin with the synthetic simulator rather than waiting for a long full-data model run.
It should make illegal future access structurally difficult and test every time boundary.
It should not change splits, thresholds, metrics, block-bootstrap policy, or candidate schedulers after final-period inspection.
It should preserve method equations and citations in code comments and documentation.
It should keep method selection and scheduler selection as separate studies.
It should report negative findings rather than tuning until a claim becomes positive.
It should not add a FastAPI service, live dashboard, causal bidding simulation, private-ad stress test, or complex hazard model before V1 is complete.
It should run the nearest complete CLI path after every milestone.

## 32. Authoritative references

- [Criteo Sponsored Search Conversion Log dataset](https://ailab.criteo.com/criteo-sponsored-search-conversion-log-dataset/)
- [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- [Modeling delayed feedback in display advertising](https://doi.org/10.1145/2623330.2623634)
- [A nonparametric delayed feedback model for conversion rate prediction](https://arxiv.org/abs/1907.06558)
- [Handling delayed feedback in conversion rate prediction](https://arxiv.org/abs/2002.02068)
- [ES-DFM paper](https://ojs.aaai.org/index.php/AAAI/article/view/16587)
- [ES-DFM preprint](https://arxiv.org/abs/2012.03245)
- [ES-DFM reference implementation](https://github.com/ThyrixYang/es_dfm)
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [Polars user guide](https://docs.pola.rs/)
- [Apache Parquet format](https://parquet.apache.org/docs/)

Later literature such as GDFM, DEFUSE, DEFER, and IF-DFM may be discussed as context after the required baselines are complete.
They should not expand V1 unless the locked core study exposes a specific scientific need.
