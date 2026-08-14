# Leakage model

LateSignal treats information availability as a security boundary.
A statistically plausible result is invalid when any learner, scheduler, transform, or selection decision observes information that was unavailable at the simulated time.

## Protected information

The protected fields are the final conversion label, conversion delay, sales amount, positive reveal time, negative maturity status, and any statistic derived using future clicks or outcomes.
The protected periods are the maturity embargo on click days 35 through 64 and the final evaluation period on click days 65 through 89.

The threat model includes accidental feature inclusion, incorrect same-time ordering, early truth exposure, preprocessing on future rows, global rather than past-only frequency features, monitoring reuse, selection on embargo outcomes, final-period tuning, and mismatched comparison IDs.

## Structural controls

Feature and truth rows are written under separate roots with different Arrow schemas.
The feature schema excludes `Sale`, `SalesAmountInEuro`, `time_delay_for_conversion`, `final_label`, and `available_at_seconds`.
The feature policy exposes an exact allowlist and rejects both missing and extra training columns.

The learner, feature, sampler, trainer, and scheduler layers consume emitted observations rather than opening truth partitions directly.
Truth is delivered by the oracle only when `available_at <= simulator_time`.
The availability ledger and trainer assert that boundary again before sampling and optimization.

The engine writes predictions before processing reveals at an equal timestamp.
Prediction ledgers become immutable when sealed.
Final evaluation rejects unsealed rows, duplicate click IDs, out-of-period rows, and comparisons with mismatched IDs, truth, click days, or seeds.

Numeric statistics and price boundaries are fitted on burn-in only.
Cold-user, cold-product, and frequency values are based only on preceding clicks.
Monitoring membership uses a deterministic hash split, and the sampler permanently excludes monitoring IDs.

Protocol selection accepts outcomes only for click days 25 through 34.
Strict selection contracts reject evidence that accessed embargo outcomes or final-period metrics.
The final protocol, data, code, environment, seeds, and compute choice are hashed before final scoring.

## Deliberate mutations

The leakage suite must prove that each nearby legal control passes and each seeded defect fails for its intended reason.
The required mutations are:

| Mutation | Required detection |
| --- | --- |
| Add `Sale` to a training batch | Feature allowlist rejection. |
| Add conversion delay to a training batch | Feature allowlist rejection. |
| Fit a normalizer on final rows | Preparation or manifest identity mismatch. |
| Compute cold status from global counts | Divergence from the past-only golden result. |
| Process reveals before prediction | Same-time boundary trace mismatch. |
| Add monitoring IDs to training | Sampler or exposure-ledger rejection. |
| Shift truth earlier than legal | Availability assertion failure. |

The publication report lists each mutation, its pass status, and concise evidence.
A missing or failed leakage control blocks a final report.

## Residual risks

Code can still contain a logic error that is not represented by a mutation or property test.
Hash locks establish identity and order but do not prove scientific validity.
External libraries and hardware may change numerical behavior within the supported statistical reproducibility boundary.
These residual risks are addressed with simple interfaces, strict schemas, paired controls, multiple seeds, reviewable ledgers, and narrow public claims.
