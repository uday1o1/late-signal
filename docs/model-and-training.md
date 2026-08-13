# Shared model and training contract

LateSignal uses one field-aware PyTorch conversion MLP for matched online studies.
Each of the 17 categorical fields owns a separate embedding table, even when two fields use the same bucket count.
The four numeric inputs are the two transformed values and their two missing-value indicators.

The authored backbone is `256 -> 128 -> 64 -> 1` with SiLU activations, LayerNorm after the first two hidden layers, and the selected dropout value after those same layers.
The initial optimizer is AdamW with a batch size of 2,048 and gradient-norm clipping at 5.0.
The selection grid remains locked in `BUILD_PLAN.md` and must be materialized in the protocol configuration before selection outcomes are read.

The sampler draws half of each batch from the recent legal window and half from a deterministic hash-priority reservoir over older legal records.
Sampling uses replacement whenever at least one legal record exists.
A scheduled update with no legal records fails with `INSUFFICIENT_LEGAL_POOL`.
Sampler random state, retained records, and event time are checkpointed.

Every sampled record produces one exposure-ledger row.
The budget counter must exactly equal the number of exposure rows before a credit is considered complete.
This accounts for core optimizer examples independently from wall-clock time or future method-specific auxiliary work.

Logistic regression and LightGBM are offline sanity references only.
Their training split rejects clicks after the training cutoff and labels unavailable at that cutoff.
Their evaluation clicks must occur strictly after the cutoff.
Both references are explicitly ineligible for the equal-compute online ranking because their optimization and representation differ.

Calibration evaluation uses ten predeclared equal-width probability bins.
It reports calibration intercept, calibration slope, expected calibration error, and a complete reliability table including empty bins.
