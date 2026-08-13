# Compute-matched credit scheduling

Study B holds the delayed-label method, conversion model, loss, optimizer, sampler, steps per credit, batch size, total credits, and seeds fixed.
Only the time at which each allocation window spends its one credit changes.

The V1 horizon is the half-open interval from click day 31 through the start of day 90.
Five-day windows produce 12 credits, including the final partial window from day 86 through day 90.
Early, midpoint, and deadline schedules spend at the exact daily boundaries defined in `BUILD_PLAN.md`.
Credits cannot roll over or be spent twice.

`CalibrationDriftCreditScheduler` evaluates a deterministic 10 percent split of the most recent seven fully mature click cohorts every day.
Membership is `xxhash64(click_id, monitor_seed) mod 10 == 0` with seed `20260813`.
Monitoring identifiers are permanent sampler exclusions.

Fresh checkpoint predictions are grouped into ten predeclared equal-width probability bins.
A bin is eligible only with at least 1,000 examples and Bernoulli variance sum of at least 25.
The scheduler computes the absolute signed residual sum divided by the square root of the variance sum plus `1e-8`.
It spends when the maximum eligible residual strictly exceeds `3.0`, otherwise the window deadline forces the spend.

Every daily audit record includes the checkpoint hash, mature cohort range, sample count, each bin's residual components, the contributing bin, trigger score, window, decision time, and reason.
The claim is limited to scheduling fixed retraining credits using mature calibration evidence.
It is not an instantaneous drift detector.
