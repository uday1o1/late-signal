# Limitations and threats to validity

LateSignal is an offline research benchmark rather than a production bidding system.
Its primary value is the leakage-audited simulator, equal-compute design, reproducibility boundary, and honest uncertainty analysis.

## Dataset scope and license

The study uses one sponsored-search conversion log with a fixed 30-day attribution window.
Results may not generalize to display advertising, other auctions, different attribution policies, organic traffic, or current production distributions.
The source data has a separate CC BY-NC-SA 4.0 license and is restricted to permitted noncommercial use.
No raw row, prepared row, checkpoint trained from restricted data, or row-level prediction is published by this repository.

## Method transfer

DFM, FNW, and ES-DFM are equation-level transfers into one shared event-time and model framework.
The original papers may use different datasets, features, sampling rules, architectures, or optimization details.
LateSignal therefore does not claim to reproduce published numbers.
The ES-DFM constant-wait variant is deliberately narrower than later extensions in the literature.

## Attribution semantics

A negative becomes known only at the end of the 30-day window.
A positive becomes known at its recorded conversion time.
These rules model the dataset contract, not lifetime user behavior or causal incrementality.
The project does not model multi-touch attribution, cancellations, refunds, repeated conversions, auction prices, or delayed revenue.

## Scheduler signal

Calibration evidence comes only from fully mature reserved cohorts and is therefore at least 30 days stale.
The scheduler allocates a fixed training budget from delayed evidence rather than detecting drift in real time.
A trigger may arrive too late for an operational response, and a statistically supported final result does not establish causal business value.

The calibration scheduler has a predeclared multi-part success criterion.
If the paired log-loss interval, bounded Brier and ECE degradation, identical compute, seed consistency, or block-size sensitivity condition fails, the result is reported as negative or inconclusive.

## Selection and multiplicity

The finite candidate grids and staged selection reduce flexibility but do not eliminate selection bias.
Only click days 25 through 34 may influence hyperparameters.
Days 35 through 64 form a maturity embargo, and days 65 through 89 are inspected only after locking.
The project does not claim that the selected configuration is globally optimal.

The final comparisons include multiple methods, metrics, slices, and sensitivity analyses.
The primary metric and scheduler criterion are predeclared, while secondary values are descriptive and should not be read as independently corrected discoveries.

## Compute comparability

Matched core optimizer examples do not make every operation identical.
DFM has an additional delay head, ES-DFM trains auxiliary models, and monitoring policies perform inference outside the core budget.
The report therefore separates core optimizer examples, auxiliary optimizer work, monitoring forwards, wall time, peak memory, and total system cost.
Wall-clock results remain machine specific.

## Calibration and slices

Fixed-bin ECE depends on the locked ten-bin definition.
Calibration intercept and slope can be unstable with limited support or nearly constant predictions.
Slices with insufficient examples or positives display their counts and suppression reason without a fabricated ranking metric.
Broad slice results remain observational and can reflect feature availability and population composition.

## Reproducibility

Exact ledger reproduction is expected only on the same supported software and hardware stack.
GPU kernels, drivers, compilers, and dependency changes can alter floating-point order.
Fixed seeds and deterministic settings reduce variation but do not prove cross-hardware bitwise identity.
Final claims therefore require three seeds and paired uncertainty.

## Non-goals

LateSignal does not provide a live dashboard, online service, causal bidding policy, private advertising stress test, multi-objective value model, or complex nonparametric hazard system.
Those additions would require new data, security, evaluation, and operational assumptions outside V1.
