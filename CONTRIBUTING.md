# Contributing

LateSignal is an event-time delayed-feedback benchmark with strict temporal and licensing boundaries.

Before committing and pushing a change, run `make check` and the closest public CLI workflow affected by the change.
Tests must preserve prediction-before-reveal ordering, legal label availability, monitoring exclusion, deterministic ledgers, and equal-compute accounting.
Do not weaken a protocol gate to accommodate an implementation or infrastructure failure.

Do not commit raw or prepared Criteo rows, acknowledgement records, quarantine rows, checkpoints, credentials, profiler output, or ordinary experiment artifacts.
Do not update locked final-period choices after inspecting final-period outcomes.
Generated changelogs and generated results should be updated through their owning command rather than edited manually.
