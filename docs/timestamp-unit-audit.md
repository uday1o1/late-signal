# Timestamp-unit audit

The public field description does not state the encoding unit shared by `click_timestamp` and `time_delay_for_conversion`.
LateSignal infers the unit before preparation and refuses ambiguous input.

The inspector tests the shared seconds-per-raw-unit multipliers `1`, `1e-3`, `1e-6`, and `1e-9`.
A multiplier passes only when the accepted click span lies between 89 and 91 days and every accepted positive conversion delay lies between zero and 30 days inclusive.
Negative `-1` delay sentinels are validated before inference and are never interpreted as times.
Exactly one multiplier must pass.

The immutable inspection manifest records every candidate, each constraint result, and the selected multiplier.
Zero or multiple passing candidates produce `AMBIGUOUS_TIME_UNIT` and no inspection manifest is published.

After selection, `T0` is the minimum accepted normalized click time.
Click day `d` is defined by `floor((click_time_seconds - T0) / 86400)`.
Every day interval is half-open as `[T0 + d * 86400, T0 + (d + 1) * 86400)`.
This exact rule is recorded in the inspection manifest and is reused by later preparation and simulation stages.

Malformed rows and label-delay inconsistencies are written to the quarantine report by zero-based raw data-row index and reason code.
The inspector asserts that accepted rows plus quarantined rows equals parsed data rows.
Duplicate raw rows are counted but remain accepted when otherwise valid.
