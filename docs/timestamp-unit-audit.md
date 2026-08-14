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

## Observed official artifact audit

The reviewed official archive has SHA-256 `b49b4135b8564235eba04f6400e663f5456b1a153f49ff504501923a9f47dbf5` and contains a 6,426,808,162-byte data member with SHA-256 `a2ee46feec6008f901ab1ce5e51a44b591b5194285ef86a141b5cc54dbde0567`.
The complete inspection parsed 15,995,634 data rows, accepted 15,924,859 rows, and quarantined 70,775 rows.
The reconciliation gate passed exactly.
The quarantine reasons were 65,417 empty `product_title` values and 5,380 inconsistent sale-delay combinations.

Only the `1` second-per-raw-unit candidate passed.
The accepted click span was 90.999988 days and the maximum accepted positive delay was 29.974745 days.
The other millisecond, microsecond, and nanosecond candidates failed the click-span constraint.

The official member is not globally ordered by click time and contains 7,954,968 adjacent order violations.
The inspector records every violating raw-row index.
Preparation therefore performs a deterministic bounded-memory sort by numeric click timestamp and then original raw-row index before calculating any past-only feature.
