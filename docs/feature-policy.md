# Feature policy and truth isolation

The V1 model receives only 17 hashed categorical click attributes, two transformed numeric click attributes, and two numeric missing indicators.
The field allowlist is authored in `configs/features.yaml` and validated as an exact set whenever a model training batch is constructed.
An injected outcome, delay, timestamp, history slice, or unknown field causes a consistency failure.

Categorical values use a field-specific input namespace and the locked seed:

```text
xxhash64(field_name + ":" + raw_value, field_seed) mod bucket_count
```

High-cardinality fields and other categorical fields use separate authored bucket counts.
Later models allocate separate embedding tables per field, so equal numeric buckets across fields do not imply a shared category.

`nb_clicks_1week` and `product_price` use `log1p` for inspected nonnegative values.
The inspected `-1` sentinel becomes a missing indicator and a zero standardized value.
Clipping quantiles, clipped means, and clipped scales are fit only on click days 0 through 14 and recorded in the preparation manifest.

The official source is not globally ordered by click timestamp.
Preparation uses Polars' streaming engine to sort accepted rows by numeric click timestamp with the original raw-row index as the deterministic tie breaker.
Cold user, cold product, and prior-click counts are calculated while the resulting chronological batches are streamed.
The current row is classified from counters before those counters are incremented.
These fields support evaluation slices and are not part of the V1 model allowlist.

Prepared feature files and truth files have different explicit Arrow schemas and different physical roots.
Feature Parquet files contain no final label, conversion delay, reveal time, or maturity time.
Positive truth is partitioned by reveal day and negative truth by maturity day.
Learner-facing code receives legal typed records from the oracle rather than opening truth files directly.

## Observed preparation evidence

The reviewed official corpus produced 15,924,859 feature rows and 15,924,859 truth rows.
The truth store contains 1,718,566 positive reveals and 14,206,293 negative maturities.
The preparation manifest inventories 7,639 published Parquet files totaling 2,569,185,501 bytes, and an independent verification rehashed every file successfully.
The feature store spans normalized click days 0 through 90 and contains no `Sale`, `SalesAmountInEuro`, `time_delay_for_conversion`, `final_label`, or `available_at_seconds` column.
The locked experimental protocol uses click days 0 through 89 and does not silently reinterpret the observed extra boundary day.
Independent scans found zero disagreements between `cold_user` and zero prior-user clicks and zero disagreements between `cold_product` and zero prior-product clicks.
