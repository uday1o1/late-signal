"""Bounded in-memory access to verified runtime feature caches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from numpy.typing import NDArray
from torch import Tensor

from latesignal.data.schema import CATEGORICAL_CLICK_FIELDS, NUMERIC_CLICK_FIELDS
from latesignal.errors import ConsistencyError
from latesignal.features.cache import FeatureCache
from latesignal.models.conversion_mlp import CategoricalSpec


@dataclass(frozen=True, slots=True)
class FeatureTensorBatch:
    categorical: dict[str, Tensor]
    numeric: Tensor

    def to(self, device: torch.device) -> FeatureTensorBatch:
        return FeatureTensorBatch(
            categorical={field: values.to(device) for field, values in self.categorical.items()},
            numeric=self.numeric.to(device),
        )


class RuntimeFeatureStore:
    """Load one verified policy once and serve compact feature references to runs."""

    def __init__(self, cache: FeatureCache, *, build_id_lookup: bool = True) -> None:
        self.cache = cache
        self.categorical_fields = tuple(sorted(CATEGORICAL_CLICK_FIELDS))
        self.numeric_fields = tuple(sorted(NUMERIC_CLICK_FIELDS))
        rows = cache.rows
        self.click_ids: NDArray[np.void] = np.empty(rows, dtype="V32")
        self.click_times = np.empty(rows, dtype=np.float64)
        self.click_days = np.empty(rows, dtype=np.int16)
        self.categorical = np.empty((rows, len(self.categorical_fields)), dtype=np.uint32)
        self.numeric = np.empty((rows, len(self.numeric_fields) * 2), dtype=np.float32)
        self.cold_user = np.empty(rows, dtype=np.bool_)
        self.cold_product = np.empty(rows, dtype=np.bool_)
        self.prior_user_clicks = np.empty(rows, dtype=np.int64)
        self.prior_product_clicks = np.empty(rows, dtype=np.int64)
        self.product_price = np.empty(rows, dtype=np.float64)
        self.device_type_codes = np.empty(rows, dtype=np.uint16)
        device_values: list[str] = []
        device_indexes: dict[str, int] = {}
        cursor = 0
        try:
            for path in cache.files:
                table = pq.ParquetFile(path).read()
                count = table.num_rows
                end = cursor + count
                if end > rows:
                    raise ConsistencyError("Runtime feature cache exceeds its declared row count")
                self.click_ids[cursor:end] = np.asarray(table["click_id"].to_pylist(), dtype="V32")
                self.click_times[cursor:end] = table["click_time_seconds"].to_numpy(
                    zero_copy_only=False
                )
                self.click_days[cursor:end] = table["click_day"].to_numpy(zero_copy_only=False)
                for column, field in enumerate(self.categorical_fields):
                    values = table[f"{field}_bucket"].to_numpy(zero_copy_only=False)
                    if np.any(values >= cache.policy.bucket_count(field)):
                        raise ConsistencyError("Runtime categorical bucket exceeds its policy")
                    self.categorical[cursor:end, column] = values
                numeric_column = 0
                for field in self.numeric_fields:
                    self.numeric[cursor:end, numeric_column] = table[f"{field}_value"].to_numpy(
                        zero_copy_only=False
                    )
                    self.numeric[cursor:end, numeric_column + 1] = table[
                        f"{field}_missing"
                    ].to_numpy(zero_copy_only=False)
                    numeric_column += 2
                self.cold_user[cursor:end] = table["cold_user"].to_numpy(zero_copy_only=False)
                self.cold_product[cursor:end] = table["cold_product"].to_numpy(zero_copy_only=False)
                self.prior_user_clicks[cursor:end] = table["prior_user_clicks"].to_numpy(
                    zero_copy_only=False
                )
                self.prior_product_clicks[cursor:end] = table["prior_product_clicks"].to_numpy(
                    zero_copy_only=False
                )
                self.product_price[cursor:end] = table["product_price"].to_numpy(
                    zero_copy_only=False
                )
                part_devices = [str(value) for value in table["device_type"].to_pylist()]
                for value in part_devices:
                    if value not in device_indexes:
                        if len(device_values) >= np.iinfo(np.uint16).max:
                            raise ConsistencyError("Runtime device-type dictionary is too large")
                        device_indexes[value] = len(device_values)
                        device_values.append(value)
                self.device_type_codes[cursor:end] = np.fromiter(
                    (device_indexes[value] for value in part_devices),
                    dtype=np.uint16,
                    count=count,
                )
                cursor = end
        except (OSError, pa.ArrowException) as error:
            raise ConsistencyError("Verified runtime feature cache could not be loaded") from error
        if cursor != rows:
            raise ConsistencyError("Runtime feature cache row count changed during load")
        if rows == 0 or np.any(np.diff(self.click_times) < 0.0):
            raise ConsistencyError("Runtime features are not globally chronological")
        if (
            np.any(self.click_days < 0)
            or np.any(self.click_days > 90)
            or np.any(np.diff(self.click_days) < 0)
        ):
            raise ConsistencyError("Runtime feature cache contains an invalid click day")
        if np.any(self.cold_user != (self.prior_user_clicks == 0)) or np.any(
            self.cold_product != (self.prior_product_clicks == 0)
        ):
            raise ConsistencyError("Runtime feature history slices are internally inconsistent")
        if not np.isfinite(self.numeric).all() or not np.isfinite(self.product_price).all():
            raise ConsistencyError("Runtime feature cache contains a nonfinite numeric value")
        self.device_types = tuple(device_values)
        self.day_ranges = self._day_ranges()
        self._id_lookup: dict[bytes, int] | None = None
        if build_id_lookup:
            self._id_lookup = self._build_id_lookup()

    def _day_ranges(self) -> dict[int, slice]:
        ranges: dict[int, slice] = {}
        for day in range(91):
            start = int(np.searchsorted(self.click_days, day, side="left"))
            end = int(np.searchsorted(self.click_days, day, side="right"))
            if start != end:
                ranges[day] = slice(start, end)
        if any(np.any(self.click_days[item] != day) for day, item in ranges.items()):
            raise ConsistencyError("Runtime feature click days are not contiguous")
        return ranges

    def _build_id_lookup(self) -> dict[bytes, int]:
        lookup = {bytes(value): index for index, value in enumerate(self.click_ids)}
        if len(lookup) != self.cache.rows:
            raise ConsistencyError("Runtime feature cache contains duplicate click IDs")
        return lookup

    @property
    def prepared_manifest_sha256(self) -> str:
        return self.cache.prepared_manifest_sha256

    @property
    def id_lookup(self) -> dict[bytes, int]:
        if self._id_lookup is None:
            self._id_lookup = self._build_id_lookup()
        return self._id_lookup

    @property
    def categorical_specs(self) -> dict[str, CategoricalSpec]:
        return {
            field: CategoricalSpec(
                bucket_count=self.cache.policy.bucket_count(field),
                embedding_dim=(16 if field in self.cache.policy.high_cardinality_fields else 8),
            )
            for field in self.categorical_fields
        }

    def references_for_ids(self, click_ids: list[bytes]) -> NDArray[np.int32]:
        try:
            return np.fromiter(
                (self.id_lookup[value] for value in click_ids),
                dtype=np.int32,
                count=len(click_ids),
            )
        except KeyError as error:
            raise ConsistencyError("Truth references an unknown click ID") from error

    def references_for_day(self, day: int) -> NDArray[np.int32]:
        item = self.day_ranges.get(day)
        if item is None:
            return np.empty(0, dtype=np.int32)
        return np.arange(item.start, item.stop, dtype=np.int32)

    def tensor_batch(self, references: NDArray[np.integer]) -> FeatureTensorBatch:
        refs = np.asarray(references, dtype=np.int64)
        if refs.ndim != 1 or np.any(refs < 0) or np.any(refs >= self.cache.rows):
            raise ConsistencyError("Feature batch contains an invalid reference")
        categorical = {
            field: torch.from_numpy(self.categorical[refs, column].astype(np.int64, copy=False))
            for column, field in enumerate(self.categorical_fields)
        }
        numeric = torch.from_numpy(self.numeric[refs])
        return FeatureTensorBatch(categorical=categorical, numeric=numeric)
