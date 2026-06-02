"""Normalization utilities with fit/transform metadata."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ville_indices.operations.ranking import percentile_rank


def normalize_series(
    values: pd.Series,
    *,
    method: str = "none",
    parameters: dict[str, Any] | None = None,
    fit: bool = True,
) -> tuple[pd.Series, dict[str, Any]]:
    """Normalize a series and return normalized values plus fitted metadata."""

    parameters = dict(parameters or {})
    method = method or "none"
    series = pd.Series(values, dtype="float64")

    if method == "none":
        return series.copy(), {"method": "none", "scope": "series"}

    if method == "minmax":
        min_value = float(series.min(skipna=True)) if fit else float(parameters["min"])
        max_value = float(series.max(skipna=True)) if fit else float(parameters["max"])
        denominator = max_value - min_value
        constant_behavior = parameters.get("constant_column_behavior", "zeros")
        if denominator == 0 or np.isnan(denominator):
            if constant_behavior == "zeros":
                normalized = pd.Series(0.0, index=series.index, dtype="float64")
            elif constant_behavior == "nan":
                normalized = pd.Series(np.nan, index=series.index, dtype="float64")
            else:
                raise ValueError(f"Unsupported constant column behavior: {constant_behavior}")
        else:
            normalized = (series - min_value) / denominator
        metadata = {
            "method": "minmax",
            "min": min_value,
            "max": max_value,
            "constant_column_behavior": constant_behavior,
            "scope": "series",
        }
        return normalized, metadata

    if method == "zscore":
        mean = float(series.mean(skipna=True)) if fit else float(parameters["mean"])
        std = float(series.std(skipna=True, ddof=0)) if fit else float(parameters["std"])
        constant_behavior = parameters.get("constant_column_behavior", "zeros")
        if std == 0 or np.isnan(std):
            if constant_behavior == "zeros":
                normalized = pd.Series(0.0, index=series.index, dtype="float64")
            elif constant_behavior == "nan":
                normalized = pd.Series(np.nan, index=series.index, dtype="float64")
            else:
                raise ValueError(f"Unsupported constant column behavior: {constant_behavior}")
        else:
            normalized = (series - mean) / std
        metadata = {
            "method": "zscore",
            "mean": mean,
            "std": std,
            "constant_column_behavior": constant_behavior,
            "scope": "series",
        }
        return normalized, metadata

    if method == "percentile_rank":
        ascending = bool(parameters.get("ascending", True))
        tie_method = parameters.get("tie_method", "min")
        normalized, rank_metadata = percentile_rank(
            series, ascending=ascending, tie_method=tie_method
        )
        metadata = {"method": "percentile_rank", **rank_metadata}
        return normalized.astype("float64"), metadata

    if method == "vector_normalization":
        denominator = (
            float(np.sqrt(np.nansum(np.square(series.to_numpy(dtype=float)))))
            if fit
            else float(parameters["denominator"])
        )
        zero_behavior = parameters.get("zero_denominator_behavior", "zeros")
        if denominator == 0 or np.isnan(denominator):
            if zero_behavior == "zeros":
                normalized = pd.Series(0.0, index=series.index, dtype="float64")
            elif zero_behavior == "nan":
                normalized = pd.Series(np.nan, index=series.index, dtype="float64")
            else:
                raise ValueError(f"Unsupported zero denominator behavior: {zero_behavior}")
        else:
            normalized = series / denominator
        metadata = {
            "method": "vector_normalization",
            "denominator": denominator,
            "zero_denominator_behavior": zero_behavior,
            "scope": "series",
        }
        return normalized, metadata

    raise NotImplementedError(f"Normalization method '{method}' is not implemented.")
