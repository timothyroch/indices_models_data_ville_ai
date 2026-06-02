"""Standardization helpers for multivariate index methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class StandardizationResult:
    standardized: pd.DataFrame
    metadata: dict[str, Any]
    dropped_variables: list[str]


def zscore_standardize(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    parameters: dict[str, dict[str, float]] | None = None,
    fit: bool = True,
    ddof: int = 0,
    zero_variance_behavior: str = "drop",
    zero_variance_tol: float = 1.0e-12,
    prefix: str | None = None,
) -> StandardizationResult:
    """Z-score standardize columns and return metadata.

    The default zero-variance behavior is to drop constant variables. This is
    explicit and metadata-backed because PCA/factor analysis cannot use columns
    with zero variance.
    """

    if zero_variance_behavior not in {"drop", "zeros", "error"}:
        raise ValueError(f"Unsupported zero variance behavior: {zero_variance_behavior}")

    parameters = parameters or {}
    standardized = pd.DataFrame(index=frame.index)
    metadata: dict[str, Any] = {
        "method": "zscore",
        "center": True,
        "scale": True,
        "ddof": ddof,
        "zero_variance_behavior": zero_variance_behavior,
        "zero_variance_tol": zero_variance_tol,
        "variables": {},
    }
    dropped: list[str] = []

    for column in columns:
        series = pd.Series(frame[column], index=frame.index, dtype="float64")
        if fit:
            mean = float(series.mean(skipna=True))
            std = float(series.std(skipna=True, ddof=ddof))
        else:
            mean = float(parameters[column]["mean"])
            std = float(parameters[column]["std"])

        output_column = f"{prefix}{column}" if prefix else column
        variable_metadata = {
            "mean": mean,
            "std": std,
            "output_column": output_column,
        }

        if std <= zero_variance_tol or np.isnan(std):
            variable_metadata["zero_variance"] = True
            if zero_variance_behavior == "drop":
                dropped.append(column)
                metadata["variables"][column] = variable_metadata
                continue
            if zero_variance_behavior == "zeros":
                standardized[output_column] = 0.0
            else:
                raise ValueError(f"Variable '{column}' has zero variance.")
        else:
            variable_metadata["zero_variance"] = False
            standardized[output_column] = (series - mean) / std

        metadata["variables"][column] = variable_metadata

    metadata["dropped_variables"] = dropped
    metadata["used_variables"] = [
        column for column in columns if column not in set(dropped)
    ]
    return StandardizationResult(
        standardized=standardized,
        metadata=metadata,
        dropped_variables=dropped,
    )
