"""Generic aggregation utilities and extension points."""

from __future__ import annotations

from typing import Any

import pandas as pd


def aggregate(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    method: str,
    weights: dict[str, float] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Aggregate columns with a generic method.

    Method-specific future algorithms such as PCA, TOPSIS, OWA, and SVI
    domain-sum-then-rank should be added as explicit operations or index logic,
    not faked through this generic helper.
    """

    if method == "sum":
        result = frame[columns].sum(axis=1)
        metadata = {"method": "sum", "columns": columns}
        return result, metadata

    if method == "mean":
        result = frame[columns].mean(axis=1)
        metadata = {"method": "mean", "columns": columns}
        return result, metadata

    if method == "weighted_sum":
        if not weights:
            raise ValueError("weighted_sum aggregation requires weights.")
        missing_weights = [column for column in columns if column not in weights]
        if missing_weights:
            raise ValueError(f"Missing weights for columns: {missing_weights}")
        result = sum(frame[column] * float(weights[column]) for column in columns)
        metadata = {
            "method": "weighted_sum",
            "columns": columns,
            "weights": {column: float(weights[column]) for column in columns},
            "weight_sum": float(sum(float(weights[column]) for column in columns)),
        }
        return result, metadata

    if method in {
        "domain_sum_then_rank",
        "factor_score_sum",
        "subindex_average",
        "topsis",
        "entropy_correction",
        "owa",
    }:
        raise NotImplementedError(
            f"Aggregation method '{method}' is a future index-specific extension point."
        )

    raise NotImplementedError(f"Aggregation method '{method}' is not implemented.")
