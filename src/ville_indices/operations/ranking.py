"""Ranking utilities, including SVI-style percentile ranks."""

from __future__ import annotations

from typing import Any

import pandas as pd


def percentile_rank(
    values: pd.Series,
    *,
    ascending: bool = True,
    tie_method: str = "min",
    n_equals_one: str = "zero",
) -> tuple[pd.Series, dict[str, Any]]:
    """Return percentile ranks on a 0-1 scale.

    The default formula is `(rank - 1) / (N - 1)`, matching the SVI-style
    percentile convention needed by future SVI-like implementations.
    """

    if tie_method not in {"min", "average", "max", "first", "dense"}:
        raise ValueError(f"Unsupported tie method: {tie_method}")
    series = pd.Series(values, dtype="float64")
    nonmissing_count = int(series.notna().sum())
    result = pd.Series(pd.NA, index=series.index, dtype="Float64")
    if nonmissing_count == 0:
        metadata = {
            "number_of_units": 0,
            "tie_method": tie_method,
            "ranking_direction": "ascending" if ascending else "descending",
            "formula": "(rank - 1) / (N - 1)",
            "n_equals_one_behavior": n_equals_one,
        }
        return result, metadata
    if nonmissing_count == 1:
        if n_equals_one == "zero":
            result.loc[series.notna()] = 0.0
        elif n_equals_one == "one":
            result.loc[series.notna()] = 1.0
        else:
            raise ValueError(f"Unsupported N=1 behavior: {n_equals_one}")
    else:
        ranks = series.rank(method=tie_method, ascending=ascending, na_option="keep")
        result = (ranks - 1.0) / (nonmissing_count - 1.0)
        result = result.astype("Float64")
    metadata = {
        "number_of_units": nonmissing_count,
        "tie_method": tie_method,
        "ranking_direction": "ascending" if ascending else "descending",
        "formula": "(rank - 1) / (N - 1)",
        "n_equals_one_behavior": n_equals_one,
        "output_range": "0_1",
    }
    return result, metadata
