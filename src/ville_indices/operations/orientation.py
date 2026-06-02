"""Orientation utilities for positive and negative score contribution directions."""

from __future__ import annotations

from typing import Any

import pandas as pd


def orient_series(
    values: pd.Series,
    *,
    direction: str,
    phase: str = "after_normalization",
    negative_mode: str = "invert_0_1",
) -> tuple[pd.Series, dict[str, Any]]:
    """Orient values so larger oriented values imply larger score contribution."""

    series = pd.Series(values, dtype="float64")
    if direction == "positive":
        oriented = series.copy()
        transformation = "identity"
    elif direction == "negative":
        if negative_mode == "invert_0_1":
            oriented = 1.0 - series
            transformation = "1 - value"
        elif negative_mode == "multiply_by_minus_one":
            oriented = -1.0 * series
            transformation = "-value"
        else:
            raise ValueError(f"Unsupported negative orientation mode: {negative_mode}")
    elif direction == "none":
        oriented = series.copy()
        transformation = "identity"
    elif direction == "custom":
        raise NotImplementedError("Custom orientation must be implemented by an index module.")
    else:
        raise ValueError(f"Unsupported variable direction: {direction}")

    metadata = {
        "original_direction": direction,
        "transformation_applied": transformation,
        "orientation_phase": phase,
        "negative_mode": negative_mode if direction == "negative" else None,
    }
    return oriented, metadata
