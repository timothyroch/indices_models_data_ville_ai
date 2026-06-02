"""Classification utilities for benchmark outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd


def classify_series(
    scores: pd.Series,
    *,
    method: str = "none",
    n_classes: int = 5,
) -> tuple[pd.Series | None, dict[str, Any]]:
    series = pd.Series(scores, dtype="float64")

    if method == "none" or method is None:
        return None, {"method": "none"}

    if n_classes < 1:
        raise ValueError("n_classes must be at least 1.")

    if series.dropna().nunique() <= 1:
        classes = pd.Series(1, index=series.index, dtype="Int64")
        return classes, {
            "method": method,
            "n_classes_requested": n_classes,
            "n_classes_created": 1,
            "constant_score_behavior": "single_class",
        }

    if method == "quantile":
        ranked = series.rank(method="first")
        classes = pd.qcut(
            ranked,
            q=min(n_classes, int(series.notna().sum())),
            labels=False,
            duplicates="drop",
        )
        classes = pd.Series(classes, index=series.index).astype("Int64") + 1
        return classes, {
            "method": "quantile",
            "n_classes_requested": n_classes,
            "n_classes_created": int(classes.dropna().nunique()),
            "class_order": "higher_score_higher_class",
        }

    if method == "equal_interval":
        classes = pd.cut(series, bins=n_classes, labels=False, include_lowest=True)
        classes = pd.Series(classes, index=series.index).astype("Int64") + 1
        return classes, {
            "method": "equal_interval",
            "n_classes_requested": n_classes,
            "n_classes_created": int(classes.dropna().nunique()),
            "class_order": "higher_score_higher_class",
        }

    if method == "standard_deviation_bands":
        zscores, metadata = standard_deviation_zscores(series)
        classes = standard_deviation_band_classes(zscores)
        metadata.update(
            {
                "method": "standard_deviation_bands",
                "class_order": "higher_z_higher_vulnerability",
            }
        )
        return classes, metadata

    if method in {"natural_breaks", "standard_deviation"}:
        raise NotImplementedError(
            f"Classification method '{method}' is reserved as a future extension point."
        )

    raise NotImplementedError(f"Classification method '{method}' is not implemented.")


def standard_deviation_zscores(
    scores: pd.Series,
) -> tuple[pd.Series, dict[str, Any]]:
    series = pd.Series(scores, dtype="float64")
    mean = float(series.mean(skipna=True))
    std = float(series.std(skipna=True, ddof=0))
    if std == 0 or pd.isna(std):
        zscores = pd.Series(0.0, index=series.index, dtype="float64")
        zero_behavior = "zeros"
    else:
        zscores = (series - mean) / std
        zero_behavior = None
    return zscores, {
        "mean": mean,
        "std": std,
        "zero_std_behavior": zero_behavior,
    }


def standard_deviation_band_classes(zscores: pd.Series) -> pd.Series:
    z = pd.Series(zscores, dtype="float64")
    classes = pd.Series("moderate_vulnerability", index=z.index, dtype="object")
    classes[z < -1.0] = "least_vulnerable"
    classes[(z >= -1.0) & (z < -0.5)] = "low_vulnerability"
    classes[(z > 0.5) & (z <= 1.0)] = "high_vulnerability"
    classes[z > 1.0] = "most_vulnerable"
    classes[z.isna()] = pd.NA
    return classes
