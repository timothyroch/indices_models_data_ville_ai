import pandas as pd
import pytest

from ville_indices.operations.normalization import normalize_series


def test_minmax_normalization() -> None:
    values, metadata = normalize_series(pd.Series([10, 20, 30]), method="minmax")

    assert values.tolist() == [0.0, 0.5, 1.0]
    assert metadata["min"] == 10
    assert metadata["max"] == 30


def test_constant_column_handling() -> None:
    values, metadata = normalize_series(pd.Series([5, 5, 5]), method="minmax")

    assert values.tolist() == [0.0, 0.0, 0.0]
    assert metadata["constant_column_behavior"] == "zeros"


def test_zscore_normalization() -> None:
    values, metadata = normalize_series(pd.Series([1, 2, 3]), method="zscore")

    assert pytest.approx(float(values.mean())) == 0.0
    assert pytest.approx(float(values.std(ddof=0))) == 1.0
    assert metadata["mean"] == 2
