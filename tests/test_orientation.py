import pandas as pd

from ville_indices.operations.orientation import orient_series


def test_positive_orientation() -> None:
    values, metadata = orient_series(pd.Series([0.2, 0.8]), direction="positive")

    assert values.tolist() == [0.2, 0.8]
    assert metadata["transformation_applied"] == "identity"


def test_negative_orientation() -> None:
    values, metadata = orient_series(pd.Series([0.2, 0.8]), direction="negative")

    assert values.tolist() == [0.8, 0.19999999999999996]
    assert metadata["transformation_applied"] == "1 - value"
