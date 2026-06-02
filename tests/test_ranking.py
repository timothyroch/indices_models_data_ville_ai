import pandas as pd

from ville_indices.operations.ranking import percentile_rank


def test_percentile_rank_formula() -> None:
    values, metadata = percentile_rank(pd.Series([10, 20, 30]), ascending=True)

    assert values.tolist() == [0.0, 0.5, 1.0]
    assert metadata["formula"] == "(rank - 1) / (N - 1)"


def test_percentile_rank_tie_min() -> None:
    values, _ = percentile_rank(pd.Series([10, 10, 30]), ascending=True, tie_method="min")

    assert values.tolist() == [0.0, 0.0, 1.0]


def test_percentile_rank_n_equals_one() -> None:
    values, metadata = percentile_rank(pd.Series([42]), ascending=True)

    assert values.tolist() == [0.0]
    assert metadata["number_of_units"] == 1
