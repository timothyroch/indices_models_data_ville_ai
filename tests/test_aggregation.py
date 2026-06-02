import pandas as pd

from ville_indices.operations.aggregation import aggregate


def test_weighted_sum_aggregation() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    values, metadata = aggregate(
        frame,
        columns=["a", "b"],
        method="weighted_sum",
        weights={"a": 0.25, "b": 0.75},
    )

    assert values.tolist() == [2.5, 3.5]
    assert metadata["weight_sum"] == 1.0
