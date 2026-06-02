import pandas as pd

from ville_indices.indices.dummy_index import DummyAdditiveIndex
from ville_indices.run import run_index


def test_dummy_index_end_to_end_direction(recipe, feature_table) -> None:
    index = DummyAdditiveIndex(recipe, run_id="test-run")

    output = index.fit_transform(feature_table)

    scores = dict(zip(output["zone_id"], output["score_raw"]))
    assert scores["B"] > scores["A"]
    assert output.loc[output["zone_id"] == "B", "rank"].item() == 1
    assert index.intermediate_output is not None
    assert "income__oriented" in index.intermediate_output.columns


def test_runner_writes_expected_outputs(tmp_path) -> None:
    outputs = run_index(
        recipe_path="recipes/dummy_index.yaml",
        feature_table_path="data/example/synthetic_feature_table.csv",
        output_dir=tmp_path / "dummy_run",
    )

    for path in outputs.values():
        assert path.exists()

    standard = pd.read_csv(outputs["standard_output"])
    scores = dict(zip(standard["zone_id"], standard["score_raw"]))
    assert scores["B"] > scores["A"]
