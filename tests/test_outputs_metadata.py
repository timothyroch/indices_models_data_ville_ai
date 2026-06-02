import json

import pandas as pd

from ville_indices.core.metadata import build_run_metadata
from ville_indices.core.outputs import assert_standard_output_schema, create_standard_output
from ville_indices.core.validation import FeatureTableValidator
from ville_indices.operations.missing_data import MissingDataHandler


def test_standard_output_schema(recipe, feature_table) -> None:
    output = create_standard_output(
        feature_table=feature_table,
        recipe=recipe,
        run_id="test-run",
        score=pd.Series([0.1, 0.9, 0.5, 0.7, 0.2]),
    )

    assert_standard_output_schema(output)
    assert output.loc[1, "rank"] == 1


def test_metadata_export(tmp_path, recipe, feature_table) -> None:
    validation_report = FeatureTableValidator(recipe).validate(feature_table)
    _, missing_report = MissingDataHandler(recipe).fit_transform(feature_table)
    metadata = build_run_metadata(
        recipe=recipe,
        run_id="test-run",
        number_input_units=len(feature_table),
        number_output_units=len(feature_table),
        validation_report=validation_report,
        missing_report=missing_report,
        process_metadata={
            "normalization": {"income": {"method": "minmax"}},
            "orientation": {},
            "weighting_method": "variable_weights_from_recipe",
        },
    )
    path = tmp_path / "metadata.json"

    metadata.to_json(path)

    loaded = json.loads(path.read_text())
    assert loaded["index_name"] == "dummy_additive_index"
    assert loaded["validation_summary"]["is_valid"] is True
