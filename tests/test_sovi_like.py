from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ville_indices.core.outputs import assert_standard_output_schema
from ville_indices.core.recipe import Recipe, load_recipe
from ville_indices.indices.sovi_like import SOVI_METHOD_NOTE, SoviLikeIndex
from ville_indices.run import run_index


SOVI_RECIPE_PATH = Path("recipes/sovi_like_synthetic.yaml")
SOVI_DATA_PATH = Path("data/example/synthetic_sovi_feature_table.csv")


def load_sovi_recipe_dict() -> dict:
    return yaml.safe_load(SOVI_RECIPE_PATH.read_text())


def load_sovi_recipe() -> Recipe:
    return Recipe.from_dict(load_sovi_recipe_dict())


def load_sovi_features() -> pd.DataFrame:
    return pd.read_csv(SOVI_DATA_PATH)


def run_sovi() -> tuple[SoviLikeIndex, pd.DataFrame, pd.DataFrame]:
    index = SoviLikeIndex(load_sovi_recipe(), run_id="sovi-test")
    standard = index.fit_transform(load_sovi_features())
    assert index.intermediate_output is not None
    return index, standard, index.intermediate_output


def test_sovi_recipes_load() -> None:
    recipe = load_recipe("recipes/sovi_like.yaml")
    synthetic = load_recipe(SOVI_RECIPE_PATH)

    assert recipe.name == "sovi_like"
    assert len(recipe.variables) == 42
    assert synthetic.name == "sovi_like"
    assert len(synthetic.variables) == 12


def test_sovi_validation_missing_duplicate_nonnumeric_and_small_cases() -> None:
    recipe = load_sovi_recipe()
    features = load_sovi_features()

    with pytest.raises(ValueError, match="Required variable"):
        SoviLikeIndex(recipe).fit(features.drop(columns=["pct_poverty"]))

    duplicate = features.copy()
    duplicate.loc[1, "zone_id"] = duplicate.loc[0, "zone_id"]
    with pytest.raises(ValueError, match="duplicate IDs"):
        SoviLikeIndex(recipe).fit(duplicate)

    nonnumeric = features.copy()
    nonnumeric["pct_poverty"] = "bad"
    with pytest.raises(ValueError, match="must be numeric"):
        SoviLikeIndex(recipe).fit(nonnumeric)

    with pytest.raises(ValueError, match="at least 2 observations"):
        SoviLikeIndex(recipe).fit(features.head(1))

    one_variable = load_sovi_recipe_dict()
    one_variable["variables"] = {
        "median_income": one_variable["variables"]["median_income"]
    }
    with pytest.raises(ValueError, match="at least 2 usable numeric variables"):
        SoviLikeIndex(Recipe.from_dict(one_variable)).fit(features)


def test_sovi_missing_variable_requires_explicit_partial_mode() -> None:
    recipe_dict = load_sovi_recipe_dict()
    recipe_dict["reproduction_level"] = "partial_sovi_like"
    recipe_dict["partial"] = {"allow_missing_required_variables": True}
    recipe = Recipe.from_dict(recipe_dict)
    features = load_sovi_features().drop(columns=["pct_mobile_homes"])

    index = SoviLikeIndex(recipe, run_id="sovi-partial")
    standard = index.fit_transform(features)

    assert set(standard["quality_flag"]) == {"partial_sovi_like"}
    assert "pct_mobile_homes" not in index.process_metadata["variables_used"]


def test_sovi_missing_data_imputation_metadata_for_zero_mean_median() -> None:
    for strategy in ["zero_imputation", "mean_imputation", "median_imputation"]:
        recipe_dict = load_sovi_recipe_dict()
        recipe_dict["missing_data"]["strategy"] = strategy
        index = SoviLikeIndex(Recipe.from_dict(recipe_dict), run_id=f"sovi-{strategy}")
        index.fit_transform(load_sovi_features())

        assert index.missing_report is not None
        assert index.missing_report.missing_count_per_variable["pct_mobile_homes"] == 1
        assert "pct_mobile_homes" in index.missing_report.imputation_values
        if strategy == "zero_imputation":
            assert index.missing_report.imputation_values["pct_mobile_homes"] == 0.0


def test_sovi_standardization_outputs_are_centered_and_scaled() -> None:
    index, _, _ = run_sovi()
    standardized = index.extra_outputs["sovi_standardized_variables"]

    assert np.allclose(standardized.mean().to_numpy(), 0.0, atol=1.0e-12)
    assert np.allclose(standardized.std(ddof=0).to_numpy(), 1.0, atol=1.0e-12)
    assert "median_income" in index.process_metadata["standardization"]["variables"]


def test_sovi_constant_variable_is_reported_and_dropped() -> None:
    features = load_sovi_features()
    features["pct_children"] = 0.2
    index = SoviLikeIndex(load_sovi_recipe(), run_id="sovi-constant")
    report = index.validate_inputs(features)
    codes = {issue.code for issue in report.issues}

    assert "constant_column" in codes
    index.fit_transform(features)
    assert "pct_children" in index.process_metadata["variables_dropped"]


def test_sovi_pca_retention_rotation_and_factor_shapes() -> None:
    index, _, _ = run_sovi()

    eigenvalues = index.extra_outputs["sovi_eigenvalues"]
    unrotated = index.extra_outputs["sovi_loadings_unrotated"]
    rotated = index.extra_outputs["sovi_loadings_rotated"]
    scores = index.extra_outputs["sovi_factor_scores"]

    assert len(eigenvalues) >= 3
    assert index.process_metadata["factor_retention"]["n_factors_retained"] == 3
    assert unrotated.shape == rotated.shape
    assert {"factor_1_rotated", "factor_1_oriented"} <= set(scores.columns)
    assert index.process_metadata["rotation"]["method"] == "varimax"


def test_sovi_eigenvalue_gt_and_zero_retention_rules() -> None:
    recipe_dict = load_sovi_recipe_dict()
    recipe_dict["factor_retention"] = {"method": "eigenvalue_gt", "threshold": 1.0}
    index = SoviLikeIndex(Recipe.from_dict(recipe_dict), run_id="sovi-eigen")
    index.fit_transform(load_sovi_features())
    assert index.process_metadata["factor_retention"]["n_factors_retained"] >= 1

    recipe_dict["factor_retention"] = {"method": "eigenvalue_gt", "threshold": 99.0}
    with pytest.raises(ValueError, match="retained zero factors"):
        SoviLikeIndex(Recipe.from_dict(recipe_dict)).fit(load_sovi_features())


def test_sovi_rotation_can_be_disabled_explicitly() -> None:
    recipe_dict = load_sovi_recipe_dict()
    recipe_dict["rotation"] = {"method": "none"}
    index = SoviLikeIndex(Recipe.from_dict(recipe_dict), run_id="sovi-no-rotation")
    index.fit_transform(load_sovi_features())

    assert index.process_metadata["rotation"]["method"] == "none"
    assert any("skipped" in warning for warning in index.process_metadata["warnings"])


def test_sovi_factor_orientation_methods_and_missing_orientation() -> None:
    index, _, detailed = run_sovi()

    assert np.allclose(
        detailed["sovi_factor_1_oriented"],
        detailed["sovi_factor_1_score"],
    )
    assert np.allclose(
        detailed["sovi_factor_2_oriented"],
        -detailed["sovi_factor_2_score"],
    )
    assert np.allclose(
        detailed["sovi_factor_3_oriented"],
        np.abs(detailed["sovi_factor_3_score"]),
    )
    assert index.process_metadata["orientation"]["factor_3"]["method"] == "absolute"

    recipe_dict = load_sovi_recipe_dict()
    del recipe_dict["factor_orientation"]["factors"]["factor_2"]
    with pytest.raises(ValueError, match="Missing SoVI orientation decision"):
        SoviLikeIndex(Recipe.from_dict(recipe_dict)).fit(load_sovi_features())


def test_sovi_aggregation_classification_and_benchmark_fields() -> None:
    _, standard, detailed = run_sovi()

    oriented_cols = [
        "sovi_factor_1_oriented",
        "sovi_factor_2_oriented",
        "sovi_factor_3_oriented",
    ]
    expected_score = detailed[oriented_cols].sum(axis=1)
    pd.testing.assert_series_equal(
        detailed["sovi_score_raw"],
        expected_score,
        check_names=False,
    )
    assert np.isclose(float(detailed["sovi_score_z"].mean()), 0.0)
    assert np.isclose(float(detailed["sovi_score_z"].std(ddof=0)), 1.0)
    assert "sovi_class" in detailed.columns
    assert standard["score_raw"].equals(detailed["sovi_score_raw"])
    assert standard["percentile"].between(0, 1).all()
    assert "class" in standard.columns
    assert_standard_output_schema(standard)


def test_sovi_outputs_metadata_runner_and_explain(tmp_path) -> None:
    outputs = run_index(
        index_name="sovi_like",
        recipe_path=SOVI_RECIPE_PATH,
        feature_table_path=SOVI_DATA_PATH,
        output_dir=tmp_path / "sovi_run",
    )

    expected = {
        "standard_output",
        "intermediate_output",
        "sovi_eigenvalues",
        "sovi_loadings_unrotated",
        "sovi_loadings_rotated",
        "sovi_factor_scores",
        "sovi_factor_summary",
        "metadata_json",
        "validation_report_json",
        "run_report",
    }
    assert expected <= set(outputs)
    for path in outputs.values():
        assert path.exists()

    metadata = json.loads(outputs["metadata_json"].read_text())
    assert metadata["index_name"] == "sovi_like"
    assert metadata["extra"]["methodology_note"] == SOVI_METHOD_NOTE
    assert metadata["extra"]["factor_retention"]["n_factors_retained"] == 3

    index, _, _ = run_sovi()
    explanation = index.explain("Z07")
    assert explanation["zone_id"] == "Z07"
    assert "oriented_factor_scores" in explanation
    assert "individual" in explanation["interpretation"]
