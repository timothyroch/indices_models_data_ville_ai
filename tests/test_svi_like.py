from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ville_indices.core.outputs import assert_standard_output_schema
from ville_indices.core.recipe import Recipe
from ville_indices.indices.svi_like import SVI_METHOD_NOTE, SviLikeIndex
from ville_indices.operations.ranking import percentile_rank
from ville_indices.run import run_index


SVI_RECIPE_PATH = Path("recipes/svi_like.yaml")
SVI_DATA_PATH = Path("data/example/synthetic_svi_feature_table.csv")


def load_svi_recipe_dict() -> dict:
    return yaml.safe_load(SVI_RECIPE_PATH.read_text())


def load_svi_recipe() -> Recipe:
    return Recipe.from_dict(load_svi_recipe_dict())


def load_svi_features() -> pd.DataFrame:
    return pd.read_csv(SVI_DATA_PATH)


def run_svi_on_synthetic() -> tuple[SviLikeIndex, pd.DataFrame, pd.DataFrame]:
    index = SviLikeIndex(load_svi_recipe(), run_id="svi-test")
    standard = index.fit_transform(load_svi_features())
    assert index.intermediate_output is not None
    return index, standard, index.intermediate_output


def test_svi_variable_percentile_formula_ties_and_directions() -> None:
    _, _, detailed = run_svi_on_synthetic()
    by_zone = detailed.set_index("zone_id")

    assert by_zone.loc["LOW", "svi_pr_pct_below_poverty"] == 0.0
    assert by_zone.loc["HIGH", "svi_pr_pct_below_poverty"] == 1.0
    assert by_zone.loc["MID2", "svi_pr_pct_below_poverty"] == 0.4
    assert by_zone.loc["MID3", "svi_pr_pct_below_poverty"] == 0.4
    assert by_zone.loc["LOW", "svi_pr_per_capita_income"] == 0.0
    assert by_zone.loc["HIGH", "svi_pr_per_capita_income"] == 1.0


def test_svi_domain_and_final_aggregation_are_staged() -> None:
    _, _, detailed = run_svi_on_synthetic()

    socioeconomic_pr_cols = [
        "svi_pr_pct_below_poverty",
        "svi_pr_pct_unemployed",
        "svi_pr_per_capita_income",
        "svi_pr_pct_no_high_school",
    ]
    expected_ses_sum = detailed[socioeconomic_pr_cols].sum(axis=1)
    pd.testing.assert_series_equal(
        detailed["svi_socioeconomic_sum"],
        expected_ses_sum,
        check_names=False,
    )

    expected_ses_percentile, _ = percentile_rank(
        detailed["svi_socioeconomic_sum"], ascending=True, tie_method="min"
    )
    pd.testing.assert_series_equal(
        detailed["svi_socioeconomic_percentile"],
        expected_ses_percentile.astype("float64"),
        check_names=False,
    )

    domain_percentile_cols = [
        "svi_socioeconomic_percentile",
        "svi_household_disability_percentile",
        "svi_minority_language_percentile",
        "svi_housing_transportation_percentile",
    ]
    expected_overall_sum = detailed[domain_percentile_cols].sum(axis=1)
    pd.testing.assert_series_equal(
        detailed["svi_overall_sum"],
        expected_overall_sum,
        check_names=False,
    )

    expected_overall_percentile, _ = percentile_rank(
        detailed["svi_overall_sum"], ascending=True, tie_method="min"
    )
    pd.testing.assert_series_equal(
        detailed["svi_overall_percentile"],
        expected_overall_percentile.astype("float64"),
        check_names=False,
    )

    variable_pr_cols = [column for column in detailed.columns if column.startswith("svi_pr_")]
    direct_variable_sum = detailed[variable_pr_cols].sum(axis=1)
    assert not direct_variable_sum.equals(detailed["svi_overall_sum"])


def test_svi_flags_and_flag_counts() -> None:
    _, _, detailed = run_svi_on_synthetic()
    high = detailed.set_index("zone_id").loc["HIGH"]

    assert high["svi_flag_pct_below_poverty"] == 1
    assert high["svi_flag_per_capita_income"] == 1
    assert high["svi_socioeconomic_flag_count"] == 4
    assert high["svi_household_disability_flag_count"] == 4
    assert high["svi_minority_language_flag_count"] == 2
    assert high["svi_housing_transportation_flag_count"] == 5
    assert high["svi_total_flag_count"] == 15


def test_svi_end_to_end_standard_output_and_explain() -> None:
    index, standard, detailed = run_svi_on_synthetic()
    assert_standard_output_schema(standard)

    scores = dict(zip(standard["zone_id"], standard["score_normalized_0_1"]))
    assert scores["HIGH"] > scores["LOW"]
    assert standard.loc[standard["zone_id"] == "HIGH", "rank"].item() == 1
    assert "svi_overall_percentile" in detailed.columns

    explanation = index.explain("HIGH")
    assert explanation["svi_overall_percentile"] == 1.0
    assert explanation["total_flag_count"] == 15
    assert "individual" in explanation["interpretation"]


def test_svi_missing_required_variable_fails_in_local_adaptation() -> None:
    features = load_svi_features().drop(columns=["pct_no_vehicle"])
    index = SviLikeIndex(load_svi_recipe())

    with pytest.raises(ValueError, match="Required variable"):
        index.fit(features)


def test_svi_missing_required_variable_requires_explicit_partial_mode() -> None:
    recipe_dict = load_svi_recipe_dict()
    recipe_dict["reproduction_level"] = "partial_svi_like"
    recipe_dict["partial"] = {"allow_missing_required_variables": True}
    recipe = Recipe.from_dict(recipe_dict)
    features = load_svi_features().drop(columns=["pct_mobile_homes"])

    index = SviLikeIndex(recipe, run_id="svi-partial")
    standard = index.fit_transform(features)
    assert index.intermediate_output is not None

    assert set(standard["quality_flag"]) == {"partial_svi_like"}
    assert (index.intermediate_output["svi_missing_variable_count"] == 1).all()


def test_svi_duplicate_zone_id_fails_validation() -> None:
    features = load_svi_features()
    features.loc[1, "zone_id"] = "LOW"
    index = SviLikeIndex(load_svi_recipe())

    with pytest.raises(ValueError, match="duplicate IDs"):
        index.fit(features)


def test_svi_zero_population_units_are_excluded() -> None:
    features = load_svi_features()
    features.loc[features["zone_id"] == "LOW", "population"] = 0
    index = SviLikeIndex(load_svi_recipe(), run_id="svi-zero-pop")
    standard = index.fit_transform(features)

    assert "LOW" not in set(standard["zone_id"])
    assert index.process_metadata["population_filter"]["excluded_count"] == 1
    assert index.process_metadata["population_filter"]["excluded_units"] == ["LOW"]


def test_svi_n_equals_one_scope_is_handled() -> None:
    features = load_svi_features().head(1)
    index = SviLikeIndex(load_svi_recipe(), run_id="svi-one")
    standard = index.fit_transform(features)

    assert standard.loc[0, "score_normalized_0_1"] == 0.0
    assert standard.loc[0, "percentile"] == 0.0
    assert index.process_metadata["ranking"]["overall_rank_metadata"]["number_of_units"] == 1


def test_svi_validation_reports_constant_and_scale_warnings() -> None:
    features = load_svi_features()
    features["pct_mobile_homes"] = 0.01
    features["pct_no_vehicle"] = 50
    index = SviLikeIndex(load_svi_recipe())

    report = index.validate_inputs(features)
    codes = {issue.code for issue in report.issues}

    assert "constant_column" in codes
    assert "proportion_out_of_range" in codes
    assert "suspicious_percentage_scale" in codes


def test_svi_proxy_mapping_is_recipe_driven() -> None:
    recipe_dict = load_svi_recipe_dict()
    recipe_dict["variables"]["per_capita_income"]["canonical_name"] = "median_household_income"
    recipe_dict["variables"]["per_capita_income"]["proxy_used"] = "median_household_income"
    recipe_dict["variables"]["per_capita_income"]["proxy_quality"] = "medium"
    recipe_dict["variables"]["per_capita_income"]["status"] = "local_adaptation"
    recipe_dict["variables"]["per_capita_income"][
        "conceptual_risk"
    ] = "household income is not identical to per-capita income"
    recipe = Recipe.from_dict(recipe_dict)
    features = load_svi_features().rename(
        columns={"per_capita_income": "median_household_income"}
    )

    index = SviLikeIndex(recipe, run_id="svi-proxy")
    standard = index.fit_transform(features)

    assert set(standard["quality_flag"]) == {"proxy_used"}
    assert index.process_metadata["variables_proxied"][0]["variable"] == "per_capita_income"


def test_svi_runner_produces_outputs_and_metadata(tmp_path) -> None:
    outputs = run_index(
        index_name="svi_like",
        recipe_path=SVI_RECIPE_PATH,
        feature_table_path=SVI_DATA_PATH,
        output_dir=tmp_path / "svi_run",
    )

    for path in outputs.values():
        assert path.exists()

    metadata = json.loads(outputs["metadata_json"].read_text())
    assert metadata["index_name"] == "svi_like"
    assert metadata["extra"]["methodology_note"] == SVI_METHOD_NOTE
    assert metadata["extra"]["ranking"]["formula"] == "(rank - 1) / (N - 1)"

    detailed = pd.read_csv(outputs["intermediate_output"])
    standard = pd.read_csv(outputs["standard_output"])
    assert "svi_overall_percentile" in detailed.columns
    assert "score_normalized_0_1" in standard.columns
