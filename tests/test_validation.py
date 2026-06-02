import pandas as pd

from ville_indices.core.validation import FeatureTableValidator


def issue_codes(report):
    return {issue.code for issue in report.issues}


def test_required_variable_validation_passes(recipe, feature_table) -> None:
    report = FeatureTableValidator(recipe).validate(feature_table)

    assert report.is_valid
    assert set(report.variables_present) == {
        "median_household_income",
        "pct_65_plus",
        "floodplain_pct",
    }


def test_duplicate_zone_id_detection(recipe, feature_table) -> None:
    feature_table.loc[1, "zone_id"] = "A"

    report = FeatureTableValidator(recipe).validate(feature_table)

    assert not report.is_valid
    assert "duplicate_spatial_ids" in issue_codes(report)


def test_missing_required_variable_behavior(recipe, feature_table) -> None:
    report = FeatureTableValidator(recipe).validate(
        feature_table.drop(columns=["pct_65_plus"])
    )

    assert not report.is_valid
    assert "missing_required_variable" in issue_codes(report)
    assert "pct_65_plus" in report.variables_missing


def test_missing_value_reporting_with_error_strategy(recipe, feature_table) -> None:
    feature_table.loc[2, "floodplain_pct"] = pd.NA

    report = FeatureTableValidator(recipe).validate(feature_table)

    assert not report.is_valid
    assert "missing_values_detected" in issue_codes(report)
    assert report.missingness["floodplain_pct"]["missing_count"] == 1


def test_optional_variable_missing_is_warning(recipe_dict, feature_table) -> None:
    recipe_dict["variables"]["optional_demo"] = {
        "canonical_name": "optional_metric",
        "required": False,
        "direction": "positive",
        "normalization": {"method": "none"},
    }
    from ville_indices.core.recipe import Recipe

    recipe = Recipe.from_dict(recipe_dict)
    report = FeatureTableValidator(recipe).validate(feature_table)

    assert report.is_valid
    assert "missing_optional_variable" in issue_codes(report)
    assert "optional_metric" in report.optional_variables_missing


def test_constant_column_and_out_of_range_proportion_warn(recipe, feature_table) -> None:
    feature_table["pct_65_plus"] = 2.0

    report = FeatureTableValidator(recipe).validate(feature_table)

    assert "constant_column" in issue_codes(report)
    assert "proportion_out_of_range" in issue_codes(report)
