import pandas as pd

from ville_indices.operations.missing_data import MissingDataHandler


def test_missing_data_report_and_median_imputation(recipe_dict, feature_table) -> None:
    recipe_dict["missing_data"] = {
        "strategy": "median_imputation",
        "add_missing_flags": True,
    }
    from ville_indices.core.recipe import Recipe

    recipe = Recipe.from_dict(recipe_dict)
    feature_table.loc[0, "pct_65_plus"] = pd.NA

    handled, report = MissingDataHandler(recipe).fit_transform(feature_table)

    assert report.missing_count_per_variable["pct_65_plus"] == 1
    assert report.affected_spatial_units_count == 1
    assert "pct_65_plus__missing_flag" in handled.columns
    assert handled.loc[0, "pct_65_plus"] == report.imputation_values["pct_65_plus"]
