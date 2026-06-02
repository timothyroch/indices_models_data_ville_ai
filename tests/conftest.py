from __future__ import annotations

import pandas as pd
import pytest

from ville_indices.core.recipe import Recipe


@pytest.fixture()
def recipe_dict() -> dict:
    return {
        "name": "dummy_additive_index",
        "version": "0.1",
        "construct_measured": "synthetic_vulnerability",
        "score_direction": "higher_is_more_vulnerable",
        "reproduction_level": "toy_validation_only",
        "spatial_id_column": "zone_id",
        "variables": {
            "income": {
                "canonical_name": "median_household_income",
                "required": True,
                "unit": "CAD",
                "direction": "negative",
                "numeric": True,
                "nonnegative": True,
                "normalization": {"method": "minmax"},
                "weight": {"value": 0.4},
            },
            "age": {
                "canonical_name": "pct_65_plus",
                "required": True,
                "unit": "proportion",
                "scale": "proportion_0_1",
                "direction": "positive",
                "numeric": True,
                "nonnegative": True,
                "normalization": {"method": "minmax"},
                "weight": {"value": 0.3},
            },
            "flood": {
                "canonical_name": "floodplain_pct",
                "required": True,
                "unit": "proportion",
                "scale": "proportion_0_1",
                "direction": "positive",
                "numeric": True,
                "nonnegative": True,
                "normalization": {"method": "minmax"},
                "weight": {"value": 0.3},
            },
        },
        "missing_data": {"strategy": "error", "add_missing_flags": True},
        "aggregation": {"method": "weighted_sum"},
        "classification": {"method": "quantile", "n_classes": 5},
        "outputs": {"include_intermediate_columns": True},
    }


@pytest.fixture()
def recipe(recipe_dict: dict) -> Recipe:
    return Recipe.from_dict(recipe_dict)


@pytest.fixture()
def feature_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone_id": ["A", "B", "C", "D", "E"],
            "median_household_income": [90000, 30000, 60000, 45000, 75000],
            "pct_65_plus": [0.05, 0.30, 0.15, 0.20, 0.10],
            "floodplain_pct": [0.00, 0.80, 0.40, 0.60, 0.10],
        }
    )
