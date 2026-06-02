from pathlib import Path

from ville_indices.core.recipe import load_recipe


def test_recipe_loading() -> None:
    recipe = load_recipe(Path("recipes/dummy_index.yaml"))

    assert recipe.name == "dummy_additive_index"
    assert recipe.spatial_id_column == "zone_id"
    assert recipe.required_variables == [
        "median_household_income",
        "pct_65_plus",
        "floodplain_pct",
    ]
