"""Standardized benchmark output helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ville_indices.core.constants import STANDARD_OUTPUT_COLUMNS
from ville_indices.core.recipe import Recipe
from ville_indices.operations.normalization import normalize_series
from ville_indices.operations.ranking import percentile_rank


@dataclass
class IndexResult:
    standard_output: pd.DataFrame
    intermediate_output: pd.DataFrame
    process_metadata: dict[str, Any]


def create_standard_output(
    *,
    feature_table: pd.DataFrame,
    recipe: Recipe,
    run_id: str,
    score: pd.Series,
    score_normalized_0_1: pd.Series | None = None,
    percentile: pd.Series | None = None,
    rank: pd.Series | None = None,
    classification: pd.Series | None = None,
    missing_count: pd.Series | None = None,
    quality_flag: pd.Series | None = None,
) -> pd.DataFrame:
    spatial_id = recipe.spatial_id_column
    score = pd.Series(score, index=feature_table.index, dtype="float64")
    if score_normalized_0_1 is None:
        normalized_score, normalized_meta = normalize_series(score, method="minmax")
    else:
        normalized_score = pd.Series(
            score_normalized_0_1, index=feature_table.index, dtype="float64"
        )
        normalized_meta = {"method": "provided_by_index"}
    if percentile is None:
        percentile, percentile_meta = percentile_rank(score, ascending=True, tie_method="min")
    else:
        percentile = pd.Series(percentile, index=feature_table.index, dtype="float64")
        percentile_meta = {"method": "provided_by_index"}
    if rank is None:
        rank = score.rank(method="min", ascending=False, na_option="bottom").astype(int)
    else:
        rank = pd.Series(rank, index=feature_table.index).astype(int)

    if missing_count is None:
        missing_count = pd.Series(0, index=feature_table.index, dtype="int64")
    if quality_flag is None:
        quality_flag = pd.Series(
            np.where(pd.Series(missing_count, index=feature_table.index) > 0, "warning", "ok"),
            index=feature_table.index,
        )

    output = pd.DataFrame(
        {
            "zone_id": feature_table[spatial_id].values,
            "index_name": recipe.name,
            "run_id": run_id,
            "score_raw": score.values,
            "score_normalized_0_1": normalized_score.values,
            "score_direction": recipe.score_direction,
            "rank": rank.values,
            "percentile": percentile.values,
            "missing_count": pd.Series(missing_count, index=feature_table.index).values,
            "quality_flag": pd.Series(quality_flag, index=feature_table.index).values,
            "reproduction_level": recipe.reproduction_level,
        }
    )
    if classification is not None:
        output["class"] = pd.Series(classification, index=feature_table.index).values
    output.attrs["score_normalization_metadata"] = normalized_meta
    output.attrs["score_percentile_metadata"] = percentile_meta
    return output


def assert_standard_output_schema(output: pd.DataFrame) -> None:
    missing = [column for column in STANDARD_OUTPUT_COLUMNS if column not in output.columns]
    if missing:
        raise ValueError(f"Standard output is missing required columns: {missing}")
