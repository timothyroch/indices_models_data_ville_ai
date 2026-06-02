"""Toy dummy additive index used only to validate the framework architecture."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ville_indices.core.base import CompositeIndex
from ville_indices.core.outputs import create_standard_output
from ville_indices.operations.aggregation import aggregate
from ville_indices.operations.classification import classify_series
from ville_indices.operations.missing_data import MissingDataHandler
from ville_indices.operations.normalization import normalize_series
from ville_indices.operations.orientation import orient_series


class DummyAdditiveIndex(CompositeIndex):
    """A synthetic toy index proving the shared framework works end to end."""

    index_name = "dummy_additive_index"
    construct_measured = "synthetic_vulnerability"
    score_direction = "higher_is_more_vulnerable"

    def fit(self, feature_table: pd.DataFrame) -> "DummyAdditiveIndex":
        report = self.validate_inputs(feature_table, raise_on_error=True)
        self.validation_report = report
        self.missing_handler = MissingDataHandler(self.recipe)
        handled, missing_report = self.missing_handler.fit_transform(feature_table)
        self.missing_report = missing_report

        normalization: dict[str, dict[str, Any]] = {}
        for key, variable in self.recipe.variables.items():
            normalized, metadata = normalize_series(
                handled[variable.canonical_name],
                method=variable.normalization.method,
                parameters=variable.normalization.parameters,
                fit=True,
            )
            normalization[key] = metadata
            normalization[key]["canonical_name"] = variable.canonical_name
            normalization[key]["recipe_key"] = key
            # Keep the fitted values reachable for sanity/debugging, but only
            # metadata parameters are reused during transform.
            _ = normalized

        self.process_metadata["normalization"] = normalization
        self.process_metadata["weighting_method"] = "variable_weights_from_recipe"
        self._is_fitted = True
        return self

    def transform(self, feature_table: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("DummyAdditiveIndex must be fitted before transform.")

        report = self.validate_inputs(feature_table, raise_on_error=True)
        self.validation_report = report
        handled, missing_report = self.missing_handler.transform(feature_table)
        self.missing_report = missing_report

        spatial_id = self.recipe.spatial_id_column
        intermediate = pd.DataFrame({spatial_id: handled[spatial_id].values}, index=handled.index)
        oriented_columns: list[str] = []
        weights: dict[str, float] = {}
        orientation_metadata: dict[str, Any] = {}

        for key, variable in self.recipe.variables.items():
            column = variable.canonical_name
            raw_column = f"{key}__raw"
            normalized_column = f"{key}__normalized"
            oriented_column = f"{key}__oriented"
            contribution_column = f"{key}__weighted_contribution"

            intermediate[raw_column] = handled[column]
            params = self.process_metadata["normalization"][key]
            normalized, normalization_metadata = normalize_series(
                handled[column],
                method=variable.normalization.method,
                parameters=params,
                fit=False,
            )
            self.process_metadata["normalization"][key].update(normalization_metadata)
            negative_mode = (
                "multiply_by_minus_one"
                if variable.normalization.method == "zscore"
                else "invert_0_1"
            )
            oriented, orientation_meta = orient_series(
                normalized,
                direction=variable.direction,
                phase="after_normalization",
                negative_mode=negative_mode,
            )
            weight = variable.weight.value
            if weight is None:
                raise ValueError(f"Dummy index requires a numeric weight for '{key}'.")

            intermediate[normalized_column] = normalized
            intermediate[oriented_column] = oriented
            intermediate[contribution_column] = oriented * float(weight)
            oriented_columns.append(oriented_column)
            weights[oriented_column] = float(weight)
            orientation_metadata[key] = {
                **orientation_meta,
                "canonical_name": column,
                "recipe_key": key,
            }

            missing_flag_column = f"{column}__missing_flag"
            if missing_flag_column in handled.columns:
                intermediate[f"{key}__missing_flag"] = handled[missing_flag_column]

        score, aggregation_metadata = aggregate(
            intermediate,
            columns=oriented_columns,
            method=self.recipe.aggregation.get("method", "weighted_sum"),
            weights=weights,
        )
        intermediate["score_raw"] = score

        classes, classification_metadata = classify_series(
            score,
            method=self.recipe.classification.get("method", "none"),
            n_classes=int(self.recipe.classification.get("n_classes", 5)),
        )

        row_missing_counts = pd.Series(
            {
                zone_id: count
                for zone_id, count in missing_report.row_missing_counts.items()
            }
        )
        missing_count = handled[spatial_id].map(row_missing_counts).fillna(0).astype(int)
        quality_flag = pd.Series(
            ["warning" if value > 0 else "ok" for value in missing_count],
            index=handled.index,
        )
        standard_output = create_standard_output(
            feature_table=handled,
            recipe=self.recipe,
            run_id=self.run_id,
            score=score,
            classification=classes,
            missing_count=missing_count,
            quality_flag=quality_flag,
        )

        if classes is not None:
            intermediate["class"] = classes
        self.standard_output = standard_output
        self.intermediate_output = intermediate
        self.process_metadata["orientation"] = orientation_metadata
        self.process_metadata["aggregation"] = aggregation_metadata
        self.process_metadata["classification"] = classification_metadata
        return standard_output
