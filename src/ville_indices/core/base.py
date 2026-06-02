"""Base interface for composite indices."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from ville_indices.core.metadata import new_run_id
from ville_indices.core.recipe import Recipe, load_recipe
from ville_indices.core.validation import FeatureTableValidator, ValidationReport


class CompositeIndex(ABC):
    """Lifecycle interface for composite-index implementations.

    The base class enforces input/output discipline but intentionally does not
    assume that indices are additive weighted sums.
    """

    index_name: str = "abstract_composite_index"
    construct_measured: str = "unspecified"
    score_direction: str = "custom"

    def __init__(self, recipe: Recipe | str | Path, run_id: str | None = None):
        self.recipe = load_recipe(recipe) if isinstance(recipe, (str, Path)) else recipe
        self.run_id = run_id or new_run_id(self.recipe.name)
        self.validation_report: ValidationReport | None = None
        self.missing_report: Any | None = None
        self.process_metadata: dict[str, Any] = {}
        self.standard_output: pd.DataFrame | None = None
        self.intermediate_output: pd.DataFrame | None = None
        self._is_fitted = False

    def required_variables(self) -> list[str]:
        return self.recipe.required_variables

    def validate_inputs(
        self, feature_table: pd.DataFrame, *, raise_on_error: bool = False
    ) -> ValidationReport:
        report = FeatureTableValidator(self.recipe).validate(feature_table)
        self.validation_report = report
        if raise_on_error and report.has_errors:
            messages = "; ".join(issue.message for issue in report.errors)
            raise ValueError(f"Input validation failed: {messages}")
        return report

    @abstractmethod
    def fit(self, feature_table: pd.DataFrame) -> "CompositeIndex":
        """Fit any data-derived parameters needed for this index."""

    @abstractmethod
    def transform(self, feature_table: pd.DataFrame) -> pd.DataFrame:
        """Transform a canonical feature table into standardized outputs."""

    def fit_transform(self, feature_table: pd.DataFrame) -> pd.DataFrame:
        self.fit(feature_table)
        return self.transform(feature_table)

    def get_metadata(self) -> dict[str, Any]:
        return self.process_metadata

    def explain(self, zone_id: Any) -> dict[str, Any]:
        if self.intermediate_output is None:
            raise RuntimeError("No intermediate output is available. Run transform first.")
        spatial_id = self.recipe.spatial_id_column
        matches = self.intermediate_output[self.intermediate_output[spatial_id] == zone_id]
        if matches.empty:
            raise KeyError(f"Zone '{zone_id}' is not present in intermediate output.")
        return matches.iloc[0].to_dict()
