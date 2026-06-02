"""Configurable missing-data handling with structured reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ville_indices.core.recipe import Recipe
from ville_indices.core.serialization import to_jsonable, write_json


@dataclass
class MissingDataReport:
    strategy: str
    missing_count_per_variable: dict[str, int] = field(default_factory=dict)
    missing_pct_per_variable: dict[str, float] = field(default_factory=dict)
    affected_spatial_units_count: int = 0
    affected_spatial_units: list[Any] = field(default_factory=list)
    add_missing_flags: bool = False
    imputation_values: dict[str, float] = field(default_factory=dict)
    variables_dropped: list[str] = field(default_factory=list)
    units_dropped: list[Any] = field(default_factory=list)
    row_missing_counts: dict[Any, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))

    def to_json(self, path: str | Path) -> None:
        write_json(self.to_dict(), path)

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)


class MissingDataHandler:
    """Fit/apply a missing-data strategy from a recipe."""

    def __init__(self, recipe: Recipe):
        self.recipe = recipe
        self.strategy = recipe.missing_data.get("strategy", "error")
        self.add_missing_flags = bool(recipe.missing_data.get("add_missing_flags", False))
        self.imputation_values_: dict[str, float] = {}
        self._is_fitted = False

    def _columns(self, frame: pd.DataFrame) -> list[str]:
        return [column for column in self.recipe.variable_columns if column in frame.columns]

    def _build_report(self, frame: pd.DataFrame) -> MissingDataReport:
        columns = self._columns(frame)
        spatial_id = self.recipe.spatial_id_column
        missing_mask = frame[columns].isna() if columns else pd.DataFrame(index=frame.index)
        row_counts = missing_mask.sum(axis=1).astype(int) if columns else pd.Series(0, index=frame.index)
        affected = frame.loc[row_counts > 0, spatial_id].tolist() if spatial_id in frame.columns else []
        return MissingDataReport(
            strategy=self.strategy,
            missing_count_per_variable={
                column: int(frame[column].isna().sum()) for column in columns
            },
            missing_pct_per_variable={
                column: float(frame[column].isna().sum() / len(frame)) if len(frame) else 0.0
                for column in columns
            },
            affected_spatial_units_count=len(affected),
            affected_spatial_units=affected,
            add_missing_flags=self.add_missing_flags,
            row_missing_counts={
                frame.loc[index, spatial_id]: int(count)
                for index, count in row_counts.items()
                if spatial_id in frame.columns
            },
        )

    def fit(self, frame: pd.DataFrame) -> "MissingDataHandler":
        columns = self._columns(frame)
        if self.strategy == "median_imputation":
            self.imputation_values_ = {
                column: float(frame[column].median(skipna=True)) for column in columns
            }
        elif self.strategy == "mean_imputation":
            self.imputation_values_ = {
                column: float(frame[column].mean(skipna=True)) for column in columns
            }
        elif self.strategy == "zero_imputation":
            self.imputation_values_ = {column: 0.0 for column in columns}
        self._is_fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, MissingDataReport]:
        if not self._is_fitted:
            raise RuntimeError("MissingDataHandler must be fitted before transform.")

        output = frame.copy()
        report = self._build_report(output)
        columns = self._columns(output)

        if self.add_missing_flags:
            for column in columns:
                output[f"{column}__missing_flag"] = output[column].isna()

        if self.strategy == "error":
            if any(report.missing_count_per_variable.values()):
                raise ValueError("Missing values found and missing_data.strategy is 'error'.")
        elif self.strategy == "drop_units":
            drop_mask = output[columns].isna().any(axis=1) if columns else pd.Series(False, index=output.index)
            spatial_id = self.recipe.spatial_id_column
            report.units_dropped = (
                output.loc[drop_mask, spatial_id].tolist() if spatial_id in output.columns else []
            )
            output = output.loc[~drop_mask].copy()
        elif self.strategy in {"median_imputation", "mean_imputation", "zero_imputation"}:
            report.imputation_values = dict(self.imputation_values_)
            for column, value in self.imputation_values_.items():
                output[column] = output[column].fillna(value)
        elif self.strategy == "keep_missing_with_flags":
            pass
        else:
            raise NotImplementedError(f"Missing-data strategy '{self.strategy}' is not implemented.")

        return output, report

    def fit_transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, MissingDataReport]:
        self.fit(frame)
        return self.transform(frame)
