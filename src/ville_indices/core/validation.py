"""Structured feature-table validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ville_indices.core.recipe import Recipe, VariableConfig
from ville_indices.core.serialization import to_jsonable, write_json


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    variable: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    spatial_id_column: str | None = None
    required_variables: list[str] = field(default_factory=list)
    optional_variables: list[str] = field(default_factory=list)
    variables_present: list[str] = field(default_factory=list)
    variables_missing: list[str] = field(default_factory=list)
    optional_variables_missing: list[str] = field(default_factory=list)
    missingness: dict[str, dict[str, float | int]] = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["issues"] = [asdict(issue) for issue in self.issues]
        return to_jsonable(data)

    def to_json(self, path: str | Path) -> None:
        write_json(self.to_dict(), path)

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)


def _infer_nonnegative(variable: VariableConfig) -> bool:
    if variable.nonnegative is not None:
        return bool(variable.nonnegative)
    text = " ".join(
        str(part).lower()
        for part in [variable.unit, variable.scale, variable.canonical_name]
        if part is not None
    )
    return any(token in text for token in ["pct", "percent", "proportion", "ratio"])


def _is_proportion(variable: VariableConfig) -> bool:
    text = " ".join(
        str(part).lower()
        for part in [variable.unit, variable.scale]
        if part is not None
    )
    return "proportion" in text or "0_1" in text


def _is_percent(variable: VariableConfig) -> bool:
    text = " ".join(
        str(part).lower()
        for part in [variable.unit, variable.scale]
        if part is not None
    )
    return "percent" in text or "0_100" in text or "%" in text


class FeatureTableValidator:
    """Validate canonical feature tables against a recipe."""

    def __init__(self, recipe: Recipe, high_missingness_threshold: float = 0.2):
        self.recipe = recipe
        self.high_missingness_threshold = high_missingness_threshold

    def validate(self, feature_table: pd.DataFrame) -> ValidationReport:
        issues: list[ValidationIssue] = []
        report = ValidationReport(
            is_valid=True,
            row_count=int(len(feature_table)),
            column_count=int(feature_table.shape[1]),
            spatial_id_column=self.recipe.spatial_id_column,
            required_variables=self.recipe.required_variables,
            optional_variables=self.recipe.optional_variables,
        )

        duplicated_columns = feature_table.columns[feature_table.columns.duplicated()].tolist()
        if duplicated_columns:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="duplicate_columns",
                    message="Feature table contains duplicate column names.",
                    details={"columns": duplicated_columns},
                )
            )

        spatial_id = self.recipe.spatial_id_column
        if spatial_id not in feature_table.columns:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_spatial_id_column",
                    message=f"Spatial ID column '{spatial_id}' is missing.",
                    variable=spatial_id,
                )
            )
        else:
            if feature_table[spatial_id].isna().any():
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="missing_spatial_id_values",
                        message=f"Spatial ID column '{spatial_id}' contains missing values.",
                        variable=spatial_id,
                    )
                )
            duplicated_ids = feature_table.loc[
                feature_table[spatial_id].duplicated(keep=False), spatial_id
            ].tolist()
            if duplicated_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="duplicate_spatial_ids",
                        message=f"Spatial ID column '{spatial_id}' contains duplicate IDs.",
                        variable=spatial_id,
                        details={"duplicate_ids": duplicated_ids},
                    )
                )

        missing_strategy = self.recipe.missing_data.get("strategy", "error")
        for variable in self.recipe.variables.values():
            column = variable.canonical_name
            if column not in feature_table.columns:
                if variable.required:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="missing_required_variable",
                            message=f"Required variable '{column}' is missing.",
                            variable=column,
                            details={"recipe_key": variable.key},
                        )
                    )
                    report.variables_missing.append(column)
                else:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            code="missing_optional_variable",
                            message=f"Optional variable '{column}' is missing.",
                            variable=column,
                            details={"recipe_key": variable.key},
                        )
                    )
                    report.optional_variables_missing.append(column)
                continue

            report.variables_present.append(column)
            series = feature_table[column]

            if variable.numeric and not pd.api.types.is_numeric_dtype(series):
                severity = "error" if variable.required else "warning"
                issues.append(
                    ValidationIssue(
                        severity=severity,
                        code="nonnumeric_variable",
                        message=f"Variable '{column}' must be numeric for this operation.",
                        variable=column,
                        details={"dtype": str(series.dtype), "recipe_key": variable.key},
                    )
                )
                continue

            missing_count = int(series.isna().sum())
            missing_pct = float(missing_count / len(series)) if len(series) else 0.0
            report.missingness[column] = {
                "missing_count": missing_count,
                "missing_pct": missing_pct,
            }
            if missing_count:
                severity = "error" if variable.required and missing_strategy == "error" else "warning"
                issues.append(
                    ValidationIssue(
                        severity=severity,
                        code="missing_values_detected",
                        message=f"Variable '{column}' contains missing values.",
                        variable=column,
                        details={
                            "missing_count": missing_count,
                            "missing_pct": missing_pct,
                            "strategy": missing_strategy,
                        },
                    )
                )
            if missing_pct >= self.high_missingness_threshold and missing_count:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="high_missingness",
                        message=f"Variable '{column}' has high missingness.",
                        variable=column,
                        details={
                            "missing_pct": missing_pct,
                            "threshold": self.high_missingness_threshold,
                        },
                    )
                )

            nonmissing = series.dropna()
            if nonmissing.empty:
                severity = "error" if variable.required else "warning"
                issues.append(
                    ValidationIssue(
                        severity=severity,
                        code="all_null_variable",
                        message=f"Variable '{column}' contains only null values.",
                        variable=column,
                    )
                )
                continue

            unique_count = int(nonmissing.nunique(dropna=True))
            if unique_count <= 1:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="constant_column",
                        message=f"Variable '{column}' is constant among nonmissing values.",
                        variable=column,
                        details={"unique_count": unique_count},
                    )
                )

            if variable.numeric:
                min_value = float(np.nanmin(nonmissing.to_numpy(dtype=float)))
                max_value = float(np.nanmax(nonmissing.to_numpy(dtype=float)))
                if _infer_nonnegative(variable) and min_value < 0:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            code="negative_values_for_nonnegative_variable",
                            message=f"Variable '{column}' has negative values but is declared nonnegative.",
                            variable=column,
                            details={"min": min_value},
                        )
                    )
                if _is_proportion(variable):
                    if min_value < 0 or max_value > 1:
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                code="proportion_out_of_range",
                                message=f"Variable '{column}' is declared as a 0-1 proportion but falls outside that range.",
                                variable=column,
                                details={"min": min_value, "max": max_value},
                            )
                        )
                    if 1 < max_value <= 100:
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                code="suspicious_percentage_scale",
                                message=f"Variable '{column}' may be encoded as 0-100 percent instead of 0-1 proportion.",
                                variable=column,
                                details={"max": max_value},
                            )
                        )
                if _is_percent(variable):
                    if min_value < 0 or max_value > 100:
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                code="percent_out_of_range",
                                message=f"Variable '{column}' is declared as percent but falls outside 0-100.",
                                variable=column,
                                details={"min": min_value, "max": max_value},
                            )
                        )
                    if 0 <= min_value and max_value <= 1:
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                code="suspicious_proportion_scale",
                                message=f"Variable '{column}' may be encoded as 0-1 proportion instead of 0-100 percent.",
                                variable=column,
                                details={"max": max_value},
                            )
                        )

        report.issues = issues
        report.is_valid = not any(issue.severity == "error" for issue in issues)
        return report
