"""SVI-like Social Vulnerability Index implementation.

This module implements the rank-based CDC/ATSDR SVI-style methodology:
variable percentile ranks -> domain raw sums -> domain percentile ranks ->
overall domain-percentile sum -> final overall percentile -> flags.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ville_indices.core.base import CompositeIndex
from ville_indices.core.outputs import create_standard_output
from ville_indices.core.validation import FeatureTableValidator, ValidationIssue, ValidationReport
from ville_indices.operations.classification import classify_series
from ville_indices.operations.missing_data import MissingDataHandler
from ville_indices.operations.ranking import percentile_rank


SVI_METHOD_NOTE = (
    "SVI does not use PCA/factor analysis. It uses percentile ranks and "
    "predefined conceptual domains."
)

EXPECTED_SVI_VARIABLES = [
    "pct_below_poverty",
    "pct_unemployed",
    "per_capita_income",
    "pct_no_high_school",
    "pct_age_65_plus",
    "pct_age_17_or_younger",
    "pct_disability",
    "pct_single_parent_households",
    "pct_minority",
    "pct_limited_language",
    "pct_multiunit_structures",
    "pct_mobile_homes",
    "pct_crowding",
    "pct_no_vehicle",
    "pct_group_quarters",
]


class SviLikeIndex(CompositeIndex):
    """SVI-like social vulnerability index.

    The implementation is recipe-driven: variable keys define SVI conceptual
    variables, while each variable's `canonical_name` defines the input feature
    table column to read. This supports local proxies without hardcoding raw
    dataset-specific column names.
    """

    index_name = "svi_like"
    construct_measured = "social_vulnerability"
    score_direction = "higher_is_more_vulnerable"

    def __init__(self, recipe, run_id: str | None = None):
        super().__init__(recipe, run_id=run_id)
        self.domains = self._recipe_domains()
        self.variable_keys = [
            variable_key
            for domain_variables in self.domains.values()
            for variable_key in domain_variables
        ]
        self.primary_scope_name = "global"
        self._validate_svi_recipe()

    def _recipe_domains(self) -> dict[str, list[str]]:
        domains = self.recipe.extra.get("domains")
        if not isinstance(domains, dict) or not domains:
            raise ValueError("SVI recipe must define top-level domains.")
        parsed: dict[str, list[str]] = {}
        for domain_name, domain_config in domains.items():
            if not isinstance(domain_config, dict) or "variables" not in domain_config:
                raise ValueError(f"SVI domain '{domain_name}' must define a variables list.")
            parsed[str(domain_name)] = list(domain_config["variables"])
        return parsed

    def _validate_svi_recipe(self) -> None:
        missing_from_recipe = [
            key for key in self.variable_keys if key not in self.recipe.variables
        ]
        if missing_from_recipe:
            raise ValueError(
                "SVI recipe domains reference variables that are not defined: "
                + ", ".join(missing_from_recipe)
            )
        if self.recipe.reproduction_level not in {
            "strict_original_like",
            "local_adaptation",
            "partial_svi_like",
        }:
            raise ValueError(
                "SVI reproduction_level must be strict_original_like, "
                "local_adaptation, or partial_svi_like."
            )
        if self.recipe.reproduction_level != "partial_svi_like":
            missing_expected = [
                key for key in EXPECTED_SVI_VARIABLES if key not in self.variable_keys
            ]
            if missing_expected:
                raise ValueError(
                    "Strict/local SVI recipes must include all 15 SVI variables. "
                    f"Missing: {missing_expected}"
                )
        for key in self.variable_keys:
            direction = self.recipe.variables[key].direction
            if direction not in {"positive", "negative"}:
                raise ValueError(
                    f"SVI variable '{key}' must have direction positive or negative."
                )

    @property
    def _partial_allowed(self) -> bool:
        partial_config = self.recipe.extra.get("partial", {})
        return (
            self.recipe.reproduction_level == "partial_svi_like"
            and bool(partial_config.get("allow_missing_required_variables", False))
        )

    @property
    def _ranking_config(self) -> dict[str, Any]:
        return dict(self.recipe.extra.get("ranking", {}))

    @property
    def _flag_threshold(self) -> float:
        flag_config = self.recipe.extra.get("flags", {})
        return float(flag_config.get("variable_flag_threshold", 0.90))

    @property
    def _population_column(self) -> str | None:
        population_column = self.recipe.extra.get("population_column")
        return str(population_column) if population_column else None

    def _append_issue(self, report: ValidationReport, issue: ValidationIssue) -> None:
        report.issues.append(issue)
        report.is_valid = not report.has_errors

    def validate_inputs(
        self, feature_table: pd.DataFrame, *, raise_on_error: bool = False
    ) -> ValidationReport:
        report = FeatureTableValidator(self.recipe).validate(feature_table)
        if self._partial_allowed:
            for issue in report.issues:
                if issue.code == "missing_required_variable":
                    issue.severity = "warning"
                    issue.code = "missing_required_variable_partial_svi_like"
                    issue.message = (
                        issue.message
                        + " Continuing because partial_svi_like explicitly allows this."
                    )
        report.is_valid = not report.has_errors

        population_column = self._population_column
        if population_column:
            if population_column not in feature_table.columns:
                self._append_issue(
                    report,
                    ValidationIssue(
                        severity="error",
                        code="missing_population_column",
                        message=(
                            f"Configured population column '{population_column}' is missing."
                        ),
                        variable=population_column,
                    ),
                )
            elif not pd.api.types.is_numeric_dtype(feature_table[population_column]):
                self._append_issue(
                    report,
                    ValidationIssue(
                        severity="error",
                        code="nonnumeric_population_column",
                        message=(
                            f"Configured population column '{population_column}' must be numeric."
                        ),
                        variable=population_column,
                    ),
                )
            else:
                nonpositive_count = int((feature_table[population_column] <= 0).sum())
                if nonpositive_count:
                    self._append_issue(
                        report,
                        ValidationIssue(
                            severity="warning",
                            code="nonpositive_population_units",
                            message=(
                                "SVI will apply the configured zero/invalid population rule "
                                f"to {nonpositive_count} unit(s)."
                            ),
                            variable=population_column,
                            details={"count": nonpositive_count},
                        ),
                    )

        comparison = self.recipe.extra.get("comparison", {})
        for scope in comparison.get("scopes", []):
            group_column = scope.get("group_column")
            if group_column and group_column not in feature_table.columns:
                self._append_issue(
                    report,
                    ValidationIssue(
                        severity="error",
                        code="missing_comparison_group_column",
                        message=(
                            f"Comparison group column '{group_column}' is configured but missing."
                        ),
                        variable=group_column,
                    ),
                )

        if raise_on_error and report.has_errors:
            messages = "; ".join(issue.message for issue in report.errors)
            raise ValueError(f"Input validation failed: {messages}")
        self.validation_report = report
        return report

    def fit(self, feature_table: pd.DataFrame) -> "SviLikeIndex":
        self.validate_inputs(feature_table, raise_on_error=True)
        self.missing_handler = MissingDataHandler(self.recipe)
        self.missing_handler.fit(feature_table)
        self._is_fitted = True
        return self

    def transform(self, feature_table: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("SviLikeIndex must be fitted before transform.")

        self.validate_inputs(feature_table, raise_on_error=True)
        handled, missing_report = self.missing_handler.transform(feature_table)
        working, population_metadata = self._apply_population_rule(handled)
        self.missing_report = missing_report

        intermediate = self._compute_svi_global(working)
        classes, classification_metadata = classify_series(
            intermediate["svi_overall_percentile"],
            method=self.recipe.classification.get("method", "none"),
            n_classes=int(self.recipe.classification.get("n_classes", 5)),
        )

        standard_output = create_standard_output(
            feature_table=working,
            recipe=self.recipe,
            run_id=self.run_id,
            score=intermediate["svi_overall_sum"],
            score_normalized_0_1=intermediate["svi_overall_percentile"],
            percentile=intermediate["svi_overall_percentile"],
            rank=intermediate["svi_rank"],
            classification=classes,
            missing_count=intermediate["svi_missing_variable_count"],
            quality_flag=intermediate["svi_quality_flag"],
        )
        if classes is not None:
            intermediate["class"] = classes

        self.standard_output = standard_output
        self.intermediate_output = intermediate
        self.process_metadata = self._build_process_metadata(
            intermediate=intermediate,
            population_metadata=population_metadata,
            classification_metadata=classification_metadata,
        )
        return standard_output

    def _apply_population_rule(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        population_column = self._population_column
        if not population_column:
            return frame.copy(), {
                "population_column": None,
                "action": "not_configured",
                "excluded_units": [],
                "excluded_count": 0,
            }

        config = dict(self.recipe.extra.get("zero_population", {}))
        action = config.get("action", "exclude")
        threshold = float(config.get("threshold", 0))
        invalid_mask = frame[population_column].isna() | (frame[population_column] <= threshold)
        excluded_units = frame.loc[invalid_mask, self.recipe.spatial_id_column].tolist()
        metadata = {
            "population_column": population_column,
            "action": action,
            "threshold": threshold,
            "excluded_units": excluded_units if action == "exclude" else [],
            "excluded_count": len(excluded_units) if action == "exclude" else 0,
            "flagged_units": excluded_units if action == "flag" else [],
            "flagged_count": len(excluded_units) if action == "flag" else 0,
        }
        if action == "exclude":
            return frame.loc[~invalid_mask].copy(), metadata
        if action == "flag":
            flagged = frame.copy()
            flagged["svi_invalid_population_flag"] = invalid_mask
            return flagged, metadata
        if action == "error" and invalid_mask.any():
            raise ValueError(
                "Population column contains zero/invalid values and zero_population.action is error."
            )
        if action != "error":
            raise NotImplementedError(f"Unsupported zero_population.action: {action}")
        return frame.copy(), metadata

    def _rank_variable(self, series: pd.Series, *, direction: str) -> tuple[pd.Series, dict[str, Any]]:
        ranking = self._ranking_config
        tie_method = ranking.get("tie_method", "min")
        n_equals_one = ranking.get("n_equals_one", "zero")
        ascending = direction == "positive"
        ranked, metadata = percentile_rank(
            series,
            ascending=ascending,
            tie_method=tie_method,
            n_equals_one=n_equals_one,
        )
        metadata["direction"] = direction
        metadata["direction_handling"] = (
            "positive variables rank ascending; negative variables rank descending"
        )
        return ranked.astype("float64"), metadata

    def _compute_svi_global(self, frame: pd.DataFrame) -> pd.DataFrame:
        spatial_id = self.recipe.spatial_id_column
        intermediate = pd.DataFrame({spatial_id: frame[spatial_id].values}, index=frame.index)
        population_column = self._population_column
        if population_column and population_column in frame.columns:
            intermediate[population_column] = frame[population_column]
        if "svi_invalid_population_flag" in frame.columns:
            intermediate["svi_invalid_population_flag"] = frame["svi_invalid_population_flag"]

        threshold = self._flag_threshold
        self._variable_rank_metadata: dict[str, Any] = {}

        for key in self.variable_keys:
            variable = self.recipe.variables[key]
            raw_column = f"svi_raw_{key}"
            missing_column = f"svi_missing_{key}"
            proxy_column = f"svi_proxy_{key}"
            percentile_column = f"svi_pr_{key}"
            flag_column = f"svi_flag_{key}"

            is_proxy = self._is_proxy_variable(key)
            intermediate[proxy_column] = bool(is_proxy)

            if variable.canonical_name not in frame.columns:
                intermediate[raw_column] = pd.NA
                intermediate[missing_column] = True
                intermediate[percentile_column] = pd.NA
                intermediate[flag_column] = 0
                self._variable_rank_metadata[key] = {
                    "method": "percentile_rank",
                    "status": "not_computed_missing_variable",
                    "canonical_name": variable.canonical_name,
                    "recipe_key": key,
                }
                continue

            intermediate[raw_column] = frame[variable.canonical_name]
            intermediate[missing_column] = frame[variable.canonical_name].isna()
            percentile, metadata = self._rank_variable(
                frame[variable.canonical_name],
                direction=variable.direction,
            )
            intermediate[percentile_column] = percentile
            intermediate[flag_column] = percentile.ge(threshold).fillna(False).astype(int)
            self._variable_rank_metadata[key] = {
                **metadata,
                "method": "percentile_rank",
                "canonical_name": variable.canonical_name,
                "recipe_key": key,
                "proxy_used": is_proxy,
            }

        self._domain_rank_metadata: dict[str, Any] = {}
        domain_percentile_columns: list[str] = []
        for domain_name, variables in self.domains.items():
            pr_columns = [f"svi_pr_{key}" for key in variables]
            flag_columns = [f"svi_flag_{key}" for key in variables]
            sum_column = f"svi_{domain_name}_sum"
            percentile_column = f"svi_{domain_name}_percentile"
            flag_count_column = f"svi_{domain_name}_flag_count"
            intermediate[sum_column] = intermediate[pr_columns].sum(axis=1, min_count=1)
            domain_percentile, metadata = percentile_rank(
                intermediate[sum_column],
                ascending=True,
                tie_method=self._ranking_config.get("tie_method", "min"),
                n_equals_one=self._ranking_config.get("n_equals_one", "zero"),
            )
            intermediate[percentile_column] = domain_percentile.astype("float64")
            intermediate[flag_count_column] = intermediate[flag_columns].sum(axis=1).astype(int)
            domain_percentile_columns.append(percentile_column)
            self._domain_rank_metadata[domain_name] = {
                **metadata,
                "method": "percentile_rank",
                "source_column": sum_column,
                "variables": variables,
            }

        intermediate["svi_overall_sum"] = intermediate[domain_percentile_columns].sum(
            axis=1, min_count=1
        )
        overall_percentile, overall_metadata = percentile_rank(
            intermediate["svi_overall_sum"],
            ascending=True,
            tie_method=self._ranking_config.get("tie_method", "min"),
            n_equals_one=self._ranking_config.get("n_equals_one", "zero"),
        )
        intermediate["svi_overall_percentile"] = overall_percentile.astype("float64")
        intermediate["svi_rank"] = (
            intermediate["svi_overall_percentile"]
            .rank(method="min", ascending=False, na_option="bottom")
            .astype(int)
        )

        variable_missing_columns = [f"svi_missing_{key}" for key in self.variable_keys]
        variable_proxy_columns = [f"svi_proxy_{key}" for key in self.variable_keys]
        variable_flag_columns = [f"svi_flag_{key}" for key in self.variable_keys]
        intermediate["svi_missing_variable_count"] = intermediate[variable_missing_columns].sum(
            axis=1
        ).astype(int)
        intermediate["svi_proxy_variable_count"] = intermediate[variable_proxy_columns].sum(
            axis=1
        ).astype(int)
        intermediate["svi_total_flag_count"] = intermediate[variable_flag_columns].sum(
            axis=1
        ).astype(int)
        intermediate["svi_quality_flag"] = [
            self._quality_flag(row)
            for _, row in intermediate[
                [
                    "svi_missing_variable_count",
                    "svi_proxy_variable_count",
                    *(
                        ["svi_invalid_population_flag"]
                        if "svi_invalid_population_flag" in intermediate.columns
                        else []
                    ),
                ]
            ].iterrows()
        ]
        self._overall_rank_metadata = {
            **overall_metadata,
            "method": "percentile_rank",
            "source_column": "svi_overall_sum",
        }
        return intermediate

    def _quality_flag(self, row: pd.Series) -> str:
        if bool(row.get("svi_invalid_population_flag", False)):
            return "invalid_population"
        if int(row["svi_missing_variable_count"]) > 0:
            return "partial_svi_like"
        if int(row["svi_proxy_variable_count"]) > 0:
            return "proxy_used"
        return "ok"

    def _is_proxy_variable(self, key: str) -> bool:
        variable = self.recipe.variables[key]
        proxy_used = variable.extra.get("proxy_used")
        if proxy_used:
            return True
        original_variable = variable.extra.get("original_variable")
        if original_variable and original_variable in EXPECTED_SVI_VARIABLES:
            return variable.canonical_name != original_variable
        return variable.canonical_name != key

    def _proxied_variables(self) -> list[dict[str, Any]]:
        proxied: list[dict[str, Any]] = []
        for key in self.variable_keys:
            if self._is_proxy_variable(key):
                variable = self.recipe.variables[key]
                proxied.append(
                    {
                        "variable": key,
                        "canonical_name": variable.canonical_name,
                        "original_variable": variable.extra.get("original_variable", key),
                        "proxy_used": variable.extra.get("proxy_used", variable.canonical_name),
                        "proxy_quality": variable.extra.get("proxy_quality"),
                        "conceptual_risk": variable.extra.get("conceptual_risk"),
                        "status": variable.extra.get("status"),
                    }
                )
        return proxied

    def _missing_recipe_variables(self) -> list[dict[str, Any]]:
        if self.intermediate_output is None:
            return []
        missing: list[dict[str, Any]] = []
        for key in self.variable_keys:
            variable = self.recipe.variables[key]
            if f"svi_raw_{key}" in self.intermediate_output.columns:
                if self.intermediate_output[f"svi_raw_{key}"].isna().all():
                    missing.append({"variable": key, "canonical_name": variable.canonical_name})
        return missing

    def _domain_summary(self, intermediate: pd.DataFrame) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for domain_name in self.domains:
            percentile_column = f"svi_{domain_name}_percentile"
            sum_column = f"svi_{domain_name}_sum"
            summary[domain_name] = {
                "mean_sum": float(intermediate[sum_column].mean(skipna=True)),
                "mean_percentile": float(intermediate[percentile_column].mean(skipna=True)),
                "max_percentile": float(intermediate[percentile_column].max(skipna=True)),
            }
        return summary

    def _build_process_metadata(
        self,
        *,
        intermediate: pd.DataFrame,
        population_metadata: dict[str, Any],
        classification_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        warnings = []
        if self.recipe.reproduction_level == "partial_svi_like":
            warnings.append(
                "This run is partial_svi_like and is not a full SVI-like reproduction."
            )
        if self._proxied_variables():
            warnings.append("One or more SVI variables use documented local proxies.")
        if population_metadata.get("excluded_count", 0):
            warnings.append(
                f"{population_metadata['excluded_count']} unit(s) excluded because population <= "
                f"{population_metadata.get('threshold', 0)}."
            )

        ranking_config = {
            "method": "percentile_rank",
            "formula": self._ranking_config.get("formula", "(rank - 1) / (N - 1)"),
            "tie_method": self._ranking_config.get("tie_method", "min"),
            "n_equals_one_behavior": self._ranking_config.get("n_equals_one", "zero"),
            "higher_percentile_means": "higher_vulnerability",
            "variable_rank_metadata": self._variable_rank_metadata,
            "domain_rank_metadata": self._domain_rank_metadata,
            "overall_rank_metadata": self._overall_rank_metadata,
        }
        return {
            "normalization": {
                key: {
                    "method": "percentile_rank",
                    "canonical_name": self.recipe.variables[key].canonical_name,
                    "direction": self.recipe.variables[key].direction,
                    "formula": ranking_config["formula"],
                    "tie_method": ranking_config["tie_method"],
                }
                for key in self.variable_keys
            },
            "orientation": {
                key: {
                    "direction": self.recipe.variables[key].direction,
                    "transformation_applied": "ranking_direction",
                    "canonical_name": self.recipe.variables[key].canonical_name,
                }
                for key in self.variable_keys
            },
            "weighting_method": "SVI equal variable ranks within domains; equal domain percentiles overall",
            "aggregation": {
                "method": "svi_domain_sum_then_domain_percentile_then_overall_percentile",
                "not_direct_sum_of_15_variable_percentiles": True,
                "domains": self.domains,
            },
            "classification": classification_metadata,
            "ranking": ranking_config,
            "flags": {
                "variable_flag_threshold": self._flag_threshold,
                "rule": "flag = 1 when variable percentile >= threshold",
            },
            "domains": self.domains,
            "comparison": self.recipe.extra.get(
                "comparison", {"scopes": [{"name": "global", "group_column": None}]}
            ),
            "population_filter": population_metadata,
            "population_column": self._population_column,
            "variables_proxied": self._proxied_variables(),
            "variables_missing": self._missing_recipe_variables(),
            "methodology_note": SVI_METHOD_NOTE,
            "source_reference": self.recipe.extra.get(
                "source_reference", self.recipe.method_reference
            ),
            "warnings": warnings,
            "svi": {
                "reproduction_level": self.recipe.reproduction_level,
                "methodology_note": SVI_METHOD_NOTE,
                "domain_summary": self._domain_summary(intermediate),
                "overall_percentile_summary": {
                    "min": float(intermediate["svi_overall_percentile"].min(skipna=True)),
                    "mean": float(intermediate["svi_overall_percentile"].mean(skipna=True)),
                    "max": float(intermediate["svi_overall_percentile"].max(skipna=True)),
                },
                "area_level_warning": (
                    "SVI is an area-level index and should not be interpreted as saying "
                    "that every person in a high-SVI zone is vulnerable."
                ),
            },
        }

    def explain(self, zone_id: Any) -> dict[str, Any]:
        if self.intermediate_output is None or self.standard_output is None:
            raise RuntimeError("No SVI output is available. Run transform first.")
        spatial_id = self.recipe.spatial_id_column
        matches = self.intermediate_output[self.intermediate_output[spatial_id] == zone_id]
        standard_matches = self.standard_output[self.standard_output["zone_id"] == zone_id]
        if matches.empty or standard_matches.empty:
            raise KeyError(f"Zone '{zone_id}' is not present in SVI output.")
        row = matches.iloc[0]
        standard_row = standard_matches.iloc[0]
        domain_percentiles = {
            domain: float(row[f"svi_{domain}_percentile"]) for domain in self.domains
        }
        domain_flag_counts = {
            domain: int(row[f"svi_{domain}_flag_count"]) for domain in self.domains
        }
        variable_percentiles = {
            key: float(row[f"svi_pr_{key}"])
            for key in self.variable_keys
            if pd.notna(row[f"svi_pr_{key}"])
        }
        flagged_variables = [
            key for key in self.variable_keys if int(row[f"svi_flag_{key}"]) == 1
        ]
        proxy_variables = [
            key for key in self.variable_keys if bool(row[f"svi_proxy_{key}"])
        ]
        missing_variables = [
            key for key in self.variable_keys if bool(row[f"svi_missing_{key}"])
        ]
        highest_domain = max(domain_percentiles, key=domain_percentiles.get)
        top_contributors = sorted(
            variable_percentiles.items(), key=lambda item: item[1], reverse=True
        )[:5]
        return {
            "zone_id": zone_id,
            "svi_overall_percentile": float(row["svi_overall_percentile"]),
            "svi_overall_sum": float(row["svi_overall_sum"]),
            "final_rank": int(standard_row["rank"]),
            "highest_domain": highest_domain,
            "domain_percentiles": domain_percentiles,
            "domain_flag_counts": domain_flag_counts,
            "top_contributing_variable_percentiles": dict(top_contributors),
            "flagged_variables": flagged_variables,
            "total_flag_count": int(row["svi_total_flag_count"]),
            "missing_variables": missing_variables,
            "proxy_variables": proxy_variables,
            "quality_flag": row["svi_quality_flag"],
            "interpretation": (
                "This is an area-level SVI-like percentile. Higher values indicate "
                "higher relative social vulnerability for the spatial unit, not a "
                "deterministic claim about every individual in that area."
            ),
        }
