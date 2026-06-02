"""SoVI-like Social Vulnerability Index implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ville_indices.core.base import CompositeIndex
from ville_indices.core.outputs import create_standard_output
from ville_indices.core.validation import FeatureTableValidator, ValidationIssue, ValidationReport
from ville_indices.operations.classification import (
    standard_deviation_band_classes,
    standard_deviation_zscores,
)
from ville_indices.operations.factor_analysis import (
    PCAResult,
    retain_factors,
    rotated_factor_scores,
    run_pca,
)
from ville_indices.operations.missing_data import MissingDataHandler
from ville_indices.operations.normalization import normalize_series
from ville_indices.operations.ranking import percentile_rank
from ville_indices.operations.rotation import varimax
from ville_indices.operations.standardization import zscore_standardize


SOVI_METHOD_NOTE = (
    "SoVI-like uses standardized variables, PCA/factor analysis, varimax rotation, "
    "recipe-driven factor orientation, and an additive sum of oriented factor scores."
)

SOVI_AREA_LEVEL_WARNING = (
    "SoVI is an area-level social vulnerability score and should not be interpreted "
    "as an individual-level diagnosis."
)

ORIGINAL_SOVI_CODES = [
    "MED_AGE90",
    "PERCAP89",
    "MVALOO90",
    "MEDRENT90",
    "PHYSICN90",
    "PCTVOTE92",
    "BRATE90",
    "MIGRA_97",
    "PCTFARMS92",
    "PCTBLACK90",
    "PCTINDIAN90",
    "PCTASIAN90",
    "PCTHISPANIC90",
    "PCTKIDS90",
    "PCTOLD90",
    "PCTVLUN91",
    "AVGPERHH",
    "PCTHH7589",
    "PCTPOV90",
    "PCTRENTER90",
    "PCTRFRM90",
    "DEBREV92",
    "PCTMOBL90",
    "PCTNOHS90",
    "HODENUT90",
    "HUPTDEN90",
    "MAESDEN92",
    "EARNDEN90",
    "COMDEVDN92",
    "RPROPDEN92",
    "CVBRPC91",
    "FEMLBR90",
    "AGRIPC90",
    "TRANPC90",
    "SERVPC90",
    "NRRESPC91",
    "HOSPTPC91",
    "PCCHGPOP90",
    "PCTURB90",
    "PCTFEM90",
    "PCTF_HH90",
    "SSBENPC90",
]


class SoviLikeIndex(CompositeIndex):
    """SoVI-like PCA/factor-analysis index."""

    index_name = "sovi_like"
    construct_measured = "social_vulnerability"
    score_direction = "higher_is_more_vulnerable"

    def __init__(self, recipe, run_id: str | None = None):
        super().__init__(recipe, run_id=run_id)
        if self.recipe.reproduction_level not in {
            "strict_original_like",
            "local_adaptation",
            "partial_sovi_like",
        }:
            raise ValueError(
                "SoVI reproduction_level must be strict_original_like, "
                "local_adaptation, or partial_sovi_like."
            )
        self.extra_outputs: dict[str, pd.DataFrame] = {}

    @property
    def _partial_allowed(self) -> bool:
        partial = self.recipe.extra.get("partial", {})
        return (
            self.recipe.reproduction_level == "partial_sovi_like"
            and bool(partial.get("allow_missing_required_variables", False))
        )

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
                    issue.code = "missing_required_variable_partial_sovi_like"
                    issue.message = (
                        issue.message
                        + " Continuing because partial_sovi_like explicitly allows this."
                    )
        report.is_valid = not report.has_errors

        if len(feature_table) < 2:
            self._append_issue(
                report,
                ValidationIssue(
                    severity="error",
                    code="too_few_observations",
                    message="SoVI PCA/factor analysis requires at least 2 observations.",
                    details={"row_count": len(feature_table)},
                ),
            )

        present_columns = [
            variable.canonical_name
            for variable in self.recipe.variables.values()
            if variable.canonical_name in feature_table.columns
        ]
        numeric_present = [
            column
            for column in present_columns
            if pd.api.types.is_numeric_dtype(feature_table[column])
        ]
        if len(numeric_present) < 2:
            self._append_issue(
                report,
                ValidationIssue(
                    severity="error",
                    code="too_few_usable_variables",
                    message="SoVI PCA/factor analysis requires at least 2 usable numeric variables.",
                    details={"usable_variable_count": len(numeric_present)},
                ),
            )

        if len(feature_table) <= len(numeric_present):
            self._append_issue(
                report,
                ValidationIssue(
                    severity="warning",
                    code="small_n_relative_to_variables",
                    message=(
                        "The number of observations is not larger than the number of "
                        "configured numeric variables; factor structure may be unstable."
                    ),
                    details={
                        "row_count": len(feature_table),
                        "usable_variable_count": len(numeric_present),
                    },
                ),
            )

        if raise_on_error and report.has_errors:
            messages = "; ".join(issue.message for issue in report.errors)
            raise ValueError(f"Input validation failed: {messages}")
        self.validation_report = report
        return report

    def fit(self, feature_table: pd.DataFrame) -> "SoviLikeIndex":
        self.validate_inputs(feature_table, raise_on_error=True)
        self.missing_handler = MissingDataHandler(self.recipe)
        handled, fit_missing_report = self.missing_handler.fit_transform(feature_table)
        self._fit_missing_report = fit_missing_report

        self._available_variable_keys = [
            key
            for key, variable in self.recipe.variables.items()
            if variable.canonical_name in handled.columns
        ]
        self._available_columns = [
            self.recipe.variables[key].canonical_name for key in self._available_variable_keys
        ]

        standard_config = self.recipe.extra.get("standardization", {})
        standard_result = zscore_standardize(
            handled,
            columns=self._available_columns,
            ddof=int(standard_config.get("ddof", 0)),
            zero_variance_behavior=standard_config.get("zero_variance_behavior", "drop"),
        )
        self._standardization_metadata = standard_result.metadata
        self._dropped_variables = list(standard_result.dropped_variables)
        self._used_columns = standard_result.metadata["used_variables"]
        self._used_variable_keys = [
            key
            for key in self._available_variable_keys
            if self.recipe.variables[key].canonical_name in self._used_columns
        ]

        if len(self._used_columns) < 2:
            raise ValueError("SoVI has fewer than 2 usable variables after standardization.")

        self._pca_result = run_pca(standard_result.standardized)
        retention = self.recipe.extra.get("factor_retention", {})
        n_factors, retention_metadata = retain_factors(
            self._pca_result.eigenvalues,
            method=retention.get("method", "eigenvalue_gt"),
            threshold=float(retention.get("threshold", 1.0)),
            n_factors=retention.get("n_factors"),
        )
        self._n_factors = n_factors
        self._retention_metadata = retention_metadata

        self._unrotated_loadings = self._pca_result.loadings[:, :n_factors]
        self._unrotated_scores_fit = self._pca_result.scores[:, :n_factors]
        rotation = self.recipe.extra.get("rotation", {})
        rotation_method = rotation.get("method", "varimax")
        if rotation_method == "varimax":
            (
                self._rotated_loadings,
                self._rotation_matrix,
                self._rotation_metadata,
            ) = varimax(
                self._unrotated_loadings,
                normalize=bool(rotation.get("normalize", True)),
                max_iter=int(rotation.get("max_iter", 500)),
                tol=float(rotation.get("tol", 1.0e-6)),
            )
        elif rotation_method == "none":
            self._rotated_loadings = self._unrotated_loadings.copy()
            self._rotation_matrix = np.eye(n_factors)
            self._rotation_metadata = {
                "method": "none",
                "methodological_variant": "varimax rotation skipped by recipe",
            }
        else:
            raise NotImplementedError(f"Rotation method '{rotation_method}' is not implemented.")

        (
            self._rotated_scores_fit,
            self._score_coefficients,
            self._factor_score_metadata,
        ) = rotated_factor_scores(standard_result.standardized, self._rotated_loadings)
        self._orientation_metadata = self._resolve_factor_orientation()
        self._is_fitted = True
        return self

    def transform(self, feature_table: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("SoviLikeIndex must be fitted before transform.")

        self.validate_inputs(feature_table, raise_on_error=True)
        handled, missing_report = self.missing_handler.transform(feature_table)
        self.missing_report = missing_report
        spatial_id = self.recipe.spatial_id_column

        standard_result = zscore_standardize(
            handled,
            columns=self._used_columns,
            parameters=self._standardization_metadata["variables"],
            fit=False,
            zero_variance_behavior="error",
        )
        standardized = standard_result.standardized
        unrotated_scores = standardized.to_numpy(dtype=float) @ self._pca_result.eigenvectors[
            :, : self._n_factors
        ]
        rotated_scores = standardized.to_numpy(dtype=float) @ self._score_coefficients
        oriented_scores = self._orient_factor_scores(rotated_scores)
        score_raw = oriented_scores.sum(axis=1)
        score = pd.Series(score_raw, index=handled.index, dtype="float64")
        score_z, z_metadata = standard_deviation_zscores(score)
        sovi_class = standard_deviation_band_classes(score_z)
        score_normalized, _ = normalize_series(score, method="minmax")
        percentile, _ = percentile_rank(score, ascending=True, tie_method="min")
        rank = score.rank(method="min", ascending=False, na_option="bottom").astype(int)

        intermediate = pd.DataFrame({spatial_id: handled[spatial_id].values}, index=handled.index)
        for key in self._available_variable_keys:
            column = self.recipe.variables[key].canonical_name
            if column in handled.columns:
                intermediate[f"sovi_raw_{key}"] = handled[column]
                missing_flag = f"{column}__missing_flag"
                if missing_flag in handled.columns:
                    intermediate[f"sovi_missing_{key}"] = handled[missing_flag]
        for column in self._used_columns:
            key = self._key_for_column(column)
            intermediate[f"sovi_z_{key}"] = standardized[column]

        for factor_index in range(self._n_factors):
            number = factor_index + 1
            intermediate[f"sovi_factor_{number}_score_unrotated"] = unrotated_scores[
                :, factor_index
            ]
            intermediate[f"sovi_factor_{number}_score"] = rotated_scores[:, factor_index]
            intermediate[f"sovi_factor_{number}_oriented"] = oriented_scores[:, factor_index]

        intermediate["sovi_score_raw"] = score
        intermediate["sovi_score_z"] = score_z
        intermediate["sovi_class"] = sovi_class
        intermediate["score_normalized_0_1"] = score_normalized
        intermediate["percentile"] = percentile.astype("float64")
        intermediate["rank"] = rank
        row_missing_counts = pd.Series(missing_report.row_missing_counts)
        missing_count = handled[spatial_id].map(row_missing_counts).fillna(0).astype(int)
        proxy_count = self._proxy_count()
        intermediate["sovi_missing_count"] = missing_count
        intermediate["sovi_proxy_count"] = proxy_count
        intermediate["sovi_quality_flag"] = [
            self._quality_flag(missing, proxy_count)
            for missing in missing_count
        ]

        standard_output = create_standard_output(
            feature_table=handled,
            recipe=self.recipe,
            run_id=self.run_id,
            score=score,
            score_normalized_0_1=score_normalized,
            percentile=percentile,
            rank=rank,
            classification=sovi_class,
            missing_count=missing_count,
            quality_flag=intermediate["sovi_quality_flag"],
        )

        self.standard_output = standard_output
        self.intermediate_output = intermediate
        self.extra_outputs = self._build_extra_outputs(
            unrotated_scores=unrotated_scores,
            rotated_scores=rotated_scores,
            oriented_scores=oriented_scores,
            standardized=standardized,
        )
        self.process_metadata = self._build_process_metadata(
            score=score,
            score_z_metadata=z_metadata,
        )
        return standard_output

    def _key_for_column(self, column: str) -> str:
        for key, variable in self.recipe.variables.items():
            if variable.canonical_name == column:
                return key
        return column

    def _resolve_factor_orientation(self) -> dict[str, dict[str, Any]]:
        orientation = self.recipe.extra.get("factor_orientation", {})
        factors = orientation.get("factors", {})
        default_behavior = orientation.get("default_unconfigured_behavior", "error")
        self._orientation_warnings: list[str] = []
        metadata: dict[str, dict[str, Any]] = {}
        for factor_index in range(self._n_factors):
            factor_name = f"factor_{factor_index + 1}"
            config = factors.get(factor_name)
            if config is None:
                if default_behavior == "error":
                    raise ValueError(f"Missing SoVI orientation decision for {factor_name}.")
                if default_behavior == "warn":
                    self._orientation_warnings.append(
                        f"Missing SoVI orientation decision for {factor_name}; leaving factor as-is."
                    )
                config = {"method": "none", "rationale": "No orientation configured."}
            method = config.get("method", "none")
            if method not in {"positive", "negative", "absolute", "none", "auto_by_anchor_variable"}:
                raise ValueError(f"Unsupported orientation method '{method}' for {factor_name}.")
            multiplier = 1.0
            if method == "negative":
                multiplier = -1.0
            if method == "auto_by_anchor_variable":
                anchor = config.get("anchor_variable")
                if anchor not in self._used_variable_keys:
                    raise ValueError(
                        f"Orientation anchor variable '{anchor}' is not a retained SoVI variable."
                    )
                anchor_row = self._used_variable_keys.index(anchor)
                multiplier = 1.0 if self._rotated_loadings[anchor_row, factor_index] >= 0 else -1.0
            metadata[factor_name] = {
                "method": method,
                "rationale": config.get("rationale"),
                "configured": factor_name in factors,
                "multiplier": multiplier,
                "anchor_variable": config.get("anchor_variable"),
            }
        return metadata

    def _orient_factor_scores(self, rotated_scores: np.ndarray) -> np.ndarray:
        oriented = np.asarray(rotated_scores, dtype=float).copy()
        for factor_index in range(self._n_factors):
            factor_name = f"factor_{factor_index + 1}"
            config = self._orientation_metadata[factor_name]
            method = config["method"]
            if method == "absolute":
                oriented[:, factor_index] = np.abs(oriented[:, factor_index])
            elif method in {"negative", "auto_by_anchor_variable"}:
                oriented[:, factor_index] = oriented[:, factor_index] * float(config["multiplier"])
            elif method in {"positive", "none"}:
                pass
        return oriented

    def _quality_flag(self, missing_count: int, proxy_count: int) -> str:
        if self.recipe.reproduction_level == "partial_sovi_like":
            return "partial_sovi_like"
        if missing_count > 0:
            return "imputed_missing_values"
        if proxy_count > 0:
            return "proxy_used"
        if self._dropped_variables:
            return "variables_dropped"
        return "ok"

    def _is_proxy_variable(self, key: str) -> bool:
        variable = self.recipe.variables[key]
        proxy_used = variable.extra.get("proxy_used")
        if proxy_used:
            return True
        return False

    def _proxy_count(self) -> int:
        return len([key for key in self._available_variable_keys if self._is_proxy_variable(key)])

    def _proxied_variables(self) -> list[dict[str, Any]]:
        proxied: list[dict[str, Any]] = []
        for key in self._available_variable_keys:
            if self._is_proxy_variable(key):
                variable = self.recipe.variables[key]
                proxied.append(
                    {
                        "variable": key,
                        "canonical_name": variable.canonical_name,
                        "original_variable": variable.extra.get("original_variable"),
                        "proxy_used": variable.extra.get("proxy_used", variable.canonical_name),
                        "proxy_quality": variable.extra.get("proxy_quality"),
                        "conceptual_risk": variable.extra.get("conceptual_risk"),
                        "status": variable.extra.get("status"),
                    }
                )
        return proxied

    def _eigenvalue_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "factor": [f"factor_{i + 1}" for i in range(len(self._pca_result.eigenvalues))],
                "eigenvalue": self._pca_result.eigenvalues,
                "explained_variance_ratio": self._pca_result.explained_variance_ratio,
                "cumulative_explained_variance": self._pca_result.cumulative_explained_variance,
                "retained": [
                    i < self._n_factors for i in range(len(self._pca_result.eigenvalues))
                ],
            }
        )

    def _loadings_table(self, loadings: np.ndarray) -> pd.DataFrame:
        data = {"variable": self._used_variable_keys}
        for factor_index in range(self._n_factors):
            data[f"factor_{factor_index + 1}"] = loadings[:, factor_index]
        return pd.DataFrame(data)

    def _factor_scores_table(
        self,
        *,
        unrotated_scores: np.ndarray,
        rotated_scores: np.ndarray,
        oriented_scores: np.ndarray,
    ) -> pd.DataFrame:
        if self.intermediate_output is None:
            raise RuntimeError("Intermediate output must exist before building factor score table.")
        data = {"zone_id": self.intermediate_output[self.recipe.spatial_id_column].values}
        for factor_index in range(self._n_factors):
            number = factor_index + 1
            data[f"factor_{number}_unrotated"] = unrotated_scores[:, factor_index]
            data[f"factor_{number}_rotated"] = rotated_scores[:, factor_index]
            data[f"factor_{number}_oriented"] = oriented_scores[:, factor_index]
        return pd.DataFrame(data)

    def _factor_summary_table(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for factor_index in range(self._n_factors):
            factor_name = f"factor_{factor_index + 1}"
            loadings = self._rotated_loadings[:, factor_index]
            order = np.argsort(np.abs(loadings))[::-1]
            top = [
                f"{self._used_variable_keys[i]}:{loadings[i]:.4f}"
                for i in order[: min(5, len(order))]
            ]
            positive = [
                f"{self._used_variable_keys[i]}:{loadings[i]:.4f}"
                for i in np.argsort(loadings)[::-1][: min(3, len(order))]
            ]
            negative = [
                f"{self._used_variable_keys[i]}:{loadings[i]:.4f}"
                for i in np.argsort(loadings)[: min(3, len(order))]
            ]
            orientation = self._orientation_metadata[factor_name]
            rows.append(
                {
                    "factor": factor_name,
                    "eigenvalue_before_rotation": self._pca_result.eigenvalues[factor_index],
                    "explained_variance_before_rotation": self._pca_result.explained_variance_ratio[
                        factor_index
                    ],
                    "dominant_variables_by_abs_loading": "; ".join(top),
                    "top_positive_loadings": "; ".join(positive),
                    "top_negative_loadings": "; ".join(negative),
                    "orientation_method": orientation["method"],
                    "orientation_rationale": orientation.get("rationale"),
                    "orientation_configured": orientation["configured"],
                }
            )
        return pd.DataFrame(rows)

    def _build_extra_outputs(
        self,
        *,
        unrotated_scores: np.ndarray,
        rotated_scores: np.ndarray,
        oriented_scores: np.ndarray,
        standardized: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        return {
            "sovi_eigenvalues": self._eigenvalue_table(),
            "sovi_explained_variance": self._eigenvalue_table(),
            "sovi_loadings_unrotated": self._loadings_table(self._unrotated_loadings),
            "sovi_loadings_rotated": self._loadings_table(self._rotated_loadings),
            "sovi_factor_scores": self._factor_scores_table(
                unrotated_scores=unrotated_scores,
                rotated_scores=rotated_scores,
                oriented_scores=oriented_scores,
            ),
            "sovi_factor_summary": self._factor_summary_table(),
            "sovi_standardized_variables": standardized.reset_index(drop=True),
        }

    def _build_process_metadata(
        self,
        *,
        score: pd.Series,
        score_z_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        warnings: list[str] = [
            "Factor signs are arbitrary and require orientation decisions.",
            SOVI_AREA_LEVEL_WARNING,
        ]
        if self.recipe.reproduction_level != "strict_original_like":
            warnings.append("This run is a SoVI-like local/partial adaptation, not exact original SoVI.")
        if self.recipe.missing_data.get("strategy") == "zero_imputation":
            warnings.append(
                "Zero imputation was used; this may distort factor structure when zero is not substantively meaningful."
            )
        if self._dropped_variables:
            warnings.append(
                "Constant variables were dropped before PCA: " + ", ".join(self._dropped_variables)
            )
        if self._rotation_metadata.get("method") == "none":
            warnings.append("Varimax rotation was skipped by explicit recipe configuration.")
        warnings.extend(getattr(self, "_orientation_warnings", []))

        eigen_table = self._eigenvalue_table()
        return {
            "normalization": {
                key: {
                    "method": "zscore",
                    "canonical_name": self.recipe.variables[key].canonical_name,
                    **self._standardization_metadata["variables"].get(
                        self.recipe.variables[key].canonical_name, {}
                    ),
                }
                for key in self._available_variable_keys
            },
            "orientation": self._orientation_metadata,
            "weighting_method": "equal additive factor scores",
            "aggregation": {
                "method": "additive_factor_sum",
                "weights": "equal",
                "score_column": "sovi_score_raw",
            },
            "classification": {
                "method": self.recipe.classification.get("method"),
                "score_z": score_z_metadata,
            },
            "standardization": self._standardization_metadata,
            "factor_analysis": {
                **self._pca_result.metadata,
                "method": self.recipe.extra.get("factor_analysis", {}).get("method", "pca"),
                "eigenvalues": self._pca_result.eigenvalues.tolist(),
                "explained_variance_ratio": self._pca_result.explained_variance_ratio.tolist(),
                "cumulative_explained_variance": self._pca_result.cumulative_explained_variance.tolist(),
            },
            "factor_retention": self._retention_metadata,
            "rotation": self._rotation_metadata,
            "factor_scores": self._factor_score_metadata,
            "variables_used": self._used_variable_keys,
            "variables_dropped": self._dropped_variables,
            "variables_proxied": self._proxied_variables(),
            "source_reference": self.recipe.extra.get("source_reference", self.recipe.method_reference),
            "methodology_note": SOVI_METHOD_NOTE,
            "sovi": {
                "methodology_note": SOVI_METHOD_NOTE,
                "area_level_warning": SOVI_AREA_LEVEL_WARNING,
                "n_factors_retained": self._n_factors,
                "retained_cumulative_explained_variance": float(
                    eigen_table.loc[eigen_table["retained"], "explained_variance_ratio"].sum()
                ),
                "score_summary": {
                    "min": float(score.min(skipna=True)),
                    "mean": float(score.mean(skipna=True)),
                    "max": float(score.max(skipna=True)),
                    "std": float(score.std(skipna=True, ddof=0)),
                },
                "dominant_variables_by_factor": self._factor_summary_table().to_dict("records"),
            },
            "warnings": warnings,
        }

    def explain(self, zone_id: Any) -> dict[str, Any]:
        if self.intermediate_output is None or self.standard_output is None:
            raise RuntimeError("No SoVI output is available. Run transform first.")
        spatial_id = self.recipe.spatial_id_column
        matches = self.intermediate_output[self.intermediate_output[spatial_id] == zone_id]
        standard_matches = self.standard_output[self.standard_output["zone_id"] == zone_id]
        if matches.empty or standard_matches.empty:
            raise KeyError(f"Zone '{zone_id}' is not present in SoVI output.")
        row = matches.iloc[0]
        standard_row = standard_matches.iloc[0]
        oriented = {
            f"factor_{i + 1}": float(row[f"sovi_factor_{i + 1}_oriented"])
            for i in range(self._n_factors)
        }
        factor_scores = {
            f"factor_{i + 1}": float(row[f"sovi_factor_{i + 1}_score"])
            for i in range(self._n_factors)
        }
        top_positive = sorted(oriented.items(), key=lambda item: item[1], reverse=True)[:3]
        top_negative = sorted(oriented.items(), key=lambda item: item[1])[:3]
        standardized_values = {
            key: float(row[f"sovi_z_{key}"])
            for key in self._used_variable_keys
            if f"sovi_z_{key}" in row.index
        }
        largest_standardized = sorted(
            standardized_values.items(), key=lambda item: abs(item[1]), reverse=True
        )[:5]
        return {
            "zone_id": zone_id,
            "sovi_score_raw": float(row["sovi_score_raw"]),
            "sovi_score_z": float(row["sovi_score_z"]),
            "sovi_class": row["sovi_class"],
            "rank": int(standard_row["rank"]),
            "percentile": float(standard_row["percentile"]),
            "factor_scores": factor_scores,
            "oriented_factor_scores": oriented,
            "largest_positive_factor_contributors": dict(top_positive),
            "largest_negative_or_protective_factor_contributors": dict(top_negative),
            "variables_with_highest_abs_standardized_values": dict(largest_standardized),
            "missing_variables": [],
            "proxy_variables": self._proxied_variables(),
            "quality_flag": row["sovi_quality_flag"],
            "interpretation": (
                "This is an area-level SoVI-like score. Higher values indicate higher "
                "relative social vulnerability for the spatial unit, not a deterministic "
                "claim about every individual in that area."
            ),
        }
