#!/usr/bin/env python3
"""
B3 tabular feature-parity baseline for the Québec CD civil-security / SoVI benchmark.

Purpose:
    Run non-graph ML using the same non-graph node features that a later CD graph
    model can receive:
        - SoVI score and numeric SoVI/static features
        - current/history event features
        - hazard-specific lag/rolling features when present
        - month/year seasonality

This is the crucial feature-parity baseline. Any later graph model should be
compared against B3 to test whether topology adds value beyond the same node
features.

Default input:
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/datasets/cd_month_panel.parquet

Default output directory:
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B3_tabular_feature_parity/

Models:
    - ridge
    - random_forest
    - hist_gradient_boosting, if scikit-learn provides it

Public API:
    run_b3_cd_tabular_feature_parity(config: Config) -> dict[str, Any]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ville_hgnn.baselines.qc_cd_sovi_common import (
    BASELINES_DIR,
    CD_ID_COL,
    CD_NAME_COL,
    DEFAULT_PANEL_PATH,
    SOVI_SCORE_COL,
    SPLIT_COL,
    ensure_dir,
    evaluate_predictions_by_split,
    evaluate_standard_prediction_frame,
    write_metadata_json,
    write_table,
)


DEFAULT_OUTPUT_DIR = BASELINES_DIR / "B3_tabular_feature_parity"

DEFAULT_MODELS = [
    "ridge",
    "random_forest",
    "hist_gradient_boosting",
]

RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]

RANDOM_FOREST_GRID = [
    {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 5},
    {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 10},
    {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 20},
]

HIST_GRADIENT_BOOSTING_GRID = [
    {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 0.0},
    {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0},
    {"max_iter": 100, "learning_rate": 0.10, "max_leaf_nodes": 15, "l2_regularization": 1.0},
]


@dataclass
class Config:
    """Configuration for B3 tabular feature-parity ML."""

    panel_path: Path = DEFAULT_PANEL_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR

    target_col: str = "target_next_3_months"

    cd_id_col: str = CD_ID_COL
    cd_name_col: str = CD_NAME_COL
    period_month_col: str = "period_month"
    split_col: str = SPLIT_COL

    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    ridge_alphas: list[float] = field(default_factory=lambda: list(RIDGE_ALPHAS))

    selection_metric: str = "mae"
    random_seed: int = 42

    # Feature controls.
    include_sovi_features: bool = True
    include_history_features: bool = True
    include_hazard_history_features: bool = True
    include_current_month_counts: bool = True
    include_seasonality: bool = True
    include_year_trend: bool = True

    # Use all eligible numeric columns after leakage-safe exclusions. Keeping
    # this True makes B3 a strong feature-parity baseline when new graph node
    # features are added later.
    include_all_other_numeric_features: bool = True

    # Missing targets come from incomplete future windows at the panel boundary.
    drop_missing_target: bool = True

    # Count predictions should be nonnegative.
    clip_predictions_at_zero: bool = True

    # Conservative defaults for tree models.
    random_forest_grid: list[dict[str, Any]] = field(
        default_factory=lambda: [dict(x) for x in RANDOM_FOREST_GRID]
    )
    hist_gradient_boosting_grid: list[dict[str, Any]] = field(
        default_factory=lambda: [dict(x) for x in HIST_GRADIENT_BOOSTING_GRID]
    )

    n_jobs: int = -1


def _lazy_import_sklearn() -> dict[str, Any]:
    """Import sklearn components lazily."""
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "B3 tabular feature-parity requires scikit-learn. Install scikit-learn "
            "in the active environment or run inside the project .venv."
        ) from exc

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except Exception:
        HistGradientBoostingRegressor = None

    return {
        "ColumnTransformer": ColumnTransformer,
        "RandomForestRegressor": RandomForestRegressor,
        "SimpleImputer": SimpleImputer,
        "Ridge": Ridge,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
    }


def read_panel(path: Path) -> pd.DataFrame:
    """Read the predictive panel."""
    if not path.exists():
        raise FileNotFoundError(f"Panel file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported panel file suffix: {path}")


def validate_config(config: Config) -> None:
    """Validate B3 configuration."""
    if not isinstance(config.panel_path, Path):
        config.panel_path = Path(config.panel_path)
    if not isinstance(config.output_dir, Path):
        config.output_dir = Path(config.output_dir)

    if not config.models:
        raise ValueError("At least one B3 model must be requested.")

    unknown = sorted(set(config.models) - set(DEFAULT_MODELS))
    if unknown:
        raise ValueError(f"Unknown B3 models: {unknown}. Allowed models: {DEFAULT_MODELS}")

    if not config.ridge_alphas and "ridge" in config.models:
        raise ValueError("At least one ridge alpha must be supplied.")

    valid_selection_metrics = {
        "mae",
        "rmse",
        "mean_poisson_deviance",
        "spearman",
        "ndcg_at_25",
    }
    if config.selection_metric not in valid_selection_metrics:
        raise ValueError(
            f"Invalid selection metric: {config.selection_metric}. "
            f"Allowed: {sorted(valid_selection_metrics)}"
        )


def validate_panel_columns(df: pd.DataFrame, config: Config) -> None:
    """Validate required panel columns."""
    required = [
        config.cd_id_col,
        config.target_col,
        config.split_col,
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            "The B3 panel is missing required columns: "
            f"{missing}. Available columns: {list(df.columns)}"
        )


def split_masks(df: pd.DataFrame, config: Config) -> dict[str, pd.Series]:
    split = df[config.split_col].astype("string")
    return {
        "train": split.eq("train"),
        "validation": split.isin(["val", "validation"]),
        "test": split.eq("test"),
        "all": pd.Series(True, index=df.index),
    }


def add_seasonality_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add leakage-safe origin-month seasonality/trend features."""
    out = df.copy()

    if config.include_seasonality:
        if "month" in out.columns:
            month = pd.to_numeric(out["month"], errors="coerce")
        elif config.period_month_col in out.columns:
            parsed = pd.to_datetime(out[config.period_month_col].astype("string"), errors="coerce")
            month = parsed.dt.month
        else:
            month = pd.Series(np.nan, index=out.index)

        radians = 2.0 * np.pi * (month - 1.0) / 12.0
        out["season_month_sin"] = np.sin(radians)
        out["season_month_cos"] = np.cos(radians)

    if config.include_year_trend:
        if "year" in out.columns:
            year = pd.to_numeric(out["year"], errors="coerce")
        elif config.period_month_col in out.columns:
            parsed = pd.to_datetime(out[config.period_month_col].astype("string"), errors="coerce")
            year = parsed.dt.year
        else:
            year = pd.Series(np.nan, index=out.index)

        out["origin_year_centered"] = year - year.dropna().min() if year.notna().any() else np.nan

    return out


def looks_like_target_or_future(col: str) -> bool:
    """Detect leakage-prone target/future/completeness columns."""
    lower = col.lower()

    if lower.startswith("target_"):
        return True
    if lower.startswith("prediction") or lower.startswith("pred_"):
        return True
    if lower.endswith("_complete"):
        return True
    if "future" in lower:
        return True

    return False


def base_excluded_columns(config: Config) -> set[str]:
    """Columns never used as B3 features."""
    return {
        config.target_col,
        config.cd_id_col,
        config.cd_name_col,
        config.period_month_col,
        config.split_col,
        "model_name",
        "candidate_name",
        "target_col",
        "target",
        "prediction",
        "features",
        "alpha",
        "_original_cd_id_norm_for_targets",
        "recommended_primary_target",
        "recommended_primary_target_name",
        "recommended_primary_target_granularity",
        "recommended_primary_precision_filter",
    }


def feature_category(col: str, config: Config) -> str | None:
    """
    Categorize a candidate feature column.

    Returning None excludes the column.
    """
    lower = col.lower()

    if col in base_excluded_columns(config):
        return None
    if looks_like_target_or_future(col):
        return None

    # Explicit seasonality features created in this module.
    if lower in {"season_month_sin", "season_month_cos", "origin_year_centered"}:
        return "seasonality"

    # Avoid raw identifiers even if numeric/coercible.
    if lower in {"zone_id", "run_id"}:
        return None
    if lower.endswith("_id") or lower.endswith("_dguid"):
        return None
    if "dguid" in lower:
        return None

    # SoVI score and common SoVI metadata/features.
    if config.include_sovi_features and (
        lower in {
            "score_raw",
            "score_normalized_0_1",
            "score_normalized",
            "sovi_score",
            "rank",
            "percentile",
            "missing_count",
        }
        or "sovi" in lower
    ):
        return "sovi"

    # Current-month event count features available at origin t.
    if lower.startswith("event_count_current_month"):
        if not config.include_current_month_counts:
            return None
        # Hazard-specific current count features are useful graph-node features.
        if any(hazard in lower for hazard in HAZARD_KEYWORDS):
            return "hazard_current"
        return "history_current"

    # Generic history features.
    if config.include_history_features and (
        lower in {"lag_1", "rolling_3", "rolling_6", "rolling_12"}
        or lower.endswith("_lag_1")
        or "_rolling_" in lower
    ):
        if any(hazard in lower for hazard in HAZARD_KEYWORDS):
            return "hazard_history" if config.include_hazard_history_features else None
        return "history"

    # Calendar raw columns are excluded; we use sin/cos/year-centered instead.
    if lower in {"month", "year"}:
        return None

    if config.include_all_other_numeric_features:
        return "other_numeric"

    return None


HAZARD_KEYWORDS = [
    "flood_water",
    "land_ground",
    "weather_climate",
    "infrastructure",
    "wildfire",
    "hazmat_health_social",
    "transport_accident",
    "other",
    "unmapped",
    "moderate_or_worse",
    "important_or_extreme",
    "possible_duplicate",
]


def infer_feature_columns(
    df: pd.DataFrame,
    config: Config,
) -> tuple[list[str], pd.DataFrame]:
    """
    Infer leakage-safe numeric feature columns and return a feature audit table.
    """
    feature_rows: list[dict[str, Any]] = []
    feature_cols: list[str] = []

    for col in df.columns:
        category = feature_category(col, config)

        numeric = pd.to_numeric(df[col], errors="coerce")
        nonmissing = int(numeric.notna().sum())
        unique = int(numeric.nunique(dropna=True))

        use = category is not None and nonmissing > 0 and unique > 1

        feature_rows.append(
            {
                "column": col,
                "category": category if category is not None else "excluded",
                "use_as_feature": bool(use),
                "nonmissing": nonmissing,
                "missing": int(len(df) - nonmissing),
                "unique": unique,
                "min": float(numeric.min()) if nonmissing else np.nan,
                "max": float(numeric.max()) if nonmissing else np.nan,
            }
        )

        if use:
            feature_cols.append(col)

    if not feature_cols:
        raise RuntimeError("No usable B3 feature columns were inferred.")

    audit = pd.DataFrame(feature_rows)
    return feature_cols, audit


def make_design_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Build numeric feature matrix."""
    X = pd.DataFrame(index=df.index)
    for col in feature_cols:
        X[col] = pd.to_numeric(df[col], errors="coerce")
    return X


def make_candidate_specs(config: Config) -> list[dict[str, Any]]:
    """Build conservative model candidate specs."""
    specs: list[dict[str, Any]] = []

    if "ridge" in config.models:
        for alpha in config.ridge_alphas:
            specs.append(
                {
                    "model_name": "ridge",
                    "candidate_name": f"ridge__alpha_{str(alpha).replace('.', 'p')}",
                    "params": {"alpha": float(alpha)},
                }
            )

    if "random_forest" in config.models:
        for idx, params in enumerate(config.random_forest_grid, start=1):
            name = (
                "random_forest"
                f"__d{params.get('max_depth', 'none')}"
                f"_leaf{params.get('min_samples_leaf', 'na')}"
                f"_n{params.get('n_estimators', 'na')}"
            )
            specs.append(
                {
                    "model_name": "random_forest",
                    "candidate_name": name,
                    "params": dict(params),
                    "grid_index": idx,
                }
            )

    if "hist_gradient_boosting" in config.models:
        for idx, params in enumerate(config.hist_gradient_boosting_grid, start=1):
            name = (
                "hist_gradient_boosting"
                f"__iter{params.get('max_iter', 'na')}"
                f"_lr{str(params.get('learning_rate', 'na')).replace('.', 'p')}"
                f"_leaf{params.get('max_leaf_nodes', 'na')}"
                f"_l2{str(params.get('l2_regularization', 'na')).replace('.', 'p')}"
            )
            specs.append(
                {
                    "model_name": "hist_gradient_boosting",
                    "candidate_name": name,
                    "params": dict(params),
                    "grid_index": idx,
                }
            )

    return specs


def make_estimator(spec: Mapping[str, Any], config: Config) -> Any:
    """Create a sklearn Pipeline for a candidate spec."""
    sk = _lazy_import_sklearn()
    Pipeline = sk["Pipeline"]
    SimpleImputer = sk["SimpleImputer"]

    model_name = spec["model_name"]
    params = dict(spec.get("params", {}))

    if model_name == "ridge":
        Ridge = sk["Ridge"]
        StandardScaler = sk["StandardScaler"]
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    Ridge(
                        alpha=float(params["alpha"]),
                        random_state=config.random_seed,
                    ),
                ),
            ]
        )

    if model_name == "random_forest":
        RandomForestRegressor = sk["RandomForestRegressor"]
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=int(params.get("n_estimators", 300)),
                        max_depth=params.get("max_depth"),
                        min_samples_leaf=int(params.get("min_samples_leaf", 5)),
                        random_state=config.random_seed,
                        n_jobs=config.n_jobs,
                    ),
                ),
            ]
        )

    if model_name == "hist_gradient_boosting":
        HistGradientBoostingRegressor = sk["HistGradientBoostingRegressor"]
        if HistGradientBoostingRegressor is None:
            raise ImportError("HistGradientBoostingRegressor is not available in this scikit-learn install.")

        # Use squared_error instead of poisson here to keep this baseline robust
        # across scikit-learn versions and target edge cases. Predictions are
        # clipped to nonnegative counts after inference.
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        loss="squared_error",
                        max_iter=int(params.get("max_iter", 150)),
                        learning_rate=float(params.get("learning_rate", 0.05)),
                        max_leaf_nodes=int(params.get("max_leaf_nodes", 15)),
                        l2_regularization=float(params.get("l2_regularization", 0.0)),
                        random_state=config.random_seed,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown B3 model name: {model_name}")


def valid_training_mask(
    df: pd.DataFrame,
    X: pd.DataFrame,
    config: Config,
) -> pd.Series:
    """Rows usable for supervised training."""
    masks = split_masks(df, config)
    y = pd.to_numeric(df[config.target_col], errors="coerce")
    has_any_feature = X.notna().any(axis=1)

    return masks["train"] & y.notna() & has_any_feature


def fit_predict_candidate(
    df: pd.DataFrame,
    X: pd.DataFrame,
    *,
    spec: Mapping[str, Any],
    config: Config,
) -> tuple[pd.Series, Any]:
    """Fit one B3 candidate on train rows and predict all rows."""
    y = pd.to_numeric(df[config.target_col], errors="coerce")
    train_mask = valid_training_mask(df, X, config)

    if int(train_mask.sum()) < 10:
        raise ValueError(
            f"Not enough training rows for candidate {spec['candidate_name']}. "
            f"Usable train rows: {int(train_mask.sum())}"
        )

    estimator = make_estimator(spec, config)
    estimator.fit(X.loc[train_mask], y.loc[train_mask].astype(float))

    has_any_feature = X.notna().any(axis=1)
    pred = pd.Series(np.nan, index=df.index, dtype=float)
    pred.loc[has_any_feature] = estimator.predict(X.loc[has_any_feature])

    if config.clip_predictions_at_zero:
        pred = pred.clip(lower=0)

    return pred, estimator


def build_prediction_frame(
    df: pd.DataFrame,
    *,
    prediction: pd.Series,
    spec: Mapping[str, Any],
    feature_cols: list[str],
    config: Config,
) -> pd.DataFrame:
    """Build standardized prediction table for one candidate."""
    out = pd.DataFrame(index=df.index)

    out[CD_ID_COL] = df[config.cd_id_col].astype("string")

    if config.cd_name_col in df.columns:
        out[CD_NAME_COL] = df[config.cd_name_col].astype("string")

    if config.period_month_col in df.columns:
        out["period_month"] = df[config.period_month_col].astype("string")
    if "year" in df.columns:
        out["year"] = df["year"]
    if "month" in df.columns:
        out["month"] = df["month"]

    out[SPLIT_COL] = df[config.split_col].astype("string")
    out["model_name"] = str(spec["model_name"])
    out["candidate_name"] = str(spec["candidate_name"])
    out["target_col"] = config.target_col
    out["feature_count"] = len(feature_cols)
    out["features_hashable"] = ",".join(feature_cols)
    out["target"] = pd.to_numeric(df[config.target_col], errors="coerce")
    out["prediction"] = pd.to_numeric(prediction, errors="coerce")

    if SOVI_SCORE_COL in df.columns:
        out["sovi_score"] = pd.to_numeric(df[SOVI_SCORE_COL], errors="coerce")

    return out.reset_index(drop=True)


def evaluate_candidate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all candidate predictions by candidate and split."""
    rows = []

    for (model_name, candidate_name), sub in predictions.groupby(
        ["model_name", "candidate_name"], dropna=False
    ):
        metrics = evaluate_standard_prediction_frame(
            sub,
            target_col="target",
            prediction_col="prediction",
            split_col=SPLIT_COL,
            id_col=CD_ID_COL,
        )
        metrics["model_name"] = model_name
        metrics["candidate_name"] = candidate_name
        if "feature_count" in sub.columns:
            metrics["feature_count"] = int(sub["feature_count"].iloc[0])
        rows.append(metrics)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def select_best_candidates(
    candidate_metrics: pd.DataFrame,
    *,
    selection_metric: str,
) -> pd.DataFrame:
    """
    Select one candidate per model family using validation metric.
    """
    if candidate_metrics.empty:
        return pd.DataFrame()

    validation = candidate_metrics[
        candidate_metrics[SPLIT_COL].astype("string").isin(["val", "validation"])
    ].copy()

    if validation.empty:
        validation = candidate_metrics[
            candidate_metrics[SPLIT_COL].astype("string").eq("all")
        ].copy()

    if validation.empty:
        return pd.DataFrame()

    if selection_metric not in validation.columns:
        raise KeyError(
            f"Selection metric '{selection_metric}' not found in metrics. "
            f"Available columns: {list(validation.columns)}"
        )

    lower_is_better = selection_metric in {"mae", "rmse", "mean_poisson_deviance"}

    selected_rows = []
    for model_name, sub in validation.groupby("model_name", dropna=False):
        sort_cols = [selection_metric]
        ascending = [lower_is_better]

        for col, asc in [
            ("mae", True),
            ("rmse", True),
            ("mean_poisson_deviance", True),
            ("spearman", False),
            ("ndcg_at_25", False),
            ("candidate_name", True),
        ]:
            if col in sub.columns and col not in sort_cols:
                sort_cols.append(col)
                ascending.append(asc)

        best = sub.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()
        best["selection_rule"] = f"per_model_best_by_validation_{selection_metric}"
        selected_rows.append(best)

    selected = pd.DataFrame(selected_rows)

    selected = selected.sort_values(
        [selection_metric, "model_name"],
        ascending=[lower_is_better, True],
    ).reset_index(drop=True)
    selected.insert(0, "selection_rank_within_family", np.arange(1, len(selected) + 1))
    return selected


def filter_selected_predictions(
    candidate_predictions: pd.DataFrame,
    selected_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only the selected candidate for each model family."""
    if candidate_predictions.empty or selected_candidates.empty:
        return pd.DataFrame()

    selected_keys = set(
        zip(
            selected_candidates["model_name"].astype(str),
            selected_candidates["candidate_name"].astype(str),
        )
    )

    keep = [
        (str(model), str(candidate)) in selected_keys
        for model, candidate in zip(
            candidate_predictions["model_name"],
            candidate_predictions["candidate_name"],
        )
    ]

    return candidate_predictions.loc[keep].reset_index(drop=True)


def extract_feature_importance(
    fitted_estimators: dict[str, Any],
    *,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Extract model-native feature importances/coefs when available.

    Notes:
        - Ridge coefficients are coefficients after median imputation and scaling.
        - Random forest importances are model-native impurity importances.
        - HistGradientBoostingRegressor does not expose stable feature_importances_
          in many sklearn versions, so it may not produce rows here.
    """
    rows: list[dict[str, Any]] = []

    for candidate_name, estimator in fitted_estimators.items():
        try:
            model = estimator.named_steps["model"]
        except Exception:
            continue

        model_class = type(model).__name__

        if hasattr(model, "coef_"):
            values = np.ravel(model.coef_).astype(float)
            for feature, value in zip(feature_cols, values):
                rows.append(
                    {
                        "candidate_name": candidate_name,
                        "model_class": model_class,
                        "importance_type": "standardized_coefficient",
                        "feature": feature,
                        "value": float(value),
                        "abs_value": float(abs(value)),
                    }
                )

        if hasattr(model, "feature_importances_"):
            values = np.ravel(model.feature_importances_).astype(float)
            for feature, value in zip(feature_cols, values):
                rows.append(
                    {
                        "candidate_name": candidate_name,
                        "model_class": model_class,
                        "importance_type": "model_feature_importance",
                        "feature": feature,
                        "value": float(value),
                        "abs_value": float(abs(value)),
                    }
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["candidate_name", "importance_type", "abs_value"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
    return out


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compact audit summary by model and split."""
    if predictions.empty:
        return pd.DataFrame()

    group_cols = ["model_name"]
    if SPLIT_COL in predictions.columns:
        group_cols.append(SPLIT_COL)

    rows = []
    for key, sub in predictions.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        row = {col: value for col, value in zip(group_cols, key)}

        target = pd.to_numeric(sub["target"], errors="coerce")
        pred = pd.to_numeric(sub["prediction"], errors="coerce")

        row.update(
            {
                "n": int(len(sub)),
                "target_nonmissing": int(target.notna().sum()),
                "prediction_nonmissing": int(pred.notna().sum()),
                "target_mean": float(target.mean()) if target.notna().any() else np.nan,
                "prediction_mean": float(pred.mean()) if pred.notna().any() else np.nan,
                "target_sum": float(target.sum()) if target.notna().any() else 0.0,
                "prediction_sum": float(pred.sum()) if pred.notna().any() else 0.0,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def write_split_predictions(predictions: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    """Write split-specific prediction files."""
    outputs: dict[str, str] = {}

    if SPLIT_COL not in predictions.columns:
        return outputs

    for split_name in ["train", "val", "validation", "test"]:
        sub = predictions[predictions[SPLIT_COL].astype("string").eq(split_name)]
        if sub.empty:
            continue

        split_label = "validation" if split_name == "val" else split_name
        split_path = output_dir / f"predictions_{split_label}.parquet"
        written = write_table(sub, split_path, write_csv_copy=True, index=False)
        for kind, path in written.items():
            outputs[f"predictions_{split_label}_{kind}"] = path

    return outputs


def write_failure_log(failures: list[dict[str, Any]], output_dir: Path) -> Path:
    """Write candidate failures, if any."""
    path = output_dir / "candidate_failures.csv"
    pd.DataFrame(failures).to_csv(path, index=False)
    return path


def run_b3_cd_tabular_feature_parity(config: Config) -> dict[str, Any]:
    """
    Run B3 tabular feature-parity ML and write standardized outputs.
    """
    validate_config(config)
    output_dir = ensure_dir(config.output_dir)

    panel = read_panel(config.panel_path)
    validate_panel_columns(panel, config)

    panel = add_seasonality_features(panel, config)

    sort_cols = [config.cd_id_col]
    if config.period_month_col in panel.columns:
        sort_cols.append(config.period_month_col)
    panel = panel.sort_values(sort_cols).reset_index(drop=True)

    feature_cols, feature_audit = infer_feature_columns(panel, config)
    X = make_design_matrix(panel, feature_cols)

    specs = make_candidate_specs(config)

    candidate_predictions_frames: list[pd.DataFrame] = []
    fitted_estimators: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []

    for spec in specs:
        try:
            pred, estimator = fit_predict_candidate(panel, X, spec=spec, config=config)
            fitted_estimators[str(spec["candidate_name"])] = estimator

            frame = build_prediction_frame(
                panel,
                prediction=pred,
                spec=spec,
                feature_cols=feature_cols,
                config=config,
            )
            candidate_predictions_frames.append(frame)
        except Exception as exc:
            failures.append(
                {
                    "model_name": spec.get("model_name"),
                    "candidate_name": spec.get("candidate_name"),
                    "params": json.dumps(spec.get("params", {}), sort_keys=True),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    if not candidate_predictions_frames:
        failure_text = "\n".join(
            f"{f['candidate_name']}: {f['error_type']} - {f['error_message']}"
            for f in failures
        )
        raise RuntimeError(f"No B3 candidates completed successfully.\n{failure_text}")

    candidate_predictions = pd.concat(candidate_predictions_frames, ignore_index=True)

    if config.drop_missing_target:
        candidate_predictions = candidate_predictions[
            candidate_predictions["target"].notna()
        ].copy()

    candidate_metrics = evaluate_candidate_predictions(candidate_predictions)
    model_selection = select_best_candidates(
        candidate_metrics,
        selection_metric=config.selection_metric,
    )

    predictions = filter_selected_predictions(candidate_predictions, model_selection)

    metrics = evaluate_standard_prediction_frame(
        predictions,
        target_col="target",
        prediction_col="prediction",
        split_col=SPLIT_COL,
        id_col=CD_ID_COL,
    )

    prediction_summary = summarize_predictions(predictions)
    feature_importance = extract_feature_importance(fitted_estimators, feature_cols=feature_cols)

    prediction_paths = write_table(
        predictions,
        output_dir / "predictions.parquet",
        write_csv_copy=True,
        index=False,
    )

    candidate_prediction_paths = write_table(
        candidate_predictions,
        output_dir / "candidate_predictions.parquet",
        write_csv_copy=True,
        index=False,
    )

    split_outputs = write_split_predictions(predictions, output_dir)

    metrics_path = output_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    candidate_metrics_path = output_dir / "candidate_metrics.csv"
    candidate_metrics.to_csv(candidate_metrics_path, index=False)

    model_selection_path = output_dir / "model_selection.csv"
    model_selection.to_csv(model_selection_path, index=False)

    prediction_summary_path = output_dir / "prediction_summary.csv"
    prediction_summary.to_csv(prediction_summary_path, index=False)

    feature_audit_path = output_dir / "feature_columns.csv"
    feature_audit.to_csv(feature_audit_path, index=False)

    feature_importance_path = output_dir / "feature_importance.csv"
    feature_importance.to_csv(feature_importance_path, index=False)

    failure_path = write_failure_log(failures, output_dir)

    feature_category_summary = (
        feature_audit[feature_audit["use_as_feature"]]
        .groupby("category", dropna=False)
        .size()
        .rename("feature_count")
        .reset_index()
        .sort_values("feature_count", ascending=False)
        .to_dict(orient="records")
    )

    metadata = {
        "baseline_family": "B3_tabular_feature_parity",
        "module": "ville_hgnn.baselines.b3_cd_tabular_feature_parity",
        "purpose": (
            "Non-graph ML using the same leakage-safe node features later "
            "available to graph models: SoVI/static features, event history, "
            "hazard-specific history, and seasonality."
        ),
        "config": {
            **asdict(config),
            "panel_path": str(config.panel_path),
            "output_dir": str(config.output_dir),
        },
        "inputs": {
            "panel_path": str(config.panel_path),
            "panel_rows": int(len(panel)),
            "panel_columns": int(panel.shape[1]),
            "panel_cd_count": int(panel[config.cd_id_col].nunique()),
        },
        "target_col": config.target_col,
        "feature_count": int(len(feature_cols)),
        "feature_category_summary": feature_category_summary,
        "models_requested": list(config.models),
        "candidate_count_requested": int(len(specs)),
        "candidate_count_completed": int(candidate_predictions["candidate_name"].nunique()),
        "candidate_count_failed": int(len(failures)),
        "prediction_rows": int(len(predictions)),
        "outputs": {
            **{f"predictions_{k}": v for k, v in prediction_paths.items()},
            **{f"candidate_predictions_{k}": v for k, v in candidate_prediction_paths.items()},
            **split_outputs,
            "metrics_csv": str(metrics_path),
            "candidate_metrics_csv": str(candidate_metrics_path),
            "model_selection_csv": str(model_selection_path),
            "prediction_summary_csv": str(prediction_summary_path),
            "feature_columns_csv": str(feature_audit_path),
            "feature_importance_csv": str(feature_importance_path),
            "candidate_failures_csv": str(failure_path),
        },
    }

    if not model_selection.empty:
        best = model_selection.iloc[0].to_dict()
        metadata["selected_model"] = {
            "model_name": best.get("model_name"),
            "candidate_name": best.get("candidate_name"),
            "selection_split": best.get(SPLIT_COL),
            "selection_metric": config.selection_metric,
            "mae": best.get("mae"),
            "rmse": best.get("rmse"),
            "mean_poisson_deviance": best.get("mean_poisson_deviance"),
            "spearman": best.get("spearman"),
            "ndcg_at_25": best.get("ndcg_at_25"),
        }

    metadata_path = write_metadata_json(metadata, output_dir / "metadata.json")
    metadata["outputs"]["metadata_json"] = str(metadata_path)

    return metadata


def parse_float_list(values: Sequence[str] | None, default: Sequence[float]) -> list[float]:
    """Parse CLI float lists."""
    if not values:
        return [float(v) for v in default]
    return [float(v) for v in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run B3 tabular feature-parity ML for the Québec CD civil-security / "
            "SoVI benchmark."
        )
    )
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=DEFAULT_PANEL_PATH,
        help="Predictive CD-month panel created by script 11.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for B3 tabular feature-parity results.",
    )
    parser.add_argument(
        "--target-col",
        default="target_next_3_months",
        help="Target column to forecast.",
    )
    parser.add_argument(
        "--cd-id-col",
        default=CD_ID_COL,
        help="CD ID column in the panel.",
    )
    parser.add_argument(
        "--cd-name-col",
        default=CD_NAME_COL,
        help="CD name column in the panel.",
    )
    parser.add_argument(
        "--period-month-col",
        default="period_month",
        help="Panel period-month column.",
    )
    parser.add_argument(
        "--split-col",
        default=SPLIT_COL,
        help="Train/validation/test split column.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=list(DEFAULT_MODELS),
        help="B3 model families to run.",
    )
    parser.add_argument(
        "--ridge-alphas",
        nargs="+",
        default=None,
        help="Ridge alpha values. Default: 0.1 1.0 10.0 100.0.",
    )
    parser.add_argument(
        "--selection-metric",
        default="mae",
        choices=["mae", "rmse", "mean_poisson_deviance", "spearman", "ndcg_at_25"],
        help="Validation metric used to select one candidate per model.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of jobs for random forest.",
    )
    parser.add_argument(
        "--keep-missing-targets",
        action="store_true",
        help="Keep rows with missing targets in prediction outputs. Metrics ignore missing targets.",
    )
    parser.add_argument(
        "--no-clip-predictions",
        action="store_true",
        help="Do not clip predictions at zero.",
    )
    parser.add_argument(
        "--no-sovi-features",
        action="store_true",
        help="Exclude SoVI score/static features.",
    )
    parser.add_argument(
        "--no-history-features",
        action="store_true",
        help="Exclude generic history features.",
    )
    parser.add_argument(
        "--no-hazard-history-features",
        action="store_true",
        help="Exclude hazard-specific history features.",
    )
    parser.add_argument(
        "--no-current-month-counts",
        action="store_true",
        help="Exclude current-month event count features.",
    )
    parser.add_argument(
        "--no-seasonality",
        action="store_true",
        help="Exclude month sin/cos features.",
    )
    parser.add_argument(
        "--no-year-trend",
        action="store_true",
        help="Exclude origin year trend feature.",
    )
    parser.add_argument(
        "--no-other-numeric-features",
        action="store_true",
        help="Exclude otherwise eligible numeric features.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        panel_path=args.panel_path,
        output_dir=args.output_dir,
        target_col=args.target_col,
        cd_id_col=args.cd_id_col,
        cd_name_col=args.cd_name_col,
        period_month_col=args.period_month_col,
        split_col=args.split_col,
        models=list(args.models),
        ridge_alphas=parse_float_list(args.ridge_alphas, RIDGE_ALPHAS),
        selection_metric=args.selection_metric,
        random_seed=args.random_seed,
        n_jobs=args.n_jobs,
        drop_missing_target=not args.keep_missing_targets,
        clip_predictions_at_zero=not args.no_clip_predictions,
        include_sovi_features=not args.no_sovi_features,
        include_history_features=not args.no_history_features,
        include_hazard_history_features=not args.no_hazard_history_features,
        include_current_month_counts=not args.no_current_month_counts,
        include_seasonality=not args.no_seasonality,
        include_year_trend=not args.no_year_trend,
        include_all_other_numeric_features=not args.no_other_numeric_features,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)

    metadata = run_b3_cd_tabular_feature_parity(config)

    print("B3 tabular feature-parity baseline completed.")
    print(f"Panel: {config.panel_path}")
    print(f"Output directory: {config.output_dir}")
    print(f"Target column: {config.target_col}")
    print(f"Feature count: {metadata.get('feature_count')}")
    print("Models:")
    for model in config.models:
        print(f"  - {model}")

    selected = metadata.get("selected_model")
    if selected:
        print("Selected model by validation metric:")
        print(f"  model_name: {selected.get('model_name')}")
        print(f"  candidate_name: {selected.get('candidate_name')}")
        print(f"  split: {selected.get('selection_split')}")
        print(f"  metric: {selected.get('selection_metric')}")
        print(f"  MAE: {selected.get('mae')}")
        print(f"  RMSE: {selected.get('rmse')}")
        print(f"  Spearman: {selected.get('spearman')}")

    print("Outputs:")
    for key, value in metadata["outputs"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
