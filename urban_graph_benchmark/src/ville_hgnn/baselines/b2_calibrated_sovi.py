#!/usr/bin/env python3
"""
B2 calibrated SoVI predictors for the Québec CD civil-security / SoVI benchmark.

Purpose:
    Use the static SoVI score as an input to simple calibrated predictors of
    future civil-security event burden.

This is intentionally a simple non-history, non-graph baseline. It tests whether
the SoVI score becomes more useful after numerical calibration, without using
event-history features.

Default input:
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/datasets/cd_month_panel.parquet

Default output directory:
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B2_calibrated_sovi/

Models:
    - linear_sovi
    - ridge_sovi
    - poisson_sovi
    - linear_sovi_seasonal
    - ridge_sovi_seasonal
    - poisson_sovi_seasonal

The seasonal variants add simple month sin/cos features. They still do not use
event-history features.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    evaluate_standard_prediction_frame,
    write_metadata_json,
    write_table,
)


DEFAULT_OUTPUT_DIR = BASELINES_DIR / "B2_calibrated_sovi"

DEFAULT_MODELS = [
    "linear_sovi",
    "ridge_sovi",
    "poisson_sovi",
    "linear_sovi_seasonal",
    "ridge_sovi_seasonal",
    "poisson_sovi_seasonal",
]

RIDGE_ALPHAS = [0.1, 1.0, 10.0]
POISSON_ALPHAS = [0.0, 0.1, 1.0]


@dataclass
class Config:
    """Configuration for B2 calibrated SoVI predictors."""

    panel_path: Path = DEFAULT_PANEL_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR

    target_col: str = "target_next_3_months"
    sovi_score_col: str = SOVI_SCORE_COL

    cd_id_col: str = CD_ID_COL
    cd_name_col: str = CD_NAME_COL
    period_month_col: str = "period_month"
    month_col: str = "month"
    split_col: str = SPLIT_COL

    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))

    ridge_alphas: list[float] = field(default_factory=lambda: list(RIDGE_ALPHAS))
    poisson_alphas: list[float] = field(default_factory=lambda: list(POISSON_ALPHAS))

    # Select hyperparameters by validation MAE.
    selection_metric: str = "mae"

    # By default, prediction rows with missing target values are removed before
    # metrics are computed/written. This avoids incomplete future-window rows.
    drop_missing_target: bool = True

    # Linear/ridge models may produce negative values; these are count forecasts.
    clip_predictions_at_zero: bool = True

    random_seed: int = 42
    poisson_max_iter: int = 1000


def _lazy_import_sklearn() -> dict[str, Any]:
    """Import sklearn components lazily so this module can be imported cheaply."""
    try:
        from sklearn.linear_model import LinearRegression, PoissonRegressor, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "B2 calibrated SoVI requires scikit-learn. Install scikit-learn "
            "in the active environment or run inside the project .venv."
        ) from exc

    return {
        "LinearRegression": LinearRegression,
        "PoissonRegressor": PoissonRegressor,
        "Ridge": Ridge,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
    }


def read_panel(path: Path) -> pd.DataFrame:
    """Read the predictive panel."""
    if not path.exists():
        raise FileNotFoundError(f"Panel file does not exist: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported panel file suffix: {path}")


def validate_config(config: Config) -> None:
    """Validate configuration values."""
    if not isinstance(config.panel_path, Path):
        config.panel_path = Path(config.panel_path)
    if not isinstance(config.output_dir, Path):
        config.output_dir = Path(config.output_dir)

    if not config.models:
        raise ValueError("At least one B2 model must be requested.")

    unknown = sorted(set(config.models) - set(DEFAULT_MODELS))
    if unknown:
        raise ValueError(f"Unknown B2 models: {unknown}. Allowed models: {DEFAULT_MODELS}")

    if not config.ridge_alphas:
        raise ValueError("At least one ridge alpha must be supplied.")

    if not config.poisson_alphas:
        raise ValueError("At least one Poisson alpha must be supplied.")


def validate_panel_columns(df: pd.DataFrame, config: Config) -> None:
    """Validate required input columns."""
    required = [
        config.cd_id_col,
        config.target_col,
        config.sovi_score_col,
        config.split_col,
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            "The B2 panel is missing required columns: "
            f"{missing}. Available columns: {list(df.columns)}"
        )

    seasonal_requested = any(model.endswith("_seasonal") for model in config.models)
    if seasonal_requested and config.month_col not in df.columns:
        raise KeyError(
            f"Seasonal B2 models require month column '{config.month_col}', "
            f"but it is missing. Available columns: {list(df.columns)}"
        )


def split_masks(df: pd.DataFrame, config: Config) -> dict[str, pd.Series]:
    """Return train/validation/test masks."""
    split = df[config.split_col].astype("string")

    train = split.eq("train")
    validation = split.isin(["val", "validation"])
    test = split.eq("test")

    return {
        "train": train,
        "validation": validation,
        "test": test,
        "all": pd.Series(True, index=df.index),
    }


def numeric_target_mask(df: pd.DataFrame, config: Config) -> pd.Series:
    """Rows with a usable numerical target and SoVI score."""
    target = pd.to_numeric(df[config.target_col], errors="coerce")
    score = pd.to_numeric(df[config.sovi_score_col], errors="coerce")
    return target.notna() & score.notna()


def make_design_matrix(
    df: pd.DataFrame,
    *,
    config: Config,
    seasonal: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a simple feature matrix.

    Base feature:
        SoVI score

    Seasonal variants:
        SoVI score + sin/cos month encodings
    """
    score = pd.to_numeric(df[config.sovi_score_col], errors="coerce")

    X = pd.DataFrame(
        {
            "sovi_score": score,
        },
        index=df.index,
    )

    if seasonal:
        month = pd.to_numeric(df[config.month_col], errors="coerce")
        radians = 2.0 * np.pi * (month - 1.0) / 12.0
        X["month_sin"] = np.sin(radians)
        X["month_cos"] = np.cos(radians)

    feature_cols = list(X.columns)
    return X, feature_cols


def model_family(model_name: str) -> str:
    """Return broad model family from model name."""
    if model_name.startswith("linear"):
        return "linear"
    if model_name.startswith("ridge"):
        return "ridge"
    if model_name.startswith("poisson"):
        return "poisson"
    raise ValueError(f"Unknown model family for model: {model_name}")


def model_is_seasonal(model_name: str) -> bool:
    """Whether this model uses month seasonality."""
    return model_name.endswith("_seasonal")


def make_estimator(
    *,
    family: str,
    alpha: float | None,
    config: Config,
) -> Any:
    """Create an sklearn estimator."""
    sk = _lazy_import_sklearn()
    Pipeline = sk["Pipeline"]
    StandardScaler = sk["StandardScaler"]

    if family == "linear":
        LinearRegression = sk["LinearRegression"]
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )

    if family == "ridge":
        Ridge = sk["Ridge"]
        if alpha is None:
            raise ValueError("Ridge requires alpha.")
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=float(alpha), random_state=config.random_seed)),
            ]
        )

    if family == "poisson":
        PoissonRegressor = sk["PoissonRegressor"]
        if alpha is None:
            raise ValueError("PoissonRegressor requires alpha.")
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    PoissonRegressor(
                        alpha=float(alpha),
                        max_iter=int(config.poisson_max_iter),
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown model family: {family}")


def candidate_alphas_for_model(model_name: str, config: Config) -> list[float | None]:
    """Return alpha candidates for a model name."""
    family = model_family(model_name)

    if family == "linear":
        return [None]
    if family == "ridge":
        return [float(a) for a in config.ridge_alphas]
    if family == "poisson":
        return [float(a) for a in config.poisson_alphas]

    raise ValueError(f"Unknown model family: {family}")


def fit_predict_candidate(
    df: pd.DataFrame,
    *,
    model_name: str,
    alpha: float | None,
    config: Config,
) -> tuple[pd.Series, Any, list[str]]:
    """Fit one model candidate on train rows and predict all rows."""
    family = model_family(model_name)
    seasonal = model_is_seasonal(model_name)

    X, feature_cols = make_design_matrix(df, config=config, seasonal=seasonal)
    y = pd.to_numeric(df[config.target_col], errors="coerce")

    masks = split_masks(df, config)
    valid = numeric_target_mask(df, config)

    train_mask = masks["train"] & valid
    if int(train_mask.sum()) < 3:
        raise ValueError(
            f"Not enough training rows for model '{model_name}'. "
            f"Usable train rows: {int(train_mask.sum())}"
        )

    # Poisson targets must be nonnegative.
    if family == "poisson" and (y.loc[train_mask] < 0).any():
        raise ValueError("Poisson model received negative target values.")

    estimator = make_estimator(family=family, alpha=alpha, config=config)
    estimator.fit(X.loc[train_mask, feature_cols], y.loc[train_mask].astype(float))

    valid_x = X[feature_cols].notna().all(axis=1)
    pred = pd.Series(np.nan, index=df.index, dtype=float)
    pred.loc[valid_x] = estimator.predict(X.loc[valid_x, feature_cols])

    if config.clip_predictions_at_zero:
        pred = pred.clip(lower=0)

    return pred, estimator, feature_cols


def build_prediction_frame(
    df: pd.DataFrame,
    *,
    prediction: pd.Series,
    model_name: str,
    candidate_name: str,
    alpha: float | None,
    feature_cols: list[str],
    config: Config,
) -> pd.DataFrame:
    """Build standardized prediction rows for one candidate."""
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
    out["model_name"] = model_name
    out["candidate_name"] = candidate_name
    out["target_col"] = config.target_col
    out["sovi_score_col"] = config.sovi_score_col
    out["alpha"] = np.nan if alpha is None else float(alpha)
    out["features"] = ",".join(feature_cols)
    out["target"] = pd.to_numeric(df[config.target_col], errors="coerce")
    out["prediction"] = pd.to_numeric(prediction, errors="coerce")

    if config.sovi_score_col in df.columns:
        out["sovi_score"] = pd.to_numeric(df[config.sovi_score_col], errors="coerce")

    return out.reset_index(drop=True)


def evaluate_candidate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate candidate predictions by model/candidate/split."""
    if predictions.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["model_name", "candidate_name"]

    for (model_name, candidate_name), sub in predictions.groupby(group_cols, dropna=False):
        metrics = evaluate_standard_prediction_frame(
            sub,
            target_col="target",
            prediction_col="prediction",
            split_col=SPLIT_COL,
            id_col=CD_ID_COL,
        )
        metrics["model_name"] = model_name
        metrics["candidate_name"] = candidate_name

        # Carry alpha/features for audit.
        if "alpha" in sub.columns:
            alpha_values = sub["alpha"].dropna().unique()
            metrics["alpha"] = float(alpha_values[0]) if len(alpha_values) else np.nan
        if "features" in sub.columns:
            feature_values = sub["features"].dropna().unique()
            metrics["features"] = str(feature_values[0]) if len(feature_values) else ""

        rows.append(metrics)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def select_best_candidates(
    candidate_metrics: pd.DataFrame,
    *,
    selection_metric: str,
) -> pd.DataFrame:
    """
    Select one candidate per model using validation metric.

    Lower is better for MAE/RMSE/deviance.
    Higher is better for rank metrics.
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
            f"Selection metric '{selection_metric}' not found in candidate metrics. "
            f"Available columns: {list(validation.columns)}"
        )

    lower_is_better = selection_metric in {"mae", "rmse", "mean_poisson_deviance"}

    selected_rows = []
    for model_name, sub in validation.groupby("model_name", dropna=False):
        sub = sub.copy()

        sort_cols = [selection_metric]
        ascending = [lower_is_better]

        # Deterministic tie-breakers.
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
    """Keep only selected candidate per model."""
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


def extract_coefficients(
    fitted_models: dict[str, tuple[Any, list[str], float | None]],
) -> pd.DataFrame:
    """Extract coefficients/intercepts for audit where available."""
    rows = []

    for candidate_name, (estimator, feature_cols, alpha) in fitted_models.items():
        try:
            model = estimator.named_steps["model"]
        except Exception:
            continue

        row_base = {
            "candidate_name": candidate_name,
            "alpha": np.nan if alpha is None else float(alpha),
            "estimator_class": type(model).__name__,
        }

        intercept = getattr(model, "intercept_", np.nan)
        try:
            intercept_value = float(np.ravel(intercept)[0])
        except Exception:
            intercept_value = np.nan

        rows.append(
            {
                **row_base,
                "term": "intercept",
                "coefficient": intercept_value,
            }
        )

        coef = getattr(model, "coef_", None)
        if coef is None:
            continue

        coef_values = np.ravel(coef).astype(float)
        for feature, value in zip(feature_cols, coef_values):
            rows.append(
                {
                    **row_base,
                    "term": feature,
                    "coefficient": float(value),
                }
            )

    return pd.DataFrame(rows)


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


def run_b2_calibrated_sovi(config: Config) -> dict[str, Any]:
    """
    Run B2 calibrated SoVI predictors and write standardized outputs.
    """
    validate_config(config)
    output_dir = ensure_dir(config.output_dir)

    panel = read_panel(config.panel_path)
    validate_panel_columns(panel, config)

    # Keep panel order deterministic.
    sort_cols = [config.cd_id_col]
    if config.period_month_col in panel.columns:
        sort_cols.append(config.period_month_col)
    panel = panel.sort_values(sort_cols).reset_index(drop=True)

    candidate_predictions = []
    fitted_models: dict[str, tuple[Any, list[str], float | None]] = {}

    for model_name in config.models:
        for alpha in candidate_alphas_for_model(model_name, config):
            alpha_part = "none" if alpha is None else str(alpha).replace(".", "p")
            candidate_name = f"{model_name}__alpha_{alpha_part}"

            pred, estimator, feature_cols = fit_predict_candidate(
                panel,
                model_name=model_name,
                alpha=alpha,
                config=config,
            )

            fitted_models[candidate_name] = (estimator, feature_cols, alpha)

            pred_frame = build_prediction_frame(
                panel,
                prediction=pred,
                model_name=model_name,
                candidate_name=candidate_name,
                alpha=alpha,
                feature_cols=feature_cols,
                config=config,
            )
            candidate_predictions.append(pred_frame)

    if not candidate_predictions:
        raise RuntimeError("No B2 candidates were fitted.")

    candidate_predictions_df = pd.concat(candidate_predictions, ignore_index=True)

    if config.drop_missing_target:
        candidate_predictions_df = candidate_predictions_df[
            candidate_predictions_df["target"].notna()
        ].copy()

    candidate_metrics = evaluate_candidate_predictions(candidate_predictions_df)
    model_selection = select_best_candidates(
        candidate_metrics,
        selection_metric=config.selection_metric,
    )

    predictions = filter_selected_predictions(candidate_predictions_df, model_selection)

    metrics = evaluate_standard_prediction_frame(
        predictions,
        target_col="target",
        prediction_col="prediction",
        split_col=SPLIT_COL,
        id_col=CD_ID_COL,
    )

    prediction_summary = summarize_predictions(predictions)
    coefficients = extract_coefficients(fitted_models)

    prediction_paths = write_table(
        predictions,
        output_dir / "predictions.parquet",
        write_csv_copy=True,
        index=False,
    )

    candidate_prediction_paths = write_table(
        candidate_predictions_df,
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

    coefficients_path = output_dir / "coefficients.csv"
    coefficients.to_csv(coefficients_path, index=False)

    metadata = {
        "baseline_family": "B2_calibrated_sovi",
        "module": "ville_hgnn.baselines.b2_calibrated_sovi",
        "purpose": (
            "Simple calibrated predictors using the SoVI score, optionally with "
            "month seasonality. No event-history or graph features are used."
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
        "models": list(config.models),
        "ridge_alphas": list(config.ridge_alphas),
        "poisson_alphas": list(config.poisson_alphas),
        "candidate_count": int(candidate_predictions_df["candidate_name"].nunique()),
        "prediction_rows": int(len(predictions)),
        "outputs": {
            **{f"predictions_{k}": v for k, v in prediction_paths.items()},
            **{f"candidate_predictions_{k}": v for k, v in candidate_prediction_paths.items()},
            **split_outputs,
            "metrics_csv": str(metrics_path),
            "candidate_metrics_csv": str(candidate_metrics_path),
            "model_selection_csv": str(model_selection_path),
            "prediction_summary_csv": str(prediction_summary_path),
            "coefficients_csv": str(coefficients_path),
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


def candidate_alphas_for_model(model_name: str, config: Config) -> list[float | None]:
    """Return candidate alpha values for a model."""
    family = model_family(model_name)

    if family == "linear":
        return [None]
    if family == "ridge":
        return [float(a) for a in config.ridge_alphas]
    if family == "poisson":
        return [float(a) for a in config.poisson_alphas]

    raise ValueError(f"Unknown model family: {family}")


def parse_float_list(values: Sequence[str] | None, default: Sequence[float]) -> list[float]:
    """Parse a list of floats from CLI values."""
    if not values:
        return [float(v) for v in default]
    return [float(v) for v in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run B2 calibrated SoVI predictors for the Québec CD civil-security / SoVI benchmark."
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
        help="Output directory for B2 calibrated SoVI results.",
    )
    parser.add_argument(
        "--target-col",
        default="target_next_3_months",
        help="Target column to forecast.",
    )
    parser.add_argument(
        "--sovi-score-col",
        default=SOVI_SCORE_COL,
        help="SoVI score column used as predictor.",
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
        "--month-col",
        default="month",
        help="Calendar month column used for seasonal features.",
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
        help="B2 calibrated SoVI models to run.",
    )
    parser.add_argument(
        "--ridge-alphas",
        nargs="+",
        default=None,
        help="Ridge alpha values. Default: 0.1 1.0 10.0.",
    )
    parser.add_argument(
        "--poisson-alphas",
        nargs="+",
        default=None,
        help="PoissonRegressor alpha values. Default: 0.0 0.1 1.0.",
    )
    parser.add_argument(
        "--selection-metric",
        default="mae",
        choices=["mae", "rmse", "mean_poisson_deviance", "spearman", "ndcg_at_25"],
        help="Validation metric used to select one candidate per model.",
    )
    parser.add_argument(
        "--keep-missing-targets",
        action="store_true",
        help="Keep rows with missing targets in candidate/prediction outputs. Metrics ignore missing targets.",
    )
    parser.add_argument(
        "--no-clip-predictions",
        action="store_true",
        help="Do not clip linear/ridge predictions at zero.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--poisson-max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for PoissonRegressor.",
    )

    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        panel_path=args.panel_path,
        output_dir=args.output_dir,
        target_col=args.target_col,
        sovi_score_col=args.sovi_score_col,
        cd_id_col=args.cd_id_col,
        cd_name_col=args.cd_name_col,
        period_month_col=args.period_month_col,
        month_col=args.month_col,
        split_col=args.split_col,
        models=list(args.models),
        ridge_alphas=parse_float_list(args.ridge_alphas, RIDGE_ALPHAS),
        poisson_alphas=parse_float_list(args.poisson_alphas, POISSON_ALPHAS),
        selection_metric=args.selection_metric,
        drop_missing_target=not args.keep_missing_targets,
        clip_predictions_at_zero=not args.no_clip_predictions,
        random_seed=args.random_seed,
        poisson_max_iter=args.poisson_max_iter,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)

    metadata = run_b2_calibrated_sovi(config)

    print("B2 calibrated SoVI baseline completed.")
    print(f"Panel: {config.panel_path}")
    print(f"Output directory: {config.output_dir}")
    print(f"Target column: {config.target_col}")
    print(f"SoVI score column: {config.sovi_score_col}")
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
