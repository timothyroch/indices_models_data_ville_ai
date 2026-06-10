#!/usr/bin/env python3
"""
A3 spatial-block feature-parity tabular baselines WITH paper-grade model-behavior plots.

This module is intentionally separate from ``a3_tabular_feature_parity_spatial_block.py`` so
the original spatial-block A3 outputs remain safe. It imports and reuses the original A3
implementation, reruns the same candidate suite into a different output folder,
and adds paper-oriented interpretation plots:

- validation-vs-test MAE scatter
- selected-candidate MAE leaderboard
- observed-vs-predicted calibration with binned uncertainty bands
- monthly aggregate observed-vs-predicted curve
- residuals by predicted-decile
- feature-importance bars
- 1D partial dependence / ICE-style plots
- 2D response heatmaps and 3D response surfaces

The plots are designed to complement the benchmark-summary figures from
``07_compare_a0_a1_a2_a3_results.py``. That comparison script answers
"which baseline wins?" This script additionally asks "what did the tabular model
learn?"

The plotting logic is deliberately conservative:
- no retraining decisions are based on test metrics;
- reference models are selected from validation MAE;
- retrospective models are plotted separately and never described as forecasts;
- PDP/ICE figures are predictive-behavior diagnostics, not causal effects.

Save as:
    urban_graph_benchmark/src/ville_hgnn/baselines/a3_tabular_feature_parity_spatial_block_with_plots.py

Example:
    PYTHONPATH=urban_graph_benchmark/src python -m ville_hgnn.baselines.a3_tabular_feature_parity_spatial_block_with_plots \\
      --config urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml \\
      --split-scheme spatial_block \\
      --hgb-grid small
"""

from __future__ import annotations

import argparse
import inspect
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    _PANDAS_IMPORT_ERROR = exc
else:
    _PANDAS_IMPORT_ERROR = None

try:  # Optional plotting dependency.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception as exc:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    cm = None  # type: ignore[assignment]
    MaxNLocator = None  # type: ignore[assignment]
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None

try:
    import sklearn
except Exception:  # pragma: no cover
    sklearn = None  # type: ignore[assignment]

from ville_hgnn.baselines import a3_tabular_feature_parity_spatial_block as base
from ville_hgnn.baselines.a1_svi_direct_ranking import (
    build_svi_score_specs,
    validate_static_svi_scores,
)
from ville_hgnn.baselines.common import (
    BINARY_TARGET_COLUMN,
    DATASET_VERSION_DEFAULT,
    DEFAULT_CONFIG_PATH,
    TARGET_COLUMN,
    build_run_context,
    evaluate_prediction_frame,
    get_baseline_paths,
    load_benchmark_frame,
    split_column_for_scheme,
    split_counts,
    split_type_for_scheme,
)
from ville_hgnn.utils.io import config_hash, file_hash, to_jsonable, write_json, write_markdown


STAGE_SLUG = "A3_feature_parity_tabular_spatial_block_with_plots"
MODEL_STAGE = "A3_feature_parity_tabular_spatial_block_with_plots"
DEFAULT_SPLIT_SCHEME = "spatial_block"

ZONE_COL = base.ZONE_COL
PERIOD_COL = base.PERIOD_COL
PREDICTED_COL = base.PREDICTED_COL
BINARY_SCORE_COL = base.BINARY_SCORE_COL
OBSERVED_ALIAS_COL = base.OBSERVED_ALIAS_COL

STRICT_TRAIN_STATIC_SETTING = base.STRICT_TRAIN_STATIC_SETTING
ROLLING_HISTORY_SETTING = base.ROLLING_HISTORY_SETTING
RETROSPECTIVE_SETTING = base.RETROSPECTIVE_SETTING

RANDOM_SEED = base.RANDOM_SEED
MIN_PREDICTION = base.MIN_PREDICTION


class A3WithPlotsError(base.A3BaselineError):
    """Raised when A3-with-plots generation fails."""


@dataclass(frozen=True)
class PlotConfig:
    """Configuration for interpretation plots."""

    enabled: bool = True
    max_pdp_background: int = 750
    max_ice_lines: int = 40
    pdp_grid_size: int = 31
    max_pdp_features: int = 6
    surface_grid_size: int = 41
    max_surface_pairs: int = 6
    sample_scatter_points: int = 3000
    dpi: int = 180
    plot_format: str = "png"


def require_runtime_dependencies() -> None:
    """Fail clearly if required dependencies are unavailable."""

    if pd is None:
        raise A3WithPlotsError("pandas is required.") from _PANDAS_IMPORT_ERROR
    if np is None:
        raise A3WithPlotsError("numpy is required.") from _NUMPY_IMPORT_ERROR


def require_plotting_dependencies() -> None:
    """Fail clearly if plotting was requested but matplotlib is unavailable."""

    if plt is None:
        raise A3WithPlotsError(
            "matplotlib is required for plots. Install matplotlib or run with --no-plots."
        ) from _MATPLOTLIB_IMPORT_ERROR


def ensure_dir(path: Path) -> Path:
    """Create a directory and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert series to numeric float."""

    return pd.to_numeric(series, errors="coerce").astype(float)


def prepare_feature_frame_compat(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    split_scheme: str,
    svi_specs: Sequence[Any],
) -> tuple[pd.DataFrame, Mapping[str, list[str]], pd.DataFrame]:
    """Call base.prepare_feature_frame across known A3 API variants.

    Earlier local A3 files used ``prepare_feature_frame(frame, train, svi_specs)``.
    Some generated variants used ``prepare_feature_frame(frame, split_scheme, svi_specs)``.
    A short-lived draft used four arguments. This adapter keeps the plotting
    script robust without editing the base A3 implementation.
    """

    fn = base.prepare_feature_frame
    params = list(inspect.signature(fn).parameters)

    if len(params) == 3:
        second = params[1].lower()
        if "split" in second:
            return fn(frame, split_scheme, svi_specs)
        return fn(frame, train, svi_specs)

    if len(params) == 4:
        return fn(frame, train, split_scheme, svi_specs)

    # Fallback: try the most likely current signature, then older variants.
    errors: list[Exception] = []
    for args in (
        (frame, train, svi_specs),
        (frame, split_scheme, svi_specs),
        (frame, train, split_scheme, svi_specs),
    ):
        try:
            return fn(*args)
        except TypeError as exc:
            errors.append(exc)

    raise A3WithPlotsError(
        "Could not call base.prepare_feature_frame with any known signature. "
        f"Detected parameters: {params}. Last error: {errors[-1] if errors else 'unknown'}"
    )


def prediction_frame_for_model(
    frame: pd.DataFrame,
    *,
    split_scheme: str,
    fitted: base.FittedTabularModel,
    predictions: pd.Series,
) -> pd.DataFrame:
    """Create standardized long prediction rows for one A3-with-plots model."""

    split_col = split_column_for_scheme(split_scheme)

    cols = [
        ZONE_COL,
        PERIOD_COL,
        split_col,
        TARGET_COLUMN,
        BINARY_TARGET_COLUMN,
        "year",
        base.MONTH_COL,
        "population_total_2021",
        "land_area_km2",
        "population_density",
        "population_density_per_km2",
        base.SAME_MONTH_NON_WATER_COL,
    ]
    cols = [col for col in cols if col in frame.columns]

    out = frame[cols].copy()
    pred = safe_numeric(predictions).fillna(0).clip(lower=0)
    pred.index = out.index

    out[OBSERVED_ALIAS_COL] = out[TARGET_COLUMN]
    out[PREDICTED_COL] = pred
    out["predicted_score"] = pred
    out[BINARY_SCORE_COL] = base.poisson_any_probability(pred)
    out["model_stage"] = MODEL_STAGE
    out["model_name"] = fitted.model_name
    out["model_family"] = fitted.model_family
    out["hyperparameter_id"] = fitted.hyperparameter_id
    out["feature_set_name"] = fitted.feature_set_name
    out["prediction_setting"] = fitted.prediction_setting
    out["target_transform"] = fitted.target_transform

    return out


def evaluate_one_model(
    prediction_frame: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    full_frame: pd.DataFrame,
    split_scheme: str,
    fitted: base.FittedTabularModel,
) -> pd.DataFrame:
    """Evaluate one model using shared metrics conventions and the new stage slug."""

    context = build_run_context(
        config=config,
        frame=full_frame,
        split_scheme=split_scheme,
        prediction_setting=fitted.prediction_setting,
        model_stage=MODEL_STAGE,
        model_name=fitted.model_name,
        target_name=TARGET_COLUMN,
        target_type="count",
        feature_set_name=fitted.feature_set_name,
        dataset_version=DATASET_VERSION_DEFAULT,
    )

    metrics = evaluate_prediction_frame(
        prediction_frame,
        context=context,
        split_scheme=split_scheme,
        observed_col=TARGET_COLUMN,
        predicted_col=PREDICTED_COL,
        binary_observed_col=BINARY_TARGET_COLUMN,
        binary_score_col=BINARY_SCORE_COL,
        ranking_score_col="predicted_score",
    )

    out = base.standardize_a3_metric_schema(metrics)
    out["model_family"] = fitted.model_family
    out["hyperparameter_id"] = fitted.hyperparameter_id
    return out


def write_long_prediction_partitions(
    predictions_long: pd.DataFrame,
    output_dir: Path,
    *,
    split_scheme: str,
) -> dict[str, str]:
    """Write validation/test prediction files for all models in long format."""

    split_col = split_column_for_scheme(split_scheme)
    validation_path = output_dir / "predictions_validation.parquet"
    test_path = output_dir / "predictions_test.parquet"

    preferred_cols = [
        ZONE_COL,
        PERIOD_COL,
        split_col,
        OBSERVED_ALIAS_COL,
        TARGET_COLUMN,
        BINARY_TARGET_COLUMN,
        PREDICTED_COL,
        "predicted_score",
        BINARY_SCORE_COL,
        "model_stage",
        "model_name",
        "model_family",
        "hyperparameter_id",
        "feature_set_name",
        "prediction_setting",
        "target_transform",
        "year",
        base.MONTH_COL,
        "population_total_2021",
        "land_area_km2",
        "population_density",
        "population_density_per_km2",
        base.SAME_MONTH_NON_WATER_COL,
    ]
    cols = [col for col in preferred_cols if col in predictions_long.columns]

    validation = predictions_long[predictions_long[split_col].astype(str) == "validation"][cols].copy()
    test = predictions_long[predictions_long[split_col].astype(str) == "test"][cols].copy()

    validation.to_parquet(validation_path, index=False)
    test.to_parquet(test_path, index=False)

    return {
        "predictions_validation": str(validation_path),
        "predictions_test": str(test_path),
    }


def metric_value(
    metrics: pd.DataFrame,
    *,
    model_name: str,
    split_name: str,
    metric_name: str,
) -> float | None:
    """Fetch a metric value."""

    subset = metrics[
        (metrics["model_name"] == model_name)
        & (metrics["split_name"] == split_name)
        & (metrics["metric_name"] == metric_name)
    ].copy()
    if subset.empty:
        return None
    values = pd.to_numeric(subset["metric_value"], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def selected_reference_models(
    model_selection_audit: pd.DataFrame,
    feature_set_audit: pd.DataFrame,
) -> dict[str, str]:
    """Select reference model names for interpretation plots.

    For the spatial-block run, the most important reference is the validation-
    selected strict/rolling forecasting model. In the current benchmark this is
    expected to be the lagged-reporting HGB model. The function still falls back
    to validation MAE if explicit selection flags are missing.
    """

    if model_selection_audit.empty:
        return {}

    audit = model_selection_audit.copy()
    audit["validation_mae_num"] = pd.to_numeric(audit["validation_mae"], errors="coerce")
    primary_names = set(
        feature_set_audit.loc[
            feature_set_audit["is_primary_feature_set"].astype(bool),
            "feature_set_name",
        ].astype(str)
    )

    refs: dict[str, str] = {}

    def _first_flagged(flag: str) -> str | None:
        if flag not in audit.columns:
            return None
        flagged = audit[audit[flag].astype(bool)].copy()
        if flagged.empty:
            return None
        flagged = flagged.sort_values("validation_mae_num", na_position="last")
        return str(flagged.iloc[0]["model_name"])

    strict_flagged = _first_flagged("selected_overall_strict_forecasting")
    if strict_flagged is not None:
        refs["selected_strict_forecasting"] = strict_flagged

    retro_flagged = _first_flagged("selected_overall_retrospective")
    if retro_flagged is not None:
        refs["selected_retrospective"] = retro_flagged

    summary_flagged = _first_flagged("selected_for_test_summary")
    if summary_flagged is not None:
        refs["selected_for_test_summary"] = summary_flagged

    strict_primary = audit[
        audit["prediction_setting"].isin([STRICT_TRAIN_STATIC_SETTING, ROLLING_HISTORY_SETTING])
        & audit["feature_set_name"].astype(str).isin(primary_names)
        & audit["validation_mae_num"].notna()
    ].copy()
    if not strict_primary.empty:
        row = strict_primary.sort_values("validation_mae_num").iloc[0]
        refs.setdefault("best_strict_primary", str(row["model_name"]))

    strict_overall = audit[
        audit["prediction_setting"].isin([STRICT_TRAIN_STATIC_SETTING, ROLLING_HISTORY_SETTING])
        & audit["validation_mae_num"].notna()
    ].copy()
    if not strict_overall.empty:
        row = strict_overall.sort_values("validation_mae_num").iloc[0]
        refs.setdefault("best_strict_overall", str(row["model_name"]))

    # Explicitly expose the best lagged-reporting HGB candidate when present,
    # because it is the scientific focus of the spatial-block result.
    lagged_hgb = audit[
        audit["feature_set_name"].astype(str).eq("A3_lagged_reporting_forecasting")
        & audit["model_family"].astype(str).eq("hist_gradient_boosting_poisson")
        & audit["validation_mae_num"].notna()
    ].copy()
    if not lagged_hgb.empty:
        row = lagged_hgb.sort_values("validation_mae_num").iloc[0]
        refs["lagged_reporting_hgb_focus"] = str(row["model_name"])

    retrospective_primary = audit[
        (audit["prediction_setting"] == RETROSPECTIVE_SETTING)
        & audit["feature_set_name"].astype(str).isin(primary_names)
        & audit["validation_mae_num"].notna()
    ].copy()
    if not retrospective_primary.empty:
        row = retrospective_primary.sort_values("validation_mae_num").iloc[0]
        refs.setdefault("best_retrospective_primary", str(row["model_name"]))

    static_primary = audit[
        (audit["prediction_setting"] == STRICT_TRAIN_STATIC_SETTING)
        & audit["feature_set_name"].astype(str).isin(primary_names)
        & audit["validation_mae_num"].notna()
    ].copy()
    if not static_primary.empty:
        row = static_primary.sort_values("validation_mae_num").iloc[0]
        refs.setdefault("best_static_primary", str(row["model_name"]))

    # Preserve insertion order while removing duplicates.
    deduped: dict[str, str] = {}
    seen: set[str] = set()
    for key, value in refs.items():
        if value in seen:
            continue
        seen.add(value)
        deduped[key] = value
    return deduped



def imputed_feature_matrix(
    frame: pd.DataFrame,
    fitted: base.FittedTabularModel,
) -> np.ndarray:
    """Build an imputed/scaled feature matrix for a fitted model."""

    features = frame[fitted.feature_columns].copy()
    out = pd.DataFrame(index=features.index)

    for col in fitted.feature_columns:
        values = safe_numeric(features[col])
        median = float(fitted.feature_medians.get(col, 0.0))
        filled = values.fillna(median)

        if fitted.standardize:
            mean = float(fitted.feature_means.get(col, 0.0))
            std = float(fitted.feature_stds.get(col, 1.0))
            if not math.isfinite(std) or std <= 1e-12:
                std = 1.0
            out[col] = (filled - mean) / std
        else:
            out[col] = filled

    return out.to_numpy(dtype=float)


def predict_with_fitted(
    frame: pd.DataFrame,
    fitted: base.FittedTabularModel,
) -> pd.Series:
    """Predict counts with a fitted A3 model on a frame containing its feature columns."""

    X = imputed_feature_matrix(frame, fitted)

    if fitted.model_family == "ridge_log_count":
        if fitted.coefficients is None:
            raise A3WithPlotsError(f"Ridge model missing coefficients: {fitted.model_name}")
        beta = np.asarray(fitted.coefficients, dtype=float)
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        pred = np.expm1(X_with_intercept @ beta)
    elif fitted.model_family == "hist_gradient_boosting_poisson":
        if fitted.sklearn_model is None:
            raise A3WithPlotsError(f"Missing sklearn model: {fitted.model_name}")
        pred = fitted.sklearn_model.predict(X)
    elif fitted.model_family == "random_forest_log_count":
        if fitted.sklearn_model is None:
            raise A3WithPlotsError(f"Missing sklearn model: {fitted.model_name}")
        pred = np.expm1(fitted.sklearn_model.predict(X))
    else:
        raise A3WithPlotsError(f"Unsupported fitted model family for plotting: {fitted.model_family}")

    pred = np.clip(pred, MIN_PREDICTION, None)
    return pd.Series(pred, index=frame.index, dtype=float)


def clean_label(value: str, max_len: int = 72) -> str:
    """Make long feature/model names readable."""

    replacements = {
        "target_history__water_drainage_count_": "target ",
        "target_train_summary__": "target train ",
        "reporting_history__total_311_count_non_water_drainage_": "non-water ",
        "requests_history__requests_total_": "all 311 ",
        "static__log1p_": "log ",
        "static_spatial__": "",
        "svi_primary__": "",
        "svi_diagnostic__": "",
        "calendar__": "",
        "_shift1": " shifted",
        "_": " ",
    }
    out = str(value)
    for old, new in replacements.items():
        out = out.replace(old, new)
    out = " ".join(out.split())
    if len(out) > max_len:
        return out[: max_len - 1] + "…"
    return out


def savefig(path: Path, *, dpi: int) -> str:
    """Save current matplotlib figure with tight layout."""

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_validation_vs_test_mae(
    model_selection_audit: pd.DataFrame,
    plots_dir: Path,
    cfg: PlotConfig,
) -> str | None:
    """Scatter plot of validation vs test MAE for all A3 candidates."""

    if model_selection_audit.empty:
        return None

    df = model_selection_audit.copy()
    df["validation_mae"] = pd.to_numeric(df["validation_mae"], errors="coerce")
    df["test_mae"] = pd.to_numeric(df["test_mae"], errors="coerce")
    df = df.dropna(subset=["validation_mae", "test_mae"])
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    families = list(df["model_family"].dropna().astype(str).unique())
    markers = ["o", "s", "^", "D", "P", "X"]

    for i, family in enumerate(families):
        part = df[df["model_family"] == family]
        ax.scatter(
            part["validation_mae"],
            part["test_mae"],
            alpha=0.78,
            s=58,
            marker=markers[i % len(markers)],
            label=clean_label(family, 36),
            edgecolor="black",
            linewidth=0.35,
        )

    lo = float(min(df["validation_mae"].min(), df["test_mae"].min()))
    hi = float(max(df["validation_mae"].max(), df["test_mae"].max()))
    pad = (hi - lo) * 0.06 if hi > lo else 0.1
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.2, color="gray")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title("A3 validation vs test MAE")
    ax.set_xlabel("Validation MAE")
    ax.set_ylabel("Test MAE")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=True, fontsize=8)

    return savefig(plots_dir / f"a3_validation_vs_test_mae.{cfg.plot_format}", dpi=cfg.dpi)


def plot_selected_test_mae(
    model_selection_audit: pd.DataFrame,
    plots_dir: Path,
    cfg: PlotConfig,
) -> str | None:
    """Horizontal leaderboard of selected candidates by feature set/family."""

    if model_selection_audit.empty:
        return None

    df = model_selection_audit[model_selection_audit["selected_for_test_summary"].astype(bool)].copy()
    if df.empty:
        return None

    df["test_mae"] = pd.to_numeric(df["test_mae"], errors="coerce")
    df = df.dropna(subset=["test_mae"])
    if df.empty:
        return None

    df["label"] = df["feature_set_name"].astype(str).map(lambda s: clean_label(s, 52)) + "\n" + df[
        "model_family"
    ].astype(str).map(lambda s: clean_label(s, 36))
    df = df.sort_values("test_mae", ascending=True)

    height = max(5.2, 0.48 * len(df))
    fig, ax = plt.subplots(figsize=(9.2, height))
    ax.barh(df["label"], df["test_mae"], alpha=0.88)
    ax.invert_yaxis()
    ax.set_xlabel("Test MAE")
    ax.set_title("Selected A3 candidates by test MAE")
    ax.grid(axis="x", alpha=0.28)
    for y, value in enumerate(df["test_mae"]):
        ax.text(value + 0.015, y, f"{value:.3f}", va="center", fontsize=8)

    return savefig(plots_dir / f"a3_selected_test_mae.{cfg.plot_format}", dpi=cfg.dpi)


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    reference_model_name: str,
    plots_dir: Path,
    cfg: PlotConfig,
    top_n: int = 22,
) -> str | None:
    """Feature-importance bar chart for a selected model."""

    if feature_importance.empty:
        return None

    df = feature_importance[feature_importance["model_name"] == reference_model_name].copy()
    if df.empty:
        return None

    df["absolute_importance"] = pd.to_numeric(df["absolute_importance"], errors="coerce")
    df = df.dropna(subset=["absolute_importance"])
    df = df[df["feature"].astype(str) != "intercept"].copy()
    if df.empty:
        return None

    df = df.sort_values("absolute_importance", ascending=False).head(top_n)
    df["label"] = df["feature"].astype(str).map(clean_label)
    df = df.sort_values("absolute_importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9.0, max(5.2, 0.34 * len(df))))
    ax.barh(df["label"], df["absolute_importance"], alpha=0.88)
    ax.set_xlabel("Absolute importance")
    ax.set_title(f"Feature importance — {clean_label(reference_model_name, 72)}")
    ax.grid(axis="x", alpha=0.28)

    return savefig(plots_dir / f"feature_importance__{safe_filename(reference_model_name)}.{cfg.plot_format}", dpi=cfg.dpi)


def sample_rows(frame: pd.DataFrame, max_rows: int, random_state: int = RANDOM_SEED) -> pd.DataFrame:
    """Sample rows for plotting while preserving all rows if small."""

    if len(frame) <= max_rows:
        return frame.copy()
    return frame.sample(n=max_rows, random_state=random_state).copy()


def plot_observed_predicted_calibration(
    predictions_long: pd.DataFrame,
    model_name: str,
    plots_dir: Path,
    cfg: PlotConfig,
    split_scheme: str,
) -> str | None:
    """Observed vs predicted calibration curve with binned interval."""

    split_col = split_column_for_scheme(split_scheme)
    split_name = "test"
    df = predictions_long[
        (predictions_long["model_name"] == model_name)
        & (predictions_long[split_col].astype(str) == split_name)
    ].copy()
    if df.empty:
        return None

    df["observed"] = safe_numeric(df[TARGET_COLUMN])
    df["predicted"] = safe_numeric(df[PREDICTED_COL]).clip(lower=0)
    df = df.dropna(subset=["observed", "predicted"])
    if df.empty:
        return None

    # Quantile bins in predicted space.
    n_bins = min(30, max(8, int(math.sqrt(len(df)) // 2)))
    df["_bin"] = pd.qcut(df["predicted"].rank(method="first"), q=n_bins, duplicates="drop")
    binned = (
        df.groupby("_bin", observed=False)
        .agg(
            predicted_mean=("predicted", "mean"),
            observed_mean=("observed", "mean"),
            observed_q05=("observed", lambda s: float(np.quantile(s, 0.05))),
            observed_q95=("observed", lambda s: float(np.quantile(s, 0.95))),
            n=("observed", "size"),
        )
        .reset_index(drop=True)
    )

    scatter = sample_rows(df, cfg.sample_scatter_points)

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.scatter(
        scatter["predicted"],
        scatter["observed"],
        alpha=0.18,
        s=10,
        label="Tract-month samples",
        rasterized=True,
    )
    ax.fill_between(
        binned["predicted_mean"],
        binned["observed_q05"],
        binned["observed_q95"],
        alpha=0.22,
        label="Observed 5–95% band by prediction bin",
    )
    ax.plot(
        binned["predicted_mean"],
        binned["observed_mean"],
        linewidth=2.4,
        label="Observed mean by prediction bin",
    )
    ax.plot(
        binned["predicted_mean"],
        binned["predicted_mean"],
        linestyle="--",
        color="gray",
        linewidth=1.2,
        label="Perfect calibration",
    )
    ax.set_title(f"Observed vs predicted calibration — {clean_label(model_name, 60)}")
    ax.set_xlabel("Predicted water/drainage requests")
    ax.set_ylabel("Observed water/drainage requests")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=True)

    return savefig(plots_dir / f"calibration__{safe_filename(model_name)}.{cfg.plot_format}", dpi=cfg.dpi)


def plot_monthly_aggregate(
    predictions_long: pd.DataFrame,
    model_name: str,
    plots_dir: Path,
    cfg: PlotConfig,
    split_scheme: str,
) -> str | None:
    """Monthly aggregate observed-vs-predicted line chart."""

    split_col = split_column_for_scheme(split_scheme)
    df = predictions_long[
        (predictions_long["model_name"] == model_name)
        & (predictions_long[split_col].astype(str).isin(["validation", "test"]))
    ].copy()
    if df.empty:
        return None

    df["period_dt"] = pd.to_datetime(df[PERIOD_COL].astype(str), errors="coerce")
    df["observed"] = safe_numeric(df[TARGET_COLUMN])
    df["predicted"] = safe_numeric(df[PREDICTED_COL]).clip(lower=0)
    df = df.dropna(subset=["period_dt", "observed", "predicted"])
    if df.empty:
        return None

    monthly = (
        df.groupby(["period_dt", split_col])
        .agg(observed_total=("observed", "sum"), predicted_total=("predicted", "sum"))
        .reset_index()
        .sort_values("period_dt")
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(monthly["period_dt"], monthly["observed_total"], marker="o", linewidth=2.0, label="Observed total")
    ax.plot(monthly["period_dt"], monthly["predicted_total"], marker="s", linewidth=2.0, label="Predicted total")
    ax.set_title(f"Monthly aggregate burden — {clean_label(model_name, 60)}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Water/drainage request count")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.autofmt_xdate()

    return savefig(plots_dir / f"monthly_aggregate__{safe_filename(model_name)}.{cfg.plot_format}", dpi=cfg.dpi)


def plot_residual_by_decile(
    predictions_long: pd.DataFrame,
    model_name: str,
    plots_dir: Path,
    cfg: PlotConfig,
    split_scheme: str,
) -> str | None:
    """Residual distribution by predicted-risk decile."""

    split_col = split_column_for_scheme(split_scheme)
    df = predictions_long[
        (predictions_long["model_name"] == model_name)
        & (predictions_long[split_col].astype(str) == "test")
    ].copy()
    if df.empty:
        return None

    df["observed"] = safe_numeric(df[TARGET_COLUMN])
    df["predicted"] = safe_numeric(df[PREDICTED_COL]).clip(lower=0)
    df["residual"] = df["observed"] - df["predicted"]
    df = df.dropna(subset=["observed", "predicted", "residual"])
    if df.empty:
        return None

    df["predicted_decile"] = pd.qcut(df["predicted"].rank(method="first"), q=10, labels=False, duplicates="drop") + 1
    summary = (
        df.groupby("predicted_decile")
        .agg(
            residual_mean=("residual", "mean"),
            residual_q10=("residual", lambda s: float(np.quantile(s, 0.10))),
            residual_q90=("residual", lambda s: float(np.quantile(s, 0.90))),
            predicted_mean=("predicted", "mean"),
            observed_mean=("observed", "mean"),
            n=("residual", "size"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.1)
    ax.fill_between(
        summary["predicted_decile"],
        summary["residual_q10"],
        summary["residual_q90"],
        alpha=0.24,
        label="Residual 10–90% band",
    )
    ax.plot(summary["predicted_decile"], summary["residual_mean"], marker="o", linewidth=2.3, label="Mean residual")
    ax.set_title(f"Residuals by predicted decile — {clean_label(model_name, 60)}")
    ax.set_xlabel("Predicted-risk decile")
    ax.set_ylabel("Observed − predicted")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    return savefig(plots_dir / f"residual_deciles__{safe_filename(model_name)}.{cfg.plot_format}", dpi=cfg.dpi)


def top_features_for_pdp(
    feature_importance: pd.DataFrame,
    model_name: str,
    fitted: base.FittedTabularModel,
    max_features: int,
) -> list[str]:
    """Choose features for PDP plots."""

    if not feature_importance.empty:
        df = feature_importance[
            (feature_importance["model_name"] == model_name)
            & (feature_importance["feature"].astype(str) != "intercept")
        ].copy()
        if not df.empty:
            df["absolute_importance"] = pd.to_numeric(df["absolute_importance"], errors="coerce")
            df = df.dropna(subset=["absolute_importance"]).sort_values("absolute_importance", ascending=False)
            ordered = [f for f in df["feature"].astype(str).tolist() if f in fitted.feature_columns]
            if ordered:
                return ordered[:max_features]

    # Fallback: meaningful features first.
    priority_tokens = [
        # Spatial-block focus: transferable lagged reporting signal.
        "reporting_history__total_311_count_non_water_drainage_lag_1",
        "requests_history__requests_total_lag_1",
        "reporting_history__total_311_count_non_water_drainage_roll3_mean_shift1",
        "requests_history__requests_total_roll3_mean_shift1",
        "reporting_history__total_311_count_non_water_drainage_lag_12",
        "requests_history__requests_total_lag_12",
        # Secondary features when the selected model includes target/static families.
        "target_train_summary__mean",
        "target_history__water_drainage_count_lag_1",
        "target_history__water_drainage_count_roll3_mean",
        "svi_primary__svi_percentile",
        "static__log1p_population_density",
        "static_spatial__tract_centroid_lon",
        "static_spatial__tract_centroid_lat",
    ]

    selected: list[str] = []
    for token in priority_tokens:
        for feature in fitted.feature_columns:
            if token in feature and feature not in selected:
                selected.append(feature)
                break

    for feature in fitted.feature_columns:
        if feature not in selected:
            selected.append(feature)

    return selected[:max_features]


def feature_grid(values: pd.Series, grid_size: int) -> np.ndarray:
    """Build a robust feature grid from observed values."""

    v = safe_numeric(values).replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return np.asarray([], dtype=float)

    unique = np.sort(v.unique())
    if len(unique) <= min(grid_size, 12):
        return unique.astype(float)

    lo = float(np.quantile(v, 0.05))
    hi = float(np.quantile(v, 0.95))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo = float(v.min())
        hi = float(v.max())

    if hi <= lo:
        return np.asarray([lo], dtype=float)

    return np.linspace(lo, hi, grid_size)


def plot_pdp_1d(
    frame: pd.DataFrame,
    fitted: base.FittedTabularModel,
    feature: str,
    plots_dir: Path,
    cfg: PlotConfig,
) -> str | None:
    """Plot 1D partial dependence with ICE-style sampled curves."""

    if feature not in fitted.feature_columns or feature not in frame.columns:
        return None

    background = sample_rows(frame, cfg.max_pdp_background)
    grid = feature_grid(background[feature], cfg.pdp_grid_size)
    if len(grid) <= 1:
        return None

    pdp_mean: list[float] = []
    pdp_q10: list[float] = []
    pdp_q90: list[float] = []
    ice_matrix: list[np.ndarray] = []

    ice_background = sample_rows(background, min(cfg.max_ice_lines, len(background)))

    for value in grid:
        modified = background.copy()
        modified[feature] = value
        preds = predict_with_fitted(modified, fitted).to_numpy(dtype=float)
        pdp_mean.append(float(np.mean(preds)))
        pdp_q10.append(float(np.quantile(preds, 0.10)))
        pdp_q90.append(float(np.quantile(preds, 0.90)))

        modified_ice = ice_background.copy()
        modified_ice[feature] = value
        ice_matrix.append(predict_with_fitted(modified_ice, fitted).to_numpy(dtype=float))

    ice = np.vstack(ice_matrix)  # grid x n_ice

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for i in range(ice.shape[1]):
        ax.plot(grid, ice[:, i], color="gray", alpha=0.10, linewidth=0.8)

    ax.fill_between(grid, pdp_q10, pdp_q90, alpha=0.22, label="PDP 10–90% band")
    ax.plot(grid, pdp_mean, linewidth=2.8, label="Mean partial dependence")
    ax.set_title(f"Partial dependence — {clean_label(feature, 68)}")
    ax.set_xlabel(clean_label(feature, 60))
    ax.set_ylabel("Predicted water/drainage count")
    ax.grid(True, alpha=0.25)

    # Rug marks for observed feature distribution.
    ymin, ymax = ax.get_ylim()
    rug_y = ymin + 0.02 * (ymax - ymin)
    observed = safe_numeric(background[feature]).dropna()
    if not observed.empty:
        rug = observed.sample(n=min(len(observed), 250), random_state=RANDOM_SEED)
        ax.plot(rug, np.full(len(rug), rug_y), "|", color="black", alpha=0.25, markersize=7)

    ax.legend(frameon=True, fontsize=8)

    path = plots_dir / f"pdp_1d__{safe_filename(fitted.model_name)}__{safe_filename(feature)}.{cfg.plot_format}"
    return savefig(path, dpi=cfg.dpi)


def preferred_surface_pairs(fitted: base.FittedTabularModel) -> list[tuple[str, str]]:
    """Choose meaningful 2D response-surface feature pairs when present.

    The first candidate pairs intentionally target the spatial-block finding:
    lagged non-water 311 reporting and lagged all-311 activity transfer across
    spatial blocks better than the heavier full-feature models.
    """

    features = set(fitted.feature_columns)

    candidates = [
        # Main spatial-block story: lagged reporting dynamics.
        (
            "reporting_history__total_311_count_non_water_drainage_lag_1",
            "requests_history__requests_total_lag_1",
        ),
        (
            "reporting_history__total_311_count_non_water_drainage_roll3_mean_shift1",
            "requests_history__requests_total_roll3_mean_shift1",
        ),
        (
            "reporting_history__total_311_count_non_water_drainage_lag_12",
            "requests_history__requests_total_lag_12",
        ),
        (
            "reporting_history__total_311_count_non_water_drainage_lag_1",
            "reporting_history__total_311_count_non_water_drainage_roll3_mean_shift1",
        ),
        (
            "requests_history__requests_total_lag_1",
            "requests_history__requests_total_roll3_mean_shift1",
        ),
        # Target-history and static/SVI variants, useful if the selected model changes.
        (
            "target_history__water_drainage_count_lag_1",
            "reporting_history__total_311_count_non_water_drainage_lag_1",
        ),
        (
            "target_train_summary__mean",
            "reporting_history__total_311_count_non_water_drainage_lag_1",
        ),
        ("target_train_summary__mean", "svi_primary__svi_percentile"),
        ("svi_primary__svi_percentile", "static__log1p_population_density"),
        ("static_spatial__tract_centroid_lon", "static_spatial__tract_centroid_lat"),
        (
            "reporting_retro__log1p_total_311_count_non_water_drainage",
            "target_train_summary__mean",
        ),
    ]

    selected: list[tuple[str, str]] = []
    for a, b in candidates:
        if a in features and b in features:
            selected.append((a, b))

    return selected[: getattr(PlotConfig, "max_surface_pairs", 6)] if False else selected[:6]



def median_reference_row(frame: pd.DataFrame, fitted: base.FittedTabularModel) -> pd.DataFrame:
    """Build a one-row reference frame using medians for model features."""

    row = frame.iloc[[0]].copy()
    for col in fitted.feature_columns:
        values = safe_numeric(frame[col]) if col in frame.columns else pd.Series(dtype=float)
        if values.notna().any():
            row[col] = float(values.median(skipna=True))
        else:
            row[col] = float(fitted.feature_medians.get(col, 0.0))
    return row


def response_surface(
    frame: pd.DataFrame,
    fitted: base.FittedTabularModel,
    feature_x: str,
    feature_y: str,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute a 2D response surface from a median reference row."""

    if feature_x not in fitted.feature_columns or feature_y not in fitted.feature_columns:
        return None

    gx = feature_grid(frame[feature_x], grid_size)
    gy = feature_grid(frame[feature_y], grid_size)
    if len(gx) <= 1 or len(gy) <= 1:
        return None

    ref = median_reference_row(frame, fitted)
    rows: list[pd.DataFrame] = []

    for y in gy:
        block = pd.concat([ref] * len(gx), ignore_index=True)
        block[feature_x] = gx
        block[feature_y] = y
        rows.append(block)

    grid_frame = pd.concat(rows, ignore_index=True)
    preds = predict_with_fitted(grid_frame, fitted).to_numpy(dtype=float)
    Z = preds.reshape(len(gy), len(gx))
    X, Y = np.meshgrid(gx, gy)
    return X, Y, Z


def plot_surface_heatmap_and_3d(
    frame: pd.DataFrame,
    fitted: base.FittedTabularModel,
    feature_x: str,
    feature_y: str,
    plots_dir: Path,
    cfg: PlotConfig,
) -> list[str]:
    """Plot true PDP-style 2D heatmap and 3D response surface for two features.

    Unlike the earlier post-hoc empirical plots, these surfaces perturb two
    features over a grid while holding other features at a median reference row.
    They therefore visualize the fitted model's response, not merely binned
    observed averages.
    """

    result = response_surface(frame, fitted, feature_x, feature_y, cfg.surface_grid_size)
    if result is None:
        return []

    X, Y, Z = result
    paths: list[str] = []

    # 2D heatmap/contour: usually cleaner than 3D in papers.
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    im = ax.contourf(X, Y, Z, levels=28, cmap="viridis")
    contour = ax.contour(X, Y, Z, levels=10, colors="black", linewidths=0.35, alpha=0.35)
    ax.clabel(contour, inline=True, fontsize=7, fmt="%.2f")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("Predicted water/drainage count")
    ax.set_title(f"Model response heatmap — {clean_label(fitted.model_name, 58)}", pad=14)
    ax.set_xlabel(clean_label(feature_x, 62))
    ax.set_ylabel(clean_label(feature_y, 62))
    ax.grid(True, alpha=0.16)
    heatmap_path = plots_dir / (
        f"surface_heatmap__{safe_filename(fitted.model_name)}__"
        f"{safe_filename(feature_x)}__{safe_filename(feature_y)}.{cfg.plot_format}"
    )
    paths.append(savefig(heatmap_path, dpi=cfg.dpi))

    # 3D surface: larger canvas and explicit margins prevent clipped z-axis.
    fig = plt.figure(figsize=(12.8, 8.8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="viridis",
        linewidth=0.12,
        antialiased=True,
        alpha=0.96,
        edgecolor=(0, 0, 0, 0.10),
    )
    ax.contour(
        X,
        Y,
        Z,
        zdir="z",
        offset=float(np.nanmin(Z)),
        cmap="viridis",
        alpha=0.55,
        linewidths=0.7,
    )
    ax.set_title(f"3D model response surface — {clean_label(fitted.model_name, 64)}", pad=22)
    ax.set_xlabel(clean_label(feature_x, 48), labelpad=15)
    ax.set_ylabel(clean_label(feature_y, 48), labelpad=17)
    ax.set_zlabel("Predicted count", labelpad=16)
    ax.view_init(elev=28, azim=-132)
    try:
        ax.dist = 10  # type: ignore[attr-defined]
    except Exception:
        pass
    cbar = fig.colorbar(surf, ax=ax, shrink=0.62, aspect=18, pad=0.08)
    cbar.set_label("Predicted count")
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.04, top=0.90)
    surface_path = plots_dir / (
        f"surface_3d__{safe_filename(fitted.model_name)}__"
        f"{safe_filename(feature_x)}__{safe_filename(feature_y)}.{cfg.plot_format}"
    )
    plt.savefig(surface_path, dpi=cfg.dpi, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    paths.append(str(surface_path))

    return paths



def safe_filename(value: str, max_len: int = 110) -> str:
    """Make a safe filename stem."""

    keep: list[str] = []
    for ch in str(value):
        if ch.isalnum() or ch in {"_", "-", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    if len(out) > max_len:
        out = out[:max_len].rstrip("_")
    return out or "unnamed"


def generate_plots(
    *,
    frame: pd.DataFrame,
    predictions_long: pd.DataFrame,
    metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    model_selection_audit: pd.DataFrame,
    feature_set_audit: pd.DataFrame,
    fitted_models: Mapping[str, base.FittedTabularModel],
    split_scheme: str,
    plots_dir: Path,
    cfg: PlotConfig,
) -> dict[str, str]:
    """Generate all A3 model-behavior plots."""

    if not cfg.enabled:
        return {}

    require_plotting_dependencies()
    ensure_dir(plots_dir)

    plot_paths: dict[str, str] = {}

    maybe = plot_validation_vs_test_mae(model_selection_audit, plots_dir, cfg)
    if maybe:
        plot_paths["validation_vs_test_mae"] = maybe

    maybe = plot_selected_test_mae(model_selection_audit, plots_dir, cfg)
    if maybe:
        plot_paths["selected_test_mae"] = maybe

    refs = selected_reference_models(model_selection_audit, feature_set_audit)
    for ref_label, model_name in refs.items():
        fitted = fitted_models.get(model_name)
        if fitted is None:
            continue

        maybe = plot_feature_importance(feature_importance, model_name, plots_dir, cfg)
        if maybe:
            plot_paths[f"{ref_label}_feature_importance"] = maybe

        maybe = plot_observed_predicted_calibration(predictions_long, model_name, plots_dir, cfg, split_scheme)
        if maybe:
            plot_paths[f"{ref_label}_calibration"] = maybe

        maybe = plot_monthly_aggregate(predictions_long, model_name, plots_dir, cfg, split_scheme)
        if maybe:
            plot_paths[f"{ref_label}_monthly_aggregate"] = maybe

        maybe = plot_residual_by_decile(predictions_long, model_name, plots_dir, cfg, split_scheme)
        if maybe:
            plot_paths[f"{ref_label}_residual_deciles"] = maybe

    # Deep model-behavior plots for the validation-selected strict spatial-block model.
    # This usually resolves to A3_lagged_reporting_forecasting / HGB Poisson.
    reference_name = (
        refs.get("selected_strict_forecasting")
        or refs.get("lagged_reporting_hgb_focus")
        or refs.get("best_strict_primary")
        or refs.get("best_strict_overall")
    )
    reference_model = fitted_models.get(reference_name or "")
    if reference_model is not None:
        test_split_col = split_column_for_scheme(split_scheme)
        plot_frame = frame[frame[test_split_col].astype(str) == "test"].copy()
        if plot_frame.empty:
            plot_frame = frame.copy()

        features = top_features_for_pdp(
            feature_importance,
            reference_model.model_name,
            reference_model,
            cfg.max_pdp_features,
        )
        for feature in features:
            maybe = plot_pdp_1d(plot_frame, reference_model, feature, plots_dir, cfg)
            if maybe:
                plot_paths[f"pdp_1d__{safe_filename(feature)}"] = maybe

        for i, (fx, fy) in enumerate(preferred_surface_pairs(reference_model)[: cfg.max_surface_pairs], start=1):
            paths = plot_surface_heatmap_and_3d(plot_frame, reference_model, fx, fy, plots_dir, cfg)
            for j, path in enumerate(paths, start=1):
                plot_paths[f"surface_{i}_{j}"] = path

    return plot_paths


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 80) -> str:
    """Render dataframe to Markdown with fallback."""

    if df.empty:
        return "_No rows._"

    display = df.head(max_rows).copy()
    try:
        return display.to_markdown(index=False)
    except Exception:
        return "```text\n" + display.to_string(index=False) + "\n```"


def render_plot_index(plot_paths: Mapping[str, str], output_dir: Path) -> str:
    """Render a Markdown plot index with embedded relative images."""

    lines: list[str] = []
    lines.append("# A3 model-behavior plots\n")
    lines.append(
        "These figures are generated by `a3_tabular_feature_parity_spatial_block_with_plots.py`. "
        "They are predictive-behavior diagnostics, not causal effect estimates.\n"
    )

    if not plot_paths:
        lines.append("_No plots were generated._\n")
        return "\n".join(lines)

    for label, path in plot_paths.items():
        p = Path(path)
        try:
            rel = p.relative_to(output_dir)
        except ValueError:
            rel = p
        lines.append(f"## {clean_label(label, 80)}\n")
        lines.append(f"![{label}]({rel.as_posix()})\n")
        lines.append(f"`{path}`\n")

    return "\n".join(lines)


def compact_selected_metrics(metrics: pd.DataFrame, model_selection_audit: pd.DataFrame) -> pd.DataFrame:
    """Return compact metrics for selected candidates."""

    if metrics.empty or model_selection_audit.empty:
        return pd.DataFrame()

    selected = set(
        model_selection_audit.loc[
            model_selection_audit["selected_for_test_summary"].astype(bool),
            "model_name",
        ].astype(str)
    )

    wanted = [
        "count__mae",
        "count__rmse",
        "count__mean_poisson_deviance",
        "ranking__spearman_corr",
        "ranking__ndcg_at_100",
        "ranking__top_10pct_overlap_rate",
    ]

    out = metrics[
        metrics["model_name"].astype(str).isin(selected)
        & metrics["metric_name"].astype(str).isin(wanted)
    ].copy()

    cols = [
        "split_name",
        "prediction_setting",
        "model_name",
        "model_family",
        "hyperparameter_id",
        "feature_set_name",
        "metric_name",
        "metric_value",
        "higher_is_better",
        "n_rows",
    ]
    return out[[col for col in cols if col in out.columns]].reset_index(drop=True)


def render_report(
    *,
    generated_at: str,
    split_scheme: str,
    row_counts: Mapping[str, Any],
    feature_set_audit: pd.DataFrame,
    model_selection_audit: pd.DataFrame,
    metrics: pd.DataFrame,
    feature_lineage: pd.DataFrame,
    feature_importance: pd.DataFrame,
    plot_paths: Mapping[str, str],
    outputs: Mapping[str, str],
    plot_index_path: Path | None,
) -> str:
    """Render the A3-with-plots report."""

    compact = compact_selected_metrics(metrics, model_selection_audit)
    refs = selected_reference_models(model_selection_audit, feature_set_audit)

    lines: list[str] = []
    lines.append("# A3 Spatial-Block Feature-Parity Tabular Baselines with Plots — Montréal 311 Water/Drainage v0\n")
    lines.append(f"Generated at: `{generated_at}`\n")
    lines.append(f"Split scheme: `{split_scheme}`\n")
    lines.append(f"Split type: `{split_type_for_scheme(split_scheme)}`\n")

    lines.append("## Purpose\n")
    lines.append(
        "This run keeps the original A3 output safe by writing to a separate stage folder. "
        "It reruns the A3 feature-parity tabular benchmark and adds richer model-behavior "
        "figures: calibration plots, residual diagnostics, partial dependence/ICE curves, "
        "2D heatmaps, and 3D response surfaces.\n"
    )

    lines.append("## Prediction-setting interpretation\n")
    lines.append(
        "- `forecasting_v0`: static/calendar/train-summary features only.\n"
        "- `rolling_observed_history_v0`: uses lagged/rolling observed history. This is valid for rolling monthly forecasting, not for forecasting the whole future horizon from the train endpoint.\n"
        "- `retrospective_explanatory_v0`: uses same-month non-water 311 reporting exposure and is not a forecasting baseline.\n"
    )

    lines.append("## Row counts\n")
    lines.append("| Partition | Rows |")
    lines.append("|---|---:|")
    for key in ["train", "validation", "test"]:
        lines.append(f"| `{key}` | {row_counts.get(key)} |")
    lines.append("")

    lines.append("## Reference models selected for plots\n")
    if refs:
        lines.append("| Role | Model |")
        lines.append("|---|---|")
        for key, value in refs.items():
            lines.append(f"| `{key}` | `{value}` |")
        lines.append("")
    else:
        lines.append("_No reference models were selected._\n")

    lines.append("## Feature sets\n")
    lines.append(dataframe_to_markdown(feature_set_audit, max_rows=30))
    lines.append("")

    lines.append("## Validation-only model selection audit\n")
    lines.append(
        "Selection uses validation MAE only. Test metrics are reported after selection and are not used for model choice.\n"
    )
    cols = [
        "model_name",
        "model_family",
        "feature_set_name",
        "prediction_setting",
        "hyperparameter_id",
        "validation_mae",
        "validation_spearman",
        "test_mae",
        "test_spearman",
        "selected_overall_strict_forecasting",
        "selected_overall_retrospective",
    ]
    lines.append(dataframe_to_markdown(model_selection_audit[[c for c in cols if c in model_selection_audit.columns]], max_rows=80))
    lines.append("")

    lines.append("## Compact metrics for selected candidates\n")
    lines.append(dataframe_to_markdown(compact, max_rows=220))
    lines.append("")

    lines.append("## Generated plots\n")
    if plot_index_path is not None:
        lines.append(f"Plot index: `{plot_index_path}`\n")
    if plot_paths:
        lines.append("| Plot | Path |")
        lines.append("|---|---|")
        for label, path in plot_paths.items():
            lines.append(f"| `{label}` | `{path}` |")
        lines.append("")
    else:
        lines.append("_No plots generated._\n")

    lines.append("## Feature-lineage audit preview\n")
    lineage_cols = [
        "feature_name",
        "feature_family",
        "source_column",
        "transformation",
        "prediction_setting_allowed",
        "uses_target_history",
        "uses_reporting_history",
        "uses_same_month_information",
        "lag_months",
        "rolling_window",
        "shift_applied",
        "is_strict_forecasting_safe",
    ]
    lineage_cols = [col for col in lineage_cols if col in feature_lineage.columns]
    lines.append(dataframe_to_markdown(feature_lineage[lineage_cols], max_rows=80))
    lines.append("")

    lines.append("## Feature importance preview\n")
    if feature_importance.empty:
        lines.append("_No feature importance available._\n")
    else:
        preview = feature_importance.copy()
        preview["absolute_importance"] = pd.to_numeric(preview["absolute_importance"], errors="coerce")
        preview = preview.sort_values("absolute_importance", ascending=False)
        cols = [
            "model_name",
            "model_family",
            "feature_set_name",
            "feature",
            "importance_type",
            "importance",
            "absolute_importance",
        ]
        lines.append(dataframe_to_markdown(preview[[c for c in cols if c in preview.columns]], max_rows=80))
        lines.append("")

    lines.append("## Output artifacts\n")
    lines.append("| Artifact | Path |")
    lines.append("|---|---|")
    for key, value in outputs.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")

    lines.append("## Interpretation warning\n")
    lines.append(
        "The PDP/ICE, heatmap, response-surface, and feature-importance figures show model behavior under controlled perturbations. "
        "They do not identify causal effects. Retrospective plots involving same-month non-water 311 activity are explanatory diagnostics, not forecasting claims.\n"
    )

    return "\n".join(lines)


def build_metadata(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    panel_path: Path,
    split_path: Path,
    outputs: Mapping[str, str],
    split_scheme: str,
    row_counts: Mapping[str, Any],
    model_summaries: Sequence[Mapping[str, Any]],
    feature_sets: Sequence[base.FeatureSetSpec],
    metrics: pd.DataFrame,
    generated_at: str,
    ridge_alphas: Sequence[float],
    include_sklearn_models: bool,
    include_random_forest: bool,
    hgb_grid: str,
    plot_config: PlotConfig,
    plot_paths: Mapping[str, str],
) -> dict[str, Any]:
    """Build metadata."""

    return to_jsonable(
        {
            "benchmark_id": str(config.get("benchmark_id", "mtl_311_water_v0")),
            "dataset_version": DATASET_VERSION_DEFAULT,
            "generated_at": generated_at,
            "config_path": str(config_path),
            "config_hash": config_hash(config),
            "panel_path": str(panel_path),
            "panel_sha256": file_hash(panel_path),
            "split_assignments_path": str(split_path),
            "split_assignments_sha256": file_hash(split_path),
            "model_stage": MODEL_STAGE,
            "model_name": "A3_feature_parity_tabular_spatial_block_with_plots_suite",
            "split_scheme": split_scheme,
            "split_type": split_type_for_scheme(split_scheme),
            "target_name": TARGET_COLUMN,
            "target_type": "count",
            "row_counts": row_counts,
            "ridge_alphas": list(ridge_alphas),
            "include_sklearn_models": include_sklearn_models,
            "include_random_forest": include_random_forest,
            "hgb_grid": hgb_grid,
            "random_seed": RANDOM_SEED,
            "sklearn_version": getattr(sklearn, "__version__", None) if sklearn is not None else None,
            "matplotlib_available": plt is not None,
            "plot_config": plot_config.__dict__,
            "plot_paths": dict(plot_paths),
            "prediction_settings": {
                "forecasting_v0": "static/calendar/train-summary features only",
                "rolling_observed_history_v0": "lagged and rolling observed-history features shifted by at least one month",
                "retrospective_explanatory_v0": "same-month non-water 311 reporting exposure allowed",
            },
            "feature_sets": [
                {
                    "name": spec.name,
                    "prediction_setting": spec.prediction_setting,
                    "n_features": len(spec.feature_columns),
                    "is_primary_feature_set": spec.is_primary_feature_set,
                    "feature_columns": spec.feature_columns,
                    "description": spec.description,
                }
                for spec in feature_sets
            ],
            "models": list(model_summaries),
            "metric_rows": int(len(metrics)),
            "outputs": dict(outputs),
            "notes": (
                "This is a safe augmented A3 run. It does not overwrite the original "
                "A3_feature_parity_tabular_spatial_block output folder."
            ),
        }
    )


def run_a3_tabular_feature_parity_spatial_block_with_plots(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    repo_root: str | Path | None = None,
    split_scheme: str = DEFAULT_SPLIT_SCHEME,
    ridge_alphas: Sequence[float] = base.DEFAULT_RIDGE_ALPHAS,
    include_sklearn_models: bool = True,
    include_random_forest: bool = True,
    include_diagnostic_svi_sets: bool = True,
    hgb_grid: str = "small",
    plot_config: PlotConfig = PlotConfig(),
) -> dict[str, Any]:
    """Run A3 in a separate output folder and generate model-behavior plots."""

    require_runtime_dependencies()
    ridge_alphas = base.parse_float_list(ridge_alphas)

    config, root, resolved_config_path, panel_path, split_path, frame = load_benchmark_frame(
        config_path=config_path,
        repo_root=repo_root,
    )

    frame = base.normalize_frame_for_a3(frame, split_scheme=split_scheme)
    frame, svi_specs, score_audit = build_svi_score_specs(frame)
    static_audit = validate_static_svi_scores(frame, svi_specs)

    row_counts = split_counts(frame, split_scheme=split_scheme)
    missing_required = [part for part in ["train", "validation", "test"] if row_counts.get(part, 0) <= 0]
    if missing_required:
        raise A3WithPlotsError(
            f"Split scheme {split_scheme!r} missing required partitions: {missing_required}. Counts: {row_counts}"
        )

    train = base.train_frame(frame, split_scheme=split_scheme)
    frame, feature_groups, feature_lineage = prepare_feature_frame_compat(
        frame,
        train,
        split_scheme,
        svi_specs,
    )

    feature_sets = base.build_feature_set_specs(
        feature_groups,
        include_diagnostic_svi_sets=include_diagnostic_svi_sets,
    )
    base.validate_feature_sets(feature_sets, feature_lineage)

    candidates = base.candidate_specs(
        ridge_alphas=ridge_alphas,
        include_sklearn_models=include_sklearn_models,
        include_random_forest=include_random_forest,
        hgb_grid=hgb_grid,
    )

    if not candidates:
        raise A3WithPlotsError("No model candidates available.")

    paths = get_baseline_paths(config, root, STAGE_SLUG)
    output_dir = paths.output_dir
    ensure_dir(output_dir)
    plots_dir = ensure_dir(output_dir / "plots")

    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    all_importances: list[pd.DataFrame] = []
    model_summaries: list[dict[str, Any]] = []
    fitted_models: dict[str, base.FittedTabularModel] = {}

    for feature_set in feature_sets:
        for candidate in candidates:
            fitted, pred = base.fit_candidate_model(frame, train, feature_set, candidate)
            fitted_models[fitted.model_name] = fitted

            pred_frame = prediction_frame_for_model(
                frame,
                split_scheme=split_scheme,
                fitted=fitted,
                predictions=pred,
            )
            metrics = evaluate_one_model(
                pred_frame,
                config=config,
                full_frame=frame,
                split_scheme=split_scheme,
                fitted=fitted,
            )
            importance = fitted.feature_importance_table()

            all_predictions.append(pred_frame)
            all_metrics.append(metrics)
            if not importance.empty:
                all_importances.append(importance)

            model_summaries.append(
                {
                    "model_name": fitted.model_name,
                    "model_family": fitted.model_family,
                    "hyperparameter_id": fitted.hyperparameter_id,
                    "feature_set_name": fitted.feature_set_name,
                    "prediction_setting": fitted.prediction_setting,
                    "n_features": len(fitted.feature_columns),
                    "feature_columns": fitted.feature_columns,
                    "parameters": fitted.parameters,
                    "standardize": fitted.standardize,
                    "target_transform": fitted.target_transform,
                    "inverse_transform": fitted.inverse_transform,
                }
            )

    predictions_long = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    metrics = base.standardize_a3_metric_schema(metrics)
    feature_importance = pd.concat(all_importances, ignore_index=True) if all_importances else pd.DataFrame()

    feature_set_audit = pd.DataFrame(base.feature_set_audit_rows(feature_sets, feature_lineage))
    model_selection_audit = base.build_model_selection_audit(model_summaries, metrics, split_scheme)

    # Generate plots after model selection so the reference model is validation-selected.
    plot_paths = generate_plots(
        frame=frame,
        predictions_long=predictions_long,
        metrics=metrics,
        feature_importance=feature_importance,
        model_selection_audit=model_selection_audit,
        feature_set_audit=feature_set_audit,
        fitted_models=fitted_models,
        split_scheme=split_scheme,
        plots_dir=plots_dir,
        cfg=plot_config,
    )

    # Output paths.
    feature_set_audit_path = output_dir / "feature_set_audit.csv"
    feature_lineage_audit_path = output_dir / "feature_lineage_audit.csv"
    feature_importance_path = output_dir / "feature_importance.csv"
    model_audit_path = output_dir / "model_audit.csv"
    model_selection_audit_path = output_dir / "model_selection_audit.csv"
    score_audit_path = output_dir / "svi_score_audit.csv"
    static_audit_path = output_dir / "svi_static_score_audit.csv"
    plot_index_path = output_dir / "plot_index.md"

    feature_set_audit.to_csv(feature_set_audit_path, index=False)
    feature_lineage.to_csv(feature_lineage_audit_path, index=False)
    feature_importance.to_csv(feature_importance_path, index=False)
    pd.DataFrame(model_summaries).to_csv(model_audit_path, index=False)
    model_selection_audit.to_csv(model_selection_audit_path, index=False)
    pd.DataFrame(score_audit).to_csv(score_audit_path, index=False)
    pd.DataFrame(static_audit).to_csv(static_audit_path, index=False)

    written_predictions = write_long_prediction_partitions(
        predictions_long,
        output_dir,
        split_scheme=split_scheme,
    )

    write_markdown(plot_index_path, render_plot_index(plot_paths, output_dir))

    generated_at = datetime.now(timezone.utc).isoformat()

    outputs = {
        "metrics": str(paths.metrics),
        "model_metadata": str(paths.model_metadata),
        "baseline_report": str(paths.baseline_report),
        "plot_index": str(plot_index_path),
        "plots_dir": str(plots_dir),
        "feature_set_audit": str(feature_set_audit_path),
        "feature_lineage_audit": str(feature_lineage_audit_path),
        "feature_importance": str(feature_importance_path),
        "model_audit": str(model_audit_path),
        "model_selection_audit": str(model_selection_audit_path),
        "svi_score_audit": str(score_audit_path),
        "svi_static_score_audit": str(static_audit_path),
        **written_predictions,
    }

    metadata = build_metadata(
        config=config,
        config_path=resolved_config_path,
        panel_path=panel_path,
        split_path=split_path,
        outputs=outputs,
        split_scheme=split_scheme,
        row_counts=row_counts,
        model_summaries=model_summaries,
        feature_sets=feature_sets,
        metrics=metrics,
        generated_at=generated_at,
        ridge_alphas=ridge_alphas,
        include_sklearn_models=include_sklearn_models,
        include_random_forest=include_random_forest,
        hgb_grid=hgb_grid,
        plot_config=plot_config,
        plot_paths=plot_paths,
    )

    report = render_report(
        generated_at=generated_at,
        split_scheme=split_scheme,
        row_counts=row_counts,
        feature_set_audit=feature_set_audit,
        model_selection_audit=model_selection_audit,
        metrics=metrics,
        feature_lineage=feature_lineage,
        feature_importance=feature_importance,
        plot_paths=plot_paths,
        outputs=outputs,
        plot_index_path=plot_index_path,
    )

    metrics.to_csv(paths.metrics, index=False)
    write_json(paths.model_metadata, metadata)
    write_markdown(paths.baseline_report, report)

    return {
        "status": "completed",
        "model_stage": MODEL_STAGE,
        "model_name": "A3_feature_parity_tabular_spatial_block_with_plots_suite",
        "split_scheme": split_scheme,
        "outputs": outputs,
        "row_counts": row_counts,
        "feature_set_count": len(feature_sets),
        "candidate_count": len(candidates),
        "model_count": len(model_summaries),
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(len(predictions_long)),
        "feature_importance_rows": int(len(feature_importance)),
        "plot_count": len(plot_paths),
        "plot_paths": plot_paths,
        "selected_models": (
            model_selection_audit.loc[
                model_selection_audit["selected_for_test_summary"].astype(bool),
                "model_name",
            ].tolist()
            if not model_selection_audit.empty else []
        ),
    }


def a3_with_plots_brief(result: Mapping[str, Any]) -> str:
    """Return concise run summary."""

    outputs = result.get("outputs", {})
    return (
        "A3 spatial-block feature-parity tabular baselines with plots completed.\n"
        f"Status: {result.get('status')}\n"
        f"Split scheme: {result.get('split_scheme')}\n"
        f"Feature sets: {result.get('feature_set_count')}\n"
        f"Candidates per feature set: {result.get('candidate_count')}\n"
        f"Models: {result.get('model_count')}\n"
        f"Metric rows: {result.get('metric_rows')}\n"
        f"Prediction rows: {result.get('prediction_rows')}\n"
        f"Feature importance rows: {result.get('feature_importance_rows')}\n"
        f"Plots: {result.get('plot_count')}\n"
        f"Metrics: {outputs.get('metrics')}\n"
        f"Report: {outputs.get('baseline_report')}\n"
        f"Plot index: {outputs.get('plot_index')}\n"
    )


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Run A3 feature-parity tabular baselines with model-behavior plots."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument(
        "--split-scheme",
        default=DEFAULT_SPLIT_SCHEME,
        choices=["spatial_block"],
    )
    parser.add_argument(
        "--ridge-alphas",
        default=",".join(str(x) for x in base.DEFAULT_RIDGE_ALPHAS),
        help="Comma-separated ridge alphas. Default: 0.1,1.0,10.0",
    )
    parser.add_argument("--no-sklearn-models", action="store_true")
    parser.add_argument("--no-random-forest", action="store_true")
    parser.add_argument("--no-diagnostic-svi-sets", action="store_true")
    parser.add_argument("--hgb-grid", default="small", choices=["small", "medium"])
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-pdp-background", type=int, default=750)
    parser.add_argument("--max-ice-lines", type=int, default=40)
    parser.add_argument("--pdp-grid-size", type=int, default=31)
    parser.add_argument("--max-pdp-features", type=int, default=6)
    parser.add_argument("--surface-grid-size", type=int, default=41)
    parser.add_argument("--max-surface-pairs", type=int, default=6)
    parser.add_argument("--sample-scatter-points", type=int, default=3000)
    parser.add_argument("--plot-dpi", type=int, default=170)
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf", "svg"])

    args = parser.parse_args()

    plot_config = PlotConfig(
        enabled=not args.no_plots,
        max_pdp_background=args.max_pdp_background,
        max_ice_lines=args.max_ice_lines,
        pdp_grid_size=args.pdp_grid_size,
        max_pdp_features=args.max_pdp_features,
        surface_grid_size=args.surface_grid_size,
        max_surface_pairs=args.max_surface_pairs,
        sample_scatter_points=args.sample_scatter_points,
        dpi=args.plot_dpi,
        plot_format=args.plot_format,
    )

    result = run_a3_tabular_feature_parity_spatial_block_with_plots(
        config_path=args.config,
        repo_root=args.repo_root,
        split_scheme=args.split_scheme,
        ridge_alphas=base.parse_float_list(args.ridge_alphas),
        include_sklearn_models=not args.no_sklearn_models,
        include_random_forest=not args.no_random_forest,
        include_diagnostic_svi_sets=not args.no_diagnostic_svi_sets,
        hgb_grid=args.hgb_grid,
        plot_config=plot_config,
    )

    print(a3_with_plots_brief(result).rstrip())
    print("\nWritten outputs:")
    for label, path in result.get("outputs", {}).items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "A3WithPlotsError",
    "DEFAULT_SPLIT_SCHEME",
    "MODEL_STAGE",
    "PlotConfig",
    "STAGE_SLUG",
    "a3_with_plots_brief",
    "generate_plots",
    "run_a3_tabular_feature_parity_spatial_block_with_plots",
]
