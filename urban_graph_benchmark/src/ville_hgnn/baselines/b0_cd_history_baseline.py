#!/usr/bin/env python3
"""
B0 history-only baselines for the Québec CD civil-security / SoVI benchmark.

This module implements naive predictors that use only each census division's past
civil-security event burden. It intentionally does not use SoVI scores, SoVI
variables, tabular ML, or graph structure.

Expected input:
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/datasets/cd_month_panel.parquet

Default output directory:
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B0_history_only/

Public API:
    run_b0_cd_history_baseline(config: Config) -> dict[str, Any]
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ville_hgnn.baselines.qc_cd_sovi_common import (
    BASELINES_DIR,
    CD_ID_COL,
    CD_NAME_COL,
    DEFAULT_PANEL_PATH,
    SPLIT_COL,
    ensure_dir,
    evaluate_standard_prediction_frame,
    write_metadata_json,
    write_table,
)


DEFAULT_OUTPUT_DIR = BASELINES_DIR / "B0_history_only"

DEFAULT_MODELS = [
    "previous_month",
    "rolling_3_months",
    "rolling_6_months",
    "rolling_12_months",
    "seasonal_historical_mean",
]

MODEL_TO_FEATURE = {
    "previous_month": "lag_1",
    "rolling_3_months": "rolling_3",
    "rolling_6_months": "rolling_6",
    "rolling_12_months": "rolling_12",
}


@dataclass
class Config:
    """Configuration for B0 history-only baselines."""

    panel_path: Path = DEFAULT_PANEL_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR

    target_col: str = "target_next_3_months"
    current_count_col: str = "event_count_current_month_all"

    cd_id_col: str = CD_ID_COL
    cd_name_col: str = CD_NAME_COL
    period_month_col: str = "period_month"
    split_col: str = SPLIT_COL

    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))

    drop_missing_target: bool = True
    clip_predictions_at_zero: bool = True
    fallback_to_train_global_mean: bool = True


def read_panel(path: Path) -> pd.DataFrame:
    """Read the predictive panel from parquet or csv."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Panel file does not exist: {path}")

    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported panel suffix: {path}")


def parse_period_month(series: pd.Series) -> pd.Series:
    """Parse month-like values to pandas Period[M]."""
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.dt.asfreq("M")

    parsed = pd.to_datetime(series.astype("string"), errors="coerce")
    if parsed.isna().any():
        bad = series.loc[parsed.isna()].head(10).tolist()
        raise ValueError(f"Could not parse some period_month values: {bad}")

    return parsed.dt.to_period("M")


def validate_config(config: Config) -> None:
    """Validate config before running."""
    config.panel_path = Path(config.panel_path)
    config.output_dir = Path(config.output_dir)

    if not config.models:
        raise ValueError("At least one B0 model must be requested.")

    unknown = sorted(set(config.models) - set(DEFAULT_MODELS))
    if unknown:
        raise ValueError(f"Unknown B0 model(s): {unknown}. Allowed: {DEFAULT_MODELS}")


def validate_panel_columns(df: pd.DataFrame, config: Config) -> None:
    """Validate required panel columns."""
    required = [
        config.cd_id_col,
        config.period_month_col,
        config.target_col,
        config.current_count_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Panel is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def add_history_features_if_missing(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Ensure lag_1 / rolling_3 / rolling_6 / rolling_12 exist.

    The panel builder normally creates them. This keeps B0 robust if a panel is
    regenerated with a smaller schema.
    """
    out = df.copy()
    out["_period_for_sort"] = parse_period_month(out[config.period_month_col])
    out = out.sort_values([config.cd_id_col, "_period_for_sort"]).reset_index(drop=True)

    out[config.current_count_col] = pd.to_numeric(
        out[config.current_count_col], errors="coerce"
    ).fillna(0.0)

    grouped = out.groupby(config.cd_id_col, sort=False)[config.current_count_col]

    if "lag_1" not in out.columns:
        out["lag_1"] = grouped.shift(1).fillna(0.0)

    for col, window in {"rolling_3": 3, "rolling_6": 6, "rolling_12": 12}.items():
        if col not in out.columns:
            out[col] = grouped.transform(
                lambda s, w=window: s.rolling(window=w, min_periods=1).sum()
            )

    for col in ["lag_1", "rolling_3", "rolling_6", "rolling_12"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out.drop(columns=["_period_for_sort"])


def add_global_fallbacks(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add global train means used only for cold-start fallback predictions."""
    out = df.copy()

    if config.split_col in out.columns:
        train_mask = out[config.split_col].astype("string").eq("train")
    else:
        train_mask = pd.Series(True, index=out.index)

    train_target = pd.to_numeric(out.loc[train_mask, config.target_col], errors="coerce")
    train_current = pd.to_numeric(
        out.loc[train_mask, config.current_count_col], errors="coerce"
    )

    out["_global_train_target_mean"] = (
        float(train_target.mean()) if train_target.notna().any() else 0.0
    )
    out["_global_train_current_mean"] = (
        float(train_current.mean()) if train_current.notna().any() else 0.0
    )
    return out


def add_seasonal_historical_mean(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Add leakage-safe seasonal historical mean predictions.

    For each CD and calendar month, prediction at origin t is the expanding mean
    of previous years' observed target values for the same origin calendar month.
    For example, 2024-03 uses 2021-03, 2022-03, and 2023-03 targets for that CD.
    """
    out = df.copy()
    out["_period_for_sort"] = parse_period_month(out[config.period_month_col])
    out["_calendar_month"] = out["_period_for_sort"].dt.month.astype(int)
    out["_target_numeric"] = pd.to_numeric(out[config.target_col], errors="coerce")

    out = out.sort_values(
        [config.cd_id_col, "_calendar_month", "_period_for_sort"]
    ).reset_index(drop=True)

    def prior_expanding_mean(s: pd.Series) -> pd.Series:
        return s.shift(1).expanding(min_periods=1).mean()

    out["pred_seasonal_historical_mean"] = (
        out.groupby([config.cd_id_col, "_calendar_month"], sort=False)["_target_numeric"]
        .transform(prior_expanding_mean)
    )

    if config.fallback_to_train_global_mean:
        out["pred_seasonal_historical_mean"] = out[
            "pred_seasonal_historical_mean"
        ].fillna(out["_global_train_target_mean"])

    return out.drop(columns=["_period_for_sort", "_calendar_month", "_target_numeric"])


def add_predictions(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add one pred_<model_name> column per requested model."""
    out = df.copy()

    for model_name, feature_col in MODEL_TO_FEATURE.items():
        if model_name not in config.models:
            continue
        if feature_col not in out.columns:
            raise KeyError(f"Model {model_name!r} needs missing feature {feature_col!r}")

        pred_col = f"pred_{model_name}"
        out[pred_col] = pd.to_numeric(out[feature_col], errors="coerce")

        if config.fallback_to_train_global_mean:
            out[pred_col] = out[pred_col].fillna(out["_global_train_current_mean"])

    if "seasonal_historical_mean" in config.models:
        out = add_seasonal_historical_mean(out, config)

    if config.clip_predictions_at_zero:
        for model_name in config.models:
            pred_col = f"pred_{model_name}"
            if pred_col in out.columns:
                out[pred_col] = pd.to_numeric(out[pred_col], errors="coerce").clip(lower=0)

    return out


def prediction_frame_for_model(df: pd.DataFrame, model_name: str, config: Config) -> pd.DataFrame:
    """Create a standard long-format prediction frame for one model."""
    pred_col = f"pred_{model_name}"
    if pred_col not in df.columns:
        raise KeyError(f"Missing prediction column: {pred_col}")

    data: dict[str, Any] = {
        CD_ID_COL: df[config.cd_id_col].astype("string"),
        "model_name": model_name,
        "target_col": config.target_col,
        "target": pd.to_numeric(df[config.target_col], errors="coerce"),
        "prediction": pd.to_numeric(df[pred_col], errors="coerce"),
    }

    if config.cd_name_col in df.columns:
        data[CD_NAME_COL] = df[config.cd_name_col].astype("string")
    if config.period_month_col in df.columns:
        data["period_month"] = df[config.period_month_col].astype("string")
    if config.split_col in df.columns:
        data[SPLIT_COL] = df[config.split_col].astype("string")
    if "year" in df.columns:
        data["year"] = df["year"]
    if "month" in df.columns:
        data["month"] = df["month"]

    return pd.DataFrame(data)


def build_predictions(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Build a standard prediction table for all requested models."""
    frames = [prediction_frame_for_model(df, m, config) for m in config.models]
    predictions = pd.concat(frames, ignore_index=True)

    if config.drop_missing_target:
        predictions = predictions[predictions["target"].notna()].copy()

    return predictions.reset_index(drop=True)


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compact audit summary by model and split."""
    group_cols = ["model_name"]
    if SPLIT_COL in predictions.columns:
        group_cols.append(SPLIT_COL)

    rows: list[dict[str, Any]] = []
    for key, sub in predictions.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key)}
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


def select_model_by_validation(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select models using validation MAE, with deterministic tie-breakers."""
    if metrics.empty:
        return pd.DataFrame()

    val_metrics = metrics[metrics["split"].astype("string").isin(["val", "validation"])].copy()
    if val_metrics.empty:
        val_metrics = metrics[metrics["split"].astype("string").eq("all")].copy()
    if val_metrics.empty:
        return pd.DataFrame()

    sort_cols: list[str] = []
    ascending: list[bool] = []
    for col, asc in [
        ("mae", True),
        ("rmse", True),
        ("mean_poisson_deviance", True),
        ("spearman", False),
        ("ndcg_at_25", False),
    ]:
        if col in val_metrics.columns:
            sort_cols.append(col)
            ascending.append(asc)

    out = val_metrics.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    out.insert(0, "selection_rank", np.arange(1, len(out) + 1))
    out["selection_rule"] = "primary=validation_mae_min; ties=rmse/deviance/rank_metrics"
    return out


def run_b0_cd_history_baseline(config: Config) -> dict[str, Any]:
    """Run B0 history-only baselines and write standardized outputs."""
    validate_config(config)
    output_dir = ensure_dir(config.output_dir)

    panel = read_panel(config.panel_path)
    validate_panel_columns(panel, config)

    working = add_history_features_if_missing(panel, config)
    working = add_global_fallbacks(working, config)
    working = add_predictions(working, config)
    predictions = build_predictions(working, config)

    metrics = evaluate_standard_prediction_frame(
        predictions,
        target_col="target",
        prediction_col="prediction",
        split_col=SPLIT_COL,
        id_col=CD_ID_COL,
    )
    model_selection = select_model_by_validation(metrics)
    prediction_summary = summarize_predictions(predictions)

    written_predictions = write_table(
        predictions,
        output_dir / "predictions.parquet",
        write_csv_copy=True,
        index=False,
    )

    metrics_path = output_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    model_selection_path = output_dir / "model_selection.csv"
    model_selection.to_csv(model_selection_path, index=False)

    prediction_summary_path = output_dir / "prediction_summary.csv"
    prediction_summary.to_csv(prediction_summary_path, index=False)

    split_outputs: dict[str, str] = {}
    if SPLIT_COL in predictions.columns:
        for split_name in ["train", "val", "validation", "test"]:
            sub = predictions[predictions[SPLIT_COL].astype("string").eq(split_name)]
            if sub.empty:
                continue
            canonical = "validation" if split_name == "val" else split_name
            split_written = write_table(
                sub,
                output_dir / f"predictions_{canonical}.parquet",
                write_csv_copy=True,
                index=False,
            )
            for kind, path in split_written.items():
                split_outputs[f"predictions_{canonical}_{kind}"] = path

    metadata: dict[str, Any] = {
        "baseline_family": "B0_history_only",
        "module": "ville_hgnn.baselines.b0_cd_history_baseline",
        "purpose": "Naive history-only predictors for Québec CD civil-security event burden.",
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
        "prediction_rows": int(len(predictions)),
        "outputs": {
            **{f"predictions_{k}": v for k, v in written_predictions.items()},
            "metrics_csv": str(metrics_path),
            "model_selection_csv": str(model_selection_path),
            "prediction_summary_csv": str(prediction_summary_path),
            **split_outputs,
        },
    }

    if not model_selection.empty:
        best = model_selection.iloc[0].to_dict()
        metadata["selected_model"] = {
            "model_name": best.get("model_name"),
            "selection_split": best.get("split"),
            "mae": best.get("mae"),
            "rmse": best.get("rmse"),
            "spearman": best.get("spearman"),
            "ndcg_at_25": best.get("ndcg_at_25"),
        }

    metadata_path = write_metadata_json(metadata, output_dir / "metadata.json")
    metadata["outputs"]["metadata_json"] = str(metadata_path)

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run B0 history-only baselines for the Québec CD civil-security / SoVI benchmark."
    )
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-col", default="target_next_3_months")
    parser.add_argument("--current-count-col", default="event_count_current_month_all")
    parser.add_argument("--cd-id-col", default=CD_ID_COL)
    parser.add_argument("--cd-name-col", default=CD_NAME_COL)
    parser.add_argument("--period-month-col", default="period_month")
    parser.add_argument("--split-col", default=SPLIT_COL)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=list(DEFAULT_MODELS))
    parser.add_argument(
        "--keep-missing-targets",
        action="store_true",
        help="Keep missing target rows in prediction outputs. Metrics ignore them.",
    )
    parser.add_argument("--no-clip-predictions", action="store_true")
    parser.add_argument("--no-global-fallback", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        panel_path=args.panel_path,
        output_dir=args.output_dir,
        target_col=args.target_col,
        current_count_col=args.current_count_col,
        cd_id_col=args.cd_id_col,
        cd_name_col=args.cd_name_col,
        period_month_col=args.period_month_col,
        split_col=args.split_col,
        models=list(args.models),
        drop_missing_target=not args.keep_missing_targets,
        clip_predictions_at_zero=not args.no_clip_predictions,
        fallback_to_train_global_mean=not args.no_global_fallback,
    )


def main() -> None:
    config = config_from_args(parse_args())
    metadata = run_b0_cd_history_baseline(config)

    print("B0 history-only baseline completed.")
    print(f"Panel: {config.panel_path}")
    print(f"Output directory: {config.output_dir}")
    print("Models:")
    for model in config.models:
        print(f"  - {model}")

    selected = metadata.get("selected_model")
    if selected:
        print("Selected model by validation MAE:")
        print(f"  model_name: {selected.get('model_name')}")
        print(f"  split: {selected.get('selection_split')}")
        print(f"  MAE: {selected.get('mae')}")
        print(f"  RMSE: {selected.get('rmse')}")
        print(f"  Spearman: {selected.get('spearman')}")

    print("Outputs:")
    for key, value in metadata["outputs"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
