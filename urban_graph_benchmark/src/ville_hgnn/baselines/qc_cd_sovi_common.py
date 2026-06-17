#!/usr/bin/env python3
"""
Shared utilities for the Québec CD civil-security / SoVI benchmark.

Intended destination:
    urban_graph_benchmark/src/ville_hgnn/baselines/qc_cd_sovi_common.py

Purpose:
    - Shared benchmark paths.
    - Shared target/score constants.
    - Shared train/validation/test split helpers.
    - Shared metric functions for B0/B2/B3/B4.
    - Shared prediction evaluation and writing helpers.

This file intentionally contains no model-specific logic.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

OUTPUT_ROOT = Path("urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0")
DATASETS_DIR = OUTPUT_ROOT / "datasets"
BASELINES_DIR = OUTPUT_ROOT / "baselines"
COMPARISONS_DIR = OUTPUT_ROOT / "comparisons"
REPORTS_DIR = OUTPUT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_EXTERNAL_CIVIL_SECURITY_DIR = Path("data/external/quebec_civil_security_events")
DATA_EXTERNAL_PROCESSED_DIR = DATA_EXTERNAL_CIVIL_SECURITY_DIR / "processed"
DATA_EXTERNAL_AUDIT_DIR = DATA_EXTERNAL_CIVIL_SECURITY_DIR / "audits"

DEFAULT_CUMULATIVE_TARGETS_PATH = (
    DATA_EXTERNAL_PROCESSED_DIR / "cd_civil_security_sovi_validation_targets_cumulative.parquet"
)
DEFAULT_CD_YEAR_TARGETS_PATH = (
    DATA_EXTERNAL_PROCESSED_DIR / "cd_civil_security_sovi_validation_targets_cd_year.parquet"
)
DEFAULT_CD_MONTH_TARGETS_PATH = (
    DATA_EXTERNAL_PROCESSED_DIR / "cd_civil_security_sovi_validation_targets_cd_month.parquet"
)
DEFAULT_PANEL_PATH = DATASETS_DIR / "cd_month_panel.parquet"
DEFAULT_SOVI_STANDARD_OUTPUT_PATH = Path(
    "data/sovi_2021/output/"
    "sovi_like_quebec_cd_2021_38var_oriented_run/"
    "standard_output.csv"
)


# -----------------------------------------------------------------------------
# Column and target constants
# -----------------------------------------------------------------------------

CD_ID_COL = "cd_id_norm"
CD_NAME_COL = "cd_name"
ORIGIN_MONTH_COL = "origin_month"
TARGET_START_MONTH_COL = "target_start_month"
TARGET_END_MONTH_COL = "target_end_month"
PERIOD_MONTH_COL = "period_month"
SPLIT_COL = "split"

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"

SOVI_SCORE_COL = "score_normalized_0_1"
SOVI_RAW_SCORE_COL = "score_raw"
SOVI_RANK_COL = "rank"
SOVI_PERCENTILE_COL = "percentile"

PRIMARY_CUMULATIVE_TARGET = "event_count_2021_2025_all"
FORWARD_CUMULATIVE_TARGET = "event_count_2022_2025_all"
RECOMMENDED_PRIMARY_TARGET = PRIMARY_CUMULATIVE_TARGET
RECOMMENDED_PRIMARY_TARGET_GRANULARITY = "cd_cumulative_2021_2025"
RECOMMENDED_PRIMARY_PRECISION_FILTER = "all"

FORECAST_TARGET_NEXT_1M_ALL = "target_next_1m_all"
FORECAST_TARGET_NEXT_3M_ALL = "target_next_3m_all"
FORECAST_TARGET_NEXT_6M_ALL = "target_next_6m_all"
DEFAULT_FORECAST_TARGET = FORECAST_TARGET_NEXT_3M_ALL

HAZARD_GROUPS = [
    "flood_water",
    "land_ground",
    "weather_climate",
    "infrastructure",
    "wildfire",
    "hazmat_health_social",
    "transport_accident",
    "other",
    "unmapped",
]

PRECISION_FILTERS = ["all", "precise_or_very_precise", "very_precise"]
SEVERITY_TARGETS = ["moderate_or_worse", "important_or_extreme"]

CUMULATIVE_TARGET_COLUMNS = {
    "B1a_2021_2025_all": "event_count_2021_2025_all",
    "B1b_2022_2025_all": "event_count_2022_2025_all",
    "B1c_2021_2025_precise_or_very_precise": "event_count_2021_2025_precise_or_very_precise",
    "B1d_2021_2025_very_precise": "event_count_2021_2025_very_precise",
    "B1e_2021_2025_flood_water": "event_count_2021_2025_all_flood_water",
    "B1f_2021_2025_land_ground": "event_count_2021_2025_all_land_ground",
    "B1f_2021_2025_weather_climate": "event_count_2021_2025_all_weather_climate",
    "B1f_2021_2025_infrastructure": "event_count_2021_2025_all_infrastructure",
    "B1f_2021_2025_wildfire": "event_count_2021_2025_all_wildfire",
    "B1g_2021_2025_moderate_or_worse": "event_count_2021_2025_all_moderate_or_worse",
    "B1g_2021_2025_important_or_extreme": "event_count_2021_2025_all_important_or_extreme",
}

FORECAST_TARGET_COLUMNS = {
    "next_1m_all": FORECAST_TARGET_NEXT_1M_ALL,
    "next_3m_all": FORECAST_TARGET_NEXT_3M_ALL,
    "next_6m_all": FORECAST_TARGET_NEXT_6M_ALL,
}

METRIC_COLUMNS = [
    "mae",
    "rmse",
    "mean_poisson_deviance",
    "spearman",
    "kendall",
    "ndcg_at_10",
    "ndcg_at_25",
    "ndcg_at_50",
    "top10_overlap_rate",
    "top25_overlap_rate",
    "top50_overlap_rate",
    "top_5pct_overlap_rate",
    "top_10pct_overlap_rate",
]
HEADLINE_RANK_METRICS = [
    "spearman",
    "kendall",
    "ndcg_at_10",
    "ndcg_at_25",
    "top10_overlap_rate",
    "top25_overlap_rate",
]
REGRESSION_METRICS = ["mae", "rmse", "mean_poisson_deviance"]
TOP_KS = [10, 25, 50]
PCT_TOPS = {"top_5pct_overlap_rate": 0.05, "top_10pct_overlap_rate": 0.10}


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def baseline_output_dir(baseline_name: str) -> Path:
    return BASELINES_DIR / baseline_name


def dataset_path(filename: str) -> Path:
    return DATASETS_DIR / filename


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Period):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def write_metadata_json(metadata: Mapping[str, Any], path: str | Path) -> Path:
    out = ensure_parent(path)
    with out.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(metadata), f, indent=2, ensure_ascii=False)
    return out


def write_table(
    df: pd.DataFrame,
    path: str | Path,
    *,
    write_csv_copy: bool = True,
    index: bool = False,
) -> dict[str, str]:
    out = ensure_parent(path)
    suffix = out.suffix.lower()
    outputs: dict[str, str] = {}
    if suffix == ".parquet":
        df.to_parquet(out, index=index)
        outputs["parquet"] = str(out)
        if write_csv_copy:
            csv_path = out.with_suffix(".csv")
            df.to_csv(csv_path, index=index)
            outputs["csv"] = str(csv_path)
    elif suffix == ".csv":
        df.to_csv(out, index=index)
        outputs["csv"] = str(out)
    else:
        raise ValueError(f"Unsupported table output suffix: {out}")
    return outputs


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input table does not exist: {p}")
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(p)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    raise ValueError(f"Unsupported table suffix: {p}")


# -----------------------------------------------------------------------------
# Temporal split helpers
# -----------------------------------------------------------------------------

def parse_month_series(series: pd.Series) -> pd.Series:
    """Parse a month-like series into pandas Period[M]."""
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.dt.asfreq("M")
    parsed = pd.to_datetime(series.astype("string"), errors="coerce")
    return parsed.dt.to_period("M")


def month_to_period(value: str | pd.Timestamp | pd.Period) -> pd.Period:
    if isinstance(value, pd.Period):
        return value.asfreq("M")
    return pd.Period(pd.to_datetime(value), freq="M")


def add_target_window_columns(
    df: pd.DataFrame,
    *,
    origin_month_col: str = ORIGIN_MONTH_COL,
    horizon_months: int = 3,
    target_start_col: str = TARGET_START_MONTH_COL,
    target_end_col: str = TARGET_END_MONTH_COL,
) -> pd.DataFrame:
    """
    Add target-window columns for forecasting.

    For origin t and horizon 3, the target covers t+1, t+2, t+3.
    Splits should be based on target_end_col to avoid leakage.
    """
    out = df.copy()
    origin = parse_month_series(out[origin_month_col])
    out[target_start_col] = (origin + 1).astype(str)
    out[target_end_col] = (origin + horizon_months).astype(str)
    return out


def assign_temporal_split_by_target_end(
    df: pd.DataFrame,
    *,
    target_end_month_col: str = TARGET_END_MONTH_COL,
    train_end: str = "2023-12",
    validation_start: str = "2024-01",
    validation_end: str = "2024-12",
    test_start: str = "2025-01",
    test_end: str = "2025-12",
    split_col: str = SPLIT_COL,
    drop_outside: bool = False,
) -> pd.DataFrame:
    """Assign train/validation/test using the future target-window end month."""
    out = df.copy()
    target_end = parse_month_series(out[target_end_month_col])
    train_end_p = month_to_period(train_end)
    val_start_p = month_to_period(validation_start)
    val_end_p = month_to_period(validation_end)
    test_start_p = month_to_period(test_start)
    test_end_p = month_to_period(test_end)

    out[split_col] = pd.NA
    out.loc[target_end <= train_end_p, split_col] = TRAIN_SPLIT
    out.loc[(target_end >= val_start_p) & (target_end <= val_end_p), split_col] = VALIDATION_SPLIT
    out.loc[(target_end >= test_start_p) & (target_end <= test_end_p), split_col] = TEST_SPLIT
    if drop_outside:
        out = out[out[split_col].notna()].copy()
    return out


def split_counts(df: pd.DataFrame, split_col: str = SPLIT_COL) -> pd.DataFrame:
    if split_col not in df.columns:
        return pd.DataFrame(columns=[split_col, "n", "share"])
    out = df[split_col].astype("string").fillna("<MISSING>").value_counts(dropna=False)
    out = out.rename_axis(split_col).reset_index(name="n")
    out["share"] = out["n"] / len(df) if len(df) else np.nan
    return out


# -----------------------------------------------------------------------------
# Metric functions
# -----------------------------------------------------------------------------

def safe_pearson(x: pd.Series, y: pd.Series) -> float:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    valid = x_num.notna() & y_num.notna()
    if int(valid.sum()) < 3:
        return math.nan
    xv = x_num.loc[valid].to_numpy(dtype=float)
    yv = y_num.loc[valid].to_numpy(dtype=float)
    if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
        return math.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def spearman_corr(y_true: pd.Series, y_pred: pd.Series) -> float:
    return safe_pearson(
        pd.to_numeric(y_true, errors="coerce").rank(method="average"),
        pd.to_numeric(y_pred, errors="coerce").rank(method="average"),
    )


def kendall_tau_b(y_true: pd.Series, y_pred: pd.Series) -> float:
    x = pd.to_numeric(y_true, errors="coerce")
    y = pd.to_numeric(y_pred, errors="coerce")
    valid = x.notna() & y.notna()
    xv = x.loc[valid].to_numpy(dtype=float)
    yv = y.loc[valid].to_numpy(dtype=float)
    n = len(xv)
    if n < 3:
        return math.nan

    concordant = discordant = ties_x_only = ties_y_only = 0
    for i in range(n - 1):
        dx = xv[i] - xv[i + 1:]
        dy = yv[i] - yv[i + 1:]
        sx = np.sign(dx)
        sy = np.sign(dy)
        both = (sx != 0) & (sy != 0)
        concordant += int(np.sum((sx[both] * sy[both]) > 0))
        discordant += int(np.sum((sx[both] * sy[both]) < 0))
        ties_x_only += int(np.sum((sx == 0) & (sy != 0)))
        ties_y_only += int(np.sum((sx != 0) & (sy == 0)))

    denom = math.sqrt(
        (concordant + discordant + ties_x_only)
        * (concordant + discordant + ties_y_only)
    )
    return math.nan if denom == 0 else float((concordant - discordant) / denom)


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    yt = pd.to_numeric(y_true, errors="coerce")
    yp = pd.to_numeric(y_pred, errors="coerce")
    valid = yt.notna() & yp.notna()
    if not valid.any():
        return math.nan
    return float(np.mean(np.abs(yt.loc[valid].to_numpy(float) - yp.loc[valid].to_numpy(float))))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    yt = pd.to_numeric(y_true, errors="coerce")
    yp = pd.to_numeric(y_pred, errors="coerce")
    valid = yt.notna() & yp.notna()
    if not valid.any():
        return math.nan
    err = yt.loc[valid].to_numpy(float) - yp.loc[valid].to_numpy(float)
    return float(np.sqrt(np.mean(err ** 2)))


def mean_poisson_deviance(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-9) -> float:
    yt = pd.to_numeric(y_true, errors="coerce")
    yp = pd.to_numeric(y_pred, errors="coerce")
    valid = yt.notna() & yp.notna()
    if not valid.any():
        return math.nan
    y = np.clip(yt.loc[valid].to_numpy(float), 0, None)
    mu = np.clip(yp.loc[valid].to_numpy(float), eps, None)
    term = np.where(y == 0, 0.0, y * np.log(np.clip(y, eps, None) / mu))
    return float(np.mean(2.0 * (term - y + mu)))


def stable_descending_order(values: pd.Series, ids: pd.Series | None = None) -> list[int]:
    vals = pd.to_numeric(values, errors="coerce").reset_index(drop=True)
    ids_s = (
        pd.Series(np.arange(len(vals))).astype("string")
        if ids is None
        else pd.Series(ids).reset_index(drop=True).astype("string").fillna("")
    )
    tmp = pd.DataFrame({"_value": vals, "_id": ids_s, "_orig_index": np.arange(len(vals))})
    tmp = tmp.sort_values(["_value", "_id"], ascending=[False, True], na_position="last", kind="mergesort")
    return tmp["_orig_index"].tolist()


def _linear_dcg(relevance: np.ndarray) -> float:
    if len(relevance) == 0:
        return math.nan
    discounts = 1.0 / np.log2(np.arange(2, len(relevance) + 2))
    return float(np.sum(relevance * discounts))


def ndcg_at_k(y_true: pd.Series, y_score: pd.Series, *, k: int, ids: pd.Series | None = None) -> float:
    yt = pd.to_numeric(y_true, errors="coerce")
    ys = pd.to_numeric(y_score, errors="coerce")
    valid = yt.notna() & ys.notna()
    if not valid.any():
        return math.nan
    rel = yt.loc[valid].reset_index(drop=True).clip(lower=0)
    score = ys.loc[valid].reset_index(drop=True)
    id_valid = pd.Series(ids).loc[valid].reset_index(drop=True) if ids is not None else None
    if rel.sum() <= 0:
        return math.nan
    k_eff = min(int(k), len(rel))
    pred_order = stable_descending_order(score, id_valid)[:k_eff]
    ideal_order = stable_descending_order(rel, id_valid)[:k_eff]
    dcg = _linear_dcg(rel.iloc[pred_order].to_numpy(float))
    ideal = _linear_dcg(rel.iloc[ideal_order].to_numpy(float))
    return math.nan if ideal == 0 else float(dcg / ideal)


def top_overlap_rate(y_true: pd.Series, y_score: pd.Series, *, k: int, ids: pd.Series | None = None) -> float:
    yt = pd.to_numeric(y_true, errors="coerce")
    ys = pd.to_numeric(y_score, errors="coerce")
    valid = yt.notna() & ys.notna()
    if not valid.any():
        return math.nan
    true_v = yt.loc[valid].reset_index(drop=True)
    score_v = ys.loc[valid].reset_index(drop=True)
    ids_v = (
        pd.Series(np.arange(len(true_v))).astype("string")
        if ids is None
        else pd.Series(ids).loc[valid].reset_index(drop=True).astype("string")
    )
    k_eff = min(int(k), len(true_v))
    if k_eff <= 0:
        return math.nan
    pred_top = set(ids_v.iloc[stable_descending_order(score_v, ids_v)[:k_eff]].astype(str))
    true_top = set(ids_v.iloc[stable_descending_order(true_v, ids_v)[:k_eff]].astype(str))
    return float(len(pred_top & true_top) / k_eff)


def percentile_top_k(n: int, pct: float) -> int:
    return max(1, int(math.ceil(int(n) * float(pct))))


def evaluate_rank_metrics(y_true: pd.Series, y_score: pd.Series, *, ids: pd.Series | None = None) -> dict[str, Any]:
    yt = pd.to_numeric(y_true, errors="coerce")
    ys = pd.to_numeric(y_score, errors="coerce")
    valid = yt.notna() & ys.notna()
    n = int(valid.sum())
    out: dict[str, Any] = {
        "n": n,
        "target_total": float(yt.loc[valid].clip(lower=0).sum()) if n else 0.0,
        "target_nonzero_count": int((yt.loc[valid].fillna(0) > 0).sum()) if n else 0,
        "target_nonzero_rate": float((yt.loc[valid].fillna(0) > 0).mean()) if n else math.nan,
    }
    rank_cols = [
        "spearman", "kendall", "ndcg_at_10", "ndcg_at_25", "ndcg_at_50",
        "top10_overlap_rate", "top25_overlap_rate", "top50_overlap_rate",
        "top_5pct_overlap_rate", "top_10pct_overlap_rate",
    ]
    if n < 3 or yt.loc[valid].nunique(dropna=True) < 2 or ys.loc[valid].nunique(dropna=True) < 2:
        out.update({c: math.nan for c in rank_cols})
        out["top_5pct_overlap_rate_k"] = percentile_top_k(n, 0.05) if n else 0
        out["top_10pct_overlap_rate_k"] = percentile_top_k(n, 0.10) if n else 0
        return out
    out["spearman"] = spearman_corr(yt, ys)
    out["kendall"] = kendall_tau_b(yt, ys)
    for k in TOP_KS:
        out[f"ndcg_at_{k}"] = ndcg_at_k(yt, ys, k=k, ids=ids)
        out[f"top{k}_overlap_rate"] = top_overlap_rate(yt, ys, k=k, ids=ids)
    for metric_name, pct in PCT_TOPS.items():
        k = percentile_top_k(n, pct)
        out[metric_name] = top_overlap_rate(yt, ys, k=k, ids=ids)
        out[f"{metric_name}_k"] = k
    return out


def evaluate_predictions(df: pd.DataFrame, *, y_true_col: str, y_pred_col: str, id_col: str = CD_ID_COL) -> dict[str, Any]:
    if y_true_col not in df.columns:
        raise KeyError(f"Target column not found: {y_true_col}")
    if y_pred_col not in df.columns:
        raise KeyError(f"Prediction column not found: {y_pred_col}")
    y_true = pd.to_numeric(df[y_true_col], errors="coerce")
    y_pred = pd.to_numeric(df[y_pred_col], errors="coerce")
    ids = df[id_col].astype("string") if id_col in df.columns else None
    valid = y_true.notna() & y_pred.notna()
    out: dict[str, Any] = {
        "n": int(valid.sum()),
        "target_col": y_true_col,
        "prediction_col": y_pred_col,
        "target_sum": float(y_true.loc[valid].sum()) if valid.any() else 0.0,
        "prediction_sum": float(y_pred.loc[valid].sum()) if valid.any() else 0.0,
        "target_mean": float(y_true.loc[valid].mean()) if valid.any() else math.nan,
        "prediction_mean": float(y_pred.loc[valid].mean()) if valid.any() else math.nan,
        "target_nonzero_count": int((y_true.loc[valid].fillna(0) > 0).sum()) if valid.any() else 0,
        "target_nonzero_rate": float((y_true.loc[valid].fillna(0) > 0).mean()) if valid.any() else math.nan,
        "missing_target_count": int(y_true.isna().sum()),
        "missing_prediction_count": int(y_pred.isna().sum()),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mean_poisson_deviance": mean_poisson_deviance(y_true, y_pred),
    }
    out.update(evaluate_rank_metrics(y_true, y_pred, ids=ids))
    return out


def evaluate_predictions_by_split(
    df: pd.DataFrame,
    *,
    y_true_col: str,
    y_pred_col: str,
    split_col: str = SPLIT_COL,
    id_col: str = CD_ID_COL,
    include_all: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if split_col not in df.columns:
        row = evaluate_predictions(df, y_true_col=y_true_col, y_pred_col=y_pred_col, id_col=id_col)
        row["split"] = "all"
        return pd.DataFrame([row])
    ordered = [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT]
    observed = [s for s in ordered if s in set(df[split_col].dropna().astype(str))]
    extra = sorted(set(df[split_col].dropna().astype(str)) - set(ordered))
    for split in [*observed, *extra]:
        sub = df[df[split_col].astype("string").eq(split)].copy()
        if sub.empty:
            continue
        row = evaluate_predictions(sub, y_true_col=y_true_col, y_pred_col=y_pred_col, id_col=id_col)
        row["split"] = split
        rows.append(row)
    if include_all:
        row = evaluate_predictions(df, y_true_col=y_true_col, y_pred_col=y_pred_col, id_col=id_col)
        row["split"] = "all"
        rows.append(row)
    return pd.DataFrame(rows)


def rank_positions(values: pd.Series, *, ids: pd.Series | None = None, ascending: bool = False) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").reset_index(drop=True)
    ids_s = pd.Series(np.arange(len(vals))).astype("string") if ids is None else pd.Series(ids).reset_index(drop=True).astype("string").fillna("")
    tmp = pd.DataFrame({"_value": vals, "_id": ids_s, "_orig_index": np.arange(len(vals))})
    tmp = tmp.sort_values(["_value", "_id"], ascending=[ascending, True], na_position="last", kind="mergesort")
    ranks = pd.Series(index=np.arange(len(vals)), dtype="Int64")
    for pos, idx in enumerate(tmp["_orig_index"].tolist(), start=1):
        ranks.iloc[idx] = pos
    return ranks.astype("Int64")


# -----------------------------------------------------------------------------
# Prediction-output helpers
# -----------------------------------------------------------------------------

def standard_prediction_columns() -> list[str]:
    return [
        CD_ID_COL,
        CD_NAME_COL,
        ORIGIN_MONTH_COL,
        TARGET_START_MONTH_COL,
        TARGET_END_MONTH_COL,
        SPLIT_COL,
        "model_name",
        "target_col",
        "target",
        "prediction",
    ]


def build_prediction_frame(
    df: pd.DataFrame,
    *,
    prediction: Sequence[float] | pd.Series | np.ndarray,
    target_col: str,
    model_name: str,
    id_col: str = CD_ID_COL,
    name_col: str = CD_NAME_COL,
    origin_month_col: str = ORIGIN_MONTH_COL,
    target_start_col: str = TARGET_START_MONTH_COL,
    target_end_col: str = TARGET_END_MONTH_COL,
    split_col: str = SPLIT_COL,
    extra_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    col_map = {
        id_col: CD_ID_COL,
        name_col: CD_NAME_COL,
        origin_month_col: ORIGIN_MONTH_COL,
        target_start_col: TARGET_START_MONTH_COL,
        target_end_col: TARGET_END_MONTH_COL,
        split_col: SPLIT_COL,
    }
    for source_col, dest_col in col_map.items():
        if source_col in df.columns and dest_col not in out.columns:
            out[dest_col] = df[source_col].values
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")
    out["model_name"] = model_name
    out["target_col"] = target_col
    out["target"] = pd.to_numeric(df[target_col], errors="coerce").values
    out["prediction"] = np.asarray(prediction, dtype=float)
    if extra_cols:
        for col in extra_cols:
            if col in df.columns and col not in out.columns:
                out[col] = df[col].values
    return out.reset_index(drop=True)


def evaluate_standard_prediction_frame(
    predictions: pd.DataFrame,
    *,
    target_col: str = "target",
    prediction_col: str = "prediction",
    split_col: str = SPLIT_COL,
    id_col: str = CD_ID_COL,
) -> pd.DataFrame:
    group_cols = [col for col in ["model_name", "target_col"] if col in predictions.columns]
    if not group_cols:
        return evaluate_predictions_by_split(
            predictions,
            y_true_col=target_col,
            y_pred_col=prediction_col,
            split_col=split_col,
            id_col=id_col,
            include_all=True,
        )
    rows = []
    for key, sub in predictions.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        metrics = evaluate_predictions_by_split(
            sub,
            y_true_col=target_col,
            y_pred_col=prediction_col,
            split_col=split_col,
            id_col=id_col,
            include_all=True,
        )
        for col, value in zip(group_cols, key):
            metrics[col] = value
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_baseline_outputs(
    *,
    output_dir: str | Path,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame | None = None,
    metadata: Mapping[str, Any] | None = None,
    prefix: str = "",
) -> dict[str, str]:
    out_dir = ensure_dir(output_dir)
    prefix_str = f"{prefix}_" if prefix else ""
    outputs: dict[str, str] = {}
    pred_paths = write_table(predictions, out_dir / f"{prefix_str}predictions.parquet", write_csv_copy=True, index=False)
    outputs.update({f"predictions_{k}": v for k, v in pred_paths.items()})
    if metrics is not None:
        metrics_path = out_dir / f"{prefix_str}metrics.csv"
        metrics.to_csv(metrics_path, index=False)
        outputs["metrics_csv"] = str(metrics_path)
    if metadata is not None:
        metadata_path = write_metadata_json(metadata, out_dir / f"{prefix_str}metadata.json")
        outputs["metadata_json"] = str(metadata_path)
    return outputs
