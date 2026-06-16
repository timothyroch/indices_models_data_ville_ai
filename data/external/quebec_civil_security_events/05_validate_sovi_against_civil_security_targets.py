#!/usr/bin/env python3
"""
Direct external-validation benchmark: SoVI score vs Québec civil-security event burden.

Scientific question:
    Do Québec CDs with higher SoVI-like vulnerability scores tend to rank higher
    in observed civil-security event burden, and is this relationship stable
    across target definitions, hazard groups, precision filters, and severity
    thresholds?

Scope:
    - Direct/static SoVI validation only.
    - No supervised ML.
    - No graph model.
    - No calibration model.
    - No tuning of SoVI using the civil-security target.

Main input:
    data/external/quebec_civil_security_events/processed/
    cd_civil_security_sovi_validation_targets_cumulative.parquet

Default output directory:
    data/external/quebec_civil_security_events/benchmarks/sovi_civil_security_validation/

Outputs:
    sovi_civil_security_validation_metrics.csv
    sovi_civil_security_validation_rankings.csv
    sovi_civil_security_validation_target_summary.csv
    sovi_civil_security_validation_interpretation.md
    sovi_civil_security_validation_metadata.json

Typical run:
    python data/external/quebec_civil_security_events/05_validate_sovi_against_civil_security_targets.py

If the SoVI score column is not detected automatically:
    python data/external/quebec_civil_security_events/05_validate_sovi_against_civil_security_targets.py \
      --sovi-score-col sovi_score_normalized_0_1
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_TARGETS_PATH = Path(
    "data/external/quebec_civil_security_events/processed/"
    "cd_civil_security_sovi_validation_targets_cumulative.parquet"
)

DEFAULT_OUTPUT_DIR = Path(
    "data/external/quebec_civil_security_events/benchmarks/"
    "sovi_civil_security_validation"
)

HEADLINE_METRICS = [
    "spearman",
    "kendall",
    "ndcg_at_10",
    "ndcg_at_25",
    "top10_overlap_rate",
    "top25_overlap_rate",
]

ALL_METRICS = [
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

TOP_KS = [10, 25, 50]

PCT_TOPS = {
    "top_5pct_overlap_rate": 0.05,
    "top_10pct_overlap_rate": 0.10,
}

SOVI_SCORE_CANDIDATES = [
    "sovi_score_normalized_0_1",
    "sovi_score_normalized",
    "sovi_score_oriented_normalized_0_1",
    "sovi_oriented_normalized_0_1",
    "sovi_index_normalized_0_1",
    "sovi_index_score",
    "sovi_score",
    "score_normalized_0_1",
    "score_normalized",
    "index_score_normalized_0_1",
    "index_score",
    "score",
]

ID_CANDIDATES = [
    "cd_id_norm",
    "zone_id",
    "census_division_dguid",
    "cd_id",
    "cd_uid",
    "CDUID",
    "DGUID",
    "dguid",
]

NAME_CANDIDATES = [
    "cd_name",
    "CDNAME",
    "name",
    "geography_name",
    "GEO_NAME",
    "zone_name",
]

TARGET_DEFINITIONS = [
    {
        "target_id": "B1a",
        "target_label": "2021–2025 all events",
        "target_column": "event_count_2021_2025_all",
        "target_family": "main_cumulative",
        "temporal_interpretation": "descriptive_near_snapshot_validation",
        "is_primary": True,
        "interpretation_note": "Main descriptive CD-level validation target aligned to the static 2021 SoVI frame.",
    },
    {
        "target_id": "B1b",
        "target_label": "2022–2025 all events",
        "target_column": "event_count_2022_2025_all",
        "target_family": "forward_cumulative",
        "temporal_interpretation": "cleaner_forward_looking_validation",
        "is_primary": False,
        "interpretation_note": "Forward-looking robustness target after the mostly-2021 SoVI construction year.",
    },
    {
        "target_id": "B1c",
        "target_label": "2021–2025 precise or very precise events",
        "target_column": "event_count_2021_2025_precise_or_very_precise",
        "target_family": "precision_filter",
        "temporal_interpretation": "descriptive_near_snapshot_validation",
        "is_primary": False,
        "interpretation_note": "Localization-precision robustness target.",
    },
    {
        "target_id": "B1d",
        "target_label": "2021–2025 very precise events",
        "target_column": "event_count_2021_2025_very_precise",
        "target_family": "precision_filter",
        "temporal_interpretation": "descriptive_near_snapshot_validation",
        "is_primary": False,
        "interpretation_note": "Strict localization-precision robustness target.",
    },
    {
        "target_id": "B1e",
        "target_label": "2021–2025 flood/water events",
        "target_column": "event_count_2021_2025_all_flood_water",
        "target_family": "hazard_group",
        "temporal_interpretation": "descriptive_near_snapshot_validation",
        "is_primary": False,
        "interpretation_note": "Hazard-specific target for flood and water-control events.",
    },
    {
        "target_id": "B1f_land_ground",
        "target_label": "2021–2025 land/ground events",
        "target_column": "event_count_2021_2025_all_land_ground",
        "target_family": "hazard_group",
        "temporal_interpretation": "descriptive_near_snapshot_validation",
        "is_primary": False,
        "interpretation_note": "Hazard-group robustness target.",
    },
    {
        "target_id": "B1f_weather_climate",
        "target_label": "2021–2025 weather/climate events",
        "target_column": "event_count_2021_2025_all_weather_climate",
        "target_family": "hazard_group",
        "temporal_interpretation": "descriptive_near_snapshot_validation",
        "is_primary": False,
        "interpretation_note": "Hazard-group robustness target.",
    },
    {
        "target_id": "B1f_infrastructure",
        "target_label": "2021–2025 infrastructure events",
        "target_column": "event_count_2021_2025_all_infrastructure",
        "target_family": "hazard_group",
        "temporal_interpretation": "descriptive_near_snapshot_validation",
        "is_primary": False,
        "interpretation_note": "Hazard-group robustness target.",
    },
    {
        "target_id": "B1f_wildfire",
        "target_label": "2021–2025 wildfire events",
        "target_column": "event_count_2021_2025_all_wildfire",
        "target_family": "hazard_group",
        "temporal_interpretation": "descriptive_near_snapshot_validation",
        "is_primary": False,
        "interpretation_note": "Hazard-group robustness target.",
    },
    {
        "target_id": "B1g_moderate_or_worse",
        "target_label": "2021–2025 moderate-or-worse events",
        "target_column": "event_count_2021_2025_all_moderate_or_worse",
        "target_family": "severity_threshold",
        "temporal_interpretation": "descriptive_near_snapshot_validation",
        "is_primary": False,
        "interpretation_note": "Severity-threshold robustness target.",
    },
    {
        "target_id": "B1g_important_or_extreme",
        "target_label": "2021–2025 important-or-extreme events",
        "target_column": "event_count_2021_2025_all_important_or_extreme",
        "target_family": "severity_threshold",
        "temporal_interpretation": "descriptive_near_snapshot_validation",
        "is_primary": False,
        "interpretation_note": "Strict severity-threshold robustness target.",
    },
]


@dataclass(frozen=True)
class Config:
    targets_path: Path
    output_dir: Path
    sovi_score_col: str | None
    cd_id_col: str | None
    cd_name_col: str | None
    reverse_sovi_score: bool
    random_seed: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Validate static SoVI rankings against civil-security event-burden targets."
    )
    parser.add_argument("--targets-path", type=Path, default=DEFAULT_TARGETS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sovi-score-col", default=None)
    parser.add_argument("--cd-id-col", default=None)
    parser.add_argument("--cd-name-col", default=None)
    parser.add_argument(
        "--reverse-sovi-score",
        action="store_true",
        help="Use -SoVI score if the selected score is oriented with lower values as more vulnerable.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    return Config(
        targets_path=args.targets_path,
        output_dir=args.output_dir,
        sovi_score_col=args.sovi_score_col,
        cd_id_col=args.cd_id_col,
        cd_name_col=args.cd_name_col,
        reverse_sovi_score=bool(args.reverse_sovi_score),
        random_seed=int(args.random_seed),
    )


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input table does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported input table extension: {path}")


def normalize_col_name(col: str) -> str:
    return "".join(ch for ch in str(col).lower() if ch.isalnum())


def choose_first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = [str(c) for c in columns]
    exact = {c: c for c in cols}
    normalized = {normalize_col_name(c): c for c in cols}

    for cand in candidates:
        if cand in exact:
            return exact[cand]

    for cand in candidates:
        key = normalize_col_name(cand)
        if key in normalized:
            return normalized[key]

    return None


def detect_sovi_score_column(df: pd.DataFrame, explicit_col: str | None) -> tuple[str, pd.DataFrame]:
    if explicit_col is not None:
        if explicit_col not in df.columns:
            raise ValueError(f"Requested SoVI score column not found: {explicit_col}")
        return explicit_col, pd.DataFrame(
            [
                {
                    "column": explicit_col,
                    "selection": "explicit",
                    "reason": "provided by --sovi-score-col",
                    "non_missing_numeric": int(pd.to_numeric(df[explicit_col], errors="coerce").notna().sum()),
                }
            ]
        )

    rows = []
    best_col = None
    best_score = (-1, -1, -1)

    normalized_candidates = {normalize_col_name(c): c for c in SOVI_SCORE_CANDIDATES}
    target_like_prefixes = ("event_count_", "recommended_primary_target")

    for col in df.columns:
        col_s = str(col)
        low = col_s.lower()
        if low.startswith(target_like_prefixes):
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")
        non_missing_numeric = int(numeric.notna().sum())
        if non_missing_numeric == 0:
            continue

        norm = normalize_col_name(col_s)
        exact_candidate = norm in normalized_candidates
        has_sovi = "sovi" in low
        has_score = "score" in low or "index" in low
        has_normalized = "normal" in low or "0_1" in low or "01" in low
        has_rank = "rank" in low

        candidate_score = 0
        if exact_candidate:
            candidate_score += 100
        if has_sovi:
            candidate_score += 50
        if has_score:
            candidate_score += 30
        if has_normalized:
            candidate_score += 20
        if has_rank:
            candidate_score -= 30
        if "quality" in low or "missing" in low or "flag" in low:
            candidate_score -= 50

        if candidate_score > 0:
            rows.append(
                {
                    "column": col_s,
                    "selection": "candidate",
                    "candidate_score": candidate_score,
                    "non_missing_numeric": non_missing_numeric,
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                    "has_sovi": has_sovi,
                    "has_score_or_index": has_score,
                    "has_normalized": has_normalized,
                    "has_rank": has_rank,
                }
            )
            score_tuple = (candidate_score, non_missing_numeric, -len(col_s))
            if score_tuple > best_score:
                best_score = score_tuple
                best_col = col_s

    if best_col is None:
        raise ValueError(
            "Could not automatically detect a SoVI score column. "
            "Pass --sovi-score-col explicitly. Candidate columns containing 'sovi' or 'score' were not found."
        )

    audit = pd.DataFrame(rows).sort_values(
        ["candidate_score", "non_missing_numeric"],
        ascending=False,
    )
    audit["selected"] = audit["column"].eq(best_col)
    return best_col, audit


def detect_id_name_columns(df: pd.DataFrame, cd_id_col: str | None, cd_name_col: str | None) -> tuple[str, str | None]:
    if cd_id_col is not None:
        if cd_id_col not in df.columns:
            raise ValueError(f"Requested CD ID column not found: {cd_id_col}")
        chosen_id = cd_id_col
    else:
        chosen_id = choose_first_existing(df.columns, ID_CANDIDATES)
        if chosen_id is None:
            raise ValueError("Could not detect CD ID column. Pass --cd-id-col explicitly.")

    if cd_name_col is not None:
        if cd_name_col not in df.columns:
            raise ValueError(f"Requested CD name column not found: {cd_name_col}")
        chosen_name = cd_name_col
    else:
        chosen_name = choose_first_existing(df.columns, NAME_CANDIDATES)

    return chosen_id, chosen_name


def available_targets(df: pd.DataFrame) -> list[dict[str, Any]]:
    targets = []
    for spec in TARGET_DEFINITIONS:
        col = spec["target_column"]
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() == 0:
            continue

        spec_copy = dict(spec)
        spec_copy["target_total"] = float(values.fillna(0).sum())
        spec_copy["target_nonzero_count"] = int((values.fillna(0) > 0).sum())
        spec_copy["target_nonzero_rate"] = float((values.fillna(0) > 0).mean()) if len(values) else math.nan
        targets.append(spec_copy)

    if not targets:
        missing = [spec["target_column"] for spec in TARGET_DEFINITIONS if spec["target_column"] not in df.columns]
        raise ValueError(
            "No target definitions were available in the input table. "
            f"Missing expected columns include: {missing[:5]}"
        )

    return targets


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    return safe_pearson(xr, yr)


def safe_pearson(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return math.nan

    xv = x[valid].to_numpy(dtype=float)
    yv = y[valid].to_numpy(dtype=float)
    if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
        return math.nan

    return float(np.corrcoef(xv, yv)[0, 1])


def kendall_tau_b(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    valid = x.notna() & y.notna()
    xv = x[valid].to_numpy(dtype=float)
    yv = y[valid].to_numpy(dtype=float)

    n = len(xv)
    if n < 3:
        return math.nan

    concordant = 0
    discordant = 0
    ties_x_only = 0
    ties_y_only = 0

    for i in range(n - 1):
        dx = xv[i] - xv[i + 1 :]
        dy = yv[i] - yv[i + 1 :]

        sx = np.sign(dx)
        sy = np.sign(dy)

        both_nonzero = (sx != 0) & (sy != 0)
        concordant += int(np.sum((sx[both_nonzero] * sy[both_nonzero]) > 0))
        discordant += int(np.sum((sx[both_nonzero] * sy[both_nonzero]) < 0))

        ties_x_only += int(np.sum((sx == 0) & (sy != 0)))
        ties_y_only += int(np.sum((sx != 0) & (sy == 0)))
        # Pairs tied on both variables do not contribute to tau-b denominator.

    denominator = math.sqrt(
        (concordant + discordant + ties_x_only)
        * (concordant + discordant + ties_y_only)
    )
    if denominator == 0:
        return math.nan

    return float((concordant - discordant) / denominator)


def stable_descending_order(values: pd.Series, ids: pd.Series) -> list[int]:
    tmp = pd.DataFrame(
        {
            "_value": pd.to_numeric(values, errors="coerce"),
            "_id": ids.astype("string").fillna(""),
        }
    )
    tmp["_orig_index"] = np.arange(len(tmp))
    tmp = tmp.sort_values(
        ["_value", "_id"],
        ascending=[False, True],
        na_position="last",
        kind="mergesort",
    )
    return tmp["_orig_index"].tolist()


def linear_dcg(relevance: np.ndarray) -> float:
    if len(relevance) == 0:
        return math.nan
    discounts = 1.0 / np.log2(np.arange(2, len(relevance) + 2))
    return float(np.sum(relevance * discounts))


def ndcg_at_k(score: pd.Series, target: pd.Series, ids: pd.Series, k: int) -> float:
    valid = score.notna() & target.notna()
    if valid.sum() == 0:
        return math.nan

    score_v = score[valid].reset_index(drop=True)
    target_v = target[valid].reset_index(drop=True)
    ids_v = ids[valid].reset_index(drop=True)

    rel = pd.to_numeric(target_v, errors="coerce").fillna(0).clip(lower=0)
    if rel.sum() <= 0:
        return math.nan

    k_eff = min(k, len(rel))
    pred_order = stable_descending_order(score_v, ids_v)[:k_eff]
    ideal_order = stable_descending_order(rel, ids_v)[:k_eff]

    dcg = linear_dcg(rel.iloc[pred_order].to_numpy(dtype=float))
    ideal = linear_dcg(rel.iloc[ideal_order].to_numpy(dtype=float))
    if ideal == 0:
        return math.nan

    return float(dcg / ideal)


def top_overlap_rate(score: pd.Series, target: pd.Series, ids: pd.Series, k: int) -> float:
    valid = score.notna() & target.notna()
    if valid.sum() == 0:
        return math.nan

    score_v = score[valid].reset_index(drop=True)
    target_v = target[valid].reset_index(drop=True)
    ids_v = ids[valid].reset_index(drop=True)

    k_eff = min(k, len(score_v))
    if k_eff <= 0:
        return math.nan

    pred_order = stable_descending_order(score_v, ids_v)[:k_eff]
    target_order = stable_descending_order(target_v, ids_v)[:k_eff]

    pred_top = set(ids_v.iloc[pred_order].astype(str))
    target_top = set(ids_v.iloc[target_order].astype(str))

    return float(len(pred_top & target_top) / k_eff)


def percentile_top_k(n: int, pct: float) -> int:
    return max(1, int(math.ceil(n * pct)))


def evaluate_ranking(
    df: pd.DataFrame,
    *,
    score_col: str,
    target_col: str,
    id_col: str,
) -> dict[str, Any]:
    score = pd.to_numeric(df[score_col], errors="coerce")
    target = pd.to_numeric(df[target_col], errors="coerce")
    ids = df[id_col].astype("string")

    valid = score.notna() & target.notna()
    eval_df = df.loc[valid].copy()
    score_v = score.loc[valid]
    target_v = target.loc[valid]
    ids_v = ids.loc[valid]

    row: dict[str, Any] = {
        "n": int(valid.sum()),
        "target_total": float(target_v.fillna(0).sum()),
        "target_nonzero_count": int((target_v.fillna(0) > 0).sum()),
        "target_nonzero_rate": float((target_v.fillna(0) > 0).mean()) if len(target_v) else math.nan,
        "score_missing_count": int(score.isna().sum()),
        "target_missing_count": int(target.isna().sum()),
    }

    if valid.sum() < 3 or target_v.nunique(dropna=True) < 2 or score_v.nunique(dropna=True) < 2:
        for metric in ALL_METRICS:
            row[metric] = math.nan
        return row

    row["spearman"] = spearman_corr(score_v, target_v)
    row["kendall"] = kendall_tau_b(score_v, target_v)

    for k in TOP_KS:
        row[f"ndcg_at_{k}"] = ndcg_at_k(score_v, target_v, ids_v, k)
        row[f"top{k}_overlap_rate"] = top_overlap_rate(score_v, target_v, ids_v, k)

    n = int(valid.sum())
    for metric_name, pct in PCT_TOPS.items():
        k = percentile_top_k(n, pct)
        row[metric_name] = top_overlap_rate(score_v, target_v, ids_v, k)
        row[f"{metric_name}_k"] = k

    return row


def rank_positions(values: pd.Series, ids: pd.Series) -> pd.Series:
    order = stable_descending_order(values, ids)
    ranks = pd.Series(index=np.arange(len(values)), dtype="Int64")
    for pos, idx in enumerate(order, start=1):
        ranks.iloc[idx] = pos
    return ranks.astype("Int64")


def build_rankings(
    df: pd.DataFrame,
    *,
    targets: list[dict[str, Any]],
    score_col: str,
    id_col: str,
    name_col: str | None,
) -> pd.DataFrame:
    rows = []

    ids = df[id_col].astype("string")
    names = df[name_col].astype("string") if name_col and name_col in df.columns else pd.Series(pd.NA, index=df.index)
    sovi_score = pd.to_numeric(df[score_col], errors="coerce")
    sovi_rank = rank_positions(sovi_score, ids)

    n = len(df)
    top5pct_k = percentile_top_k(n, 0.05)
    top10pct_k = percentile_top_k(n, 0.10)

    for spec in targets:
        target_col = spec["target_column"]
        target = pd.to_numeric(df[target_col], errors="coerce")
        target_rank = rank_positions(target, ids)

        for i in range(len(df)):
            row = {
                "target_id": spec["target_id"],
                "target_label": spec["target_label"],
                "target_column": target_col,
                "target_family": spec["target_family"],
                "temporal_interpretation": spec["temporal_interpretation"],
                "cd_id_norm": ids.iloc[i],
                "cd_name": names.iloc[i],
                "sovi_score": sovi_score.iloc[i],
                "target_value": target.iloc[i],
                "sovi_rank_desc": int(sovi_rank.iloc[i]) if pd.notna(sovi_rank.iloc[i]) else pd.NA,
                "target_rank_desc": int(target_rank.iloc[i]) if pd.notna(target_rank.iloc[i]) else pd.NA,
                "is_primary_target": bool(spec.get("is_primary", False)),
                "is_sovi_top10": bool(sovi_rank.iloc[i] <= 10) if pd.notna(sovi_rank.iloc[i]) else False,
                "is_target_top10": bool(target_rank.iloc[i] <= 10) if pd.notna(target_rank.iloc[i]) else False,
                "is_sovi_top25": bool(sovi_rank.iloc[i] <= 25) if pd.notna(sovi_rank.iloc[i]) else False,
                "is_target_top25": bool(target_rank.iloc[i] <= 25) if pd.notna(target_rank.iloc[i]) else False,
                "is_sovi_top50": bool(sovi_rank.iloc[i] <= 50) if pd.notna(sovi_rank.iloc[i]) else False,
                "is_target_top50": bool(target_rank.iloc[i] <= 50) if pd.notna(target_rank.iloc[i]) else False,
                "is_sovi_top_5pct": bool(sovi_rank.iloc[i] <= top5pct_k) if pd.notna(sovi_rank.iloc[i]) else False,
                "is_target_top_5pct": bool(target_rank.iloc[i] <= top5pct_k) if pd.notna(target_rank.iloc[i]) else False,
                "is_sovi_top_10pct": bool(sovi_rank.iloc[i] <= top10pct_k) if pd.notna(sovi_rank.iloc[i]) else False,
                "is_target_top_10pct": bool(target_rank.iloc[i] <= top10pct_k) if pd.notna(target_rank.iloc[i]) else False,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def target_summary(df: pd.DataFrame, targets: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for spec in targets:
        col = spec["target_column"]
        values = pd.to_numeric(df[col], errors="coerce").fillna(0)
        rows.append(
            {
                "target_id": spec["target_id"],
                "target_label": spec["target_label"],
                "target_column": col,
                "target_family": spec["target_family"],
                "temporal_interpretation": spec["temporal_interpretation"],
                "is_primary": bool(spec.get("is_primary", False)),
                "n": int(len(values)),
                "total_events": int(values.sum()),
                "nonzero_cds": int((values > 0).sum()),
                "zero_cds": int((values == 0).sum()),
                "nonzero_rate": float((values > 0).mean()) if len(values) else math.nan,
                "mean": float(values.mean()) if len(values) else math.nan,
                "median": float(values.median()) if len(values) else math.nan,
                "max": float(values.max()) if len(values) else math.nan,
                "p75": float(values.quantile(0.75)) if len(values) else math.nan,
                "p90": float(values.quantile(0.90)) if len(values) else math.nan,
                "interpretation_note": spec.get("interpretation_note", ""),
            }
        )
    return pd.DataFrame(rows)


def find_numeric_control_column(df: pd.DataFrame, candidate_names: list[str]) -> str | None:
    normalized_candidates = [normalize_col_name(c) for c in candidate_names]
    exact = choose_first_existing(df.columns, candidate_names)
    if exact is not None and pd.to_numeric(df[exact], errors="coerce").notna().sum() > 0:
        return exact

    best_col = None
    best_score = -1
    for col in df.columns:
        norm = normalize_col_name(col)
        low = str(col).lower()
        if str(col).startswith("event_count_") or str(col).startswith("recommended_primary_target"):
            continue
        if pd.to_numeric(df[col], errors="coerce").notna().sum() == 0:
            continue

        score = 0
        for cand in normalized_candidates:
            if norm == cand:
                score += 100
            elif cand in norm:
                score += 10
        if "sovi" in low or "rank" in low or "score" in low:
            score -= 20

        if score > best_score and score > 0:
            best_score = score
            best_col = str(col)

    return best_col


def build_scorer_columns(df: pd.DataFrame, score_col: str, reverse_sovi_score: bool, random_seed: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = df.copy()
    scorers = []

    out["_score_sovi_direct"] = pd.to_numeric(out[score_col], errors="coerce")
    if reverse_sovi_score:
        out["_score_sovi_direct"] = -out["_score_sovi_direct"]

    scorers.append(
        {
            "scorer_id": "SoVI_direct",
            "scorer_label": "SoVI direct oriented score",
            "scorer_type": "static_index",
            "score_column": "_score_sovi_direct",
            "source_column": score_col,
            "is_main_sovi_scorer": True,
            "note": "Raw/static SoVI score used directly as a ranking signal.",
        }
    )

    rng = np.random.default_rng(random_seed)
    out["_score_random_seed"] = rng.random(len(out))
    scorers.append(
        {
            "scorer_id": f"B0_random_seed_{random_seed}",
            "scorer_label": f"Random ranking, seed {random_seed}",
            "scorer_type": "null_control",
            "score_column": "_score_random_seed",
            "source_column": None,
            "is_main_sovi_scorer": False,
            "note": "Small sanity/null control; not a substantive exposure model.",
        }
    )

    pop_col = find_numeric_control_column(
        out,
        [
            "population",
            "total_population",
            "population_total",
            "pop_total",
            "C1_COUNT_TOTAL",
            "c1_count_total",
        ],
    )
    if pop_col is not None:
        out["_score_population_control"] = pd.to_numeric(out[pop_col], errors="coerce")
        scorers.append(
            {
                "scorer_id": "B0_population",
                "scorer_label": "Population control",
                "scorer_type": "null_control",
                "score_column": "_score_population_control",
                "source_column": pop_col,
                "is_main_sovi_scorer": False,
                "note": "Simple exposure sanity control available in the aligned frame.",
            }
        )

    area_col = find_numeric_control_column(
        out,
        ["land_area", "landarea", "LANDAREA", "area_km2", "land_area_km2"],
    )
    if area_col is not None:
        out["_score_land_area_control"] = pd.to_numeric(out[area_col], errors="coerce")
        scorers.append(
            {
                "scorer_id": "B0_land_area",
                "scorer_label": "Land area control",
                "scorer_type": "null_control",
                "score_column": "_score_land_area_control",
                "source_column": area_col,
                "is_main_sovi_scorer": False,
                "note": "Simple geography/exposure sanity control available in the aligned frame.",
            }
        )

    density_col = find_numeric_control_column(
        out,
        [
            "population_density",
            "pop_density",
            "density",
            "population_per_km2",
        ],
    )
    if density_col is not None:
        out["_score_density_control"] = pd.to_numeric(out[density_col], errors="coerce")
        scorers.append(
            {
                "scorer_id": "B0_density",
                "scorer_label": "Population density control",
                "scorer_type": "null_control",
                "score_column": "_score_density_control",
                "source_column": density_col,
                "is_main_sovi_scorer": False,
                "note": "Simple density sanity control available in the aligned frame.",
            }
        )
    elif pop_col is not None and area_col is not None:
        pop = pd.to_numeric(out[pop_col], errors="coerce")
        area = pd.to_numeric(out[area_col], errors="coerce")
        density = pop / area.replace(0, np.nan)
        if density.notna().sum() > 0:
            out["_score_density_control"] = density
            scorers.append(
                {
                    "scorer_id": "B0_density_computed",
                    "scorer_label": "Computed population density control",
                    "scorer_type": "null_control",
                    "score_column": "_score_density_control",
                    "source_column": f"{pop_col}/{area_col}",
                    "is_main_sovi_scorer": False,
                    "note": "Simple computed density sanity control; not a tuned model.",
                }
            )

    return out, scorers


def evaluate_all(
    df: pd.DataFrame,
    *,
    targets: list[dict[str, Any]],
    scorers: list[dict[str, Any]],
    id_col: str,
) -> pd.DataFrame:
    rows = []

    for spec in targets:
        for scorer in scorers:
            metric_row = evaluate_ranking(
                df,
                score_col=scorer["score_column"],
                target_col=spec["target_column"],
                id_col=id_col,
            )
            row = {
                "target_id": spec["target_id"],
                "target_label": spec["target_label"],
                "target_column": spec["target_column"],
                "target_family": spec["target_family"],
                "temporal_interpretation": spec["temporal_interpretation"],
                "is_primary_target": bool(spec.get("is_primary", False)),
                "scorer_id": scorer["scorer_id"],
                "scorer_label": scorer["scorer_label"],
                "scorer_type": scorer["scorer_type"],
                "score_column": scorer["score_column"],
                "source_column": scorer.get("source_column"),
                "is_main_sovi_scorer": bool(scorer.get("is_main_sovi_scorer", False)),
                "note": scorer.get("note", ""),
                **metric_row,
            }
            rows.append(row)

    metrics = pd.DataFrame(rows)
    sort_cols = ["target_id", "is_main_sovi_scorer", "scorer_type", "scorer_id"]
    metrics = metrics.sort_values(sort_cols, ascending=[True, False, True, True]).reset_index(drop=True)
    return metrics


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_interpretation(
    *,
    output_path: Path,
    metrics: pd.DataFrame,
    target_summary_df: pd.DataFrame,
    score_col: str,
    reverse_sovi_score: bool,
    scorers: list[dict[str, Any]],
) -> None:
    sovi_metrics = metrics[metrics["scorer_id"].eq("SoVI_direct")].copy()
    primary = sovi_metrics[sovi_metrics["is_primary_target"].eq(True)]

    if not primary.empty:
        p = primary.iloc[0]
        primary_lines = [
            f"- Spearman: {fmt(p.get('spearman'))}",
            f"- Kendall: {fmt(p.get('kendall'))}",
            f"- NDCG@10: {fmt(p.get('ndcg_at_10'))}",
            f"- NDCG@25: {fmt(p.get('ndcg_at_25'))}",
            f"- Top-10 overlap: {fmt(p.get('top10_overlap_rate'))}",
            f"- Top-25 overlap: {fmt(p.get('top25_overlap_rate'))}",
        ]
    else:
        primary_lines = ["- Primary target metrics unavailable."]

    sensitivity = sovi_metrics[
        [
            "target_id",
            "target_label",
            "target_family",
            "spearman",
            "kendall",
            "ndcg_at_10",
            "ndcg_at_25",
            "top10_overlap_rate",
            "top25_overlap_rate",
        ]
    ].copy()

    if not sensitivity.empty:
        sensitivity = sensitivity.sort_values("target_id")
        sensitivity_md = sensitivity.to_markdown(index=False, floatfmt=".3f")
    else:
        sensitivity_md = "No SoVI sensitivity metrics available."

    controls = metrics[metrics["scorer_type"].eq("null_control")].copy()
    if not controls.empty:
        control_primary = controls[controls["is_primary_target"].eq(True)]
        control_cols = [
            "scorer_id",
            "spearman",
            "kendall",
            "ndcg_at_10",
            "ndcg_at_25",
            "top10_overlap_rate",
            "top25_overlap_rate",
        ]
        controls_md = control_primary[control_cols].to_markdown(index=False, floatfmt=".3f")
    else:
        controls_md = "No optional null controls were available."

    target_md = target_summary_df[
        [
            "target_id",
            "target_label",
            "target_family",
            "total_events",
            "nonzero_cds",
            "nonzero_rate",
            "median",
            "max",
        ]
    ].to_markdown(index=False, floatfmt=".3f")

    orientation_note = (
        "The selected SoVI score was reversed before evaluation."
        if reverse_sovi_score
        else "The selected SoVI score was evaluated as higher = more vulnerable."
    )

    text = f"""# SoVI direct external-validation against Québec civil-security event burden

## Scope

This benchmark is a direct external-validation benchmark. It asks whether the static SoVI-like CD score ranks Québec census divisions in a way that aligns with observed civil-security event burden.

It does **not** run supervised ML, graph modeling, calibration, or target-based tuning. The SoVI score is used directly as a ranking signal.

Selected SoVI score column:

```text
{score_col}
```

{orientation_note}

## Primary target

The primary target is:

```text
event_count_2021_2025_all
```

Interpretation:

```text
2021–2025 = descriptive / near-snapshot validation
2022–2025 = cleaner forward-looking validation from mostly-2021 SoVI variables
```

The primary result should be read as external alignment evidence, not causal proof and not proof that SoVI predicts future operational disruption.

## Primary result: SoVI vs 2021–2025 all civil-security events

{chr(10).join(primary_lines)}

Headline metrics are Spearman, Kendall, NDCG@10, NDCG@25, top-10 overlap, and top-25 overlap.

NDCG@50 and top-50 overlap are still reported in the CSV files for consistency with the SVI benchmark family, but they should not be headlined strongly because there are only 98 CDs. Top-50 is roughly half the province, so it is a broad/global diagnostic rather than a sharp prioritization metric.

## Target summary

{target_md}

## SoVI target-sensitivity results

{sensitivity_md}

## Optional B0 sanity/null controls

These controls are included only as small sanity checks when the required columns are already present in the aligned frame. They are not a supervised exposure model and should not become the center of the benchmark.

{controls_md}

## Recommended interpretation

Use this benchmark to answer:

```text
Do Québec CDs with higher SoVI-like vulnerability scores tend to rank higher in observed civil-security event burden, and is this relationship stable across target definitions, hazard groups, precision filters, and severity thresholds?
```

A strong result would show stable positive rank association across the primary target, the 2022–2025 forward-looking target, localization-precision targets, and at least some hazard/severity targets.

A weak or unstable result would not invalidate SoVI as a social vulnerability index. It would mean that this particular event-burden layer is only partially aligned with the social-vulnerability construct, or that observed event counts are driven heavily by exposure, hazard geography, reporting practices, administrative definitions, and event-detection processes.

## Caveats

- Civil-security event burden is not the same construct as social vulnerability.
- CD-level event counts mix hazard exposure, reporting intensity, administrative practice, and vulnerability.
- The current benchmark is ranking-based because SoVI is an index, not a calibrated event-count predictor.
- No SoVI score was tuned using the civil-security target.
- Monthly CD targets are intentionally not the main target because earlier density checks showed sparse CD-month cells.
"""
    output_path.write_text(text, encoding="utf-8")


def write_json(data: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(config.targets_path)

    id_col, name_col = detect_id_name_columns(df, config.cd_id_col, config.cd_name_col)
    score_col, score_audit = detect_sovi_score_column(df, config.sovi_score_col)
    targets = available_targets(df)

    df_eval, scorers = build_scorer_columns(
        df,
        score_col=score_col,
        reverse_sovi_score=config.reverse_sovi_score,
        random_seed=config.random_seed,
    )

    metrics = evaluate_all(
        df_eval,
        targets=targets,
        scorers=scorers,
        id_col=id_col,
    )

    rankings = build_rankings(
        df_eval,
        targets=targets,
        score_col="_score_sovi_direct",
        id_col=id_col,
        name_col=name_col,
    )

    target_summary_df = target_summary(df_eval, targets)

    metrics_path = config.output_dir / "sovi_civil_security_validation_metrics.csv"
    rankings_path = config.output_dir / "sovi_civil_security_validation_rankings.csv"
    target_summary_path = config.output_dir / "sovi_civil_security_validation_target_summary.csv"
    interpretation_path = config.output_dir / "sovi_civil_security_validation_interpretation.md"
    metadata_path = config.output_dir / "sovi_civil_security_validation_metadata.json"
    score_audit_path = config.output_dir / "sovi_civil_security_validation_score_column_audit.csv"

    metrics.to_csv(metrics_path, index=False)
    rankings.to_csv(rankings_path, index=False)
    target_summary_df.to_csv(target_summary_path, index=False)
    score_audit.to_csv(score_audit_path, index=False)

    write_interpretation(
        output_path=interpretation_path,
        metrics=metrics,
        target_summary_df=target_summary_df,
        score_col=score_col,
        reverse_sovi_score=config.reverse_sovi_score,
        scorers=scorers,
    )

    primary_metrics = metrics[
        metrics["is_primary_target"].eq(True)
        & metrics["scorer_id"].eq("SoVI_direct")
    ].copy()

    primary_row = primary_metrics.iloc[0].to_dict() if not primary_metrics.empty else {}

    metadata = {
        "status": "completed",
        "benchmark_type": "SoVI direct external-validation benchmark",
        "scientific_question": (
            "Do Québec CDs with higher SoVI-like vulnerability scores tend to rank higher "
            "in observed civil-security event burden, and is this relationship stable across "
            "target definitions, hazard groups, precision filters, and severity thresholds?"
        ),
        "constraints": {
            "supervised_ml_run": False,
            "graph_model_run": False,
            "calibration_model_run": False,
            "sovi_tuned_on_civil_security_target": False,
            "direct_static_index_validation_only": True,
        },
        "input_targets_path": str(config.targets_path),
        "output_dir": str(config.output_dir),
        "cd_id_col": id_col,
        "cd_name_col": name_col,
        "sovi_score_col": score_col,
        "reverse_sovi_score": config.reverse_sovi_score,
        "random_seed": config.random_seed,
        "n_rows": int(len(df_eval)),
        "n_targets_evaluated": int(len(targets)),
        "targets_evaluated": targets,
        "scorers_evaluated": scorers,
        "headline_metrics": HEADLINE_METRICS,
        "all_metrics": ALL_METRICS,
        "metric_notes": {
            "ndcg": "Linear-relevance NDCG using event counts as relevance.",
            "ndcg_at_50": "Reported for SVI-family consistency but should not be headlined strongly with only 98 CDs.",
            "top50_overlap_rate": "Reported for SVI-family consistency but broad/global with only 98 CDs.",
            "ndcg_at_100": "Not reported because there are only 98 CDs; no full-ranking NDCG is used here.",
        },
        "primary_target": {
            "target_id": "B1a",
            "target_column": "event_count_2021_2025_all",
            "interpretation": "descriptive / near-snapshot validation",
        },
        "forward_target": {
            "target_id": "B1b",
            "target_column": "event_count_2022_2025_all",
            "interpretation": "cleaner forward-looking validation from mostly-2021 SoVI variables",
        },
        "primary_sovi_result": primary_row,
        "outputs": {
            "metrics": str(metrics_path),
            "rankings": str(rankings_path),
            "target_summary": str(target_summary_path),
            "interpretation": str(interpretation_path),
            "metadata": str(metadata_path),
            "score_column_audit": str(score_audit_path),
        },
    }

    write_json(metadata, metadata_path)

    print("SoVI civil-security direct validation benchmark completed.")
    print(f"Input: {config.targets_path}")
    print(f"Output directory: {config.output_dir}")
    print(f"CD ID column: {id_col}")
    print(f"CD name column: {name_col}")
    print(f"SoVI score column: {score_col}")
    print()
    print("Outputs:")
    print(f"  metrics: {metrics_path}")
    print(f"  rankings: {rankings_path}")
    print(f"  target summary: {target_summary_path}")
    print(f"  interpretation: {interpretation_path}")
    print(f"  metadata: {metadata_path}")
    print()
    if primary_row:
        print("Primary SoVI result on B1a / event_count_2021_2025_all:")
        for metric in HEADLINE_METRICS:
            print(f"  {metric}: {fmt(primary_row.get(metric))}")


if __name__ == "__main__":
    main()
