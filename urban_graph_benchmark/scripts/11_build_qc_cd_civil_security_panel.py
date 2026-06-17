#!/usr/bin/env python3
"""
Build the Québec CD × month civil-security / SoVI predictive panel.

Output:
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/datasets/cd_month_panel.parquet

This script is intentionally a dataset-building script, not a model baseline.
It prepares a shared forecasting panel that later B0/B2/B3/B4 baselines can use.

Default split convention requested for this benchmark:
    train: 2021-2023 origin months
    val:   2024 origin months
    test:  2025 origin months

Forecast targets:
    target_next_1_month   = event count in t+1
    target_next_3_months  = event count in t+1 + t+2 + t+3

Rows near the end of the available date range are kept by default, but future
targets are set to NaN when the full future window is not observable.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from ville_hgnn.baselines.qc_cd_sovi_common import (
        CD_ID_COL,
        CD_NAME_COL,
        DATASETS_DIR,
        DEFAULT_CD_MONTH_TARGETS_PATH,
        DEFAULT_CUMULATIVE_TARGETS_PATH,
        DEFAULT_PANEL_PATH,
        HAZARD_GROUPS,
        OUTPUT_ROOT,
        ensure_dir,
        write_metadata_json,
    )
except Exception:  # pragma: no cover - fallback for early bootstrapping only
    CD_ID_COL = "cd_id_norm"
    CD_NAME_COL = "cd_name"
    OUTPUT_ROOT = Path("urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0")
    DATASETS_DIR = OUTPUT_ROOT / "datasets"
    DEFAULT_CUMULATIVE_TARGETS_PATH = Path(
        "data/external/quebec_civil_security_events/processed/"
        "cd_civil_security_sovi_validation_targets_cumulative.parquet"
    )
    DEFAULT_CD_MONTH_TARGETS_PATH = Path(
        "data/external/quebec_civil_security_events/processed/"
        "cd_civil_security_sovi_validation_targets_cd_month.parquet"
    )
    DEFAULT_PANEL_PATH = DATASETS_DIR / "cd_month_panel.parquet"
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

    def ensure_dir(path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        return out

    def write_metadata_json(metadata: dict[str, Any], path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        return out


DEFAULT_OUTPUT_PATH = DEFAULT_PANEL_PATH


ID_COLUMN_CANDIDATES = [
    "cd_id_norm",
    "cd_id",
    "CDUID",
    "cduid",
    "zone_id",
    "census_division_dguid",
    "DGUID",
    "dguid",
]

NAME_COLUMN_CANDIDATES = [
    "cd_name",
    "CDNAME",
    "cdname",
    "name",
    "NAME",
    "census_division_name",
]

MONTH_COLUMN_CANDIDATES = [
    "period_month",
    "event_period_month",
    "month_period",
    "year_month",
    "date_month",
    "month",
]

NON_SOVI_PREFIXES = (
    "event_count_",
    "target_",
    "recommended_primary_",
)

NON_SOVI_EXACT = {
    "_original_cd_id_norm_for_targets",
}


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input table does not exist: {p}")

    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    raise ValueError(f"Unsupported input table suffix: {p}")


def write_table_with_csv_copy(df: pd.DataFrame, path: str | Path) -> dict[str, str]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    if p.suffix.lower() == ".parquet":
        df.to_parquet(p, index=False)
        outputs["parquet"] = str(p)
        csv_path = p.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        outputs["csv"] = str(csv_path)
    elif p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
        outputs["csv"] = str(p)
    else:
        raise ValueError(f"Unsupported output suffix: {p}")

    return outputs


def normalize_cd_id(value: Any) -> str | None:
    """Normalize CD identifiers to compact 4-digit strings such as '2401'."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    # Remove common float artifacts from CSV round trips.
    text = re.sub(r"\.0$", "", text)

    # DGUID-like values often end in the 4-digit CDUID.
    digit_groups = re.findall(r"\d+", text)
    if digit_groups:
        joined = "".join(digit_groups)
        if len(joined) >= 4:
            return joined[-4:]
        return joined.zfill(4)

    return text


def find_first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower_to_original = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def ensure_cd_id_column(df: pd.DataFrame, *, preferred_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    columns = list(out.columns)

    id_col = preferred_col if preferred_col and preferred_col in out.columns else None
    if id_col is None:
        id_col = find_first_existing(columns, ID_COLUMN_CANDIDATES)

    if id_col is None:
        raise KeyError(
            "Could not find a CD identifier column. "
            f"Tried: {ID_COLUMN_CANDIDATES}. Available columns: {columns}"
        )

    out[CD_ID_COL] = out[id_col].map(normalize_cd_id)
    return out


def ensure_cd_name_column(df: pd.DataFrame, *, preferred_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    columns = list(out.columns)

    name_col = preferred_col if preferred_col and preferred_col in out.columns else None
    if name_col is None:
        name_col = find_first_existing(columns, NAME_COLUMN_CANDIDATES)

    if name_col is None:
        out[CD_NAME_COL] = pd.NA
    else:
        out[CD_NAME_COL] = out[name_col].astype("string")

    return out


def parse_period_month_value(value: Any) -> pd.Period | pd.NaT:
    if pd.isna(value):
        return pd.NaT

    if isinstance(value, pd.Period):
        return value.asfreq("M")

    text = str(value).strip()
    if not text:
        return pd.NaT

    # Accept YYYY-MM quickly to avoid pandas interpreting some strings oddly.
    match = re.match(r"^(\d{4})[-/](\d{1,2})$", text)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        if 1 <= month <= 12:
            return pd.Period(year=year, month=month, freq="M")

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT

    return pd.Period(parsed, freq="M")


def ensure_period_month_column(df: pd.DataFrame, *, preferred_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    columns = list(out.columns)

    month_col = preferred_col if preferred_col and preferred_col in out.columns else None
    if month_col is None:
        month_col = find_first_existing(columns, MONTH_COLUMN_CANDIDATES)

    # If a bare "month" column is numeric and a year column exists, build YYYY-MM.
    if month_col is not None and month_col.lower() == "month" and "year" in out.columns:
        month_num = pd.to_numeric(out[month_col], errors="coerce").astype("Int64")
        year_num = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
        out["period_month"] = [
            pd.Period(year=int(y), month=int(m), freq="M") if pd.notna(y) and pd.notna(m) else pd.NaT
            for y, m in zip(year_num, month_num)
        ]
        return out

    if month_col is not None:
        out["period_month"] = out[month_col].map(parse_period_month_value)
        return out

    if {"year", "month"}.issubset(out.columns):
        month_num = pd.to_numeric(out["month"], errors="coerce").astype("Int64")
        year_num = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
        out["period_month"] = [
            pd.Period(year=int(y), month=int(m), freq="M") if pd.notna(y) and pd.notna(m) else pd.NaT
            for y, m in zip(year_num, month_num)
        ]
        return out

    raise KeyError(
        "Could not find a month column. "
        f"Tried: {MONTH_COLUMN_CANDIDATES}, or year+month. "
        f"Available columns: {columns}"
    )


def is_sovi_feature_column(col: str) -> bool:
    lower = col.lower()

    if col in NON_SOVI_EXACT:
        return False
    if lower.startswith(NON_SOVI_PREFIXES):
        return False
    if lower in {
        "recommended_primary_target",
        "recommended_primary_target_name",
        "recommended_primary_target_granularity",
        "recommended_primary_precision_filter",
    }:
        return False

    return True


def load_sovi_cd_frame(
    path: str | Path,
    *,
    cd_id_col: str | None = None,
    cd_name_col: str | None = None,
) -> pd.DataFrame:
    raw = read_table(path)
    raw = ensure_cd_id_column(raw, preferred_col=cd_id_col)
    raw = ensure_cd_name_column(raw, preferred_col=cd_name_col)

    keep_cols = [c for c in raw.columns if is_sovi_feature_column(c)]

    # Ensure canonical ID/name columns are present and first.
    for col in [CD_ID_COL, CD_NAME_COL]:
        if col not in keep_cols:
            keep_cols.insert(0, col)

    # Drop duplicate column names while preserving order.
    seen = set()
    keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]

    out = raw[keep_cols].copy()
    out = out.drop_duplicates(subset=[CD_ID_COL], keep="first")

    if out[CD_ID_COL].isna().any():
        raise ValueError("Some SoVI rows have missing normalized CD IDs.")
    if out[CD_ID_COL].duplicated().any():
        dupes = out.loc[out[CD_ID_COL].duplicated(), CD_ID_COL].tolist()
        raise ValueError(f"Duplicate normalized CD IDs in SoVI frame: {dupes[:10]}")

    return out.sort_values(CD_ID_COL).reset_index(drop=True)


def detect_event_count_columns(df: pd.DataFrame) -> list[str]:
    columns = list(df.columns)

    event_cols = [
        c
        for c in columns
        if c.lower().startswith("event_count")
        and "2021_2025" not in c.lower()
        and "2022_2025" not in c.lower()
    ]

    if event_cols:
        return event_cols

    fallback = []
    for c in ["event_count", "n_events", "count", "events"]:
        if c in columns:
            fallback.append(c)

    return fallback


def standardize_event_count_col_name(col: str) -> str:
    """
    Standardize monthly event-count column names without collapsing
    precision-filtered variants.

    Examples:
        event_count_all
          -> event_count_current_month_all

        event_count_all_flood_water
          -> event_count_current_month_all_flood_water

        event_count_precise_or_very_precise_flood_water
          -> event_count_current_month_precise_or_very_precise_flood_water

        event_count_very_precise_flood_water
          -> event_count_current_month_very_precise_flood_water
    """
    lower = col.lower()

    # Generic aliases for the total all-events monthly count.
    if lower in {
        "event_count",
        "event_count_all",
        "event_count_month_all",
        "n_events",
        "count",
        "events",
    }:
        return "event_count_current_month_all"

    # Detect precision filter first.
    if "precise_or_very_precise" in lower:
        precision = "precise_or_very_precise"
    elif "very_precise" in lower:
        precision = "very_precise"
    elif re.search(r"(^|_)all($|_)", lower):
        precision = "all"
    else:
        precision = "all"

    # Detect special subset/audit targets.
    if "moderate_or_worse" in lower:
        return f"event_count_current_month_{precision}_moderate_or_worse"
    if "important_or_extreme" in lower:
        return f"event_count_current_month_{precision}_important_or_extreme"
    if "possible_duplicate" in lower:
        return f"event_count_current_month_{precision}_possible_duplicate"

    # Detect hazard group.
    for hazard in HAZARD_GROUPS:
        if hazard in lower:
            return f"event_count_current_month_{precision}_{hazard}"

    # If this is just event_count_precise_or_very_precise or event_count_very_precise.
    if precision != "all":
        return f"event_count_current_month_{precision}"

    # Last-resort clean fallback.
    clean = re.sub(r"^event_count_?", "", lower)
    clean = re.sub(r"[^a-z0-9]+", "_", clean).strip("_")
    if not clean:
        clean = "all"
    return f"event_count_current_month_{clean}"

def load_monthly_event_counts(
    path: str | Path,
    *,
    cd_id_col: str | None = None,
    month_col: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    raw = read_table(path)
    raw = ensure_cd_id_column(raw, preferred_col=cd_id_col)
    raw = ensure_period_month_column(raw, preferred_col=month_col)

    event_cols = detect_event_count_columns(raw)
    if not event_cols:
        raise KeyError(
            "Could not detect monthly event count columns. "
            "Expected columns starting with 'event_count' or fallback names "
            "event_count/n_events/count/events."
        )

    base = raw[[CD_ID_COL, "period_month"]].copy()

    # Build standardized count columns without creating duplicate column names.
    # Important: multiple source columns can be aliases of the same semantic count.
    # In that case, we verify they are identical/compatible and keep one.
    standardized: dict[str, pd.Series] = {}
    source_columns_by_standard_name: dict[str, list[str]] = {}

    for source_col in event_cols:
        standard_col = standardize_event_count_col_name(source_col)
        values = pd.to_numeric(raw[source_col], errors="coerce")

        source_columns_by_standard_name.setdefault(standard_col, []).append(source_col)

        if standard_col not in standardized:
            standardized[standard_col] = values
            continue

        existing = standardized[standard_col]

        both_present = existing.notna() & values.notna()
        disagree = both_present & ~np.isclose(
            existing.astype(float),
            values.astype(float),
            rtol=0.0,
            atol=1e-9,
        )

        if disagree.any():
            examples = raw.loc[
                disagree,
                [CD_ID_COL, "period_month", *source_columns_by_standard_name[standard_col]],
            ].head(10)

            raise ValueError(
                "Multiple source columns map to the same standardized event-count "
                f"column '{standard_col}', but their values differ. "
                f"Source columns: {source_columns_by_standard_name[standard_col]}. "
                "These appear not to be safe aliases. Inspect examples:\n"
                f"{examples.to_string(index=False)}"
            )

        # Same semantic alias: keep the first non-missing value.
        standardized[standard_col] = existing.combine_first(values)

    work = base.copy()
    for standard_col, values in standardized.items():
        work[standard_col] = values.fillna(0.0)

    count_cols = sorted(
        standardized.keys(),
        key=lambda c: (c != "event_count_current_month_all", c),
    )

    # Now group duplicate ROWS by CD-month. This is legitimate aggregation.
    # We are no longer summing duplicate alias COLUMNS.
    work = (
        work.groupby([CD_ID_COL, "period_month"], as_index=False)[count_cols]
        .sum()
        .sort_values([CD_ID_COL, "period_month"])
        .reset_index(drop=True)
    )

    # If no explicit all-events monthly count exists, reconstruct it from hazard
    # components only. Do not include severity/audit subset columns.
    if "event_count_current_month_all" not in work.columns:
        hazard_cols = [
            f"event_count_current_month_all_{h}"
            for h in HAZARD_GROUPS
            if f"event_count_current_month_all_{h}" in work.columns
        ]

        if hazard_cols:
            work["event_count_current_month_all"] = work[hazard_cols].sum(axis=1)
            count_cols = ["event_count_current_month_all", *count_cols]
        else:
            raise KeyError(
                "No explicit all-events count exists and no hazard-component "
                "columns were available to reconstruct it."
            )

    count_cols = sorted(
        set(count_cols),
        key=lambda c: (c != "event_count_current_month_all", c),
    )

    return work, count_cols


def make_cd_month_grid(
    sovi_cd: pd.DataFrame,
    *,
    start_month: str,
    end_month: str,
) -> pd.DataFrame:
    months = pd.period_range(start=start_month, end=end_month, freq="M")
    cd_base = sovi_cd[[CD_ID_COL, CD_NAME_COL]].drop_duplicates().copy()

    grid = cd_base.assign(_key=1).merge(
        pd.DataFrame({"period_month": months, "_key": 1}),
        on="_key",
        how="outer",
    ).drop(columns="_key")

    grid["period_month"] = grid["period_month"].astype("period[M]")
    return grid.sort_values([CD_ID_COL, "period_month"]).reset_index(drop=True)


def add_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    period = out["period_month"].astype("period[M]")
    out["year"] = period.dt.year.astype(int)
    out["month"] = period.dt.month.astype(int)
    out["period_month"] = period.astype(str)
    return out


def add_history_features(
    df: pd.DataFrame,
    *,
    count_cols: list[str],
    id_col: str = CD_ID_COL,
) -> pd.DataFrame:
    out = df.copy()
    out["_period_for_sort"] = out["period_month"].map(parse_period_month_value)
    out = out.sort_values([id_col, "_period_for_sort"]).reset_index(drop=True)

    for count_col in count_cols:
        if count_col not in out.columns:
            continue

        out[count_col] = pd.to_numeric(out[count_col], errors="coerce").fillna(0.0)

        suffix = count_col.replace("event_count_current_month_", "")
        feature_base = f"event_count_{suffix}"

        group = out.groupby(id_col, sort=False)[count_col]

        out[f"{feature_base}_lag_1"] = group.shift(1).fillna(0.0)

        for window in [3, 6, 12]:
            out[f"{feature_base}_rolling_{window}"] = group.transform(
                lambda s, w=window: s.rolling(window=w, min_periods=1).sum()
            )

    # Simple aliases requested in the specification, based on all-event counts.
    if "event_count_all_lag_1" in out.columns:
        out["lag_1"] = out["event_count_all_lag_1"]
    if "event_count_all_rolling_3" in out.columns:
        out["rolling_3"] = out["event_count_all_rolling_3"]
    if "event_count_all_rolling_6" in out.columns:
        out["rolling_6"] = out["event_count_all_rolling_6"]
    if "event_count_all_rolling_12" in out.columns:
        out["rolling_12"] = out["event_count_all_rolling_12"]

    return out.drop(columns=["_period_for_sort"])


def add_future_targets(
    df: pd.DataFrame,
    *,
    count_col: str = "event_count_current_month_all",
    id_col: str = CD_ID_COL,
    panel_end_month: str,
) -> pd.DataFrame:
    out = df.copy()
    out["_period_for_sort"] = out["period_month"].map(parse_period_month_value)
    out = out.sort_values([id_col, "_period_for_sort"]).reset_index(drop=True)

    if count_col not in out.columns:
        raise KeyError(f"Required current-month count column is missing: {count_col}")

    out[count_col] = pd.to_numeric(out[count_col], errors="coerce").fillna(0.0)

    group = out.groupby(id_col, sort=False)[count_col]

    # Sum strictly future months: t+1 ... t+h.
    future_1 = group.shift(-1)
    out["target_next_1_month"] = future_1
    out["target_next_1m_all"] = future_1

    target_3 = None
    for step in [1, 2, 3]:
        shifted = group.shift(-step)
        target_3 = shifted if target_3 is None else target_3 + shifted
    out["target_next_3_months"] = target_3
    out["target_next_3m_all"] = target_3

    target_6 = None
    for step in [1, 2, 3, 4, 5, 6]:
        shifted = group.shift(-step)
        target_6 = shifted if target_6 is None else target_6 + shifted
    out["target_next_6_months"] = target_6
    out["target_next_6m_all"] = target_6

    period = out["_period_for_sort"].astype("period[M]")
    end_period = pd.Period(panel_end_month, freq="M")

    for horizon, col in [
        (1, "target_next_1_month"),
        (3, "target_next_3_months"),
        (6, "target_next_6_months"),
    ]:
        complete_col = f"{col}_complete"
        out[complete_col] = (period + horizon) <= end_period
        out.loc[~out[complete_col], col] = np.nan

    # Keep aliases consistent after masking.
    out["target_next_1m_all"] = out["target_next_1_month"]
    out["target_next_3m_all"] = out["target_next_3_months"]
    out["target_next_6m_all"] = out["target_next_6_months"]

    return out.drop(columns=["_period_for_sort"])


def add_origin_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    years = pd.to_numeric(out["year"], errors="coerce")

    out["split"] = pd.NA
    out.loc[(years >= 2021) & (years <= 2023), "split"] = "train"
    out.loc[years == 2024, "split"] = "val"
    out.loc[years == 2025, "split"] = "test"
    return out


def repair_cd_name_from_sovi(panel: pd.DataFrame, sovi_cd: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()

    if CD_NAME_COL not in out.columns:
        out[CD_NAME_COL] = pd.NA

    if CD_NAME_COL not in sovi_cd.columns:
        return out

    name_map = (
        sovi_cd[[CD_ID_COL, CD_NAME_COL]]
        .dropna(subset=[CD_ID_COL])
        .drop_duplicates(subset=[CD_ID_COL])
        .set_index(CD_ID_COL)[CD_NAME_COL]
    )

    missing = out[CD_NAME_COL].isna() | out[CD_NAME_COL].astype("string").eq("<NA")
    out.loc[missing, CD_NAME_COL] = out.loc[missing, CD_ID_COL].map(name_map)

    return out


def build_panel(
    *,
    cumulative_targets_path: Path,
    cd_month_targets_path: Path,
    output_path: Path,
    start_month: str,
    end_month: str,
    cd_id_col: str | None,
    cd_name_col: str | None,
    month_col: str | None,
    drop_incomplete_targets: bool,
) -> dict[str, Any]:
    sovi_cd = load_sovi_cd_frame(
        cumulative_targets_path,
        cd_id_col=cd_id_col,
        cd_name_col=cd_name_col,
    )

    monthly_counts, count_cols = load_monthly_event_counts(
        cd_month_targets_path,
        cd_id_col=cd_id_col,
        month_col=month_col,
    )

    grid = make_cd_month_grid(sovi_cd, start_month=start_month, end_month=end_month)

    # Merge SoVI/static CD attributes into every CD-month row.
    static_cols = [c for c in sovi_cd.columns if c not in {CD_NAME_COL}]
    panel = grid.merge(
        sovi_cd[static_cols],
        on=CD_ID_COL,
        how="left",
        validate="many_to_one",
    )

    # Repair/keep CD name.
    panel = repair_cd_name_from_sovi(panel, sovi_cd)

    panel["period_month"] = panel["period_month"].astype("period[M]")
    monthly_counts["period_month"] = monthly_counts["period_month"].astype("period[M]")

    panel = panel.merge(
        monthly_counts,
        on=[CD_ID_COL, "period_month"],
        how="left",
        validate="one_to_one",
    )

    for col in count_cols:
        if col not in panel.columns:
            panel[col] = 0.0
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0.0)

    # Ensure all-event current-month count exists.
    if "event_count_current_month_all" not in panel.columns:
        hazard_cols = [
            c
            for c in panel.columns
            if c.startswith("event_count_current_month_all_")
            and c not in {
                "event_count_current_month_all_moderate_or_worse",
                "event_count_current_month_all_important_or_extreme",
                "event_count_current_month_all_possible_duplicate",
            }
        ]
        if hazard_cols:
            panel["event_count_current_month_all"] = panel[hazard_cols].sum(axis=1)
            count_cols = ["event_count_current_month_all", *count_cols]
        else:
            raise KeyError("No all-event current-month count could be constructed.")

    count_cols = sorted(
        set([*count_cols, "event_count_current_month_all"]),
        key=lambda c: (c != "event_count_current_month_all", c),
    )

    # Calendar columns before split.
    panel = add_calendar_columns(panel)

    # History and future target features.
    panel = add_history_features(panel, count_cols=count_cols)
    panel = add_future_targets(
        panel,
        count_col="event_count_current_month_all",
        panel_end_month=end_month,
    )

    panel = add_origin_split(panel)

    if drop_incomplete_targets:
        panel = panel[panel["target_next_3_months"].notna()].copy()

    # Order high-value columns first, then the remaining static/feature columns.
    preferred_front = [
        CD_ID_COL,
        CD_NAME_COL,
        "month",
        "year",
        "period_month",
        "split",
        "score_normalized_0_1",
        "score_raw",
        "rank",
        "percentile",
        "event_count_current_month_all",
        "lag_1",
        "rolling_3",
        "rolling_6",
        "rolling_12",
        "target_next_1_month",
        "target_next_3_months",
        "target_next_6_months",
        "target_next_1_month_complete",
        "target_next_3_months_complete",
        "target_next_6_months_complete",
    ]

    front = [c for c in preferred_front if c in panel.columns]
    rest = [c for c in panel.columns if c not in front]
    panel = panel[front + rest].sort_values([CD_ID_COL, "period_month"]).reset_index(drop=True)

    output_paths = write_table_with_csv_copy(panel, output_path)

    metadata = {
        "script": "urban_graph_benchmark/scripts/11_build_qc_cd_civil_security_panel.py",
        "purpose": "Build predictive Québec CD × month civil-security / SoVI panel.",
        "output_root": str(OUTPUT_ROOT),
        "inputs": {
            "cumulative_targets_path": str(cumulative_targets_path),
            "cd_month_targets_path": str(cd_month_targets_path),
        },
        "outputs": output_paths,
        "panel": {
            "rows": int(len(panel)),
            "columns": int(panel.shape[1]),
            "n_cd": int(panel[CD_ID_COL].nunique()),
            "start_month": start_month,
            "end_month": end_month,
            "drop_incomplete_targets": bool(drop_incomplete_targets),
        },
        "splits": (
            panel["split"]
            .astype("string")
            .fillna("<MISSING>")
            .value_counts(dropna=False)
            .rename_axis("split")
            .reset_index(name="n")
            .to_dict(orient="records")
        ),
        "event_count_columns": count_cols,
        "history_features": [
            c
            for c in panel.columns
            if c.endswith("_lag_1")
            or "_rolling_3" in c
            or "_rolling_6" in c
            or "_rolling_12" in c
            or c in {"lag_1", "rolling_3", "rolling_6", "rolling_12"}
        ],
        "target_columns": [
            "target_next_1_month",
            "target_next_3_months",
            "target_next_6_months",
            "target_next_1m_all",
            "target_next_3m_all",
            "target_next_6m_all",
        ],
        "target_completeness": {
            "target_next_1_month_complete_rows": int(panel["target_next_1_month"].notna().sum()),
            "target_next_3_months_complete_rows": int(panel["target_next_3_months"].notna().sum()),
            "target_next_6_months_complete_rows": int(panel["target_next_6_months"].notna().sum()),
        },
        "split_definition": {
            "train": "origin year 2021-2023",
            "val": "origin year 2024",
            "test": "origin year 2025",
            "note": (
                "Future target values are NaN when the complete future window is "
                "not observable within the requested panel end month."
            ),
        },
    }

    metadata_path = output_path.with_name("cd_month_panel_metadata.json")
    write_metadata_json(metadata, metadata_path)

    # Also write a compact schema/audit file for quick inspection.
    schema_rows = []
    for col in panel.columns:
        schema_rows.append(
            {
                "column": col,
                "dtype": str(panel[col].dtype),
                "nonmissing": int(panel[col].notna().sum()),
                "missing": int(panel[col].isna().sum()),
                "unique": int(panel[col].nunique(dropna=True)),
            }
        )
    schema = pd.DataFrame(schema_rows)
    schema_path = output_path.with_name("cd_month_panel_schema.csv")
    schema.to_csv(schema_path, index=False)

    metadata["outputs"]["metadata"] = str(metadata_path)
    metadata["outputs"]["schema"] = str(schema_path)

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Québec CD × month civil-security / SoVI predictive panel."
    )
    parser.add_argument(
        "--cumulative-targets-path",
        type=Path,
        default=DEFAULT_CUMULATIVE_TARGETS_PATH,
        help="CD-level cumulative SoVI/civil-security target table from step 04.",
    )
    parser.add_argument(
        "--cd-month-targets-path",
        type=Path,
        default=DEFAULT_CD_MONTH_TARGETS_PATH,
        help="CD-month SoVI/civil-security target table from step 04.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output parquet path for the predictive panel.",
    )
    parser.add_argument(
        "--start-month",
        default="2021-01",
        help="First origin month in the panel.",
    )
    parser.add_argument(
        "--end-month",
        default="2025-12",
        help="Last origin month in the panel.",
    )
    parser.add_argument(
        "--cd-id-col",
        default=None,
        help="Optional source CD ID column override.",
    )
    parser.add_argument(
        "--cd-name-col",
        default=None,
        help="Optional source CD name column override for the cumulative/SoVI frame.",
    )
    parser.add_argument(
        "--month-col",
        default=None,
        help="Optional source month column override for the CD-month event table.",
    )
    parser.add_argument(
        "--drop-incomplete-targets",
        action="store_true",
        help=(
            "Drop rows where target_next_3_months is not fully observable. "
            "By default, rows are kept and incomplete future targets are NaN."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_dir(DATASETS_DIR)

    metadata = build_panel(
        cumulative_targets_path=args.cumulative_targets_path,
        cd_month_targets_path=args.cd_month_targets_path,
        output_path=args.output_path,
        start_month=args.start_month,
        end_month=args.end_month,
        cd_id_col=args.cd_id_col,
        cd_name_col=args.cd_name_col,
        month_col=args.month_col,
        drop_incomplete_targets=args.drop_incomplete_targets,
    )

    print("Québec CD × month civil-security / SoVI panel completed.")
    print(f"Output panel: {args.output_path}")
    print(f"Rows: {metadata['panel']['rows']:,}")
    print(f"Columns: {metadata['panel']['columns']:,}")
    print(f"CDs: {metadata['panel']['n_cd']:,}")
    print("Splits:")
    for row in metadata["splits"]:
        print(f"  {row['split']}: {row['n']:,}")
    print("Outputs:")
    for key, value in metadata["outputs"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
