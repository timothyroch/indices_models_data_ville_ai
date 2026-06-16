#!/usr/bin/env python3
"""
Build CD-level civil-security event targets for SoVI validation.

This script is the target-construction step after:
    01_clean_quebec_civil_security_events.py
    02_spatial_join_quebec_civil_security_events.py
    03_check_cd_event_sovi_alignment.py

It does NOT run a SoVI benchmark/model/validation. It only builds clean target
tables aligned to the existing 98-CD SoVI frame.

Default inputs:
    data/external/quebec_civil_security_events/processed/quebec_civil_security_events_with_geographies.parquet
    data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/standard_output.csv

Default outputs:
    data/external/quebec_civil_security_events/processed/cd_civil_security_sovi_validation_targets_cumulative.parquet
    data/external/quebec_civil_security_events/processed/cd_civil_security_sovi_validation_targets_cumulative.csv
    data/external/quebec_civil_security_events/processed/cd_civil_security_sovi_validation_targets_cd_year.parquet
    data/external/quebec_civil_security_events/processed/cd_civil_security_sovi_validation_targets_cd_year.csv
    data/external/quebec_civil_security_events/processed/cd_civil_security_sovi_validation_targets_cd_month.parquet
    data/external/quebec_civil_security_events/processed/cd_civil_security_sovi_validation_targets_cd_month.csv

Default audits:
    data/external/quebec_civil_security_events/audits/cd_civil_security_sovi_validation_targets_summary.json
    data/external/quebec_civil_security_events/audits/cd_civil_security_excluded_events.csv
    data/external/quebec_civil_security_events/audits/cd_civil_security_target_density_audit.csv
    data/external/quebec_civil_security_events/audits/cd_civil_security_target_columns.csv
    data/external/quebec_civil_security_events/audits/cd_civil_security_sovi_column_audit.csv

Typical run:
    python data/external/quebec_civil_security_events/04_build_cd_civil_security_sovi_validation_targets.py

If needed:
    python data/external/quebec_civil_security_events/04_build_cd_civil_security_sovi_validation_targets.py \
      --sovi-cd-path data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/standard_output.csv \
      --sovi-cd-id-col zone_id

Scientific convention:
    - Keep exactly the 98 CDs in the SoVI frame.
    - Drop event CDs outside the SoVI frame, such as 1010 and 1314.
    - Drop unmatched CD events.
    - Preserve possible duplicate records, but expose duplicate-count target columns.
    - Primary recommended target: cumulative 2021-2025, all localization precision.
    - CD-year and CD-month tables are produced for diagnostics/robustness, not because
      monthly density is recommended as the first validation target.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


DEFAULT_EVENTS_WITH_GEOGRAPHIES = Path(
    "data/external/quebec_civil_security_events/processed/"
    "quebec_civil_security_events_with_geographies.parquet"
)

DEFAULT_SOVI_CD_PATH = Path(
    "data/sovi_2021/output/"
    "sovi_like_quebec_cd_2021_38var_oriented_run/"
    "standard_output.csv"
)

DEFAULT_PROCESSED_DIR = Path("data/external/quebec_civil_security_events/processed")
DEFAULT_AUDIT_DIR = Path("data/external/quebec_civil_security_events/audits")

SOVI_SEARCH_ROOTS = [
    Path("data/sovi_2021/output"),
    Path("urban_graph_benchmark/outputs"),
    Path("data"),
    Path("."),
]

TABLE_EXTENSIONS = {".parquet", ".csv", ".txt", ".xlsx", ".xls", ".geojson", ".json", ".gpkg"}

SOVI_ID_CANDIDATES = [
    "zone_id",
    "census_division_dguid",
    "cd_id",
    "cd_uid",
    "CDUID",
    "CDUID_2021",
    "DGUID",
    "dguid",
    "geo_id",
    "geo_uid",
    "GEO_CODE",
    "geocode",
    "geography_id",
    "census_division_id",
    "census_division_uid",
]

SOVI_NAME_CANDIDATES = [
    "cd_name",
    "CDNAME",
    "CDNAME_2021",
    "name",
    "GEO_NAME",
    "geography_name",
    "census_division_name",
    "zone_name",
]

EVENT_CD_ID_COL = "cd_id"
EVENT_CD_NAME_COL = "cd_name"
EVENT_CD_JOIN_COL = "cd_join_success"

PRIMARY_WINDOW_NAME = "2021_2025"
ROBUSTNESS_WINDOW_NAME = "2022_2025"

WINDOWS = {
    "2021_2025": [2021, 2022, 2023, 2024, 2025],
    "2022_2025": [2022, 2023, 2024, 2025],
}

PRECISION_FILTERS = {
    "all": None,
    "precise_or_very_precise": "is_precise_or_very_precise",
    "very_precise": "is_very_precise",
}

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

SEVERITY_TARGETS = {
    "moderate_or_worse": "is_moderate_or_worse",
    "important_or_extreme": "is_important_or_extreme",
}

RECOMMENDED_PRIMARY_TARGET_GRANULARITY = "cd_cumulative_2021_2025"
RECOMMENDED_PRIMARY_PRECISION_FILTER = "all"
RECOMMENDED_REASON = (
    "The SoVI score is static and CD-level; all 98 SoVI CDs have event coverage "
    "over 2021-2025, while monthly CD cells are sparse. Cumulative 2021-2025 "
    "therefore provides the cleanest first validation target."
)


@dataclass(frozen=True)
class Config:
    events_with_geographies_path: Path
    sovi_cd_path: Path | None
    sovi_cd_id_col: str | None
    sovi_cd_name_col: str | None
    processed_dir: Path
    audit_dir: Path
    include_monthly: bool
    fail_if_sovi_missing: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Build CD-level civil-security event targets aligned to the SoVI CD frame."
    )
    parser.add_argument("--events-with-geographies", type=Path, default=DEFAULT_EVENTS_WITH_GEOGRAPHIES)
    parser.add_argument("--sovi-cd-path", type=Path, default=None)
    parser.add_argument("--sovi-cd-id-col", default=None)
    parser.add_argument("--sovi-cd-name-col", default=None)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument(
        "--no-monthly",
        action="store_true",
        help="Do not write the CD-month target table.",
    )
    parser.add_argument(
        "--fail-if-sovi-missing",
        action="store_true",
        help="Fail if the default/auto-detected SoVI file is unavailable.",
    )
    args = parser.parse_args()

    sovi_path = args.sovi_cd_path
    if sovi_path is None and DEFAULT_SOVI_CD_PATH.exists():
        sovi_path = DEFAULT_SOVI_CD_PATH

    return Config(
        events_with_geographies_path=args.events_with_geographies,
        sovi_cd_path=sovi_path,
        sovi_cd_id_col=args.sovi_cd_id_col,
        sovi_cd_name_col=args.sovi_cd_name_col,
        processed_dir=args.processed_dir,
        audit_dir=args.audit_dir,
        include_monthly=not args.no_monthly,
        fail_if_sovi_missing=bool(args.fail_if_sovi_missing),
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
    if suffix in {".geojson", ".json", ".gpkg"}:
        try:
            import geopandas as gpd
        except Exception as exc:
            raise RuntimeError(f"Reading {suffix} requires geopandas: {path}") from exc
        return pd.DataFrame(gpd.read_file(path))

    raise ValueError(f"Unsupported table extension: {path}")


def normalize_col_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(col).lower())


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


def normalize_cd_id(value: Any) -> str | None:
    """
    Normalize CD identifiers to a comparable 4-digit CDUID where possible.

    Handles:
    - CDUID values like 2401
    - numeric values like 2401.0
    - DGUID values like 2021A00032401
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "<missing>", "<empty>"}:
        return None

    m = re.search(r"A0003(\d{4})$", text)
    if m:
        return m.group(1)

    if "A0003" in text:
        digits = re.findall(r"\d+", text)
        if digits:
            tail = digits[-1]
            if len(tail) >= 4:
                return tail[-4:]

    if re.fullmatch(r"\d+(\.0+)?", text):
        return f"{int(float(text)):04d}"

    digits_only = re.sub(r"\D", "", text)
    if len(digits_only) == 4:
        return digits_only
    if len(digits_only) > 4 and "24" in digits_only:
        q = re.search(r"(24\d{2})", digits_only)
        if q:
            return q.group(1)

    return text


def candidate_sovi_files() -> list[Path]:
    candidates = []
    for root in SOVI_SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TABLE_EXTENSIONS:
                continue

            low = str(path).lower()
            if "civil_security" in low or "quebec_civil_security_events" in low:
                continue
            if "sovi" not in low:
                continue
            if any(skip in low for skip in ["/audits/", "/plots/", "/figures/", "/maps/"]):
                continue
            if path.name.startswith("~$"):
                continue
            candidates.append(path)

    def score(path: Path) -> tuple[int, int, str]:
        low = str(path).lower()
        s = 0
        if "sovi_like_quebec_cd_2021_38var_oriented_run" in low:
            s -= 100
        if "standard_output" in low:
            s -= 50
        if "quebec" in low:
            s -= 20
        if "/cd" in low or "_cd" in low or "census_division" in low:
            s -= 20
        if path.suffix.lower() == ".parquet":
            s -= 5
        if path.suffix.lower() == ".csv":
            s -= 3
        if "raw" in low:
            s += 20
        return (s, len(str(path)), str(path))

    return sorted(set(candidates), key=score)


def detect_sovi_file(event_ids_norm: set[str]) -> tuple[Path | None, pd.DataFrame]:
    candidates = candidate_sovi_files()
    rows = []
    best_path = None
    best_score = (-1, -1, -1, -1)

    for path in candidates:
        try:
            df = read_table(path)
        except Exception as exc:
            rows.append(
                {
                    "path": str(path),
                    "read_ok": False,
                    "error": str(exc),
                    "rows": None,
                    "columns": None,
                    "best_id_col": None,
                    "overlap_with_event_cd_ids": 0,
                    "unique_normalized_ids": 0,
                    "has_sovi_column": False,
                }
            )
            continue

        id_col, diag = detect_sovi_id_column(df, event_ids_norm, None)
        if id_col is None:
            overlap = 0
            unique_ids = 0
        else:
            norm = df[id_col].map(normalize_cd_id)
            overlap = len(set(norm.dropna()) & event_ids_norm)
            unique_ids = int(norm.dropna().nunique())

        has_sovi_col = any("sovi" in str(c).lower() for c in df.columns)
        rows.append(
            {
                "path": str(path),
                "read_ok": True,
                "error": "",
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "best_id_col": id_col,
                "overlap_with_event_cd_ids": int(overlap),
                "unique_normalized_ids": int(unique_ids),
                "has_sovi_column": bool(has_sovi_col),
            }
        )

        score = (
            int(overlap),
            int(has_sovi_col),
            int(unique_ids),
            -abs(len(df) - 98),
        )
        if score > best_score and overlap > 0:
            best_score = score
            best_path = path

    audit = pd.DataFrame(rows)
    if not audit.empty:
        audit = audit.sort_values(
            ["overlap_with_event_cd_ids", "has_sovi_column", "unique_normalized_ids"],
            ascending=False,
        )
    return best_path, audit


def detect_sovi_id_column(
    sovi: pd.DataFrame,
    event_ids_norm: set[str],
    explicit_col: str | None,
) -> tuple[str | None, pd.DataFrame]:
    diagnostics = []

    if explicit_col is not None:
        if explicit_col not in sovi.columns:
            raise ValueError(f"Requested SoVI CD ID column not found: {explicit_col}")
        norm = sovi[explicit_col].map(normalize_cd_id)
        diagnostics.append(
            {
                "column": explicit_col,
                "candidate_type": "explicit",
                "non_missing_normalized": int(norm.notna().sum()),
                "unique_normalized": int(norm.dropna().nunique()),
                "overlap_with_event_cd_ids": int(len(set(norm.dropna()) & event_ids_norm)),
            }
        )
        return explicit_col, pd.DataFrame(diagnostics)

    candidate_cols = []
    normalized_candidates = {normalize_col_name(c) for c in SOVI_ID_CANDIDATES}
    for col in sovi.columns:
        col_s = str(col)
        low = col_s.lower()
        if col_s in SOVI_ID_CANDIDATES or normalize_col_name(col_s) in normalized_candidates:
            candidate_cols.append(col_s)
        elif any(token in low for token in ["zone", "cduid", "dguid", "cd_id", "cd_uid", "geo_code", "geocode"]):
            candidate_cols.append(col_s)

    seen = set()
    candidate_cols = [c for c in candidate_cols if not (c in seen or seen.add(c))]

    best_col = None
    best_score = (-1, -1, -1)

    for col in candidate_cols:
        norm = sovi[col].map(normalize_cd_id)
        normalized_set = set(norm.dropna())
        overlap = len(normalized_set & event_ids_norm)
        score = (overlap, int(norm.notna().sum()), int(norm.dropna().nunique()))
        diagnostics.append(
            {
                "column": col,
                "candidate_type": "auto",
                "non_missing_normalized": int(norm.notna().sum()),
                "unique_normalized": int(norm.dropna().nunique()),
                "overlap_with_event_cd_ids": int(overlap),
            }
        )
        if score > best_score:
            best_score = score
            best_col = col

    diag = pd.DataFrame(diagnostics)
    if not diag.empty:
        diag = diag.sort_values(
            ["overlap_with_event_cd_ids", "non_missing_normalized", "unique_normalized"],
            ascending=False,
        )
    return best_col, diag


def detect_sovi_name_column(sovi: pd.DataFrame, explicit_col: str | None) -> str | None:
    if explicit_col is not None:
        if explicit_col not in sovi.columns:
            raise ValueError(f"Requested SoVI CD name column not found: {explicit_col}")
        return explicit_col
    return choose_first_existing(sovi.columns, SOVI_NAME_CANDIDATES)


def ensure_event_columns(events: pd.DataFrame) -> None:
    required = [
        "event_id",
        "event_date_primary",
        "event_period_month",
        EVENT_CD_ID_COL,
        EVENT_CD_JOIN_COL,
        "alea_group",
        "severite",
        "is_precise_or_very_precise",
        "is_very_precise",
        "is_moderate_or_worse",
        "is_important_or_extreme",
        "possible_duplicate_flag",
    ]
    missing = [c for c in required if c not in events.columns]
    if missing:
        raise ValueError(f"Event-with-geographies file is missing required columns: {missing}")


def prepare_events(events: pd.DataFrame, sovi_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_event_columns(events)
    out = events.copy()

    out["cd_id_norm"] = out[EVENT_CD_ID_COL].map(normalize_cd_id)
    out["event_date_primary"] = pd.to_datetime(out["event_date_primary"], errors="coerce")
    out["event_year_clean"] = out["event_date_primary"].dt.year.astype("Int64")
    out["event_period_month"] = out["event_date_primary"].dt.to_period("M").astype("string")

    out["exclude_reason"] = pd.NA
    out.loc[~out[EVENT_CD_JOIN_COL].fillna(False), "exclude_reason"] = "unmatched_cd"
    out.loc[
        out["exclude_reason"].isna() & out["cd_id_norm"].notna() & ~out["cd_id_norm"].isin(sovi_ids),
        "exclude_reason",
    ] = "event_cd_not_in_sovi"
    out.loc[
        out["exclude_reason"].isna() & out["cd_id_norm"].isna(),
        "exclude_reason",
    ] = "missing_cd_id"
    out.loc[
        out["exclude_reason"].isna() & out["event_date_primary"].isna(),
        "exclude_reason",
    ] = "missing_event_date_primary"

    excluded = out[out["exclude_reason"].notna()].copy()
    keep = out[out["exclude_reason"].isna()].copy()

    return keep, excluded


def precision_mask(events: pd.DataFrame, precision_filter: str) -> pd.Series:
    filter_col = PRECISION_FILTERS[precision_filter]
    if filter_col is None:
        return pd.Series(True, index=events.index)
    return events[filter_col].fillna(False)


def window_mask(events: pd.DataFrame, years: list[int]) -> pd.Series:
    return events["event_year_clean"].isin(years)


def init_target_frame(sovi: pd.DataFrame, id_norm_col: str, name_col: str | None) -> pd.DataFrame:
    out = sovi.copy()
    if "cd_id_norm" not in out.columns:
        out = out.rename(columns={id_norm_col: "_original_cd_id_norm_for_targets"})
        out["cd_id_norm"] = out["_original_cd_id_norm_for_targets"]
    else:
        out["cd_id_norm"] = out[id_norm_col]

    if name_col is not None and "cd_name" not in out.columns:
        out["cd_name"] = out[name_col]
    elif "cd_name" not in out.columns:
        out["cd_name"] = pd.NA

    # Put key columns first.
    key_cols = ["cd_id_norm", "cd_name"]
    remaining = [c for c in out.columns if c not in key_cols]
    return out[key_cols + remaining].copy()


def add_count_column(
    target: pd.DataFrame,
    events: pd.DataFrame,
    *,
    column_name: str,
    mask: pd.Series,
    cd_col: str = "cd_id_norm",
) -> None:
    counts = events.loc[mask].groupby(cd_col, dropna=False)["event_id"].size()
    target[column_name] = target["cd_id_norm"].map(counts).fillna(0).astype(int)


def add_window_targets(
    target: pd.DataFrame,
    events: pd.DataFrame,
    *,
    window_name: str,
    years: list[int],
) -> None:
    base_window = window_mask(events, years)

    for precision_name in PRECISION_FILTERS:
        base = base_window & precision_mask(events, precision_name)
        prefix = f"event_count_{window_name}_{precision_name}"

        add_count_column(target, events, column_name=prefix, mask=base)

        for group in HAZARD_GROUPS:
            add_count_column(
                target,
                events,
                column_name=f"{prefix}_{group}",
                mask=base & events["alea_group"].astype("string").eq(group),
            )

        for sev_name, sev_col in SEVERITY_TARGETS.items():
            add_count_column(
                target,
                events,
                column_name=f"{prefix}_{sev_name}",
                mask=base & events[sev_col].fillna(False),
            )

        add_count_column(
            target,
            events,
            column_name=f"{prefix}_possible_duplicate",
            mask=base & events["possible_duplicate_flag"].fillna(False),
        )


def build_cumulative_targets(sovi: pd.DataFrame, id_norm_col: str, name_col: str | None, events: pd.DataFrame) -> pd.DataFrame:
    target = init_target_frame(sovi, id_norm_col, name_col)

    for window_name, years in WINDOWS.items():
        add_window_targets(target, events, window_name=window_name, years=years)

    # Convenience aliases for the recommended target.
    target["recommended_primary_target"] = target[f"event_count_{PRIMARY_WINDOW_NAME}_all"]
    target["recommended_primary_target_name"] = f"event_count_{PRIMARY_WINDOW_NAME}_all"
    target["recommended_primary_target_granularity"] = RECOMMENDED_PRIMARY_TARGET_GRANULARITY
    target["recommended_primary_precision_filter"] = RECOMMENDED_PRIMARY_PRECISION_FILTER

    return target.sort_values("cd_id_norm").reset_index(drop=True)


def build_cd_year_targets(sovi: pd.DataFrame, id_norm_col: str, name_col: str | None, events: pd.DataFrame) -> pd.DataFrame:
    years = sorted(events["event_year_clean"].dropna().astype(int).unique().tolist())
    if not years:
        return pd.DataFrame()

    base = init_target_frame(sovi, id_norm_col, name_col)[["cd_id_norm", "cd_name"]].copy()
    grid = pd.MultiIndex.from_product(
        [base["cd_id_norm"].astype(str).tolist(), years],
        names=["cd_id_norm", "event_year"],
    ).to_frame(index=False)
    grid = grid.merge(base, on="cd_id_norm", how="left")

    for precision_name in PRECISION_FILTERS:
        mask_precision = precision_mask(events, precision_name)
        prefix = f"event_count_{precision_name}"

        grouped = (
            events.loc[mask_precision]
            .groupby(["cd_id_norm", "event_year_clean"], dropna=False)["event_id"]
            .size()
            .reset_index(name=prefix)
            .rename(columns={"event_year_clean": "event_year"})
        )
        grid = grid.merge(grouped, on=["cd_id_norm", "event_year"], how="left")
        grid[prefix] = grid[prefix].fillna(0).astype(int)

        for group in HAZARD_GROUPS:
            col = f"{prefix}_{group}"
            grouped_g = (
                events.loc[mask_precision & events["alea_group"].astype("string").eq(group)]
                .groupby(["cd_id_norm", "event_year_clean"], dropna=False)["event_id"]
                .size()
                .reset_index(name=col)
                .rename(columns={"event_year_clean": "event_year"})
            )
            grid = grid.merge(grouped_g, on=["cd_id_norm", "event_year"], how="left")
            grid[col] = grid[col].fillna(0).astype(int)

        for sev_name, sev_col in SEVERITY_TARGETS.items():
            col = f"{prefix}_{sev_name}"
            grouped_s = (
                events.loc[mask_precision & events[sev_col].fillna(False)]
                .groupby(["cd_id_norm", "event_year_clean"], dropna=False)["event_id"]
                .size()
                .reset_index(name=col)
                .rename(columns={"event_year_clean": "event_year"})
            )
            grid = grid.merge(grouped_s, on=["cd_id_norm", "event_year"], how="left")
            grid[col] = grid[col].fillna(0).astype(int)

    grid["has_any_event_all"] = grid["event_count_all"] > 0
    return grid.sort_values(["cd_id_norm", "event_year"]).reset_index(drop=True)


def build_cd_month_targets(sovi: pd.DataFrame, id_norm_col: str, name_col: str | None, events: pd.DataFrame) -> pd.DataFrame:
    months = sorted(events["event_period_month"].dropna().astype(str).unique().tolist())
    if not months:
        return pd.DataFrame()

    base = init_target_frame(sovi, id_norm_col, name_col)[["cd_id_norm", "cd_name"]].copy()
    grid = pd.MultiIndex.from_product(
        [base["cd_id_norm"].astype(str).tolist(), months],
        names=["cd_id_norm", "event_period_month"],
    ).to_frame(index=False)
    grid = grid.merge(base, on="cd_id_norm", how="left")

    for precision_name in PRECISION_FILTERS:
        mask_precision = precision_mask(events, precision_name)
        prefix = f"event_count_{precision_name}"

        grouped = (
            events.loc[mask_precision]
            .groupby(["cd_id_norm", "event_period_month"], dropna=False)["event_id"]
            .size()
            .reset_index(name=prefix)
        )
        grid = grid.merge(grouped, on=["cd_id_norm", "event_period_month"], how="left")
        grid[prefix] = grid[prefix].fillna(0).astype(int)

        for group in HAZARD_GROUPS:
            col = f"{prefix}_{group}"
            grouped_g = (
                events.loc[mask_precision & events["alea_group"].astype("string").eq(group)]
                .groupby(["cd_id_norm", "event_period_month"], dropna=False)["event_id"]
                .size()
                .reset_index(name=col)
            )
            grid = grid.merge(grouped_g, on=["cd_id_norm", "event_period_month"], how="left")
            grid[col] = grid[col].fillna(0).astype(int)

        for sev_name, sev_col in SEVERITY_TARGETS.items():
            col = f"{prefix}_{sev_name}"
            grouped_s = (
                events.loc[mask_precision & events[sev_col].fillna(False)]
                .groupby(["cd_id_norm", "event_period_month"], dropna=False)["event_id"]
                .size()
                .reset_index(name=col)
            )
            grid = grid.merge(grouped_s, on=["cd_id_norm", "event_period_month"], how="left")
            grid[col] = grid[col].fillna(0).astype(int)

    grid["has_any_event_all"] = grid["event_count_all"] > 0
    return grid.sort_values(["cd_id_norm", "event_period_month"]).reset_index(drop=True)


def target_density_audit(cumulative: pd.DataFrame, cd_year: pd.DataFrame, cd_month: pd.DataFrame) -> pd.DataFrame:
    rows = []

    count_cols_cum = [c for c in cumulative.columns if c.startswith("event_count_")]
    for col in count_cols_cum:
        values = pd.to_numeric(cumulative[col], errors="coerce").fillna(0)
        rows.append(
            {
                "table": "cumulative",
                "target_column": col,
                "rows": int(len(values)),
                "total_events": int(values.sum()),
                "nonzero_rows": int((values > 0).sum()),
                "nonzero_rate": float((values > 0).mean()) if len(values) else math.nan,
                "mean": float(values.mean()) if len(values) else math.nan,
                "median": float(values.median()) if len(values) else math.nan,
                "max": float(values.max()) if len(values) else math.nan,
            }
        )

    for table_name, df in [("cd_year", cd_year), ("cd_month", cd_month)]:
        if df.empty:
            continue
        count_cols = [c for c in df.columns if c.startswith("event_count_")]
        for col in count_cols:
            values = pd.to_numeric(df[col], errors="coerce").fillna(0)
            rows.append(
                {
                    "table": table_name,
                    "target_column": col,
                    "rows": int(len(values)),
                    "total_events": int(values.sum()),
                    "nonzero_rows": int((values > 0).sum()),
                    "nonzero_rate": float((values > 0).mean()) if len(values) else math.nan,
                    "mean": float(values.mean()) if len(values) else math.nan,
                    "median": float(values.median()) if len(values) else math.nan,
                    "max": float(values.max()) if len(values) else math.nan,
                }
            )

    return pd.DataFrame(rows)


def column_audit(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "non_missing": [int(df[c].notna().sum()) for c in df.columns],
            "missing": [int(df[c].isna().sum()) for c in df.columns],
            "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )


def write_table(df: pd.DataFrame, parquet_path: Path) -> dict[str, str]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = parquet_path.with_suffix(".csv")
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    return {"parquet": str(parquet_path), "csv": str(csv_path)}


def write_json(data: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def main() -> None:
    config = parse_args()
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.audit_dir.mkdir(parents=True, exist_ok=True)

    events_raw = read_table(config.events_with_geographies_path)
    event_ids_for_sovi_detection = set(events_raw[EVENT_CD_ID_COL].map(normalize_cd_id).dropna()) if EVENT_CD_ID_COL in events_raw.columns else set()

    sovi_path = config.sovi_cd_path
    sovi_candidate_audit = pd.DataFrame()
    if sovi_path is None:
        sovi_path, sovi_candidate_audit = detect_sovi_file(event_ids_for_sovi_detection)
        sovi_candidate_audit.to_csv(
            config.audit_dir / "cd_civil_security_sovi_validation_sovi_file_candidates.csv",
            index=False,
        )

    if sovi_path is None:
        if config.fail_if_sovi_missing:
            raise FileNotFoundError("Could not locate SoVI CD output. Pass --sovi-cd-path explicitly.")
        raise FileNotFoundError(
            "Could not locate SoVI CD output. Pass --sovi-cd-path explicitly, for example:\n"
            "  --sovi-cd-path data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/standard_output.csv"
        )

    sovi = read_table(sovi_path)

    # Need event IDs before detecting SoVI ID column.
    # Use joined event CDs because outside-SoVI event CDs can exist.
    ensure_event_columns(events_raw)
    joined_event_ids = set(
        events_raw.loc[events_raw[EVENT_CD_JOIN_COL].fillna(False), EVENT_CD_ID_COL]
        .map(normalize_cd_id)
        .dropna()
    )

    sovi_id_col, sovi_id_diag = detect_sovi_id_column(sovi, joined_event_ids, config.sovi_cd_id_col)
    if sovi_id_col is None:
        raise ValueError(
            f"Could not detect the SoVI CD ID column in {sovi_path}. "
            "Pass --sovi-cd-id-col zone_id or --sovi-cd-id-col census_division_dguid."
        )

    sovi_name_col = detect_sovi_name_column(sovi, config.sovi_cd_name_col)
    sovi["_cd_id_norm_for_targets"] = sovi[sovi_id_col].map(normalize_cd_id)
    sovi_ids = set(sovi["_cd_id_norm_for_targets"].dropna())

    if len(sovi_ids) != len(sovi):
        duplicate_sovi_ids = (
            sovi["_cd_id_norm_for_targets"]
            .value_counts(dropna=False)
            .loc[lambda s: s > 1]
            .reset_index()
            .rename(columns={"index": "cd_id_norm", "_cd_id_norm_for_targets": "n"})
        )
        duplicate_sovi_ids.to_csv(config.audit_dir / "cd_civil_security_duplicate_sovi_ids.csv", index=False)
        raise ValueError(
            f"SoVI CD frame has {len(sovi)} rows but {len(sovi_ids)} unique normalized CD IDs. "
            "Duplicate/invalid IDs were written to cd_civil_security_duplicate_sovi_ids.csv."
        )

    sovi_id_diag.to_csv(
        config.audit_dir / "cd_civil_security_sovi_validation_sovi_id_column_diagnostics.csv",
        index=False,
    )

    events_kept, events_excluded = prepare_events(events_raw, sovi_ids)

    cumulative = build_cumulative_targets(
        sovi=sovi,
        id_norm_col="_cd_id_norm_for_targets",
        name_col=sovi_name_col,
        events=events_kept,
    )

    cd_year = build_cd_year_targets(
        sovi=sovi,
        id_norm_col="_cd_id_norm_for_targets",
        name_col=sovi_name_col,
        events=events_kept,
    )

    cd_month = (
        build_cd_month_targets(
            sovi=sovi,
            id_norm_col="_cd_id_norm_for_targets",
            name_col=sovi_name_col,
            events=events_kept,
        )
        if config.include_monthly
        else pd.DataFrame()
    )

    outputs = {}
    outputs["cumulative"] = write_table(
        cumulative,
        config.processed_dir / "cd_civil_security_sovi_validation_targets_cumulative.parquet",
    )
    outputs["cd_year"] = write_table(
        cd_year,
        config.processed_dir / "cd_civil_security_sovi_validation_targets_cd_year.parquet",
    )
    if config.include_monthly:
        outputs["cd_month"] = write_table(
            cd_month,
            config.processed_dir / "cd_civil_security_sovi_validation_targets_cd_month.parquet",
        )

    # Audits.
    exclude_cols = [
        c for c in [
            "event_id",
            "exclude_reason",
            "cd_id",
            "cd_id_norm",
            "cd_name",
            "cd_join_success",
            "alea",
            "alea_group",
            "severite",
            "precision_localisation",
            "municipalite",
            "code_municipalite",
            "event_date_primary",
            "event_period_month",
            "lon",
            "lat",
            "inside_loose_quebec_bbox",
        ]
        if c in events_excluded.columns
    ]
    events_excluded[exclude_cols].to_csv(
        config.audit_dir / "cd_civil_security_excluded_events.csv",
        index=False,
    )

    target_density = target_density_audit(cumulative, cd_year, cd_month)
    target_density.to_csv(config.audit_dir / "cd_civil_security_target_density_audit.csv", index=False)

    column_audit(cumulative).to_csv(config.audit_dir / "cd_civil_security_target_columns.csv", index=False)
    column_audit(sovi).to_csv(config.audit_dir / "cd_civil_security_sovi_column_audit.csv", index=False)

    excluded_summary = (
        events_excluded["exclude_reason"]
        .astype("string")
        .fillna("<MISSING>")
        .value_counts(dropna=False)
        .rename_axis("exclude_reason")
        .reset_index(name="n")
    )
    excluded_summary["share_of_all_events"] = excluded_summary["n"] / len(events_raw)
    excluded_summary.to_csv(config.audit_dir / "cd_civil_security_excluded_events_summary.csv", index=False)

    # Direct sanity checks.
    primary_col = f"event_count_{PRIMARY_WINDOW_NAME}_all"
    cd_year_2021_2025 = cd_year[cd_year["event_year"].isin(WINDOWS[PRIMARY_WINDOW_NAME])] if not cd_year.empty else pd.DataFrame()
    cd_month_2021_2025 = cd_month[
        cd_month["event_period_month"].astype("string").str.slice(0, 4).astype("Int64").isin(WINDOWS[PRIMARY_WINDOW_NAME])
    ] if not cd_month.empty else pd.DataFrame()

    summary = {
        "status": "completed",
        "events_with_geographies_path": str(config.events_with_geographies_path),
        "sovi_cd_path": str(sovi_path),
        "sovi_cd_id_col": sovi_id_col,
        "sovi_cd_name_col": sovi_name_col,
        "processed_dir": str(config.processed_dir),
        "audit_dir": str(config.audit_dir),

        "n_raw_events": int(len(events_raw)),
        "n_events_kept_for_sovi_targets": int(len(events_kept)),
        "n_events_excluded": int(len(events_excluded)),
        "excluded_events_by_reason": excluded_summary.set_index("exclude_reason")["n"].to_dict() if not excluded_summary.empty else {},

        "n_sovi_rows": int(len(sovi)),
        "n_sovi_unique_cd_ids": int(len(sovi_ids)),
        "target_rows_cumulative": int(len(cumulative)),
        "target_rows_cd_year": int(len(cd_year)),
        "target_rows_cd_month": int(len(cd_month)) if config.include_monthly else None,

        "primary_target_column": primary_col,
        "primary_target_total_events": int(cumulative[primary_col].sum()) if primary_col in cumulative.columns else None,
        "primary_target_nonzero_cds": int((cumulative[primary_col] > 0).sum()) if primary_col in cumulative.columns else None,
        "primary_target_zero_cds": int((cumulative[primary_col] == 0).sum()) if primary_col in cumulative.columns else None,
        "primary_target_mean_events_per_cd": float(cumulative[primary_col].mean()) if primary_col in cumulative.columns else None,
        "primary_target_median_events_per_cd": float(cumulative[primary_col].median()) if primary_col in cumulative.columns else None,

        "cd_year_2021_2025_nonzero_cells": int((cd_year_2021_2025["event_count_all"] > 0).sum()) if not cd_year_2021_2025.empty and "event_count_all" in cd_year_2021_2025.columns else None,
        "cd_year_2021_2025_total_cells": int(len(cd_year_2021_2025)) if not cd_year_2021_2025.empty else None,
        "cd_year_2021_2025_nonzero_rate": float((cd_year_2021_2025["event_count_all"] > 0).mean()) if not cd_year_2021_2025.empty and "event_count_all" in cd_year_2021_2025.columns else None,

        "cd_month_2021_2025_nonzero_cells": int((cd_month_2021_2025["event_count_all"] > 0).sum()) if not cd_month_2021_2025.empty and "event_count_all" in cd_month_2021_2025.columns else None,
        "cd_month_2021_2025_total_cells": int(len(cd_month_2021_2025)) if not cd_month_2021_2025.empty else None,
        "cd_month_2021_2025_nonzero_rate": float((cd_month_2021_2025["event_count_all"] > 0).mean()) if not cd_month_2021_2025.empty and "event_count_all" in cd_month_2021_2025.columns else None,

        "recommended_primary_target_granularity": RECOMMENDED_PRIMARY_TARGET_GRANULARITY,
        "recommended_primary_precision_filter": RECOMMENDED_PRIMARY_PRECISION_FILTER,
        "recommended_primary_target_column": primary_col,
        "recommended_reason": RECOMMENDED_REASON,

        "outputs": outputs,
    }

    write_json(summary, config.audit_dir / "cd_civil_security_sovi_validation_targets_summary.json")

    print("CD civil-security SoVI validation target construction completed.")
    print(f"SoVI file: {sovi_path}")
    print(f"SoVI ID column: {sovi_id_col}")
    print(f"SoVI rows: {len(sovi):,}")
    print(f"Raw events: {len(events_raw):,}")
    print(f"Events kept for SoVI targets: {len(events_kept):,}")
    print(f"Events excluded: {len(events_excluded):,}")
    print()
    print("Outputs:")
    for name, paths in outputs.items():
        print(f"  {name}: {paths['parquet']}")
    print()
    print("Recommended primary target:")
    print(f"  {primary_col}")
    print(f"  granularity: {RECOMMENDED_PRIMARY_TARGET_GRANULARITY}")
    print(f"  precision filter: {RECOMMENDED_PRIMARY_PRECISION_FILTER}")
    print(f"  reason: {RECOMMENDED_REASON}")


if __name__ == "__main__":
    main()
