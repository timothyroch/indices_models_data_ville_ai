#!/usr/bin/env python3
"""
Check CD-level alignment between Québec civil-security events and the existing SoVI CD frame.

This is an audit/feasibility script only. It does not run a SoVI validation model.

Default inputs:
    data/external/quebec_civil_security_events/processed/quebec_civil_security_events_with_geographies.parquet
    data/external/quebec_civil_security_events/processed/civil_security_events_by_cd__all.parquet
    data/external/quebec_civil_security_events/processed/civil_security_events_by_cd_month__all.parquet

Required or auto-detected:
    an existing SoVI CD output file from the sovi_like_quebec_cd_2021_38var_oriented_run family

Example:
    python data/external/quebec_civil_security_events/03_check_cd_event_sovi_alignment.py

If auto-detection fails:
    python data/external/quebec_civil_security_events/03_check_cd_event_sovi_alignment.py \
      --sovi-cd-path <PATH_TO_SOVI_CD_OUTPUT>

Optional:
    python data/external/quebec_civil_security_events/03_check_cd_event_sovi_alignment.py \
      --sovi-cd-id-col CDUID \
      --sovi-cd-name-col CDNAME

Outputs:
    data/external/quebec_civil_security_events/audits/cd_sovi_alignment_summary.json
    data/external/quebec_civil_security_events/audits/cd_sovi_alignment_event_ids_not_in_sovi.csv
    data/external/quebec_civil_security_events/audits/cd_sovi_alignment_sovi_ids_not_in_events.csv
    data/external/quebec_civil_security_events/audits/unmatched_cd_events.csv
    data/external/quebec_civil_security_events/audits/event_density_by_year.csv
    data/external/quebec_civil_security_events/audits/event_density_by_cd.csv
    data/external/quebec_civil_security_events/audits/event_density_by_cd_year.csv
    data/external/quebec_civil_security_events/audits/event_density_by_cd_month.csv
    data/external/quebec_civil_security_events/audits/event_density_by_hazard_group.csv
    data/external/quebec_civil_security_events/audits/event_density_by_severity.csv
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
DEFAULT_EVENTS_BY_CD = Path(
    "data/external/quebec_civil_security_events/processed/"
    "civil_security_events_by_cd__all.parquet"
)
DEFAULT_EVENTS_BY_CD_MONTH = Path(
    "data/external/quebec_civil_security_events/processed/"
    "civil_security_events_by_cd_month__all.parquet"
)
DEFAULT_AUDIT_DIR = Path("data/external/quebec_civil_security_events/audits")

SOVI_SEARCH_ROOTS = [
    Path("urban_graph_benchmark/outputs"),
    Path("data"),
    Path("."),
]

TABLE_EXTENSIONS = {
    ".parquet",
    ".csv",
    ".txt",
    ".xlsx",
    ".xls",
    ".geojson",
    ".json",
    ".gpkg",
}

SOVI_ID_CANDIDATES = [
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
    "municipality_name",
]

EVENT_CD_ID_COL = "cd_id"
EVENT_CD_NAME_COL = "cd_name"
EVENT_CD_JOIN_COL = "cd_join_success"

TARGET_FULL_YEARS_2021_2025 = [2021, 2022, 2023, 2024, 2025]
TARGET_FULL_YEARS_2022_2025 = [2022, 2023, 2024, 2025]


@dataclass(frozen=True)
class Config:
    events_with_geographies_path: Path
    events_by_cd_path: Path
    events_by_cd_month_path: Path
    sovi_cd_path: Path | None
    sovi_cd_id_col: str | None
    sovi_cd_name_col: str | None
    audit_dir: Path
    fail_if_sovi_missing: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Audit CD alignment between civil-security events and the existing SoVI CD frame."
    )
    parser.add_argument("--events-with-geographies", type=Path, default=DEFAULT_EVENTS_WITH_GEOGRAPHIES)
    parser.add_argument("--events-by-cd", type=Path, default=DEFAULT_EVENTS_BY_CD)
    parser.add_argument("--events-by-cd-month", type=Path, default=DEFAULT_EVENTS_BY_CD_MONTH)
    parser.add_argument("--sovi-cd-path", type=Path, default=None)
    parser.add_argument("--sovi-cd-id-col", default=None)
    parser.add_argument("--sovi-cd-name-col", default=None)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument(
        "--fail-if-sovi-missing",
        action="store_true",
        help="Fail instead of writing partial audits if the SoVI CD frame cannot be auto-detected.",
    )
    args = parser.parse_args()
    return Config(
        events_with_geographies_path=args.events_with_geographies,
        events_by_cd_path=args.events_by_cd,
        events_by_cd_month_path=args.events_by_cd_month,
        sovi_cd_path=args.sovi_cd_path,
        sovi_cd_id_col=args.sovi_cd_id_col,
        sovi_cd_name_col=args.sovi_cd_name_col,
        audit_dir=args.audit_dir,
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

    raise ValueError(f"Unsupported table extension for {path}")


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
    - CDUID values such as 2401
    - numeric values such as 2401.0
    - DGUID values such as 2021A00032401
    - strings with surrounding spaces
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "<missing>", "<empty>"}:
        return None

    # DGUID format for census divisions often contains A0003 + CDUID.
    m = re.search(r"A0003(\d{4})$", text)
    if m:
        return m.group(1)

    # Generic DGUID-like fallback: trailing 4 digits after non-digit prefix.
    if "A0003" in text:
        digits = re.findall(r"\d+", text)
        if digits:
            tail = digits[-1]
            if len(tail) >= 4:
                return tail[-4:]

    # Plain numeric, including values read as 2401.0.
    if re.fullmatch(r"\d+(\.0+)?", text):
        number = int(float(text))
        return f"{number:04d}"

    # Plain CDUID-ish string.
    digits_only = re.sub(r"\D", "", text)
    if len(digits_only) == 4:
        return digits_only
    if len(digits_only) > 4 and digits_only.startswith("24"):
        # Some accidentally concatenated Québec IDs may include prefix/suffix.
        # Keep the first likely Québec CDUID.
        q = re.search(r"(24\d{2})", digits_only)
        if q:
            return q.group(1)

    return text


def detect_sovi_id_column(sovi: pd.DataFrame, event_ids_norm: set[str], explicit_col: str | None) -> tuple[str | None, pd.DataFrame]:
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
    for col in sovi.columns:
        col_s = str(col)
        low = col_s.lower()
        if col_s in SOVI_ID_CANDIDATES or normalize_col_name(col_s) in {normalize_col_name(c) for c in SOVI_ID_CANDIDATES}:
            candidate_cols.append(col_s)
        elif any(token in low for token in ["cduid", "dguid", "cd_id", "cd_uid", "geo_code", "geocode"]):
            candidate_cols.append(col_s)

    # Keep unique order.
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

    if best_col is None:
        return None, pd.DataFrame(diagnostics)

    return best_col, pd.DataFrame(diagnostics).sort_values(
        ["overlap_with_event_cd_ids", "non_missing_normalized", "unique_normalized"],
        ascending=False,
    )


def detect_name_column(df: pd.DataFrame, explicit_col: str | None) -> str | None:
    if explicit_col is not None:
        if explicit_col not in df.columns:
            raise ValueError(f"Requested name column not found: {explicit_col}")
        return explicit_col
    return choose_first_existing(df.columns, SOVI_NAME_CANDIDATES)


def candidate_sovi_files() -> list[Path]:
    candidates = []
    for root in SOVI_SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in TABLE_EXTENSIONS:
                continue

            low = str(path).lower()
            name_low = path.name.lower()

            if "civil_security" in low or "quebec_civil_security_events" in low:
                continue
            if "sovi" not in low:
                continue
            if any(skip in low for skip in ["/audits/", "/plots/", "/figures/", "/maps/"]):
                continue
            if name_low.startswith("~$"):
                continue

            candidates.append(path)

    def score(path: Path) -> tuple[int, int, str]:
        low = str(path).lower()
        s = 0
        if "sovi_like_quebec_cd_2021_38var_oriented_run" in low:
            s -= 100
        if "quebec" in low:
            s -= 20
        if "/cd" in low or "_cd" in low or "census_division" in low:
            s -= 20
        if any(tok in low for tok in ["score", "rank", "result", "final", "oriented", "normalized", "output"]):
            s -= 10
        if "raw" in low:
            s += 20
        if path.suffix.lower() == ".parquet":
            s -= 5
        elif path.suffix.lower() == ".csv":
            s -= 3
        return (s, len(str(path)), str(path))

    return sorted(set(candidates), key=score)


def autodetect_sovi_file(event_ids_norm: set[str]) -> tuple[Path | None, pd.DataFrame]:
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
                    "has_sovi_like_column": False,
                }
            )
            continue

        best_col, id_diag = detect_sovi_id_column(df, event_ids_norm, None)
        overlap = 0
        unique_norm = 0
        if best_col is not None:
            norm = df[best_col].map(normalize_cd_id)
            overlap = len(set(norm.dropna()) & event_ids_norm)
            unique_norm = int(norm.dropna().nunique())

        has_sovi_like_column = any("sovi" in str(c).lower() for c in df.columns)

        rows.append(
            {
                "path": str(path),
                "read_ok": True,
                "error": "",
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "best_id_col": best_col,
                "overlap_with_event_cd_ids": int(overlap),
                "unique_normalized_ids": int(unique_norm),
                "has_sovi_like_column": bool(has_sovi_like_column),
            }
        )

        # Prefer high overlap, SoVI-like columns, CD-sized tables, and compact outputs.
        size_score = -abs(len(df) - 98)
        score = (
            int(overlap),
            int(has_sovi_like_column),
            int(unique_norm),
            int(size_score),
        )
        if score > best_score and overlap > 0:
            best_score = score
            best_path = path

    return best_path, pd.DataFrame(rows).sort_values(
        ["overlap_with_event_cd_ids", "has_sovi_like_column", "unique_normalized_ids"],
        ascending=False,
    )


def write_json(data: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def ensure_event_cd_columns(events: pd.DataFrame) -> None:
    missing = [c for c in [EVENT_CD_ID_COL, EVENT_CD_JOIN_COL] if c not in events.columns]
    if missing:
        raise ValueError(f"Event-with-geographies file is missing required CD columns: {missing}")


def joined_events(events: pd.DataFrame) -> pd.DataFrame:
    ensure_event_cd_columns(events)
    return events[events[EVENT_CD_JOIN_COL].fillna(False)].copy()


def unmatched_events(events: pd.DataFrame) -> pd.DataFrame:
    ensure_event_cd_columns(events)
    cols = [
        c for c in [
            "event_id",
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
            "coord_source",
            "date_signalement",
            "date_debut",
            "date_fin",
            EVENT_CD_ID_COL,
            EVENT_CD_NAME_COL,
            EVENT_CD_JOIN_COL,
        ]
        if c in events.columns
    ]
    out = events[~events[EVENT_CD_JOIN_COL].fillna(False)][cols].copy()
    return out.sort_values([c for c in ["event_date_primary", "municipalite", "alea"] if c in out.columns])


def event_cd_set(events: pd.DataFrame) -> set[str]:
    je = joined_events(events)
    return set(je[EVENT_CD_ID_COL].map(normalize_cd_id).dropna())


def density_by_year(events: pd.DataFrame) -> pd.DataFrame:
    je = joined_events(events)
    year = pd.to_datetime(je["event_date_primary"], errors="coerce").dt.year
    je = je.assign(event_year_clean=year.astype("Int64"))

    rows = []
    for y, sub in je.groupby("event_year_clean", dropna=True):
        rows.append(
            {
                "event_year": int(y),
                "event_count_total": int(len(sub)),
                "n_cd_with_events": int(sub[EVENT_CD_ID_COL].map(normalize_cd_id).nunique()),
                "event_count_precise_or_very_precise": int(sub.get("is_precise_or_very_precise", False).fillna(False).sum()),
                "event_count_very_precise": int(sub.get("is_very_precise", False).fillna(False).sum()),
                "event_count_moderate_or_worse": int(sub.get("is_moderate_or_worse", False).fillna(False).sum()),
                "event_count_important_or_extreme": int(sub.get("is_important_or_extreme", False).fillna(False).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("event_year").reset_index(drop=True)


def add_hazard_counts(row: dict[str, Any], sub: pd.DataFrame) -> None:
    if "alea_group" not in sub.columns:
        return
    for group, n in sub["alea_group"].astype("string").fillna("<MISSING>").value_counts().items():
        row[f"event_count_{group}"] = int(n)


def density_by_cd(events: pd.DataFrame, sovi_ids: set[str] | None = None) -> pd.DataFrame:
    je = joined_events(events).copy()
    je["cd_id_norm"] = je[EVENT_CD_ID_COL].map(normalize_cd_id)

    rows = []
    for cd_id, sub in je.groupby("cd_id_norm", dropna=False):
        row = {
            "cd_id_norm": cd_id,
            "cd_id_source": sub[EVENT_CD_ID_COL].astype("string").dropna().iloc[0] if sub[EVENT_CD_ID_COL].notna().any() else pd.NA,
            "cd_name": sub[EVENT_CD_NAME_COL].astype("string").dropna().iloc[0] if EVENT_CD_NAME_COL in sub.columns and sub[EVENT_CD_NAME_COL].notna().any() else pd.NA,
            "event_count_total": int(len(sub)),
            "event_count_precise_or_very_precise": int(sub.get("is_precise_or_very_precise", False).fillna(False).sum()),
            "event_count_very_precise": int(sub.get("is_very_precise", False).fillna(False).sum()),
            "event_count_moderate_or_worse": int(sub.get("is_moderate_or_worse", False).fillna(False).sum()),
            "event_count_important_or_extreme": int(sub.get("is_important_or_extreme", False).fillna(False).sum()),
            "event_count_possible_duplicate": int(sub.get("possible_duplicate_flag", False).fillna(False).sum()),
            "first_event_date": pd.to_datetime(sub["event_date_primary"], errors="coerce").min(),
            "last_event_date": pd.to_datetime(sub["event_date_primary"], errors="coerce").max(),
            "n_event_years_with_events": int(pd.to_datetime(sub["event_date_primary"], errors="coerce").dt.year.nunique()),
            "n_event_months_with_events": int(sub["event_period_month"].astype("string").nunique()) if "event_period_month" in sub.columns else 0,
        }
        if sovi_ids is not None:
            row["in_sovi_frame"] = cd_id in sovi_ids
        add_hazard_counts(row, sub)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("event_count_total", ascending=False).reset_index(drop=True)
    return out


def complete_geo_year_grid(
    events: pd.DataFrame,
    cd_ids: list[str],
    cd_names: Mapping[str, Any],
    years: list[int],
) -> pd.DataFrame:
    je = joined_events(events).copy()
    je["cd_id_norm"] = je[EVENT_CD_ID_COL].map(normalize_cd_id)
    je["event_year_clean"] = pd.to_datetime(je["event_date_primary"], errors="coerce").dt.year.astype("Int64")

    grouped = (
        je.dropna(subset=["cd_id_norm", "event_year_clean"])
        .groupby(["cd_id_norm", "event_year_clean"])
        .agg(
            event_count_total=("event_id", "size"),
            event_count_precise_or_very_precise=("is_precise_or_very_precise", lambda s: int(s.fillna(False).sum())),
            event_count_very_precise=("is_very_precise", lambda s: int(s.fillna(False).sum())),
            event_count_moderate_or_worse=("is_moderate_or_worse", lambda s: int(s.fillna(False).sum())),
            event_count_important_or_extreme=("is_important_or_extreme", lambda s: int(s.fillna(False).sum())),
        )
        .reset_index()
        .rename(columns={"event_year_clean": "event_year"})
    )

    grid = pd.MultiIndex.from_product([cd_ids, years], names=["cd_id_norm", "event_year"]).to_frame(index=False)
    grid["cd_name"] = grid["cd_id_norm"].map(cd_names)

    out = grid.merge(grouped, on=["cd_id_norm", "event_year"], how="left")
    count_cols = [c for c in out.columns if c.startswith("event_count_")]
    out[count_cols] = out[count_cols].fillna(0).astype(int)
    out["has_any_event"] = out["event_count_total"] > 0
    return out.sort_values(["cd_id_norm", "event_year"]).reset_index(drop=True)


def complete_geo_month_grid(
    events: pd.DataFrame,
    cd_ids: list[str],
    cd_names: Mapping[str, Any],
    months: list[str],
) -> pd.DataFrame:
    je = joined_events(events).copy()
    je["cd_id_norm"] = je[EVENT_CD_ID_COL].map(normalize_cd_id)
    je["event_period_month"] = je["event_period_month"].astype("string")

    grouped = (
        je.dropna(subset=["cd_id_norm", "event_period_month"])
        .groupby(["cd_id_norm", "event_period_month"])
        .agg(
            event_count_total=("event_id", "size"),
            event_count_precise_or_very_precise=("is_precise_or_very_precise", lambda s: int(s.fillna(False).sum())),
            event_count_very_precise=("is_very_precise", lambda s: int(s.fillna(False).sum())),
            event_count_moderate_or_worse=("is_moderate_or_worse", lambda s: int(s.fillna(False).sum())),
            event_count_important_or_extreme=("is_important_or_extreme", lambda s: int(s.fillna(False).sum())),
        )
        .reset_index()
    )

    grid = pd.MultiIndex.from_product([cd_ids, months], names=["cd_id_norm", "event_period_month"]).to_frame(index=False)
    grid["cd_name"] = grid["cd_id_norm"].map(cd_names)

    out = grid.merge(grouped, on=["cd_id_norm", "event_period_month"], how="left")
    count_cols = [c for c in out.columns if c.startswith("event_count_")]
    out[count_cols] = out[count_cols].fillna(0).astype(int)
    out["has_any_event"] = out["event_count_total"] > 0
    return out.sort_values(["cd_id_norm", "event_period_month"]).reset_index(drop=True)


def density_by_hazard_group(events: pd.DataFrame) -> pd.DataFrame:
    je = joined_events(events).copy()
    je["cd_id_norm"] = je[EVENT_CD_ID_COL].map(normalize_cd_id)
    je["event_year_clean"] = pd.to_datetime(je["event_date_primary"], errors="coerce").dt.year.astype("Int64")
    out = (
        je.groupby("alea_group", dropna=False)
        .agg(
            event_count_total=("event_id", "size"),
            n_cd_with_events=("cd_id_norm", "nunique"),
            n_years_with_events=("event_year_clean", "nunique"),
            event_count_precise_or_very_precise=("is_precise_or_very_precise", lambda s: int(s.fillna(False).sum())),
            event_count_very_precise=("is_very_precise", lambda s: int(s.fillna(False).sum())),
            event_count_moderate_or_worse=("is_moderate_or_worse", lambda s: int(s.fillna(False).sum())),
            event_count_important_or_extreme=("is_important_or_extreme", lambda s: int(s.fillna(False).sum())),
        )
        .reset_index()
    )
    out["share"] = out["event_count_total"] / out["event_count_total"].sum()
    return out.sort_values("event_count_total", ascending=False).reset_index(drop=True)


def density_by_severity(events: pd.DataFrame) -> pd.DataFrame:
    je = joined_events(events).copy()
    je["cd_id_norm"] = je[EVENT_CD_ID_COL].map(normalize_cd_id)
    je["event_year_clean"] = pd.to_datetime(je["event_date_primary"], errors="coerce").dt.year.astype("Int64")
    out = (
        je.groupby("severite", dropna=False)
        .agg(
            event_count_total=("event_id", "size"),
            n_cd_with_events=("cd_id_norm", "nunique"),
            n_years_with_events=("event_year_clean", "nunique"),
            event_count_precise_or_very_precise=("is_precise_or_very_precise", lambda s: int(s.fillna(False).sum())),
            event_count_very_precise=("is_very_precise", lambda s: int(s.fillna(False).sum())),
        )
        .reset_index()
    )
    out["share"] = out["event_count_total"] / out["event_count_total"].sum()
    return out.sort_values("event_count_total", ascending=False).reset_index(drop=True)


def set_difference_report(
    left_ids: set[str],
    right_ids: set[str],
    *,
    left_name: str,
    right_name: str,
    events_density: pd.DataFrame | None = None,
    sovi: pd.DataFrame | None = None,
    sovi_id_col_norm: str | None = None,
    sovi_name_col: str | None = None,
) -> pd.DataFrame:
    diff = sorted(left_ids - right_ids)
    rows = []
    for cd_id in diff:
        row = {"cd_id_norm": cd_id, "present_in": left_name, "missing_from": right_name}
        if events_density is not None and "cd_id_norm" in events_density.columns:
            match = events_density[events_density["cd_id_norm"].eq(cd_id)]
            if not match.empty:
                for col in ["cd_name", "event_count_total", "first_event_date", "last_event_date"]:
                    if col in match.columns:
                        row[col] = match.iloc[0][col]
        if sovi is not None and sovi_id_col_norm is not None:
            match = sovi[sovi[sovi_id_col_norm].eq(cd_id)]
            if not match.empty:
                if sovi_name_col and sovi_name_col in match.columns:
                    row["sovi_cd_name"] = match.iloc[0][sovi_name_col]
                row["sovi_row_index"] = int(match.index[0])
        rows.append(row)
    return pd.DataFrame(rows)


def infer_cd_name_map(events_density: pd.DataFrame, sovi: pd.DataFrame | None, sovi_id_norm_col: str | None, sovi_name_col: str | None) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    if not events_density.empty and "cd_id_norm" in events_density.columns and "cd_name" in events_density.columns:
        for _, row in events_density.iterrows():
            if pd.notna(row["cd_id_norm"]) and pd.notna(row["cd_name"]):
                mapping[str(row["cd_id_norm"])] = row["cd_name"]

    if sovi is not None and sovi_id_norm_col and sovi_name_col and sovi_name_col in sovi.columns:
        for _, row in sovi.iterrows():
            if pd.notna(row[sovi_id_norm_col]) and pd.notna(row[sovi_name_col]):
                mapping.setdefault(str(row[sovi_id_norm_col]), row[sovi_name_col])

    return mapping


def window_stats(cd_year: pd.DataFrame, years: list[int], cd_ids: list[str]) -> dict[str, Any]:
    sub = cd_year[cd_year["event_year"].isin(years)].copy()
    if sub.empty:
        return {
            "years": years,
            "possible_cd_year_cells": len(cd_ids) * len(years),
            "nonzero_cd_year_cells": 0,
            "nonzero_cd_year_rate": 0.0,
            "total_events": 0,
            "cds_with_any_event": 0,
            "cd_coverage_rate": 0.0,
            "mean_events_per_cd": 0.0,
            "median_events_per_cd": 0.0,
        }

    by_cd = sub.groupby("cd_id_norm", dropna=False)["event_count_total"].sum()
    total_events = int(sub["event_count_total"].sum())
    cds_with_any = int((by_cd > 0).sum())
    possible_cells = len(cd_ids) * len(years)
    nonzero_cells = int((sub["event_count_total"] > 0).sum())

    return {
        "years": years,
        "possible_cd_year_cells": int(possible_cells),
        "nonzero_cd_year_cells": int(nonzero_cells),
        "nonzero_cd_year_rate": float(nonzero_cells / possible_cells) if possible_cells else math.nan,
        "total_events": total_events,
        "cds_with_any_event": cds_with_any,
        "cd_coverage_rate": float(cds_with_any / len(cd_ids)) if cd_ids else math.nan,
        "mean_events_per_cd": float(by_cd.mean()) if len(by_cd) else 0.0,
        "median_events_per_cd": float(by_cd.median()) if len(by_cd) else 0.0,
        "zero_event_cds": int((by_cd == 0).sum()),
    }


def monthly_feasibility(cd_month: pd.DataFrame, cd_ids: list[str], months_2021_2025: list[str]) -> dict[str, Any]:
    sub = cd_month[cd_month["event_period_month"].isin(months_2021_2025)].copy()
    possible = len(cd_ids) * len(months_2021_2025)
    nonzero = int((sub["event_count_total"] > 0).sum()) if not sub.empty else 0

    nonzero_counts = sub.loc[sub["event_count_total"] > 0, "event_count_total"] if not sub.empty else pd.Series(dtype=float)

    return {
        "months": months_2021_2025,
        "possible_cd_month_cells": int(possible),
        "nonzero_cd_month_cells": int(nonzero),
        "nonzero_cd_month_rate": float(nonzero / possible) if possible else math.nan,
        "total_events": int(sub["event_count_total"].sum()) if not sub.empty else 0,
        "median_events_per_nonzero_cd_month": float(nonzero_counts.median()) if len(nonzero_counts) else 0.0,
        "mean_events_per_cd_month": float(sub["event_count_total"].mean()) if not sub.empty else 0.0,
    }


def choose_recommendation(
    stats_2021_2025: Mapping[str, Any],
    stats_2022_2025: Mapping[str, Any],
    monthly_stats: Mapping[str, Any],
    yearly_nonzero_rate: float,
) -> tuple[str, str, str]:
    # CD-level aggregation is robust enough to use all precision classes. The
    # precision sensitivity files can still be used later, but the first SoVI
    # validation should maximize coverage.
    precision = "all"

    monthly_rate = float(monthly_stats.get("nonzero_cd_month_rate", 0.0) or 0.0)
    monthly_median_nonzero = float(monthly_stats.get("median_events_per_nonzero_cd_month", 0.0) or 0.0)
    coverage_2021_2025 = float(stats_2021_2025.get("cd_coverage_rate", 0.0) or 0.0)
    coverage_2022_2025 = float(stats_2022_2025.get("cd_coverage_rate", 0.0) or 0.0)

    if monthly_rate >= 0.35 and monthly_median_nonzero >= 2:
        return (
            "cd_month",
            precision,
            "Monthly CD density appears sufficiently populated for a first validation target.",
        )

    if yearly_nonzero_rate >= 0.65:
        return (
            "cd_year",
            precision,
            "CD-year density appears reasonably populated and preserves temporal variation.",
        )

    # Prefer 2021-2025 if it has good coverage and aligns with 2021 census/SoVI.
    if coverage_2021_2025 >= 0.80:
        return (
            "cd_cumulative_2021_2025",
            precision,
            "Monthly/CD-year cells are sparse; cumulative 2021-2025 burden aligns with the 2021 SoVI frame and keeps broad CD coverage.",
        )

    if coverage_2022_2025 >= 0.80:
        return (
            "cd_cumulative_2022_2025",
            precision,
            "Monthly/CD-year cells are sparse; 2022-2025 cumulative burden gives stronger complete-year coverage than 2021-2025.",
        )

    return (
        "cd_cumulative_2021_2025",
        precision,
        "Defaulting to cumulative 2021-2025 because monthly and yearly targets are likely sparse for a first SoVI validation.",
    )


def main() -> None:
    config = parse_args()
    config.audit_dir.mkdir(parents=True, exist_ok=True)

    events = read_table(config.events_with_geographies_path)
    events_by_cd = read_table(config.events_by_cd_path) if config.events_by_cd_path.exists() else pd.DataFrame()
    events_by_cd_month = read_table(config.events_by_cd_month_path) if config.events_by_cd_month_path.exists() else pd.DataFrame()

    ensure_event_cd_columns(events)

    event_ids = event_cd_set(events)
    event_density_cd = density_by_cd(events, None)

    sovi_path = config.sovi_cd_path
    candidate_audit = pd.DataFrame()

    if sovi_path is None:
        sovi_path, candidate_audit = autodetect_sovi_file(event_ids)
        candidate_audit.to_csv(config.audit_dir / "cd_sovi_alignment_sovi_file_candidates.csv", index=False)

    if sovi_path is None:
        if config.fail_if_sovi_missing:
            raise FileNotFoundError(
                "Could not auto-detect SoVI CD output. Pass --sovi-cd-path explicitly."
            )
        sovi = None
        sovi_id_col = None
        sovi_name_col = None
        sovi_id_diag = pd.DataFrame()
        sovi_ids = set()
    else:
        sovi = read_table(sovi_path)
        sovi_id_col, sovi_id_diag = detect_sovi_id_column(sovi, event_ids, config.sovi_cd_id_col)
        if sovi_id_col is None:
            raise ValueError(
                f"Could not detect a CD ID column in SoVI file: {sovi_path}. "
                "Pass --sovi-cd-id-col explicitly."
            )
        sovi_name_col = detect_name_column(sovi, config.sovi_cd_name_col)

        sovi["_cd_id_norm_for_alignment"] = sovi[sovi_id_col].map(normalize_cd_id)
        sovi_ids = set(sovi["_cd_id_norm_for_alignment"].dropna())

        sovi_id_diag.to_csv(config.audit_dir / "cd_sovi_alignment_sovi_id_column_diagnostics.csv", index=False)

    event_density_cd = density_by_cd(events, sovi_ids if sovi_ids else None)

    event_not_in_sovi = set_difference_report(
        event_ids,
        sovi_ids,
        left_name="events",
        right_name="sovi",
        events_density=event_density_cd,
    )
    sovi_not_in_events = set_difference_report(
        sovi_ids,
        event_ids,
        left_name="sovi",
        right_name="events",
        sovi=sovi,
        sovi_id_col_norm="_cd_id_norm_for_alignment" if sovi is not None else None,
        sovi_name_col=sovi_name_col,
    )

    event_not_in_sovi.to_csv(config.audit_dir / "cd_sovi_alignment_event_ids_not_in_sovi.csv", index=False)
    sovi_not_in_events.to_csv(config.audit_dir / "cd_sovi_alignment_sovi_ids_not_in_events.csv", index=False)

    unmatched = unmatched_events(events)
    unmatched.to_csv(config.audit_dir / "unmatched_cd_events.csv", index=False)

    year_density = density_by_year(events)
    hazard_density = density_by_hazard_group(events)
    severity_density = density_by_severity(events)

    year_density.to_csv(config.audit_dir / "event_density_by_year.csv", index=False)
    event_density_cd.to_csv(config.audit_dir / "event_density_by_cd.csv", index=False)
    hazard_density.to_csv(config.audit_dir / "event_density_by_hazard_group.csv", index=False)
    severity_density.to_csv(config.audit_dir / "event_density_by_severity.csv", index=False)

    # Complete grids use SoVI CDs when available; otherwise event CDs.
    cd_ids_for_grid = sorted(sovi_ids if sovi_ids else event_ids)
    cd_name_map = infer_cd_name_map(
        event_density_cd,
        sovi,
        "_cd_id_norm_for_alignment" if sovi is not None else None,
        sovi_name_col,
    )

    all_event_years = sorted(
        pd.to_datetime(events["event_date_primary"], errors="coerce").dt.year.dropna().astype(int).unique().tolist()
    )
    cd_year = complete_geo_year_grid(events, cd_ids_for_grid, cd_name_map, all_event_years)
    cd_year.to_csv(config.audit_dir / "event_density_by_cd_year.csv", index=False)

    all_months = sorted(events["event_period_month"].dropna().astype(str).unique().tolist())
    cd_month = complete_geo_month_grid(events, cd_ids_for_grid, cd_name_map, all_months)
    cd_month.to_csv(config.audit_dir / "event_density_by_cd_month.csv", index=False)

    # Feasibility windows.
    months_2021_2025 = [
        f"{year}-{month:02d}"
        for year in TARGET_FULL_YEARS_2021_2025
        for month in range(1, 13)
    ]

    stats_2021_2025 = window_stats(cd_year, TARGET_FULL_YEARS_2021_2025, cd_ids_for_grid)
    stats_2022_2025 = window_stats(cd_year, TARGET_FULL_YEARS_2022_2025, cd_ids_for_grid)
    month_stats = monthly_feasibility(cd_month, cd_ids_for_grid, months_2021_2025)

    years_2021_2025 = cd_year[cd_year["event_year"].isin(TARGET_FULL_YEARS_2021_2025)]
    possible_year_cells = len(cd_ids_for_grid) * len(TARGET_FULL_YEARS_2021_2025)
    yearly_nonzero_rate = (
        float((years_2021_2025["event_count_total"] > 0).sum() / possible_year_cells)
        if possible_year_cells
        else math.nan
    )

    recommendation, precision, reason = choose_recommendation(
        stats_2021_2025,
        stats_2022_2025,
        month_stats,
        yearly_nonzero_rate,
    )

    # Check precomputed aggregate alignment, if available.
    precomputed_aggregate_checks: dict[str, Any] = {}
    if not events_by_cd.empty:
        agg_cd_col = choose_first_existing(events_by_cd.columns, ["cd_id", "CDUID", "DGUID", "geo_id"])
        if agg_cd_col is not None:
            agg_ids = set(events_by_cd[agg_cd_col].map(normalize_cd_id).dropna())
            precomputed_aggregate_checks["events_by_cd_rows"] = int(len(events_by_cd))
            precomputed_aggregate_checks["events_by_cd_id_col"] = agg_cd_col
            precomputed_aggregate_checks["events_by_cd_unique_ids"] = int(len(agg_ids))
            precomputed_aggregate_checks["events_by_cd_ids_match_event_level_ids"] = bool(agg_ids == event_ids)
            precomputed_aggregate_checks["events_by_cd_ids_missing_from_event_level"] = sorted(agg_ids - event_ids)
            precomputed_aggregate_checks["event_level_ids_missing_from_events_by_cd"] = sorted(event_ids - agg_ids)

    if not events_by_cd_month.empty:
        agg_cd_col = choose_first_existing(events_by_cd_month.columns, ["cd_id", "CDUID", "DGUID", "geo_id"])
        month_col = choose_first_existing(events_by_cd_month.columns, ["event_period_month", "period_month", "month"])
        precomputed_aggregate_checks["events_by_cd_month_rows"] = int(len(events_by_cd_month))
        precomputed_aggregate_checks["events_by_cd_month_id_col"] = agg_cd_col
        precomputed_aggregate_checks["events_by_cd_month_month_col"] = month_col

    unmatched_summary = {
        "unmatched_rows": int(len(unmatched)),
        "unmatched_inside_loose_quebec_bbox": int(unmatched.get("inside_loose_quebec_bbox", pd.Series(dtype=bool)).fillna(False).sum()) if not unmatched.empty else 0,
        "unmatched_with_valid_lon_lat": int((pd.to_numeric(unmatched.get("lon", pd.Series(dtype=float)), errors="coerce").notna() & pd.to_numeric(unmatched.get("lat", pd.Series(dtype=float)), errors="coerce").notna()).sum()) if not unmatched.empty else 0,
        "unmatched_by_precision": unmatched["precision_localisation"].astype("string").fillna("<MISSING>").value_counts().to_dict() if "precision_localisation" in unmatched.columns else {},
        "unmatched_by_hazard_group": unmatched["alea_group"].astype("string").fillna("<MISSING>").value_counts().to_dict() if "alea_group" in unmatched.columns else {},
    }

    summary = {
        "status": "completed",
        "events_with_geographies_path": str(config.events_with_geographies_path),
        "events_by_cd_path": str(config.events_by_cd_path),
        "events_by_cd_month_path": str(config.events_by_cd_month_path),
        "sovi_cd_path": str(sovi_path) if sovi_path is not None else None,
        "sovi_cd_id_col": sovi_id_col,
        "sovi_cd_name_col": sovi_name_col,
        "audit_dir": str(config.audit_dir),

        "event_rows": int(len(events)),
        "joined_event_rows": int(joined_events(events).shape[0]),
        "unmatched_event_rows": int(len(unmatched)),
        "event_cd_unique_ids": int(len(event_ids)),

        "sovi_rows": int(len(sovi)) if sovi is not None else None,
        "sovi_cd_unique_ids": int(len(sovi_ids)) if sovi_ids else 0,

        "overlap_cd_ids": int(len(event_ids & sovi_ids)) if sovi_ids else 0,
        "event_cd_ids_not_in_sovi": int(len(event_ids - sovi_ids)) if sovi_ids else None,
        "sovi_cd_ids_not_in_events": int(len(sovi_ids - event_ids)) if sovi_ids else None,

        "event_cd_ids_not_in_sovi_values": sorted(event_ids - sovi_ids) if sovi_ids else [],
        "sovi_cd_ids_not_in_events_values": sorted(sovi_ids - event_ids) if sovi_ids else [],

        "unmatched_summary": unmatched_summary,
        "precomputed_aggregate_checks": precomputed_aggregate_checks,

        "feasibility": {
            "cd_cumulative_2021_2025": stats_2021_2025,
            "cd_cumulative_2022_2025": stats_2022_2025,
            "cd_year_2021_2025": {
                "possible_cd_year_cells": int(possible_year_cells),
                "nonzero_cd_year_cells": int((years_2021_2025["event_count_total"] > 0).sum()),
                "nonzero_cd_year_rate": yearly_nonzero_rate,
            },
            "cd_month_2021_2025": month_stats,
        },

        "recommended_primary_target_granularity": recommendation,
        "recommended_primary_precision_filter": precision,
        "recommended_reason": reason,
    }

    write_json(summary, config.audit_dir / "cd_sovi_alignment_summary.json")

    print("CD event/SoVI alignment audit completed.")
    print(f"Audit directory: {config.audit_dir}")
    print()
    print(f"Events with geographies: {config.events_with_geographies_path}")
    print(f"SoVI CD path: {sovi_path}")
    print(f"SoVI CD ID column: {sovi_id_col}")
    print(f"SoVI CD name column: {sovi_name_col}")
    print()
    print("Alignment:")
    print(f"  Event CD IDs: {len(event_ids):,}")
    print(f"  SoVI CD IDs: {len(sovi_ids):,}" if sovi_ids else "  SoVI CD IDs: unavailable")
    print(f"  Overlap: {len(event_ids & sovi_ids):,}" if sovi_ids else "  Overlap: unavailable")
    print(f"  Event IDs not in SoVI: {len(event_ids - sovi_ids):,}" if sovi_ids else "  Event IDs not in SoVI: unavailable")
    print(f"  SoVI IDs not in events: {len(sovi_ids - event_ids):,}" if sovi_ids else "  SoVI IDs not in events: unavailable")
    print(f"  Unmatched event rows: {len(unmatched):,}")
    print()
    print("Recommendation:")
    print(f"  Granularity: {recommendation}")
    print(f"  Precision filter: {precision}")
    print(f"  Reason: {reason}")


if __name__ == "__main__":
    main()
