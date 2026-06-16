#!/usr/bin/env python3
"""
Spatially join cleaned Québec civil-security events to census geographies.

Default input:
    data/external/quebec_civil_security_events/processed/quebec_civil_security_events_clean.parquet

Default output folder:
    data/external/quebec_civil_security_events/processed/

Default audit folder:
    data/external/quebec_civil_security_events/audits/

This script consumes the event-level clean file produced by:
    01_clean_quebec_civil_security_events.py

It performs point-in-polygon joins to CD and/or CT boundaries and writes:
    - event-level file with CD/CT join columns
    - CD event-count aggregates
    - CT event-count aggregates
    - CD-month event-count aggregates
    - CT-month event-count aggregates
    - spatial-join audit files

It does not alter the raw or cleaned event data.

Typical run from repository root:

    python data/external/quebec_civil_security_events/02_spatial_join_quebec_civil_security_events.py \
      --cd-boundaries path/to/cd_boundaries.geojson \
      --ct-boundaries path/to/ct_boundaries.geojson

If boundary paths are not passed, the script tries to auto-detect plausible CD/CT
boundary files under common project folders. Auto-detection is conservative and
may ask you to pass explicit paths if several candidates exist.
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

try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script requires geopandas and shapely. Install them in your venv "
        "before running the spatial join."
    ) from exc


DEFAULT_CLEAN_EVENTS_PATH = Path(
    "data/external/quebec_civil_security_events/processed/"
    "quebec_civil_security_events_clean.parquet"
)
DEFAULT_PROCESSED_DIR = Path("data/external/quebec_civil_security_events/processed")
DEFAULT_AUDIT_DIR = Path("data/external/quebec_civil_security_events/audits")

OUTPUT_EVENTS_JOINED = "quebec_civil_security_events_with_geographies.parquet"
OUTPUT_EVENTS_JOINED_GEOJSON = "quebec_civil_security_events_with_geographies.geojson"

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

PRECISION_FILTERS = {
    "all": None,
    "precise_or_very_precise": "is_precise_or_very_precise",
    "very_precise": "is_very_precise",
}

BOUNDARY_SEARCH_ROOTS = [
    Path("data"),
    Path("urban_graph_benchmark/data"),
    Path("urban_graph_benchmark/outputs"),
]

BOUNDARY_FILE_EXTENSIONS = {
    ".parquet",
    ".geojson",
    ".json",
    ".gpkg",
    ".shp",
}


@dataclass(frozen=True)
class Config:
    clean_events_path: Path
    processed_dir: Path
    audit_dir: Path
    cd_boundaries: Path | None
    ct_boundaries: Path | None
    cd_id_col: str | None
    ct_id_col: str | None
    cd_name_col: str | None
    ct_name_col: str | None
    write_geojson: bool
    strict_boundaries: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Spatially join Québec civil-security event points to CD/CT boundaries."
    )
    parser.add_argument("--clean-events", type=Path, default=DEFAULT_CLEAN_EVENTS_PATH)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)

    parser.add_argument("--cd-boundaries", type=Path, default=None)
    parser.add_argument("--ct-boundaries", type=Path, default=None)

    parser.add_argument("--cd-id-col", default=None)
    parser.add_argument("--ct-id-col", default=None)
    parser.add_argument("--cd-name-col", default=None)
    parser.add_argument("--ct-name-col", default=None)

    parser.add_argument("--no-geojson", action="store_true")
    parser.add_argument(
        "--strict-boundaries",
        action="store_true",
        help="Fail if either CD or CT boundaries cannot be loaded.",
    )

    args = parser.parse_args()
    return Config(
        clean_events_path=args.clean_events,
        processed_dir=args.processed_dir,
        audit_dir=args.audit_dir,
        cd_boundaries=args.cd_boundaries,
        ct_boundaries=args.ct_boundaries,
        cd_id_col=args.cd_id_col,
        ct_id_col=args.ct_id_col,
        cd_name_col=args.cd_name_col,
        ct_name_col=args.ct_name_col,
        write_geojson=not args.no_geojson,
        strict_boundaries=bool(args.strict_boundaries),
    )


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input table does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".geojson", ".json", ".gpkg", ".shp"}:
        return pd.DataFrame(gpd.read_file(path))

    raise ValueError(f"Unsupported table format: {path}")


def read_boundaries(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Boundary file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        gdf = gpd.read_parquet(path)
    elif suffix in {".geojson", ".json", ".gpkg", ".shp"}:
        gdf = gpd.read_file(path)
    else:
        raise ValueError(f"Unsupported boundary file format: {path}")

    if not isinstance(gdf, gpd.GeoDataFrame):
        if "geometry" not in gdf.columns:
            raise ValueError(f"Boundary file has no geometry column: {path}")
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

    if gdf.geometry.isna().all():
        raise ValueError(f"Boundary file geometry is entirely missing: {path}")

    if gdf.crs is None:
        # Statistics Canada files are often EPSG:3347, but we do not assume that.
        # For safety, fail instead of silently assigning a CRS.
        raise ValueError(
            f"Boundary file has no CRS: {path}. Re-save it with a CRS or pass a file with CRS metadata."
        )

    return gdf


def event_points_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    required = {"lon", "lat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Clean event file is missing coordinate columns: {sorted(missing)}")

    lon = pd.to_numeric(df["lon"], errors="coerce")
    lat = pd.to_numeric(df["lat"], errors="coerce")
    valid = lon.notna() & lat.notna()

    if not valid.all():
        raise ValueError(
            f"Clean event file has {int((~valid).sum())} rows without valid lon/lat. "
            "The current cleaning inspection suggested all rows should be valid."
        )

    geometry = [Point(float(x), float(y)) for x, y in zip(lon, lat)]
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")


def normalize_col_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(col).lower())


def choose_first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = list(columns)
    exact = {str(c): str(c) for c in cols}
    normalized = {normalize_col_name(c): str(c) for c in cols}

    for cand in candidates:
        if cand in exact:
            return exact[cand]
    for cand in candidates:
        key = normalize_col_name(cand)
        if key in normalized:
            return normalized[key]
    return None


def detect_boundary_columns(
    gdf: gpd.GeoDataFrame,
    *,
    level: str,
    id_col: str | None,
    name_col: str | None,
) -> tuple[str, str | None]:
    cols = [c for c in gdf.columns if c != gdf.geometry.name]

    if level == "cd":
        id_candidates = [
            "CDUID",
            "CDUID_2021",
            "cd_uid",
            "cd_id",
            "CD_CODE",
            "CSDUID",
            "DGUID",
            "dguid",
            "geo_id",
            "geography_id",
        ]
        name_candidates = [
            "CDNAME",
            "CDNAME_2021",
            "cd_name",
            "NOM_DR",
            "name",
            "geography_name",
            "GEO_NAME",
        ]
    elif level == "ct":
        id_candidates = [
            "CTUID",
            "CTUID_2021",
            "ct_uid",
            "ct_id",
            "CT_CODE",
            "DGUID",
            "dguid",
            "geo_id",
            "geography_id",
        ]
        name_candidates = [
            "CTNAME",
            "CTNAME_2021",
            "ct_name",
            "name",
            "geography_name",
            "GEO_NAME",
        ]
    else:
        raise ValueError(f"Unknown boundary level: {level}")

    chosen_id = id_col or choose_first_existing(cols, id_candidates)
    chosen_name = name_col or choose_first_existing(cols, name_candidates)

    if chosen_id is None:
        raise ValueError(
            f"Could not infer {level.upper()} ID column. Available columns: {cols}. "
            f"Pass --{level}-id-col explicitly."
        )

    if chosen_id not in gdf.columns:
        raise ValueError(f"Requested {level.upper()} ID column not found: {chosen_id}")

    if chosen_name is not None and chosen_name not in gdf.columns:
        raise ValueError(f"Requested {level.upper()} name column not found: {chosen_name}")

    return chosen_id, chosen_name


def candidate_boundary_files(level: str) -> list[Path]:
    tokens_by_level = {
        "cd": ["cd", "census_division", "census-divisions", "division", "dr_"],
        "ct": ["ct", "census_tract", "census-tract", "tract", "sr_"],
    }
    tokens = tokens_by_level[level]

    candidates: list[Path] = []
    for root in BOUNDARY_SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in BOUNDARY_FILE_EXTENSIONS:
                continue
            low = str(path).lower()
            if any(tok in low for tok in tokens) and any(
                geo_tok in low for geo_tok in ["boundary", "boundaries", "geometry", "spatial", "geo", "carto", "lct_", "lcd_"]
            ):
                candidates.append(path)

    # Prefer non-raw and processed outputs.
    def score(path: Path) -> tuple[int, int, str]:
        low = str(path).lower()
        processed_score = 0
        if "processed" in low or "outputs" in low or "clean" in low:
            processed_score -= 10
        if "raw" in low:
            processed_score += 10
        return (processed_score, len(str(path)), str(path))

    return sorted(set(candidates), key=score)


def autodetect_boundary_path(level: str) -> Path | None:
    candidates = candidate_boundary_files(level)
    if not candidates:
        return None

    # If there is exactly one plausible candidate, use it.
    if len(candidates) == 1:
        return candidates[0]

    # If multiple, try to pick obvious 2021 Québec boundary files.
    preferred = [
        p for p in candidates
        if any(tok in str(p).lower() for tok in ["2021", "21"])
        and any(tok in str(p).lower() for tok in ["quebec", "qc", "mtl_311_water"])
    ]
    if len(preferred) == 1:
        return preferred[0]

    return None


def spatial_join(
    events: gpd.GeoDataFrame,
    boundaries: gpd.GeoDataFrame,
    *,
    level: str,
    id_col: str | None,
    name_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chosen_id, chosen_name = detect_boundary_columns(
        boundaries,
        level=level,
        id_col=id_col,
        name_col=name_col,
    )

    boundaries = boundaries[[c for c in [chosen_id, chosen_name, boundaries.geometry.name] if c is not None]].copy()
    boundaries = boundaries.rename(columns={chosen_id: f"{level}_id"})
    if chosen_name is not None:
        boundaries = boundaries.rename(columns={chosen_name: f"{level}_name"})
    else:
        boundaries[f"{level}_name"] = pd.NA

    if boundaries.crs != events.crs:
        boundaries = boundaries.to_crs(events.crs)

    joined = gpd.sjoin(
        events,
        boundaries[[f"{level}_id", f"{level}_name", boundaries.geometry.name]],
        how="left",
        predicate="within",
    )

    # Very rare: points exactly on polygon boundary may fail within. Try intersects fallback
    # only for unmatched rows.
    unmatched_mask = joined[f"{level}_id"].isna()
    if unmatched_mask.any():
        unmatched = events.loc[unmatched_mask].copy()
        fallback = gpd.sjoin(
            unmatched,
            boundaries[[f"{level}_id", f"{level}_name", boundaries.geometry.name]],
            how="left",
            predicate="intersects",
        )
        for col in [f"{level}_id", f"{level}_name"]:
            joined.loc[unmatched_mask, col] = fallback[col].to_numpy()

    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
    joined[f"{level}_join_success"] = joined[f"{level}_id"].notna()

    audit = pd.DataFrame(
        [
            {
                "level": level,
                "boundary_rows": int(len(boundaries)),
                "boundary_crs": str(boundaries.crs),
                "id_col_used": chosen_id,
                "name_col_used": chosen_name,
                "event_rows": int(len(events)),
                "joined_rows": int(joined[f"{level}_join_success"].sum()),
                "unmatched_rows": int((~joined[f"{level}_join_success"]).sum()),
                "join_success_rate": float(joined[f"{level}_join_success"].mean()) if len(joined) else math.nan,
            }
        ]
    )

    return pd.DataFrame(joined.drop(columns="geometry")), audit


def base_aggregate_columns(df: pd.DataFrame, geo_id_col: str, geo_name_col: str | None) -> list[str]:
    cols = [geo_id_col]
    if geo_name_col and geo_name_col in df.columns:
        cols.append(geo_name_col)
    return cols


def aggregate_events(
    df: pd.DataFrame,
    *,
    level: str,
    monthly: bool,
    precision_filter_name: str,
    precision_filter_col: str | None,
) -> pd.DataFrame:
    geo_id = f"{level}_id"
    geo_name = f"{level}_name"
    join_success = f"{level}_join_success"

    if geo_id not in df.columns:
        return pd.DataFrame()

    tmp = df[df[join_success].fillna(False)].copy()
    if precision_filter_col is not None:
        tmp = tmp[tmp[precision_filter_col].fillna(False)].copy()

    if tmp.empty:
        return pd.DataFrame()

    group_cols = base_aggregate_columns(tmp, geo_id, geo_name)
    if monthly:
        group_cols = [*group_cols, "event_period_month"]

    rows = []
    grouped = tmp.groupby(group_cols, dropna=False)
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        row = {col: val for col, val in zip(group_cols, key)}
        row["precision_filter"] = precision_filter_name
        row["event_count_total"] = int(len(sub))
        row["event_count_precise_or_very_precise"] = int(sub["is_precise_or_very_precise"].fillna(False).sum())
        row["event_count_very_precise"] = int(sub["is_very_precise"].fillna(False).sum())
        row["event_count_moderate_or_worse"] = int(sub["is_moderate_or_worse"].fillna(False).sum())
        row["event_count_important_or_extreme"] = int(sub["is_important_or_extreme"].fillna(False).sum())
        row["event_count_open"] = int(sub["is_open_event"].fillna(False).sum())
        row["event_count_possible_duplicate"] = int(sub["possible_duplicate_flag"].fillna(False).sum())

        for group in HAZARD_GROUPS:
            row[f"event_count_{group}"] = int(sub["alea_group"].eq(group).sum())

        rows.append(row)

    out = pd.DataFrame(rows)
    sort_cols = [c for c in [geo_id, "event_period_month", "precision_filter"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def write_json(data: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def clean_for_geojson(df: pd.DataFrame) -> gpd.GeoDataFrame:
    lon = pd.to_numeric(df["lon"], errors="coerce")
    lat = pd.to_numeric(df["lat"], errors="coerce")
    geometry = [Point(float(x), float(y)) if pd.notna(x) and pd.notna(y) else None for x, y in zip(lon, lat)]
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")


def main() -> None:
    config = parse_args()
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.audit_dir.mkdir(parents=True, exist_ok=True)

    events_df = read_table(config.clean_events_path)
    events = event_points_gdf(events_df)

    cd_path = config.cd_boundaries or autodetect_boundary_path("cd")
    ct_path = config.ct_boundaries or autodetect_boundary_path("ct")

    missing_boundary_notes = []
    if cd_path is None:
        missing_boundary_notes.append(
            "CD boundaries were not provided and could not be auto-detected. "
            "Pass --cd-boundaries explicitly."
        )
    if ct_path is None:
        missing_boundary_notes.append(
            "CT boundaries were not provided and could not be auto-detected. "
            "Pass --ct-boundaries explicitly."
        )

    if config.strict_boundaries and missing_boundary_notes:
        raise FileNotFoundError("\n".join(missing_boundary_notes))

    joined = pd.DataFrame(events.drop(columns="geometry"))
    join_audits = []
    boundary_paths_used: dict[str, str | None] = {"cd": None, "ct": None}

    if cd_path is not None:
        cd_gdf = read_boundaries(cd_path)
        cd_joined, cd_audit = spatial_join(
            events,
            cd_gdf,
            level="cd",
            id_col=config.cd_id_col,
            name_col=config.cd_name_col,
        )
        for col in ["cd_id", "cd_name", "cd_join_success"]:
            joined[col] = cd_joined[col]
        join_audits.append(cd_audit)
        boundary_paths_used["cd"] = str(cd_path)

    if ct_path is not None:
        ct_gdf = read_boundaries(ct_path)
        ct_joined, ct_audit = spatial_join(
            events,
            ct_gdf,
            level="ct",
            id_col=config.ct_id_col,
            name_col=config.ct_name_col,
        )
        for col in ["ct_id", "ct_name", "ct_join_success"]:
            joined[col] = ct_joined[col]
        join_audits.append(ct_audit)
        boundary_paths_used["ct"] = str(ct_path)

    joined_path = config.processed_dir / OUTPUT_EVENTS_JOINED
    joined.to_parquet(joined_path, index=False)

    outputs: dict[str, str] = {"events_with_geographies_parquet": str(joined_path)}

    if config.write_geojson:
        geojson_path = config.processed_dir / OUTPUT_EVENTS_JOINED_GEOJSON
        clean_for_geojson(joined).to_file(geojson_path, driver="GeoJSON")
        outputs["events_with_geographies_geojson"] = str(geojson_path)

    join_audit = pd.concat(join_audits, ignore_index=True) if join_audits else pd.DataFrame()
    join_audit.to_csv(config.audit_dir / "spatial_join_audit.csv", index=False)

    if missing_boundary_notes:
        pd.DataFrame({"issue": missing_boundary_notes}).to_csv(
            config.audit_dir / "spatial_join_missing_boundaries.csv",
            index=False,
        )

    aggregate_outputs: dict[str, str] = {}

    for level in ["cd", "ct"]:
        if f"{level}_id" not in joined.columns:
            continue

        for precision_name, precision_col in PRECISION_FILTERS.items():
            total = aggregate_events(
                joined,
                level=level,
                monthly=False,
                precision_filter_name=precision_name,
                precision_filter_col=precision_col,
            )
            monthly = aggregate_events(
                joined,
                level=level,
                monthly=True,
                precision_filter_name=precision_name,
                precision_filter_col=precision_col,
            )

            total_path = config.processed_dir / f"civil_security_events_by_{level}__{precision_name}.parquet"
            monthly_path = config.processed_dir / f"civil_security_events_by_{level}_month__{precision_name}.parquet"

            total.to_parquet(total_path, index=False)
            monthly.to_parquet(monthly_path, index=False)

            total.to_csv(total_path.with_suffix(".csv"), index=False)
            monthly.to_csv(monthly_path.with_suffix(".csv"), index=False)

            aggregate_outputs[f"{level}_{precision_name}"] = str(total_path)
            aggregate_outputs[f"{level}_month_{precision_name}"] = str(monthly_path)

    # Coverage audits by geography level and precision filter.
    coverage_rows = []
    for level in ["cd", "ct"]:
        join_col = f"{level}_join_success"
        id_col = f"{level}_id"
        if join_col not in joined.columns:
            continue
        for precision_name, precision_col in PRECISION_FILTERS.items():
            tmp = joined.copy()
            if precision_col is not None:
                tmp = tmp[tmp[precision_col].fillna(False)].copy()
            coverage_rows.append(
                {
                    "level": level,
                    "precision_filter": precision_name,
                    "n_events_considered": int(len(tmp)),
                    "n_joined": int(tmp[join_col].fillna(False).sum()),
                    "n_unmatched": int((~tmp[join_col].fillna(False)).sum()),
                    "join_rate": float(tmp[join_col].fillna(False).mean()) if len(tmp) else math.nan,
                    "n_unique_geographies_with_events": int(tmp.loc[tmp[join_col].fillna(False), id_col].nunique()),
                }
            )
    pd.DataFrame(coverage_rows).to_csv(config.audit_dir / "spatial_join_coverage_by_filter.csv", index=False)

    summary = {
        "status": "completed",
        "clean_events_path": str(config.clean_events_path),
        "processed_dir": str(config.processed_dir),
        "audit_dir": str(config.audit_dir),
        "boundary_paths_used": boundary_paths_used,
        "missing_boundary_notes": missing_boundary_notes,
        "n_input_events": int(len(events)),
        "outputs": outputs,
        "aggregate_outputs": aggregate_outputs,
    }
    write_json(summary, config.audit_dir / "spatial_join_summary.json")

    print("Québec civil-security events spatial join completed.")
    print(f"Input events: {len(events):,}")
    print(f"Event-level output: {joined_path}")
    print(f"Audit directory: {config.audit_dir}")
    print()
    print("Boundary paths used:")
    print(f"  CD: {boundary_paths_used['cd']}")
    print(f"  CT: {boundary_paths_used['ct']}")
    if missing_boundary_notes:
        print()
        print("Missing boundary notes:")
        for note in missing_boundary_notes:
            print(f"  - {note}")
    print()
    print("Aggregate outputs:")
    for key, value in aggregate_outputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
