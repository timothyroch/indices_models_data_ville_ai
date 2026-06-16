#!/usr/bin/env python3
"""
Clean Québec civil-security events.

Default raw input:
    data/external/quebec_civil_security_events/raw/quebec_civil_security_events.json

Default outputs:
    data/external/quebec_civil_security_events/processed/quebec_civil_security_events_clean.parquet
    data/external/quebec_civil_security_events/processed/quebec_civil_security_events_clean.csv
    data/external/quebec_civil_security_events/processed/quebec_civil_security_events_clean.geojson

Default audits:
    data/external/quebec_civil_security_events/audits/cleaning_summary.json
    data/external/quebec_civil_security_events/audits/cleaning_audit.csv
    data/external/quebec_civil_security_events/audits/cleaned_hazard_summary.csv
    data/external/quebec_civil_security_events/audits/cleaned_hazard_group_summary.csv
    data/external/quebec_civil_security_events/audits/cleaned_severity_summary.csv
    data/external/quebec_civil_security_events/audits/cleaned_precision_summary.csv
    data/external/quebec_civil_security_events/audits/cleaned_temporal_coverage.csv
    data/external/quebec_civil_security_events/audits/cleaned_duplicate_summary.csv

This script performs conservative event-level cleaning only. It does not spatially
join to CD/CT and does not aggregate events. That should be done by a later
02_spatial_join_quebec_civil_security_events.py script.

Run from repository root:
    python data/external/quebec_civil_security_events/01_clean_quebec_civil_security_events.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


RAW_PATH_DEFAULT = Path(
    "data/external/quebec_civil_security_events/raw/quebec_civil_security_events.json"
)
PROCESSED_DIR_DEFAULT = Path("data/external/quebec_civil_security_events/processed")
AUDIT_DIR_DEFAULT = Path("data/external/quebec_civil_security_events/audits")

QUEBEC_LON_MIN = -80.5
QUEBEC_LON_MAX = -56.0
QUEBEC_LAT_MIN = 44.0
QUEBEC_LAT_MAX = 63.5

EXPECTED_CORE_COLUMNS = [
    "code_alea",
    "alea",
    "code_municipalite",
    "municipalite",
    "precision_localisation",
    "info_compl_localisation",
    "code_severite",
    "severite",
    "date_signalement",
    "date_debut",
    "date_debut_imprecise",
    "commentaire_date_debut",
    "date_fin",
    "coord_x",
    "coord_y",
]

SEVERITY_ORDINAL = {
    "Mineure": 1,
    "Modérée": 2,
    "Importante": 3,
    "Extrême": 4,
}

LOCATION_PRECISION_RANK = {
    "Imprécise": 1,
    "Précise": 2,
    "Très précise": 3,
}

# Grouping based on observed 2026 dataset labels and close variants.
ALEA_TO_GROUP = {
    # Flood / water
    "Inondation": "flood_water",
    "Défaillance d’un ouvrage de contrôle des eaux": "flood_water",
    "Défaillance d'un ouvrage de contrôle des eaux": "flood_water",
    "Submersion côtière": "flood_water",

    # Ground / land / geomorphology
    "Mouvement de terrain": "land_ground",
    "Érosion": "land_ground",
    "Avalanche": "land_ground",
    "Séisme": "land_ground",

    # Weather / climate
    "Orage violent": "weather_climate",
    "Tempête hivernale": "weather_climate",
    "Pluie verglaçante": "weather_climate",
    "Vent violent": "weather_climate",
    "Tornade": "weather_climate",
    "Ouragan": "weather_climate",
    "Ouragan (inclut Tempête tropicale/ Tempête post-tropicale)": "weather_climate",
    "Vague de chaleur": "weather_climate",
    "Vague de froid": "weather_climate",

    # Infrastructure and essential services
    "Panne d'électricité": "infrastructure",
    "Panne de télécommunication": "infrastructure",
    "Défaillance d'infrastructure de transport": "infrastructure",
    "Défaillance d'infrastructure de transport (routier, ferroviaire, aérien, maritime)": "infrastructure",
    "Pénurie d'eau potable": "infrastructure",
    "Effondrement de bâtiment": "infrastructure",
    "Incendie d'infrastructures": "infrastructure",
    "Incendie d'infrastructure": "infrastructure",
    "Incendie d'infrastructure (multiples ou essentielles)": "infrastructure",

    # Wildfire
    "Feu de forêt": "wildfire",

    # Hazardous material / health / social
    "Matières dangereuses": "hazmat_health_social",
    "Contamination ou mauvaise qualité de l'air": "hazmat_health_social",
    "Épidémie": "hazmat_health_social",
    "Désordre social": "hazmat_health_social",

    # Transport accidents
    "Accident routier": "transport_accident",
    "Accident ferroviaire": "transport_accident",
    "Accident maritime": "transport_accident",
    "Accident aérien": "transport_accident",

    # Other
    "Autre": "other",
}

HAZARD_GROUP_ORDER = [
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


@dataclass(frozen=True)
class CleanConfig:
    raw_path: Path
    processed_dir: Path
    audit_dir: Path
    write_csv: bool
    write_geojson: bool
    fail_on_unmapped_alea: bool


def parse_args() -> CleanConfig:
    parser = argparse.ArgumentParser(
        description="Conservatively clean Québec civil-security events."
    )
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH_DEFAULT)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR_DEFAULT)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR_DEFAULT)
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write the clean CSV copy.",
    )
    parser.add_argument(
        "--no-geojson",
        action="store_true",
        help="Do not write the clean GeoJSON copy.",
    )
    parser.add_argument(
        "--fail-on-unmapped-alea",
        action="store_true",
        help="Fail if an aléa label cannot be mapped to an aléa group.",
    )
    args = parser.parse_args()
    return CleanConfig(
        raw_path=args.raw_path,
        processed_dir=args.processed_dir,
        audit_dir=args.audit_dir,
        write_csv=not args.no_csv,
        write_geojson=not args.no_geojson,
        fail_on_unmapped_alea=bool(args.fail_on_unmapped_alea),
    )


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Raw file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_mapping(
    obj: Mapping[str, Any],
    *,
    prefix: str = "",
    sep: str = "__",
    max_depth: int = 4,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def _walk(value: Any, key_prefix: str, depth: int) -> None:
        if isinstance(value, Mapping) and depth < max_depth:
            for k, v in value.items():
                next_key = f"{key_prefix}{sep}{k}" if key_prefix else str(k)
                _walk(v, next_key, depth + 1)
        elif isinstance(value, list):
            out[key_prefix] = json.dumps(value, ensure_ascii=False, default=str)
        else:
            out[key_prefix] = value

    _walk(obj, prefix, 0)
    return out


def extract_geojson_feature(feature: Mapping[str, Any], index: int) -> dict[str, Any]:
    row: dict[str, Any] = {}
    props = feature.get("properties")
    if isinstance(props, Mapping):
        row.update(flatten_mapping(props))
    else:
        row["properties_raw"] = json.dumps(props, ensure_ascii=False, default=str)

    geom = feature.get("geometry")
    row["_feature_index"] = index
    row["_geometry_present"] = isinstance(geom, Mapping)
    row["_geometry_type"] = geom.get("type") if isinstance(geom, Mapping) else None

    if isinstance(geom, Mapping):
        coords = geom.get("coordinates")
        row["_geometry_coordinates_raw"] = json.dumps(coords, ensure_ascii=False, default=str)
        if geom.get("type") == "Point" and isinstance(coords, list) and len(coords) >= 2:
            row["_geometry_x"] = coords[0]
            row["_geometry_y"] = coords[1]

    return row


def extract_records(obj: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata: dict[str, Any] = {
        "top_level_type": type(obj).__name__,
        "detected_format": None,
        "top_level_keys": sorted(map(str, obj.keys())) if isinstance(obj, Mapping) else None,
    }

    if isinstance(obj, Mapping) and obj.get("type") == "FeatureCollection":
        features = obj.get("features") or []
        if not isinstance(features, list):
            raise ValueError("GeoJSON FeatureCollection has non-list features.")
        metadata["detected_format"] = "geojson_feature_collection"
        metadata["feature_count"] = len(features)
        metadata["top_level_metadata"] = {
            k: v for k, v in obj.items() if k != "features"
        }
        return [extract_geojson_feature(feat, i) for i, feat in enumerate(features)], metadata

    if isinstance(obj, list):
        metadata["detected_format"] = "json_list"
        rows = []
        for i, item in enumerate(obj):
            row = flatten_mapping(item) if isinstance(item, Mapping) else {"value": item}
            row["_record_index"] = i
            rows.append(row)
        metadata["record_count"] = len(rows)
        return rows, metadata

    if isinstance(obj, Mapping):
        for key in ["features", "records", "data", "results", "items"]:
            value = obj.get(key)
            if isinstance(value, list):
                metadata["detected_format"] = f"dict_with_{key}"
                metadata["record_container_key"] = key
                rows = []
                for i, item in enumerate(value):
                    if key == "features" and isinstance(item, Mapping) and (
                        "properties" in item or "geometry" in item
                    ):
                        row = extract_geojson_feature(item, i)
                    elif isinstance(item, Mapping):
                        row = flatten_mapping(item)
                        row["_record_index"] = i
                    else:
                        row = {"value": item, "_record_index": i}
                    rows.append(row)
                metadata["record_count"] = len(rows)
                return rows, metadata

        metadata["detected_format"] = "single_dict"
        return [flatten_mapping(obj)], metadata

    raise ValueError(f"Unsupported JSON top-level type: {type(obj).__name__}")


def clean_string(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return pd.NA
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "<empty>", "<missing>"}:
        return pd.NA
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_text_series(series: pd.Series) -> pd.Series:
    return series.map(clean_string).astype("string")


def parse_iso_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")


def parse_signalement_filtre(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%Y/%m/%d %H:%M:%S", errors="coerce")
    missing = parsed.isna() & series.notna() & (series.astype(str).str.strip() != "")
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series.loc[missing], errors="coerce")
    return parsed


def make_event_id(df: pd.DataFrame) -> pd.Series:
    if "_feature_index" in df.columns:
        idx = pd.to_numeric(df["_feature_index"], errors="coerce")
        if idx.notna().all():
            return idx.astype(int).map(lambda x: f"qc_civil_security_event_{x:06d}")

    stable_cols = [
        c for c in [
            "code_alea",
            "alea",
            "code_municipalite",
            "municipalite",
            "date_signalement",
            "date_debut",
            "date_fin",
            "coord_x",
            "coord_y",
        ]
        if c in df.columns
    ]

    def row_hash(row: pd.Series) -> str:
        raw = "|".join(str(row.get(c, "")) for c in stable_cols)
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"qc_civil_security_event_{digest}"

    return df.apply(row_hash, axis=1)


def map_alea_group(alea: Any) -> str:
    label = clean_string(alea)
    if pd.isna(label):
        return "unmapped"
    return ALEA_TO_GROUP.get(str(label), "unmapped")


def severity_ordinal(value: Any) -> float:
    label = clean_string(value)
    if pd.isna(label):
        return math.nan
    return float(SEVERITY_ORDINAL.get(str(label), math.nan))


def precision_rank(value: Any) -> float:
    label = clean_string(value)
    if pd.isna(label):
        return math.nan
    return float(LOCATION_PRECISION_RANK.get(str(label), math.nan))


def yes_no_bool(value: Any) -> Any:
    label = clean_string(value)
    if pd.isna(label):
        return pd.NA
    low = str(label).strip().lower()
    if low in {"oui", "yes", "true", "1"}:
        return True
    if low in {"non", "no", "false", "0"}:
        return False
    return pd.NA


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def loose_quebec_bbox(lon: pd.Series, lat: pd.Series) -> pd.Series:
    return (
        lon.notna()
        & lat.notna()
        & lon.between(QUEBEC_LON_MIN, QUEBEC_LON_MAX)
        & lat.between(QUEBEC_LAT_MIN, QUEBEC_LAT_MAX)
    )


def make_duplicate_key(df: pd.DataFrame) -> pd.Series:
    key_cols = [
        c for c in [
            "code_alea",
            "alea",
            "code_municipalite",
            "municipalite",
            "date_signalement",
            "date_debut",
            "date_fin",
            "lon",
            "lat",
        ]
        if c in df.columns
    ]

    def format_value(value: Any) -> str:
        if pd.isna(value):
            return "<NA>"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    return df[key_cols].apply(lambda row: "|".join(format_value(row[c]) for c in key_cols), axis=1)


def count_by(df: pd.DataFrame, col: str, *, order: list[str] | None = None) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[col, "n", "share"])

    out = (
        df[col]
        .astype("string")
        .fillna("<MISSING>")
        .replace("", "<EMPTY>")
        .value_counts(dropna=False)
        .rename_axis(col)
        .reset_index(name="n")
    )
    out["share"] = out["n"] / len(df) if len(df) else math.nan

    if order is not None:
        rank = {v: i for i, v in enumerate(order)}
        out["_order"] = out[col].map(lambda x: rank.get(str(x), len(rank) + 1))
        out = out.sort_values(["_order", "n"], ascending=[True, False]).drop(columns="_order")
    return out.reset_index(drop=True)


def temporal_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [
        "date_signalement_parsed",
        "date_signalement_filtre_parsed",
        "date_debut_parsed",
        "date_fin_parsed",
        "event_date_primary",
    ]:
        if col not in df.columns:
            continue
        s = pd.to_datetime(df[col], errors="coerce")
        rows.append(
            {
                "column": col,
                "non_missing": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "min_date": s.min(),
                "max_date": s.max(),
                "n_unique_dates": int(s.dt.date.nunique(dropna=True)) if s.notna().any() else 0,
            }
        )
    return pd.DataFrame(rows)


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [
        "date_signalement_parsed",
        "date_debut_parsed",
        "event_date_primary",
    ]:
        if col not in df.columns:
            continue
        s = pd.to_datetime(df[col], errors="coerce")
        tmp = (
            s.dropna()
            .dt.to_period("M")
            .astype(str)
            .value_counts()
            .sort_index()
            .rename_axis("period_month")
            .reset_index(name="n")
        )
        tmp["date_column"] = col
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def cleaning_audit(df: pd.DataFrame) -> pd.DataFrame:
    checks = []

    def add(name: str, value: Any, detail: str = "") -> None:
        checks.append({"check": name, "value": value, "detail": detail})

    add("n_clean_rows", int(len(df)))
    add("n_unique_event_ids", int(df["event_id"].nunique(dropna=True)))
    add("n_missing_event_date_primary", int(df["event_date_primary"].isna().sum()))
    add("n_date_debut_missing", int(df["date_debut_missing"].sum()))
    add("n_date_fin_missing", int(df["date_fin_missing"].sum()))
    add("n_date_fin_before_date_debut", int(df["date_fin_before_date_debut_flag"].sum()))
    add("n_valid_lon_lat", int(df["has_valid_lon_lat"].sum()))
    add("n_inside_loose_quebec_bbox", int(df["inside_loose_quebec_bbox"].sum()))
    add("n_outside_loose_quebec_bbox", int((df["has_valid_lon_lat"] & ~df["inside_loose_quebec_bbox"]).sum()))
    add("n_unmapped_alea_group", int((df["alea_group"] == "unmapped").sum()))
    add("n_unknown_severity", int(df["severite_unknown"].sum()))
    add("n_possible_duplicate_records", int(df["possible_duplicate_flag"].sum()))
    add("n_open_events", int(df["is_open_event"].sum()))

    return pd.DataFrame(checks)


def dataframe_to_geojson(df: pd.DataFrame, path: Path) -> None:
    features = []
    property_exclude = {"geometry", "lon", "lat"}
    for _, row in df.iterrows():
        lon = row.get("lon")
        lat = row.get("lat")
        if pd.isna(lon) or pd.isna(lat):
            continue

        props = {}
        for col, val in row.items():
            if col in property_exclude:
                continue
            if pd.isna(val):
                props[col] = None
            elif isinstance(val, pd.Timestamp):
                props[col] = val.isoformat()
            else:
                props[col] = val.item() if hasattr(val, "item") else val

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon), float(lat)],
                },
                "properties": props,
            }
        )

    obj = {
        "type": "FeatureCollection",
        "name": "quebec_civil_security_events_clean",
        "features": features,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def write_json(data: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def clean_events(raw_obj: Any, config: CleanConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    records, source_metadata = extract_records(raw_obj)
    df = pd.DataFrame(records)
    df.columns = [str(c) for c in df.columns]
    df = df.reset_index(drop=True)

    for col in [
        "alea",
        "municipalite",
        "precision_localisation",
        "info_compl_localisation",
        "severite",
        "date_debut_imprecise",
        "commentaire_date_debut",
        "etat",
        "adresse",
        "dossier",
        "nom_bilans",
    ]:
        if col in df.columns:
            df[col] = normalize_text_series(df[col])

    df["event_id"] = make_event_id(df)
    if "_feature_index" in df.columns:
        df["source_feature_index"] = safe_numeric(df["_feature_index"]).astype("Int64")
    else:
        df["source_feature_index"] = pd.Series(range(len(df)), dtype="Int64")

    if "code_alea" in df.columns:
        df["code_alea"] = safe_numeric(df["code_alea"]).astype("Int64")
    if "code_severite" in df.columns:
        df["code_severite"] = safe_numeric(df["code_severite"]).astype("Int64")
    if "code_municipalite" in df.columns:
        df["code_municipalite"] = normalize_text_series(df["code_municipalite"])

    coord_x = safe_numeric(df["coord_x"]) if "coord_x" in df.columns else pd.Series(math.nan, index=df.index)
    coord_y = safe_numeric(df["coord_y"]) if "coord_y" in df.columns else pd.Series(math.nan, index=df.index)
    geom_x = safe_numeric(df["_geometry_x"]) if "_geometry_x" in df.columns else pd.Series(math.nan, index=df.index)
    geom_y = safe_numeric(df["_geometry_y"]) if "_geometry_y" in df.columns else pd.Series(math.nan, index=df.index)

    df["lon"] = coord_x.where(coord_x.notna(), geom_x)
    df["lat"] = coord_y.where(coord_y.notna(), geom_y)
    df["has_valid_lon_lat"] = df["lon"].notna() & df["lat"].notna()
    df["inside_loose_quebec_bbox"] = loose_quebec_bbox(df["lon"], df["lat"])
    df["coord_source"] = pd.NA
    df.loc[coord_x.notna() & coord_y.notna(), "coord_source"] = "coord_x_coord_y"
    df.loc[df["coord_source"].isna() & geom_x.notna() & geom_y.notna(), "coord_source"] = "geojson_geometry"

    if "date_signalement" in df.columns:
        df["date_signalement_parsed"] = parse_iso_date(df["date_signalement"])
    else:
        df["date_signalement_parsed"] = pd.NaT

    if "date_signalement_filtre" in df.columns:
        df["date_signalement_filtre_parsed"] = parse_signalement_filtre(df["date_signalement_filtre"])
    else:
        df["date_signalement_filtre_parsed"] = pd.NaT

    if "date_debut" in df.columns:
        df["date_debut_parsed"] = parse_iso_date(df["date_debut"])
    else:
        df["date_debut_parsed"] = pd.NaT

    if "date_fin" in df.columns:
        df["date_fin_parsed"] = parse_iso_date(df["date_fin"])
    else:
        df["date_fin_parsed"] = pd.NaT

    df["date_debut_missing"] = df["date_debut_parsed"].isna()
    df["date_fin_missing"] = df["date_fin_parsed"].isna()

    df["event_date_primary"] = df["date_debut_parsed"].where(
        df["date_debut_parsed"].notna(),
        df["date_signalement_parsed"],
    )
    df["event_date_primary_source"] = "date_debut"
    df.loc[df["date_debut_parsed"].isna() & df["date_signalement_parsed"].notna(), "event_date_primary_source"] = (
        "date_signalement_fallback"
    )

    no_primary = df["event_date_primary"].isna()
    df.loc[no_primary & df["date_signalement_filtre_parsed"].notna(), "event_date_primary"] = (
        df.loc[no_primary & df["date_signalement_filtre_parsed"].notna(), "date_signalement_filtre_parsed"]
    )
    df.loc[
        df["date_debut_parsed"].isna()
        & df["date_signalement_parsed"].isna()
        & df["date_signalement_filtre_parsed"].notna(),
        "event_date_primary_source",
    ] = "date_signalement_filtre_fallback"
    df.loc[df["event_date_primary"].isna(), "event_date_primary_source"] = pd.NA

    df["event_year"] = df["event_date_primary"].dt.year.astype("Int64")
    df["event_month"] = df["event_date_primary"].dt.month.astype("Int64")
    df["event_period_month"] = df["event_date_primary"].dt.to_period("M").astype("string")

    duration = (df["date_fin_parsed"] - df["date_debut_parsed"]).dt.total_seconds() / 86400.0
    df["event_duration_days"] = duration
    df.loc[df["date_debut_parsed"].isna() | df["date_fin_parsed"].isna(), "event_duration_days"] = math.nan
    df["date_fin_before_date_debut_flag"] = (
        df["date_debut_parsed"].notna()
        & df["date_fin_parsed"].notna()
        & (df["date_fin_parsed"] < df["date_debut_parsed"])
    )

    if "date_debut_imprecise" in df.columns:
        df["date_debut_imprecise_bool"] = df["date_debut_imprecise"].map(yes_no_bool).astype("boolean")
        df["date_debut_imprecise_missing"] = df["date_debut_imprecise_bool"].isna()
    else:
        df["date_debut_imprecise_bool"] = pd.Series(pd.NA, dtype="boolean", index=df.index)
        df["date_debut_imprecise_missing"] = True

    df["alea_group"] = df["alea"].map(map_alea_group) if "alea" in df.columns else "unmapped"
    for group in HAZARD_GROUP_ORDER:
        df[f"is_alea_group__{group}"] = df["alea_group"].eq(group)

    unmapped = sorted(df.loc[df["alea_group"].eq("unmapped"), "alea"].dropna().astype(str).unique())
    if config.fail_on_unmapped_alea and unmapped:
        raise ValueError(f"Unmapped aléa labels found: {unmapped}")

    df["severite_ordinal"] = df["severite"].map(severity_ordinal) if "severite" in df.columns else math.nan
    df["severite_ordinal"] = pd.to_numeric(df["severite_ordinal"], errors="coerce").astype("Float64")
    df["severite_unknown"] = df["severite"].astype("string").eq("Inconnue") if "severite" in df.columns else True
    df["is_moderate_or_worse"] = df["severite_ordinal"].ge(2).fillna(False)
    df["is_important_or_extreme"] = df["severite_ordinal"].ge(3).fillna(False)

    if "etat" in df.columns:
        df["is_open_event"] = df["etat"].astype("string").str.lower().eq("en cours").fillna(False)
    else:
        df["is_open_event"] = False

    df["location_precision_rank"] = (
        df["precision_localisation"].map(precision_rank)
        if "precision_localisation" in df.columns
        else math.nan
    )
    df["location_precision_rank"] = pd.to_numeric(df["location_precision_rank"], errors="coerce").astype("Float64")
    df["is_precise_or_very_precise"] = df["location_precision_rank"].ge(2).fillna(False)
    df["is_very_precise"] = df["location_precision_rank"].ge(3).fillna(False)

    df["possible_duplicate_key"] = make_duplicate_key(df)
    dup_counts = df.groupby("possible_duplicate_key", dropna=False)["event_id"].transform("size")
    df["possible_duplicate_count"] = dup_counts.astype("Int64")
    df["possible_duplicate_flag"] = df["possible_duplicate_count"].gt(1)

    df["geometry_wkt"] = df.apply(
        lambda row: (
            f"POINT ({float(row['lon'])} {float(row['lat'])})"
            if pd.notna(row["lon"]) and pd.notna(row["lat"])
            else pd.NA
        ),
        axis=1,
    )

    df = df.sort_values(["event_date_primary", "source_feature_index", "event_id"], na_position="last").reset_index(drop=True)

    metadata = {
        "source_metadata": source_metadata,
        "n_records_extracted": int(len(df)),
        "unmapped_alea_labels": unmapped,
    }
    return df, metadata


def write_outputs(df: pd.DataFrame, metadata: Mapping[str, Any], config: CleanConfig) -> dict[str, str]:
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.audit_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = config.processed_dir / "quebec_civil_security_events_clean.parquet"
    csv_path = config.processed_dir / "quebec_civil_security_events_clean.csv"
    geojson_path = config.processed_dir / "quebec_civil_security_events_clean.geojson"

    df.to_parquet(parquet_path, index=False)
    outputs = {"parquet": str(parquet_path)}

    if config.write_csv:
        df.to_csv(csv_path, index=False)
        outputs["csv"] = str(csv_path)

    if config.write_geojson:
        dataframe_to_geojson(df, geojson_path)
        outputs["geojson"] = str(geojson_path)

    cleaning_audit(df).to_csv(config.audit_dir / "cleaning_audit.csv", index=False)

    count_by(df, "alea").to_csv(config.audit_dir / "cleaned_hazard_summary.csv", index=False)
    count_by(df, "alea_group", order=HAZARD_GROUP_ORDER).to_csv(
        config.audit_dir / "cleaned_hazard_group_summary.csv", index=False
    )
    count_by(df, "severite").to_csv(config.audit_dir / "cleaned_severity_summary.csv", index=False)
    count_by(df, "precision_localisation").to_csv(
        config.audit_dir / "cleaned_precision_summary.csv", index=False
    )
    count_by(df, "etat").to_csv(config.audit_dir / "cleaned_event_state_summary.csv", index=False)
    count_by(df, "event_date_primary_source").to_csv(
        config.audit_dir / "cleaned_primary_date_source_summary.csv", index=False
    )

    temporal_coverage(df).to_csv(config.audit_dir / "cleaned_temporal_coverage.csv", index=False)
    monthly_counts(df).to_csv(config.audit_dir / "cleaned_monthly_counts.csv", index=False)

    duplicate_summary = (
        df[df["possible_duplicate_flag"]]
        .groupby("possible_duplicate_key", dropna=False)
        .agg(
            possible_duplicate_count=("event_id", "size"),
            event_ids=("event_id", lambda s: "; ".join(map(str, s.head(10)))),
            alea=("alea", "first"),
            municipalite=("municipalite", "first"),
            event_date_primary=("event_date_primary", "first"),
            lon=("lon", "first"),
            lat=("lat", "first"),
        )
        .reset_index()
        .sort_values("possible_duplicate_count", ascending=False)
    )
    duplicate_summary.to_csv(config.audit_dir / "cleaned_duplicate_summary.csv", index=False)

    coordinate_summary = pd.DataFrame(
        [
            {
                "n_rows": int(len(df)),
                "n_valid_lon_lat": int(df["has_valid_lon_lat"].sum()),
                "n_inside_loose_quebec_bbox": int(df["inside_loose_quebec_bbox"].sum()),
                "n_outside_loose_quebec_bbox": int((df["has_valid_lon_lat"] & ~df["inside_loose_quebec_bbox"]).sum()),
                "lon_min": float(df["lon"].min()),
                "lon_max": float(df["lon"].max()),
                "lat_min": float(df["lat"].min()),
                "lat_max": float(df["lat"].max()),
            }
        ]
    )
    coordinate_summary.to_csv(config.audit_dir / "cleaned_coordinate_summary.csv", index=False)

    columns_audit = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "non_missing": [int(df[c].notna().sum()) for c in df.columns],
            "missing": [int(df[c].isna().sum()) for c in df.columns],
            "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )
    columns_audit.to_csv(config.audit_dir / "cleaned_columns_audit.csv", index=False)

    summary = {
        "status": "completed",
        "raw_path": str(config.raw_path),
        "processed_dir": str(config.processed_dir),
        "audit_dir": str(config.audit_dir),
        "outputs": outputs,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "n_unique_event_ids": int(df["event_id"].nunique()),
        "n_unmapped_alea_group": int(df["alea_group"].eq("unmapped").sum()),
        "unmapped_alea_labels": metadata.get("unmapped_alea_labels", []),
        "n_possible_duplicate_records": int(df["possible_duplicate_flag"].sum()),
        "n_open_events": int(df["is_open_event"].sum()),
        "n_valid_lon_lat": int(df["has_valid_lon_lat"].sum()),
        "n_inside_loose_quebec_bbox": int(df["inside_loose_quebec_bbox"].sum()),
        "date_primary_source_counts": df["event_date_primary_source"].value_counts(dropna=False).to_dict(),
        "alea_group_counts": df["alea_group"].value_counts(dropna=False).to_dict(),
        "severity_counts": df["severite"].astype("string").fillna("<MISSING>").value_counts(dropna=False).to_dict(),
        "location_precision_counts": df["precision_localisation"].astype("string").fillna("<MISSING>").value_counts(dropna=False).to_dict(),
        "source_metadata": metadata.get("source_metadata", {}),
        "clean_config": {
            "write_csv": config.write_csv,
            "write_geojson": config.write_geojson,
            "fail_on_unmapped_alea": config.fail_on_unmapped_alea,
        },
    }
    write_json(summary, config.audit_dir / "cleaning_summary.json")
    outputs["cleaning_summary"] = str(config.audit_dir / "cleaning_summary.json")
    outputs["cleaning_audit"] = str(config.audit_dir / "cleaning_audit.csv")
    return outputs


def main() -> None:
    config = parse_args()
    raw_obj = read_json(config.raw_path)
    clean_df, metadata = clean_events(raw_obj, config)
    outputs = write_outputs(clean_df, metadata, config)

    print("Québec civil-security events cleaning completed.")
    print(f"Rows: {len(clean_df):,}")
    print(f"Columns: {len(clean_df.columns):,}")
    print(f"Primary output: {outputs['parquet']}")
    if "geojson" in outputs:
        print(f"GeoJSON output: {outputs['geojson']}")
    if "csv" in outputs:
        print(f"CSV output: {outputs['csv']}")
    print(f"Audit directory: {config.audit_dir}")
    print()
    print("Key checks:")
    print(f"  Unmapped aléa labels: {int(clean_df['alea_group'].eq('unmapped').sum()):,}")
    print(f"  Valid lon/lat: {int(clean_df['has_valid_lon_lat'].sum()):,} / {len(clean_df):,}")
    print(f"  Inside loose Québec bbox: {int(clean_df['inside_loose_quebec_bbox'].sum()):,} / {len(clean_df):,}")
    print(f"  Possible duplicate records: {int(clean_df['possible_duplicate_flag'].sum()):,}")
    print(f"  Open events: {int(clean_df['is_open_event'].sum()):,}")


if __name__ == "__main__":
    main()
