from pathlib import Path
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

RAW_LOCAL = SCRIPT_DIR / "requetes311.csv"

OUT_DIR = SCRIPT_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

# Montréal 311 gives LOC_X / LOC_Y in NAD83 MTM Zone 8.
# EPSG:32188 is commonly used for NAD83 / MTM zone 8.
SOURCE_CRS = "EPSG:32188"

# Quick test: aggregate into 25m x 25m grid cells.
# This mirrors the idea of the VILLE_IA vulnerability grid, where Montréal’s
# climate-vulnerability analysis uses 25m x 25m cells.
GRID_SIZE_METERS = 25

# Restrict the test to recent data so that the first run is manageable.
START_DATE = None
END_DATE = None  # Example: "2025-12-31"


# ---------------------------------------------------------------------
# 2. Download helper
# ---------------------------------------------------------------------

def download_if_needed(url: str, path: Path) -> None:
    """
    Downloads the raw CSV only if it is not already present.
    """
    if path.exists():
        print(f"Using existing file: {path}")
        return

    import requests

    print(f"Downloading to {path}...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print("Download complete.")


# ---------------------------------------------------------------------
# 3. Raw event cleaning
# ---------------------------------------------------------------------

def normalize_text(x):
    """
    Basic text normalization for municipal categories.
    """
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    x = (
        x.replace("é", "e")
         .replace("è", "e")
         .replace("ê", "e")
         .replace("à", "a")
         .replace("â", "a")
         .replace("î", "i")
         .replace("ï", "i")
         .replace("ô", "o")
         .replace("ù", "u")
         .replace("ç", "c")
    )
    return x


def classify_311_activity(activity_name: str) -> str:
    """
    Maps raw 311 activity labels into broad VILLE_IA feature families.

    This is deliberately simple for the first prototype.
    Later, we should replace this with a reviewed taxonomy.
    """
    s = normalize_text(activity_name)

    water_patterns = [
        "eau", "egout", "aqueduc", "fuite", "inond", "refoulement",
        "puisard", "drain", "ruissel", "borne-fontaine"
    ]

    road_patterns = [
        "chausse", "nid-de-poule", "trottoir", "rue", "signalisation",
        "circulation", "route", "pavage", "voirie"
    ]

    tree_patterns = [
        "arbre", "branche", "canop", "veget", "elagu", "feuille"
    ]

    snow_patterns = [
        "neige", "deneig", "glace", "verglas", "abrasif"
    ]

    waste_patterns = [
        "dechet", "ordure", "collecte", "recycl", "compost", "encombrant"
    ]

    if any(p in s for p in water_patterns):
        return "water_drainage"
    if any(p in s for p in road_patterns):
        return "road_mobility"
    if any(p in s for p in tree_patterns):
        return "tree_canopy"
    if any(p in s for p in snow_patterns):
        return "snow_winter"
    if any(p in s for p in waste_patterns):
        return "waste_cleanliness"

    return "other"


def clean_311_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts one raw chunk into a clean event-level table.
    """

    # Standardize column names defensively.
    df.columns = [c.strip().upper() for c in df.columns]

    required = [
        "ID_UNIQUE",
        "NATURE",
        "ACTI_NOM",
        "ARRONDISSEMENT_GEO",
        "UNITE_RESP_PARENT",
        "DDS_DATE_CREATION",
        "LOC_X",
        "LOC_Y",
        "LOC_LAT",
        "LOC_LONG",
        "DERNIER_STATUT",
        "DATE_DERNIER_STATUT",
    ]

    # Keep only columns that exist, because municipal files sometimes evolve.
    keep = [c for c in required if c in df.columns]
    df = df[keep].copy()

    # Rename into a project-wide schema.
    rename_map = {
        "ID_UNIQUE": "event_id",
        "NATURE": "request_type",
        "ACTI_NOM": "activity_raw",
        "ARRONDISSEMENT_GEO": "borough_geo",
        "UNITE_RESP_PARENT": "responsible_unit",
        "DDS_DATE_CREATION": "created_at",
        "LOC_X": "x",
        "LOC_Y": "y",
        "LOC_LAT": "lat",
        "LOC_LONG": "lon",
        "DERNIER_STATUT": "last_status",
        "DATE_DERNIER_STATUT": "last_status_at",
    }

    df = df.rename(columns=rename_map)

    # Parse dates.
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    if "last_status_at" in df.columns:
        df["last_status_at"] = pd.to_datetime(df["last_status_at"], errors="coerce")

    # Optional date filtering.
    if START_DATE is not None:
        df = df[df["created_at"] >= pd.Timestamp(START_DATE)]
    if END_DATE is not None:
        df = df[df["created_at"] <= pd.Timestamp(END_DATE)]

    # Coordinates.
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Keep only georeferenced rows.
    df = df.dropna(subset=["created_at", "x", "y"])

    # Important: after coordinate filtering, some chunks may be empty.
    if df.empty:
        return df

    # Add temporal variables.
    df["year"] = df["created_at"].dt.year
    df["month"] = df["created_at"].dt.month
    df["period_month"] = df["created_at"].dt.to_period("M").astype(str)

    # Normalize important text fields.
    df["request_type_norm"] = df["request_type"].fillna("").astype(str).map(normalize_text)
    df["last_status_norm"] = df["last_status"].fillna("").astype(str).map(normalize_text)
    df["activity_family"] = df["activity_raw"].fillna("").astype(str).map(classify_311_activity)

    # Quick binary event markers.
    df["is_complaint"] = df["request_type_norm"].eq("plainte").astype(int)
    df["is_request"] = df["request_type_norm"].eq("requete").astype(int)
    df["is_comment"] = df["request_type_norm"].eq("commentaire").astype(int)
    df["is_urgent"] = df["last_status_norm"].str.contains("urgent", na=False).astype(int)
    df["is_finished"] = df["last_status_norm"].str.contains("terminee", na=False).astype(int)

    # Compute crude delay, if available.
    if "last_status_at" in df.columns:
        df["resolution_delay_hours"] = (
            df["last_status_at"] - df["created_at"]
        ).dt.total_seconds() / 3600
    else:
        df["resolution_delay_hours"] = np.nan

    return df


# ---------------------------------------------------------------------
# 4. Build 25m grid units from point coordinates
# ---------------------------------------------------------------------

def attach_25m_grid_id(df: pd.DataFrame, grid_size: int = 25) -> pd.DataFrame:
    """
    Creates a simple projected 25m x 25m grid ID from LOC_X / LOC_Y.

    This is a fast prototype. For production, we should spatially join
    events to the official VILLE_IA grid or census dissemination areas.
    """
    df = df.copy()

    df["grid_x0"] = np.floor(df["x"] / grid_size).astype(int) * grid_size
    df["grid_y0"] = np.floor(df["y"] / grid_size).astype(int) * grid_size

    df["unit_id"] = (
        "grid25m_"
        + df["grid_x0"].astype(str)
        + "_"
        + df["grid_y0"].astype(str)
    )

    return df


# ---------------------------------------------------------------------
# 5. Aggregate event table into VILLE_IA feature table
# ---------------------------------------------------------------------

def aggregate_to_feature_table(events: pd.DataFrame) -> pd.DataFrame:
    """
    Converts event-level data into one-row-per-spatial-unit-per-month features.
    """

    group_cols = ["unit_id", "period_month"]

    base = (
        events
        .groupby(group_cols)
        .agg(
            requests_total=("event_id", "count"),
            complaints_total=("is_complaint", "sum"),
            citizen_requests_total=("is_request", "sum"),
            comments_total=("is_comment", "sum"),
            urgent_total=("is_urgent", "sum"),
            finished_total=("is_finished", "sum"),
            avg_resolution_delay_hours=("resolution_delay_hours", "mean"),
            median_resolution_delay_hours=("resolution_delay_hours", "median"),
            unique_activity_count=("activity_raw", "nunique"),
            unique_responsible_units=("responsible_unit", "nunique"),
            x_centroid=("x", "mean"),
            y_centroid=("y", "mean"),
            lat_centroid=("lat", "mean"),
            lon_centroid=("lon", "mean"),
        )
        .reset_index()
    )

    # Wide counts by activity family.
    family_counts = (
        events
        .pivot_table(
            index=group_cols,
            columns="activity_family",
            values="event_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    family_counts.columns.name = None
    family_counts = family_counts.rename(
        columns={
            "water_drainage": "water_drainage_requests",
            "road_mobility": "road_mobility_requests",
            "tree_canopy": "tree_canopy_requests",
            "snow_winter": "snow_winter_requests",
            "waste_cleanliness": "waste_cleanliness_requests",
            "other": "other_requests",
        }
    )

    features = base.merge(family_counts, on=group_cols, how="left")

    # Derived proportions: useful for ML because raw counts depend on population/density.
    for col in [
        "complaints_total",
        "urgent_total",
        "water_drainage_requests",
        "road_mobility_requests",
        "tree_canopy_requests",
        "snow_winter_requests",
        "waste_cleanliness_requests",
    ]:
        if col in features.columns:
            features[f"share_{col}"] = features[col] / features["requests_total"].replace(0, np.nan)

    return features


# ---------------------------------------------------------------------
# 6. Optional geometry output for GIS / HGNN
# ---------------------------------------------------------------------

def add_grid_geometry(features: pd.DataFrame, grid_size: int = 25) -> gpd.GeoDataFrame:
    """
    Reconstructs square grid-cell geometry from unit_id.
    """
    out = features.copy()

    extracted = out["unit_id"].str.extract(r"grid25m_(?P<x0>-?\d+)_(?P<y0>-?\d+)")
    out["grid_x0"] = extracted["x0"].astype(float)
    out["grid_y0"] = extracted["y0"].astype(float)

    out["geometry"] = [
        box(x0, y0, x0 + grid_size, y0 + grid_size)
        for x0, y0 in zip(out["grid_x0"], out["grid_y0"])
    ]

    return gpd.GeoDataFrame(out, geometry="geometry", crs=SOURCE_CRS)


# ---------------------------------------------------------------------
# 7. Full pipeline
# ---------------------------------------------------------------------

def build_311_feature_table(
    raw_csv_path: Path,
    chunksize: int = 250_000,
    max_chunks: int | None = None,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Reads the large 311 CSV in chunks and returns a clean feature table.

    max_chunks is useful for a quick first test.
    Set max_chunks=2 for a fast smoke test.
    Set max_chunks=None to process the full file.
    """

    usecols = [
        "ID_UNIQUE",
        "NATURE",
        "ACTI_NOM",
        "ARRONDISSEMENT_GEO",
        "UNITE_RESP_PARENT",
        "DDS_DATE_CREATION",
        "LOC_X",
        "LOC_Y",
        "LOC_LAT",
        "LOC_LONG",
        "DERNIER_STATUT",
        "DATE_DERNIER_STATUT",
    ]

    all_features = []

    reader = pd.read_csv(
        raw_csv_path,
        usecols=lambda c: c in usecols,
        chunksize=chunksize,
        low_memory=False,
    )

    for i, chunk in enumerate(reader):
        print(f"Processing chunk {i + 1}...")

        clean = clean_311_chunk(chunk)

        if clean.empty:
            continue

        clean = attach_25m_grid_id(clean, GRID_SIZE_METERS)
        features = aggregate_to_feature_table(clean)

        all_features.append(features)

        if max_chunks is not None and i + 1 >= max_chunks:
            break

    if not all_features:
        raise ValueError("No usable rows were found.")

    # Because each chunk is aggregated separately, we aggregate again globally.
    tmp = pd.concat(all_features, ignore_index=True)

    additive_cols = [
        c for c in tmp.columns
        if c.endswith("_total")
        or c.endswith("_requests")
        or c in ["requests_total", "urgent_total", "finished_total"]
    ]

    mean_cols = [
        "avg_resolution_delay_hours",
        "median_resolution_delay_hours",
        "unique_activity_count",
        "unique_responsible_units",
        "x_centroid",
        "y_centroid",
        "lat_centroid",
        "lon_centroid",
    ]

    agg_dict = {c: "sum" for c in additive_cols if c in tmp.columns}
    agg_dict.update({c: "mean" for c in mean_cols if c in tmp.columns})

    final = (
        tmp
        .groupby(["unit_id", "period_month"], as_index=False)
        .agg(agg_dict)
    )

    # Recompute shares after final aggregation.
    for col in [
        "complaints_total",
        "urgent_total",
        "water_drainage_requests",
        "road_mobility_requests",
        "tree_canopy_requests",
        "snow_winter_requests",
        "waste_cleanliness_requests",
    ]:
        if col in final.columns:
            final[f"share_{col}"] = final[col] / final["requests_total"].replace(0, np.nan)

    final_geo = add_grid_geometry(final, GRID_SIZE_METERS)

    return final, final_geo


# ---------------------------------------------------------------------
# 8. Run the test
# ---------------------------------------------------------------------

# For a quick test, use max_chunks=2.
# For the full file, use max_chunks=None.
feature_table, feature_geo = build_311_feature_table(
    RAW_LOCAL,
    chunksize=250_000,
    max_chunks=None,
)

# Save ML-ready table.
feature_table.to_parquet(OUT_DIR / "ville_ia_311_features_grid25m_monthly.parquet", index=False)
feature_table.to_csv(OUT_DIR / "ville_ia_311_features_grid25m_monthly.csv", index=False)

# Save GIS-ready table.
feature_geo.to_file(
    OUT_DIR / "ville_ia_311_features_grid25m_monthly.geojson",
    driver="GeoJSON",
)

print(feature_table.head())
print(feature_table.shape)