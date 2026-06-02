from pathlib import Path

import pandas as pd
import geopandas as gpd

from matplotlib import colormaps
from matplotlib.colors import Normalize, to_hex


# ============================================================
# Create Map-Ready GeoJSON for Québec 2021 Partial12 SVI
# ============================================================
#
# Purpose:
#   Join SVI score outputs back to census-tract geometry and create
#   web-map-ready spatial files.
#
# This script does NOT modify source data files.
# It only reads existing inputs and writes map outputs inside:
#
#   outputs/svi_quebec_2021_partial12_run/
#
# Main fixes:
#   1. Validate successful score-to-geometry joins, not non-null percentiles.
#   2. Export web GeoJSON in EPSG:4326 for MapLibre / Leaflet / web viewers.
#   3. Keep native high-precision GeoPackage in EPSG:3347.
#   4. Use continuous colormap-generated colors, not hardcoded vulnerability colors.
#   5. Convert pandas missing values to JSON-safe null values for GeoJSON export.
#
# Run from data/:
#   python svi_2021/create_svi_geojson_map_2021.py
#
# Or run from project root:
#   python data/svi_2021/create_svi_geojson_map_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

SCRIPT_PATH = Path(__file__).resolve()

# Expected script location:
#   <project_root>/data/svi_2021/create_svi_geojson_map_2021.py
DATA_DIR = SCRIPT_PATH.parent.parent
PROJECT_ROOT = DATA_DIR.parent

GEOMETRY_PATH = (
    DATA_DIR
    / "svi_2021"
    / "output"
    / "clean_quebec_census_tract_svi_input_2021.geojson"
)

SVI_RUN_DIR = PROJECT_ROOT / "outputs" / "svi_quebec_2021_partial12_run"

STANDARD_OUTPUT_PATH = SVI_RUN_DIR / "standard_output.csv"

# Web-facing slim GeoJSON, reprojected to EPSG:4326.
OUTPUT_WEB_GEOJSON = SVI_RUN_DIR / "svi_quebec_2021_partial12_map_web.geojson"

# Full audit spatial output, kept in native CRS.
OUTPUT_NATIVE_GPKG = SVI_RUN_DIR / "svi_quebec_2021_partial12_map_native.gpkg"

# Full audit CSV, geometry as WKT.
OUTPUT_AUDIT_CSV = SVI_RUN_DIR / "svi_quebec_2021_partial12_map_audit.csv"

# Legend for the continuous color scale.
OUTPUT_LEGEND_CSV = SVI_RUN_DIR / "svi_quebec_2021_partial12_map_legend.csv"


# -----------------------------
# Color configuration
# -----------------------------

# A continuous colormap name, not hardcoded class colors.
# Good alternatives:
#   "YlOrRd", "viridis", "plasma", "magma", "inferno"
CONTINUOUS_CMAP_NAME = "Greens"

# Neutral colormap for unscored / no-percentile cases.
NEUTRAL_CMAP_NAME = "Greys"

COLOR_MIN = 0.0
COLOR_MAX = 1.0
LEGEND_STEPS = 101


# -----------------------------
# Helpers
# -----------------------------

def require_columns(df: pd.DataFrame, required_cols: list[str], label: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {label}:\n"
            + "\n".join(missing)
        )


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def vulnerability_label(percentile: float | None) -> str:
    """
    Human-readable label only.

    Colors are continuous and do not depend on these bins.
    """
    if pd.isna(percentile):
        return "No percentile"
    if percentile < 0.20:
        return "Very low"
    if percentile < 0.40:
        return "Low"
    if percentile < 0.60:
        return "Moderate"
    if percentile < 0.80:
        return "High"
    return "Very high"


def continuous_color(value: float | None) -> str:
    """
    Convert exact SVI percentile to a continuous hex color.

    Example:
        0.29 and 0.31 produce different colors.
    """
    if pd.isna(value):
        neutral_cmap = colormaps.get_cmap(NEUTRAL_CMAP_NAME)
        return to_hex(neutral_cmap(0.35), keep_alpha=False)

    clipped = min(max(float(value), COLOR_MIN), COLOR_MAX)
    norm = Normalize(vmin=COLOR_MIN, vmax=COLOR_MAX)
    cmap = colormaps.get_cmap(CONTINUOUS_CMAP_NAME)

    return to_hex(cmap(norm(clipped)), keep_alpha=False)


def neutral_color(value: float) -> str:
    """
    Generate neutral color from neutral colormap.

    This avoids hardcoding special-case colors.
    """
    neutral_cmap = colormaps.get_cmap(NEUTRAL_CMAP_NAME)
    clipped = min(max(float(value), 0.0), 1.0)
    return to_hex(neutral_cmap(clipped), keep_alpha=False)


def sanitize_attributes_for_geojson(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert pandas missing values to JSON-safe nulls for all non-geometry columns.

    Geometry is left untouched.
    """
    out = gdf.copy()

    for col in out.columns:
        if col == out.geometry.name:
            continue

        # Object conversion helps avoid pandas extension dtypes such as pd.NA
        # leaking into Fiona/GeoJSON serialization.
        out[col] = out[col].astype(object)
        out[col] = out[col].where(pd.notna(out[col]), None)

    return out


def build_legend() -> pd.DataFrame:
    values = [i / (LEGEND_STEPS - 1) for i in range(LEGEND_STEPS)]

    return pd.DataFrame(
        {
            "svi_percentile": values,
            "svi_color": [continuous_color(value) for value in values],
            "svi_vulnerability_label": [vulnerability_label(value) for value in values],
            "colormap": CONTINUOUS_CMAP_NAME,
        }
    )


# -----------------------------
# Load inputs
# -----------------------------

if not GEOMETRY_PATH.exists():
    raise FileNotFoundError(f"Geometry file not found:\n{GEOMETRY_PATH}")

if not STANDARD_OUTPUT_PATH.exists():
    raise FileNotFoundError(f"SVI standard output not found:\n{STANDARD_OUTPUT_PATH}")

geometry = gpd.read_file(GEOMETRY_PATH)
standard = pd.read_csv(STANDARD_OUTPUT_PATH)

print("\nLoaded geometry")
print("Path:", GEOMETRY_PATH)
print("Rows:", len(geometry))
print("CRS:", geometry.crs)

print("\nLoaded SVI standard output")
print("Path:", STANDARD_OUTPUT_PATH)
print("Rows:", len(standard))
print("Columns:", list(standard.columns))


# -----------------------------
# Validate keys
# -----------------------------

require_columns(geometry, ["zone_id", "geometry"], "SVI input geometry")
require_columns(standard, ["zone_id"], "SVI standard output")

geometry = geometry.copy()
standard = standard.copy()

geometry["zone_id"] = clean_text(geometry["zone_id"])
standard["zone_id"] = clean_text(standard["zone_id"])

if geometry["zone_id"].duplicated().any():
    dupes = geometry[geometry["zone_id"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated zone_id values in geometry file:\n"
        + dupes[["zone_id"]].head(30).to_string(index=False)
    )

if standard["zone_id"].duplicated().any():
    dupes = standard[standard["zone_id"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated zone_id values in standard output:\n"
        + dupes[["zone_id"]].head(30).to_string(index=False)
    )


# -----------------------------
# Normalize SVI output columns
# -----------------------------

if "score_normalized_0_1" in standard.columns:
    standard["svi_percentile"] = pd.to_numeric(
        standard["score_normalized_0_1"],
        errors="coerce",
    )
elif "percentile" in standard.columns:
    standard["svi_percentile"] = pd.to_numeric(
        standard["percentile"],
        errors="coerce",
    )
else:
    raise ValueError(
        "Could not find score_normalized_0_1 or percentile in standard output."
    )

if "score_raw" in standard.columns:
    standard["svi_score_raw"] = pd.to_numeric(standard["score_raw"], errors="coerce")

if "rank" in standard.columns:
    standard["svi_rank"] = pd.to_numeric(standard["rank"], errors="coerce")

if "class" in standard.columns:
    standard["svi_class"] = standard["class"]

if "quality_flag" in standard.columns:
    standard["svi_quality_flag"] = standard["quality_flag"]

if "missing_count" in standard.columns:
    standard["svi_missing_count"] = pd.to_numeric(
        standard["missing_count"],
        errors="coerce",
    )

if "reproduction_level" in standard.columns:
    standard["svi_reproduction_level"] = standard["reproduction_level"]

score_cols = [
    "zone_id",
    "svi_percentile",
    "svi_score_raw",
    "svi_rank",
    "svi_class",
    "svi_quality_flag",
    "svi_missing_count",
    "svi_reproduction_level",
]

score_cols = [col for col in score_cols if col in standard.columns]
standard_map = standard[score_cols].copy()


# -----------------------------
# Join scores to full geometry
# -----------------------------

mapped = geometry.merge(
    standard_map,
    on="zone_id",
    how="left",
    validate="one_to_one",
    indicator=True,
)

mapped["svi_scored"] = mapped["_merge"] == "both"
mapped = mapped.drop(columns="_merge")

unscored = mapped[~mapped["svi_scored"]].copy()

print("\nJoin result")
print("Geometry rows:", len(geometry))
print("SVI scored rows:", len(standard))
print("Map rows:", len(mapped))
print("Scored geometries:", int(mapped["svi_scored"].sum()))
print("Unscored geometries:", int((~mapped["svi_scored"]).sum()))

if not unscored.empty:
    print("\nUnscored geometries, usually zero/non-positive population tracts excluded by SVI:")
    display_cols = [
        col for col in ["zone_id", "unit_name", "population"]
        if col in unscored.columns
    ]
    print(unscored[display_cols].head(30).to_string(index=False))


# -----------------------------
# Continuous map styling columns
# -----------------------------

mapped["svi_vulnerability_label"] = mapped["svi_percentile"].apply(vulnerability_label)

# Continuous color based on exact percentile.
mapped["svi_color"] = mapped["svi_percentile"].apply(continuous_color)

# Distinguish unscored geometry from scored rows with missing percentile.
mapped.loc[~mapped["svi_scored"], "svi_vulnerability_label"] = "Not scored"
mapped.loc[~mapped["svi_scored"], "svi_color"] = neutral_color(0.55)

scored_no_percentile = mapped["svi_scored"] & mapped["svi_percentile"].isna()
mapped.loc[scored_no_percentile, "svi_vulnerability_label"] = "No percentile"
mapped.loc[scored_no_percentile, "svi_color"] = neutral_color(0.30)

# SimpleStyle fields recognized by some GeoJSON web viewers.
mapped["fill"] = mapped["svi_color"]
mapped["fill-opacity"] = mapped["svi_scored"].map({True: 0.78, False: 0.35})
mapped["stroke"] = neutral_color(0.85)
mapped["stroke-width"] = 0.35
mapped["stroke-opacity"] = 0.75

# Useful numeric field for GIS styling.
mapped["svi_color_value_0_1"] = mapped["svi_percentile"]

# Human-readable label.
if "unit_name" in mapped.columns:
    mapped["map_label"] = (
        "CT "
        + mapped["unit_name"].astype("string")
        + " | SVI percentile: "
        + mapped["svi_percentile"].round(4).astype("string")
        + " | "
        + mapped["svi_vulnerability_label"].astype("string")
    )
else:
    mapped["map_label"] = (
        mapped["zone_id"].astype("string")
        + " | SVI percentile: "
        + mapped["svi_percentile"].round(4).astype("string")
        + " | "
        + mapped["svi_vulnerability_label"].astype("string")
    )


# -----------------------------
# Final validation
# -----------------------------

if len(mapped) != len(geometry):
    raise ValueError("Mapped output row count changed unexpectedly.")

if mapped.geometry.isna().any():
    raise ValueError("Some mapped rows have missing geometry.")

matched_count = int(mapped["svi_scored"].sum())
if matched_count != len(standard):
    raise ValueError(
        f"Matched scored geometries ({matched_count}) does not equal "
        f"standard output rows ({len(standard)})."
    )

missing_percentile_scored = int(
    mapped.loc[mapped["svi_scored"], "svi_percentile"].isna().sum()
)

if missing_percentile_scored:
    print(
        "\nWARNING: Some scored rows have missing SVI percentiles. "
        "They will be styled as 'No percentile'."
    )
    print("Scored rows with missing percentile:", missing_percentile_scored)


# -----------------------------
# Build slim web GeoJSON
# -----------------------------

web_cols = [
    "zone_id",
    "unit_name",
    "population",
    "svi_percentile",
    "svi_score_raw",
    "svi_rank",
    "svi_class",
    "svi_quality_flag",
    "svi_missing_count",
    "svi_reproduction_level",
    "svi_scored",
    "svi_vulnerability_label",
    "svi_color",
    "svi_color_value_0_1",
    "fill",
    "fill-opacity",
    "stroke",
    "stroke-width",
    "stroke-opacity",
    "map_label",
    "geometry",
]

web_cols = [col for col in web_cols if col in mapped.columns]
mapped_web = mapped[web_cols].copy()

# GeoJSON for web maps must be lon/lat WGS84.
if mapped_web.crs is None:
    raise ValueError("Mapped geometry CRS is missing; cannot safely reproject to EPSG:4326.")

mapped_web = mapped_web.to_crs(epsg=4326)

# Clean attributes after reprojection, before GeoJSON serialization.
mapped_web = sanitize_attributes_for_geojson(mapped_web)


# -----------------------------
# Build full native audit output
# -----------------------------

mapped_native = mapped.copy()

# Keep native CRS for GeoPackage.
if mapped_native.crs is None:
    print("\nWARNING: Native mapped layer has no CRS metadata.")

# Clean attributes for GPKG stability too, without touching geometry.
mapped_native_clean = sanitize_attributes_for_geojson(mapped_native)


# -----------------------------
# Build legend
# -----------------------------

legend = build_legend()


# -----------------------------
# Save outputs
# -----------------------------

# Web-map GeoJSON in EPSG:4326.
mapped_web.to_file(OUTPUT_WEB_GEOJSON, driver="GeoJSON")

# Full native GeoPackage in original CRS, expected EPSG:3347.
mapped_native_clean.to_file(
    OUTPUT_NATIVE_GPKG,
    layer="svi_quebec_2021_partial12_map_native",
    driver="GPKG",
)

# Full audit CSV with WKT geometry in native CRS.
csv_out = mapped_native.copy()
csv_out["geometry_wkt"] = csv_out.geometry.to_wkt()
csv_out = pd.DataFrame(csv_out.drop(columns="geometry"))
csv_out = csv_out.where(pd.notna(csv_out), None)
csv_out.to_csv(OUTPUT_AUDIT_CSV, index=False)

legend.to_csv(OUTPUT_LEGEND_CSV, index=False)


# -----------------------------
# Diagnostics
# -----------------------------

print("\nVulnerability label counts:")
print(
    mapped["svi_vulnerability_label"]
    .value_counts(dropna=False)
    .reindex(
        [
            "Very low",
            "Low",
            "Moderate",
            "High",
            "Very high",
            "No percentile",
            "Not scored",
        ]
    )
    .fillna(0)
    .astype(int)
    .to_string()
)

print("\nSVI percentile summary:")
print(mapped["svi_percentile"].describe().to_string())

print("\nContinuous color check:")
sample_values = [0.29, 0.30, 0.31, 0.50, 0.75, 0.90]
sample_colors = pd.DataFrame(
    {
        "svi_percentile": sample_values,
        "generated_color": [continuous_color(value) for value in sample_values],
    }
)
print(sample_colors.to_string(index=False))

print("\nTop 10 highest-vulnerability tracts:")
top_cols = [
    "zone_id",
    "unit_name",
    "population",
    "svi_percentile",
    "svi_score_raw",
    "svi_rank",
    "svi_vulnerability_label",
    "svi_color",
]
top_cols = [col for col in top_cols if col in mapped.columns]

print(
    mapped[mapped["svi_scored"] & mapped["svi_percentile"].notna()]
    .sort_values("svi_percentile", ascending=False)
    .head(10)[top_cols]
    .to_string(index=False)
)

print("\nOutput CRS:")
print("Web GeoJSON CRS: EPSG:4326")
print("Native GPKG CRS:", mapped_native.crs)

print("\nSaved:")
print(OUTPUT_WEB_GEOJSON)
print(OUTPUT_NATIVE_GPKG)
print(OUTPUT_AUDIT_CSV)
print(OUTPUT_LEGEND_CSV)

print("\nDone.")