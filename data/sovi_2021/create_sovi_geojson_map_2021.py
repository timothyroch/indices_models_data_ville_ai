from pathlib import Path

import pandas as pd
import geopandas as gpd

from matplotlib import colormaps
from matplotlib.colors import Normalize, to_hex


# ============================================================
# Create Map-Ready GeoJSON for Québec 2021 SoVI-38
# ============================================================
#
# Purpose:
#   Join SoVI-38 census-division scores back to Québec census-division
#   geometry and create web-map-ready spatial outputs.
#
# Input score run:
#   data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_first_run/
#
# Main outputs:
#   sovi_like_quebec_cd_2021_38var_map_web.geojson
#   sovi_like_quebec_cd_2021_38var_map_native.gpkg
#   sovi_like_quebec_cd_2021_38var_map_audit.csv
#   sovi_like_quebec_cd_2021_38var_map_legend.csv
#
# Run from project root:
#   python data/sovi_2021/create_sovi_geojson_map_2021.py
#
# Or run from data/:
#   python sovi_2021/create_sovi_geojson_map_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

SCRIPT_PATH = Path(__file__).resolve()

# Expected script location:
#   <project_root>/data/sovi_2021/create_sovi_geojson_map_2021.py
DATA_DIR = SCRIPT_PATH.parent.parent
PROJECT_ROOT = DATA_DIR.parent

GEOMETRY_CANDIDATES = [
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.geojson",
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.gpkg",
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.shp",
]

SOVI_RUN_DIR = (
    DATA_DIR
    / "sovi_2021"
    / "output"
    / "sovi_like_quebec_cd_2021_38var_oriented_run"
)

STANDARD_OUTPUT_PATH = SOVI_RUN_DIR / "standard_output.csv"

OUTPUT_WEB_GEOJSON = SOVI_RUN_DIR / "sovi_like_quebec_cd_2021_38var_map_web.geojson"
OUTPUT_NATIVE_GPKG = SOVI_RUN_DIR / "sovi_like_quebec_cd_2021_38var_map_native.gpkg"
OUTPUT_AUDIT_CSV = SOVI_RUN_DIR / "sovi_like_quebec_cd_2021_38var_map_audit.csv"
OUTPUT_LEGEND_CSV = SOVI_RUN_DIR / "sovi_like_quebec_cd_2021_38var_map_legend.csv"


# -----------------------------
# Color configuration
# -----------------------------

CONTINUOUS_CMAP_NAME = "Greens"
NEUTRAL_CMAP_NAME = "Greys"

COLOR_MIN = 0.0
COLOR_MAX = 1.0
LEGEND_STEPS = 101


# -----------------------------
# Helpers
# -----------------------------

def first_existing_path(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find {label}. Tried:\n"
        + "\n".join(str(path) for path in paths)
    )


def require_columns(df: pd.DataFrame, required_cols: list[str], label: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {label}:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(map(str, df.columns))
        )


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def vulnerability_label(value: float | None) -> str:
    if pd.isna(value):
        return "No score"
    if value < 0.20:
        return "Very low"
    if value < 0.40:
        return "Low"
    if value < 0.60:
        return "Moderate"
    if value < 0.80:
        return "High"
    return "Very high"


def continuous_color(value: float | None) -> str:
    if pd.isna(value):
        neutral_cmap = colormaps.get_cmap(NEUTRAL_CMAP_NAME)
        return to_hex(neutral_cmap(0.35), keep_alpha=False)

    clipped = min(max(float(value), COLOR_MIN), COLOR_MAX)
    norm = Normalize(vmin=COLOR_MIN, vmax=COLOR_MAX)
    cmap = colormaps.get_cmap(CONTINUOUS_CMAP_NAME)

    return to_hex(cmap(norm(clipped)), keep_alpha=False)


def neutral_color(value: float) -> str:
    neutral_cmap = colormaps.get_cmap(NEUTRAL_CMAP_NAME)
    clipped = min(max(float(value), 0.0), 1.0)
    return to_hex(neutral_cmap(clipped), keep_alpha=False)


def sanitize_attributes_for_geojson(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()

    for col in out.columns:
        if col == out.geometry.name:
            continue
        out[col] = out[col].astype(object)
        out[col] = out[col].where(pd.notna(out[col]), None)

    return out


def build_legend() -> pd.DataFrame:
    values = [i / (LEGEND_STEPS - 1) for i in range(LEGEND_STEPS)]

    return pd.DataFrame(
        {
            "sovi_score_normalized_0_1": values,
            "sovi_color": [continuous_color(value) for value in values],
            "sovi_vulnerability_label": [vulnerability_label(value) for value in values],
            "colormap": CONTINUOUS_CMAP_NAME,
        }
    )


def pick_score_column(standard: pd.DataFrame) -> str:
    for col in ["score_normalized_0_1", "percentile"]:
        if col in standard.columns:
            return col

    raise ValueError(
        "Could not find score_normalized_0_1 or percentile in standard_output.csv."
    )


def normalize_standard_output(standard: pd.DataFrame) -> pd.DataFrame:
    standard = standard.copy()

    score_col = pick_score_column(standard)
    standard["sovi_score_normalized_0_1"] = pd.to_numeric(
        standard[score_col],
        errors="coerce",
    )

    rename_candidates = {
        "score_raw": "sovi_score_raw",
        "score_z": "sovi_score_z",
        "rank": "sovi_rank",
        "class": "sovi_class",
        "classification": "sovi_class",
        "quality_flag": "sovi_quality_flag",
        "missing_count": "sovi_missing_count",
        "percentile": "sovi_percentile",
        "reproduction_level": "sovi_reproduction_level",
    }

    for old, new in rename_candidates.items():
        if old in standard.columns and new not in standard.columns:
            standard[new] = standard[old]

    numeric_cols = [
        "sovi_score_raw",
        "sovi_score_z",
        "sovi_rank",
        "sovi_missing_count",
        "sovi_percentile",
    ]

    for col in numeric_cols:
        if col in standard.columns:
            standard[col] = pd.to_numeric(standard[col], errors="coerce")

    score_cols = [
        "zone_id",
        "sovi_score_normalized_0_1",
        "sovi_percentile",
        "sovi_score_raw",
        "sovi_score_z",
        "sovi_rank",
        "sovi_class",
        "sovi_quality_flag",
        "sovi_missing_count",
        "sovi_reproduction_level",
    ]

    score_cols = [col for col in score_cols if col in standard.columns]

    return standard[score_cols].copy()


# -----------------------------
# Load inputs
# -----------------------------

GEOMETRY_PATH = first_existing_path(GEOMETRY_CANDIDATES, "Québec CD geometry")

if not STANDARD_OUTPUT_PATH.exists():
    raise FileNotFoundError(f"SoVI standard output not found:\n{STANDARD_OUTPUT_PATH}")

geometry = gpd.read_file(GEOMETRY_PATH)
standard = pd.read_csv(STANDARD_OUTPUT_PATH)

print("\nLoaded geometry")
print("Path:", GEOMETRY_PATH)
print("Rows:", len(geometry))
print("CRS:", geometry.crs)

print("\nLoaded SoVI standard output")
print("Path:", STANDARD_OUTPUT_PATH)
print("Rows:", len(standard))
print("Columns:", list(standard.columns))


# -----------------------------
# Validate keys
# -----------------------------

require_columns(standard, ["zone_id"], "SoVI standard output")

if "zone_id" not in geometry.columns:
    if "census_division_dguid" in geometry.columns:
        geometry["zone_id"] = geometry["census_division_dguid"]
    elif "DGUID" in geometry.columns:
        geometry["zone_id"] = geometry["DGUID"]
    else:
        raise ValueError(
            "Geometry must contain zone_id, census_division_dguid, or DGUID."
        )

require_columns(geometry, ["zone_id", "geometry"], "Québec CD geometry")

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
# Normalize SoVI output columns
# -----------------------------

standard_map = normalize_standard_output(standard)


# -----------------------------
# Join scores to geometry
# -----------------------------

mapped = geometry.merge(
    standard_map,
    on="zone_id",
    how="left",
    validate="one_to_one",
    indicator=True,
)

mapped["sovi_scored"] = mapped["_merge"] == "both"
mapped = mapped.drop(columns="_merge")

print("\nJoin result")
print("Geometry rows:", len(geometry))
print("SoVI scored rows:", len(standard))
print("Map rows:", len(mapped))
print("Scored geometries:", int(mapped["sovi_scored"].sum()))
print("Unscored geometries:", int((~mapped["sovi_scored"]).sum()))

unscored = mapped[~mapped["sovi_scored"]].copy()

if not unscored.empty:
    display_cols = [
        col
        for col in [
            "zone_id",
            "census_division_code",
            "census_division_name",
            "population_total_2021",
        ]
        if col in unscored.columns
    ]
    print("\nUnscored geometries:")
    print(unscored[display_cols].head(30).to_string(index=False))


# -----------------------------
# Map styling columns
# -----------------------------

mapped["sovi_vulnerability_label"] = mapped["sovi_score_normalized_0_1"].apply(
    vulnerability_label
)

mapped["sovi_color"] = mapped["sovi_score_normalized_0_1"].apply(continuous_color)

mapped.loc[~mapped["sovi_scored"], "sovi_vulnerability_label"] = "Not scored"
mapped.loc[~mapped["sovi_scored"], "sovi_color"] = neutral_color(0.55)

scored_no_score = mapped["sovi_scored"] & mapped["sovi_score_normalized_0_1"].isna()
mapped.loc[scored_no_score, "sovi_vulnerability_label"] = "No score"
mapped.loc[scored_no_score, "sovi_color"] = neutral_color(0.30)

mapped["fill"] = mapped["sovi_color"]
mapped["fill-opacity"] = mapped["sovi_scored"].map({True: 0.78, False: 0.35})
mapped["stroke"] = neutral_color(0.85)
mapped["stroke-width"] = 0.35
mapped["stroke-opacity"] = 0.75

mapped["sovi_color_value_0_1"] = mapped["sovi_score_normalized_0_1"]

if "census_division_name" in mapped.columns:
    mapped["map_label"] = (
        mapped["census_division_name"].astype("string")
        + " | SoVI score: "
        + mapped["sovi_score_normalized_0_1"].round(4).astype("string")
        + " | Rank: "
        + mapped.get("sovi_rank", pd.Series(pd.NA, index=mapped.index)).astype("string")
        + " | "
        + mapped["sovi_vulnerability_label"].astype("string")
    )
else:
    mapped["map_label"] = (
        mapped["zone_id"].astype("string")
        + " | SoVI score: "
        + mapped["sovi_score_normalized_0_1"].round(4).astype("string")
        + " | "
        + mapped["sovi_vulnerability_label"].astype("string")
    )


# -----------------------------
# Final validation
# -----------------------------

if len(mapped) != len(geometry):
    raise ValueError("Mapped output row count changed unexpectedly.")

if mapped.geometry.isna().any():
    raise ValueError("Some mapped rows have missing geometry.")

matched_count = int(mapped["sovi_scored"].sum())

if matched_count != len(standard):
    raise ValueError(
        f"Matched scored geometries ({matched_count}) does not equal "
        f"standard output rows ({len(standard)})."
    )

missing_score_scored = int(
    mapped.loc[mapped["sovi_scored"], "sovi_score_normalized_0_1"].isna().sum()
)

if missing_score_scored:
    print(
        "\nWARNING: Some scored rows have missing normalized SoVI scores. "
        "They will be styled as 'No score'."
    )
    print("Scored rows with missing score:", missing_score_scored)


# -----------------------------
# Build slim web GeoJSON
# -----------------------------

web_cols = [
    "zone_id",
    "census_division_code",
    "census_division_name",
    "census_division_type",
    "province_code",
    "province_name",
    "population_total_2021",
    "land_area_km2",
    "sovi_score_normalized_0_1",
    "sovi_percentile",
    "sovi_score_raw",
    "sovi_score_z",
    "sovi_rank",
    "sovi_class",
    "sovi_quality_flag",
    "sovi_missing_count",
    "sovi_reproduction_level",
    "sovi_scored",
    "sovi_vulnerability_label",
    "sovi_color",
    "sovi_color_value_0_1",
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

if mapped_web.crs is None:
    raise ValueError("Mapped geometry CRS is missing; cannot safely reproject to EPSG:4326.")

mapped_web = mapped_web.to_crs(epsg=4326)
mapped_web = sanitize_attributes_for_geojson(mapped_web)


# -----------------------------
# Build full native audit output
# -----------------------------

mapped_native = mapped.copy()

if mapped_native.crs is None:
    print("\nWARNING: Native mapped layer has no CRS metadata.")

mapped_native_clean = sanitize_attributes_for_geojson(mapped_native)


# -----------------------------
# Build legend
# -----------------------------

legend = build_legend()


# -----------------------------
# Save outputs
# -----------------------------

mapped_web.to_file(OUTPUT_WEB_GEOJSON, driver="GeoJSON")

mapped_native_clean.to_file(
    OUTPUT_NATIVE_GPKG,
    layer="sovi_like_quebec_cd_2021_38var_map_native",
    driver="GPKG",
)

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
    mapped["sovi_vulnerability_label"]
    .value_counts(dropna=False)
    .reindex(
        [
            "Very low",
            "Low",
            "Moderate",
            "High",
            "Very high",
            "No score",
            "Not scored",
        ]
    )
    .fillna(0)
    .astype(int)
    .to_string()
)

print("\nSoVI normalized score summary:")
print(mapped["sovi_score_normalized_0_1"].describe().to_string())

print("\nContinuous color check:")
sample_values = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]
sample_colors = pd.DataFrame(
    {
        "sovi_score_normalized_0_1": sample_values,
        "generated_color": [continuous_color(value) for value in sample_values],
    }
)
print(sample_colors.to_string(index=False))

print("\nTop 10 highest-vulnerability census divisions:")
top_cols = [
    "zone_id",
    "census_division_code",
    "census_division_name",
    "population_total_2021",
    "sovi_score_normalized_0_1",
    "sovi_score_raw",
    "sovi_rank",
    "sovi_vulnerability_label",
    "sovi_color",
]
top_cols = [col for col in top_cols if col in mapped.columns]

print(
    mapped[mapped["sovi_scored"] & mapped["sovi_score_normalized_0_1"].notna()]
    .sort_values("sovi_score_normalized_0_1", ascending=False)
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