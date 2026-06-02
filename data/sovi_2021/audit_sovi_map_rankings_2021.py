from pathlib import Path

import pandas as pd
import geopandas as gpd


# ============================================================
# Audit Top/Bottom SoVI Census Divisions for Québec 2021
# ============================================================
#
# Purpose:
#   Load the map-ready SoVI GeoPackage or GeoJSON created by:
#
#       create_sovi_geojson_map_2021.py
#
#   Then print and save tables showing the highest- and lowest-vulnerability
#   census divisions, with names, ranks, classes, quality flags, and missing
#   counts.
#
# Run from project root:
#
#   python data/sovi_2021/audit_sovi_map_rankings_2021.py
#
# Or run from data/:
#
#   python sovi_2021/audit_sovi_map_rankings_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

SCRIPT_PATH = Path(__file__).resolve()

# Expected script location:
#   <project_root>/data/sovi_2021/audit_sovi_map_rankings_2021.py
DATA_DIR = SCRIPT_PATH.parent.parent
PROJECT_ROOT = DATA_DIR.parent

SOVI_RUN_DIR = (
    DATA_DIR
    / "sovi_2021"
    / "output"
    / "sovi_like_quebec_cd_2021_38var_oriented_run"
)

MAP_AUDIT_CSV = SOVI_RUN_DIR / "sovi_like_quebec_cd_2021_38var_map_audit.csv"
MAP_NATIVE_GPKG = SOVI_RUN_DIR / "sovi_like_quebec_cd_2021_38var_map_native.gpkg"
MAP_WEB_GEOJSON = SOVI_RUN_DIR / "sovi_like_quebec_cd_2021_38var_map_web.geojson"

OUTPUT_TOP_20 = SOVI_RUN_DIR / "sovi_like_quebec_cd_2021_top_20_vulnerability_audit.csv"
OUTPUT_BOTTOM_20 = SOVI_RUN_DIR / "sovi_like_quebec_cd_2021_bottom_20_vulnerability_audit.csv"
OUTPUT_FULL_RANKING = SOVI_RUN_DIR / "sovi_like_quebec_cd_2021_full_ranking_audit.csv"
OUTPUT_CLASS_COUNTS = SOVI_RUN_DIR / "sovi_like_quebec_cd_2021_class_counts_audit.csv"


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


def load_map_audit() -> pd.DataFrame:
    """
    Prefer the CSV audit because it is faster to load and already contains
    all attributes. Fall back to GPKG / GeoJSON if the CSV does not exist.
    """
    if MAP_AUDIT_CSV.exists():
        print("Loading map audit CSV:")
        print(MAP_AUDIT_CSV)
        return pd.read_csv(MAP_AUDIT_CSV)

    spatial_path = first_existing_path(
        [MAP_NATIVE_GPKG, MAP_WEB_GEOJSON],
        "SoVI map output",
    )

    print("Loading spatial map output:")
    print(spatial_path)

    gdf = gpd.read_file(spatial_path)
    return pd.DataFrame(gdf.drop(columns=gdf.geometry.name))


def require_columns(df: pd.DataFrame, required_cols: list[str], label: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {label}:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(map(str, df.columns))
        )


def available_columns(df: pd.DataFrame, desired_cols: list[str]) -> list[str]:
    return [col for col in desired_cols if col in df.columns]


# -----------------------------
# Load
# -----------------------------

map_audit = load_map_audit()

require_columns(
    map_audit,
    ["zone_id", "sovi_score_normalized_0_1"],
    "SoVI map audit",
)

map_audit = map_audit.copy()

map_audit["sovi_score_normalized_0_1"] = pd.to_numeric(
    map_audit["sovi_score_normalized_0_1"],
    errors="coerce",
)

if "sovi_rank" in map_audit.columns:
    map_audit["sovi_rank"] = pd.to_numeric(map_audit["sovi_rank"], errors="coerce")

if "sovi_missing_count" in map_audit.columns:
    map_audit["sovi_missing_count"] = pd.to_numeric(
        map_audit["sovi_missing_count"],
        errors="coerce",
    )


# -----------------------------
# Build audit tables
# -----------------------------

audit_cols = available_columns(
    map_audit,
    [
        "census_division_name",
        "census_division_code",
        "zone_id",
        "population_total_2021",
        "land_area_km2",
        "sovi_score_normalized_0_1",
        "sovi_percentile",
        "sovi_score_raw",
        "sovi_score_z",
        "sovi_rank",
        "sovi_class",
        "sovi_vulnerability_label",
        "sovi_quality_flag",
        "sovi_missing_count",
        "sovi_reproduction_level",
        "sovi_color",
        "map_label",
    ],
)

full_ranking = (
    map_audit
    .sort_values("sovi_score_normalized_0_1", ascending=False, na_position="last")
    [audit_cols]
    .reset_index(drop=True)
)

top_20 = full_ranking.head(20).copy()
bottom_20 = full_ranking.tail(20).sort_values(
    "sovi_score_normalized_0_1",
    ascending=True,
    na_position="last",
).reset_index(drop=True)

class_count_cols = available_columns(
    map_audit,
    ["sovi_class", "sovi_vulnerability_label", "sovi_quality_flag"],
)

class_count_frames = []

for col in class_count_cols:
    counts = (
        map_audit[col]
        .value_counts(dropna=False)
        .reset_index()
        .rename(columns={"index": "value", col: "count"})
    )
    counts.insert(0, "field", col)
    class_count_frames.append(counts)

if class_count_frames:
    class_counts = pd.concat(class_count_frames, ignore_index=True, sort=False)
else:
    class_counts = pd.DataFrame(columns=["field", "value", "count"])


# -----------------------------
# Save
# -----------------------------

top_20.to_csv(OUTPUT_TOP_20, index=False)
bottom_20.to_csv(OUTPUT_BOTTOM_20, index=False)
full_ranking.to_csv(OUTPUT_FULL_RANKING, index=False)
class_counts.to_csv(OUTPUT_CLASS_COUNTS, index=False)


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("SoVI MAP RANKING AUDIT")
print("=" * 72)

print("\nRows:", len(map_audit))
print("Scored rows:", int(map_audit["sovi_score_normalized_0_1"].notna().sum()))
print("Missing scores:", int(map_audit["sovi_score_normalized_0_1"].isna().sum()))

print("\nTop 20 highest-vulnerability census divisions:")
print(top_20.to_string(index=False))

print("\nBottom 20 lowest-vulnerability census divisions:")
print(bottom_20.to_string(index=False))

if not class_counts.empty:
    print("\nClass / label / quality counts:")
    print(class_counts.to_string(index=False))

print("\nSaved:")
print(OUTPUT_TOP_20)
print(OUTPUT_BOTTOM_20)
print(OUTPUT_FULL_RANKING)
print(OUTPUT_CLASS_COUNTS)

print("\nDone.")