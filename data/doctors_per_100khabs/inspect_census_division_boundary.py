from pathlib import Path
import geopandas as gpd


# ============================================================
# Inspect 2021 Census Division Boundary File
# ============================================================
#
# Purpose:
#   Confirm that the downloaded 2021 census division boundary file contains
#   the fields we need for the doctors_per_100khabs health-region crosswalk.
#
# Run from data/:
#   python doctors_per_100khabs/inspect_census_division_boundary.py
#
# ============================================================


DATA_DIR = Path(__file__).resolve().parent.parent

BOUNDARY_PATH = (
    DATA_DIR
    / "2021-census-division-boundary-file"
    / "lcd_000b21a_e"
    / "lcd_000b21a_e.shp"
)

OUTPUT_DIR = DATA_DIR / "doctors_per_100khabs" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "quebec_census_divisions_2021_inventory.csv"


if not BOUNDARY_PATH.exists():
    raise FileNotFoundError(f"Boundary file not found:\n{BOUNDARY_PATH}")


gdf = gpd.read_file(BOUNDARY_PATH)

print("\nLoaded census division boundary file")
print("Rows:", len(gdf))
print("Columns:", list(gdf.columns))
print("CRS:", gdf.crs)

print("\nGeometry types:")
print(gdf.geometry.geom_type.value_counts())


# -----------------------------
# Basic required fields
# -----------------------------

expected_cols = [
    "CDUID",
    "DGUID",
    "CDNAME",
    "CDTYPE",
    "LANDAREA",
    "PRUID",
]

missing_cols = [col for col in expected_cols if col not in gdf.columns]

if missing_cols:
    print("\nWARNING: Missing expected columns:")
    for col in missing_cols:
        print(f"  {col}")
else:
    print("\nAll expected census-division columns found.")


# -----------------------------
# Filter to Quebec
# -----------------------------
# Quebec PRUID is usually 24.

if "PRUID" not in gdf.columns:
    raise ValueError("Cannot filter Quebec because PRUID is missing.")

qc = gdf[gdf["PRUID"].astype(str).str.strip() == "24"].copy()

print("\nQuebec census divisions")
print("Rows:", len(qc))

display_cols = [
    col for col in [
        "CDUID",
        "DGUID",
        "CDNAME",
        "CDTYPE",
        "LANDAREA",
        "PRUID",
    ]
    if col in qc.columns
]

print("\nQuebec census division inventory:")
print(qc[display_cols].sort_values("CDUID").to_string(index=False))


# -----------------------------
# Save non-spatial inventory
# -----------------------------

qc_inventory = qc[display_cols].sort_values("CDUID").copy()
qc_inventory.to_csv(OUTPUT_CSV, index=False)

print("\nSaved:")
print(OUTPUT_CSV)

print("\nDone.")