from pathlib import Path
import geopandas as gpd


# -----------------------------
# Configuration
# -----------------------------

RAW_SHP_PATH = Path("lct_000b21a_e.shp")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

QUEBEC_PRUID = "24"

OUTPUT_GPKG = OUTPUT_DIR / "clean_quebec_census_tracts_2021.gpkg"
OUTPUT_GEOJSON = OUTPUT_DIR / "clean_quebec_census_tracts_2021.geojson"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_quebec_census_tracts_2021.parquet"


# -----------------------------
# Load raw shapefile
# -----------------------------

gdf = gpd.read_file(RAW_SHP_PATH)

print("Loaded raw file")
print("Rows:", len(gdf))
print("Columns:", list(gdf.columns))
print("CRS:", gdf.crs)


# -----------------------------
# Basic validation
# -----------------------------

required_columns = {
    "CTUID",
    "DGUID",
    "CTNAME",
    "LANDAREA",
    "PRUID",
    "geometry",
}

missing_columns = required_columns - set(gdf.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if gdf.crs is None:
    raise ValueError("The shapefile has no CRS. Expected EPSG:3347.")

if str(gdf.crs).upper() != "EPSG:3347":
    print(f"Warning: expected EPSG:3347, got {gdf.crs}")


# -----------------------------
# Filter to Québec
# -----------------------------

gdf["PRUID"] = gdf["PRUID"].astype(str)

gdf_qc = gdf[gdf["PRUID"] == QUEBEC_PRUID].copy()

print("Québec census tracts:", len(gdf_qc))

if gdf_qc.empty:
    raise ValueError("No Québec census tracts found. Check PRUID values.")


# -----------------------------
# Create canonical clean schema
# -----------------------------

clean = gdf_qc.rename(
    columns={
        "CTUID": "unit_id",
        "DGUID": "statcan_dguid",
        "CTNAME": "unit_name",
        "LANDAREA": "land_area_km2",
        "PRUID": "province_id",
    }
).copy()

clean["unit_type"] = "census_tract"
clean["census_year"] = 2021
clean["province_name"] = "Quebec"
clean["source"] = "Statistics Canada 2021 Census Tract Cartographic Boundary File"

# Keep only useful canonical columns
clean = clean[
    [
        "unit_id",
        "statcan_dguid",
        "unit_name",
        "unit_type",
        "census_year",
        "province_id",
        "province_name",
        "land_area_km2",
        "source",
        "geometry",
    ]
]


# -----------------------------
# Validate clean table
# -----------------------------

if clean["unit_id"].duplicated().any():
    duplicated = clean[clean["unit_id"].duplicated(keep=False)]
    raise ValueError(f"Duplicated unit_id values found:\n{duplicated[['unit_id', 'unit_name']]}")

if clean["geometry"].isna().any():
    raise ValueError("Some rows have missing geometry.")

if clean["land_area_km2"].isna().any():
    raise ValueError("Some rows have missing land_area_km2.")

print("\nClean table ready")
print("Rows:", len(clean))
print("Columns:", list(clean.columns))
print(clean.drop(columns="geometry").head())


# -----------------------------
# Save outputs
# -----------------------------

# Best GIS/modeling format
clean.to_file(OUTPUT_GPKG, layer="census_tracts_2021_quebec", driver="GPKG")

# Web/map-friendly format
clean.to_file(OUTPUT_GEOJSON, driver="GeoJSON")

# Efficient analytics/modeling format
clean.to_parquet(OUTPUT_PARQUET)

print("\nSaved:")
print(OUTPUT_GPKG)
print(OUTPUT_GEOJSON)
print(OUTPUT_PARQUET)