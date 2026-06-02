from pathlib import Path
import geopandas as gpd
import pandas as pd


# -----------------------------
# Paths
# -----------------------------

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent

BOUNDARY_PATH = (
    DATA_DIR
    / "2021-census-boundaries-file"
    / "output"
    / "clean_quebec_census_tracts_2021.parquet"
)

POPULATION_PATH = (
    DATA_DIR
    / "census_profile_2021"
    / "output"
    / "clean_census_tract_population_2021.parquet"
)

OUTPUT_DIR = THIS_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_PARQUET = OUTPUT_DIR / "clean_quebec_census_tract_spatial_frame_with_population_2021.parquet"
OUTPUT_GPKG = OUTPUT_DIR / "clean_quebec_census_tract_spatial_frame_with_population_2021.gpkg"
OUTPUT_GEOJSON = OUTPUT_DIR / "clean_quebec_census_tract_spatial_frame_with_population_2021.geojson"


# -----------------------------
# Load inputs
# -----------------------------

boundaries = gpd.read_parquet(BOUNDARY_PATH)
population = pd.read_parquet(POPULATION_PATH)

print("\nLoaded boundaries")
print("Rows:", len(boundaries))
print("Columns:", list(boundaries.columns))
print("CRS:", boundaries.crs)

print("\nLoaded population")
print("Rows:", len(population))
print("Columns:", list(population.columns))


# -----------------------------
# Validate inputs
# -----------------------------

required_boundary_cols = {
    "unit_id",
    "statcan_dguid",
    "unit_name",
    "unit_type",
    "census_year",
    "province_id",
    "province_name",
    "land_area_km2",
    "geometry",
}

required_population_cols = {
    "statcan_dguid",
    "population_total",
}

missing_boundary = required_boundary_cols - set(boundaries.columns)
missing_population = required_population_cols - set(population.columns)

if missing_boundary:
    raise ValueError(f"Boundary file missing columns: {missing_boundary}")

if missing_population:
    raise ValueError(f"Population file missing columns: {missing_population}")

if boundaries["statcan_dguid"].duplicated().any():
    raise ValueError("Duplicated statcan_dguid values found in boundary table.")

if population["statcan_dguid"].duplicated().any():
    raise ValueError("Duplicated statcan_dguid values found in population table.")


# -----------------------------
# Preserve source metadata clearly
# -----------------------------

# The boundary cleaning script currently has a generic "source" column.
# Rename it before joining so the final table is explicit.
if "source" in boundaries.columns:
    boundaries = boundaries.rename(columns={"source": "source_boundary"})
else:
    boundaries["source_boundary"] = "Statistics Canada 2021 Census Tract Cartographic Boundary File"

# The population cleaning script also has a source column.
# Keep only the relevant fields and rename source explicitly.
population_cols = ["statcan_dguid", "population_total"]

if "source" in population.columns:
    population = population.rename(columns={"source": "source_population"})
    population_cols.append("source_population")
else:
    population["source_population"] = "Statistics Canada Census Profile, 2021 Census of Population"
    population_cols.append("source_population")

population = population[population_cols].copy()


# -----------------------------
# Join population onto spatial frame
# -----------------------------

joined = boundaries.merge(
    population,
    on="statcan_dguid",
    how="left",
    validate="one_to_one",
)

print("\nAfter join")
print("Rows:", len(joined))
print("Columns:", list(joined.columns))


# -----------------------------
# Post-join checks
# -----------------------------

missing_pop = joined["population_total"].isna().sum()
print("\nMissing population_total after join:", missing_pop)

if missing_pop > 0:
    print("\nRows missing population_total after join:")
    print(
        joined.loc[
            joined["population_total"].isna(),
            ["unit_id", "statcan_dguid", "unit_name", "province_id"],
        ].head(30).to_string(index=False)
    )

# Keep zero-population / unavailable-population tracts.
# Do not drop them here; index-specific scripts decide later.
joined["has_positive_population"] = joined["population_total"].fillna(0) > 0

print("\nPopulation summary:")
print(joined["population_total"].describe())

print("\nZero-population or unavailable-population tracts kept:")
print((joined["has_positive_population"] == False).sum())


# -----------------------------
# Reorder columns
# -----------------------------

joined = joined[
    [
        "unit_id",
        "statcan_dguid",
        "unit_name",
        "unit_type",
        "census_year",
        "province_id",
        "province_name",
        "land_area_km2",
        "population_total",
        "has_positive_population",
        "source_boundary",
        "source_population",
        "geometry",
    ]
]


# -----------------------------
# Save outputs
# -----------------------------

joined.to_parquet(OUTPUT_PARQUET)
joined.to_file(OUTPUT_GPKG, layer="spatial_frame_population_2021", driver="GPKG")
joined.to_file(OUTPUT_GEOJSON, driver="GeoJSON")

print("\nSaved:")
print(OUTPUT_PARQUET)
print(OUTPUT_GPKG)
print(OUTPUT_GEOJSON)

print("\nPreview:")
print(joined.drop(columns="geometry").head(10).to_string(index=False))