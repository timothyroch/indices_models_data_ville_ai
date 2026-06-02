from pathlib import Path
import pandas as pd
import geopandas as gpd


# ============================================================
# Clean Census Division Residential Care per 100k Population
# ============================================================
#
# Purpose:
#   Compute a census-division-level residential-care-facilities-per-100k
#   population feature using:
#
#   Numerator:
#       ODHF nursing/residential-care facility record counts by census division
#
#   Denominator:
#       2021 census-division population from the reusable CD spatial frame
#
# Formula:
#   residential_care_facilities_per_100k_population_odhf =
#       residential_care_facility_count_odhf / population_total_2021 * 100000
#
# Inputs:
#   residential_care_per_capita/output/
#       clean_census_division_residential_care_counts_odhf_2021.csv
#
#   census_division_spatial_frame_population_2021/output/
#       clean_quebec_census_division_spatial_frame_with_population_2021.geojson
#
# Outputs:
#   residential_care_per_capita/output/
#       clean_census_division_residential_care_per_100k_population_odhf_2021.csv
#       clean_census_division_residential_care_per_100k_population_odhf_2021.parquet
#       clean_census_division_residential_care_per_100k_population_odhf_2021.geojson
#       clean_census_division_residential_care_per_100k_population_odhf_2021.gpkg
#
# Run from data/:
#   python residential_care_per_capita/clean_census_division_residential_care_per_100k_population_odhf_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

RESIDENTIAL_CARE_COUNTS_PATH = (
    DATA_DIR
    / "residential_care_per_capita"
    / "output"
    / "clean_census_division_residential_care_counts_odhf_2021.csv"
)

CD_POPULATION_GEO_PATH = (
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.geojson"
)

OUTPUT_DIR = DATA_DIR / "residential_care_per_capita" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "clean_census_division_residential_care_per_100k_population_odhf_2021.csv"
)

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "clean_census_division_residential_care_per_100k_population_odhf_2021.parquet"
)

OUTPUT_GEOJSON = (
    OUTPUT_DIR
    / "clean_census_division_residential_care_per_100k_population_odhf_2021.geojson"
)

OUTPUT_GPKG = (
    OUTPUT_DIR
    / "clean_census_division_residential_care_per_100k_population_odhf_2021.gpkg"
)


# -----------------------------
# Constants
# -----------------------------

GEOGRAPHY_YEAR = 2021
ODHF_SOURCE_YEAR = 2020

SOURCE_RESIDENTIAL_CARE = (
    "Statistics Canada Open Database of Healthcare Facilities (ODHF), version 1.1"
)

SOURCE_POPULATION = (
    "Statistics Canada Table 98-10-0007-01, "
    "Population and dwelling counts: Canada and census divisions"
)

FEATURE_DESCRIPTION = (
    "ODHF nursing/residential-care facility records per 100,000 population by "
    "2021 Quebec census division. The numerator is "
    "residential_care_facility_count_odhf from ODHF records classified as "
    "Nursing and residential care facilities. The denominator is "
    "population_total_2021 from Statistics Canada Table 98-10-0007-01. "
    "This is a facility-record proxy for residential/institutional care "
    "infrastructure, not a count of CHSLD residents or long-term-care beds."
)


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


# -----------------------------
# Load inputs
# -----------------------------

if not RESIDENTIAL_CARE_COUNTS_PATH.exists():
    raise FileNotFoundError(
        f"Residential-care counts file not found:\n{RESIDENTIAL_CARE_COUNTS_PATH}"
    )

if not CD_POPULATION_GEO_PATH.exists():
    raise FileNotFoundError(
        f"Census division population spatial frame not found:\n{CD_POPULATION_GEO_PATH}"
    )

residential_counts = pd.read_csv(RESIDENTIAL_CARE_COUNTS_PATH, dtype=str)
cd_population = gpd.read_file(CD_POPULATION_GEO_PATH)

print("\nLoaded residential-care counts")
print("Rows:", len(residential_counts))
print("Columns:", list(residential_counts.columns))

print("\nLoaded census-division population spatial frame")
print("Rows:", len(cd_population))
print("Columns:", list(cd_population.columns))
print("CRS:", cd_population.crs)


# -----------------------------
# Validate input columns
# -----------------------------

required_residential_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "census_division_type",
    "residential_care_facility_count_odhf",
    "residential_care_facility_count_automatic_csd_uid",
    "residential_care_facility_count_manual_repair",
]

required_population_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "census_division_type",
    "population_total_2021",
    "population_total_2016",
    "population_change_pct_2016_2021",
    "total_private_dwellings_2021",
    "private_dwellings_occupied_by_usual_residents_2021",
    "land_area_km2",
    "population_density_per_km2_2021",
    "geometry",
]

require_columns(
    residential_counts,
    required_residential_cols,
    "residential-care counts file",
)

require_columns(
    cd_population,
    required_population_cols,
    "CD population spatial frame",
)


# -----------------------------
# Clean and normalize inputs
# -----------------------------

residential_counts = residential_counts.copy()
cd_population = cd_population.copy()

for col in [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "census_division_type",
]:
    residential_counts[col] = clean_text(residential_counts[col])
    cd_population[col] = clean_text(cd_population[col])

numeric_residential_cols = [
    "residential_care_facility_count_odhf",
    "residential_care_facility_count_automatic_csd_uid",
    "residential_care_facility_count_manual_repair",
]

for col in numeric_residential_cols:
    residential_counts[col] = pd.to_numeric(
        residential_counts[col],
        errors="coerce",
    )

numeric_population_cols = [
    "population_total_2021",
    "population_total_2016",
    "population_change_pct_2016_2021",
    "total_private_dwellings_2021",
    "private_dwellings_occupied_by_usual_residents_2021",
    "land_area_km2",
    "population_density_per_km2_2021",
]

for col in numeric_population_cols:
    cd_population[col] = pd.to_numeric(cd_population[col], errors="coerce")


# -----------------------------
# Validate row counts and keys
# -----------------------------

if len(residential_counts) != 98:
    raise ValueError(
        f"Expected 98 residential-care-count rows, got {len(residential_counts)}."
    )

if len(cd_population) != 98:
    raise ValueError(f"Expected 98 population rows, got {len(cd_population)}.")

if residential_counts["census_division_dguid"].duplicated().any():
    duplicated = residential_counts[
        residential_counts["census_division_dguid"].duplicated(keep=False)
    ]
    raise ValueError(
        "Duplicated census_division_dguid values in residential-care counts:\n"
        + duplicated[
            [
                "census_division_code",
                "census_division_dguid",
                "census_division_name",
            ]
        ].to_string(index=False)
    )

if cd_population["census_division_dguid"].duplicated().any():
    duplicated = cd_population[
        cd_population["census_division_dguid"].duplicated(keep=False)
    ]
    raise ValueError(
        "Duplicated census_division_dguid values in population frame:\n"
        + duplicated[
            [
                "census_division_code",
                "census_division_dguid",
                "census_division_name",
            ]
        ].to_string(index=False)
    )


# -----------------------------
# Select population columns for join
# -----------------------------

population_keep_cols = [
    "census_division_dguid",
    "population_total_2021",
    "population_total_2016",
    "population_change_pct_2016_2021",
    "total_private_dwellings_2021",
    "private_dwellings_occupied_by_usual_residents_2021",
    "land_area_km2",
    "population_density_per_km2_2021",
    "geometry",
]

population_join = cd_population[population_keep_cols].copy()


# -----------------------------
# Join residential-care counts to population frame
# -----------------------------

joined = residential_counts.merge(
    population_join,
    on="census_division_dguid",
    how="left",
    validate="one_to_one",
    indicator=True,
)

joined["matched_population_frame"] = joined["_merge"] == "both"
joined = joined.drop(columns="_merge")

unmatched = joined[~joined["matched_population_frame"]]

if not unmatched.empty:
    raise ValueError(
        "Some residential-care-count rows did not match the population frame:\n"
        + unmatched[
            [
                "census_division_code",
                "census_division_dguid",
                "census_division_name",
            ]
        ].to_string(index=False)
    )


# -----------------------------
# Validate numeric fields
# -----------------------------

critical_numeric_cols = [
    "residential_care_facility_count_odhf",
    "residential_care_facility_count_automatic_csd_uid",
    "residential_care_facility_count_manual_repair",
    "population_total_2021",
    "population_total_2016",
    "land_area_km2",
    "population_density_per_km2_2021",
]

for col in critical_numeric_cols:
    if joined[col].isna().any():
        bad = joined[joined[col].isna()]
        raise ValueError(
            f"Missing numeric values in {col}:\n"
            + bad[
                [
                    "census_division_code",
                    "census_division_name",
                    col,
                ]
            ].to_string(index=False)
        )

if (joined["population_total_2021"] <= 0).any():
    bad = joined[joined["population_total_2021"] <= 0]
    raise ValueError(
        "Some census divisions have non-positive population_total_2021:\n"
        + bad[
            [
                "census_division_code",
                "census_division_name",
                "population_total_2021",
            ]
        ].to_string(index=False)
    )

if (
    joined["residential_care_facility_count_automatic_csd_uid"]
    + joined["residential_care_facility_count_manual_repair"]
    != joined["residential_care_facility_count_odhf"]
).any():
    bad = joined[
        (
            joined["residential_care_facility_count_automatic_csd_uid"]
            + joined["residential_care_facility_count_manual_repair"]
            != joined["residential_care_facility_count_odhf"]
        )
    ]
    raise ValueError(
        "Automatic + manual residential-care counts do not equal total count:\n"
        + bad[
            [
                "census_division_code",
                "census_division_name",
                "residential_care_facility_count_odhf",
                "residential_care_facility_count_automatic_csd_uid",
                "residential_care_facility_count_manual_repair",
            ]
        ].to_string(index=False)
    )


# -----------------------------
# Compute rates
# -----------------------------

joined["residential_care_facilities_per_100k_population_odhf"] = (
    joined["residential_care_facility_count_odhf"]
    / joined["population_total_2021"]
    * 100000
)

joined["residential_care_facilities_per_10k_population_odhf"] = (
    joined["residential_care_facility_count_odhf"]
    / joined["population_total_2021"]
    * 10000
)

joined["residential_care_facility_count_per_km2_odhf"] = (
    joined["residential_care_facility_count_odhf"]
    / joined["land_area_km2"]
)

joined["has_residential_care_facility_odhf"] = (
    joined["residential_care_facility_count_odhf"] > 0
)


# -----------------------------
# Add metadata
# -----------------------------

joined["unit_type"] = "census_division"
joined["geography_year"] = GEOGRAPHY_YEAR
joined["odhf_source_year"] = ODHF_SOURCE_YEAR
joined["source_residential_care"] = SOURCE_RESIDENTIAL_CARE
joined["source_population"] = SOURCE_POPULATION
joined["feature_description"] = FEATURE_DESCRIPTION


# -----------------------------
# Build GeoDataFrame
# -----------------------------

gdf = gpd.GeoDataFrame(joined, geometry="geometry", crs=cd_population.crs)

preferred_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "census_division_type",
    "unit_type",
    "geography_year",
    "odhf_source_year",
    "residential_care_facility_count_odhf",
    "residential_care_facility_count_automatic_csd_uid",
    "residential_care_facility_count_manual_repair",
    "has_residential_care_facility_odhf",
    "population_total_2021",
    "population_total_2016",
    "population_change_pct_2016_2021",
    "total_private_dwellings_2021",
    "private_dwellings_occupied_by_usual_residents_2021",
    "land_area_km2",
    "population_density_per_km2_2021",
    "residential_care_facilities_per_100k_population_odhf",
    "residential_care_facilities_per_10k_population_odhf",
    "residential_care_facility_count_per_km2_odhf",
    "matched_population_frame",
    "source_residential_care",
    "source_population",
    "feature_description",
    "geometry",
]

existing_preferred_cols = [col for col in preferred_cols if col in gdf.columns]
remaining_cols = [col for col in gdf.columns if col not in existing_preferred_cols]

gdf = gdf[existing_preferred_cols + remaining_cols].copy()


# -----------------------------
# Final validation
# -----------------------------

if len(gdf) != 98:
    raise ValueError(f"Expected 98 rows in final output, got {len(gdf)}.")

if gdf["residential_care_facilities_per_100k_population_odhf"].isna().any():
    bad = gdf[
        gdf["residential_care_facilities_per_100k_population_odhf"].isna()
    ]
    raise ValueError(
        "Missing residential_care_facilities_per_100k_population_odhf values:\n"
        + bad[
            [
                "census_division_code",
                "census_division_name",
                "residential_care_facility_count_odhf",
                "population_total_2021",
            ]
        ].to_string(index=False)
    )

if (gdf["residential_care_facilities_per_100k_population_odhf"] < 0).any():
    bad = gdf[
        gdf["residential_care_facilities_per_100k_population_odhf"] < 0
    ]
    raise ValueError(
        "Negative residential_care_facilities_per_100k_population_odhf values:\n"
        + bad[
            [
                "census_division_code",
                "census_division_name",
                "residential_care_facilities_per_100k_population_odhf",
            ]
        ].to_string(index=False)
    )


# -----------------------------
# Save outputs
# -----------------------------

# CSV with WKT geometry.
csv_out = gdf.copy()
csv_out["geometry_wkt"] = csv_out.geometry.to_wkt()
csv_out = pd.DataFrame(csv_out.drop(columns="geometry"))
csv_out.to_csv(OUTPUT_CSV, index=False)

gdf.to_parquet(OUTPUT_PARQUET, index=False)
gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
gdf.to_file(
    OUTPUT_GPKG,
    layer="residential_care_per_100k_population_odhf_2021",
    driver="GPKG",
)


# -----------------------------
# Diagnostics
# -----------------------------

print("\nFinal residential-care-per-100k table")
print("Rows:", len(gdf))
print("CRS:", gdf.crs)

print("\nCount totals:")
print(
    "Total residential_care_facility_count_odhf:",
    int(gdf["residential_care_facility_count_odhf"].sum()),
)
print(
    "Total automatic assignments:",
    int(gdf["residential_care_facility_count_automatic_csd_uid"].sum()),
)
print(
    "Total manual repairs:",
    int(gdf["residential_care_facility_count_manual_repair"].sum()),
)
print(
    "Census divisions with at least one residential-care facility:",
    int(gdf["has_residential_care_facility_odhf"].sum()),
)
print(
    "Census divisions with zero residential-care facilities:",
    int((~gdf["has_residential_care_facility_odhf"]).sum()),
)

print("\nRate summary:")
print(gdf["residential_care_facilities_per_100k_population_odhf"].describe().to_string())

print("\nTop 20 census divisions by residential-care facilities per 100k population:")
print(
    gdf[
        [
            "census_division_code",
            "census_division_name",
            "residential_care_facility_count_odhf",
            "population_total_2021",
            "residential_care_facilities_per_100k_population_odhf",
        ]
    ]
    .sort_values("residential_care_facilities_per_100k_population_odhf", ascending=False)
    .head(20)
    .to_string(index=False)
)

print("\nTop 20 census divisions by residential-care facility count:")
print(
    gdf[
        [
            "census_division_code",
            "census_division_name",
            "residential_care_facility_count_odhf",
            "population_total_2021",
            "residential_care_facilities_per_100k_population_odhf",
        ]
    ]
    .sort_values("residential_care_facility_count_odhf", ascending=False)
    .head(20)
    .to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_PARQUET)
print(OUTPUT_GEOJSON)
print(OUTPUT_GPKG)

print("\nDone.")