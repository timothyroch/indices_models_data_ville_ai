from pathlib import Path
import pandas as pd
import geopandas as gpd


# ============================================================
# Clean Quebec Census Division Spatial Frame with Population
# ============================================================
#
# Purpose:
#   Build a reusable 2021 Quebec census-division spatial frame with population
#   and dwelling-count denominators.
#
# This layer is intended to support several SoVI-style variables:
#   - hospitals per 100k population
#   - residential-care facilities per 100k population
#   - voter-turnout spatial allocation / weighting
#   - future per-capita / density variables
#
# Inputs:
#   2021-census-division-boundary-file/lcd_000b21a_e/lcd_000b21a_e.shp
#   census_division_spatial_frame_population_2021/raw/98100007.csv
#
# Output:
#   census_division_spatial_frame_population_2021/output/
#       clean_quebec_census_division_spatial_frame_with_population_2021.csv
#       clean_quebec_census_division_spatial_frame_with_population_2021.parquet
#       clean_quebec_census_division_spatial_frame_with_population_2021.geojson
#       clean_quebec_census_division_spatial_frame_with_population_2021.gpkg
#
# Run from data/:
#   python census_division_spatial_frame_population_2021/clean_census_division_spatial_frame_population_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

BOUNDARY_PATH = (
    DATA_DIR
    / "2021-census-division-boundary-file"
    / "lcd_000b21a_e"
    / "lcd_000b21a_e.shp"
)

POPULATION_TABLE_PATH = (
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "raw"
    / "98100007.csv"
)

OUTPUT_DIR = DATA_DIR / "census_division_spatial_frame_population_2021" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "clean_quebec_census_division_spatial_frame_with_population_2021.csv"
)

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "clean_quebec_census_division_spatial_frame_with_population_2021.parquet"
)

OUTPUT_GEOJSON = (
    OUTPUT_DIR
    / "clean_quebec_census_division_spatial_frame_with_population_2021.geojson"
)

OUTPUT_GPKG = (
    OUTPUT_DIR
    / "clean_quebec_census_division_spatial_frame_with_population_2021.gpkg"
)


# -----------------------------
# Constants
# -----------------------------

CENSUS_YEAR = 2021
PROVINCE_CODE_QUEBEC = "24"
PROVINCE_NAME_QUEBEC = "Quebec"

SOURCE_BOUNDARY = "Statistics Canada 2021 Census Division Boundary File"
SOURCE_POPULATION = (
    "Statistics Canada Table 98-10-0007-01, "
    "Population and dwelling counts: Canada and census divisions"
)


# -----------------------------
# Helpers
# -----------------------------

def read_csv_fallback(path: Path) -> tuple[pd.DataFrame, str]:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1"]
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False, encoding=enc)
            return df, enc
        except UnicodeDecodeError as e:
            last_error = e
            print(f"Could not read {path.name} with encoding={enc}: {e}")

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not read {path} with tested encodings. Last error: {last_error}",
    )


def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def clean_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("r", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def find_column_containing(columns: list[str], required_terms: list[str]) -> str | None:
    """
    Find the first column whose lowercase name contains all required terms.
    """
    for col in columns:
        col_lower = col.lower()
        if all(term.lower() in col_lower for term in required_terms):
            return col
    return None


def require_columns(df: pd.DataFrame, required_cols: list[str], label: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {label}:\n"
            + "\n".join(missing)
        )


# -----------------------------
# Load inputs
# -----------------------------

if not BOUNDARY_PATH.exists():
    raise FileNotFoundError(f"Boundary shapefile not found:\n{BOUNDARY_PATH}")

if not POPULATION_TABLE_PATH.exists():
    raise FileNotFoundError(f"Population table not found:\n{POPULATION_TABLE_PATH}")

boundaries = gpd.read_file(BOUNDARY_PATH)
boundaries = normalize_colnames(boundaries)

population_raw, population_encoding = read_csv_fallback(POPULATION_TABLE_PATH)
population_raw = normalize_colnames(population_raw)

print("\nLoaded census division boundary file")
print("Rows:", len(boundaries))
print("Columns:", list(boundaries.columns))
print("CRS:", boundaries.crs)

print("\nLoaded StatCan population/dwelling table")
print("Rows:", len(population_raw))
print("Columns:", list(population_raw.columns))
print("Encoding used:", population_encoding)


# -----------------------------
# Validate boundary columns
# -----------------------------

boundary_required_cols = [
    "CDUID",
    "DGUID",
    "CDNAME",
    "CDTYPE",
    "LANDAREA",
    "PRUID",
    "geometry",
]

require_columns(boundaries, boundary_required_cols, "census division boundary file")

boundaries["PRUID"] = clean_text(boundaries["PRUID"])

quebec_boundaries = boundaries[boundaries["PRUID"] == PROVINCE_CODE_QUEBEC].copy()

print("\nQuebec census divisions from boundary file")
print("Rows:", len(quebec_boundaries))

if len(quebec_boundaries) != 98:
    raise ValueError(
        f"Expected 98 Quebec census divisions, got {len(quebec_boundaries)}."
    )

if quebec_boundaries["DGUID"].duplicated().any():
    duplicated = quebec_boundaries[
        quebec_boundaries["DGUID"].duplicated(keep=False)
    ]
    raise ValueError(
        "Duplicated DGUIDs in Quebec boundary file:\n"
        + duplicated[["CDUID", "DGUID", "CDNAME", "CDTYPE"]].to_string(index=False)
    )


# -----------------------------
# Validate and prepare population table
# -----------------------------

require_columns(population_raw, ["DGUID", "GEO"], "population/dwelling table")

population_columns = list(population_raw.columns)

population_2021_col = find_column_containing(
    population_columns,
    ["Population and dwelling counts", "Population, 2021"],
)

population_2016_col = find_column_containing(
    population_columns,
    ["Population and dwelling counts", "Population, 2016"],
)

population_change_col = find_column_containing(
    population_columns,
    ["Population percentage change", "2016 to 2021"],
)

total_private_dwellings_2021_col = find_column_containing(
    population_columns,
    ["Total private dwellings", "2021"],
)

total_private_dwellings_2016_col = find_column_containing(
    population_columns,
    ["Total private dwellings", "2016"],
)

total_private_dwellings_change_col = find_column_containing(
    population_columns,
    ["Total private dwellings percentage change", "2016 to 2021"],
)

private_dwellings_occupied_2021_col = find_column_containing(
    population_columns,
    ["Private dwellings occupied by usual residents", "2021"],
)

private_dwellings_occupied_2016_col = find_column_containing(
    population_columns,
    ["Private dwellings occupied by usual residents", "2016"],
)

private_dwellings_occupied_change_col = find_column_containing(
    population_columns,
    [
        "Private dwellings occupied by usual residents percentage change",
        "2016 to 2021",
    ],
)

land_area_2021_col = find_column_containing(
    population_columns,
    ["Land area in square kilometres", "2021"],
)

population_density_2021_col = find_column_containing(
    population_columns,
    ["Population density per square kilometre", "2021"],
)

national_population_rank_2021_col = find_column_containing(
    population_columns,
    ["National population rank", "2021"],
)

province_population_rank_2021_col = find_column_containing(
    population_columns,
    ["Province/territory population rank", "2021"],
)

detected_cols = {
    "population_2021_col": population_2021_col,
    "population_2016_col": population_2016_col,
    "population_change_col": population_change_col,
    "total_private_dwellings_2021_col": total_private_dwellings_2021_col,
    "total_private_dwellings_2016_col": total_private_dwellings_2016_col,
    "total_private_dwellings_change_col": total_private_dwellings_change_col,
    "private_dwellings_occupied_2021_col": private_dwellings_occupied_2021_col,
    "private_dwellings_occupied_2016_col": private_dwellings_occupied_2016_col,
    "private_dwellings_occupied_change_col": private_dwellings_occupied_change_col,
    "land_area_2021_col": land_area_2021_col,
    "population_density_2021_col": population_density_2021_col,
    "national_population_rank_2021_col": national_population_rank_2021_col,
    "province_population_rank_2021_col": province_population_rank_2021_col,
}

print("\nDetected StatCan value columns:")
for label, col in detected_cols.items():
    print(f"{label}: {col}")

required_detected = {
    "population_2021_col": population_2021_col,
    "population_2016_col": population_2016_col,
    "land_area_2021_col": land_area_2021_col,
    "population_density_2021_col": population_density_2021_col,
}

missing_detected = [
    label for label, col in required_detected.items() if col is None
]

if missing_detected:
    raise ValueError(
        "Could not detect required value columns:\n"
        + "\n".join(missing_detected)
    )


# -----------------------------
# Build clean population table
# -----------------------------

population = population_raw.copy()

population["population_dguid"] = clean_text(population["DGUID"])
population["population_geo_name"] = clean_text(population["GEO"])

pop_subset_cols = [
    "population_dguid",
    "population_geo_name",
    population_2021_col,
    population_2016_col,
    population_change_col,
    total_private_dwellings_2021_col,
    total_private_dwellings_2016_col,
    total_private_dwellings_change_col,
    private_dwellings_occupied_2021_col,
    private_dwellings_occupied_2016_col,
    private_dwellings_occupied_change_col,
    land_area_2021_col,
    population_density_2021_col,
    national_population_rank_2021_col,
    province_population_rank_2021_col,
]

pop_subset_cols = [
    col for col in pop_subset_cols
    if col is not None and col in population.columns
]

pop = population[pop_subset_cols].copy()

rename_map = {
    population_2021_col: "population_total_2021",
    population_2016_col: "population_total_2016",
    population_change_col: "population_change_pct_2016_2021",
    total_private_dwellings_2021_col: "total_private_dwellings_2021",
    total_private_dwellings_2016_col: "total_private_dwellings_2016",
    total_private_dwellings_change_col: "total_private_dwellings_change_pct_2016_2021",
    private_dwellings_occupied_2021_col: (
        "private_dwellings_occupied_by_usual_residents_2021"
    ),
    private_dwellings_occupied_2016_col: (
        "private_dwellings_occupied_by_usual_residents_2016"
    ),
    private_dwellings_occupied_change_col: (
        "private_dwellings_occupied_by_usual_residents_change_pct_2016_2021"
    ),
    land_area_2021_col: "land_area_km2_population_table_2021",
    population_density_2021_col: "population_density_per_km2_2021",
    national_population_rank_2021_col: "national_population_rank_2021",
    province_population_rank_2021_col: "province_population_rank_2021",
}

rename_map = {
    old: new for old, new in rename_map.items()
    if old is not None and old in pop.columns
}

pop = pop.rename(columns=rename_map)

numeric_cols = [
    "population_total_2021",
    "population_total_2016",
    "population_change_pct_2016_2021",
    "total_private_dwellings_2021",
    "total_private_dwellings_2016",
    "total_private_dwellings_change_pct_2016_2021",
    "private_dwellings_occupied_by_usual_residents_2021",
    "private_dwellings_occupied_by_usual_residents_2016",
    "private_dwellings_occupied_by_usual_residents_change_pct_2016_2021",
    "land_area_km2_population_table_2021",
    "population_density_per_km2_2021",
    "national_population_rank_2021",
    "province_population_rank_2021",
]

for col in numeric_cols:
    if col in pop.columns:
        pop[col] = clean_number(pop[col])

if pop["population_dguid"].duplicated().any():
    duplicated = pop[pop["population_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated DGUIDs in population table:\n"
        + duplicated[["population_dguid", "population_geo_name"]].to_string(index=False)
    )


# -----------------------------
# Prepare boundary table
# -----------------------------

qcd = quebec_boundaries.copy()

qcd["census_division_code"] = clean_text(qcd["CDUID"])
qcd["census_division_dguid"] = clean_text(qcd["DGUID"])
qcd["census_division_name"] = clean_text(qcd["CDNAME"])
qcd["census_division_type"] = clean_text(qcd["CDTYPE"])
qcd["province_code"] = clean_text(qcd["PRUID"])
qcd["province_name"] = PROVINCE_NAME_QUEBEC
qcd["census_year"] = CENSUS_YEAR

qcd["land_area_km2_boundary"] = pd.to_numeric(
    qcd["LANDAREA"],
    errors="coerce",
)

qcd = qcd[
    [
        "census_division_code",
        "census_division_dguid",
        "census_division_name",
        "census_division_type",
        "province_code",
        "province_name",
        "census_year",
        "land_area_km2_boundary",
        "geometry",
    ]
].copy()


# -----------------------------
# Join boundary and population table
# -----------------------------

clean = qcd.merge(
    pop,
    left_on="census_division_dguid",
    right_on="population_dguid",
    how="left",
    validate="one_to_one",
)

clean["matched_population_table"] = clean["population_total_2021"].notna()

unmatched = clean[~clean["matched_population_table"]]

if not unmatched.empty:
    raise ValueError(
        "Some Quebec census divisions did not match the population table:\n"
        + unmatched[
            [
                "census_division_code",
                "census_division_dguid",
                "census_division_name",
                "census_division_type",
            ]
        ].to_string(index=False)
    )


# -----------------------------
# Derived flags and metadata
# -----------------------------

clean["has_positive_population"] = clean["population_total_2021"] > 0

clean["land_area_difference_boundary_minus_population_table_km2"] = (
    clean["land_area_km2_boundary"]
    - clean["land_area_km2_population_table_2021"]
)

clean["source_boundary"] = SOURCE_BOUNDARY
clean["source_population"] = SOURCE_POPULATION
clean["population_table_id"] = "98-10-0007-01"
clean["geography_level"] = "census_division"

# Keep a convenient canonical land_area_km2 field.
# The boundary file is used as the spatial source, but the population table
# also provides land area. We keep both for audit.
clean["land_area_km2"] = clean["land_area_km2_boundary"]


# -----------------------------
# Final column order
# -----------------------------

preferred_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "census_division_type",
    "province_code",
    "province_name",
    "geography_level",
    "census_year",
    "population_geo_name",
    "population_total_2021",
    "population_total_2016",
    "population_change_pct_2016_2021",
    "total_private_dwellings_2021",
    "total_private_dwellings_2016",
    "total_private_dwellings_change_pct_2016_2021",
    "private_dwellings_occupied_by_usual_residents_2021",
    "private_dwellings_occupied_by_usual_residents_2016",
    "private_dwellings_occupied_by_usual_residents_change_pct_2016_2021",
    "land_area_km2",
    "land_area_km2_boundary",
    "land_area_km2_population_table_2021",
    "land_area_difference_boundary_minus_population_table_km2",
    "population_density_per_km2_2021",
    "national_population_rank_2021",
    "province_population_rank_2021",
    "has_positive_population",
    "matched_population_table",
    "source_boundary",
    "source_population",
    "population_table_id",
    "geometry",
]

existing_preferred_cols = [col for col in preferred_cols if col in clean.columns]
remaining_cols = [col for col in clean.columns if col not in existing_preferred_cols]

clean = clean[existing_preferred_cols + remaining_cols].copy()


# -----------------------------
# Final validation
# -----------------------------

if len(clean) != 98:
    raise ValueError(f"Expected 98 Quebec census divisions, got {len(clean)}.")

if clean["census_division_code"].duplicated().any():
    duplicated = clean[clean["census_division_code"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated census_division_code values:\n"
        + duplicated[
            [
                "census_division_code",
                "census_division_dguid",
                "census_division_name",
            ]
        ].to_string(index=False)
    )

critical_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "population_total_2021",
    "population_total_2016",
    "land_area_km2",
    "population_density_per_km2_2021",
    "geometry",
]

for col in critical_cols:
    missing_mask = clean[col].isna()
    if missing_mask.any():
        raise ValueError(
            f"Missing values found in critical column {col}:\n"
            + clean.loc[
                missing_mask,
                [
                    "census_division_code",
                    "census_division_dguid",
                    "census_division_name",
                    col,
                ],
            ].to_string(index=False)
        )

if not clean["has_positive_population"].all():
    non_positive = clean[~clean["has_positive_population"]]
    raise ValueError(
        "Some census divisions have non-positive population:\n"
        + non_positive[
            [
                "census_division_code",
                "census_division_name",
                "population_total_2021",
            ]
        ].to_string(index=False)
    )


# -----------------------------
# Save outputs
# -----------------------------

# Save CSV with geometry as WKT.
csv_out = clean.copy()
csv_out["geometry_wkt"] = csv_out.geometry.to_wkt()
csv_out = pd.DataFrame(csv_out.drop(columns="geometry"))
csv_out.to_csv(OUTPUT_CSV, index=False)

# Save spatial formats.
clean.to_parquet(OUTPUT_PARQUET, index=False)
clean.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
clean.to_file(OUTPUT_GPKG, layer="quebec_census_divisions_2021", driver="GPKG")


# -----------------------------
# Diagnostics
# -----------------------------

print("\nFinal reusable Quebec census-division spatial frame")
print("Rows:", len(clean))
print("CRS:", clean.crs)

print("\nPopulation summary:")
print(clean["population_total_2021"].describe().to_string())

print("\nPopulation totals:")
print("Quebec total population across CDs:", int(clean["population_total_2021"].sum()))
print("Quebec total population 2016 across CDs:", int(clean["population_total_2016"].sum()))

print("\nTop 10 census divisions by 2021 population:")
print(
    clean[
        [
            "census_division_code",
            "census_division_name",
            "population_total_2021",
            "population_density_per_km2_2021",
        ]
    ]
    .sort_values("population_total_2021", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\nSmallest 10 census divisions by 2021 population:")
print(
    clean[
        [
            "census_division_code",
            "census_division_name",
            "population_total_2021",
            "population_density_per_km2_2021",
        ]
    ]
    .sort_values("population_total_2021", ascending=True)
    .head(10)
    .to_string(index=False)
)

print("\nLand-area difference check:")
print(
    clean[
        [
            "census_division_code",
            "census_division_name",
            "land_area_km2_boundary",
            "land_area_km2_population_table_2021",
            "land_area_difference_boundary_minus_population_table_km2",
        ]
    ]
    .assign(
        abs_diff=lambda df: df[
            "land_area_difference_boundary_minus_population_table_km2"
        ].abs()
    )
    .sort_values("abs_diff", ascending=False)
    .head(10)
    .drop(columns="abs_diff")
    .to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_PARQUET)
print(OUTPUT_GEOJSON)
print(OUTPUT_GPKG)

print("\nDone.")