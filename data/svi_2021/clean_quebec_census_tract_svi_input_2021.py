from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely import wkb, wkt
from shapely.geometry.base import BaseGeometry


# ============================================================
# Clean Québec Census Tract SVI Input Table 2021
# ============================================================
#
# Purpose:
#   Build the canonical census-tract-level input table for the SVI-like
#   social vulnerability index.
#
# This script DOES NOT compute SVI scores.
# It only creates a clean wide feature table with one row per Québec census
# tract and the canonical SVI variable names expected by the SVI implementation.
#
# Missing values are preserved.
# Missing canonical variables are included as NA columns.
#
# Canonical SVI variables:
#
#   Socioeconomic status:
#       pct_below_poverty
#       pct_unemployed
#       per_capita_income
#       pct_no_high_school
#
#   Household composition / disability:
#       pct_age_65_plus
#       pct_age_17_or_younger
#       pct_disability
#       pct_single_parent_households
#
#   Minority status / language:
#       pct_minority
#       pct_limited_language
#
#   Housing / transportation:
#       pct_multiunit_structures
#       pct_mobile_homes
#       pct_crowding
#       pct_no_vehicle
#       pct_group_quarters
#
# Important:
#   - pct_disability is currently missing.
#   - pct_group_quarters is currently missing.
#   - pct_no_vehicle is currently missing as an exact SVI variable.
#     A weak commuting-based proxy candidate is preserved separately as:
#       pct_no_vehicle_weak_proxy_candidate
#
# Run from data/:
#   python svi_2021/clean_quebec_census_tract_svi_input_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = DATA_DIR / "svi_2021" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_FRAME_CANDIDATES = [
    DATA_DIR
    / "spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_tract_spatial_frame_with_population_2021.parquet",
    DATA_DIR
    / "spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_tract_spatial_frame_with_population_2021.geojson",
    DATA_DIR
    / "spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_tract_spatial_frame_with_population_2021.gpkg",
    DATA_DIR
    / "spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_tract_spatial_frame_with_population_2021.csv",
]

OUTPUT_CSV = OUTPUT_DIR / "clean_quebec_census_tract_svi_input_2021.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_quebec_census_tract_svi_input_2021.parquet"
OUTPUT_GEOJSON = OUTPUT_DIR / "clean_quebec_census_tract_svi_input_2021.geojson"
OUTPUT_GPKG = OUTPUT_DIR / "clean_quebec_census_tract_svi_input_2021.gpkg"

OUTPUT_VARIABLE_METADATA = OUTPUT_DIR / "clean_quebec_census_tract_svi_input_variable_metadata_2021.csv"
OUTPUT_MISSINGNESS_REPORT = OUTPUT_DIR / "clean_quebec_census_tract_svi_input_missingness_report_2021.csv"
OUTPUT_JOIN_REPORT = OUTPUT_DIR / "clean_quebec_census_tract_svi_input_join_report_2021.csv"


# -----------------------------
# Constants
# -----------------------------

CENSUS_YEAR = 2021
REPRODUCTION_LEVEL_RECOMMENDED = "local_adaptation_or_partial_svi_like"

SOURCE_SVI_INPUT = (
    "Built from cleaned 2021 Quebec census-tract feature tables for SVI-like "
    "local adaptation."
)

SVI_CANONICAL_VARIABLES = [
    "pct_below_poverty",
    "pct_unemployed",
    "per_capita_income",
    "pct_no_high_school",
    "pct_age_65_plus",
    "pct_age_17_or_younger",
    "pct_disability",
    "pct_single_parent_households",
    "pct_minority",
    "pct_limited_language",
    "pct_multiunit_structures",
    "pct_mobile_homes",
    "pct_crowding",
    "pct_no_vehicle",
    "pct_group_quarters",
]

SVI_DOMAINS = {
    "socioeconomic_status": [
        "pct_below_poverty",
        "pct_unemployed",
        "per_capita_income",
        "pct_no_high_school",
    ],
    "household_composition_disability": [
        "pct_age_65_plus",
        "pct_age_17_or_younger",
        "pct_disability",
        "pct_single_parent_households",
    ],
    "minority_status_language": [
        "pct_minority",
        "pct_limited_language",
    ],
    "housing_transportation": [
        "pct_multiunit_structures",
        "pct_mobile_homes",
        "pct_crowding",
        "pct_no_vehicle",
        "pct_group_quarters",
    ],
}

CANONICAL_DIRECTIONS = {
    "pct_below_poverty": "higher_more_vulnerable",
    "pct_unemployed": "higher_more_vulnerable",
    "per_capita_income": "lower_more_vulnerable",
    "pct_no_high_school": "higher_more_vulnerable",
    "pct_age_65_plus": "higher_more_vulnerable",
    "pct_age_17_or_younger": "higher_more_vulnerable",
    "pct_disability": "higher_more_vulnerable",
    "pct_single_parent_households": "higher_more_vulnerable",
    "pct_minority": "higher_more_vulnerable",
    "pct_limited_language": "higher_more_vulnerable",
    "pct_multiunit_structures": "higher_more_vulnerable",
    "pct_mobile_homes": "higher_more_vulnerable",
    "pct_crowding": "higher_more_vulnerable",
    "pct_no_vehicle": "higher_more_vulnerable",
    "pct_group_quarters": "higher_more_vulnerable",
}


# -----------------------------
# Feature mapping
# -----------------------------
#
# main_canonical = True means the selected source column is copied into the
# canonical SVI column.
#
# main_canonical = False means the feature is preserved as a side/audit/proxy
# candidate column but is NOT copied into the canonical SVI column.
#
# For missing canonical variables, source_folder/source_file/source_column are None.
#

FEATURE_SPECS = [
    {
        "canonical_variable": "pct_below_poverty",
        "domain": "socioeconomic_status",
        "source_folder": "low_income_2021",
        "source_file": "clean_census_tract_low_income_2021.csv",
        "source_column": "pct_low_income_lim_at",
        "output_column": "pct_below_poverty",
        "main_canonical": True,
        "status": "direct_or_strong_proxy",
        "proxy_used": True,
        "proxy_quality": "high",
        "conceptual_risk": (
            "Canadian low-income measure after tax is used as a poverty/low-income proxy."
        ),
    },
    {
        "canonical_variable": "pct_unemployed",
        "domain": "socioeconomic_status",
        "source_folder": "unemployment_2021",
        "source_file": "clean_census_tract_unemployment_2021.csv",
        "source_column": "pct_unemployed",
        "output_column": "pct_unemployed",
        "main_canonical": True,
        "status": "direct_or_strong_proxy",
        "proxy_used": False,
        "proxy_quality": "high",
        "conceptual_risk": "Closest available unemployment rate.",
    },
    {
        "canonical_variable": "per_capita_income",
        "domain": "socioeconomic_status",
        "source_folder": "income_2021",
        "source_file": "clean_census_tract_income_2021.csv",
        "source_column": "income_measure_default",
        "output_column": "per_capita_income",
        "main_canonical": True,
        "status": "local_adaptation_proxy",
        "proxy_used": True,
        "proxy_quality": "medium",
        "conceptual_risk": (
            "Original SVI uses per-capita income. Current table uses the cleaned "
            "Canadian income default proxy. Direction must be reversed in SVI scoring."
        ),
    },
    {
        "canonical_variable": "pct_no_high_school",
        "domain": "socioeconomic_status",
        "source_folder": "education_2021",
        "source_file": "clean_census_tract_education_2021.csv",
        "source_column": "education_measure_default",
        "output_column": "pct_no_high_school",
        "main_canonical": True,
        "status": "local_adaptation_proxy",
        "proxy_used": True,
        "proxy_quality": "medium_high",
        "conceptual_risk": (
            "Canadian no certificate/diploma/degree is used as a proxy for no high school diploma."
        ),
    },
    {
        "canonical_variable": "pct_age_65_plus",
        "domain": "household_composition_disability",
        "source_folder": "age_structure_2021",
        "source_file": "clean_census_tract_age_structure_2021.csv",
        "source_column": "pct_age_65_plus",
        "output_column": "pct_age_65_plus",
        "main_canonical": True,
        "status": "direct_or_strong_proxy",
        "proxy_used": False,
        "proxy_quality": "high",
        "conceptual_risk": "Direct age 65+ measure.",
    },
    {
        "canonical_variable": "pct_age_17_or_younger",
        "domain": "household_composition_disability",
        "source_folder": "age_structure_2021",
        "source_file": "clean_census_tract_age_structure_2021.csv",
        "source_column": "pct_age_0_14",
        "output_column": "pct_age_17_or_younger",
        "main_canonical": True,
        "status": "local_adaptation_proxy",
        "proxy_used": True,
        "proxy_quality": "medium",
        "conceptual_risk": (
            "Original SVI uses persons age 17 or younger. Current available proxy is age 0-14."
        ),
    },
    {
        "canonical_variable": "pct_disability",
        "domain": "household_composition_disability",
        "source_folder": None,
        "source_file": None,
        "source_column": None,
        "output_column": "pct_disability",
        "main_canonical": True,
        "status": "missing_no_current_proxy",
        "proxy_used": False,
        "proxy_quality": "missing",
        "conceptual_risk": (
            "No acceptable current census-tract disability/activity-limitation variable has been cleaned."
        ),
    },
    {
        "canonical_variable": "pct_single_parent_households",
        "domain": "household_composition_disability",
        "source_folder": "household_family_2021",
        "source_file": "clean_census_tract_household_family_2021.csv",
        "source_column": "single_parent_measure_default",
        "output_column": "pct_single_parent_households",
        "main_canonical": True,
        "status": "local_adaptation_proxy",
        "proxy_used": True,
        "proxy_quality": "medium",
        "conceptual_risk": (
            "Proxy for single-parent households with children under 18. Current default "
            "may not be restricted exactly to children under 18."
        ),
    },
    {
        "canonical_variable": "pct_minority",
        "domain": "minority_status_language",
        "source_folder": "immigration_ethnocultural_2021",
        "source_file": "clean_census_tract_immigration_ethnocultural_2021.csv",
        "source_column": "ethnocultural_measure_default",
        "output_column": "pct_minority",
        "main_canonical": True,
        "status": "local_adaptation_proxy",
        "proxy_used": True,
        "proxy_quality": "medium",
        "conceptual_risk": (
            "Canadian/Quebec local-adaptation proxy for the U.S. minority-status variable."
        ),
    },
    {
        "canonical_variable": "pct_limited_language",
        "domain": "minority_status_language",
        "source_folder": "language_2021",
        "source_file": "clean_census_tract_language_2021.csv",
        "source_column": "language_barrier_measure_default",
        "output_column": "pct_limited_language",
        "main_canonical": True,
        "status": "local_adaptation_proxy",
        "proxy_used": True,
        "proxy_quality": "medium",
        "conceptual_risk": (
            "Canadian/Quebec language-barrier proxy for English less than well."
        ),
    },
    {
        "canonical_variable": "pct_multiunit_structures",
        "domain": "housing_transportation",
        "source_folder": "housing_type_2021",
        "source_file": "clean_census_tract_housing_type_2021.csv",
        "source_column": "multiunit_measure_default",
        "output_column": "pct_multiunit_structures",
        "main_canonical": True,
        "status": "direct_or_strong_proxy",
        "proxy_used": True,
        "proxy_quality": "high",
        "conceptual_risk": "Strong Canadian housing-type proxy for multi-unit structures.",
    },
    {
        "canonical_variable": "pct_mobile_homes",
        "domain": "housing_transportation",
        "source_folder": "housing_type_2021",
        "source_file": "clean_census_tract_housing_type_2021.csv",
        "source_column": "mobile_home_measure_default",
        "output_column": "pct_mobile_homes",
        "main_canonical": True,
        "status": "direct_or_strong_proxy",
        "proxy_used": True,
        "proxy_quality": "high",
        "conceptual_risk": "Canadian movable-dwelling proxy for mobile homes.",
    },
    {
        "canonical_variable": "pct_crowding",
        "domain": "housing_transportation",
        "source_folder": "housing_suitability_crowding_2021",
        "source_file": "clean_census_tract_housing_suitability_crowding_2021.csv",
        "source_column": "crowding_measure_default",
        "output_column": "pct_crowding",
        "main_canonical": True,
        "status": "direct_or_strong_proxy",
        "proxy_used": True,
        "proxy_quality": "high",
        "conceptual_risk": (
            "More than one person per room is used as the crowding measure."
        ),
    },
    {
        "canonical_variable": "pct_no_vehicle",
        "domain": "housing_transportation",
        "source_folder": None,
        "source_file": None,
        "source_column": None,
        "output_column": "pct_no_vehicle",
        "main_canonical": True,
        "status": "missing_exact_weak_proxy_available",
        "proxy_used": False,
        "proxy_quality": "missing_exact",
        "conceptual_risk": (
            "No exact household no-vehicle variable has been cleaned. "
            "A weak commuting-mode proxy candidate is preserved separately."
        ),
    },
    {
        "canonical_variable": "pct_no_vehicle",
        "domain": "housing_transportation",
        "source_folder": "commuting_transport_2021",
        "source_file": "clean_census_tract_commuting_transport_2021.csv",
        "source_column": "pct_commute_non_car_modes",
        "output_column": "pct_no_vehicle_weak_proxy_candidate",
        "main_canonical": False,
        "status": "weak_proxy_candidate_only",
        "proxy_used": True,
        "proxy_quality": "low",
        "conceptual_risk": (
            "Commuting non-car modes are not household vehicle availability. "
            "Preserved for sensitivity analysis only."
        ),
    },
    {
        "canonical_variable": "pct_group_quarters",
        "domain": "housing_transportation",
        "source_folder": None,
        "source_file": None,
        "source_column": None,
        "output_column": "pct_group_quarters",
        "main_canonical": True,
        "status": "missing_no_current_proxy",
        "proxy_used": False,
        "proxy_quality": "missing",
        "conceptual_risk": (
            "Need census-tract population in collective dwellings/group quarters. "
            "ODHF facility counts are not an acceptable proxy."
        ),
    },
]


# -----------------------------
# Helpers
# -----------------------------

def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, low_memory=False)

    if suffix == ".parquet":
        # Prefer GeoPandas so GeoParquet metadata is respected when present.
        # If the file is a plain Parquet table with WKB geometry bytes,
        # fall back to pandas and decode the geometry column below.
        try:
            return gpd.read_parquet(path)
        except Exception:
            return pd.read_parquet(path)

    if suffix in [".geojson", ".gpkg", ".shp"]:
        return gpd.read_file(path)

    raise ValueError(f"Unsupported file type: {path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def ensure_geometry_objects(df: pd.DataFrame, geometry_col: str = "geometry") -> pd.DataFrame:
    """Decode geometry values if Parquet returned WKB bytes instead of Shapely objects."""
    if geometry_col not in df.columns:
        return df

    non_null_geometry = df[geometry_col].dropna()

    if non_null_geometry.empty:
        return df

    sample = non_null_geometry.iloc[0]

    if isinstance(sample, BaseGeometry):
        return df

    out = df.copy()

    if isinstance(sample, (bytes, bytearray, memoryview)):
        out[geometry_col] = out[geometry_col].apply(
            lambda geom: wkb.loads(bytes(geom))
            if isinstance(geom, (bytes, bytearray, memoryview))
            else geom
        )
        return out

    if isinstance(sample, str):
        out[geometry_col] = out[geometry_col].apply(
            lambda geom: wkt.loads(geom)
            if isinstance(geom, str) and geom.strip()
            else geom
        )
        return out

    return out


def find_existing_base_frame() -> Path:
    for path in BASE_FRAME_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find the census-tract spatial/population frame. "
        "Expected one of:\n"
        + "\n".join(str(path) for path in BASE_FRAME_CANDIDATES)
    )


def require_columns(df: pd.DataFrame, required_cols: list[str], label: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns in {label}:\n"
            + "\n".join(missing)
        )


def find_source_path(source_folder: str, source_file: str) -> Path:
    path = DATA_DIR / source_folder / "output" / source_file

    if not path.exists():
        raise FileNotFoundError(
            f"Expected source file not found:\n{path}"
        )

    return path


def to_numeric_nullable(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_domain_status(row: pd.Series, domain_variables: list[str]) -> dict:
    values = row[domain_variables]

    return {
        "available_count": int(values.notna().sum()),
        "missing_count": int(values.isna().sum()),
        "complete": bool(values.notna().all()),
    }


def source_label(spec: dict) -> str:
    if spec["source_folder"] is None:
        return ""

    return f'{spec["source_folder"]}/output/{spec["source_file"]}::{spec["source_column"]}'


# -----------------------------
# Load base frame
# -----------------------------

base_path = find_existing_base_frame()
base = read_table(base_path)
base = normalize_columns(base)

print("\nLoaded base census-tract spatial/population frame")
print("Path:", base_path.relative_to(DATA_DIR))
print("Rows:", len(base))
print("Columns:", list(base.columns))

require_columns(
    base,
    [
        "statcan_dguid",
        "unit_id",
        "unit_name",
        "unit_type",
        "census_year",
        "province_id",
        "province_name",
        "land_area_km2",
        "population_total",
        "geometry",
    ],
    "base census-tract spatial/population frame",
)

base = base.copy()
base["statcan_dguid"] = clean_text(base["statcan_dguid"])

if base["statcan_dguid"].duplicated().any():
    duplicated = base[base["statcan_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated statcan_dguid values in base frame:\n"
        + duplicated[["statcan_dguid", "unit_name"]].head(30).to_string(index=False)
    )

base["population_total"] = pd.to_numeric(base["population_total"], errors="coerce")
base["land_area_km2"] = pd.to_numeric(base["land_area_km2"], errors="coerce")
base["has_positive_population"] = base["population_total"] > 0

# Plain Parquet files may store geometry as WKB bytes.
# GeoPandas needs actual Shapely geometry objects before building a GeoDataFrame.
base = ensure_geometry_objects(base, geometry_col="geometry")

# Build initial GeoDataFrame.
clean = gpd.GeoDataFrame(base.copy(), geometry="geometry", crs=getattr(base, "crs", None))

# If reading parquet through pandas preserved geometry but not CRS, try to set CRS
# only when missing. The known StatCan census-tract frame uses EPSG:3347.
if clean.crs is None:
    try:
        clean = clean.set_crs("EPSG:3347", allow_override=True)
    except Exception:
        pass


# -----------------------------
# Rename base columns for SVI runner convenience
# -----------------------------

clean["zone_id"] = clean["statcan_dguid"]
clean["population"] = clean["population_total"]
clean["svi_input_year"] = CENSUS_YEAR
clean["svi_input_source"] = SOURCE_SVI_INPUT
clean["recommended_reproduction_level"] = REPRODUCTION_LEVEL_RECOMMENDED


# -----------------------------
# Load and join feature sources
# -----------------------------

join_report_rows = []
source_cache = {}

print("\nJoining SVI feature sources...")

for spec in FEATURE_SPECS:
    output_col = spec["output_column"]
    canonical = spec["canonical_variable"]

    # Missing placeholder column.
    if spec["source_folder"] is None:
        clean[output_col] = pd.NA

        join_report_rows.append(
            {
                "canonical_variable": canonical,
                "output_column": output_col,
                "source_folder": "",
                "source_file": "",
                "source_column": "",
                "status": spec["status"],
                "main_canonical": spec["main_canonical"],
                "source_rows": 0,
                "matched_rows": 0,
                "unmatched_rows": len(clean),
                "missing_after_join": int(clean[output_col].isna().sum()),
                "join_success": False,
                "note": "Missing placeholder column created.",
            }
        )

        print(f"- {output_col}: created missing placeholder")
        continue

    cache_key = (spec["source_folder"], spec["source_file"])

    if cache_key not in source_cache:
        source_path = find_source_path(spec["source_folder"], spec["source_file"])
        source_df = read_table(source_path)
        source_df = normalize_columns(source_df)

        require_columns(
            source_df,
            ["statcan_dguid"],
            f"{spec['source_folder']}/{spec['source_file']}",
        )

        source_df = source_df.copy()
        source_df["statcan_dguid"] = clean_text(source_df["statcan_dguid"])

        if source_df["statcan_dguid"].duplicated().any():
            duplicated = source_df[source_df["statcan_dguid"].duplicated(keep=False)]
            raise ValueError(
                f"Duplicated statcan_dguid values in source file {source_path}:\n"
                + duplicated[["statcan_dguid"]].head(30).to_string(index=False)
            )

        source_cache[cache_key] = source_df

    source = source_cache[cache_key]

    require_columns(
        source,
        [spec["source_column"]],
        f"{spec['source_folder']}/{spec['source_file']}",
    )

    source_subset = source[["statcan_dguid", spec["source_column"]]].copy()
    source_subset = source_subset.rename(columns={spec["source_column"]: output_col})

    before_rows = len(clean)

    clean = clean.merge(
        source_subset,
        on="statcan_dguid",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    matched_rows = int((clean["_merge"] == "both").sum())
    unmatched_rows = int((clean["_merge"] == "left_only").sum())

    clean = clean.drop(columns="_merge")

    clean[output_col] = to_numeric_nullable(clean[output_col])

    if len(clean) != before_rows:
        raise ValueError(
            f"Row count changed after joining {output_col}: "
            f"{before_rows} -> {len(clean)}"
        )

    join_report_rows.append(
        {
            "canonical_variable": canonical,
            "output_column": output_col,
            "source_folder": spec["source_folder"],
            "source_file": spec["source_file"],
            "source_column": spec["source_column"],
            "status": spec["status"],
            "main_canonical": spec["main_canonical"],
            "source_rows": len(source),
            "matched_rows": matched_rows,
            "unmatched_rows": unmatched_rows,
            "missing_after_join": int(clean[output_col].isna().sum()),
            "join_success": unmatched_rows == 0,
            "note": spec["conceptual_risk"],
        }
    )

    print(
        f"- {output_col}: joined from {spec['source_folder']}::{spec['source_column']} "
        f"matched={matched_rows}, unmatched={unmatched_rows}, "
        f"missing={int(clean[output_col].isna().sum())}"
    )


# -----------------------------
# Add canonical variable metadata fields
# -----------------------------

metadata_rows = []

for spec in FEATURE_SPECS:
    if not spec["main_canonical"]:
        continue

    canonical = spec["canonical_variable"]
    output_col = spec["output_column"]

    metadata_rows.append(
        {
            "canonical_variable": canonical,
            "domain": spec["domain"],
            "output_column": output_col,
            "direction": CANONICAL_DIRECTIONS[canonical],
            "status": spec["status"],
            "proxy_used": spec["proxy_used"],
            "proxy_quality": spec["proxy_quality"],
            "source_folder": spec["source_folder"] or "",
            "source_file": spec["source_file"] or "",
            "source_column": spec["source_column"] or "",
            "source_label": source_label(spec),
            "conceptual_risk": spec["conceptual_risk"],
        }
    )

    clean[f"{canonical}__status"] = spec["status"]
    clean[f"{canonical}__proxy_used"] = spec["proxy_used"]
    clean[f"{canonical}__proxy_quality"] = spec["proxy_quality"]
    clean[f"{canonical}__source"] = source_label(spec)
    clean[f"{canonical}__conceptual_risk"] = spec["conceptual_risk"]


metadata = pd.DataFrame(metadata_rows)


# -----------------------------
# Add missingness and domain diagnostics
# -----------------------------

for var in SVI_CANONICAL_VARIABLES:
    clean[f"{var}__is_missing"] = clean[var].isna()

clean["svi_available_variable_count"] = clean[SVI_CANONICAL_VARIABLES].notna().sum(axis=1)
clean["svi_missing_variable_count"] = clean[SVI_CANONICAL_VARIABLES].isna().sum(axis=1)

for domain_name, domain_vars in SVI_DOMAINS.items():
    clean[f"{domain_name}__available_count"] = clean[domain_vars].notna().sum(axis=1)
    clean[f"{domain_name}__missing_count"] = clean[domain_vars].isna().sum(axis=1)
    clean[f"{domain_name}__complete"] = clean[domain_vars].notna().all(axis=1)

clean["svi_has_all_15_canonical_variables"] = clean[SVI_CANONICAL_VARIABLES].notna().all(axis=1)

# Available main variables excluding the known missing exact placeholders.
ready_main_variables = [
    spec["canonical_variable"]
    for spec in FEATURE_SPECS
    if spec["main_canonical"]
    and spec["status"] in ["direct_or_strong_proxy", "local_adaptation_proxy"]
]

clean["svi_available_ready_variable_count"] = clean[ready_main_variables].notna().sum(axis=1)
clean["svi_missing_ready_variable_count"] = clean[ready_main_variables].isna().sum(axis=1)


# -----------------------------
# Build missingness report
# -----------------------------

missingness_rows = []

for var in SVI_CANONICAL_VARIABLES:
    positive_mask = clean["has_positive_population"] == True

    missingness_rows.append(
        {
            "variable": var,
            "domain": metadata.loc[
                metadata["canonical_variable"] == var,
                "domain",
            ].iloc[0],
            "status": metadata.loc[
                metadata["canonical_variable"] == var,
                "status",
            ].iloc[0],
            "proxy_quality": metadata.loc[
                metadata["canonical_variable"] == var,
                "proxy_quality",
            ].iloc[0],
            "total_rows": len(clean),
            "missing_all_rows": int(clean[var].isna().sum()),
            "non_missing_all_rows": int(clean[var].notna().sum()),
            "positive_population_rows": int(positive_mask.sum()),
            "missing_positive_population_rows": int(clean.loc[positive_mask, var].isna().sum()),
            "non_missing_positive_population_rows": int(clean.loc[positive_mask, var].notna().sum()),
            "min": pd.to_numeric(clean[var], errors="coerce").min(skipna=True),
            "max": pd.to_numeric(clean[var], errors="coerce").max(skipna=True),
            "mean": pd.to_numeric(clean[var], errors="coerce").mean(skipna=True),
            "median": pd.to_numeric(clean[var], errors="coerce").median(skipna=True),
        }
    )

# Add weak proxy candidate to missingness report if present.
if "pct_no_vehicle_weak_proxy_candidate" in clean.columns:
    positive_mask = clean["has_positive_population"] == True
    candidate = "pct_no_vehicle_weak_proxy_candidate"

    missingness_rows.append(
        {
            "variable": candidate,
            "domain": "housing_transportation",
            "status": "weak_proxy_candidate_only",
            "proxy_quality": "low",
            "total_rows": len(clean),
            "missing_all_rows": int(clean[candidate].isna().sum()),
            "non_missing_all_rows": int(clean[candidate].notna().sum()),
            "positive_population_rows": int(positive_mask.sum()),
            "missing_positive_population_rows": int(clean.loc[positive_mask, candidate].isna().sum()),
            "non_missing_positive_population_rows": int(clean.loc[positive_mask, candidate].notna().sum()),
            "min": pd.to_numeric(clean[candidate], errors="coerce").min(skipna=True),
            "max": pd.to_numeric(clean[candidate], errors="coerce").max(skipna=True),
            "mean": pd.to_numeric(clean[candidate], errors="coerce").mean(skipna=True),
            "median": pd.to_numeric(clean[candidate], errors="coerce").median(skipna=True),
        }
    )

missingness = pd.DataFrame(missingness_rows)
join_report = pd.DataFrame(join_report_rows)


# -----------------------------
# Final column order
# -----------------------------

identity_cols = [
    "zone_id",
    "statcan_dguid",
    "unit_id",
    "unit_name",
    "unit_type",
    "census_year",
    "svi_input_year",
    "province_id",
    "province_name",
    "population",
    "population_total",
    "has_positive_population",
    "land_area_km2",
    "source_boundary",
    "source_population",
    "svi_input_source",
    "recommended_reproduction_level",
]

canonical_cols = SVI_CANONICAL_VARIABLES

weak_proxy_cols = [
    "pct_no_vehicle_weak_proxy_candidate",
]

domain_diagnostic_cols = [
    "svi_available_variable_count",
    "svi_missing_variable_count",
    "svi_available_ready_variable_count",
    "svi_missing_ready_variable_count",
    "svi_has_all_15_canonical_variables",
    "socioeconomic_status__available_count",
    "socioeconomic_status__missing_count",
    "socioeconomic_status__complete",
    "household_composition_disability__available_count",
    "household_composition_disability__missing_count",
    "household_composition_disability__complete",
    "minority_status_language__available_count",
    "minority_status_language__missing_count",
    "minority_status_language__complete",
    "housing_transportation__available_count",
    "housing_transportation__missing_count",
    "housing_transportation__complete",
]

missing_flag_cols = [f"{var}__is_missing" for var in SVI_CANONICAL_VARIABLES]

metadata_cols = []
for var in SVI_CANONICAL_VARIABLES:
    metadata_cols.extend(
        [
            f"{var}__status",
            f"{var}__proxy_used",
            f"{var}__proxy_quality",
            f"{var}__source",
            f"{var}__conceptual_risk",
        ]
    )

geometry_cols = ["geometry"]

preferred_cols = (
    identity_cols
    + canonical_cols
    + weak_proxy_cols
    + domain_diagnostic_cols
    + missing_flag_cols
    + metadata_cols
    + geometry_cols
)

existing_preferred_cols = [col for col in preferred_cols if col in clean.columns]
remaining_cols = [col for col in clean.columns if col not in existing_preferred_cols]

clean = clean[existing_preferred_cols + remaining_cols].copy()


# -----------------------------
# Final validation
# -----------------------------

if len(clean) != len(base):
    raise ValueError(
        f"Final row count does not match base frame: {len(clean)} vs {len(base)}"
    )

if clean["zone_id"].duplicated().any():
    duplicated = clean[clean["zone_id"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated zone_id values in final table:\n"
        + duplicated[["zone_id", "unit_name"]].head(30).to_string(index=False)
    )

for var in SVI_CANONICAL_VARIABLES:
    if var not in clean.columns:
        raise ValueError(f"Missing canonical SVI column in final table: {var}")

# Do not fail on missing values. Missingness is expected and documented.
# Do not fail on all-NA canonical columns. Those are intentional placeholders.


# -----------------------------
# Save outputs
# -----------------------------

# CSV with WKT geometry.
csv_out = clean.copy()
csv_out["geometry_wkt"] = csv_out.geometry.to_wkt()
csv_out = pd.DataFrame(csv_out.drop(columns="geometry"))
csv_out.to_csv(OUTPUT_CSV, index=False)

clean.to_parquet(OUTPUT_PARQUET, index=False)
clean.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
clean.to_file(OUTPUT_GPKG, layer="quebec_census_tract_svi_input_2021", driver="GPKG")

metadata.to_csv(OUTPUT_VARIABLE_METADATA, index=False)
missingness.to_csv(OUTPUT_MISSINGNESS_REPORT, index=False)
join_report.to_csv(OUTPUT_JOIN_REPORT, index=False)


# -----------------------------
# Console diagnostics
# -----------------------------

print("\n" + "=" * 72)
print("CLEAN QUÉBEC CENSUS TRACT SVI INPUT TABLE")
print("=" * 72)

print("\nFinal table:")
print("Rows:", len(clean))
print("CRS:", clean.crs)
print("Columns:", len(clean.columns))

print("\nPopulation:")
print("Total rows:", len(clean))
print("Positive-population rows:", int(clean["has_positive_population"].sum()))
print("Zero/non-positive/unknown-population rows:", int((~clean["has_positive_population"]).sum()))

print("\nCanonical SVI variables:")
print("Canonical variables included:", len(SVI_CANONICAL_VARIABLES))
print("Ready/main variable columns:", len(ready_main_variables))
print("Missing/exact-placeholder variables:", [
    var
    for var in SVI_CANONICAL_VARIABLES
    if clean[var].isna().all()
])

print("\nVariable metadata summary:")
print(metadata["status"].value_counts(dropna=False).to_string())

print("\nMissingness report:")
print(
    missingness[
        [
            "variable",
            "status",
            "proxy_quality",
            "missing_all_rows",
            "missing_positive_population_rows",
            "non_missing_positive_population_rows",
            "mean",
            "median",
        ]
    ].to_string(index=False)
)

print("\nJoin report:")
print(
    join_report[
        [
            "output_column",
            "status",
            "main_canonical",
            "matched_rows",
            "unmatched_rows",
            "missing_after_join",
            "join_success",
        ]
    ].to_string(index=False)
)

print("\nImportant notes:")
print("- This script does not compute SVI scores.")
print("- Missing values are intentionally preserved.")
print("- pct_disability and pct_group_quarters are all-NA placeholders.")
print("- pct_no_vehicle is all-NA as an exact canonical variable.")
print("- pct_no_vehicle_weak_proxy_candidate is preserved separately but is not mapped")
print("  into pct_no_vehicle by this cleaner.")
print("- The SVI scoring/recipe layer should decide how to handle missing values and")
print("  whether to use weak proxy candidates.")

print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_PARQUET)
print(OUTPUT_GEOJSON)
print(OUTPUT_GPKG)
print(OUTPUT_VARIABLE_METADATA)
print(OUTPUT_MISSINGNESS_REPORT)
print(OUTPUT_JOIN_REPORT)

print("\nDone.")