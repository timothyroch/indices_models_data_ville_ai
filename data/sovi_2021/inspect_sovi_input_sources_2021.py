from pathlib import Path
import pandas as pd

try:
    import geopandas as gpd
except ImportError:
    gpd = None


# ============================================================
# Inspect SoVI Input Sources 2021
# ============================================================
#
# Purpose:
#   Inspect which original SoVI-inspired variables we can currently populate
#   for a Québec census-division-level SoVI-like input table.
#
# This script does NOT build the final SoVI table.
# It checks:
#   1. which candidate files exist;
#   2. which candidate source columns exist;
#   3. whether the source is already census-division-level;
#   4. whether it joins directly to the CD base frame;
#   5. whether it is only available at census-tract level;
#   6. whether it requires a custom cleaner/crosswalk;
#   7. which variables are currently missing/placeholders.
#
# Target geography:
#   Québec 2021 census divisions.
#
# Run from data/:
#   python sovi_2021/inspect_sovi_input_sources_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = DATA_DIR / "sovi_2021" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SOURCE_INVENTORY = OUTPUT_DIR / "sovi_input_source_file_inventory_2021.csv"
OUTPUT_VARIABLE_AVAILABILITY = OUTPUT_DIR / "sovi_input_variable_availability_2021.csv"
OUTPUT_CANDIDATE_COLUMN_DIAGNOSTICS = (
    OUTPUT_DIR / "sovi_input_candidate_column_diagnostics_2021.csv"
)
OUTPUT_SUMMARY = OUTPUT_DIR / "sovi_input_availability_summary_2021.csv"


# -----------------------------
# Base census-division frame
# -----------------------------

BASE_FRAME_CANDIDATES = [
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
    / "clean_quebec_census_division_spatial_frame_with_population_2021.parquet",
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.csv",
]


# -----------------------------
# Helpers
# -----------------------------

def p(relative_path: str) -> Path:
    return DATA_DIR / relative_path


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, dtype=str, low_memory=False, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return read_csv_with_fallback(path)

    if suffix == ".parquet":
        if gpd is not None:
            try:
                return gpd.read_parquet(path)
            except Exception:
                pass
        return pd.read_parquet(path)

    if suffix in [".geojson", ".gpkg", ".shp"]:
        if gpd is None:
            raise ImportError(f"geopandas is required to read spatial file: {path}")
        return gpd.read_file(path)

    raise ValueError(f"Unsupported file type: {path}")


def find_existing_base_frame() -> Path:
    for path in BASE_FRAME_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find the census-division spatial/population frame.\n"
        "Expected one of:\n"
        + "\n".join(str(path) for path in BASE_FRAME_CANDIDATES)
    )


def safe_column_list(df: pd.DataFrame, max_chars: int = 1500) -> str:
    text = ", ".join(map(str, df.columns))
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ... [truncated]"


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def numeric_summary(series: pd.Series) -> dict:
    numeric = pd.to_numeric(series, errors="coerce")

    return {
        "numeric_non_missing": int(numeric.notna().sum()),
        "numeric_missing": int(numeric.isna().sum()),
        "min": numeric.min(skipna=True),
        "max": numeric.max(skipna=True),
        "mean": numeric.mean(skipna=True),
        "median": numeric.median(skipna=True),
    }


def infer_join_method(
    source_df: pd.DataFrame,
    base_df: pd.DataFrame,
) -> tuple[str, int, int]:
    """
    Return:
        join_method, matched_rows, unmatched_base_rows
    """
    if (
        "census_division_dguid" in source_df.columns
        and "census_division_dguid" in base_df.columns
    ):
        source_keys = source_df[["census_division_dguid"]].copy()
        source_keys["census_division_dguid"] = clean_text(source_keys["census_division_dguid"])

        base_keys = base_df[["census_division_dguid"]].copy()
        base_keys["census_division_dguid"] = clean_text(base_keys["census_division_dguid"])

        joined = base_keys.merge(
            source_keys.drop_duplicates(),
            on="census_division_dguid",
            how="left",
            indicator=True,
        )

        return (
            "direct_on_census_division_dguid",
            int((joined["_merge"] == "both").sum()),
            int((joined["_merge"] == "left_only").sum()),
        )

    if (
        "census_division_code" in source_df.columns
        and "census_division_code" in base_df.columns
    ):
        source_keys = source_df[["census_division_code"]].copy()
        source_keys["census_division_code"] = clean_text(source_keys["census_division_code"])

        base_keys = base_df[["census_division_code"]].copy()
        base_keys["census_division_code"] = clean_text(base_keys["census_division_code"])

        joined = base_keys.merge(
            source_keys.drop_duplicates(),
            on="census_division_code",
            how="left",
            indicator=True,
        )

        return (
            "direct_on_census_division_code",
            int((joined["_merge"] == "both").sum()),
            int((joined["_merge"] == "left_only").sum()),
        )

    if "statcan_dguid" in source_df.columns:
        return "source_has_statcan_dguid_but_not_cd_key", 0, len(base_df)

    return "no_direct_cd_join_key_found", 0, len(base_df)


def detect_source_spatial_level(df: pd.DataFrame, declared: str | None = None) -> str:
    if declared:
        return declared

    columns = set(df.columns)

    if "census_division_dguid" in columns or "census_division_code" in columns:
        return "census_division"

    if "unit_type" in columns:
        unit_types = set(clean_text(df["unit_type"]).dropna().str.lower().unique())
        if "census_division" in unit_types:
            return "census_division"
        if "census_tract" in unit_types:
            return "census_tract"

    if "statcan_dguid" in columns:
        return "unknown_statcan_geography"

    return "unknown"


# -----------------------------
# SoVI variable specifications
# -----------------------------
#
# status_hint values:
#   ready_direct_cd
#   ready_derived_from_cd_base
#   available_requires_custom_join
#   available_only_ct_needs_cd_cleaner
#   missing_or_not_yet_collected
#   likely_available_needs_column_confirmation
#
# The specs intentionally include candidate columns generously. The inspection
# will tell us what exists locally and what does not.
#

SOVI_SPECS = [
    {
        "original_code": "MED_AGE90",
        "canonical_variable": "median_age",
        "description": "Median age",
        "candidate_sources": [
            {
                "path": p("age_structure_2021/output/clean_census_tract_age_structure_2021.csv"),
                "candidate_columns": ["median_age", "median_age_total", "age_median"],
                "declared_spatial_level": "census_tract",
                "status_hint": "available_only_ct_needs_cd_cleaner",
            }
        ],
    },
    {
        "original_code": "PERCAP89",
        "canonical_variable": "per_capita_income",
        "description": "Per capita income",
        "candidate_sources": [
            {
                "path": p("income_2021/output/clean_census_tract_income_2021.csv"),
                "candidate_columns": [
                    "income_measure_default",
                    "median_after_tax_income_15plus_2020",
                    "average_after_tax_income_15plus_2020",
                    "average_total_income_15plus_2020",
                    "per_capita_income",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "available_only_ct_needs_cd_cleaner",
            }
        ],
    },
    {
        "original_code": "MVALOO90",
        "canonical_variable": "median_owner_occupied_housing_value",
        "description": "Median dollar value of owner-occupied housing",
        "candidate_sources": [
            {
                "path": p("housing_tenure_costs_2021/output/clean_census_tract_housing_tenure_costs_2021.csv"),
                "candidate_columns": [
                    "median_value_owner_occupied_dwelling",
                    "median_owner_occupied_housing_value",
                    "owner_housing_value_measure_default",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "likely_available_needs_column_confirmation",
            }
        ],
    },
    {
        "original_code": "MEDRENT90",
        "canonical_variable": "median_rent",
        "description": "Median rent for renter-occupied housing units",
        "candidate_sources": [
            {
                "path": p("housing_tenure_costs_2021/output/clean_census_tract_housing_tenure_costs_2021.csv"),
                "candidate_columns": [
                    "median_rent",
                    "median_monthly_shelter_costs_rented",
                    "renter_shelter_cost_measure_default",
                    "median_shelter_cost_renter_households",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "likely_available_needs_column_confirmation",
            }
        ],
    },
    {
        "original_code": "PHYSICN90",
        "canonical_variable": "physicians_per_100k_population",
        "description": "Number of physicians per 100,000 population",
        "candidate_sources": [
            {
                "path": p("doctors_per_100khabs/output/clean_census_division_doctors_per_100k_2024.csv"),
                "candidate_columns": [
                    "physicians_per_100k_population",
                    "physicians_per_100k_health_region",
                    "family_medicine_physicians_per_100k",
                ],
                "declared_spatial_level": "census_division",
                "status_hint": "ready_direct_cd",
            },
            {
                "path": p("doctors_per_100khabs/lookup/quebec_census_division_to_health_region_crosswalk_filled.csv"),
                "candidate_columns": ["health_region_name"],
                "declared_spatial_level": "census_division_crosswalk_only",
                "status_hint": "available_requires_custom_join",
            },
        ],
    },
    {
        "original_code": "PCTVOTE92",
        "canonical_variable": "pct_vote_leading_party",
        "description": "Percent vote cast for leading party",
        "candidate_sources": [],
    },
    {
        "original_code": "BRATE90",
        "canonical_variable": "birth_rate_per_1000_population",
        "description": "Birth rate per 1,000 population",
        "candidate_sources": [],
    },
    {
        "original_code": "MIGRA_97",
        "canonical_variable": "net_international_migration",
        "description": "Net international migration",
        "candidate_sources": [],
    },
    {
        "original_code": "PCTFARMS92",
        "canonical_variable": "pct_land_in_farms",
        "description": "Land in farms as percent of total land",
        "candidate_sources": [],
    },
    {
        "original_code": "PCTBLACK90",
        "canonical_variable": "pct_black_or_local_proxy",
        "description": "Percent African American / local ethnocultural proxy",
        "candidate_sources": [
            {
                "path": p("immigration_ethnocultural_2021/output/clean_census_tract_immigration_ethnocultural_2021.csv"),
                "candidate_columns": [
                    "pct_black",
                    "pct_visible_minority_black",
                    "ethnocultural_measure_default",
                    "pct_visible_minority",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "available_only_ct_needs_cd_cleaner",
            }
        ],
    },
    {
        "original_code": "PCTINDIAN90",
        "canonical_variable": "pct_indigenous_or_local_proxy",
        "description": "Percent Native American / Indigenous local proxy",
        "candidate_sources": [
            {
                "path": p("immigration_ethnocultural_2021/output/clean_census_tract_immigration_ethnocultural_2021.csv"),
                "candidate_columns": [
                    "pct_indigenous",
                    "pct_aboriginal_identity",
                    "pct_first_nations_metis_inuit",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "available_only_ct_needs_cd_cleaner",
            }
        ],
    },
    {
        "original_code": "PCTASIAN90",
        "canonical_variable": "pct_asian_or_local_proxy",
        "description": "Percent Asian / local proxy",
        "candidate_sources": [
            {
                "path": p("immigration_ethnocultural_2021/output/clean_census_tract_immigration_ethnocultural_2021.csv"),
                "candidate_columns": [
                    "pct_asian",
                    "pct_visible_minority_asian",
                    "pct_south_asian",
                    "pct_chinese",
                    "pct_filipino",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "available_only_ct_needs_cd_cleaner",
            }
        ],
    },
    {
        "original_code": "PCTHISPANIC90",
        "canonical_variable": "pct_hispanic_or_local_proxy",
        "description": "Percent Hispanic / local proxy",
        "candidate_sources": [
            {
                "path": p("immigration_ethnocultural_2021/output/clean_census_tract_immigration_ethnocultural_2021.csv"),
                "candidate_columns": [
                    "pct_latin_american",
                    "pct_hispanic",
                    "pct_visible_minority_latin_american",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "available_only_ct_needs_cd_cleaner",
            }
        ],
    },
    {
        "original_code": "PCTKIDS90",
        "canonical_variable": "pct_under_5_or_child_proxy",
        "description": "Percent population under five years old",
        "candidate_sources": [
            {
                "path": p("age_structure_2021/output/clean_census_tract_age_structure_2021.csv"),
                "candidate_columns": [
                    "pct_age_0_4",
                    "pct_under_5",
                    "pct_age_0_14",
                    "pct_age_17_or_younger",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "available_only_ct_needs_cd_cleaner",
            }
        ],
    },
    {
        "original_code": "PCTOLD90",
        "canonical_variable": "pct_over_65",
        "description": "Percent population over 65",
        "candidate_sources": [
            {
                "path": p("age_structure_2021/output/clean_census_tract_age_structure_2021.csv"),
                "candidate_columns": ["pct_age_65_plus", "pct_over_65"],
                "declared_spatial_level": "census_tract",
                "status_hint": "available_only_ct_needs_cd_cleaner",
            }
        ],
    },
    {
        "original_code": "PCTVLUN91",
        "canonical_variable": "pct_unemployed",
        "description": "Percent civilian labor force unemployed",
        "candidate_sources": [
            {
                "path": p("unemployment_2021/output/clean_census_tract_unemployment_2021.csv"),
                "candidate_columns": [
                    "pct_unemployed",
                    "unemployment_measure_default",
                    "unemployment_rate",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "available_only_ct_needs_cd_cleaner",
            }
        ],
    },
    {
        "original_code": "AVGPERHH",
        "canonical_variable": "avg_people_per_household",
        "description": "Average number of people per household",
        "candidate_sources": [
            {
                "path": p("household_family_2021/output/clean_census_tract_household_family_2021.csv"),
                "candidate_columns": [
                    "avg_people_per_household",
                    "average_household_size",
                    "persons_per_household",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "likely_available_needs_column_confirmation",
            }
        ],
    },
    {
        "original_code": "PCTHH7589",
        "canonical_variable": "pct_high_income_households",
        "description": "Percent households earning more than threshold",
        "candidate_sources": [
            {
                "path": p("income_2021/output/clean_census_tract_income_2021.csv"),
                "candidate_columns": [
                    "pct_households_income_over_75000",
                    "pct_high_income_households",
                    "household_high_income_measure_default",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "likely_available_needs_column_confirmation",
            }
        ],
    },
    {
        "original_code": "PCTPOV90",
        "canonical_variable": "pct_poverty_or_low_income",
        "description": "Percent living in poverty",
        "candidate_sources": [
            {
                "path": p("low_income_2021/output/clean_census_tract_low_income_2021.csv"),
                "candidate_columns": [
                    "pct_low_income_lim_at",
                    "pct_low_income_lico_at",
                    "published_pct_low_income_lim_at",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "available_only_ct_needs_cd_cleaner",
            }
        ],
    },
    {
        "original_code": "PCTRENTER90",
        "canonical_variable": "pct_renter_occupied",
        "description": "Percent renter-occupied housing units",
        "candidate_sources": [
            {
                "path": p("housing_tenure_costs_2021/output/clean_census_tract_housing_tenure_costs_2021.csv"),
                "candidate_columns": [
                    "pct_renter_households",
                    "pct_renter_occupied",
                    "renter_measure_default",
                    "pct_tenant_households",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "available_only_ct_needs_cd_cleaner",
            }
        ],
    },
    {
        "original_code": "PCTRFRM90",
        "canonical_variable": "pct_rural_farm_population",
        "description": "Percent rural farm population",
        "candidate_sources": [],
    },
    {
        "original_code": "DEBREV92",
        "canonical_variable": "local_government_debt_to_revenue_ratio",
        "description": "Local government debt-to-revenue ratio",
        "candidate_sources": [],
    },
    {
        "original_code": "PCTMOBL90",
        "canonical_variable": "pct_mobile_homes",
        "description": "Percent housing units that are mobile homes",
        "candidate_sources": [
            {
                "path": p("housing_type_2021/output/clean_census_tract_housing_type_2021.csv"),
                "candidate_columns": [
                    "mobile_home_measure_default",
                    "pct_movable_dwelling",
                    "pct_mobile_home",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "available_only_ct_needs_cd_cleaner",
            }
        ],
    },
    {
        "original_code": "PCTNOHS90",
        "canonical_variable": "pct_no_high_school",
        "description": "Percent population age 25+ with no high school diploma",
        "candidate_sources": [
            {
                "path": p("education_2021/output/clean_census_tract_education_2021.csv"),
                "candidate_columns": [
                    "education_measure_default",
                    "pct_no_certificate_15plus",
                    "pct_no_certificate_25_64",
                    "pct_no_high_school",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "available_only_ct_needs_cd_cleaner",
            }
        ],
    },
    {
        "original_code": "HODENUT90",
        "canonical_variable": "housing_units_density",
        "description": "Number of housing units per area",
        "derived_from_base": True,
        "required_base_columns": [
            "total_private_dwellings_2021",
            "land_area_km2",
        ],
        "formula_note": "total_private_dwellings_2021 / land_area_km2",
        "candidate_sources": [],
    },
    {
        "original_code": "HUPTDEN90",
        "canonical_variable": "housing_permits_density",
        "description": "Housing permits / new residential construction density",
        "candidate_sources": [],
    },
    {
        "original_code": "MAESDEN92",
        "canonical_variable": "manufacturing_establishments_density",
        "description": "Manufacturing establishments per area",
        "candidate_sources": [],
    },
    {
        "original_code": "EARNDEN90",
        "canonical_variable": "earnings_density",
        "description": "Earnings in all industries per area",
        "candidate_sources": [],
    },
    {
        "original_code": "COMDEVDN92",
        "canonical_variable": "commercial_establishments_density",
        "description": "Commercial establishments per area",
        "candidate_sources": [],
    },
    {
        "original_code": "RPROPDEN92",
        "canonical_variable": "property_farm_value_density",
        "description": "Value of property and farm products sold per area",
        "candidate_sources": [],
    },
    {
        "original_code": "CVBRPC91",
        "canonical_variable": "labor_force_participation_rate",
        "description": "Percent population participating in labor force",
        "candidate_sources": [
            {
                "path": p("unemployment_2021/output/clean_census_tract_unemployment_2021.csv"),
                "candidate_columns": [
                    "labor_force_participation_rate",
                    "pct_labor_force_participation",
                    "participation_rate",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "likely_available_needs_column_confirmation",
            }
        ],
    },
    {
        "original_code": "FEMLBR90",
        "canonical_variable": "female_labor_force_participation_rate",
        "description": "Percent females participating in civilian labor force",
        "candidate_sources": [
            {
                "path": p("unemployment_2021/output/clean_census_tract_unemployment_2021.csv"),
                "candidate_columns": [
                    "female_labor_force_participation_rate",
                    "pct_female_labor_force_participation",
                    "female_participation_rate",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "likely_available_needs_column_confirmation",
            }
        ],
    },
    {
        "original_code": "AGRIPC90",
        "canonical_variable": "pct_extractive_employment",
        "description": "Percent employed in farming, fishing, mining, forestry",
        "candidate_sources": [
            {
                "path": p("occupation_industry_2021/output/clean_census_tract_occupation_industry_2021.csv"),
                "candidate_columns": [
                    "pct_extractive_employment",
                    "pct_agriculture_forestry_fishing_hunting_mining",
                    "primary_industry_measure_default",
                    "extractive_industry_measure_default",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "likely_available_needs_column_confirmation",
            }
        ],
    },
    {
        "original_code": "TRANPC90",
        "canonical_variable": "pct_transport_utilities_employment",
        "description": "Percent employed in transportation, communications, utilities",
        "candidate_sources": [
            {
                "path": p("occupation_industry_2021/output/clean_census_tract_occupation_industry_2021.csv"),
                "candidate_columns": [
                    "pct_transport_utilities_employment",
                    "transport_utilities_measure_default",
                    "pct_transportation_warehousing_utilities",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "likely_available_needs_column_confirmation",
            }
        ],
    },
    {
        "original_code": "SERVPC90",
        "canonical_variable": "pct_service_employment",
        "description": "Percent employed in service occupations",
        "candidate_sources": [
            {
                "path": p("occupation_industry_2021/output/clean_census_tract_occupation_industry_2021.csv"),
                "candidate_columns": [
                    "pct_service_employment",
                    "service_occupation_measure_default",
                    "pct_sales_service_occupations",
                    "pct_service_occupations",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "likely_available_needs_column_confirmation",
            }
        ],
    },
    {
        "original_code": "NRRESPC91",
        "canonical_variable": "nursing_home_residents_per_capita",
        "description": "Per capita residents in nursing homes",
        "candidate_sources": [
            {
                "path": p("residential_care_per_capita/output/clean_census_division_residential_care_per_100k_population_odhf_2021.csv"),
                "candidate_columns": [
                    "residential_care_facilities_per_100k_population_odhf",
                    "residential_care_facility_count_odhf",
                ],
                "declared_spatial_level": "census_division",
                "status_hint": "ready_direct_cd_proxy",
            }
        ],
    },
    {
        "original_code": "HOSPTPC91",
        "canonical_variable": "hospitals_per_capita",
        "description": "Per capita number of community hospitals",
        "candidate_sources": [
            {
                "path": p("hospitals_per_capita/output/clean_census_division_hospitals_per_100k_population_odhf_2021.csv"),
                "candidate_columns": [
                    "hospitals_per_100k_population_odhf",
                    "hospital_count_odhf",
                ],
                "declared_spatial_level": "census_division",
                "status_hint": "ready_direct_cd_proxy",
            }
        ],
    },
    {
        "original_code": "PCCHGPOP90",
        "canonical_variable": "pct_population_change",
        "description": "Percent population change",
        "derived_from_base": True,
        "required_base_columns": [
            "population_change_pct_2016_2021",
        ],
        "formula_note": "use population_change_pct_2016_2021 from CD population frame",
        "candidate_sources": [],
    },
    {
        "original_code": "PCTURB90",
        "canonical_variable": "pct_urban_population",
        "description": "Percent urban population",
        "candidate_sources": [],
    },
    {
        "original_code": "PCTFEM90",
        "canonical_variable": "pct_female",
        "description": "Percent females",
        "candidate_sources": [
            {
                "path": p("age_structure_2021/output/clean_census_tract_age_structure_2021.csv"),
                "candidate_columns": [
                    "pct_female",
                    "female_share",
                    "pct_population_female",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "likely_available_needs_column_confirmation",
            }
        ],
    },
    {
        "original_code": "PCTF_HH90",
        "canonical_variable": "pct_female_headed_households",
        "description": "Percent female-headed households, no spouse present",
        "candidate_sources": [
            {
                "path": p("household_family_2021/output/clean_census_tract_household_family_2021.csv"),
                "candidate_columns": [
                    "pct_female_headed_households",
                    "pct_lone_parent_female_households",
                    "single_parent_measure_default",
                    "pct_one_parent_family_households",
                ],
                "declared_spatial_level": "census_tract",
                "status_hint": "available_only_ct_needs_cd_cleaner",
            }
        ],
    },
    {
        "original_code": "SSBENPC90",
        "canonical_variable": "social_security_recipients_per_capita",
        "description": "Per capita Social Security / public benefit recipients",
        "candidate_sources": [],
    },
]


# -----------------------------
# Load base CD frame
# -----------------------------

base_path = find_existing_base_frame()
base = read_table(base_path)
base = normalize_columns(base)

print("\nLoaded Québec census-division base frame")
print("Path:", base_path.relative_to(DATA_DIR))
print("Rows:", len(base))
print("Columns:", list(base.columns))

required_base_keys = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
]

missing_base = [col for col in required_base_keys if col not in base.columns]
if missing_base:
    raise ValueError(
        "Base CD frame missing required columns:\n"
        + "\n".join(missing_base)
    )

base = base.copy()
base["census_division_code"] = clean_text(base["census_division_code"])
base["census_division_dguid"] = clean_text(base["census_division_dguid"])

if base["census_division_dguid"].duplicated().any():
    duplicated = base[base["census_division_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated census_division_dguid in base frame:\n"
        + duplicated[
            ["census_division_code", "census_division_dguid", "census_division_name"]
        ].head(30).to_string(index=False)
    )

base_rows = len(base)


# -----------------------------
# Read unique candidate files
# -----------------------------

source_cache: dict[Path, pd.DataFrame] = {}
source_inventory_rows = []

unique_paths = sorted(
    {
        source["path"]
        for spec in SOVI_SPECS
        for source in spec.get("candidate_sources", [])
    },
    key=lambda x: str(x),
)

print("\nInspecting candidate source files...")

for path in unique_paths:
    if not path.exists():
        source_inventory_rows.append(
            {
                "relative_path": str(path.relative_to(DATA_DIR)),
                "exists": False,
                "readable": False,
                "row_count": 0,
                "detected_spatial_level": "",
                "has_cd_dguid": False,
                "has_cd_code": False,
                "has_statcan_dguid": False,
                "columns": "",
                "error": "file_not_found",
            }
        )
        print(f"- MISSING: {path.relative_to(DATA_DIR)}")
        continue

    try:
        df = read_table(path)
        df = normalize_columns(df)

        source_cache[path] = df

        source_inventory_rows.append(
            {
                "relative_path": str(path.relative_to(DATA_DIR)),
                "exists": True,
                "readable": True,
                "row_count": len(df),
                "detected_spatial_level": detect_source_spatial_level(df),
                "has_cd_dguid": "census_division_dguid" in df.columns,
                "has_cd_code": "census_division_code" in df.columns,
                "has_statcan_dguid": "statcan_dguid" in df.columns,
                "columns": safe_column_list(df),
                "error": "",
            }
        )

        print(f"- OK: {path.relative_to(DATA_DIR)} ({len(df)} rows)")

    except Exception as exc:
        source_inventory_rows.append(
            {
                "relative_path": str(path.relative_to(DATA_DIR)),
                "exists": True,
                "readable": False,
                "row_count": 0,
                "detected_spatial_level": "",
                "has_cd_dguid": False,
                "has_cd_code": False,
                "has_statcan_dguid": False,
                "columns": "",
                "error": str(exc),
            }
        )

        print(f"- ERROR: {path.relative_to(DATA_DIR)} -> {exc}")


source_inventory = pd.DataFrame(source_inventory_rows)
source_inventory.to_csv(OUTPUT_SOURCE_INVENTORY, index=False)


# -----------------------------
# Inspect variable availability
# -----------------------------

availability_rows = []
candidate_column_rows = []

print("\nInspecting SoVI variable availability...")

for spec in SOVI_SPECS:
    original_code = spec["original_code"]
    canonical = spec["canonical_variable"]
    description = spec["description"]

    print(f"\nVariable {original_code} / {canonical}")

    if spec.get("derived_from_base", False):
        required_cols = spec.get("required_base_columns", [])
        missing_required = [col for col in required_cols if col not in base.columns]

        ready = len(missing_required) == 0

        availability_rows.append(
            {
                "original_code": original_code,
                "canonical_variable": canonical,
                "description": description,
                "status_current": (
                    "ready_derived_from_cd_base"
                    if ready
                    else "missing_required_base_columns"
                ),
                "selected_source_file": str(base_path.relative_to(DATA_DIR)),
                "selected_source_column": ", ".join(required_cols),
                "source_spatial_level": "census_division",
                "target_spatial_level": "census_division",
                "join_method": "base_frame_derived",
                "matched_cd_rows": base_rows if ready else 0,
                "unmatched_cd_rows": 0 if ready else base_rows,
                "source_rows": base_rows,
                "candidate_columns_found": ", ".join(required_cols if ready else []),
                "candidate_columns_missing": ", ".join(missing_required),
                "proxy_or_derivation_note": spec.get("formula_note", ""),
                "ready_for_first_sovi_table": ready,
                "needs_cd_cleaner": False,
                "needs_custom_join": False,
                "needs_new_data": not ready,
            }
        )

        print("  Derived from CD base:", "READY" if ready else "MISSING BASE COLUMNS")
        continue

    candidate_sources = spec.get("candidate_sources", [])

    if not candidate_sources:
        availability_rows.append(
            {
                "original_code": original_code,
                "canonical_variable": canonical,
                "description": description,
                "status_current": "missing_or_not_yet_collected",
                "selected_source_file": "",
                "selected_source_column": "",
                "source_spatial_level": "",
                "target_spatial_level": "census_division",
                "join_method": "",
                "matched_cd_rows": 0,
                "unmatched_cd_rows": base_rows,
                "source_rows": 0,
                "candidate_columns_found": "",
                "candidate_columns_missing": "",
                "proxy_or_derivation_note": "",
                "ready_for_first_sovi_table": False,
                "needs_cd_cleaner": False,
                "needs_custom_join": False,
                "needs_new_data": True,
            }
        )

        print("  No current candidate source.")
        continue

    best_row = None

    for source in candidate_sources:
        path = source["path"]
        candidate_columns = source.get("candidate_columns", [])
        declared_level = source.get("declared_spatial_level")
        status_hint = source.get("status_hint", "")

        df = source_cache.get(path)

        if df is None:
            row = {
                "original_code": original_code,
                "canonical_variable": canonical,
                "description": description,
                "status_current": "candidate_file_missing_or_unreadable",
                "selected_source_file": str(path.relative_to(DATA_DIR)),
                "selected_source_column": "",
                "source_spatial_level": declared_level or "",
                "target_spatial_level": "census_division",
                "join_method": "",
                "matched_cd_rows": 0,
                "unmatched_cd_rows": base_rows,
                "source_rows": 0,
                "candidate_columns_found": "",
                "candidate_columns_missing": ", ".join(candidate_columns),
                "proxy_or_derivation_note": status_hint,
                "ready_for_first_sovi_table": False,
                "needs_cd_cleaner": False,
                "needs_custom_join": status_hint == "available_requires_custom_join",
                "needs_new_data": True,
            }

            if best_row is None:
                best_row = row

            continue

        source_level = detect_source_spatial_level(df, declared_level)
        selected_col = first_existing_column(df, candidate_columns)
        found_cols = [col for col in candidate_columns if col in df.columns]
        missing_cols = [col for col in candidate_columns if col not in df.columns]

        for candidate_col in candidate_columns:
            if candidate_col in df.columns:
                summary = numeric_summary(df[candidate_col])
                candidate_column_rows.append(
                    {
                        "original_code": original_code,
                        "canonical_variable": canonical,
                        "source_file": str(path.relative_to(DATA_DIR)),
                        "candidate_column": candidate_col,
                        "column_found": True,
                        "source_spatial_level": source_level,
                        **summary,
                    }
                )
            else:
                candidate_column_rows.append(
                    {
                        "original_code": original_code,
                        "canonical_variable": canonical,
                        "source_file": str(path.relative_to(DATA_DIR)),
                        "candidate_column": candidate_col,
                        "column_found": False,
                        "source_spatial_level": source_level,
                        "numeric_non_missing": 0,
                        "numeric_missing": None,
                        "min": None,
                        "max": None,
                        "mean": None,
                        "median": None,
                    }
                )

        join_method, matched_rows, unmatched_rows = infer_join_method(df, base)

        if selected_col is None:
            status_current = "source_exists_but_no_candidate_column_found"
            ready = False
            needs_cd_cleaner = source_level == "census_tract"
            needs_custom_join = status_hint == "available_requires_custom_join"
            needs_new_data = False
        elif source_level == "census_division" and matched_rows == base_rows:
            status_current = status_hint if status_hint else "ready_direct_cd"
            ready = True
            needs_cd_cleaner = False
            needs_custom_join = False
            needs_new_data = False
        elif source_level == "census_division" and matched_rows > 0:
            status_current = "partial_cd_join_needs_review"
            ready = False
            needs_cd_cleaner = False
            needs_custom_join = True
            needs_new_data = False
        elif source_level == "census_tract":
            status_current = "available_only_ct_needs_cd_cleaner"
            ready = False
            needs_cd_cleaner = True
            needs_custom_join = False
            needs_new_data = False
        elif "crosswalk" in source_level or status_hint == "available_requires_custom_join":
            status_current = "available_requires_custom_join"
            ready = False
            needs_cd_cleaner = False
            needs_custom_join = True
            needs_new_data = False
        else:
            status_current = "source_exists_but_not_directly_usable"
            ready = False
            needs_cd_cleaner = False
            needs_custom_join = True
            needs_new_data = False

        row = {
            "original_code": original_code,
            "canonical_variable": canonical,
            "description": description,
            "status_current": status_current,
            "selected_source_file": str(path.relative_to(DATA_DIR)),
            "selected_source_column": selected_col or "",
            "source_spatial_level": source_level,
            "target_spatial_level": "census_division",
            "join_method": join_method,
            "matched_cd_rows": matched_rows,
            "unmatched_cd_rows": unmatched_rows,
            "source_rows": len(df),
            "candidate_columns_found": ", ".join(found_cols),
            "candidate_columns_missing": ", ".join(missing_cols),
            "proxy_or_derivation_note": status_hint,
            "ready_for_first_sovi_table": ready,
            "needs_cd_cleaner": needs_cd_cleaner,
            "needs_custom_join": needs_custom_join,
            "needs_new_data": needs_new_data,
        }

        if best_row is None:
            best_row = row
        else:
            # Prefer a ready direct CD source over anything else.
            if row["ready_for_first_sovi_table"] and not best_row["ready_for_first_sovi_table"]:
                best_row = row
            # Then prefer a source with a selected column.
            elif row["selected_source_column"] and not best_row["selected_source_column"]:
                best_row = row

    availability_rows.append(best_row)

    print("  Status:", best_row["status_current"])
    print("  Source:", best_row["selected_source_file"])
    print("  Column:", best_row["selected_source_column"] or "[none]")
    print("  Join:", best_row["join_method"])
    print("  Ready:", best_row["ready_for_first_sovi_table"])


availability = pd.DataFrame(availability_rows)
candidate_columns = pd.DataFrame(candidate_column_rows)

availability.to_csv(OUTPUT_VARIABLE_AVAILABILITY, index=False)
candidate_columns.to_csv(OUTPUT_CANDIDATE_COLUMN_DIAGNOSTICS, index=False)


# -----------------------------
# Summary
# -----------------------------

summary_rows = []

summary_rows.append({"metric": "base_frame_path", "value": str(base_path.relative_to(DATA_DIR))})
summary_rows.append({"metric": "base_rows", "value": base_rows})
summary_rows.append({"metric": "total_sovi_variables", "value": len(availability)})

for status, count in availability["status_current"].value_counts(dropna=False).items():
    summary_rows.append({"metric": f"status_{status}", "value": int(count)})

summary_rows.append(
    {
        "metric": "ready_for_first_sovi_table",
        "value": int(availability["ready_for_first_sovi_table"].sum()),
    }
)

summary_rows.append(
    {
        "metric": "needs_cd_cleaner",
        "value": int(availability["needs_cd_cleaner"].sum()),
    }
)

summary_rows.append(
    {
        "metric": "needs_custom_join",
        "value": int(availability["needs_custom_join"].sum()),
    }
)

summary_rows.append(
    {
        "metric": "needs_new_data",
        "value": int(availability["needs_new_data"].sum()),
    }
)

summary_rows.append(
    {
        "metric": "ready_variables",
        "value": ", ".join(
            availability.loc[
                availability["ready_for_first_sovi_table"],
                "canonical_variable",
            ].tolist()
        ),
    }
)

summary_rows.append(
    {
        "metric": "ct_only_variables_needing_cd_cleaner",
        "value": ", ".join(
            availability.loc[
                availability["needs_cd_cleaner"],
                "canonical_variable",
            ].tolist()
        ),
    }
)

summary_rows.append(
    {
        "metric": "variables_needing_new_data_or_no_source",
        "value": ", ".join(
            availability.loc[
                availability["needs_new_data"],
                "canonical_variable",
            ].tolist()
        ),
    }
)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False)


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("SoVI INPUT SOURCE INSPECTION SUMMARY")
print("=" * 72)

print("\nBase census-division frame:")
print("Path:", base_path.relative_to(DATA_DIR))
print("Rows:", base_rows)

print("\nAvailability by current status:")
print(availability["status_current"].value_counts(dropna=False).to_string())

print("\nReady for first SoVI table:")
ready_cols = [
    "original_code",
    "canonical_variable",
    "selected_source_file",
    "selected_source_column",
    "status_current",
]
ready = availability[availability["ready_for_first_sovi_table"]]
if ready.empty:
    print("[none]")
else:
    print(ready[ready_cols].to_string(index=False))

print("\nAvailable only at census-tract level / needs CD cleaner:")
ct_only = availability[availability["needs_cd_cleaner"]]
if ct_only.empty:
    print("[none]")
else:
    print(
        ct_only[
            [
                "original_code",
                "canonical_variable",
                "selected_source_file",
                "selected_source_column",
                "status_current",
            ]
        ].to_string(index=False)
    )

print("\nNeeds custom join/crosswalk:")
custom = availability[availability["needs_custom_join"]]
if custom.empty:
    print("[none]")
else:
    print(
        custom[
            [
                "original_code",
                "canonical_variable",
                "selected_source_file",
                "selected_source_column",
                "status_current",
            ]
        ].to_string(index=False)
    )

print("\nNo current source / likely future data collection:")
missing = availability[availability["needs_new_data"]]
if missing.empty:
    print("[none]")
else:
    print(
        missing[
            [
                "original_code",
                "canonical_variable",
                "description",
                "status_current",
            ]
        ].to_string(index=False)
    )

print("\nSaved:")
print(OUTPUT_SOURCE_INVENTORY)
print(OUTPUT_VARIABLE_AVAILABILITY)
print(OUTPUT_CANDIDATE_COLUMN_DIAGNOSTICS)
print(OUTPUT_SUMMARY)

print("\nDone.")