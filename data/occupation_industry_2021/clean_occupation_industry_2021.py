from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------
# Paths
# -----------------------------

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent

RAW_PROFILE_PATH = (
    DATA_DIR
    / "census_profile_2021"
    / "98-401-X2021007_English_CSV_data.csv"
)

OUTPUT_DIR = THIS_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "clean_census_tract_occupation_industry_2021.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_census_tract_occupation_industry_2021.parquet"


# -----------------------------
# Characteristic IDs
# -----------------------------
# Corrected from targeted inspection of IDs 2180-2310.
#
# Relevant block:
# 2223-2230 = labour force status
# 2231-2236 = work activity during reference year
# 2237-2245 = class of worker / job permanency
# 2246-2258 = occupation, broad NOC 2021 categories
# 2259-2281 = industry, NAICS 2017 sectors
#
# IDs 2180-2222 are location-of-study variables, not occupation/industry.
# IDs 2282+ are language-at-work variables, not occupation/industry.
#
# This folder is named occupation_industry_2021, but the clean table also
# includes adjacent labour-force structure variables because they are useful
# for SoVI-style features, HGNN node features, and contextual labour-market
# vulnerability analysis.

CHARACTERISTICS = {
    # Labour force status
    2223: "labour_force_status_total_population_15_plus",
    2224: "in_labour_force",
    2225: "employed",
    2226: "unemployed",
    2227: "not_in_labour_force",
    2228: "published_participation_rate",
    2229: "published_employment_rate",
    2230: "published_unemployment_rate",

    # Work activity during reference year
    2231: "work_activity_total_population_15_plus",
    2232: "did_not_work_reference_year",
    2233: "worked_reference_year",
    2234: "worked_full_year_full_time",
    2235: "worked_part_year_or_part_time",
    2236: "average_weeks_worked_reference_year",

    # Class of worker / job permanency
    2237: "class_worker_total_labour_force",
    2238: "class_worker_not_applicable",
    2239: "all_classes_of_workers",
    2240: "employee",
    2241: "permanent_position",
    2242: "temporary_position",
    2243: "fixed_term_position_1_year_or_more",
    2244: "casual_seasonal_short_term_position",
    2245: "self_employed",

    # Occupation - broad NOC 2021 categories
    2246: "occupation_total_labour_force",
    2247: "occupation_not_applicable",
    2248: "all_occupations",
    2249: "occupation_management",
    2250: "occupation_business_finance_administration",
    2251: "occupation_natural_applied_sciences",
    2252: "occupation_health",
    2253: "occupation_education_law_social_community_government",
    2254: "occupation_art_culture_recreation_sport",
    2255: "occupation_sales_service",
    2256: "occupation_trades_transport_equipment_operators",
    2257: "occupation_natural_resources_agriculture_production",
    2258: "occupation_manufacturing_utilities",

    # Industry - NAICS 2017 sectors
    2259: "industry_total_labour_force",
    2260: "industry_not_applicable",
    2261: "all_industries",
    2262: "industry_agriculture_forestry_fishing_hunting",
    2263: "industry_mining_quarrying_oil_gas",
    2264: "industry_utilities",
    2265: "industry_construction",
    2266: "industry_manufacturing",
    2267: "industry_wholesale_trade",
    2268: "industry_retail_trade",
    2269: "industry_transportation_warehousing",
    2270: "industry_information_cultural",
    2271: "industry_finance_insurance",
    2272: "industry_real_estate_rental_leasing",
    2273: "industry_professional_scientific_technical",
    2274: "industry_management_companies_enterprises",
    2275: "industry_admin_support_waste_management_remediation",
    2276: "industry_educational_services",
    2277: "industry_health_care_social_assistance",
    2278: "industry_arts_entertainment_recreation",
    2279: "industry_accommodation_food_services",
    2280: "industry_other_services_except_public_admin",
    2281: "industry_public_administration",
}

# These are published percentage/rate rows in the Census Profile.
# They are kept as 0-100 values and converted to 0-1 proportion versions.
PUBLISHED_RATE_IDS = {
    2228,
    2229,
    2230,
}

SOURCE_OCCUPATION_INDUSTRY = (
    "Statistics Canada Census Profile, 2021 Census of Population; "
    "labour force status, work activity, class of worker, occupation broad NOC 2021 categories, "
    "and industry NAICS 2017 sectors, 25% sample data"
)


# -----------------------------
# Helper functions
# -----------------------------

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Divide two numeric pandas Series safely.

    If the denominator is 0 or missing, the result is np.nan.
    This avoids inf values while keeping the output column numeric.
    """
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")

    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], np.nan)

    return result.astype(float)


def bounded_proportion(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Compute a proportion and clip it to [0, 1].

    This protects derived proportions from minor Census rounding artifacts.
    Raw counts are preserved unchanged.
    """
    result = safe_divide(numerator, denominator)
    result = result.clip(lower=0, upper=1)
    return result.astype(float)


def published_percent_to_proportion(series: pd.Series) -> pd.Series:
    """
    Convert a published StatCan percentage from 0-100 into a proportion from 0-1.
    Missing values remain missing.
    """
    result = pd.to_numeric(series, errors="coerce") / 100.0
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.clip(lower=0, upper=1)
    return result.astype(float)


# -----------------------------
# Load only needed columns
# -----------------------------

usecols = [
    "DGUID",
    "GEO_LEVEL",
    "GEO_NAME",
    "CHARACTERISTIC_ID",
    "CHARACTERISTIC_NAME",
    "C1_COUNT_TOTAL",
    "C10_RATE_TOTAL",
    "SYMBOL",
]

df = pd.read_csv(
    RAW_PROFILE_PATH,
    usecols=usecols,
    encoding="iso-8859-1",
    low_memory=False,
)

print("Loaded Census Profile")
print("Rows:", len(df))


# -----------------------------
# Filter to census tract occupation / industry rows
# -----------------------------

df["CHARACTERISTIC_ID"] = pd.to_numeric(
    df["CHARACTERISTIC_ID"],
    errors="coerce",
).astype("Int64")

rows = df[
    (df["GEO_LEVEL"].astype(str).str.strip() == "Census tract")
    & (df["CHARACTERISTIC_ID"].isin(CHARACTERISTICS.keys()))
].copy()

print("\nOccupation / industry rows found:", len(rows))

if rows.empty:
    raise ValueError("No census tract occupation / industry rows found.")


# -----------------------------
# Inspect selected characteristics
# -----------------------------

selected_characteristics = (
    rows[
        [
            "CHARACTERISTIC_ID",
            "CHARACTERISTIC_NAME",
        ]
    ]
    .drop_duplicates()
    .sort_values("CHARACTERISTIC_ID")
)

print("\nSelected characteristics:")
print(selected_characteristics.to_string(index=False))


# -----------------------------
# Check that all expected characteristics were found
# -----------------------------

found_ids = set(rows["CHARACTERISTIC_ID"].dropna().astype(int).unique())
expected_ids = set(CHARACTERISTICS.keys())
missing_ids = sorted(expected_ids - found_ids)

if missing_ids:
    print("\nWarning: some expected characteristic IDs were not found:")
    for characteristic_id in missing_ids:
        print(f"  {characteristic_id}: {CHARACTERISTICS[characteristic_id]}")


# -----------------------------
# Prepare value column
# -----------------------------
# For this block, the useful numeric values are read from C1_COUNT_TOTAL.
# Published rate rows are stored as 0-100 values and later converted to 0-1.

rows["value"] = pd.to_numeric(
    rows["C1_COUNT_TOTAL"],
    errors="coerce",
)


# -----------------------------
# Report missing / suppressed values
# -----------------------------

missing = rows[rows["value"].isna()].copy()

print("\nMissing or non-numeric occupation / industry values:", len(missing))

if not missing.empty:
    print("\nMissing values by characteristic:")
    print(
        missing["CHARACTERISTIC_ID"]
        .map(CHARACTERISTICS)
        .value_counts(dropna=False)
    )

    print("\nSYMBOL counts for missing rows:")
    print(missing["SYMBOL"].value_counts(dropna=False))

    print("\nMissing preview:")
    print(
        missing[
            [
                "DGUID",
                "GEO_NAME",
                "CHARACTERISTIC_ID",
                "CHARACTERISTIC_NAME",
                "C1_COUNT_TOTAL",
                "C10_RATE_TOTAL",
                "SYMBOL",
            ]
        ]
        .head(50)
        .to_string(index=False)
    )


# -----------------------------
# Pivot to one row per census tract
# -----------------------------

rows["feature_name"] = rows["CHARACTERISTIC_ID"].map(CHARACTERISTICS)

wide = (
    rows
    .set_index(["DGUID", "GEO_NAME", "feature_name"])["value"]
    .unstack("feature_name")
    .reset_index()
)

wide.columns.name = None

clean = wide.rename(
    columns={
        "DGUID": "statcan_dguid",
        "GEO_NAME": "geo_name",
    }
).copy()


# -----------------------------
# Ensure all expected output columns exist
# -----------------------------

for column_name in CHARACTERISTICS.values():
    if column_name not in clean.columns:
        clean[column_name] = np.nan


# -----------------------------
# Labour-force status proportions
# -----------------------------

clean["pct_in_labour_force"] = bounded_proportion(
    clean["in_labour_force"],
    clean["labour_force_status_total_population_15_plus"],
)

clean["pct_employed"] = bounded_proportion(
    clean["employed"],
    clean["labour_force_status_total_population_15_plus"],
)

clean["pct_unemployed_count_based"] = bounded_proportion(
    clean["unemployed"],
    clean["labour_force_status_total_population_15_plus"],
)

clean["pct_not_in_labour_force"] = bounded_proportion(
    clean["not_in_labour_force"],
    clean["labour_force_status_total_population_15_plus"],
)

# Published 0-100 rates converted to 0-1.
clean["participation_rate_published"] = published_percent_to_proportion(
    clean["published_participation_rate"]
)

clean["employment_rate_published"] = published_percent_to_proportion(
    clean["published_employment_rate"]
)

clean["unemployment_rate_published"] = published_percent_to_proportion(
    clean["published_unemployment_rate"]
)


# -----------------------------
# Work activity proportions
# -----------------------------

clean["pct_did_not_work_reference_year"] = bounded_proportion(
    clean["did_not_work_reference_year"],
    clean["work_activity_total_population_15_plus"],
)

clean["pct_worked_reference_year"] = bounded_proportion(
    clean["worked_reference_year"],
    clean["work_activity_total_population_15_plus"],
)

clean["pct_worked_full_year_full_time"] = bounded_proportion(
    clean["worked_full_year_full_time"],
    clean["work_activity_total_population_15_plus"],
)

clean["pct_worked_part_year_or_part_time"] = bounded_proportion(
    clean["worked_part_year_or_part_time"],
    clean["work_activity_total_population_15_plus"],
)

clean["pct_part_time_or_part_year_among_workers"] = bounded_proportion(
    clean["worked_part_year_or_part_time"],
    clean["worked_reference_year"],
)


# -----------------------------
# Class of worker / job permanency proportions
# -----------------------------

clean["pct_all_classes_of_workers"] = bounded_proportion(
    clean["all_classes_of_workers"],
    clean["class_worker_total_labour_force"],
)

clean["pct_employee"] = bounded_proportion(
    clean["employee"],
    clean["all_classes_of_workers"],
)

clean["pct_permanent_position"] = bounded_proportion(
    clean["permanent_position"],
    clean["employee"],
)

clean["pct_temporary_position"] = bounded_proportion(
    clean["temporary_position"],
    clean["employee"],
)

clean["pct_fixed_term_position_1_year_or_more"] = bounded_proportion(
    clean["fixed_term_position_1_year_or_more"],
    clean["employee"],
)

clean["pct_casual_seasonal_short_term_position"] = bounded_proportion(
    clean["casual_seasonal_short_term_position"],
    clean["employee"],
)

clean["pct_self_employed"] = bounded_proportion(
    clean["self_employed"],
    clean["all_classes_of_workers"],
)


# -----------------------------
# Occupation proportions
# -----------------------------

occupation_denominator = clean["all_occupations"]

clean["pct_occupation_management"] = bounded_proportion(
    clean["occupation_management"],
    occupation_denominator,
)

clean["pct_occupation_business_finance_administration"] = bounded_proportion(
    clean["occupation_business_finance_administration"],
    occupation_denominator,
)

clean["pct_occupation_natural_applied_sciences"] = bounded_proportion(
    clean["occupation_natural_applied_sciences"],
    occupation_denominator,
)

clean["pct_occupation_health"] = bounded_proportion(
    clean["occupation_health"],
    occupation_denominator,
)

clean["pct_occupation_education_law_social_community_government"] = bounded_proportion(
    clean["occupation_education_law_social_community_government"],
    occupation_denominator,
)

clean["pct_occupation_art_culture_recreation_sport"] = bounded_proportion(
    clean["occupation_art_culture_recreation_sport"],
    occupation_denominator,
)

clean["pct_occupation_sales_service"] = bounded_proportion(
    clean["occupation_sales_service"],
    occupation_denominator,
)

clean["pct_occupation_trades_transport_equipment_operators"] = bounded_proportion(
    clean["occupation_trades_transport_equipment_operators"],
    occupation_denominator,
)

clean["pct_occupation_natural_resources_agriculture_production"] = bounded_proportion(
    clean["occupation_natural_resources_agriculture_production"],
    occupation_denominator,
)

clean["pct_occupation_manufacturing_utilities"] = bounded_proportion(
    clean["occupation_manufacturing_utilities"],
    occupation_denominator,
)

# Composite occupation features useful for SoVI/HGNN.
clean["occupation_frontline_service_count"] = (
    clean["occupation_sales_service"]
    + clean["occupation_trades_transport_equipment_operators"]
    + clean["occupation_manufacturing_utilities"]
    + clean["occupation_natural_resources_agriculture_production"]
)

clean["pct_occupation_frontline_service"] = bounded_proportion(
    clean["occupation_frontline_service_count"],
    occupation_denominator,
)

clean["occupation_knowledge_professional_count"] = (
    clean["occupation_business_finance_administration"]
    + clean["occupation_natural_applied_sciences"]
    + clean["occupation_health"]
    + clean["occupation_education_law_social_community_government"]
)

clean["pct_occupation_knowledge_professional"] = bounded_proportion(
    clean["occupation_knowledge_professional_count"],
    occupation_denominator,
)


# -----------------------------
# Industry proportions
# -----------------------------

industry_denominator = clean["all_industries"]

clean["pct_industry_agriculture_forestry_fishing_hunting"] = bounded_proportion(
    clean["industry_agriculture_forestry_fishing_hunting"],
    industry_denominator,
)

clean["pct_industry_mining_quarrying_oil_gas"] = bounded_proportion(
    clean["industry_mining_quarrying_oil_gas"],
    industry_denominator,
)

clean["pct_industry_utilities"] = bounded_proportion(
    clean["industry_utilities"],
    industry_denominator,
)

clean["pct_industry_construction"] = bounded_proportion(
    clean["industry_construction"],
    industry_denominator,
)

clean["pct_industry_manufacturing"] = bounded_proportion(
    clean["industry_manufacturing"],
    industry_denominator,
)

clean["pct_industry_wholesale_trade"] = bounded_proportion(
    clean["industry_wholesale_trade"],
    industry_denominator,
)

clean["pct_industry_retail_trade"] = bounded_proportion(
    clean["industry_retail_trade"],
    industry_denominator,
)

clean["pct_industry_transportation_warehousing"] = bounded_proportion(
    clean["industry_transportation_warehousing"],
    industry_denominator,
)

clean["pct_industry_information_cultural"] = bounded_proportion(
    clean["industry_information_cultural"],
    industry_denominator,
)

clean["pct_industry_finance_insurance"] = bounded_proportion(
    clean["industry_finance_insurance"],
    industry_denominator,
)

clean["pct_industry_real_estate_rental_leasing"] = bounded_proportion(
    clean["industry_real_estate_rental_leasing"],
    industry_denominator,
)

clean["pct_industry_professional_scientific_technical"] = bounded_proportion(
    clean["industry_professional_scientific_technical"],
    industry_denominator,
)

clean["pct_industry_management_companies_enterprises"] = bounded_proportion(
    clean["industry_management_companies_enterprises"],
    industry_denominator,
)

clean["pct_industry_admin_support_waste_management_remediation"] = bounded_proportion(
    clean["industry_admin_support_waste_management_remediation"],
    industry_denominator,
)

clean["pct_industry_educational_services"] = bounded_proportion(
    clean["industry_educational_services"],
    industry_denominator,
)

clean["pct_industry_health_care_social_assistance"] = bounded_proportion(
    clean["industry_health_care_social_assistance"],
    industry_denominator,
)

clean["pct_industry_arts_entertainment_recreation"] = bounded_proportion(
    clean["industry_arts_entertainment_recreation"],
    industry_denominator,
)

clean["pct_industry_accommodation_food_services"] = bounded_proportion(
    clean["industry_accommodation_food_services"],
    industry_denominator,
)

clean["pct_industry_other_services_except_public_admin"] = bounded_proportion(
    clean["industry_other_services_except_public_admin"],
    industry_denominator,
)

clean["pct_industry_public_administration"] = bounded_proportion(
    clean["industry_public_administration"],
    industry_denominator,
)

# Composite industry features useful for SoVI/HGNN.
clean["industry_public_essential_services_count"] = (
    clean["industry_educational_services"]
    + clean["industry_health_care_social_assistance"]
    + clean["industry_public_administration"]
)

clean["pct_industry_public_essential_services"] = bounded_proportion(
    clean["industry_public_essential_services_count"],
    industry_denominator,
)

clean["industry_service_retail_accommodation_count"] = (
    clean["industry_retail_trade"]
    + clean["industry_accommodation_food_services"]
    + clean["industry_arts_entertainment_recreation"]
    + clean["industry_other_services_except_public_admin"]
)

clean["pct_industry_service_retail_accommodation"] = bounded_proportion(
    clean["industry_service_retail_accommodation_count"],
    industry_denominator,
)

clean["industry_physical_infrastructure_count"] = (
    clean["industry_construction"]
    + clean["industry_manufacturing"]
    + clean["industry_transportation_warehousing"]
    + clean["industry_utilities"]
    + clean["industry_agriculture_forestry_fishing_hunting"]
    + clean["industry_mining_quarrying_oil_gas"]
)

clean["pct_industry_physical_infrastructure"] = bounded_proportion(
    clean["industry_physical_infrastructure_count"],
    industry_denominator,
)

clean["industry_professional_knowledge_count"] = (
    clean["industry_information_cultural"]
    + clean["industry_finance_insurance"]
    + clean["industry_professional_scientific_technical"]
    + clean["industry_management_companies_enterprises"]
)

clean["pct_industry_professional_knowledge"] = bounded_proportion(
    clean["industry_professional_knowledge_count"],
    industry_denominator,
)


# -----------------------------
# Add default named fields
# -----------------------------

clean["sales_service_occupation_measure_default"] = clean[
    "pct_occupation_sales_service"
].astype(float)

clean["sales_service_occupation_measure_default_description"] = (
    "pct_occupation_sales_service; share of labour force in sales and service occupations"
)

clean["trades_transport_occupation_measure_default"] = clean[
    "pct_occupation_trades_transport_equipment_operators"
].astype(float)

clean["trades_transport_occupation_measure_default_description"] = (
    "pct_occupation_trades_transport_equipment_operators; share of labour force in trades, transport, and equipment operator occupations"
)

clean["temporary_work_measure_default"] = clean["pct_temporary_position"].astype(float)

clean["temporary_work_measure_default_description"] = (
    "pct_temporary_position; share of employees in temporary positions"
)

clean["part_time_or_part_year_work_measure_default"] = clean[
    "pct_part_time_or_part_year_among_workers"
].astype(float)

clean["part_time_or_part_year_work_measure_default_description"] = (
    "pct_part_time_or_part_year_among_workers; share of workers working part year and/or part time"
)

clean["service_retail_accommodation_industry_measure_default"] = clean[
    "pct_industry_service_retail_accommodation"
].astype(float)

clean["service_retail_accommodation_industry_measure_default_description"] = (
    "pct_industry_service_retail_accommodation; retail, accommodation/food, arts/recreation, and other services"
)


# -----------------------------
# Add metadata
# -----------------------------

clean["unit_type"] = "census_tract"
clean["census_year"] = 2021
clean["source_occupation_industry"] = SOURCE_OCCUPATION_INDUSTRY


# -----------------------------
# Reorder columns
# -----------------------------

ordered_columns = [
    "statcan_dguid",
    "geo_name",
    "unit_type",
    "census_year",

    # Labour force status
    "labour_force_status_total_population_15_plus",
    "in_labour_force",
    "employed",
    "unemployed",
    "not_in_labour_force",
    "pct_in_labour_force",
    "pct_employed",
    "pct_unemployed_count_based",
    "pct_not_in_labour_force",
    "published_participation_rate",
    "participation_rate_published",
    "published_employment_rate",
    "employment_rate_published",
    "published_unemployment_rate",
    "unemployment_rate_published",

    # Work activity
    "work_activity_total_population_15_plus",
    "did_not_work_reference_year",
    "worked_reference_year",
    "worked_full_year_full_time",
    "worked_part_year_or_part_time",
    "average_weeks_worked_reference_year",
    "pct_did_not_work_reference_year",
    "pct_worked_reference_year",
    "pct_worked_full_year_full_time",
    "pct_worked_part_year_or_part_time",
    "pct_part_time_or_part_year_among_workers",

    # Class of worker / job permanency
    "class_worker_total_labour_force",
    "class_worker_not_applicable",
    "all_classes_of_workers",
    "employee",
    "permanent_position",
    "temporary_position",
    "fixed_term_position_1_year_or_more",
    "casual_seasonal_short_term_position",
    "self_employed",
    "pct_all_classes_of_workers",
    "pct_employee",
    "pct_permanent_position",
    "pct_temporary_position",
    "pct_fixed_term_position_1_year_or_more",
    "pct_casual_seasonal_short_term_position",
    "pct_self_employed",

    # Occupation raw counts
    "occupation_total_labour_force",
    "occupation_not_applicable",
    "all_occupations",
    "occupation_management",
    "occupation_business_finance_administration",
    "occupation_natural_applied_sciences",
    "occupation_health",
    "occupation_education_law_social_community_government",
    "occupation_art_culture_recreation_sport",
    "occupation_sales_service",
    "occupation_trades_transport_equipment_operators",
    "occupation_natural_resources_agriculture_production",
    "occupation_manufacturing_utilities",

    # Occupation proportions
    "pct_occupation_management",
    "pct_occupation_business_finance_administration",
    "pct_occupation_natural_applied_sciences",
    "pct_occupation_health",
    "pct_occupation_education_law_social_community_government",
    "pct_occupation_art_culture_recreation_sport",
    "pct_occupation_sales_service",
    "pct_occupation_trades_transport_equipment_operators",
    "pct_occupation_natural_resources_agriculture_production",
    "pct_occupation_manufacturing_utilities",
    "occupation_frontline_service_count",
    "pct_occupation_frontline_service",
    "occupation_knowledge_professional_count",
    "pct_occupation_knowledge_professional",

    # Industry raw counts
    "industry_total_labour_force",
    "industry_not_applicable",
    "all_industries",
    "industry_agriculture_forestry_fishing_hunting",
    "industry_mining_quarrying_oil_gas",
    "industry_utilities",
    "industry_construction",
    "industry_manufacturing",
    "industry_wholesale_trade",
    "industry_retail_trade",
    "industry_transportation_warehousing",
    "industry_information_cultural",
    "industry_finance_insurance",
    "industry_real_estate_rental_leasing",
    "industry_professional_scientific_technical",
    "industry_management_companies_enterprises",
    "industry_admin_support_waste_management_remediation",
    "industry_educational_services",
    "industry_health_care_social_assistance",
    "industry_arts_entertainment_recreation",
    "industry_accommodation_food_services",
    "industry_other_services_except_public_admin",
    "industry_public_administration",

    # Industry proportions
    "pct_industry_agriculture_forestry_fishing_hunting",
    "pct_industry_mining_quarrying_oil_gas",
    "pct_industry_utilities",
    "pct_industry_construction",
    "pct_industry_manufacturing",
    "pct_industry_wholesale_trade",
    "pct_industry_retail_trade",
    "pct_industry_transportation_warehousing",
    "pct_industry_information_cultural",
    "pct_industry_finance_insurance",
    "pct_industry_real_estate_rental_leasing",
    "pct_industry_professional_scientific_technical",
    "pct_industry_management_companies_enterprises",
    "pct_industry_admin_support_waste_management_remediation",
    "pct_industry_educational_services",
    "pct_industry_health_care_social_assistance",
    "pct_industry_arts_entertainment_recreation",
    "pct_industry_accommodation_food_services",
    "pct_industry_other_services_except_public_admin",
    "pct_industry_public_administration",
    "industry_public_essential_services_count",
    "pct_industry_public_essential_services",
    "industry_service_retail_accommodation_count",
    "pct_industry_service_retail_accommodation",
    "industry_physical_infrastructure_count",
    "pct_industry_physical_infrastructure",
    "industry_professional_knowledge_count",
    "pct_industry_professional_knowledge",

    # Defaults
    "sales_service_occupation_measure_default",
    "sales_service_occupation_measure_default_description",
    "trades_transport_occupation_measure_default",
    "trades_transport_occupation_measure_default_description",
    "temporary_work_measure_default",
    "temporary_work_measure_default_description",
    "part_time_or_part_year_work_measure_default",
    "part_time_or_part_year_work_measure_default_description",
    "service_retail_accommodation_industry_measure_default",
    "service_retail_accommodation_industry_measure_default_description",

    "source_occupation_industry",
]

clean = clean[ordered_columns]


# -----------------------------
# Validation
# -----------------------------

if clean["statcan_dguid"].duplicated().any():
    duplicated = clean[clean["statcan_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated statcan_dguid values found:\n"
        + duplicated.to_string(index=False)
    )

computed_cols = [
    "pct_in_labour_force",
    "pct_employed",
    "pct_unemployed_count_based",
    "pct_not_in_labour_force",
    "participation_rate_published",
    "employment_rate_published",
    "unemployment_rate_published",

    "pct_did_not_work_reference_year",
    "pct_worked_reference_year",
    "pct_worked_full_year_full_time",
    "pct_worked_part_year_or_part_time",
    "pct_part_time_or_part_year_among_workers",

    "pct_all_classes_of_workers",
    "pct_employee",
    "pct_permanent_position",
    "pct_temporary_position",
    "pct_fixed_term_position_1_year_or_more",
    "pct_casual_seasonal_short_term_position",
    "pct_self_employed",

    "pct_occupation_management",
    "pct_occupation_business_finance_administration",
    "pct_occupation_natural_applied_sciences",
    "pct_occupation_health",
    "pct_occupation_education_law_social_community_government",
    "pct_occupation_art_culture_recreation_sport",
    "pct_occupation_sales_service",
    "pct_occupation_trades_transport_equipment_operators",
    "pct_occupation_natural_resources_agriculture_production",
    "pct_occupation_manufacturing_utilities",
    "pct_occupation_frontline_service",
    "pct_occupation_knowledge_professional",

    "pct_industry_agriculture_forestry_fishing_hunting",
    "pct_industry_mining_quarrying_oil_gas",
    "pct_industry_utilities",
    "pct_industry_construction",
    "pct_industry_manufacturing",
    "pct_industry_wholesale_trade",
    "pct_industry_retail_trade",
    "pct_industry_transportation_warehousing",
    "pct_industry_information_cultural",
    "pct_industry_finance_insurance",
    "pct_industry_real_estate_rental_leasing",
    "pct_industry_professional_scientific_technical",
    "pct_industry_management_companies_enterprises",
    "pct_industry_admin_support_waste_management_remediation",
    "pct_industry_educational_services",
    "pct_industry_health_care_social_assistance",
    "pct_industry_arts_entertainment_recreation",
    "pct_industry_accommodation_food_services",
    "pct_industry_other_services_except_public_admin",
    "pct_industry_public_administration",
    "pct_industry_public_essential_services",
    "pct_industry_service_retail_accommodation",
    "pct_industry_physical_infrastructure",
    "pct_industry_professional_knowledge",

    "sales_service_occupation_measure_default",
    "trades_transport_occupation_measure_default",
    "temporary_work_measure_default",
    "part_time_or_part_year_work_measure_default",
    "service_retail_accommodation_industry_measure_default",
]

for col in computed_cols:
    clean[col] = pd.to_numeric(clean[col], errors="coerce").astype(float)

    if np.isinf(clean[col]).any():
        raise ValueError(f"Infinite values found in computed column: {col}")

    min_value = clean[col].min(skipna=True)
    max_value = clean[col].max(skipna=True)

    if min_value < 0 or max_value > 1:
        raise ValueError(
            f"Out-of-bounds values remain in {col}: "
            f"min={min_value}, max={max_value}"
        )


# Sanity checks to catch denominator mistakes.
if clean["sales_service_occupation_measure_default"].mean(skipna=True) > 0.7:
    raise ValueError(
        "Sales/service occupation share is suspiciously high. "
        "Check occupation denominator mapping."
    )

if clean["temporary_work_measure_default"].mean(skipna=True) > 0.7:
    raise ValueError(
        "Temporary work share is suspiciously high. "
        "Check class-of-worker denominator mapping."
    )

if clean["service_retail_accommodation_industry_measure_default"].mean(skipna=True) > 0.7:
    raise ValueError(
        "Service/retail/accommodation industry share is suspiciously high. "
        "Check industry denominator mapping."
    )


print("\nClean occupation / industry table")
print("Rows:", len(clean))
print("Columns:", list(clean.columns))

print("\nMissing values by column:")
print(clean.isna().sum())

print("\nPublished unemployment rate summary:")
print(clean["unemployment_rate_published"].describe())

print("\nSales/service occupation measure summary:")
print(clean["sales_service_occupation_measure_default"].describe())

print("\nTrades/transport occupation measure summary:")
print(clean["trades_transport_occupation_measure_default"].describe())

print("\nTemporary work measure summary:")
print(clean["temporary_work_measure_default"].describe())

print("\nPart-time or part-year work among workers summary:")
print(clean["part_time_or_part_year_work_measure_default"].describe())

print("\nService/retail/accommodation industry measure summary:")
print(clean["service_retail_accommodation_industry_measure_default"].describe())

print("\nPhysical infrastructure industry measure summary:")
print(clean["pct_industry_physical_infrastructure"].describe())

print("\nProfessional knowledge industry measure summary:")
print(clean["pct_industry_professional_knowledge"].describe())

print("\nPreview:")
print(clean.head(10).to_string(index=False))


# -----------------------------
# Save outputs
# -----------------------------

clean.to_csv(OUTPUT_CSV, index=False)
clean.to_parquet(OUTPUT_PARQUET, index=False)

print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_PARQUET)