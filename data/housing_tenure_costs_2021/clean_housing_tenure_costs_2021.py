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

OUTPUT_CSV = OUTPUT_DIR / "clean_census_tract_housing_tenure_costs_2021.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_census_tract_housing_tenure_costs_2021.parquet"


# -----------------------------
# Characteristic IDs
# -----------------------------
# Corrected from targeted inspection of IDs 1414-1495.
#
# Tenure:
# 1414 = Total - Private households by tenure - 25% sample data
# 1415 = Owner
# 1416 = Renter
# 1417 = Dwelling provided by local government, First Nation or Indian band
#
# Condominium status:
# 1418 = Total - Occupied private dwellings by condominium status - 25% sample data
# 1419 = Condominium
# 1420 = Not condominium
#
# Dwelling condition:
# 1449 = Total - Occupied private dwellings by dwelling condition - 25% sample data
# 1450 = Only regular maintenance and minor repairs needed
# 1451 = Major repairs needed
#
# Shelter-cost-to-income ratio:
# 1465 = Total - Owner and tenant households with household total income > 0,
#        in non-farm, non-reserve private dwellings by shelter-cost-to-income ratio
# 1466 = Spending less than 30% of income on shelter costs
# 1467 = Spending 30% or more of income on shelter costs
# 1468 = 30% to less than 100%
#
# Housing indicators:
# 1469 = Total - Occupied private dwellings by housing indicators - 25% sample data
# 1470 = Total - households with at least one housing indicator problem:
#        spending 30%+ OR not suitable OR major repairs needed
# 1471 = Spending 30%+ only
# 1472 = Not suitable only
# 1473 = Major repairs needed only
# 1474 = Spending 30%+ and not suitable
# 1475 = Spending 30%+ and major repairs needed
# 1476 = Not suitable and major repairs needed
# 1477 = Spending 30%+ and not suitable and major repairs needed
# 1478 = Acceptable housing
#
# Core housing need:
# 1479 = Total - Owner and tenant households with income > 0 and
#        shelter-cost-to-income ratio < 100%
# 1480 = In core need
# 1481 = Not in core need
#
# Owner housing costs and values:
# 1482 = Total - Owner households in non-farm, non-reserve private dwellings
# 1483 = % of owner households with a mortgage
# 1484 = % of owner households spending 30%+ on shelter costs
# 1485 = % owner households in core housing need
# 1486 = Median monthly shelter costs for owned dwellings ($)
# 1487 = Average monthly shelter costs for owned dwellings ($)
# 1488 = Median value of dwellings ($)
# 1489 = Average value of dwellings ($)
#
# Tenant housing costs:
# 1490 = Total - Tenant households in non-farm, non-reserve private dwellings
# 1491 = % of tenant households in subsidized housing
# 1492 = % of tenant households spending 30%+ on shelter costs
# 1493 = % tenant households in core housing need
# 1494 = Median monthly shelter costs for rented dwellings ($)
# 1495 = Average monthly shelter costs for rented dwellings ($)

CHARACTERISTICS = {
    # Tenure
    1414: "tenure_total_households",
    1415: "owner_households",
    1416: "renter_households",
    1417: "government_or_band_provided_dwelling_households",

    # Condominium status
    1418: "condominium_status_total_dwellings",
    1419: "condominium_dwellings",
    1420: "not_condominium_dwellings",

    # Dwelling condition
    1449: "dwelling_condition_total",
    1450: "regular_maintenance_or_minor_repairs_dwellings",
    1451: "major_repairs_needed_dwellings",

    # Shelter-cost-to-income ratio
    1465: "shelter_cost_income_ratio_total_households",
    1466: "shelter_cost_less_than_30pct_households",
    1467: "shelter_cost_30pct_or_more_households",
    1468: "shelter_cost_30pct_to_less_than_100pct_households",

    # Housing indicators
    1469: "housing_indicators_total_dwellings",
    1470: "housing_indicator_problem_any",
    1471: "housing_indicator_cost_burden_only",
    1472: "housing_indicator_not_suitable_only",
    1473: "housing_indicator_major_repairs_only",
    1474: "housing_indicator_cost_burden_and_not_suitable",
    1475: "housing_indicator_cost_burden_and_major_repairs",
    1476: "housing_indicator_not_suitable_and_major_repairs",
    1477: "housing_indicator_all_three",
    1478: "acceptable_housing",

    # Core housing need
    1479: "core_housing_need_total_households",
    1480: "core_housing_need_households",
    1481: "not_core_housing_need_households",

    # Owner housing costs / values
    1482: "owner_households_nonfarm_nonreserve_total",
    1483: "published_pct_owner_households_with_mortgage",
    1484: "published_pct_owner_shelter_cost_30pct_or_more",
    1485: "published_pct_owner_core_housing_need",
    1486: "median_monthly_shelter_costs_owned_dwellings",
    1487: "average_monthly_shelter_costs_owned_dwellings",
    1488: "median_value_owned_dwellings",
    1489: "average_value_owned_dwellings",

    # Tenant housing costs
    1490: "tenant_households_nonfarm_nonreserve_total",
    1491: "published_pct_tenant_subsidized_housing",
    1492: "published_pct_tenant_shelter_cost_30pct_or_more",
    1493: "published_pct_tenant_core_housing_need",
    1494: "median_monthly_shelter_costs_rented_dwellings",
    1495: "average_monthly_shelter_costs_rented_dwellings",
}

# These IDs are published percentages in the Census Profile.
# We keep the original 0-100 values and also create 0-1 proportion versions.
PUBLISHED_PERCENTAGE_IDS = {
    1483,
    1484,
    1485,
    1491,
    1492,
    1493,
}

SOURCE_HOUSING_TENURE_COSTS = (
    "Statistics Canada Census Profile, 2021 Census of Population; "
    "housing tenure, dwelling condition, shelter costs, housing indicators, "
    "core housing need, and owner/tenant housing costs, 25% sample data"
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
# Filter to census tract housing tenure/cost rows
# -----------------------------

df["CHARACTERISTIC_ID"] = pd.to_numeric(
    df["CHARACTERISTIC_ID"],
    errors="coerce",
).astype("Int64")

rows = df[
    (df["GEO_LEVEL"].astype(str).str.strip() == "Census tract")
    & (df["CHARACTERISTIC_ID"].isin(CHARACTERISTICS.keys()))
].copy()

print("\nHousing tenure/cost rows found:", len(rows))

if rows.empty:
    raise ValueError("No census tract housing tenure/cost rows found.")


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
# In this Census Profile file, the relevant numeric values are in C1_COUNT_TOTAL
# for counts, dollar values, and the published percentage rows.
#
# We therefore use C1_COUNT_TOTAL consistently.
# Published percentage rows remain 0-100 here; separate 0-1 columns are created
# after pivoting.

rows["value"] = pd.to_numeric(
    rows["C1_COUNT_TOTAL"],
    errors="coerce",
)


# -----------------------------
# Report missing / suppressed values
# -----------------------------

missing = rows[rows["value"].isna()].copy()

print("\nMissing or non-numeric housing tenure/cost values:", len(missing))

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

for_column_names = list(CHARACTERISTICS.values())

for column_name in for_column_names:
    if column_name not in clean.columns:
        clean[column_name] = np.nan


# -----------------------------
# Compute tenure and condominium proportions
# -----------------------------

clean["pct_owner"] = bounded_proportion(
    clean["owner_households"],
    clean["tenure_total_households"],
)

clean["pct_renter"] = bounded_proportion(
    clean["renter_households"],
    clean["tenure_total_households"],
)

clean["pct_government_or_band_provided_dwelling"] = bounded_proportion(
    clean["government_or_band_provided_dwelling_households"],
    clean["tenure_total_households"],
)

clean["pct_condominium"] = bounded_proportion(
    clean["condominium_dwellings"],
    clean["condominium_status_total_dwellings"],
)

clean["pct_not_condominium"] = bounded_proportion(
    clean["not_condominium_dwellings"],
    clean["condominium_status_total_dwellings"],
)


# -----------------------------
# Compute dwelling condition proportions
# -----------------------------

clean["pct_regular_maintenance_or_minor_repairs"] = bounded_proportion(
    clean["regular_maintenance_or_minor_repairs_dwellings"],
    clean["dwelling_condition_total"],
)

clean["pct_major_repairs_needed"] = bounded_proportion(
    clean["major_repairs_needed_dwellings"],
    clean["dwelling_condition_total"],
)


# -----------------------------
# Compute shelter-cost burden proportions
# -----------------------------

clean["pct_shelter_cost_less_than_30pct"] = bounded_proportion(
    clean["shelter_cost_less_than_30pct_households"],
    clean["shelter_cost_income_ratio_total_households"],
)

clean["pct_shelter_cost_30pct_or_more"] = bounded_proportion(
    clean["shelter_cost_30pct_or_more_households"],
    clean["shelter_cost_income_ratio_total_households"],
)

clean["pct_shelter_cost_30pct_to_less_than_100pct"] = bounded_proportion(
    clean["shelter_cost_30pct_to_less_than_100pct_households"],
    clean["shelter_cost_income_ratio_total_households"],
)


# -----------------------------
# Compute housing-indicator proportions
# -----------------------------

clean["pct_housing_indicator_problem_any"] = bounded_proportion(
    clean["housing_indicator_problem_any"],
    clean["housing_indicators_total_dwellings"],
)

clean["pct_housing_indicator_cost_burden_only"] = bounded_proportion(
    clean["housing_indicator_cost_burden_only"],
    clean["housing_indicators_total_dwellings"],
)

clean["pct_housing_indicator_not_suitable_only"] = bounded_proportion(
    clean["housing_indicator_not_suitable_only"],
    clean["housing_indicators_total_dwellings"],
)

clean["pct_housing_indicator_major_repairs_only"] = bounded_proportion(
    clean["housing_indicator_major_repairs_only"],
    clean["housing_indicators_total_dwellings"],
)

clean["pct_housing_indicator_cost_burden_and_not_suitable"] = bounded_proportion(
    clean["housing_indicator_cost_burden_and_not_suitable"],
    clean["housing_indicators_total_dwellings"],
)

clean["pct_housing_indicator_cost_burden_and_major_repairs"] = bounded_proportion(
    clean["housing_indicator_cost_burden_and_major_repairs"],
    clean["housing_indicators_total_dwellings"],
)

clean["pct_housing_indicator_not_suitable_and_major_repairs"] = bounded_proportion(
    clean["housing_indicator_not_suitable_and_major_repairs"],
    clean["housing_indicators_total_dwellings"],
)

clean["pct_housing_indicator_all_three"] = bounded_proportion(
    clean["housing_indicator_all_three"],
    clean["housing_indicators_total_dwellings"],
)

clean["pct_acceptable_housing"] = bounded_proportion(
    clean["acceptable_housing"],
    clean["housing_indicators_total_dwellings"],
)


# -----------------------------
# Compute core housing need proportions
# -----------------------------

clean["pct_core_housing_need"] = bounded_proportion(
    clean["core_housing_need_households"],
    clean["core_housing_need_total_households"],
)

clean["pct_not_core_housing_need"] = bounded_proportion(
    clean["not_core_housing_need_households"],
    clean["core_housing_need_total_households"],
)


# -----------------------------
# Convert published percentage fields to 0-1 proportions
# -----------------------------

clean["pct_owner_households_with_mortgage_published"] = published_percent_to_proportion(
    clean["published_pct_owner_households_with_mortgage"]
)

clean["pct_owner_shelter_cost_30pct_or_more_published"] = published_percent_to_proportion(
    clean["published_pct_owner_shelter_cost_30pct_or_more"]
)

clean["pct_owner_core_housing_need_published"] = published_percent_to_proportion(
    clean["published_pct_owner_core_housing_need"]
)

clean["pct_tenant_subsidized_housing_published"] = published_percent_to_proportion(
    clean["published_pct_tenant_subsidized_housing"]
)

clean["pct_tenant_shelter_cost_30pct_or_more_published"] = published_percent_to_proportion(
    clean["published_pct_tenant_shelter_cost_30pct_or_more"]
)

clean["pct_tenant_core_housing_need_published"] = published_percent_to_proportion(
    clean["published_pct_tenant_core_housing_need"]
)


# -----------------------------
# Add default SoVI/HGNN-style fields
# -----------------------------

clean["renter_measure_default"] = clean["pct_renter"].astype(float)

clean["renter_measure_default_description"] = (
    "pct_renter; share of private households that are renters"
)

clean["housing_cost_burden_measure_default"] = clean[
    "pct_shelter_cost_30pct_or_more"
].astype(float)

clean["housing_cost_burden_measure_default_description"] = (
    "pct_shelter_cost_30pct_or_more; share of owner/tenant households spending 30%+ of income on shelter"
)

clean["major_repairs_measure_default"] = clean["pct_major_repairs_needed"].astype(float)

clean["major_repairs_measure_default_description"] = (
    "pct_major_repairs_needed; share of occupied private dwellings needing major repairs"
)

clean["core_housing_need_measure_default"] = clean["pct_core_housing_need"].astype(float)

clean["core_housing_need_measure_default_description"] = (
    "pct_core_housing_need; share of owner/tenant households in core housing need"
)

clean["acceptable_housing_measure_default"] = clean["pct_acceptable_housing"].astype(float)

clean["acceptable_housing_measure_default_description"] = (
    "pct_acceptable_housing; share of occupied private dwellings classified as acceptable housing"
)


# -----------------------------
# Add metadata
# -----------------------------

clean["unit_type"] = "census_tract"
clean["census_year"] = 2021
clean["source_housing_tenure_costs"] = SOURCE_HOUSING_TENURE_COSTS


# -----------------------------
# Reorder columns
# -----------------------------

ordered_columns = [
    "statcan_dguid",
    "geo_name",
    "unit_type",
    "census_year",

    # Tenure
    "tenure_total_households",
    "owner_households",
    "renter_households",
    "government_or_band_provided_dwelling_households",
    "pct_owner",
    "pct_renter",
    "pct_government_or_band_provided_dwelling",

    # Condominium status
    "condominium_status_total_dwellings",
    "condominium_dwellings",
    "not_condominium_dwellings",
    "pct_condominium",
    "pct_not_condominium",

    # Dwelling condition
    "dwelling_condition_total",
    "regular_maintenance_or_minor_repairs_dwellings",
    "major_repairs_needed_dwellings",
    "pct_regular_maintenance_or_minor_repairs",
    "pct_major_repairs_needed",

    # Shelter cost burden
    "shelter_cost_income_ratio_total_households",
    "shelter_cost_less_than_30pct_households",
    "shelter_cost_30pct_or_more_households",
    "shelter_cost_30pct_to_less_than_100pct_households",
    "pct_shelter_cost_less_than_30pct",
    "pct_shelter_cost_30pct_or_more",
    "pct_shelter_cost_30pct_to_less_than_100pct",

    # Housing indicators
    "housing_indicators_total_dwellings",
    "housing_indicator_problem_any",
    "housing_indicator_cost_burden_only",
    "housing_indicator_not_suitable_only",
    "housing_indicator_major_repairs_only",
    "housing_indicator_cost_burden_and_not_suitable",
    "housing_indicator_cost_burden_and_major_repairs",
    "housing_indicator_not_suitable_and_major_repairs",
    "housing_indicator_all_three",
    "acceptable_housing",
    "pct_housing_indicator_problem_any",
    "pct_housing_indicator_cost_burden_only",
    "pct_housing_indicator_not_suitable_only",
    "pct_housing_indicator_major_repairs_only",
    "pct_housing_indicator_cost_burden_and_not_suitable",
    "pct_housing_indicator_cost_burden_and_major_repairs",
    "pct_housing_indicator_not_suitable_and_major_repairs",
    "pct_housing_indicator_all_three",
    "pct_acceptable_housing",

    # Core housing need
    "core_housing_need_total_households",
    "core_housing_need_households",
    "not_core_housing_need_households",
    "pct_core_housing_need",
    "pct_not_core_housing_need",

    # Owner costs / values
    "owner_households_nonfarm_nonreserve_total",
    "published_pct_owner_households_with_mortgage",
    "pct_owner_households_with_mortgage_published",
    "published_pct_owner_shelter_cost_30pct_or_more",
    "pct_owner_shelter_cost_30pct_or_more_published",
    "published_pct_owner_core_housing_need",
    "pct_owner_core_housing_need_published",
    "median_monthly_shelter_costs_owned_dwellings",
    "average_monthly_shelter_costs_owned_dwellings",
    "median_value_owned_dwellings",
    "average_value_owned_dwellings",

    # Tenant costs
    "tenant_households_nonfarm_nonreserve_total",
    "published_pct_tenant_subsidized_housing",
    "pct_tenant_subsidized_housing_published",
    "published_pct_tenant_shelter_cost_30pct_or_more",
    "pct_tenant_shelter_cost_30pct_or_more_published",
    "published_pct_tenant_core_housing_need",
    "pct_tenant_core_housing_need_published",
    "median_monthly_shelter_costs_rented_dwellings",
    "average_monthly_shelter_costs_rented_dwellings",

    # Defaults
    "renter_measure_default",
    "renter_measure_default_description",
    "housing_cost_burden_measure_default",
    "housing_cost_burden_measure_default_description",
    "major_repairs_measure_default",
    "major_repairs_measure_default_description",
    "core_housing_need_measure_default",
    "core_housing_need_measure_default_description",
    "acceptable_housing_measure_default",
    "acceptable_housing_measure_default_description",

    "source_housing_tenure_costs",
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
    "pct_owner",
    "pct_renter",
    "pct_government_or_band_provided_dwelling",
    "pct_condominium",
    "pct_not_condominium",
    "pct_regular_maintenance_or_minor_repairs",
    "pct_major_repairs_needed",
    "pct_shelter_cost_less_than_30pct",
    "pct_shelter_cost_30pct_or_more",
    "pct_shelter_cost_30pct_to_less_than_100pct",
    "pct_housing_indicator_problem_any",
    "pct_housing_indicator_cost_burden_only",
    "pct_housing_indicator_not_suitable_only",
    "pct_housing_indicator_major_repairs_only",
    "pct_housing_indicator_cost_burden_and_not_suitable",
    "pct_housing_indicator_cost_burden_and_major_repairs",
    "pct_housing_indicator_not_suitable_and_major_repairs",
    "pct_housing_indicator_all_three",
    "pct_acceptable_housing",
    "pct_core_housing_need",
    "pct_not_core_housing_need",
    "pct_owner_households_with_mortgage_published",
    "pct_owner_shelter_cost_30pct_or_more_published",
    "pct_owner_core_housing_need_published",
    "pct_tenant_subsidized_housing_published",
    "pct_tenant_shelter_cost_30pct_or_more_published",
    "pct_tenant_core_housing_need_published",
    "renter_measure_default",
    "housing_cost_burden_measure_default",
    "major_repairs_measure_default",
    "core_housing_need_measure_default",
    "acceptable_housing_measure_default",
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


# Compare computed shelter-cost burden with the implied published-style value.
# There is no separate published all-household % field in this block; 1467 is
# a count, not a published percent. This check simply confirms the computed
# measure has realistic values.
if clean["housing_cost_burden_measure_default"].mean(skipna=True) > 0.8:
    raise ValueError(
        "Housing cost burden mean is suspiciously high. "
        "This may indicate a count/percentage mapping error."
    )

if clean["major_repairs_measure_default"].mean(skipna=True) > 0.5:
    raise ValueError(
        "Major repairs mean is suspiciously high. "
        "This may indicate a wrong dwelling-condition denominator."
    )

print("\nClean housing tenure/costs table")
print("Rows:", len(clean))
print("Columns:", list(clean.columns))

print("\nMissing values by column:")
print(clean.isna().sum())

print("\nDefault renter measure summary:")
print(clean["renter_measure_default"].describe())

print("\nDefault housing cost-burden measure summary:")
print(clean["housing_cost_burden_measure_default"].describe())

print("\nDefault major-repairs measure summary:")
print(clean["major_repairs_measure_default"].describe())

print("\nDefault core-housing-need measure summary:")
print(clean["core_housing_need_measure_default"].describe())

print("\nAcceptable housing measure summary:")
print(clean["acceptable_housing_measure_default"].describe())

print("\nPublished tenant shelter-cost burden proportion summary:")
print(clean["pct_tenant_shelter_cost_30pct_or_more_published"].describe())

print("\nPublished owner shelter-cost burden proportion summary:")
print(clean["pct_owner_shelter_cost_30pct_or_more_published"].describe())

print("\nMedian monthly rented-dwelling shelter cost summary:")
print(clean["median_monthly_shelter_costs_rented_dwellings"].describe())

print("\nMedian value of owned dwellings summary:")
print(clean["median_value_owned_dwellings"].describe())

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