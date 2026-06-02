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

OUTPUT_CSV = OUTPUT_DIR / "clean_census_tract_immigration_ethnocultural_2021.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_census_tract_immigration_ethnocultural_2021.parquet"


# -----------------------------
# Characteristic IDs
# -----------------------------
# Corrected and enriched from targeted Census Profile inspection.
#
# Indigenous identity:
# 1402 = Total - Indigenous identity for the population in private households - 25% sample data
# 1403 = Indigenous identity
# 1410 = Non-Indigenous identity
#
# Citizenship:
# 1522 = Total - Citizenship for the population in private households - 25% sample data
# 1523 = Canadian citizens
# 1526 = Not Canadian citizens
#
# Immigrant status:
# 1527 = Total - Immigrant status and period of immigration for the population in private households - 25% sample data
# 1528 = Non-immigrants
# 1529 = Immigrants
# 1537 = Non-permanent residents
#
# Place of birth:
# 1544 = Total - Place of birth for the immigrant population in private households - 25% sample data
# 1604 = Total - Place of birth for the recent immigrant population in private households - 25% sample data
#
# Generation status:
# 1665 = Total - Generation status for the population in private households - 25% sample data
# 1666 = First generation
# 1667 = Second generation
# 1668 = Third generation or more
#
# Admission category:
# 1669 = Total - Admission category and applicant type for the immigrant population
#        in private households who were admitted between 1980 and 2021 - 25% sample data
# 1670 = Economic immigrants
# 1673 = Immigrants sponsored by family
# 1674 = Refugees
# 1675 = Other immigrants
#
# Important correction:
# - 1670, 1673, 1674, and 1675 should use 1669 as their denominator.
# - They should not use the full immigrant-status total 1527 as their primary denominator.
#
# Visible minority:
# 1683 = Total - Visible minority for the population in private households - 25% sample data
# 1684 = Total visible minority population
# 1695 = Visible minority, n.i.e.
# 1697 = Not a visible minority
#
# Ethnic or cultural origin:
# 1698 = Total - Ethnic or cultural origin for the population in private households - 25% sample data
#
# Notes:
# - These are area-level demographic/contextual variables.
# - They should not be interpreted as individual vulnerability labels.
# - For SVI-like construction, the likely default structural/demographic proxy is:
#       pct_visible_minority
#   but index-specific scripts may choose immigration, non-permanent resident,
#   citizenship, generation-status, or Indigenous-identity variants depending
#   on the theoretical framing.

CHARACTERISTICS = {
    # Indigenous identity
    1402: "indigenous_identity_total",
    1403: "indigenous_identity_population",
    1410: "non_indigenous_identity_population",

    # Citizenship
    1522: "citizenship_total",
    1523: "canadian_citizens_population",
    1526: "not_canadian_citizens_population",

    # Immigrant status
    1527: "immigrant_status_total",
    1528: "non_immigrant_population",
    1529: "immigrant_population",
    1537: "non_permanent_resident_population",

    # Place of birth
    1544: "place_of_birth_immigrant_population_total",
    1604: "place_of_birth_recent_immigrant_population_total",

    # Generation status
    1665: "generation_status_total",
    1666: "first_generation_population",
    1667: "second_generation_population",
    1668: "third_generation_or_more_population",

    # Admission category
    1669: "admission_category_total_immigrants_admitted_1980_2021",
    1670: "economic_immigrants_population",
    1673: "family_sponsored_immigrants_population",
    1674: "refugees_population",
    1675: "other_immigrants_population",

    # Visible minority
    1683: "visible_minority_total",
    1684: "visible_minority_population",
    1695: "visible_minority_nie_population",
    1697: "not_visible_minority_population",

    # Ethnic or cultural origin
    1698: "ethnic_or_cultural_origin_total",
}

SOURCE_IMMIGRATION_ETHNOCULTURAL = (
    "Statistics Canada Census Profile, 2021 Census of Population; "
    "immigration, citizenship, Indigenous identity, generation status, "
    "admission category, visible minority, and ethnocultural variables, "
    "25% sample data"
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
# Filter to census tract rows
# -----------------------------

df["CHARACTERISTIC_ID"] = pd.to_numeric(
    df["CHARACTERISTIC_ID"],
    errors="coerce",
).astype("Int64")

rows = df[
    (df["GEO_LEVEL"].astype(str).str.strip() == "Census tract")
    & (df["CHARACTERISTIC_ID"].isin(CHARACTERISTICS.keys()))
].copy()

print("\nImmigration/ethnocultural rows found:", len(rows))

if rows.empty:
    raise ValueError("No census tract immigration/ethnocultural rows found.")


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

rows["value"] = pd.to_numeric(
    rows["C1_COUNT_TOTAL"],
    errors="coerce",
)


# -----------------------------
# Report missing / suppressed values
# -----------------------------

missing = rows[rows["value"].isna()].copy()

print("\nMissing or non-numeric immigration/ethnocultural values:", len(missing))

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
        .head(40)
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
# Compute reusable proportions
# -----------------------------

# Indigenous identity
clean["pct_indigenous_identity"] = bounded_proportion(
    clean["indigenous_identity_population"],
    clean["indigenous_identity_total"],
)

clean["pct_non_indigenous_identity"] = bounded_proportion(
    clean["non_indigenous_identity_population"],
    clean["indigenous_identity_total"],
)


# Citizenship
clean["pct_canadian_citizens"] = bounded_proportion(
    clean["canadian_citizens_population"],
    clean["citizenship_total"],
)

clean["pct_not_canadian_citizens"] = bounded_proportion(
    clean["not_canadian_citizens_population"],
    clean["citizenship_total"],
)


# Immigrant status
clean["pct_immigrant"] = bounded_proportion(
    clean["immigrant_population"],
    clean["immigrant_status_total"],
)

clean["pct_non_immigrant"] = bounded_proportion(
    clean["non_immigrant_population"],
    clean["immigrant_status_total"],
)

clean["pct_non_permanent_resident"] = bounded_proportion(
    clean["non_permanent_resident_population"],
    clean["immigrant_status_total"],
)


# Generation status
clean["pct_first_generation"] = bounded_proportion(
    clean["first_generation_population"],
    clean["generation_status_total"],
)

clean["pct_second_generation"] = bounded_proportion(
    clean["second_generation_population"],
    clean["generation_status_total"],
)

clean["pct_third_generation_or_more"] = bounded_proportion(
    clean["third_generation_or_more_population"],
    clean["generation_status_total"],
)


# Admission-category proportions
# Correct denominator: 1669 admission category total, not 1527 immigrant status total.
clean["pct_economic_immigrants_of_admission_category_total"] = bounded_proportion(
    clean["economic_immigrants_population"],
    clean["admission_category_total_immigrants_admitted_1980_2021"],
)

clean["pct_family_sponsored_immigrants_of_admission_category_total"] = bounded_proportion(
    clean["family_sponsored_immigrants_population"],
    clean["admission_category_total_immigrants_admitted_1980_2021"],
)

clean["pct_refugees_of_admission_category_total"] = bounded_proportion(
    clean["refugees_population"],
    clean["admission_category_total_immigrants_admitted_1980_2021"],
)

clean["pct_other_immigrants_of_admission_category_total"] = bounded_proportion(
    clean["other_immigrants_population"],
    clean["admission_category_total_immigrants_admitted_1980_2021"],
)

# Legacy-style comparison fields.
# These are kept only as contextual ratios, not recommended as primary
# admission-category fields, because their denominator is broader.
clean["pct_economic_immigrants_of_immigrant_status_total"] = bounded_proportion(
    clean["economic_immigrants_population"],
    clean["immigrant_status_total"],
)

clean["pct_family_sponsored_immigrants_of_immigrant_status_total"] = bounded_proportion(
    clean["family_sponsored_immigrants_population"],
    clean["immigrant_status_total"],
)

clean["pct_refugees_of_immigrant_status_total"] = bounded_proportion(
    clean["refugees_population"],
    clean["immigrant_status_total"],
)

clean["pct_other_immigrants_of_immigrant_status_total"] = bounded_proportion(
    clean["other_immigrants_population"],
    clean["immigrant_status_total"],
)


# Visible minority
clean["pct_visible_minority"] = bounded_proportion(
    clean["visible_minority_population"],
    clean["visible_minority_total"],
)

clean["pct_not_visible_minority"] = bounded_proportion(
    clean["not_visible_minority_population"],
    clean["visible_minority_total"],
)

clean["pct_visible_minority_nie"] = bounded_proportion(
    clean["visible_minority_nie_population"],
    clean["visible_minority_total"],
)


# -----------------------------
# Add default SVI-like variable
# -----------------------------
# The original SVI contains a minority-status variable. For the Canadian
# Census Profile adaptation, we keep a broader feature table and set the
# current default to pct_visible_minority.
#
# This is a structural/demographic area-level proxy and should be interpreted
# carefully in index-specific documentation.

clean["ethnocultural_measure_default"] = clean["pct_visible_minority"].astype(float)

clean["ethnocultural_measure_default_description"] = (
    "pct_visible_minority; area-level structural/demographic proxy for SVI/SoVI-style construction"
)


# -----------------------------
# Add additional named contextual fields
# -----------------------------

clean["immigrant_measure_default"] = clean["pct_immigrant"].astype(float)

clean["immigrant_measure_default_description"] = (
    "pct_immigrant; share of population in private households classified as immigrants"
)

clean["non_permanent_resident_measure_default"] = clean[
    "pct_non_permanent_resident"
].astype(float)

clean["non_permanent_resident_measure_default_description"] = (
    "pct_non_permanent_resident; share of population in private households classified as non-permanent residents"
)

clean["not_canadian_citizen_measure_default"] = clean[
    "pct_not_canadian_citizens"
].astype(float)

clean["not_canadian_citizen_measure_default_description"] = (
    "pct_not_canadian_citizens; share of population in private households that are not Canadian citizens"
)

clean["first_generation_measure_default"] = clean["pct_first_generation"].astype(float)

clean["first_generation_measure_default_description"] = (
    "pct_first_generation; share of population in private households classified as first generation"
)

clean["refugee_admission_category_measure_default"] = clean[
    "pct_refugees_of_admission_category_total"
].astype(float)

clean["refugee_admission_category_measure_default_description"] = (
    "pct_refugees_of_admission_category_total; share of immigrants admitted between 1980 and 2021 whose admission category is refugee"
)


# -----------------------------
# Add metadata
# -----------------------------

clean["unit_type"] = "census_tract"
clean["census_year"] = 2021
clean["source_immigration_ethnocultural"] = SOURCE_IMMIGRATION_ETHNOCULTURAL


# -----------------------------
# Reorder columns
# -----------------------------

ordered_columns = [
    "statcan_dguid",
    "geo_name",
    "unit_type",
    "census_year",

    # Indigenous identity
    "indigenous_identity_total",
    "indigenous_identity_population",
    "non_indigenous_identity_population",
    "pct_indigenous_identity",
    "pct_non_indigenous_identity",

    # Citizenship
    "citizenship_total",
    "canadian_citizens_population",
    "not_canadian_citizens_population",
    "pct_canadian_citizens",
    "pct_not_canadian_citizens",

    # Immigrant status
    "immigrant_status_total",
    "non_immigrant_population",
    "immigrant_population",
    "non_permanent_resident_population",
    "pct_non_immigrant",
    "pct_immigrant",
    "pct_non_permanent_resident",

    # Place of birth totals
    "place_of_birth_immigrant_population_total",
    "place_of_birth_recent_immigrant_population_total",

    # Generation status
    "generation_status_total",
    "first_generation_population",
    "second_generation_population",
    "third_generation_or_more_population",
    "pct_first_generation",
    "pct_second_generation",
    "pct_third_generation_or_more",

    # Admission category
    "admission_category_total_immigrants_admitted_1980_2021",
    "economic_immigrants_population",
    "family_sponsored_immigrants_population",
    "refugees_population",
    "other_immigrants_population",
    "pct_economic_immigrants_of_admission_category_total",
    "pct_family_sponsored_immigrants_of_admission_category_total",
    "pct_refugees_of_admission_category_total",
    "pct_other_immigrants_of_admission_category_total",

    # Legacy/contextual broad-denominator admission ratios
    "pct_economic_immigrants_of_immigrant_status_total",
    "pct_family_sponsored_immigrants_of_immigrant_status_total",
    "pct_refugees_of_immigrant_status_total",
    "pct_other_immigrants_of_immigrant_status_total",

    # Visible minority
    "visible_minority_total",
    "visible_minority_population",
    "visible_minority_nie_population",
    "not_visible_minority_population",
    "pct_visible_minority",
    "pct_visible_minority_nie",
    "pct_not_visible_minority",

    # Ethnic/cultural origin denominator
    "ethnic_or_cultural_origin_total",

    # Default and named contextual fields
    "ethnocultural_measure_default",
    "ethnocultural_measure_default_description",
    "immigrant_measure_default",
    "immigrant_measure_default_description",
    "non_permanent_resident_measure_default",
    "non_permanent_resident_measure_default_description",
    "not_canadian_citizen_measure_default",
    "not_canadian_citizen_measure_default_description",
    "first_generation_measure_default",
    "first_generation_measure_default_description",
    "refugee_admission_category_measure_default",
    "refugee_admission_category_measure_default_description",

    "source_immigration_ethnocultural",
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

proportion_cols = [
    "pct_indigenous_identity",
    "pct_non_indigenous_identity",
    "pct_canadian_citizens",
    "pct_not_canadian_citizens",
    "pct_non_immigrant",
    "pct_immigrant",
    "pct_non_permanent_resident",
    "pct_first_generation",
    "pct_second_generation",
    "pct_third_generation_or_more",
    "pct_economic_immigrants_of_admission_category_total",
    "pct_family_sponsored_immigrants_of_admission_category_total",
    "pct_refugees_of_admission_category_total",
    "pct_other_immigrants_of_admission_category_total",
    "pct_economic_immigrants_of_immigrant_status_total",
    "pct_family_sponsored_immigrants_of_immigrant_status_total",
    "pct_refugees_of_immigrant_status_total",
    "pct_other_immigrants_of_immigrant_status_total",
    "pct_visible_minority",
    "pct_visible_minority_nie",
    "pct_not_visible_minority",
    "ethnocultural_measure_default",
    "immigrant_measure_default",
    "non_permanent_resident_measure_default",
    "not_canadian_citizen_measure_default",
    "first_generation_measure_default",
    "refugee_admission_category_measure_default",
]

for col in proportion_cols:
    clean[col] = pd.to_numeric(clean[col], errors="coerce").astype(float)

    if np.isinf(clean[col]).any():
        raise ValueError(f"Infinite values found in computed proportion column: {col}")

    min_value = clean[col].min(skipna=True)
    max_value = clean[col].max(skipna=True)

    if min_value < 0 or max_value > 1:
        raise ValueError(
            f"Out-of-bounds values remain in {col}: "
            f"min={min_value}, max={max_value}"
        )


print("\nClean immigration/ethnocultural table")
print("Rows:", len(clean))
print("Columns:", list(clean.columns))

print("\nMissing values by column:")
print(clean.isna().sum())

print("\nDefault ethnocultural measure summary:")
print(clean["ethnocultural_measure_default"].describe())

print("\nVisible minority proportion summary:")
print(clean["pct_visible_minority"].describe())

print("\nImmigrant proportion summary:")
print(clean["pct_immigrant"].describe())

print("\nNon-permanent resident proportion summary:")
print(clean["pct_non_permanent_resident"].describe())

print("\nIndigenous identity proportion summary:")
print(clean["pct_indigenous_identity"].describe())

print("\nNot Canadian citizen measure summary:")
print(clean["not_canadian_citizen_measure_default"].describe())

print("\nFirst generation measure summary:")
print(clean["first_generation_measure_default"].describe())

print("\nRefugee admission-category measure summary:")
print(clean["refugee_admission_category_measure_default"].describe())

print("\nAdmission-category denominator summary:")
print(clean["admission_category_total_immigrants_admitted_1980_2021"].describe())

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