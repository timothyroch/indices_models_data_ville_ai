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

OUTPUT_CSV = OUTPUT_DIR / "clean_census_tract_household_family_2021.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_census_tract_household_family_2021.parquet"


# -----------------------------
# Characteristic IDs
# -----------------------------

CHARACTERISTICS = {
    # Household size
    50: "private_households_total_by_size",
    56: "persons_in_private_households",

    # Census families
    71: "census_families_total_by_family_size",
    76: "average_size_census_families",
    77: "average_children_in_census_families_with_children",
    78: "census_families_total",
    79: "couple_families_total",
    86: "one_parent_families_total",

    # Persons in families
    89: "persons_private_households_total",
    90: "persons_in_census_families",
    92: "parents_in_one_parent_families",
    95: "persons_in_one_parent_family",
    96: "persons_not_in_census_families",
    97: "persons_living_alone",

    # Household type
    100: "household_type_total",
    105: "one_parent_family_households",
    106: "multigenerational_households",
    107: "multiple_census_family_households",
    110: "one_person_households",

    # One-parent economic-family income and size
    313: "one_parent_economic_families_income_total_100pct_denominator",
    314: "median_total_income_one_parent_economic_families_2020",
    315: "median_after_tax_income_one_parent_economic_families_2020",
    316: "average_size_one_parent_economic_families",
    329: "one_parent_economic_families_income_total_25pct_denominator",
    330: "average_total_income_one_parent_economic_families_2020",
    331: "average_after_tax_income_one_parent_economic_families_2020",
}

SOURCE_HOUSEHOLD_FAMILY = (
    "Statistics Canada Census Profile, 2021 Census of Population; "
    "household and family composition"
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
# Filter to census tract household/family rows
# -----------------------------

df["CHARACTERISTIC_ID"] = pd.to_numeric(
    df["CHARACTERISTIC_ID"],
    errors="coerce",
).astype("Int64")

household_rows = df[
    (df["GEO_LEVEL"].astype(str).str.strip() == "Census tract")
    & (df["CHARACTERISTIC_ID"].isin(CHARACTERISTICS.keys()))
].copy()

print("\nHousehold/family rows found:", len(household_rows))

if household_rows.empty:
    raise ValueError("No census tract household/family rows found.")


# -----------------------------
# Inspect selected characteristics
# -----------------------------

selected_characteristics = (
    household_rows[
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

found_ids = set(household_rows["CHARACTERISTIC_ID"].dropna().astype(int).unique())
expected_ids = set(CHARACTERISTICS.keys())
missing_ids = sorted(expected_ids - found_ids)

if missing_ids:
    print("\nWarning: some expected household/family characteristic IDs were not found:")
    for characteristic_id in missing_ids:
        print(f"  {characteristic_id}: {CHARACTERISTICS[characteristic_id]}")


# -----------------------------
# Prepare value column
# -----------------------------

household_rows["value"] = pd.to_numeric(
    household_rows["C1_COUNT_TOTAL"],
    errors="coerce",
)


# -----------------------------
# Report missing / suppressed values
# -----------------------------

missing = household_rows[household_rows["value"].isna()].copy()

print("\nMissing or non-numeric household/family values:", len(missing))

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

household_rows["feature_name"] = household_rows["CHARACTERISTIC_ID"].map(
    CHARACTERISTICS
)

wide = (
    household_rows
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
# Compute reusable measures safely
# -----------------------------

# This is not a proportion. Values greater than 1 are normal.
clean["average_private_household_size_computed"] = safe_divide(
    clean["persons_in_private_households"],
    clean["private_households_total_by_size"],
)

# These are proportions between 0 and 1.
clean["pct_one_parent_families_among_census_families"] = bounded_proportion(
    clean["one_parent_families_total"],
    clean["census_families_total"],
)

clean["pct_couple_families_among_census_families"] = bounded_proportion(
    clean["couple_families_total"],
    clean["census_families_total"],
)

clean["pct_one_parent_family_households"] = bounded_proportion(
    clean["one_parent_family_households"],
    clean["household_type_total"],
)

clean["pct_multigenerational_households"] = bounded_proportion(
    clean["multigenerational_households"],
    clean["household_type_total"],
)

clean["pct_multiple_census_family_households"] = bounded_proportion(
    clean["multiple_census_family_households"],
    clean["household_type_total"],
)

clean["pct_one_person_households"] = bounded_proportion(
    clean["one_person_households"],
    clean["household_type_total"],
)

clean["pct_persons_in_one_parent_family"] = bounded_proportion(
    clean["persons_in_one_parent_family"],
    clean["persons_private_households_total"],
)

clean["pct_persons_not_in_census_families"] = bounded_proportion(
    clean["persons_not_in_census_families"],
    clean["persons_private_households_total"],
)

clean["pct_persons_in_census_families"] = bounded_proportion(
    clean["persons_in_census_families"],
    clean["persons_private_households_total"],
)

clean["pct_persons_living_alone"] = bounded_proportion(
    clean["persons_living_alone"],
    clean["persons_private_households_total"],
)


# -----------------------------
# Add default SVI-like household/family field
# -----------------------------

clean["single_parent_measure_default"] = clean[
    "pct_one_parent_family_households"
].astype(float)

clean["single_parent_measure_default_description"] = (
    "pct_one_parent_family_households; proxy for SVI single-parent household variable"
)


# -----------------------------
# Add HGNN / SoVI enrichment fields
# -----------------------------

clean["living_alone_measure_default"] = clean[
    "pct_persons_living_alone"
].astype(float)

clean["living_alone_measure_default_description"] = (
    "pct_persons_living_alone; share of persons in private households living alone"
)

clean["one_person_household_measure_default"] = clean[
    "pct_one_person_households"
].astype(float)

clean["one_person_household_measure_default_description"] = (
    "pct_one_person_households; share of household types that are one-person households"
)

clean["multigenerational_household_measure_default"] = clean[
    "pct_multigenerational_households"
].astype(float)

clean["multigenerational_household_measure_default_description"] = (
    "pct_multigenerational_households; share of household types that are multigenerational households"
)

clean["multiple_census_family_household_measure_default"] = clean[
    "pct_multiple_census_family_households"
].astype(float)

clean["multiple_census_family_household_measure_default_description"] = (
    "pct_multiple_census_family_households; share of household types that are multiple-census-family households"
)


# -----------------------------
# Add metadata
# -----------------------------

clean["unit_type"] = "census_tract"
clean["census_year"] = 2021
clean["income_reference_year"] = 2020
clean["source_household_family"] = SOURCE_HOUSEHOLD_FAMILY


# -----------------------------
# Reorder columns
# -----------------------------

ordered_columns = [
    "statcan_dguid",
    "geo_name",
    "unit_type",
    "census_year",
    "income_reference_year",

    # Household size
    "private_households_total_by_size",
    "persons_in_private_households",
    "average_private_household_size_computed",

    # Census families
    "census_families_total_by_family_size",
    "average_size_census_families",
    "average_children_in_census_families_with_children",
    "census_families_total",
    "couple_families_total",
    "one_parent_families_total",
    "pct_one_parent_families_among_census_families",
    "pct_couple_families_among_census_families",

    # Persons in families / not in families
    "persons_private_households_total",
    "persons_in_census_families",
    "parents_in_one_parent_families",
    "persons_in_one_parent_family",
    "persons_not_in_census_families",
    "persons_living_alone",
    "pct_persons_in_one_parent_family",
    "pct_persons_not_in_census_families",
    "pct_persons_in_census_families",
    "pct_persons_living_alone",

    # Household type
    "household_type_total",
    "one_parent_family_households",
    "multigenerational_households",
    "multiple_census_family_households",
    "one_person_households",
    "pct_one_parent_family_households",
    "pct_multigenerational_households",
    "pct_multiple_census_family_households",
    "pct_one_person_households",

    # One-parent economic-family income
    "one_parent_economic_families_income_total_100pct_denominator",
    "median_total_income_one_parent_economic_families_2020",
    "median_after_tax_income_one_parent_economic_families_2020",
    "average_size_one_parent_economic_families",
    "one_parent_economic_families_income_total_25pct_denominator",
    "average_total_income_one_parent_economic_families_2020",
    "average_after_tax_income_one_parent_economic_families_2020",

    # Defaults / named proxy fields
    "single_parent_measure_default",
    "single_parent_measure_default_description",
    "living_alone_measure_default",
    "living_alone_measure_default_description",
    "one_person_household_measure_default",
    "one_person_household_measure_default_description",
    "multigenerational_household_measure_default",
    "multigenerational_household_measure_default_description",
    "multiple_census_family_household_measure_default",
    "multiple_census_family_household_measure_default_description",

    "source_household_family",
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
    "pct_one_parent_families_among_census_families",
    "pct_couple_families_among_census_families",
    "pct_one_parent_family_households",
    "pct_multigenerational_households",
    "pct_multiple_census_family_households",
    "pct_one_person_households",
    "pct_persons_in_one_parent_family",
    "pct_persons_not_in_census_families",
    "pct_persons_in_census_families",
    "pct_persons_living_alone",
    "single_parent_measure_default",
    "living_alone_measure_default",
    "one_person_household_measure_default",
    "multigenerational_household_measure_default",
    "multiple_census_family_household_measure_default",
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

# Average household size is not a proportion. It can be greater than 1.
clean["average_private_household_size_computed"] = pd.to_numeric(
    clean["average_private_household_size_computed"],
    errors="coerce",
).astype(float)

if np.isinf(clean["average_private_household_size_computed"]).any():
    raise ValueError("Infinite values found in average_private_household_size_computed")

if clean["average_private_household_size_computed"].min(skipna=True) < 0:
    raise ValueError("Negative values found in average_private_household_size_computed")


print("\nClean household/family table")
print("Rows:", len(clean))
print("Columns:", list(clean.columns))

print("\nMissing values by column:")
print(clean.isna().sum())

print("\nDefault single-parent measure summary:")
print(clean["single_parent_measure_default"].describe())

print("\nLiving alone measure summary:")
print(clean["living_alone_measure_default"].describe())

print("\nOne-person household measure summary:")
print(clean["one_person_household_measure_default"].describe())

print("\nMultigenerational household measure summary:")
print(clean["multigenerational_household_measure_default"].describe())

print("\nMultiple-census-family household measure summary:")
print(clean["multiple_census_family_household_measure_default"].describe())

print("\nOne-parent families among census families summary:")
print(clean["pct_one_parent_families_among_census_families"].describe())

print("\nAverage private household size summary:")
print(clean["average_private_household_size_computed"].describe())

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