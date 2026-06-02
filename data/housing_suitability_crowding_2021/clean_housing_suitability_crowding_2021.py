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

OUTPUT_CSV = OUTPUT_DIR / "clean_census_tract_housing_suitability_crowding_2021.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_census_tract_housing_suitability_crowding_2021.parquet"


# -----------------------------
# Characteristic IDs
# -----------------------------
# From the Census Profile housing suitability / crowding block:
#
# 1434 = Total - Private households by number of persons per room - 25% sample data
# 1435 = One person or fewer per room
# 1436 = More than one person per room
#
# 1437 = Total - Private households by housing suitability - 25% sample data
# 1438 = Suitable
# 1439 = Not suitable
#
# Notes:
# - SVI uses a crowding variable.
# - Canadian Census Profile provides two useful proxies:
#       1. more than one person per room
#       2. housing not suitable
# - We keep both.
# - The default SVI-style crowding proxy is:
#       pct_more_than_one_person_per_room

CHARACTERISTICS = {
    # Persons per room
    1434: "persons_per_room_total_households",
    1435: "one_person_or_fewer_per_room_households",
    1436: "more_than_one_person_per_room_households",

    # Housing suitability
    1437: "housing_suitability_total_households",
    1438: "suitable_housing_households",
    1439: "not_suitable_housing_households",
}

SOURCE_HOUSING_SUITABILITY_CROWDING = (
    "Statistics Canada Census Profile, 2021 Census of Population; "
    "housing suitability and persons per room, 25% sample data"
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

print("\nHousing suitability / crowding rows found:", len(rows))

if rows.empty:
    raise ValueError("No census tract housing suitability / crowding rows found.")


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

print("\nMissing or non-numeric housing suitability / crowding values:", len(missing))

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
# Computed fields are proportions between 0 and 1.
# Raw counts are kept unchanged.
# Proportions are bounded to [0, 1] to avoid minor rounding artifacts.

clean["pct_one_person_or_fewer_per_room"] = bounded_proportion(
    clean["one_person_or_fewer_per_room_households"],
    clean["persons_per_room_total_households"],
)

clean["pct_more_than_one_person_per_room"] = bounded_proportion(
    clean["more_than_one_person_per_room_households"],
    clean["persons_per_room_total_households"],
)

clean["pct_suitable_housing"] = bounded_proportion(
    clean["suitable_housing_households"],
    clean["housing_suitability_total_households"],
)

clean["pct_not_suitable_housing"] = bounded_proportion(
    clean["not_suitable_housing_households"],
    clean["housing_suitability_total_households"],
)


# -----------------------------
# Add default SVI-like field
# -----------------------------
# The original SVI crowding variable is usually based on crowded housing.
# The strictest direct Canadian Census Profile proxy is:
#   more than one person per room
#
# Housing suitability is kept as a second important Canadian housing-stress
# variable because it reflects whether a dwelling has enough bedrooms for
# household size and composition.

clean["crowding_measure_default"] = clean[
    "pct_more_than_one_person_per_room"
].astype(float)

clean["crowding_measure_default_description"] = (
    "pct_more_than_one_person_per_room; Canadian Census proxy for SVI crowding"
)

clean["housing_suitability_measure_default"] = clean[
    "pct_not_suitable_housing"
].astype(float)

clean["housing_suitability_measure_default_description"] = (
    "pct_not_suitable_housing; Canadian Census housing suitability stress measure"
)


# -----------------------------
# Add metadata
# -----------------------------

clean["unit_type"] = "census_tract"
clean["census_year"] = 2021
clean["source_housing_suitability_crowding"] = SOURCE_HOUSING_SUITABILITY_CROWDING


# -----------------------------
# Reorder columns
# -----------------------------

ordered_columns = [
    "statcan_dguid",
    "geo_name",
    "unit_type",
    "census_year",

    "persons_per_room_total_households",
    "one_person_or_fewer_per_room_households",
    "more_than_one_person_per_room_households",
    "pct_one_person_or_fewer_per_room",
    "pct_more_than_one_person_per_room",

    "housing_suitability_total_households",
    "suitable_housing_households",
    "not_suitable_housing_households",
    "pct_suitable_housing",
    "pct_not_suitable_housing",

    "crowding_measure_default",
    "crowding_measure_default_description",
    "housing_suitability_measure_default",
    "housing_suitability_measure_default_description",

    "source_housing_suitability_crowding",
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
    "pct_one_person_or_fewer_per_room",
    "pct_more_than_one_person_per_room",
    "pct_suitable_housing",
    "pct_not_suitable_housing",
    "crowding_measure_default",
    "housing_suitability_measure_default",
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

print("\nClean housing suitability / crowding table")
print("Rows:", len(clean))
print("Columns:", list(clean.columns))

print("\nMissing values by column:")
print(clean.isna().sum())

print("\nDefault crowding measure summary:")
print(clean["crowding_measure_default"].describe())

print("\nHousing suitability stress measure summary:")
print(clean["housing_suitability_measure_default"].describe())

print("\nMore than one person per room summary:")
print(clean["pct_more_than_one_person_per_room"].describe())

print("\nNot suitable housing summary:")
print(clean["pct_not_suitable_housing"].describe())

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