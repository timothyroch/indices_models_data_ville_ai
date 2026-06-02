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

OUTPUT_CSV = OUTPUT_DIR / "clean_census_tract_housing_type_2021.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_census_tract_housing_type_2021.parquet"


# -----------------------------
# Characteristic IDs
# -----------------------------
# From audited Census Profile structural dwelling type block:
#
# 41 = Total - Occupied private dwellings by structural type of dwelling - 100% data
# 42 = Single-detached house
# 43 = Semi-detached house
# 44 = Row house
# 45 = Apartment or flat in a duplex
# 46 = Apartment in a building that has fewer than five storeys
# 47 = Apartment in a building that has five or more storeys
# 48 = Other single-attached house
# 49 = Movable dwelling
#
# Notes:
# - ID 48 was added after local block audit.
# - It is a real structural-dwelling category and is now preserved explicitly.
# - It is not included in the default apartment/multi-unit proxy.
# - It is implicitly included in pct_non_single_detached because that field is:
#       total occupied private dwellings - single-detached dwellings
#
# For SVI-like construction, the default proxies remain:
# - multi-unit structures: apartment/duplex + apartment <5 storeys + apartment 5+ storeys
# - mobile homes: movable dwellings

CHARACTERISTICS = {
    41: "occupied_private_dwellings_total_by_structure",
    42: "single_detached_house_dwellings",
    43: "semi_detached_house_dwellings",
    44: "row_house_dwellings",
    45: "apartment_or_flat_in_duplex_dwellings",
    46: "apartment_building_fewer_than_5_storeys_dwellings",
    47: "apartment_building_5_or_more_storeys_dwellings",
    48: "other_single_attached_house_dwellings",
    49: "movable_dwelling_dwellings",
}

SOURCE_HOUSING_TYPE = (
    "Statistics Canada Census Profile, 2021 Census of Population; "
    "occupied private dwellings by structural type of dwelling, 100% data"
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

    This is useful for derived proportions from independently rounded census
    categories. Raw counts are not changed. Only the derived proportion is
    bounded to avoid impossible values such as -0.01 or 1.02.
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
# Filter to census tract housing-type rows
# -----------------------------

df["CHARACTERISTIC_ID"] = pd.to_numeric(
    df["CHARACTERISTIC_ID"],
    errors="coerce",
).astype("Int64")

housing_rows = df[
    (df["GEO_LEVEL"].astype(str).str.strip() == "Census tract")
    & (df["CHARACTERISTIC_ID"].isin(CHARACTERISTICS.keys()))
].copy()

print("\nHousing-type rows found:", len(housing_rows))

if housing_rows.empty:
    raise ValueError("No census tract housing-type rows found.")


# -----------------------------
# Inspect selected characteristics
# -----------------------------

selected_characteristics = (
    housing_rows[
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

found_ids = set(housing_rows["CHARACTERISTIC_ID"].dropna().astype(int).unique())
expected_ids = set(CHARACTERISTICS.keys())
missing_ids = sorted(expected_ids - found_ids)

if missing_ids:
    print("\nWarning: some expected housing-type characteristic IDs were not found:")
    for characteristic_id in missing_ids:
        print(f"  {characteristic_id}: {CHARACTERISTICS[characteristic_id]}")


# -----------------------------
# Prepare value column
# -----------------------------

housing_rows["value"] = pd.to_numeric(
    housing_rows["C1_COUNT_TOTAL"],
    errors="coerce",
)


# -----------------------------
# Report missing / suppressed values
# -----------------------------

missing = housing_rows[housing_rows["value"].isna()].copy()

print("\nMissing or non-numeric housing-type values:", len(missing))

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

housing_rows["feature_name"] = housing_rows["CHARACTERISTIC_ID"].map(
    CHARACTERISTICS
)

wide = (
    housing_rows
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
# Compute reusable counts
# -----------------------------

clean["apartment_multiunit_dwellings"] = (
    clean["apartment_or_flat_in_duplex_dwellings"]
    + clean["apartment_building_fewer_than_5_storeys_dwellings"]
    + clean["apartment_building_5_or_more_storeys_dwellings"]
)

clean["single_attached_non_apartment_dwellings"] = (
    clean["semi_detached_house_dwellings"]
    + clean["row_house_dwellings"]
    + clean["other_single_attached_house_dwellings"]
)

clean["non_single_detached_dwellings"] = (
    clean["occupied_private_dwellings_total_by_structure"]
    - clean["single_detached_house_dwellings"]
)


# -----------------------------
# Compute reusable proportions
# -----------------------------

denominator = clean["occupied_private_dwellings_total_by_structure"]

clean["pct_single_detached_house"] = bounded_proportion(
    clean["single_detached_house_dwellings"],
    denominator,
)

clean["pct_semi_detached_house"] = bounded_proportion(
    clean["semi_detached_house_dwellings"],
    denominator,
)

clean["pct_row_house"] = bounded_proportion(
    clean["row_house_dwellings"],
    denominator,
)

clean["pct_apartment_or_flat_in_duplex"] = bounded_proportion(
    clean["apartment_or_flat_in_duplex_dwellings"],
    denominator,
)

clean["pct_apartment_building_fewer_than_5_storeys"] = bounded_proportion(
    clean["apartment_building_fewer_than_5_storeys_dwellings"],
    denominator,
)

clean["pct_apartment_building_5_or_more_storeys"] = bounded_proportion(
    clean["apartment_building_5_or_more_storeys_dwellings"],
    denominator,
)

clean["pct_other_single_attached_house"] = bounded_proportion(
    clean["other_single_attached_house_dwellings"],
    denominator,
)

clean["pct_movable_dwelling"] = bounded_proportion(
    clean["movable_dwelling_dwellings"],
    denominator,
)

# Default apartment/multi-unit proxy.
# This intentionally excludes semi-detached houses, row houses, and other
# single-attached houses unless an index-specific script chooses a broader
# built-form definition.
clean["pct_apartment_multiunit"] = bounded_proportion(
    clean["apartment_multiunit_dwellings"],
    denominator,
)

# Useful built-form sensitivity proxy.
clean["pct_single_attached_non_apartment"] = bounded_proportion(
    clean["single_attached_non_apartment_dwellings"],
    denominator,
)

# Broader non-single-detached proxy.
clean["pct_non_single_detached"] = bounded_proportion(
    clean["non_single_detached_dwellings"],
    denominator,
)


# -----------------------------
# Add default SVI-like fields
# -----------------------------

clean["multiunit_measure_default"] = clean["pct_apartment_multiunit"].astype(float)

clean["multiunit_measure_default_description"] = (
    "pct_apartment_multiunit; apartment/duplex-based Canadian proxy for SVI multi-unit structures"
)

clean["mobile_home_measure_default"] = clean["pct_movable_dwelling"].astype(float)

clean["mobile_home_measure_default_description"] = (
    "pct_movable_dwelling; Canadian Census structural dwelling proxy for mobile/movable homes"
)


# -----------------------------
# Add metadata
# -----------------------------

clean["unit_type"] = "census_tract"
clean["census_year"] = 2021
clean["source_housing_type"] = SOURCE_HOUSING_TYPE


# -----------------------------
# Reorder columns
# -----------------------------

ordered_columns = [
    "statcan_dguid",
    "geo_name",
    "unit_type",
    "census_year",

    "occupied_private_dwellings_total_by_structure",

    "single_detached_house_dwellings",
    "semi_detached_house_dwellings",
    "row_house_dwellings",
    "apartment_or_flat_in_duplex_dwellings",
    "apartment_building_fewer_than_5_storeys_dwellings",
    "apartment_building_5_or_more_storeys_dwellings",
    "other_single_attached_house_dwellings",
    "movable_dwelling_dwellings",

    "pct_single_detached_house",
    "pct_semi_detached_house",
    "pct_row_house",
    "pct_apartment_or_flat_in_duplex",
    "pct_apartment_building_fewer_than_5_storeys",
    "pct_apartment_building_5_or_more_storeys",
    "pct_other_single_attached_house",
    "pct_movable_dwelling",

    "apartment_multiunit_dwellings",
    "pct_apartment_multiunit",
    "single_attached_non_apartment_dwellings",
    "pct_single_attached_non_apartment",
    "non_single_detached_dwellings",
    "pct_non_single_detached",

    "multiunit_measure_default",
    "multiunit_measure_default_description",
    "mobile_home_measure_default",
    "mobile_home_measure_default_description",

    "source_housing_type",
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
    "pct_single_detached_house",
    "pct_semi_detached_house",
    "pct_row_house",
    "pct_apartment_or_flat_in_duplex",
    "pct_apartment_building_fewer_than_5_storeys",
    "pct_apartment_building_5_or_more_storeys",
    "pct_other_single_attached_house",
    "pct_movable_dwelling",
    "pct_apartment_multiunit",
    "pct_single_attached_non_apartment",
    "pct_non_single_detached",
    "multiunit_measure_default",
    "mobile_home_measure_default",
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

print("\nClean housing-type table")
print("Rows:", len(clean))
print("Columns:", list(clean.columns))

print("\nMissing values by column:")
print(clean.isna().sum())

print("\nDefault multiunit measure summary:")
print(clean["multiunit_measure_default"].describe())

print("\nDefault mobile/movable dwelling measure summary:")
print(clean["mobile_home_measure_default"].describe())

print("\nOther single-attached house proportion summary:")
print(clean["pct_other_single_attached_house"].describe())

print("\nSingle-attached non-apartment proportion summary:")
print(clean["pct_single_attached_non_apartment"].describe())

print("\nNon-single-detached proportion summary:")
print(clean["pct_non_single_detached"].describe())

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