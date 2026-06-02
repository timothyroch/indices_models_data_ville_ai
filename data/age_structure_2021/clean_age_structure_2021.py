from pathlib import Path
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

OUTPUT_CSV = OUTPUT_DIR / "clean_census_tract_age_structure_2021.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_census_tract_age_structure_2021.parquet"


# -----------------------------
# Characteristic IDs
# -----------------------------
# From the Census Profile characteristic discovery inventory:
#
# 8  = Total - Age groups of the population - 100% data
# 9  = 0 to 14 years
# 24 = 65 years and over
#
# These are 100% data variables, which is preferable for this feature family.

CHARACTERISTICS = {
    8: "age_total_population",
    9: "population_0_14",
    24: "population_65_plus",
}

SOURCE_AGE_STRUCTURE = (
    "Statistics Canada Census Profile, 2021 Census of Population; "
    "age groups of the population, 100% data"
)


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
# Filter to census tract age rows
# -----------------------------

df["CHARACTERISTIC_ID"] = pd.to_numeric(
    df["CHARACTERISTIC_ID"],
    errors="coerce",
).astype("Int64")

age_rows = df[
    (df["GEO_LEVEL"].astype(str).str.strip() == "Census tract")
    & (df["CHARACTERISTIC_ID"].isin(CHARACTERISTICS.keys()))
].copy()

print("\nAge-structure rows found:", len(age_rows))

if age_rows.empty:
    raise ValueError("No census tract age-structure rows found.")


# -----------------------------
# Inspect selected characteristics
# -----------------------------

selected_characteristics = (
    age_rows[
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
# Prepare value column
# -----------------------------

age_rows["value"] = pd.to_numeric(
    age_rows["C1_COUNT_TOTAL"],
    errors="coerce",
)


# -----------------------------
# Report missing / suppressed values
# -----------------------------

missing = age_rows[age_rows["value"].isna()].copy()

print("\nMissing or non-numeric age-structure values:", len(missing))

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
        .head(30)
        .to_string(index=False)
    )


# -----------------------------
# Pivot to one row per census tract
# -----------------------------

age_rows["feature_name"] = age_rows["CHARACTERISTIC_ID"].map(
    CHARACTERISTICS
)

wide = (
    age_rows
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
# Compute percentages
# -----------------------------
# These fields are proportions between 0 and 1.
# The original SVI uses age 17 or younger; the Census Profile gives 0 to 14
# directly here, so pct_age_0_14 is kept as the Canadian available proxy.

clean["pct_age_0_14"] = (
    clean["population_0_14"] / clean["age_total_population"]
)

clean["pct_age_65_plus"] = (
    clean["population_65_plus"] / clean["age_total_population"]
)

# Avoid infinite values if denominator is 0 or missing.
for col in ["pct_age_0_14", "pct_age_65_plus"]:
    clean.loc[~pd.notna(clean[col]), col] = pd.NA


# -----------------------------
# Add metadata
# -----------------------------

clean["unit_type"] = "census_tract"
clean["census_year"] = 2021
clean["source_age_structure"] = SOURCE_AGE_STRUCTURE


# -----------------------------
# Reorder columns
# -----------------------------

ordered_columns = [
    "statcan_dguid",
    "geo_name",
    "unit_type",
    "census_year",
    "age_total_population",
    "population_0_14",
    "pct_age_0_14",
    "population_65_plus",
    "pct_age_65_plus",
    "source_age_structure",
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

print("\nClean age-structure table")
print("Rows:", len(clean))
print("Columns:", list(clean.columns))

print("\nMissing values by column:")
print(clean.isna().sum())

print("\nAge 0-14 proportion summary:")
print(clean["pct_age_0_14"].describe())

print("\nAge 65+ proportion summary:")
print(clean["pct_age_65_plus"].describe())

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