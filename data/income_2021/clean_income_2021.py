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

OUTPUT_CSV = OUTPUT_DIR / "clean_census_tract_income_2021.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_census_tract_income_2021.parquet"


# -----------------------------
# Characteristic IDs
# -----------------------------
# Main individual income variables, 2020, population aged 15+
# 100% data:
#
# 111 = Total - Income statistics in 2020 for the population aged 15 years and over in private households - 100% data
# 112 = Number of total income recipients aged 15 years and over in private households in 2020 - 100% data
# 113 = Median total income in 2020 among recipients ($)
# 114 = Number of after-tax income recipients aged 15 years and over in private households in 2020 - 100% data
# 115 = Median after-tax income in 2020 among recipients ($)
# 116 = Number of market income recipients aged 15 years and over in private households in 2020 - 100% data
# 117 = Median market income in 2020 among recipients ($)
# 118 = Number of employment income recipients aged 15 years and over in private households in 2020 - 100% data
# 119 = Median employment income in 2020 among recipients ($)
#
# 25% sample data:
# 126 = Total - Income statistics in 2020 for the population aged 15 years and over in private households - 25% sample data
# 127 = Number of total income recipients aged 15 years and over in private households in 2020 - 25% sample data
# 128 = Average total income in 2020 among recipients ($)
# 129 = Number of after-tax income recipients aged 15 years and over in private households in 2020 - 25% sample data
# 130 = Average after-tax income in 2020 among recipients ($)
# 131 = Number of market income recipients aged 15 years and over in private households in 2020 - 25% sample data
# 132 = Average market income in 2020 among recipients ($)
# 133 = Number of employment income recipients aged 15 years and over in private households in 2020 - 25% sample data
# 134 = Average employment income in 2020 among recipients ($)
#
# Household income variables:
# 242 = Total - Income statistics for private households - 100% data
# 243 = Median total income of household in 2020 ($)
# 244 = Median after-tax income of household in 2020 ($)
# 251 = Total - Income statistics for private households - 25% sample data
# 252 = Average total income of household in 2020 ($)
# 253 = Average after-tax income of household in 2020 ($)

CHARACTERISTICS = {
    111: "individual_income_stats_total_15plus_100pct_denominator",
    112: "individual_total_income_recipients_15plus_2020_100pct",
    113: "median_total_income_15plus_2020",
    114: "individual_after_tax_income_recipients_15plus_2020_100pct",
    115: "median_after_tax_income_15plus_2020",
    116: "individual_market_income_recipients_15plus_2020_100pct",
    117: "median_market_income_15plus_2020",
    118: "individual_employment_income_recipients_15plus_2020_100pct",
    119: "median_employment_income_15plus_2020",

    126: "individual_income_stats_total_15plus_25pct_denominator",
    127: "individual_total_income_recipients_15plus_2020_25pct",
    128: "average_total_income_15plus_2020",
    129: "individual_after_tax_income_recipients_15plus_2020_25pct",
    130: "average_after_tax_income_15plus_2020",
    131: "individual_market_income_recipients_15plus_2020_25pct",
    132: "average_market_income_15plus_2020",
    133: "individual_employment_income_recipients_15plus_2020_25pct",
    134: "average_employment_income_15plus_2020",

    242: "household_income_stats_total_2020_100pct_denominator",
    243: "median_household_total_income_2020",
    244: "median_household_after_tax_income_2020",
    251: "household_income_stats_total_2020_25pct_denominator",
    252: "average_household_total_income_2020",
    253: "average_household_after_tax_income_2020",
}

SOURCE_INCOME = (
    "Statistics Canada Census Profile, 2021 Census of Population; "
    "income statistics for 2020"
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
# Filter to census tract income rows
# -----------------------------

df["CHARACTERISTIC_ID"] = pd.to_numeric(
    df["CHARACTERISTIC_ID"],
    errors="coerce",
).astype("Int64")

income_rows = df[
    (df["GEO_LEVEL"].astype(str).str.strip() == "Census tract")
    & (df["CHARACTERISTIC_ID"].isin(CHARACTERISTICS.keys()))
].copy()

print("\nIncome rows found:", len(income_rows))

if income_rows.empty:
    raise ValueError("No census tract income rows found.")


# -----------------------------
# Inspect selected characteristics
# -----------------------------

selected_characteristics = (
    income_rows[
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
# For these selected income variables, the usable numeric value is in C1_COUNT_TOTAL.
# This includes counts, medians, and averages.
#
# C10_RATE_TOTAL often duplicates values for some rows, but we use C1_COUNT_TOTAL
# consistently to avoid mixing count/rate semantics.

income_rows["value"] = pd.to_numeric(
    income_rows["C1_COUNT_TOTAL"],
    errors="coerce",
)


# -----------------------------
# Report missing / suppressed values
# -----------------------------

missing = income_rows[income_rows["value"].isna()].copy()

print("\nMissing or non-numeric income values:", len(missing))

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

income_rows["feature_name"] = income_rows["CHARACTERISTIC_ID"].map(
    CHARACTERISTICS
)

wide = (
    income_rows
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
# Add preferred SVI-like income fields
# -----------------------------
# The original U.S. SVI uses per-capita income and reverses its direction:
# lower income = higher vulnerability.
#
# The Canadian Census Profile does not give a direct per-capita income field here.
# We therefore keep multiple income measures and define a default available
# SVI-style income proxy for later index-specific scripts.

clean["income_measure_default"] = clean["median_after_tax_income_15plus_2020"]
clean["income_measure_default_description"] = (
    "median_after_tax_income_15plus_2020; lower values imply higher vulnerability"
)


# -----------------------------
# Add metadata
# -----------------------------

clean["unit_type"] = "census_tract"
clean["census_year"] = 2021
clean["income_reference_year"] = 2020
clean["source_income"] = SOURCE_INCOME


# -----------------------------
# Reorder columns
# -----------------------------

ordered_columns = [
    "statcan_dguid",
    "geo_name",
    "unit_type",
    "census_year",
    "income_reference_year",

    "individual_income_stats_total_15plus_100pct_denominator",
    "individual_total_income_recipients_15plus_2020_100pct",
    "median_total_income_15plus_2020",
    "individual_after_tax_income_recipients_15plus_2020_100pct",
    "median_after_tax_income_15plus_2020",
    "individual_market_income_recipients_15plus_2020_100pct",
    "median_market_income_15plus_2020",
    "individual_employment_income_recipients_15plus_2020_100pct",
    "median_employment_income_15plus_2020",

    "individual_income_stats_total_15plus_25pct_denominator",
    "individual_total_income_recipients_15plus_2020_25pct",
    "average_total_income_15plus_2020",
    "individual_after_tax_income_recipients_15plus_2020_25pct",
    "average_after_tax_income_15plus_2020",
    "individual_market_income_recipients_15plus_2020_25pct",
    "average_market_income_15plus_2020",
    "individual_employment_income_recipients_15plus_2020_25pct",
    "average_employment_income_15plus_2020",

    "household_income_stats_total_2020_100pct_denominator",
    "median_household_total_income_2020",
    "median_household_after_tax_income_2020",
    "household_income_stats_total_2020_25pct_denominator",
    "average_household_total_income_2020",
    "average_household_after_tax_income_2020",

    "income_measure_default",
    "income_measure_default_description",
    "source_income",
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

print("\nClean income table")
print("Rows:", len(clean))
print("Columns:", list(clean.columns))

print("\nMissing values by column:")
print(clean.isna().sum())

print("\nDefault income measure summary:")
print(clean["income_measure_default"].describe())

print("\nMedian after-tax income 15+ summary:")
print(clean["median_after_tax_income_15plus_2020"].describe())

print("\nMedian household after-tax income summary:")
print(clean["median_household_after_tax_income_2020"].describe())

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