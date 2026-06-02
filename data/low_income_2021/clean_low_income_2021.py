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

OUTPUT_CSV = OUTPUT_DIR / "clean_census_tract_low_income_2021.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_census_tract_low_income_2021.parquet"


# -----------------------------
# Characteristic IDs
# -----------------------------

CHARACTERISTICS = {
    335: "lim_at_denominator",
    340: "lim_at_low_income_population",
    345: "published_pct_low_income_lim_at",
    350: "lico_at_denominator",
    355: "lico_at_low_income_population",
    360: "published_pct_low_income_lico_at",
}

SOURCE_LOW_INCOME = (
    "Statistics Canada Census Profile, 2021 Census of Population; "
    "low-income status in 2020"
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
# Filter to census tract low-income rows
# -----------------------------

df["CHARACTERISTIC_ID"] = pd.to_numeric(
    df["CHARACTERISTIC_ID"],
    errors="coerce",
).astype("Int64")

low_income_rows = df[
    (df["GEO_LEVEL"].astype(str).str.strip() == "Census tract")
    & (df["CHARACTERISTIC_ID"].isin(CHARACTERISTICS.keys()))
].copy()

print("\nLow-income rows found:", len(low_income_rows))

if low_income_rows.empty:
    raise ValueError("No low-income census tract rows found.")


# -----------------------------
# Inspect selected characteristics
# -----------------------------

selected_characteristics = (
    low_income_rows[
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

# For count variables, Statistics Canada stores values in C1_COUNT_TOTAL.
# For percentage/prevalence variables, it also stores values in C10_RATE_TOTAL.
# We keep the published percentage from C10_RATE_TOTAL when available.

low_income_rows["count_value"] = pd.to_numeric(
    low_income_rows["C1_COUNT_TOTAL"],
    errors="coerce",
)

low_income_rows["rate_value"] = pd.to_numeric(
    low_income_rows["C10_RATE_TOTAL"],
    errors="coerce",
)

# Use rate for published percentage IDs, count for count IDs.
percentage_ids = {345, 360}

low_income_rows["value"] = low_income_rows["count_value"]

low_income_rows.loc[
    low_income_rows["CHARACTERISTIC_ID"].isin(percentage_ids),
    "value",
] = low_income_rows.loc[
    low_income_rows["CHARACTERISTIC_ID"].isin(percentage_ids),
    "rate_value",
]


# -----------------------------
# Report missing / suppressed values
# -----------------------------

missing = low_income_rows[low_income_rows["value"].isna()].copy()

print("\nMissing or non-numeric low-income values:", len(missing))

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

low_income_rows["feature_name"] = low_income_rows["CHARACTERISTIC_ID"].map(
    CHARACTERISTICS
)

wide = (
    low_income_rows
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
# Compute our own percentages
# -----------------------------

# We keep published percentages from StatCan, but also compute ratios ourselves.
# The computed fields are proportions between 0 and 1.
# The published fields are percentages between 0 and 100.

clean["pct_low_income_lim_at"] = (
    clean["lim_at_low_income_population"] / clean["lim_at_denominator"]
)

clean["pct_low_income_lico_at"] = (
    clean["lico_at_low_income_population"] / clean["lico_at_denominator"]
)

# Avoid infinite values if denominator is 0 or missing.
for col in ["pct_low_income_lim_at", "pct_low_income_lico_at"]:
    clean.loc[~pd.notna(clean[col]), col] = pd.NA


# -----------------------------
# Add metadata
# -----------------------------

clean["unit_type"] = "census_tract"
clean["census_year"] = 2021
clean["income_reference_year"] = 2020
clean["source_low_income"] = SOURCE_LOW_INCOME


# -----------------------------
# Reorder columns
# -----------------------------

ordered_columns = [
    "statcan_dguid",
    "geo_name",
    "unit_type",
    "census_year",
    "income_reference_year",
    "lim_at_denominator",
    "lim_at_low_income_population",
    "pct_low_income_lim_at",
    "published_pct_low_income_lim_at",
    "lico_at_denominator",
    "lico_at_low_income_population",
    "pct_low_income_lico_at",
    "published_pct_low_income_lico_at",
    "source_low_income",
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

print("\nClean low-income table")
print("Rows:", len(clean))
print("Columns:", list(clean.columns))

print("\nMissing values by column:")
print(clean.isna().sum())

print("\nLIM-AT percentage summary, computed proportion 0-1:")
print(clean["pct_low_income_lim_at"].describe())

print("\nLICO-AT percentage summary, computed proportion 0-1:")
print(clean["pct_low_income_lico_at"].describe())

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