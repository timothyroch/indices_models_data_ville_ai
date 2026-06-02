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

OUTPUT_CSV = OUTPUT_DIR / "clean_census_tract_unemployment_2021.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_census_tract_unemployment_2021.parquet"


# -----------------------------
# Characteristic IDs
# -----------------------------
# From Census Profile characteristic discovery:
#
# 2223 = Total - Population aged 15 years and over by labour force status - 25% sample data
# 2224 = In the labour force
# 2225 = Employed
# 2226 = Unemployed
# 2227 = Not in the labour force
# 2228 = Participation rate
# 2229 = Employment rate
# 2230 = Unemployment rate
#
# For SVI-like unemployment, the main variable is:
#   pct_unemployed = unemployed_population / labour_force_population
#
# Statistics Canada also provides unemployment_rate directly as characteristic 2230.
# We keep both computed and published versions.

CHARACTERISTICS = {
    2223: "labour_force_status_denominator_15plus",
    2224: "labour_force_population",
    2225: "employed_population",
    2226: "unemployed_population",
    2227: "not_in_labour_force_population",
    2228: "published_participation_rate",
    2229: "published_employment_rate",
    2230: "published_unemployment_rate",
}

PERCENTAGE_IDS = {
    2228,
    2229,
    2230,
}

SOURCE_UNEMPLOYMENT = (
    "Statistics Canada Census Profile, 2021 Census of Population; "
    "labour force status for population aged 15 years and over, 25% sample data"
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
# Filter to census tract unemployment rows
# -----------------------------

df["CHARACTERISTIC_ID"] = pd.to_numeric(
    df["CHARACTERISTIC_ID"],
    errors="coerce",
).astype("Int64")

unemployment_rows = df[
    (df["GEO_LEVEL"].astype(str).str.strip() == "Census tract")
    & (df["CHARACTERISTIC_ID"].isin(CHARACTERISTICS.keys()))
].copy()

print("\nUnemployment/labour-force rows found:", len(unemployment_rows))

if unemployment_rows.empty:
    raise ValueError("No census tract unemployment/labour-force rows found.")


# -----------------------------
# Inspect selected characteristics
# -----------------------------

selected_characteristics = (
    unemployment_rows[
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

unemployment_rows["count_value"] = pd.to_numeric(
    unemployment_rows["C1_COUNT_TOTAL"],
    errors="coerce",
)

unemployment_rows["rate_value"] = pd.to_numeric(
    unemployment_rows["C10_RATE_TOTAL"],
    errors="coerce",
)

# Count IDs use C1_COUNT_TOTAL.
# Rate IDs use C10_RATE_TOTAL.
unemployment_rows["value"] = unemployment_rows["count_value"]

unemployment_rows.loc[
    unemployment_rows["CHARACTERISTIC_ID"].isin(PERCENTAGE_IDS),
    "value",
] = unemployment_rows.loc[
    unemployment_rows["CHARACTERISTIC_ID"].isin(PERCENTAGE_IDS),
    "rate_value",
]


# -----------------------------
# Report missing / suppressed values
# -----------------------------

missing = unemployment_rows[unemployment_rows["value"].isna()].copy()

print("\nMissing or non-numeric unemployment/labour-force values:", len(missing))

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

unemployment_rows["feature_name"] = unemployment_rows["CHARACTERISTIC_ID"].map(
    CHARACTERISTICS
)

wide = (
    unemployment_rows
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
# Compute rates
# -----------------------------
# Computed fields are proportions between 0 and 1.
# Published fields from StatCan are percentages between 0 and 100.

clean["pct_unemployed"] = (
    clean["unemployed_population"] / clean["labour_force_population"]
)

clean["pct_employed"] = (
    clean["employed_population"] / clean["labour_force_status_denominator_15plus"]
)

clean["pct_in_labour_force"] = (
    clean["labour_force_population"] / clean["labour_force_status_denominator_15plus"]
)

clean["pct_not_in_labour_force"] = (
    clean["not_in_labour_force_population"] / clean["labour_force_status_denominator_15plus"]
)

# Avoid infinite values if denominator is 0 or missing.
computed_rate_cols = [
    "pct_unemployed",
    "pct_employed",
    "pct_in_labour_force",
    "pct_not_in_labour_force",
]

for col in computed_rate_cols:
    clean.loc[~pd.notna(clean[col]), col] = pd.NA


# -----------------------------
# Add metadata
# -----------------------------

clean["unit_type"] = "census_tract"
clean["census_year"] = 2021
clean["source_unemployment"] = SOURCE_UNEMPLOYMENT


# -----------------------------
# Reorder columns
# -----------------------------

ordered_columns = [
    "statcan_dguid",
    "geo_name",
    "unit_type",
    "census_year",
    "labour_force_status_denominator_15plus",
    "labour_force_population",
    "employed_population",
    "unemployed_population",
    "not_in_labour_force_population",
    "pct_unemployed",
    "published_unemployment_rate",
    "pct_employed",
    "published_employment_rate",
    "pct_in_labour_force",
    "published_participation_rate",
    "pct_not_in_labour_force",
    "source_unemployment",
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

print("\nClean unemployment table")
print("Rows:", len(clean))
print("Columns:", list(clean.columns))

print("\nMissing values by column:")
print(clean.isna().sum())

print("\nComputed unemployment proportion summary, 0-1:")
print(clean["pct_unemployed"].describe())

print("\nPublished unemployment rate summary, 0-100:")
print(clean["published_unemployment_rate"].describe())

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