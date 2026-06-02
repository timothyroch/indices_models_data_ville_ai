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

OUTPUT_CSV = OUTPUT_DIR / "clean_census_tract_education_2021.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_census_tract_education_2021.parquet"


# -----------------------------
# Characteristic IDs
# -----------------------------
# From the Census Profile characteristic discovery inventory:
#
# 1998 = Total - Highest certificate, diploma or degree for the population aged 15 years and over in private households - 25% sample data
# 1999 = No certificate, diploma or degree
# 2000 = High school diploma or equivalency certificate
# 2001 = Postsecondary certificate, diploma or degree
#
# 2014 = Total - Highest certificate, diploma or degree for the population aged 25 to 64 years in private households - 25% sample data
# 2015 = No certificate, diploma or degree
# 2016 = High school diploma or equivalency certificate
# 2017 = Postsecondary certificate, diploma or degree
#
# The original U.S. SVI uses "percent persons age 25+ with no high school diploma".
# The Canadian Census Profile gives a direct 25-64 version and a broader 15+ version.
# We keep both and define the default feature as the broader 15+ no-certificate share,
# while preserving the 25-64 version for sensitivity / alternative index construction.

CHARACTERISTICS = {
    1998: "education_total_15plus",
    1999: "no_certificate_15plus",
    2000: "high_school_certificate_15plus",
    2001: "postsecondary_certificate_15plus",

    2014: "education_total_25_64",
    2015: "no_certificate_25_64",
    2016: "high_school_certificate_25_64",
    2017: "postsecondary_certificate_25_64",
}

SOURCE_EDUCATION = (
    "Statistics Canada Census Profile, 2021 Census of Population; "
    "highest certificate, diploma or degree, 25% sample data"
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
# Filter to census tract education rows
# -----------------------------

df["CHARACTERISTIC_ID"] = pd.to_numeric(
    df["CHARACTERISTIC_ID"],
    errors="coerce",
).astype("Int64")

education_rows = df[
    (df["GEO_LEVEL"].astype(str).str.strip() == "Census tract")
    & (df["CHARACTERISTIC_ID"].isin(CHARACTERISTICS.keys()))
].copy()

print("\nEducation rows found:", len(education_rows))

if education_rows.empty:
    raise ValueError("No census tract education rows found.")


# -----------------------------
# Inspect selected characteristics
# -----------------------------

selected_characteristics = (
    education_rows[
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

education_rows["value"] = pd.to_numeric(
    education_rows["C1_COUNT_TOTAL"],
    errors="coerce",
)


# -----------------------------
# Report missing / suppressed values
# -----------------------------

missing = education_rows[education_rows["value"].isna()].copy()

print("\nMissing or non-numeric education values:", len(missing))

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

education_rows["feature_name"] = education_rows["CHARACTERISTIC_ID"].map(
    CHARACTERISTICS
)

wide = (
    education_rows
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
# Computed fields are proportions between 0 and 1.

clean["pct_no_certificate_15plus"] = (
    clean["no_certificate_15plus"] / clean["education_total_15plus"]
)

clean["pct_high_school_certificate_15plus"] = (
    clean["high_school_certificate_15plus"] / clean["education_total_15plus"]
)

clean["pct_postsecondary_certificate_15plus"] = (
    clean["postsecondary_certificate_15plus"] / clean["education_total_15plus"]
)

clean["pct_no_certificate_25_64"] = (
    clean["no_certificate_25_64"] / clean["education_total_25_64"]
)

clean["pct_high_school_certificate_25_64"] = (
    clean["high_school_certificate_25_64"] / clean["education_total_25_64"]
)

clean["pct_postsecondary_certificate_25_64"] = (
    clean["postsecondary_certificate_25_64"] / clean["education_total_25_64"]
)

computed_rate_cols = [
    "pct_no_certificate_15plus",
    "pct_high_school_certificate_15plus",
    "pct_postsecondary_certificate_15plus",
    "pct_no_certificate_25_64",
    "pct_high_school_certificate_25_64",
    "pct_postsecondary_certificate_25_64",
]

for col in computed_rate_cols:
    clean.loc[~pd.notna(clean[col]), col] = pd.NA


# -----------------------------
# Add default SVI-like education field
# -----------------------------
# The original SVI target is no high school diploma.
# In the Census Profile, "No certificate, diploma or degree" is the closest
# tract-level education variable.
#
# We preserve both 15+ and 25-64 variants. The default is 15+ because it is the
# broader available adult/private-household denominator. Index-specific scripts
# can switch to pct_no_certificate_25_64 if preferred.

clean["education_measure_default"] = clean["pct_no_certificate_15plus"]

clean["education_measure_default_description"] = (
    "pct_no_certificate_15plus; proxy for SVI percent without high school diploma"
)


# -----------------------------
# Add metadata
# -----------------------------

clean["unit_type"] = "census_tract"
clean["census_year"] = 2021
clean["source_education"] = SOURCE_EDUCATION


# -----------------------------
# Reorder columns
# -----------------------------

ordered_columns = [
    "statcan_dguid",
    "geo_name",
    "unit_type",
    "census_year",

    "education_total_15plus",
    "no_certificate_15plus",
    "high_school_certificate_15plus",
    "postsecondary_certificate_15plus",
    "pct_no_certificate_15plus",
    "pct_high_school_certificate_15plus",
    "pct_postsecondary_certificate_15plus",

    "education_total_25_64",
    "no_certificate_25_64",
    "high_school_certificate_25_64",
    "postsecondary_certificate_25_64",
    "pct_no_certificate_25_64",
    "pct_high_school_certificate_25_64",
    "pct_postsecondary_certificate_25_64",

    "education_measure_default",
    "education_measure_default_description",
    "source_education",
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

print("\nClean education table")
print("Rows:", len(clean))
print("Columns:", list(clean.columns))

print("\nMissing values by column:")
print(clean.isna().sum())

print("\nDefault education measure summary:")
print(clean["education_measure_default"].describe())

print("\nNo certificate 15+ proportion summary:")
print(clean["pct_no_certificate_15plus"].describe())

print("\nNo certificate 25-64 proportion summary:")
print(clean["pct_no_certificate_25_64"].describe())

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