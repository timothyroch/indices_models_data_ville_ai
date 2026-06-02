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

OUTPUT_CSV = OUTPUT_DIR / "clean_census_tract_language_2021.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_census_tract_language_2021.parquet"


# -----------------------------
# Characteristic IDs
# -----------------------------
# Corrected and enriched from targeted Census Profile inspection.
#
# Knowledge of official languages:
# 383 = Total - Knowledge of official languages for the total population excluding institutional residents - 100% data
# 384 = English only
# 385 = French only
# 386 = English and French
# 387 = Neither English nor French
#
# First official language spoken:
# 388 = Total - First official language spoken for the total population excluding institutional residents - 100% data
# 389 = English
# 390 = French
# 391 = English and French
# 392 = Neither English nor French
#
# Mother tongue:
# 393 = Total - Mother tongue for the total population excluding institutional residents - 100% data
# 398 = Non-official languages
#
# Language spoken most often at home:
# 735  = Total - Language spoken most often at home for the total population excluding institutional residents - 100% data
# 740  = Non-official languages, single-response branch
# 1060 = Multiple responses
# 1062 = English and non-official language(s)
# 1063 = French and non-official language(s)
# 1064 = English, French and non-official language(s)
# 1065 = Multiple non-official languages
#
# Other language(s) spoken regularly at home:
# 1066 = Total - Other language(s) spoken regularly at home for the total population excluding institutional residents - 100% data
# 1067 = None
# 1068 = English
# 1069 = French
# 1070 = Non-official language
# 1071 = Indigenous
# 1072 = Non-Indigenous
#
# Important correction:
# - 1070 is NOT under the "language spoken most often at home" denominator 735.
# - 1070 belongs under 1066, "Other language(s) spoken regularly at home".
#
# Therefore:
# - pct_home_language_most_often_non_official_including_multiple uses 735 as denominator.
# - pct_other_home_language_regularly_non_official uses 1066 as denominator.
#
# Notes:
# - The original U.S. SVI language variable is about limited English proficiency.
# - For Québec / Canada, we preserve several language-access options.
# - The default language-barrier proxy remains:
#       pct_knows_neither_english_nor_french
#
# This is not exactly the same as "speaks English less than well", but it is a
# clean Canadian Census Profile equivalent for official-language access.

CHARACTERISTICS = {
    # Knowledge of official languages
    383: "official_language_knowledge_total",
    384: "knows_english_only",
    385: "knows_french_only",
    386: "knows_english_and_french",
    387: "knows_neither_english_nor_french",

    # First official language spoken
    388: "first_official_language_total",
    389: "first_official_language_english",
    390: "first_official_language_french",
    391: "first_official_language_english_and_french",
    392: "first_official_language_neither_english_nor_french",

    # Mother tongue
    393: "mother_tongue_total",
    398: "mother_tongue_non_official_single_response",

    # Language spoken most often at home
    735: "home_language_most_often_total",
    740: "home_language_most_often_non_official_single_response",
    1060: "home_language_most_often_multiple_responses",
    1062: "home_language_most_often_english_and_non_official_multiple",
    1063: "home_language_most_often_french_and_non_official_multiple",
    1064: "home_language_most_often_english_french_and_non_official_multiple",
    1065: "home_language_most_often_multiple_non_official_languages",

    # Other language(s) spoken regularly at home
    1066: "other_home_language_regularly_total",
    1067: "other_home_language_regularly_none",
    1068: "other_home_language_regularly_english",
    1069: "other_home_language_regularly_french",
    1070: "other_home_language_regularly_non_official",
    1071: "other_home_language_regularly_non_official_indigenous",
    1072: "other_home_language_regularly_non_official_non_indigenous",
}

SOURCE_LANGUAGE = (
    "Statistics Canada Census Profile, 2021 Census of Population; "
    "official-language knowledge, first official language spoken, mother tongue, "
    "language spoken most often at home, and other languages spoken regularly at home, 100% data"
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
# Filter to census tract language rows
# -----------------------------

df["CHARACTERISTIC_ID"] = pd.to_numeric(
    df["CHARACTERISTIC_ID"],
    errors="coerce",
).astype("Int64")

language_rows = df[
    (df["GEO_LEVEL"].astype(str).str.strip() == "Census tract")
    & (df["CHARACTERISTIC_ID"].isin(CHARACTERISTICS.keys()))
].copy()

print("\nLanguage rows found:", len(language_rows))

if language_rows.empty:
    raise ValueError("No census tract language rows found.")


# -----------------------------
# Inspect selected characteristics
# -----------------------------

selected_characteristics = (
    language_rows[
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

found_ids = set(language_rows["CHARACTERISTIC_ID"].dropna().astype(int).unique())
expected_ids = set(CHARACTERISTICS.keys())
missing_ids = sorted(expected_ids - found_ids)

if missing_ids:
    print("\nWarning: some expected language characteristic IDs were not found:")
    for characteristic_id in missing_ids:
        print(f"  {characteristic_id}: {CHARACTERISTICS[characteristic_id]}")


# -----------------------------
# Prepare value column
# -----------------------------

language_rows["value"] = pd.to_numeric(
    language_rows["C1_COUNT_TOTAL"],
    errors="coerce",
)


# -----------------------------
# Report missing / suppressed values
# -----------------------------

missing = language_rows[language_rows["value"].isna()].copy()

print("\nMissing or non-numeric language values:", len(missing))

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

language_rows["feature_name"] = language_rows["CHARACTERISTIC_ID"].map(
    CHARACTERISTICS
)

wide = (
    language_rows
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
# Compute official-language knowledge proportions
# -----------------------------

clean["pct_knows_english_only"] = bounded_proportion(
    clean["knows_english_only"],
    clean["official_language_knowledge_total"],
)

clean["pct_knows_french_only"] = bounded_proportion(
    clean["knows_french_only"],
    clean["official_language_knowledge_total"],
)

clean["pct_knows_english_and_french"] = bounded_proportion(
    clean["knows_english_and_french"],
    clean["official_language_knowledge_total"],
)

clean["pct_knows_neither_english_nor_french"] = bounded_proportion(
    clean["knows_neither_english_nor_french"],
    clean["official_language_knowledge_total"],
)


# -----------------------------
# Compute first-official-language proportions
# -----------------------------

clean["pct_first_official_language_english"] = bounded_proportion(
    clean["first_official_language_english"],
    clean["first_official_language_total"],
)

clean["pct_first_official_language_french"] = bounded_proportion(
    clean["first_official_language_french"],
    clean["first_official_language_total"],
)

clean["pct_first_official_language_english_and_french"] = bounded_proportion(
    clean["first_official_language_english_and_french"],
    clean["first_official_language_total"],
)

clean["pct_first_official_language_neither_english_nor_french"] = bounded_proportion(
    clean["first_official_language_neither_english_nor_french"],
    clean["first_official_language_total"],
)


# -----------------------------
# Compute mother-tongue proportions
# -----------------------------
# Note:
#   ID 398 is the non-official-language branch under single responses.
#   We keep the name explicit to avoid overstating it as all non-official
#   mother-tongue responses.

clean["pct_mother_tongue_non_official_single_response"] = bounded_proportion(
    clean["mother_tongue_non_official_single_response"],
    clean["mother_tongue_total"],
)


# -----------------------------
# Compute language-spoken-most-often-at-home proportions
# -----------------------------

clean["home_language_most_often_non_official_including_multiple_count"] = (
    clean["home_language_most_often_non_official_single_response"]
    + clean["home_language_most_often_english_and_non_official_multiple"]
    + clean["home_language_most_often_french_and_non_official_multiple"]
    + clean["home_language_most_often_english_french_and_non_official_multiple"]
    + clean["home_language_most_often_multiple_non_official_languages"]
)

clean["pct_home_language_most_often_non_official_single_response"] = bounded_proportion(
    clean["home_language_most_often_non_official_single_response"],
    clean["home_language_most_often_total"],
)

clean["pct_home_language_most_often_non_official_including_multiple"] = bounded_proportion(
    clean["home_language_most_often_non_official_including_multiple_count"],
    clean["home_language_most_often_total"],
)

clean["pct_home_language_most_often_multiple_responses"] = bounded_proportion(
    clean["home_language_most_often_multiple_responses"],
    clean["home_language_most_often_total"],
)


# -----------------------------
# Compute other-language-regularly-at-home proportions
# -----------------------------

clean["pct_other_home_language_regularly_none"] = bounded_proportion(
    clean["other_home_language_regularly_none"],
    clean["other_home_language_regularly_total"],
)

clean["pct_other_home_language_regularly_english"] = bounded_proportion(
    clean["other_home_language_regularly_english"],
    clean["other_home_language_regularly_total"],
)

clean["pct_other_home_language_regularly_french"] = bounded_proportion(
    clean["other_home_language_regularly_french"],
    clean["other_home_language_regularly_total"],
)

clean["pct_other_home_language_regularly_non_official"] = bounded_proportion(
    clean["other_home_language_regularly_non_official"],
    clean["other_home_language_regularly_total"],
)

clean["pct_other_home_language_regularly_non_official_indigenous"] = bounded_proportion(
    clean["other_home_language_regularly_non_official_indigenous"],
    clean["other_home_language_regularly_total"],
)

clean["pct_other_home_language_regularly_non_official_non_indigenous"] = bounded_proportion(
    clean["other_home_language_regularly_non_official_non_indigenous"],
    clean["other_home_language_regularly_total"],
)


# -----------------------------
# Add default SVI-like language field
# -----------------------------
# Default proxy:
#   pct_knows_neither_english_nor_french
#
# This is a strict official-language-access variable.
# It is very conservative, especially in Québec, because it captures people
# who report knowing neither official language.

clean["language_barrier_measure_default"] = clean[
    "pct_knows_neither_english_nor_french"
].astype(float)

clean["language_barrier_measure_default_description"] = (
    "pct_knows_neither_english_nor_french; Canadian official-language-access proxy"
)


# -----------------------------
# Add named language-context fields
# -----------------------------

clean["home_language_most_often_non_official_measure_default"] = clean[
    "pct_home_language_most_often_non_official_including_multiple"
].astype(float)

clean["home_language_most_often_non_official_measure_default_description"] = (
    "pct_home_language_most_often_non_official_including_multiple; share whose language spoken most often at home includes a non-official language"
)

clean["other_home_language_regularly_non_official_measure_default"] = clean[
    "pct_other_home_language_regularly_non_official"
].astype(float)

clean["other_home_language_regularly_non_official_measure_default_description"] = (
    "pct_other_home_language_regularly_non_official; share who regularly speak a non-official language at home as another language"
)

clean["mother_tongue_non_official_measure_default"] = clean[
    "pct_mother_tongue_non_official_single_response"
].astype(float)

clean["mother_tongue_non_official_measure_default_description"] = (
    "pct_mother_tongue_non_official_single_response; non-official mother-tongue single-response share"
)


# -----------------------------
# Add metadata
# -----------------------------

clean["unit_type"] = "census_tract"
clean["census_year"] = 2021
clean["source_language"] = SOURCE_LANGUAGE


# -----------------------------
# Reorder columns
# -----------------------------

ordered_columns = [
    "statcan_dguid",
    "geo_name",
    "unit_type",
    "census_year",

    # Official-language knowledge
    "official_language_knowledge_total",
    "knows_english_only",
    "knows_french_only",
    "knows_english_and_french",
    "knows_neither_english_nor_french",
    "pct_knows_english_only",
    "pct_knows_french_only",
    "pct_knows_english_and_french",
    "pct_knows_neither_english_nor_french",

    # First official language spoken
    "first_official_language_total",
    "first_official_language_english",
    "first_official_language_french",
    "first_official_language_english_and_french",
    "first_official_language_neither_english_nor_french",
    "pct_first_official_language_english",
    "pct_first_official_language_french",
    "pct_first_official_language_english_and_french",
    "pct_first_official_language_neither_english_nor_french",

    # Mother tongue
    "mother_tongue_total",
    "mother_tongue_non_official_single_response",
    "pct_mother_tongue_non_official_single_response",

    # Language spoken most often at home
    "home_language_most_often_total",
    "home_language_most_often_non_official_single_response",
    "home_language_most_often_multiple_responses",
    "home_language_most_often_english_and_non_official_multiple",
    "home_language_most_often_french_and_non_official_multiple",
    "home_language_most_often_english_french_and_non_official_multiple",
    "home_language_most_often_multiple_non_official_languages",
    "home_language_most_often_non_official_including_multiple_count",
    "pct_home_language_most_often_non_official_single_response",
    "pct_home_language_most_often_non_official_including_multiple",
    "pct_home_language_most_often_multiple_responses",

    # Other language(s) spoken regularly at home
    "other_home_language_regularly_total",
    "other_home_language_regularly_none",
    "other_home_language_regularly_english",
    "other_home_language_regularly_french",
    "other_home_language_regularly_non_official",
    "other_home_language_regularly_non_official_indigenous",
    "other_home_language_regularly_non_official_non_indigenous",
    "pct_other_home_language_regularly_none",
    "pct_other_home_language_regularly_english",
    "pct_other_home_language_regularly_french",
    "pct_other_home_language_regularly_non_official",
    "pct_other_home_language_regularly_non_official_indigenous",
    "pct_other_home_language_regularly_non_official_non_indigenous",

    # Defaults / named proxy fields
    "language_barrier_measure_default",
    "language_barrier_measure_default_description",
    "home_language_most_often_non_official_measure_default",
    "home_language_most_often_non_official_measure_default_description",
    "other_home_language_regularly_non_official_measure_default",
    "other_home_language_regularly_non_official_measure_default_description",
    "mother_tongue_non_official_measure_default",
    "mother_tongue_non_official_measure_default_description",

    "source_language",
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
    "pct_knows_english_only",
    "pct_knows_french_only",
    "pct_knows_english_and_french",
    "pct_knows_neither_english_nor_french",

    "pct_first_official_language_english",
    "pct_first_official_language_french",
    "pct_first_official_language_english_and_french",
    "pct_first_official_language_neither_english_nor_french",

    "pct_mother_tongue_non_official_single_response",

    "pct_home_language_most_often_non_official_single_response",
    "pct_home_language_most_often_non_official_including_multiple",
    "pct_home_language_most_often_multiple_responses",

    "pct_other_home_language_regularly_none",
    "pct_other_home_language_regularly_english",
    "pct_other_home_language_regularly_french",
    "pct_other_home_language_regularly_non_official",
    "pct_other_home_language_regularly_non_official_indigenous",
    "pct_other_home_language_regularly_non_official_non_indigenous",

    "language_barrier_measure_default",
    "home_language_most_often_non_official_measure_default",
    "other_home_language_regularly_non_official_measure_default",
    "mother_tongue_non_official_measure_default",
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


print("\nClean language table")
print("Rows:", len(clean))
print("Columns:", list(clean.columns))

print("\nMissing values by column:")
print(clean.isna().sum())

print("\nDefault language-barrier measure summary:")
print(clean["language_barrier_measure_default"].describe())

print("\nNeither English nor French knowledge proportion summary:")
print(clean["pct_knows_neither_english_nor_french"].describe())

print("\nHome language most often non-official including multiple responses summary:")
print(clean["home_language_most_often_non_official_measure_default"].describe())

print("\nOther home language regularly non-official summary:")
print(clean["other_home_language_regularly_non_official_measure_default"].describe())

print("\nMother tongue non-official single-response summary:")
print(clean["mother_tongue_non_official_measure_default"].describe())

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