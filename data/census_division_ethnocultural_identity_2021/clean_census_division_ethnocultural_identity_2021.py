from pathlib import Path
import pandas as pd


# ============================================================
# Clean Census Division Ethnocultural / Indigenous Identity 2021
# ============================================================
#
# Purpose:
#   Create clean Québec census-division features for SoVI-like
#   ethnocultural / Indigenous identity variables:
#
#       pct_black
#       pct_hispanic
#       pct_indigenous
#       pct_asian
#
# Original SoVI variables:
#
#       PCTBLACK90      -> pct_black
#       PCTHISPANIC90   -> pct_hispanic
#       PCTINDIAN90     -> pct_indigenous
#       PCTASIAN90      -> pct_asian
#
# Method:
#   Uses forced Census Profile CHARACTERISTIC_ID values verified by the
#   inspection script.
#
# Direct variables:
#
#       pct_black       -> 1687 Black
#       pct_hispanic    -> 1690 Latin American
#       pct_indigenous  -> 1403 Indigenous identity
#
# Derived variable:
#
#       pct_asian =
#           1685 South Asian
#         + 1686 Chinese
#         + 1688 Filipino
#         + 1691 Southeast Asian
#         + 1692 West Asian
#         + 1693 Korean
#         + 1694 Japanese
#
# Important:
#   pct_asian is derived from visible-minority population-group rate rows.
#   It intentionally avoids language rows such as Chinese languages and
#   Tagalog.
#
# Run from data/:
#   python census_division_ethnocultural_identity_2021/clean_census_division_ethnocultural_identity_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_ethnocultural_identity_2021"
OUTPUT_DIR = SECTION_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CENSUS_PROFILE = (
    DATA_DIR
    / "census_profile_census_division_2021"
    / "raw"
    / "98-401-X2021004_English_CSV_data.csv"
)

BASE_CD_FRAME = (
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.csv"
)

OUTPUT_CLEAN = OUTPUT_DIR / "clean_census_division_ethnocultural_identity_2021.csv"
OUTPUT_COMPONENT_LONG = OUTPUT_DIR / "clean_census_division_ethnocultural_identity_component_long_2021.csv"
OUTPUT_METADATA = OUTPUT_DIR / "clean_census_division_ethnocultural_identity_variable_metadata_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "clean_census_division_ethnocultural_identity_summary_2021.csv"


# -----------------------------
# Config
# -----------------------------

ENCODING_CANDIDATES = [
    "cp1252",
    "utf-8-sig",
    "utf-8",
    "latin1",
]

CHUNKSIZE = 200_000

QUEBEC_CD_DGUID_PREFIX = "2021A000324"
EXPECTED_QC_CD_COUNT = 98

VALUE_RATE_COLUMN = "C10_RATE_TOTAL"
VALUE_COUNT_COLUMN = "C1_COUNT_TOTAL"
SYMBOL_RATE_COLUMN = "SYMBOL.3"
SYMBOL_COUNT_COLUMN = "SYMBOL"

RAW_COLUMNS_NEEDED = [
    "CENSUS_YEAR",
    "DGUID",
    "ALT_GEO_CODE",
    "GEO_LEVEL",
    "GEO_NAME",
    "TNR_SF",
    "TNR_LF",
    "DATA_QUALITY_FLAG",
    "CHARACTERISTIC_ID",
    "CHARACTERISTIC_NAME",
    "CHARACTERISTIC_NOTE",
    "C1_COUNT_TOTAL",
    "SYMBOL",
    "C10_RATE_TOTAL",
    "SYMBOL.3",
]

DIRECT_TARGETS = {
    "pct_black": {
        "original_code": "PCTBLACK90",
        "characteristic_id": "1687",
        "expected_name": "Black",
        "unit": "percent",
        "source_role": "visible_minority_population_group",
        "method_note": (
            "Uses Census Profile visible-minority / population-group row 'Black'. "
            "This avoids Blackfoot language or ethnocultural-origin rows."
        ),
    },
    "pct_hispanic": {
        "original_code": "PCTHISPANIC90",
        "characteristic_id": "1690",
        "expected_name": "Latin American",
        "unit": "percent",
        "source_role": "visible_minority_population_group_proxy",
        "method_note": (
            "Canadian Census Profile does not use the U.S. Hispanic category directly. "
            "Uses Latin American visible-minority / population-group row as the closest proxy."
        ),
    },
    "pct_indigenous": {
        "original_code": "PCTINDIAN90",
        "characteristic_id": "1403",
        "expected_name": "Indigenous identity",
        "unit": "percent",
        "source_role": "indigenous_identity",
        "method_note": (
            "Uses broad Indigenous identity rate. This avoids total-denominator rows, "
            "Non-Indigenous identity rows, and narrower First Nations / Métis / Inuit component rows."
        ),
    },
}

ASIAN_COMPONENTS = {
    "pct_asian_component_south_asian": {
        "component_key": "south_asian",
        "characteristic_id": "1685",
        "expected_name": "South Asian",
    },
    "pct_asian_component_chinese": {
        "component_key": "chinese",
        "characteristic_id": "1686",
        "expected_name": "Chinese",
    },
    "pct_asian_component_filipino": {
        "component_key": "filipino",
        "characteristic_id": "1688",
        "expected_name": "Filipino",
    },
    "pct_asian_component_southeast_asian": {
        "component_key": "southeast_asian",
        "characteristic_id": "1691",
        "expected_name": "Southeast Asian",
    },
    "pct_asian_component_west_asian": {
        "component_key": "west_asian",
        "characteristic_id": "1692",
        "expected_name": "West Asian",
    },
    "pct_asian_component_korean": {
        "component_key": "korean",
        "characteristic_id": "1693",
        "expected_name": "Korean",
    },
    "pct_asian_component_japanese": {
        "component_key": "japanese",
        "characteristic_id": "1694",
        "expected_name": "Japanese",
    },
}

ALL_CHARACTERISTIC_IDS = (
    [target["characteristic_id"] for target in DIRECT_TARGETS.values()]
    + [component["characteristic_id"] for component in ASIAN_COMPONENTS.values()]
)


# -----------------------------
# Helpers
# -----------------------------

def select_encoding(path: Path) -> str:
    for encoding in ENCODING_CANDIDATES:
        try:
            pd.read_csv(path, encoding=encoding, nrows=5, low_memory=False)
            return encoding
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not decode {path} with candidates {ENCODING_CANDIDATES}",
    )


def clean_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        return str(path)


def contains_mojibake(series: pd.Series) -> int:
    text = series.astype("string")
    return int(
        text.str.contains("Ã|Â|�", regex=True, na=False).sum()
    )


def numeric_summary(series: pd.Series) -> dict:
    numeric = clean_numeric(series)

    return {
        "non_missing": int(numeric.notna().sum()),
        "missing": int(numeric.isna().sum()),
        "min": numeric.min(skipna=True),
        "max": numeric.max(skipna=True),
        "mean": numeric.mean(skipna=True),
        "median": numeric.median(skipna=True),
    }


def validate_single_characteristic(
    df: pd.DataFrame,
    characteristic_id: str,
    expected_name: str,
    alias: str,
) -> None:
    if df.empty:
        raise ValueError(
            f"No rows found for {alias} / CHARACTERISTIC_ID={characteristic_id}."
        )

    unique_cds = df["DGUID"].nunique()
    if unique_cds != EXPECTED_QC_CD_COUNT:
        raise ValueError(
            f"{alias} / CHARACTERISTIC_ID={characteristic_id} has {unique_cds} unique Québec CDs; "
            f"expected {EXPECTED_QC_CD_COUNT}."
        )

    values = clean_numeric(df[VALUE_RATE_COLUMN])
    missing = int(values.isna().sum())
    if missing != 0:
        raise ValueError(
            f"{alias} / CHARACTERISTIC_ID={characteristic_id} has {missing} missing rate values."
        )

    names = sorted(df["CHARACTERISTIC_NAME"].astype(str).str.strip().unique())
    if len(names) != 1:
        raise ValueError(
            f"{alias} / CHARACTERISTIC_ID={characteristic_id} has multiple names:\n{names}"
        )

    actual_name = names[0].strip().lower()
    expected = expected_name.strip().lower()

    if expected not in actual_name:
        raise ValueError(
            f"{alias} / CHARACTERISTIC_ID={characteristic_id} name mismatch. "
            f"Expected to contain '{expected_name}', got '{names[0]}'."
        )


# -----------------------------
# Validate inputs
# -----------------------------

if not RAW_CENSUS_PROFILE.exists():
    raise FileNotFoundError(f"Missing raw Census Profile CD file:\n{RAW_CENSUS_PROFILE}")

if not BASE_CD_FRAME.exists():
    raise FileNotFoundError(f"Missing base CD frame:\n{BASE_CD_FRAME}")


# -----------------------------
# Load base frame
# -----------------------------

base = pd.read_csv(BASE_CD_FRAME, dtype=str, low_memory=False)
base.columns = [str(col).strip() for col in base.columns]

required_base_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
]

missing_base_cols = [col for col in required_base_cols if col not in base.columns]
if missing_base_cols:
    raise ValueError(
        "Base CD frame is missing required columns:\n"
        + "\n".join(missing_base_cols)
        + "\n\nAvailable columns:\n"
        + "\n".join(base.columns)
    )

base["census_division_dguid"] = clean_key(base["census_division_dguid"])

if len(base) != EXPECTED_QC_CD_COUNT:
    raise ValueError(f"Base frame has {len(base)} rows; expected {EXPECTED_QC_CD_COUNT}.")

if base["census_division_dguid"].duplicated().any():
    dupes = base[base["census_division_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicate census_division_dguid values in base frame:\n"
        + dupes[required_base_cols].to_string(index=False)
    )

print("\nCleaning Census Division Ethnocultural / Indigenous Identity 2021")
print("Raw Census Profile:", safe_relative(RAW_CENSUS_PROFILE))
print("Base CD frame:", safe_relative(BASE_CD_FRAME))
print("Base rows:", len(base))


# -----------------------------
# Scan raw Census Profile for forced IDs
# -----------------------------

encoding = select_encoding(RAW_CENSUS_PROFILE)
print("Raw encoding selected:", encoding)

selected_chunks = []
total_rows_scanned = 0
quebec_cd_rows_scanned = 0

print("\nScanning raw Census Profile file...")

for chunk_idx, chunk in enumerate(
    pd.read_csv(
        RAW_CENSUS_PROFILE,
        encoding=encoding,
        dtype=str,
        usecols=lambda col: col in RAW_COLUMNS_NEEDED,
        chunksize=CHUNKSIZE,
        low_memory=False,
    ),
    start=1,
):
    chunk.columns = [str(col).strip() for col in chunk.columns]
    total_rows_scanned += len(chunk)

    qc_cd = chunk[
        (chunk["GEO_LEVEL"].astype("string").str.strip() == "Census division")
        & (chunk["DGUID"].astype("string").str.startswith(QUEBEC_CD_DGUID_PREFIX, na=False))
    ].copy()

    quebec_cd_rows_scanned += len(qc_cd)

    if qc_cd.empty:
        continue

    qc_cd["CHARACTERISTIC_ID_STR"] = qc_cd["CHARACTERISTIC_ID"].astype(str).str.strip()

    selected = qc_cd[qc_cd["CHARACTERISTIC_ID_STR"].isin(ALL_CHARACTERISTIC_IDS)].copy()

    if not selected.empty:
        selected_chunks.append(selected)

    if chunk_idx % 10 == 0:
        print(f"  Processed chunks: {chunk_idx}, rows so far: {total_rows_scanned}")


if selected_chunks:
    selected_rows = pd.concat(selected_chunks, ignore_index=True)
else:
    selected_rows = pd.DataFrame(columns=RAW_COLUMNS_NEEDED + ["CHARACTERISTIC_ID_STR"])

print("Selected forced-ID rows:", len(selected_rows))


# -----------------------------
# Validate selected rows
# -----------------------------

expected_selected_rows = EXPECTED_QC_CD_COUNT * len(ALL_CHARACTERISTIC_IDS)

if len(selected_rows) != expected_selected_rows:
    raise ValueError(
        f"Expected {expected_selected_rows} selected rows "
        f"({EXPECTED_QC_CD_COUNT} CDs × {len(ALL_CHARACTERISTIC_IDS)} characteristics), "
        f"got {len(selected_rows)}."
    )

for alias, target in DIRECT_TARGETS.items():
    rows = selected_rows[
        selected_rows["CHARACTERISTIC_ID_STR"] == target["characteristic_id"]
    ].copy()

    validate_single_characteristic(
        df=rows,
        characteristic_id=target["characteristic_id"],
        expected_name=target["expected_name"],
        alias=alias,
    )

for alias, component in ASIAN_COMPONENTS.items():
    rows = selected_rows[
        selected_rows["CHARACTERISTIC_ID_STR"] == component["characteristic_id"]
    ].copy()

    validate_single_characteristic(
        df=rows,
        characteristic_id=component["characteristic_id"],
        expected_name=component["expected_name"],
        alias=alias,
    )


# -----------------------------
# Build component-long audit table
# -----------------------------

component_long_rows = []

for canonical_variable, target in DIRECT_TARGETS.items():
    rows = selected_rows[
        selected_rows["CHARACTERISTIC_ID_STR"] == target["characteristic_id"]
    ].copy()

    temp = rows[
        [
            "DGUID",
            "ALT_GEO_CODE",
            "GEO_NAME",
            "CHARACTERISTIC_ID",
            "CHARACTERISTIC_NAME",
            VALUE_RATE_COLUMN,
            SYMBOL_RATE_COLUMN,
            VALUE_COUNT_COLUMN,
            SYMBOL_COUNT_COLUMN,
        ]
    ].copy()

    temp = temp.rename(
        columns={
            "DGUID": "census_division_dguid",
            "ALT_GEO_CODE": "census_division_code",
            "GEO_NAME": "profile_geo_name",
            VALUE_RATE_COLUMN: "rate_value",
            SYMBOL_RATE_COLUMN: "rate_symbol",
            VALUE_COUNT_COLUMN: "count_value",
            SYMBOL_COUNT_COLUMN: "count_symbol",
        }
    )

    temp["canonical_variable"] = canonical_variable
    temp["component_alias"] = canonical_variable
    temp["original_code"] = target["original_code"]
    temp["unit"] = target["unit"]
    temp["source_role"] = target["source_role"]
    temp["method_note"] = target["method_note"]
    temp["rate_value"] = clean_numeric(temp["rate_value"])
    temp["count_value"] = clean_numeric(temp["count_value"])

    component_long_rows.append(temp)

for component_alias, component in ASIAN_COMPONENTS.items():
    rows = selected_rows[
        selected_rows["CHARACTERISTIC_ID_STR"] == component["characteristic_id"]
    ].copy()

    temp = rows[
        [
            "DGUID",
            "ALT_GEO_CODE",
            "GEO_NAME",
            "CHARACTERISTIC_ID",
            "CHARACTERISTIC_NAME",
            VALUE_RATE_COLUMN,
            SYMBOL_RATE_COLUMN,
            VALUE_COUNT_COLUMN,
            SYMBOL_COUNT_COLUMN,
        ]
    ].copy()

    temp = temp.rename(
        columns={
            "DGUID": "census_division_dguid",
            "ALT_GEO_CODE": "census_division_code",
            "GEO_NAME": "profile_geo_name",
            VALUE_RATE_COLUMN: "rate_value",
            SYMBOL_RATE_COLUMN: "rate_symbol",
            VALUE_COUNT_COLUMN: "count_value",
            SYMBOL_COUNT_COLUMN: "count_symbol",
        }
    )

    temp["canonical_variable"] = "pct_asian"
    temp["component_alias"] = component_alias
    temp["original_code"] = "PCTASIAN90"
    temp["unit"] = "percent"
    temp["source_role"] = "visible_minority_population_group_component"
    temp["method_note"] = (
        "Component of derived pct_asian. Uses visible-minority population-group rate row, "
        "not language or ethnic-origin row."
    )
    temp["rate_value"] = clean_numeric(temp["rate_value"])
    temp["count_value"] = clean_numeric(temp["count_value"])

    component_long_rows.append(temp)

component_long = pd.concat(component_long_rows, ignore_index=True)

component_long = component_long[
    [
        "census_division_dguid",
        "census_division_code",
        "profile_geo_name",
        "canonical_variable",
        "component_alias",
        "original_code",
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "rate_value",
        "rate_symbol",
        "count_value",
        "count_symbol",
        "unit",
        "source_role",
        "method_note",
    ]
].copy()

component_long.to_csv(OUTPUT_COMPONENT_LONG, index=False, encoding="utf-8")


# -----------------------------
# Pivot to wide clean table
# -----------------------------

rate_wide = (
    component_long
    .pivot_table(
        index="census_division_dguid",
        columns="component_alias",
        values="rate_value",
        aggfunc="first",
    )
    .reset_index()
)

rate_wide.columns.name = None

clean = base.copy()

identity_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "census_division_type",
    "province_code",
    "province_name",
    "geography_level",
    "census_year",
    "population_total_2021",
    "land_area_km2",
    "has_positive_population",
]

identity_cols = [col for col in identity_cols if col in clean.columns]
clean = clean[identity_cols].copy()

clean = clean.merge(
    rate_wide,
    on="census_division_dguid",
    how="left",
    validate="one_to_one",
)

asian_component_cols = list(ASIAN_COMPONENTS.keys())

clean["pct_asian"] = clean[asian_component_cols].sum(axis=1, skipna=False)

# Reorder output columns.
main_feature_cols = [
    "pct_black",
    "pct_hispanic",
    "pct_indigenous",
    "pct_asian",
]

audit_component_cols = [
    "pct_asian_component_south_asian",
    "pct_asian_component_chinese",
    "pct_asian_component_filipino",
    "pct_asian_component_southeast_asian",
    "pct_asian_component_west_asian",
    "pct_asian_component_korean",
    "pct_asian_component_japanese",
]

other_cols = [
    "source_file",
    "method_note",
]

clean["source_file"] = safe_relative(RAW_CENSUS_PROFILE)
clean["method_note"] = (
    "Census Profile 2021 census-division visible-minority / Indigenous identity variables. "
    "pct_black uses ID 1687 Black; pct_hispanic uses ID 1690 Latin American; "
    "pct_indigenous uses ID 1403 Indigenous identity; pct_asian is the sum of visible-minority "
    "Asian components 1685, 1686, 1688, 1691, 1692, 1693, and 1694."
)

ordered_cols = (
    identity_cols
    + main_feature_cols
    + audit_component_cols
    + other_cols
)

ordered_cols = [col for col in ordered_cols if col in clean.columns]
clean = clean[ordered_cols].copy()


# -----------------------------
# Validation
# -----------------------------

if len(clean) != EXPECTED_QC_CD_COUNT:
    raise ValueError(f"Clean output has {len(clean)} rows; expected {EXPECTED_QC_CD_COUNT}.")

if clean["census_division_dguid"].duplicated().any():
    dupes = clean[clean["census_division_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicate census_division_dguid values in clean output:\n"
        + dupes[["census_division_code", "census_division_dguid", "census_division_name"]].to_string(index=False)
    )

required_features = main_feature_cols + audit_component_cols

for col in required_features:
    missing = int(clean[col].isna().sum())
    if missing != 0:
        raise ValueError(f"Unexpected missing values in {col}: {missing}")

# Check pct_asian formula exactly against component sum.
asian_formula_check = clean[asian_component_cols].sum(axis=1, skipna=False)
formula_diff = (clean["pct_asian"] - asian_formula_check).abs().max(skipna=True)

if formula_diff != 0:
    raise ValueError(f"pct_asian formula check failed. Max difference: {formula_diff}")

# Defensive bounds.
for col in main_feature_cols + audit_component_cols:
    if (clean[col] < 0).any():
        raise ValueError(f"Negative values found in {col}.")
    if (clean[col] > 100).any():
        raise ValueError(f"Values over 100 found in {col}.")

base_names_with_mojibake = contains_mojibake(base["census_division_name"])
clean_names_with_mojibake = contains_mojibake(clean["census_division_name"])
profile_names_with_mojibake = contains_mojibake(component_long["profile_geo_name"])


# -----------------------------
# Metadata
# -----------------------------

metadata_rows = [
    {
        "variable": "pct_black",
        "original_sovi_code": "PCTBLACK90",
        "source_characteristic_id": "1687",
        "source_characteristic_name": "Black",
        "unit": "percent",
        "derivation": "direct_rate",
        "role": "visible_minority_population_group_proxy",
        "notes": "Uses visible-minority / population-group row Black. Avoids Blackfoot false match.",
    },
    {
        "variable": "pct_hispanic",
        "original_sovi_code": "PCTHISPANIC90",
        "source_characteristic_id": "1690",
        "source_characteristic_name": "Latin American",
        "unit": "percent",
        "derivation": "direct_rate",
        "role": "latin_american_proxy_for_hispanic",
        "notes": "Canadian proxy for U.S. Hispanic category.",
    },
    {
        "variable": "pct_indigenous",
        "original_sovi_code": "PCTINDIAN90",
        "source_characteristic_id": "1403",
        "source_characteristic_name": "Indigenous identity",
        "unit": "percent",
        "derivation": "direct_rate",
        "role": "indigenous_identity_proxy",
        "notes": "Broad Indigenous identity rate.",
    },
    {
        "variable": "pct_asian",
        "original_sovi_code": "PCTASIAN90",
        "source_characteristic_id": "1685 + 1686 + 1688 + 1691 + 1692 + 1693 + 1694",
        "source_characteristic_name": (
            "South Asian + Chinese + Filipino + Southeast Asian + West Asian + Korean + Japanese"
        ),
        "unit": "percent",
        "derivation": "sum_of_visible_minority_component_rates",
        "role": "derived_asian_visible_minority_proxy",
        "notes": (
            "Derived Canadian visible-minority Asian-group proxy. Avoids language rows such as "
            "Chinese languages and Tagalog."
        ),
    },
]

for alias, component in ASIAN_COMPONENTS.items():
    metadata_rows.append(
        {
            "variable": alias,
            "original_sovi_code": "PCTASIAN90_COMPONENT_AUDIT",
            "source_characteristic_id": component["characteristic_id"],
            "source_characteristic_name": component["expected_name"],
            "unit": "percent",
            "derivation": "direct_component_rate",
            "role": "pct_asian_component_audit",
            "notes": "Retained for audit and reproducibility of pct_asian.",
        }
    )

metadata = pd.DataFrame(metadata_rows)
metadata.to_csv(OUTPUT_METADATA, index=False, encoding="utf-8")


# -----------------------------
# Summary
# -----------------------------

summary_rows = [
    {"metric": "raw_csv", "value": safe_relative(RAW_CENSUS_PROFILE)},
    {"metric": "base_cd_frame", "value": safe_relative(BASE_CD_FRAME)},
    {"metric": "raw_encoding", "value": encoding},
    {"metric": "total_rows_scanned", "value": total_rows_scanned},
    {"metric": "quebec_cd_rows_scanned", "value": quebec_cd_rows_scanned},
    {"metric": "selected_forced_id_rows", "value": len(selected_rows)},
    {"metric": "expected_selected_forced_id_rows", "value": expected_selected_rows},
    {"metric": "clean_rows", "value": len(clean)},
    {"metric": "unique_census_divisions", "value": clean["census_division_dguid"].nunique()},
    {"metric": "variables_cleaned", "value": ", ".join(main_feature_cols)},
    {"metric": "asian_components_cleaned", "value": ", ".join(asian_component_cols)},
    {"metric": "all_main_variables_complete", "value": bool(clean[main_feature_cols].notna().all().all())},
    {"metric": "all_asian_components_complete", "value": bool(clean[asian_component_cols].notna().all().all())},
    {"metric": "pct_asian_formula_max_abs_difference", "value": formula_diff},
    {"metric": "base_names_with_mojibake", "value": base_names_with_mojibake},
    {"metric": "clean_names_with_mojibake", "value": clean_names_with_mojibake},
    {"metric": "profile_names_with_mojibake", "value": profile_names_with_mojibake},
]

for variable in main_feature_cols + asian_component_cols:
    stats = numeric_summary(clean[variable])

    for key, value in stats.items():
        summary_rows.append(
            {
                "metric": f"{variable}_{key}",
                "value": value,
            }
        )

summary_rows.append(
    {
        "metric": "recommended_next_step",
        "value": (
            "If the summary shows 98 rows, complete main variables, complete Asian components, "
            "and no mojibake, generate the README and then update the SoVI input-source inspection "
            "to map pct_black, pct_indigenous, pct_asian, and pct_hispanic."
        ),
    }
)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Save clean output
# -----------------------------

clean.to_csv(OUTPUT_CLEAN, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CLEAN CENSUS DIVISION ETHNOCULTURAL / INDIGENOUS IDENTITY 2021")
print("=" * 72)

print("\nClean output:")
print("Rows:", len(clean))
print("Unique census divisions:", clean["census_division_dguid"].nunique())
print("Variables cleaned:", ", ".join(main_feature_cols))
print("Asian components:", ", ".join(asian_component_cols))
print("All main variables complete:", bool(clean[main_feature_cols].notna().all().all()))
print("All Asian components complete:", bool(clean[asian_component_cols].notna().all().all()))
print("pct_asian formula max abs difference:", formula_diff)

print("\nMojibake check:")
print("Base names with mojibake:", base_names_with_mojibake)
print("Clean names with mojibake:", clean_names_with_mojibake)
print("Profile names with mojibake:", profile_names_with_mojibake)

print("\nMain variable summaries:")
for variable in main_feature_cols:
    print("\n", variable)
    print(clean[variable].describe())

print("\nPreview:")
preview_cols = [
    "census_division_code",
    "census_division_name",
    "pct_black",
    "pct_hispanic",
    "pct_indigenous",
    "pct_asian",
]
print(clean[preview_cols].head(20).to_string(index=False))

print("\nSaved:")
print(OUTPUT_CLEAN)
print(OUTPUT_COMPONENT_LONG)
print(OUTPUT_METADATA)
print(OUTPUT_SUMMARY)

print("\nDone.")