from pathlib import Path
import pandas as pd


# ============================================================
# Clean Census Division Employment Sector 2021
# ============================================================
#
# Purpose:
#   Create clean Québec census-division features for SoVI-like employment
#   sector / occupation variables:
#
#       pct_extractive_employment
#       pct_transport_utility_employment
#       pct_service_employment
#
# Original SoVI variables:
#
#       AGRIPC90  -> pct_extractive_employment
#       TRANPC90  -> pct_transport_utility_employment
#       SERVPC90  -> pct_service_employment
#
# Methodological choices:
#
#   pct_extractive_employment:
#       NAICS industry sum:
#           2262  Agriculture, forestry, fishing and hunting
#         + 2263  Mining, quarrying, and oil and gas extraction
#
#   pct_transport_utility_employment:
#       NAICS-style industry sum:
#           2269  Transportation and warehousing
#         + 2270  Information and cultural industries
#         + 2264  Utilities
#
#       Note:
#           The inspection also found 2258 "Occupations in manufacturing and
#           utilities", but this cleaner intentionally does not use it because it
#           is an occupation row, not an industry row.
#
#   pct_service_employment:
#       NOC occupation row:
#           2255  Sales and service occupations
#
#       Note:
#           SERVPC90 is interpreted as service occupations, not service
#           industries. Therefore this cleaner does not sum retail, health care,
#           education, finance, accommodation, and other service-industry rows.
#
# Run from data/:
#   python census_division_employment_sector_2021/clean_census_division_employment_sector_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_employment_sector_2021"
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

OUTPUT_CLEAN = OUTPUT_DIR / "clean_census_division_employment_sector_2021.csv"
OUTPUT_COMPONENT_LONG = OUTPUT_DIR / "clean_census_division_employment_sector_component_long_2021.csv"
OUTPUT_METADATA = OUTPUT_DIR / "clean_census_division_employment_sector_variable_metadata_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "clean_census_division_employment_sector_summary_2021.csv"


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

COMPONENTS = {
    "pct_extractive_component_agriculture_forestry_fishing_hunting": {
        "canonical_variable": "pct_extractive_employment",
        "original_code": "AGRIPC90",
        "characteristic_id": "2262",
        "expected_name_contains": "Agriculture, forestry, fishing and hunting",
        "unit": "percent",
        "source_role": "naics_industry_component",
        "method_note": "Component of derived extractive / primary-industry employment proxy.",
    },
    "pct_extractive_component_mining_quarrying_oil_gas": {
        "canonical_variable": "pct_extractive_employment",
        "original_code": "AGRIPC90",
        "characteristic_id": "2263",
        "expected_name_contains": "Mining, quarrying, and oil and gas extraction",
        "unit": "percent",
        "source_role": "naics_industry_component",
        "method_note": "Component of derived extractive / primary-industry employment proxy.",
    },
    "pct_transport_utility_component_transportation_warehousing": {
        "canonical_variable": "pct_transport_utility_employment",
        "original_code": "TRANPC90",
        "characteristic_id": "2269",
        "expected_name_contains": "Transportation and warehousing",
        "unit": "percent",
        "source_role": "naics_industry_component",
        "method_note": "Component of derived transportation / utility / communication industry proxy.",
    },
    "pct_transport_utility_component_information_cultural_industries": {
        "canonical_variable": "pct_transport_utility_employment",
        "original_code": "TRANPC90",
        "characteristic_id": "2270",
        "expected_name_contains": "Information and cultural industries",
        "unit": "percent",
        "source_role": "naics_industry_component",
        "method_note": (
            "Component of derived transportation / utility / communication industry proxy. "
            "Used as the Canadian NAICS-style proxy for the communications part of the original concept."
        ),
    },
    "pct_transport_utility_component_utilities": {
        "canonical_variable": "pct_transport_utility_employment",
        "original_code": "TRANPC90",
        "characteristic_id": "2264",
        "expected_name_contains": "Utilities",
        "unit": "percent",
        "source_role": "naics_industry_component",
        "method_note": (
            "Component of derived transportation / utility / communication industry proxy. "
            "Uses the industry row Utilities, not the occupation row 'Occupations in manufacturing and utilities'."
        ),
    },
    "pct_service_component_sales_service_occupations": {
        "canonical_variable": "pct_service_employment",
        "original_code": "SERVPC90",
        "characteristic_id": "2255",
        "expected_name_contains": "Sales and service occupations",
        "unit": "percent",
        "source_role": "noc_occupation_row",
        "method_note": (
            "Main service-employment proxy. Uses the NOC occupation row Sales and service occupations, "
            "rather than a sum of service-industry rows."
        ),
    },
}

ALL_CHARACTERISTIC_IDS = [
    component["characteristic_id"]
    for component in COMPONENTS.values()
]

IDENTITY_COLUMNS = [
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


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        return str(path)


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


def contains_mojibake(series: pd.Series) -> int:
    text = series.astype("string")
    return int(text.str.contains("Ã|Â|�", regex=True, na=False).sum())


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
    expected_name_contains: str,
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

    actual_name = names[0].lower()
    expected = expected_name_contains.lower()

    if expected not in actual_name:
        raise ValueError(
            f"{alias} / CHARACTERISTIC_ID={characteristic_id} name mismatch. "
            f"Expected to contain '{expected_name_contains}', got '{names[0]}'."
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

print("\nCleaning Census Division Employment Sector 2021")
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

for alias, component in COMPONENTS.items():
    rows = selected_rows[
        selected_rows["CHARACTERISTIC_ID_STR"] == component["characteristic_id"]
    ].copy()

    validate_single_characteristic(
        df=rows,
        characteristic_id=component["characteristic_id"],
        expected_name_contains=component["expected_name_contains"],
        alias=alias,
    )


# -----------------------------
# Build component-long audit table
# -----------------------------

component_long_rows = []

for component_alias, component in COMPONENTS.items():
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

    temp["canonical_variable"] = component["canonical_variable"]
    temp["component_alias"] = component_alias
    temp["original_code"] = component["original_code"]
    temp["unit"] = component["unit"]
    temp["source_role"] = component["source_role"]
    temp["method_note"] = component["method_note"]
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

identity_cols = [col for col in IDENTITY_COLUMNS if col in base.columns]
clean = base[identity_cols].copy()

clean = clean.merge(
    rate_wide,
    on="census_division_dguid",
    how="left",
    validate="one_to_one",
)

extractive_component_cols = [
    "pct_extractive_component_agriculture_forestry_fishing_hunting",
    "pct_extractive_component_mining_quarrying_oil_gas",
]

transport_utility_component_cols = [
    "pct_transport_utility_component_transportation_warehousing",
    "pct_transport_utility_component_information_cultural_industries",
    "pct_transport_utility_component_utilities",
]

service_component_cols = [
    "pct_service_component_sales_service_occupations",
]

clean["pct_extractive_employment"] = clean[extractive_component_cols].sum(axis=1, skipna=False)

clean["pct_transport_utility_employment"] = clean[transport_utility_component_cols].sum(axis=1, skipna=False)

clean["pct_service_employment"] = clean["pct_service_component_sales_service_occupations"]

main_feature_cols = [
    "pct_extractive_employment",
    "pct_transport_utility_employment",
    "pct_service_employment",
]

audit_component_cols = (
    extractive_component_cols
    + transport_utility_component_cols
    + service_component_cols
)

clean["source_file"] = safe_relative(RAW_CENSUS_PROFILE)
clean["method_note"] = (
    "Employment-sector variables derived from 2021 Census Profile census-division rows. "
    "pct_extractive_employment is the sum of NAICS rows 2262 and 2263. "
    "pct_transport_utility_employment is the sum of NAICS-style rows 2269, 2270, and 2264. "
    "pct_service_employment uses NOC occupation row 2255 Sales and service occupations. "
    "The cleaner intentionally avoids using 2258 Occupations in manufacturing and utilities for TRANPC90 "
    "because that row is occupation-based rather than industry-based."
)

ordered_cols = (
    identity_cols
    + main_feature_cols
    + audit_component_cols
    + ["source_file", "method_note"]
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

    negative = int((clean[col] < 0).sum())
    if negative != 0:
        raise ValueError(f"Unexpected negative values in {col}: {negative}")

    over_100 = int((clean[col] > 100).sum())
    if over_100 != 0:
        raise ValueError(f"Values over 100 found in {col}: {over_100}")

extractive_formula_diff = (
    clean["pct_extractive_employment"]
    - clean[extractive_component_cols].sum(axis=1, skipna=False)
).abs().max(skipna=True)

transport_utility_formula_diff = (
    clean["pct_transport_utility_employment"]
    - clean[transport_utility_component_cols].sum(axis=1, skipna=False)
).abs().max(skipna=True)

service_formula_diff = (
    clean["pct_service_employment"]
    - clean["pct_service_component_sales_service_occupations"]
).abs().max(skipna=True)

if extractive_formula_diff != 0:
    raise ValueError(f"pct_extractive_employment formula check failed: {extractive_formula_diff}")

if transport_utility_formula_diff != 0:
    raise ValueError(f"pct_transport_utility_employment formula check failed: {transport_utility_formula_diff}")

if service_formula_diff != 0:
    raise ValueError(f"pct_service_employment formula check failed: {service_formula_diff}")

base_names_with_mojibake = contains_mojibake(base["census_division_name"])
clean_names_with_mojibake = contains_mojibake(clean["census_division_name"])
profile_names_with_mojibake = contains_mojibake(component_long["profile_geo_name"])


# -----------------------------
# Metadata
# -----------------------------

metadata_rows = [
    {
        "variable": "pct_extractive_employment",
        "original_sovi_code": "AGRIPC90",
        "source_characteristic_id": "2262 + 2263",
        "source_characteristic_name": (
            "Agriculture, forestry, fishing and hunting + "
            "Mining, quarrying, and oil and gas extraction"
        ),
        "unit": "percent",
        "derivation": "sum_of_naics_industry_component_rates",
        "role": "recommended_sovi_extractive_employment_proxy",
        "notes": "Derived primary/extractive industry employment proxy.",
    },
    {
        "variable": "pct_transport_utility_employment",
        "original_sovi_code": "TRANPC90",
        "source_characteristic_id": "2269 + 2270 + 2264",
        "source_characteristic_name": (
            "Transportation and warehousing + Information and cultural industries + Utilities"
        ),
        "unit": "percent",
        "derivation": "sum_of_naics_industry_component_rates",
        "role": "recommended_sovi_transport_utility_employment_proxy",
        "notes": (
            "Derived industry proxy for transportation, communications, and utilities. "
            "Uses Utilities industry row 2264, not occupation row 2258."
        ),
    },
    {
        "variable": "pct_service_employment",
        "original_sovi_code": "SERVPC90",
        "source_characteristic_id": "2255",
        "source_characteristic_name": "Sales and service occupations",
        "unit": "percent",
        "derivation": "direct_occupation_rate",
        "role": "recommended_sovi_service_employment_proxy",
        "notes": (
            "Uses NOC occupation row Sales and service occupations. This preserves the service-occupation "
            "interpretation rather than converting the variable into a service-industry sum."
        ),
    },
]

for alias, component in COMPONENTS.items():
    metadata_rows.append(
        {
            "variable": alias,
            "original_sovi_code": f"{component['original_code']}_COMPONENT_AUDIT",
            "source_characteristic_id": component["characteristic_id"],
            "source_characteristic_name": component["expected_name_contains"],
            "unit": component["unit"],
            "derivation": "direct_component_rate",
            "role": component["source_role"],
            "notes": component["method_note"],
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
    {"metric": "components_cleaned", "value": ", ".join(audit_component_cols)},
    {"metric": "all_main_variables_complete", "value": bool(clean[main_feature_cols].notna().all().all())},
    {"metric": "all_components_complete", "value": bool(clean[audit_component_cols].notna().all().all())},
    {"metric": "pct_extractive_employment_formula_max_abs_difference", "value": extractive_formula_diff},
    {"metric": "pct_transport_utility_employment_formula_max_abs_difference", "value": transport_utility_formula_diff},
    {"metric": "pct_service_employment_formula_max_abs_difference", "value": service_formula_diff},
    {"metric": "base_names_with_mojibake", "value": base_names_with_mojibake},
    {"metric": "clean_names_with_mojibake", "value": clean_names_with_mojibake},
    {"metric": "profile_names_with_mojibake", "value": profile_names_with_mojibake},
]

for variable in main_feature_cols + audit_component_cols:
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
            "If this summary shows 98 rows, complete main variables, complete components, "
            "zero formula differences, and no mojibake, generate the README and add a SoVI YAML mapping "
            "for pct_extractive_employment, pct_transport_utility_employment, and pct_service_employment."
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
print("CLEAN CENSUS DIVISION EMPLOYMENT SECTOR 2021")
print("=" * 72)

print("\nClean output:")
print("Rows:", len(clean))
print("Unique census divisions:", clean["census_division_dguid"].nunique())
print("Variables cleaned:", ", ".join(main_feature_cols))
print("Components:", ", ".join(audit_component_cols))
print("All main variables complete:", bool(clean[main_feature_cols].notna().all().all()))
print("All components complete:", bool(clean[audit_component_cols].notna().all().all()))

print("\nFormula checks:")
print("pct_extractive_employment max abs difference:", extractive_formula_diff)
print("pct_transport_utility_employment max abs difference:", transport_utility_formula_diff)
print("pct_service_employment max abs difference:", service_formula_diff)

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
    "pct_extractive_employment",
    "pct_transport_utility_employment",
    "pct_service_employment",
]
print(clean[preview_cols].head(20).to_string(index=False))

print("\nSaved:")
print(OUTPUT_CLEAN)
print(OUTPUT_COMPONENT_LONG)
print(OUTPUT_METADATA)
print(OUTPUT_SUMMARY)

print("\nDone.")