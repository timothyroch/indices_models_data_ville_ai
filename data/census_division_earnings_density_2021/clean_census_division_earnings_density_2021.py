from pathlib import Path
import codecs
import re
import pandas as pd


# ============================================================
# Clean Census Division Earnings Density 2021
# ============================================================
#
# Purpose:
#   Clean a Québec census-division Canadian proxy for:
#
#       EARNDEN90 -> earnings_density
#
# Original SoVI concept:
#   Earnings, in $1,000, in all industries per square mile.
#
# Canadian adaptation:
#   The 2021 Census Profile did not expose a direct aggregate employment
#   income row in the inspected census-division file. Therefore, this cleaner
#   reconstructs estimated aggregate employment income from:
#
#       number of employment income recipients
#       *
#       average employment income among recipients
#
# Main default:
#   CHARACTERISTIC_ID = 133
#       Number of employment income recipients aged 15 years and over
#       in private households in 2020 - 25% sample data
#
#   CHARACTERISTIC_ID = 134
#       Average employment income in 2020 among recipients ($)
#
# Main formula:
#   estimated_aggregate_employment_income_2020 =
#       recipients_2020 * average_employment_income_2020
#
#   earnings_density =
#       estimated_aggregate_employment_income_2020 / land_area_km2
#
# Original-style audit:
#   earnings_density_thousands_per_square_mile =
#       (estimated_aggregate_employment_income_2020 / 1000)
#       / land_area_square_miles
#
# Audit sensitivity:
#   The analogous 2019 pair is retained:
#
#       ID 224 * ID 225
#
# Important:
#   This is a derived proxy, not a direct aggregate income table row.
#
# Run from data/:
#
#   python census_division_earnings_density_2021/clean_census_division_earnings_density_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_earnings_density_2021"
OUTPUT_DIR = SECTION_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_PROFILE = (
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

OUTPUT_CLEAN = OUTPUT_DIR / "clean_census_division_earnings_density_2021.csv"
OUTPUT_SOURCE_ROWS = OUTPUT_DIR / "clean_census_division_earnings_density_source_rows_2021.csv"
OUTPUT_METADATA = OUTPUT_DIR / "clean_census_division_earnings_density_variable_metadata_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "clean_census_division_earnings_density_summary_2021.csv"


# -----------------------------
# Config
# -----------------------------

EXPECTED_QC_CD_COUNT = 98

ENCODING_CANDIDATES = [
    "utf-8-sig",
    "utf-8",
    "cp1252",
    "latin1",
]

KM2_TO_SQUARE_MILES = 0.3861021585424458

CHARACTERISTICS = {
    "employment_income_recipients_2020_25pct": {
        "id": "133",
        "expected_name_contains": (
            "Number of employment income recipients aged 15 years and over "
            "in private households in 2020 - 25% sample data"
        ),
        "role": "main_default_count_component",
        "value_label": "recipients",
    },
    "average_employment_income_2020_25pct": {
        "id": "134",
        "expected_name_contains": "Average employment income in 2020 among recipients ($)",
        "role": "main_default_average_component",
        "value_label": "dollars",
    },
    "employment_income_recipients_2019_25pct": {
        "id": "224",
        "expected_name_contains": (
            "Number of employment income recipients aged 15 years and over "
            "in private households in 2019 - 25% sample data"
        ),
        "role": "audit_2019_count_component",
        "value_label": "recipients",
    },
    "average_employment_income_2019_25pct": {
        "id": "225",
        "expected_name_contains": "Average employment income in 2019 among recipients ($)",
        "role": "audit_2019_average_component",
        "value_label": "dollars",
    },
}

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

def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        return str(path)


def detect_file_encoding_strict(path: Path, encodings: list[str]) -> str:
    for encoding in encodings:
        decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
        try:
            with path.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    decoder.decode(chunk)
                decoder.decode(b"", final=True)
            return encoding
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not strictly decode {path} with candidates {encodings}",
    )


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = text.replace("\u00a0", " ")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_lower(value: object) -> str:
    return normalize_text(value).lower()


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def contains_mojibake(series: pd.Series) -> int:
    return int(series.astype("string").str.contains("Ã|Â|�", regex=True, na=False).sum())


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {col.lower(): col for col in columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for candidate in candidates:
        candidate_lower = candidate.lower()
        for col in columns:
            if candidate_lower in col.lower():
                return col

    return None


def require_column(columns: list[str], candidates: list[str], label: str) -> str:
    col = find_column(columns, candidates)
    if col is None:
        raise ValueError(
            f"Could not detect {label} column.\nCandidates: {candidates}\n\nAvailable columns:\n"
            + "\n".join(columns)
        )
    return col


def numeric_summary(series: pd.Series) -> dict:
    values = clean_numeric(series)
    return {
        "non_missing": int(values.notna().sum()),
        "missing": int(values.isna().sum()),
        "min": values.min(skipna=True),
        "max": values.max(skipna=True),
        "mean": values.mean(skipna=True),
        "median": values.median(skipna=True),
    }


def extract_characteristic(
    qc_rows: pd.DataFrame,
    alias: str,
    config: dict,
    dguid_col: str,
    geo_col: str | None,
    char_id_col: str,
    char_name_col: str,
    value_col: str,
    symbol_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    char_id = str(config["id"])

    rows = qc_rows[qc_rows[char_id_col].astype(str).str.strip() == char_id].copy()

    if rows.empty:
        raise ValueError(f"No rows found for characteristic ID {char_id} ({alias}).")

    rows[char_name_col] = rows[char_name_col].map(normalize_text)
    unique_names = sorted(rows[char_name_col].dropna().astype(str).unique())

    if len(unique_names) != 1:
        raise ValueError(
            f"Expected one characteristic name for ID {char_id}, got:\n"
            + "\n".join(unique_names)
        )

    characteristic_name = unique_names[0]
    expected_fragment = normalize_lower(config["expected_name_contains"])

    if expected_fragment not in normalize_lower(characteristic_name):
        raise ValueError(
            f"Characteristic ID {char_id} name does not match expectation.\n"
            f"Expected to contain: {expected_fragment}\n"
            f"Actual: {characteristic_name}"
        )

    source_cols = [dguid_col]
    if geo_col and geo_col in rows.columns:
        source_cols.append(geo_col)

    source_cols += [char_id_col, char_name_col, value_col]

    if symbol_col and symbol_col in rows.columns:
        source_cols.append(symbol_col)

    source_long = rows[source_cols].copy()
    source_long.insert(0, "source_alias", alias)
    source_long.insert(1, "source_role", config["role"])
    source_long["source_value_numeric"] = clean_numeric(source_long[value_col])

    merge_cols = [dguid_col, char_id_col, char_name_col, value_col]
    if symbol_col and symbol_col in rows.columns:
        merge_cols.append(symbol_col)

    out = rows[merge_cols].copy()
    out[dguid_col] = out[dguid_col].astype("string").str.strip()
    out[value_col] = clean_numeric(out[value_col])

    duplicate_dguid_count = int(out[dguid_col].duplicated().sum())
    if duplicate_dguid_count != 0:
        raise ValueError(
            f"Duplicate DGUID rows for characteristic ID {char_id} ({alias}): {duplicate_dguid_count}"
        )

    out = out.rename(
        columns={
            dguid_col: "census_division_dguid",
            char_id_col: f"{alias}_characteristic_id",
            char_name_col: f"{alias}_characteristic_name",
            value_col: f"{alias}_value",
        }
    )

    if symbol_col and symbol_col in rows.columns:
        out = out.rename(columns={symbol_col: f"{alias}_symbol"})

    values = clean_numeric(out[f"{alias}_value"])

    inventory = {
        "alias": alias,
        "characteristic_id": char_id,
        "characteristic_name": characteristic_name,
        "role": config["role"],
        "value_label": config["value_label"],
        "rows_extracted": len(out),
        "unique_census_divisions": out["census_division_dguid"].nunique(),
        "value_non_missing": int(values.notna().sum()),
        "value_missing": int(values.isna().sum()),
        "value_min": values.min(skipna=True),
        "value_max": values.max(skipna=True),
        "value_mean": values.mean(skipna=True),
        "value_median": values.median(skipna=True),
        "coverage_is_98_cds": out["census_division_dguid"].nunique() == EXPECTED_QC_CD_COUNT,
    }

    return out, source_long, inventory


# -----------------------------
# Validate inputs
# -----------------------------

if not RAW_PROFILE.exists():
    raise FileNotFoundError(f"Missing Census Profile raw CSV:\n{RAW_PROFILE}")

if not BASE_CD_FRAME.exists():
    raise FileNotFoundError(f"Missing base CD frame:\n{BASE_CD_FRAME}")


# -----------------------------
# Load base frame
# -----------------------------

base_encoding = detect_file_encoding_strict(BASE_CD_FRAME, ENCODING_CANDIDATES)

base = pd.read_csv(
    BASE_CD_FRAME,
    encoding=base_encoding,
    dtype=str,
    low_memory=False,
)

base.columns = [str(col).strip() for col in base.columns]

required_base_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "population_total_2021",
    "land_area_km2",
]

missing_base_cols = [col for col in required_base_cols if col not in base.columns]
if missing_base_cols:
    raise ValueError(
        "Base frame is missing required columns:\n"
        + "\n".join(missing_base_cols)
        + "\n\nAvailable columns:\n"
        + "\n".join(base.columns)
    )

base = base.copy()
base["census_division_dguid"] = base["census_division_dguid"].astype("string").str.strip()
base["population_total_2021"] = clean_numeric(base["population_total_2021"])
base["land_area_km2"] = clean_numeric(base["land_area_km2"])

if len(base) != EXPECTED_QC_CD_COUNT:
    raise ValueError(f"Expected {EXPECTED_QC_CD_COUNT} Québec CDs, got {len(base)}.")

if base["census_division_dguid"].duplicated().any():
    raise ValueError("Duplicate census_division_dguid values in base frame.")

if base["land_area_km2"].isna().any() or (base["land_area_km2"] <= 0).any():
    raise ValueError("Missing or non-positive land_area_km2 values in base frame.")

if base["population_total_2021"].isna().any() or (base["population_total_2021"] <= 0).any():
    raise ValueError("Missing or non-positive population_total_2021 values in base frame.")

base_dguid_set = set(base["census_division_dguid"].dropna().astype(str))


# -----------------------------
# Load Census Profile with strict full-file encoding detection
# -----------------------------

raw_encoding = detect_file_encoding_strict(RAW_PROFILE, ENCODING_CANDIDATES)

raw = pd.read_csv(
    RAW_PROFILE,
    encoding=raw_encoding,
    dtype=str,
    low_memory=False,
)

raw.columns = [str(col).strip() for col in raw.columns]
columns = list(raw.columns)

dguid_col = require_column(columns, ["DGUID", "dguid"], "DGUID")
geo_col = find_column(columns, ["GEO", "Geography", "Geography name"])
char_id_col = require_column(columns, ["CHARACTERISTIC_ID", "Characteristic ID"], "characteristic ID")
char_name_col = require_column(columns, ["CHARACTERISTIC_NAME", "Characteristic Name"], "characteristic name")
value_col = require_column(columns, ["C1_COUNT_TOTAL", "Count total", "Total - Count"], "main value/count")
symbol_col = find_column(columns, ["SYMBOL.1", "C1_SYMBOL_TOTAL", "SYMBOL"])

raw[dguid_col] = raw[dguid_col].astype("string").str.strip()
qc_rows = raw[raw[dguid_col].isin(base_dguid_set)].copy()

if qc_rows.empty:
    raise ValueError("No Québec CD rows found in Census Profile raw file after DGUID filtering.")


print("\nCleaning Census Division Earnings Density 2021")
print("Raw profile:", safe_relative(RAW_PROFILE))
print("Raw encoding:", raw_encoding)
print("Base frame:", safe_relative(BASE_CD_FRAME))
print("Base encoding:", base_encoding)
print("Base rows:", len(base))


# -----------------------------
# Extract forced characteristics
# -----------------------------

source_frames = []
source_long_frames = []
inventory_rows = []

for alias, config in CHARACTERISTICS.items():
    extracted, source_long, inventory = extract_characteristic(
        qc_rows=qc_rows,
        alias=alias,
        config=config,
        dguid_col=dguid_col,
        geo_col=geo_col,
        char_id_col=char_id_col,
        char_name_col=char_name_col,
        value_col=value_col,
        symbol_col=symbol_col,
    )

    source_frames.append(extracted)
    source_long_frames.append(source_long)
    inventory_rows.append(inventory)

    print(
        f"  {alias}: ID {inventory['characteristic_id']}, "
        f"rows={inventory['rows_extracted']}, "
        f"value_non_missing={inventory['value_non_missing']}"
    )

source_rows = pd.concat(source_long_frames, ignore_index=True, sort=False)
source_rows.to_csv(OUTPUT_SOURCE_ROWS, index=False, encoding="utf-8")

inventory = pd.DataFrame(inventory_rows)


# -----------------------------
# Build clean table
# -----------------------------

identity_cols = [col for col in IDENTITY_COLUMNS if col in base.columns]
clean = base[identity_cols].copy()

for extracted in source_frames:
    clean = clean.merge(
        extracted,
        on="census_division_dguid",
        how="left",
        validate="one_to_one",
    )

# Rename value columns to readable components.
clean["employment_income_recipients_2020_25pct_count"] = clean[
    "employment_income_recipients_2020_25pct_value"
]
clean["average_employment_income_2020_25pct_dollars"] = clean[
    "average_employment_income_2020_25pct_value"
]
clean["employment_income_recipients_2019_25pct_count"] = clean[
    "employment_income_recipients_2019_25pct_value"
]
clean["average_employment_income_2019_25pct_dollars"] = clean[
    "average_employment_income_2019_25pct_value"
]

# Derived aggregate employment income.
clean["estimated_aggregate_employment_income_2020"] = (
    clean["employment_income_recipients_2020_25pct_count"]
    * clean["average_employment_income_2020_25pct_dollars"]
)

clean["estimated_aggregate_employment_income_2019"] = (
    clean["employment_income_recipients_2019_25pct_count"]
    * clean["average_employment_income_2019_25pct_dollars"]
)

# Area denominator.
clean["land_area_square_miles"] = clean["land_area_km2"] * KM2_TO_SQUARE_MILES

# Main SoVI variable: dollars per km2.
clean["earnings_density"] = (
    clean["estimated_aggregate_employment_income_2020"]
    / clean["land_area_km2"]
)

# Original-style audit: $1,000 per square mile.
clean["earnings_density_thousands_per_square_mile"] = (
    clean["estimated_aggregate_employment_income_2020"]
    / 1000
    / clean["land_area_square_miles"]
)

clean["earnings_density_thousands_per_km2"] = (
    clean["estimated_aggregate_employment_income_2020"]
    / 1000
    / clean["land_area_km2"]
)

# 2019 sensitivity.
clean["earnings_density_2019"] = (
    clean["estimated_aggregate_employment_income_2019"]
    / clean["land_area_km2"]
)

clean["earnings_density_2019_thousands_per_square_mile"] = (
    clean["estimated_aggregate_employment_income_2019"]
    / 1000
    / clean["land_area_square_miles"]
)

clean["earnings_density_2020_minus_2019"] = (
    clean["earnings_density"]
    - clean["earnings_density_2019"]
)

clean["estimated_aggregate_employment_income_2020_per_capita"] = (
    clean["estimated_aggregate_employment_income_2020"]
    / clean["population_total_2021"]
)

clean["estimated_aggregate_employment_income_2019_per_capita"] = (
    clean["estimated_aggregate_employment_income_2019"]
    / clean["population_total_2021"]
)

clean["source_file"] = safe_relative(RAW_PROFILE)
clean["method_note"] = (
    "earnings_density maps EARNDEN90 to a derived Canadian employment-income density proxy. "
    "The 2021 Census Profile did not provide a direct aggregate employment-income row in the inspected "
    "census-division file, so the numerator is reconstructed as characteristic ID 133, number of employment "
    "income recipients aged 15 years and over in private households in 2020 - 25% sample data, multiplied by "
    "characteristic ID 134, average employment income in 2020 among recipients. The main variable is dollars "
    "per square kilometre. An original-style audit variable in $1,000 per square mile is retained."
)


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

required_numeric_cols = [
    "population_total_2021",
    "land_area_km2",
    "land_area_square_miles",
    "employment_income_recipients_2020_25pct_count",
    "average_employment_income_2020_25pct_dollars",
    "employment_income_recipients_2019_25pct_count",
    "average_employment_income_2019_25pct_dollars",
    "estimated_aggregate_employment_income_2020",
    "estimated_aggregate_employment_income_2019",
    "earnings_density",
    "earnings_density_thousands_per_square_mile",
    "earnings_density_thousands_per_km2",
    "earnings_density_2019",
    "earnings_density_2019_thousands_per_square_mile",
    "earnings_density_2020_minus_2019",
    "estimated_aggregate_employment_income_2020_per_capita",
    "estimated_aggregate_employment_income_2019_per_capita",
]

for col in required_numeric_cols:
    clean[col] = clean_numeric(clean[col])
    missing = int(clean[col].isna().sum())
    if missing != 0:
        raise ValueError(f"Unexpected missing values in {col}: {missing}")

nonnegative_cols = [
    "employment_income_recipients_2020_25pct_count",
    "average_employment_income_2020_25pct_dollars",
    "employment_income_recipients_2019_25pct_count",
    "average_employment_income_2019_25pct_dollars",
    "estimated_aggregate_employment_income_2020",
    "estimated_aggregate_employment_income_2019",
    "earnings_density",
    "earnings_density_thousands_per_square_mile",
    "earnings_density_thousands_per_km2",
    "earnings_density_2019",
    "earnings_density_2019_thousands_per_square_mile",
]

for col in nonnegative_cols:
    if (clean[col] < 0).any():
        raise ValueError(f"Negative values found in {col}.")

aggregate_2020_formula_diff = (
    clean["estimated_aggregate_employment_income_2020"]
    - clean["employment_income_recipients_2020_25pct_count"]
    * clean["average_employment_income_2020_25pct_dollars"]
).abs().max(skipna=True)

aggregate_2019_formula_diff = (
    clean["estimated_aggregate_employment_income_2019"]
    - clean["employment_income_recipients_2019_25pct_count"]
    * clean["average_employment_income_2019_25pct_dollars"]
).abs().max(skipna=True)

earnings_density_formula_diff = (
    clean["earnings_density"]
    - clean["estimated_aggregate_employment_income_2020"] / clean["land_area_km2"]
).abs().max(skipna=True)

original_style_formula_diff = (
    clean["earnings_density_thousands_per_square_mile"]
    - clean["estimated_aggregate_employment_income_2020"] / 1000 / clean["land_area_square_miles"]
).abs().max(skipna=True)

tolerance = 1e-8

formula_diffs = {
    "aggregate_2020_formula_diff": aggregate_2020_formula_diff,
    "aggregate_2019_formula_diff": aggregate_2019_formula_diff,
    "earnings_density_formula_diff": earnings_density_formula_diff,
    "original_style_formula_diff": original_style_formula_diff,
}

for name, diff in formula_diffs.items():
    if pd.notna(diff) and diff > tolerance:
        raise ValueError(f"{name} exceeded tolerance {tolerance}: {diff}")

base_names_with_mojibake = contains_mojibake(base["census_division_name"])
clean_names_with_mojibake = contains_mojibake(clean["census_division_name"])
raw_characteristic_names_with_mojibake = contains_mojibake(raw[char_name_col])

if raw_characteristic_names_with_mojibake != 0:
    raise ValueError("Mojibake detected in raw characteristic names after decoding.")


# -----------------------------
# Metadata
# -----------------------------

metadata_rows = [
    {
        "variable": "earnings_density",
        "original_sovi_code": "EARNDEN90",
        "description": "Derived estimated employment income density per square kilometre",
        "source_dataset": "2021 Census Profile, census divisions",
        "source_characteristic_ids": "133 * 134",
        "source_characteristic_names": (
            "Number of employment income recipients aged 15 years and over in private households in 2020 - 25% sample data; "
            "Average employment income in 2020 among recipients ($)"
        ),
        "source_column": "C1_COUNT_TOTAL",
        "denominator": "land_area_km2 from cleaned census-division base frame",
        "unit": "dollars_per_square_kilometre",
        "derivation": "(ID133 recipients * ID134 average employment income) / land_area_km2",
        "coverage": "98/98",
        "status": "ready_full_coverage_derived_proxy",
        "notes": (
            "Derived proxy because no direct aggregate employment-income row was found in the inspected Census Profile file. "
            "This is the default SoVI input column."
        ),
    },
    {
        "variable": "earnings_density_thousands_per_square_mile",
        "original_sovi_code": "EARNDEN90",
        "description": "Derived estimated employment income density in original SoVI-style units",
        "source_dataset": "2021 Census Profile, census divisions",
        "source_characteristic_ids": "133 * 134",
        "source_characteristic_names": (
            "Number of employment income recipients aged 15 years and over in private households in 2020 - 25% sample data; "
            "Average employment income in 2020 among recipients ($)"
        ),
        "source_column": "C1_COUNT_TOTAL",
        "denominator": "land_area_square_miles derived from land_area_km2",
        "unit": "thousand_dollars_per_square_mile",
        "derivation": "(ID133 recipients * ID134 average employment income / 1000) / land_area_square_miles",
        "coverage": "98/98",
        "status": "original_style_audit_variable",
        "notes": "Retained because original SoVI EARNDEN90 was expressed as earnings in $1,000 per square mile.",
    },
    {
        "variable": "estimated_aggregate_employment_income_2020",
        "original_sovi_code": "",
        "description": "Estimated aggregate employment income in 2020",
        "source_dataset": "2021 Census Profile, census divisions",
        "source_characteristic_ids": "133 * 134",
        "source_characteristic_names": (
            "Number of employment income recipients aged 15 years and over in private households in 2020 - 25% sample data; "
            "Average employment income in 2020 among recipients ($)"
        ),
        "source_column": "C1_COUNT_TOTAL",
        "denominator": "",
        "unit": "dollars",
        "derivation": "employment_income_recipients_2020_25pct_count * average_employment_income_2020_25pct_dollars",
        "coverage": "98/98",
        "status": "component_audit_variable",
        "notes": "Estimated aggregate numerator used for earnings_density.",
    },
    {
        "variable": "earnings_density_2019",
        "original_sovi_code": "",
        "description": "Derived estimated 2019 employment income density per square kilometre",
        "source_dataset": "2021 Census Profile, census divisions",
        "source_characteristic_ids": "224 * 225",
        "source_characteristic_names": (
            "Number of employment income recipients aged 15 years and over in private households in 2019 - 25% sample data; "
            "Average employment income in 2019 among recipients ($)"
        ),
        "source_column": "C1_COUNT_TOTAL",
        "denominator": "land_area_km2",
        "unit": "dollars_per_square_kilometre",
        "derivation": "(ID224 recipients * ID225 average employment income) / land_area_km2",
        "coverage": "98/98",
        "status": "audit_sensitivity_variable",
        "notes": "Retained as a pre-2020 sensitivity check.",
    },
]

metadata = pd.DataFrame(metadata_rows)
metadata.to_csv(OUTPUT_METADATA, index=False, encoding="utf-8")


# -----------------------------
# Summary
# -----------------------------

summary_rows = [
    {"metric": "raw_profile_csv", "value": safe_relative(RAW_PROFILE)},
    {"metric": "base_cd_frame", "value": safe_relative(BASE_CD_FRAME)},
    {"metric": "raw_encoding", "value": raw_encoding},
    {"metric": "base_encoding", "value": base_encoding},
    {"metric": "raw_rows", "value": len(raw)},
    {"metric": "quebec_cd_rows_scanned", "value": len(qc_rows)},
    {"metric": "clean_rows", "value": len(clean)},
    {"metric": "unique_census_divisions", "value": clean["census_division_dguid"].nunique()},
    {"metric": "variables_cleaned", "value": "earnings_density"},
    {"metric": "main_default_characteristic_ids", "value": "133, 134"},
    {"metric": "audit_characteristic_ids", "value": "224, 225"},
    {"metric": "all_required_numeric_columns_complete", "value": bool(clean[required_numeric_cols].notna().all().all())},
    {"metric": "aggregate_2020_formula_max_abs_difference", "value": aggregate_2020_formula_diff},
    {"metric": "aggregate_2019_formula_max_abs_difference", "value": aggregate_2019_formula_diff},
    {"metric": "earnings_density_formula_max_abs_difference", "value": earnings_density_formula_diff},
    {"metric": "original_style_formula_max_abs_difference", "value": original_style_formula_diff},
    {"metric": "base_names_with_mojibake", "value": base_names_with_mojibake},
    {"metric": "clean_names_with_mojibake", "value": clean_names_with_mojibake},
    {"metric": "raw_characteristic_names_with_mojibake", "value": raw_characteristic_names_with_mojibake},
]

for _, row in inventory.iterrows():
    summary_rows.append({"metric": f"{row['alias']}_characteristic_id", "value": row["characteristic_id"]})
    summary_rows.append({"metric": f"{row['alias']}_characteristic_name", "value": row["characteristic_name"]})
    summary_rows.append({"metric": f"{row['alias']}_coverage_is_98_cds", "value": row["coverage_is_98_cds"]})
    summary_rows.append({"metric": f"{row['alias']}_value_non_missing", "value": row["value_non_missing"]})

for variable in required_numeric_cols:
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
            "If this summary shows 98 rows, full numeric coverage, zero or near-zero formula differences, "
            "and no mojibake, generate the README and add a SoVI YAML mapping for EARNDEN90 -> earnings_density."
        ),
    }
)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Save clean output
# -----------------------------

ordered_cols = (
    identity_cols
    + [
        "earnings_density",
        "earnings_density_thousands_per_square_mile",
        "earnings_density_thousands_per_km2",
        "estimated_aggregate_employment_income_2020",
        "employment_income_recipients_2020_25pct_count",
        "average_employment_income_2020_25pct_dollars",
        "estimated_aggregate_employment_income_2020_per_capita",
        "earnings_density_2019",
        "earnings_density_2019_thousands_per_square_mile",
        "estimated_aggregate_employment_income_2019",
        "employment_income_recipients_2019_25pct_count",
        "average_employment_income_2019_25pct_dollars",
        "estimated_aggregate_employment_income_2019_per_capita",
        "earnings_density_2020_minus_2019",
        "land_area_square_miles",
    ]
)

for alias in CHARACTERISTICS:
    ordered_cols += [
        f"{alias}_characteristic_id",
        f"{alias}_characteristic_name",
    ]

ordered_cols += [
    "source_file",
    "method_note",
]

ordered_cols = [col for col in ordered_cols if col in clean.columns]
clean = clean[ordered_cols].copy()
clean.to_csv(OUTPUT_CLEAN, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CLEAN CENSUS DIVISION EARNINGS DENSITY 2021")
print("=" * 72)

print("\nClean output:")
print("Rows:", len(clean))
print("Unique census divisions:", clean["census_division_dguid"].nunique())
print("Variables cleaned: earnings_density")

print("\nFormula checks:")
print("aggregate 2020 formula max abs difference:", aggregate_2020_formula_diff)
print("aggregate 2019 formula max abs difference:", aggregate_2019_formula_diff)
print("earnings_density formula max abs difference:", earnings_density_formula_diff)
print("original-style formula max abs difference:", original_style_formula_diff)

print("\nMojibake check:")
print("Base names with mojibake:", base_names_with_mojibake)
print("Clean names with mojibake:", clean_names_with_mojibake)
print("Raw characteristic names with mojibake:", raw_characteristic_names_with_mojibake)

print("\nMain summaries:")
for variable in [
    "earnings_density",
    "earnings_density_thousands_per_square_mile",
    "estimated_aggregate_employment_income_2020",
    "estimated_aggregate_employment_income_2020_per_capita",
    "earnings_density_2019",
    "earnings_density_2020_minus_2019",
]:
    print("\n", variable)
    print(clean[variable].describe())

print("\nSource characteristic inventory:")
print(inventory.to_string(index=False))

print("\nPreview:")
preview_cols = [
    "census_division_code",
    "census_division_name",
    "earnings_density",
    "earnings_density_thousands_per_square_mile",
    "employment_income_recipients_2020_25pct_count",
    "average_employment_income_2020_25pct_dollars",
    "estimated_aggregate_employment_income_2020",
]
preview_cols = [col for col in preview_cols if col in clean.columns]
print(clean[preview_cols].head(20).to_string(index=False))

print("\nSaved:")
print(OUTPUT_CLEAN)
print(OUTPUT_SOURCE_ROWS)
print(OUTPUT_METADATA)
print(OUTPUT_SUMMARY)

print("\nDone.")