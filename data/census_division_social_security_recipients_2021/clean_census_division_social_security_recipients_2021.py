from pathlib import Path
import codecs
import re
import pandas as pd


# ============================================================
# Clean Census Division Social Security Recipients 2021
# ============================================================
#
# Purpose:
#   Clean a Québec census-division Canadian proxy for:
#
#       SSBENPC90 -> social_security_recipients_per_capita
#
# Original SoVI concept:
#   Per capita Social Security recipients.
#
# Canadian adaptation:
#   The 2021 Canadian Census Profile does not expose a direct U.S.-style
#   Social Security recipient variable at census-division level. The selected
#   proxy is the number of government transfer recipients aged 15 years and
#   over in private households, divided by total 2021 population.
#
# Main default:
#   CHARACTERISTIC_ID = 213
#   "Number of government transfers recipients aged 15 years and over
#    in private households in 2019 - 100% data"
#
# Why 2019 default:
#   2020 transfer-recipient rows are temporally close to the 2021 Census but
#   include the pandemic income-transfer context. The 2019 100% row is used as
#   the default structural proxy, while 2020 and 25% rows are retained as audit
#   and sensitivity variables.
#
# Important:
#   This is a broad government-transfer proxy, not a direct public pension /
#   OAS / CPP / QPP recipient measure.
#
# Run from data/:
#
#   python census_division_social_security_recipients_2021/clean_census_division_social_security_recipients_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_social_security_recipients_2021"
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

OUTPUT_CLEAN = OUTPUT_DIR / "clean_census_division_social_security_recipients_2021.csv"
OUTPUT_SOURCE_ROWS = OUTPUT_DIR / "clean_census_division_social_security_recipients_source_rows_2021.csv"
OUTPUT_METADATA = OUTPUT_DIR / "clean_census_division_social_security_recipients_variable_metadata_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "clean_census_division_social_security_recipients_summary_2021.csv"


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

CHARACTERISTICS = {
    "government_transfer_recipients_2019_100pct": {
        "id": "213",
        "expected_name_contains": "number of government transfers recipients aged 15 years and over in private households in 2019 - 100% data",
        "role": "main_default_structural_proxy",
    },
    "government_transfer_recipients_2020_100pct": {
        "id": "120",
        "expected_name_contains": "number of government transfers recipients aged 15 years and over in private households in 2020 - 100% data",
        "role": "audit_temporally_closer_pandemic_context",
    },
    "government_transfer_recipients_2019_25pct": {
        "id": "226",
        "expected_name_contains": "number of government transfers recipients aged 15 years and over in private households in 2019 - 25% sample data",
        "role": "audit_sample_2019",
    },
    "government_transfer_recipients_2020_25pct": {
        "id": "135",
        "expected_name_contains": "number of government transfers recipients aged 15 years and over in private households in 2020 - 25% sample data",
        "role": "audit_sample_2020_pandemic_context",
    },
    "employment_insurance_recipients_2020_100pct": {
        "id": "122",
        "expected_name_contains": "number of employment insurance benefits recipients aged 15 years and over in private households in 2020 - 100% data",
        "role": "audit_narrow_employment_insurance_proxy",
    },
    "covid_benefit_recipients_2020_100pct": {
        "id": "124",
        "expected_name_contains": "number of covid-19 emergency and recovery benefits recipients aged 15 years and over in private households in 2020 - 100% data",
        "role": "audit_pandemic_benefit_context",
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
    """
    Strict full-file encoding detector.

    This deliberately scans the entire byte stream with an incremental decoder.
    It does not rely on pd.read_csv(..., nrows=...), because the first rows of
    a StatCan CSV can decode as UTF-8 while later accented bytes require cp1252.

    It also does not use encoding_errors='ignore' or 'replace', because that
    would silently corrupt characteristic names.
    """
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
    char_id_col: str,
    char_name_col: str,
    count_col: str,
    rate_col: str,
    count_symbol_col: str | None,
    rate_symbol_col: str | None,
) -> tuple[pd.DataFrame, dict]:
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

    out_cols = [
        dguid_col,
        char_id_col,
        char_name_col,
        count_col,
        rate_col,
    ]

    if count_symbol_col and count_symbol_col in rows.columns:
        out_cols.append(count_symbol_col)

    if rate_symbol_col and rate_symbol_col in rows.columns and rate_symbol_col not in out_cols:
        out_cols.append(rate_symbol_col)

    out = rows[out_cols].copy()
    out[dguid_col] = out[dguid_col].astype("string").str.strip()
    out[count_col] = clean_numeric(out[count_col])
    out[rate_col] = clean_numeric(out[rate_col])

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
            count_col: f"{alias}_count",
            rate_col: f"{alias}_source_rate",
        }
    )

    if count_symbol_col and count_symbol_col in rows.columns:
        out = out.rename(columns={count_symbol_col: f"{alias}_count_symbol"})

    if rate_symbol_col and rate_symbol_col in rows.columns:
        out = out.rename(columns={rate_symbol_col: f"{alias}_rate_symbol"})

    count_values = clean_numeric(out[f"{alias}_count"])
    rate_values = clean_numeric(out[f"{alias}_source_rate"])

    inventory = {
        "alias": alias,
        "characteristic_id": char_id,
        "characteristic_name": characteristic_name,
        "role": config["role"],
        "rows_extracted": len(out),
        "unique_census_divisions": out["census_division_dguid"].nunique(),
        "count_non_missing": int(count_values.notna().sum()),
        "count_missing": int(count_values.isna().sum()),
        "count_min": count_values.min(skipna=True),
        "count_max": count_values.max(skipna=True),
        "count_mean": count_values.mean(skipna=True),
        "count_median": count_values.median(skipna=True),
        "rate_non_missing": int(rate_values.notna().sum()),
        "rate_missing": int(rate_values.isna().sum()),
        "rate_min": rate_values.min(skipna=True),
        "rate_max": rate_values.max(skipna=True),
        "rate_mean": rate_values.mean(skipna=True),
        "rate_median": rate_values.median(skipna=True),
        "coverage_is_98_cds": out["census_division_dguid"].nunique() == EXPECTED_QC_CD_COUNT,
    }

    return out, inventory


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

base = pd.read_csv(BASE_CD_FRAME, dtype=str, low_memory=False)
base.columns = [str(col).strip() for col in base.columns]

required_base_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "population_total_2021",
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

if len(base) != EXPECTED_QC_CD_COUNT:
    raise ValueError(f"Expected {EXPECTED_QC_CD_COUNT} Québec CDs, got {len(base)}.")

if base["census_division_dguid"].duplicated().any():
    raise ValueError("Duplicate census_division_dguid values in base frame.")

if base["population_total_2021"].isna().any():
    raise ValueError("Missing population_total_2021 in base frame.")

if (base["population_total_2021"] <= 0).any():
    raise ValueError("Non-positive population_total_2021 values found in base frame.")

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
char_id_col = require_column(columns, ["CHARACTERISTIC_ID", "Characteristic ID"], "characteristic ID")
char_name_col = require_column(columns, ["CHARACTERISTIC_NAME", "Characteristic Name"], "characteristic name")
count_col = require_column(columns, ["C1_COUNT_TOTAL", "Count total", "Total - Count"], "count")
rate_col = require_column(columns, ["C10_RATE_TOTAL", "Rate total", "Total - Rate"], "rate")

count_symbol_col = find_column(columns, ["SYMBOL.1", "C1_SYMBOL_TOTAL", "SYMBOL"])
rate_symbol_col = find_column(columns, ["SYMBOL.3", "C10_SYMBOL_TOTAL", "SYMBOL"])

raw[dguid_col] = raw[dguid_col].astype("string").str.strip()
qc_rows = raw[raw[dguid_col].isin(base_dguid_set)].copy()

if qc_rows.empty:
    raise ValueError("No Québec CD rows found in Census Profile raw file after DGUID filtering.")


print("\nCleaning Census Division Social Security Recipients 2021")
print("Raw profile:", safe_relative(RAW_PROFILE))
print("Raw encoding:", raw_encoding)
print("Base frame:", safe_relative(BASE_CD_FRAME))
print("Base rows:", len(base))


# -----------------------------
# Extract forced characteristics
# -----------------------------

source_frames = []
inventory_rows = []

for alias, config in CHARACTERISTICS.items():
    extracted, inventory = extract_characteristic(
        qc_rows=qc_rows,
        alias=alias,
        config=config,
        dguid_col=dguid_col,
        char_id_col=char_id_col,
        char_name_col=char_name_col,
        count_col=count_col,
        rate_col=rate_col,
        count_symbol_col=count_symbol_col,
        rate_symbol_col=rate_symbol_col,
    )

    source_frames.append(extracted)
    inventory_rows.append(inventory)

    print(
        f"  {alias}: ID {inventory['characteristic_id']}, "
        f"rows={inventory['rows_extracted']}, "
        f"count_non_missing={inventory['count_non_missing']}, "
        f"rate_non_missing={inventory['rate_non_missing']}"
    )

source_rows = pd.concat(source_frames, ignore_index=False, sort=False)
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

# Main default variable.
clean["government_transfer_recipients_2019_100pct_count"] = clean[
    "government_transfer_recipients_2019_100pct_count"
]
clean["government_transfer_recipients_2020_100pct_count"] = clean[
    "government_transfer_recipients_2020_100pct_count"
]

clean["social_security_recipients_per_capita"] = (
    clean["government_transfer_recipients_2019_100pct_count"]
    / clean["population_total_2021"]
)

clean["social_security_recipients_per_1000"] = (
    1000 * clean["social_security_recipients_per_capita"]
)

# Audit/sensitivity variables.
clean["government_transfer_recipients_2020_100pct_per_capita"] = (
    clean["government_transfer_recipients_2020_100pct_count"]
    / clean["population_total_2021"]
)
clean["government_transfer_recipients_2019_25pct_per_capita"] = (
    clean["government_transfer_recipients_2019_25pct_count"]
    / clean["population_total_2021"]
)
clean["government_transfer_recipients_2020_25pct_per_capita"] = (
    clean["government_transfer_recipients_2020_25pct_count"]
    / clean["population_total_2021"]
)
clean["employment_insurance_recipients_2020_100pct_per_capita"] = (
    clean["employment_insurance_recipients_2020_100pct_count"]
    / clean["population_total_2021"]
)
clean["covid_benefit_recipients_2020_100pct_per_capita"] = (
    clean["covid_benefit_recipients_2020_100pct_count"]
    / clean["population_total_2021"]
)

clean["government_transfer_recipients_2020_minus_2019_count"] = (
    clean["government_transfer_recipients_2020_100pct_count"]
    - clean["government_transfer_recipients_2019_100pct_count"]
)
clean["government_transfer_recipients_2020_minus_2019_per_capita"] = (
    clean["government_transfer_recipients_2020_100pct_per_capita"]
    - clean["social_security_recipients_per_capita"]
)

clean["source_file"] = safe_relative(RAW_PROFILE)
clean["method_note"] = (
    "social_security_recipients_per_capita maps SSBENPC90 to a broad Canadian government-transfer "
    "recipient proxy. The default numerator is Census Profile characteristic ID 213, "
    "'Number of government transfers recipients aged 15 years and over in private households in 2019 - 100% data', "
    "divided by total 2021 census-division population. The 2019 row is used as the default to avoid making "
    "the main variable primarily reflect COVID-era emergency and recovery benefits; 2020 and related benefit rows "
    "are retained as audit variables."
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
    "social_security_recipients_per_capita",
    "social_security_recipients_per_1000",
    "government_transfer_recipients_2019_100pct_count",
    "government_transfer_recipients_2020_100pct_count",
    "government_transfer_recipients_2019_25pct_count",
    "government_transfer_recipients_2020_25pct_count",
    "employment_insurance_recipients_2020_100pct_count",
    "covid_benefit_recipients_2020_100pct_count",
    "government_transfer_recipients_2020_100pct_per_capita",
    "government_transfer_recipients_2019_25pct_per_capita",
    "government_transfer_recipients_2020_25pct_per_capita",
    "employment_insurance_recipients_2020_100pct_per_capita",
    "covid_benefit_recipients_2020_100pct_per_capita",
    "government_transfer_recipients_2020_minus_2019_count",
    "government_transfer_recipients_2020_minus_2019_per_capita",
]

for col in required_numeric_cols:
    clean[col] = clean_numeric(clean[col])
    missing = int(clean[col].isna().sum())
    if missing != 0:
        raise ValueError(f"Unexpected missing values in {col}: {missing}")

per_capita_cols = [
    "social_security_recipients_per_capita",
    "government_transfer_recipients_2020_100pct_per_capita",
    "government_transfer_recipients_2019_25pct_per_capita",
    "government_transfer_recipients_2020_25pct_per_capita",
    "employment_insurance_recipients_2020_100pct_per_capita",
    "covid_benefit_recipients_2020_100pct_per_capita",
]

for col in per_capita_cols:
    if (clean[col] < 0).any():
        raise ValueError(f"Negative values found in {col}.")
    if (clean[col] > 1).any():
        raise ValueError(f"Values over 1 found in {col}; check numerator/denominator choice.")

formula_diff = (
    clean["social_security_recipients_per_capita"]
    - clean["government_transfer_recipients_2019_100pct_count"] / clean["population_total_2021"]
).abs().max(skipna=True)

per_1000_formula_diff = (
    clean["social_security_recipients_per_1000"]
    - 1000 * clean["social_security_recipients_per_capita"]
).abs().max(skipna=True)

FLOAT_TOLERANCE = 1e-10

if formula_diff > FLOAT_TOLERANCE:
    raise ValueError(f"social_security_recipients_per_capita formula check failed: {formula_diff}")

if per_1000_formula_diff > FLOAT_TOLERANCE:
    raise ValueError(f"social_security_recipients_per_1000 formula check failed: {per_1000_formula_diff}")

base_names_with_mojibake = contains_mojibake(base["census_division_name"])
clean_names_with_mojibake = contains_mojibake(clean["census_division_name"])
raw_characteristic_names_with_mojibake = contains_mojibake(raw[char_name_col])


# -----------------------------
# Metadata
# -----------------------------

metadata_rows = [
    {
        "variable": "social_security_recipients_per_capita",
        "original_sovi_code": "SSBENPC90",
        "description": "Broad government-transfer recipients per total population",
        "source_dataset": "2021 Census Profile, census divisions",
        "source_characteristic_id": "213",
        "source_characteristic_name": "Number of government transfers recipients aged 15 years and over in private households in 2019 - 100% data",
        "source_column": "C1_COUNT_TOTAL",
        "denominator": "population_total_2021 from cleaned census-division base frame",
        "unit": "recipients_per_person",
        "derivation": "government_transfer_recipients_2019_100pct_count / population_total_2021",
        "coverage": "98/98",
        "status": "ready_full_coverage",
        "notes": (
            "Canadian proxy for U.S. Social Security recipients per capita. This is broader than Social Security "
            "because government transfers include multiple income-transfer programs. The 2019 100% row is used as "
            "default to reduce COVID-era transfer contamination."
        ),
    },
    {
        "variable": "social_security_recipients_per_1000",
        "original_sovi_code": "",
        "description": "Broad government-transfer recipients per 1,000 population",
        "source_dataset": "2021 Census Profile, census divisions",
        "source_characteristic_id": "213",
        "source_characteristic_name": "Number of government transfers recipients aged 15 years and over in private households in 2019 - 100% data",
        "source_column": "C1_COUNT_TOTAL",
        "denominator": "population_total_2021 from cleaned census-division base frame",
        "unit": "recipients_per_1000_population",
        "derivation": "1000 * social_security_recipients_per_capita",
        "coverage": "98/98",
        "status": "audit_scaled_variable",
        "notes": "Human-readable scaling of the main per-capita variable.",
    },
    {
        "variable": "government_transfer_recipients_2020_100pct_per_capita",
        "original_sovi_code": "",
        "description": "2020 government-transfer recipients per total population",
        "source_dataset": "2021 Census Profile, census divisions",
        "source_characteristic_id": "120",
        "source_characteristic_name": "Number of government transfers recipients aged 15 years and over in private households in 2020 - 100% data",
        "source_column": "C1_COUNT_TOTAL",
        "denominator": "population_total_2021",
        "unit": "recipients_per_person",
        "derivation": "government_transfer_recipients_2020_100pct_count / population_total_2021",
        "coverage": "98/98",
        "status": "audit_sensitivity_variable",
        "notes": "Retained because 2020 is closer to the 2021 Census, but it is affected by pandemic emergency/recovery benefits.",
    },
    {
        "variable": "employment_insurance_recipients_2020_100pct_per_capita",
        "original_sovi_code": "",
        "description": "Employment insurance benefit recipients per total population",
        "source_dataset": "2021 Census Profile, census divisions",
        "source_characteristic_id": "122",
        "source_characteristic_name": "Number of employment insurance benefits recipients aged 15 years and over in private households in 2020 - 100% data",
        "source_column": "C1_COUNT_TOTAL",
        "denominator": "population_total_2021",
        "unit": "recipients_per_person",
        "derivation": "employment_insurance_recipients_2020_100pct_count / population_total_2021",
        "coverage": "98/98",
        "status": "audit_narrow_benefit_variable",
        "notes": "Narrower benefit-recipient audit variable. Not used as default for SSBENPC90.",
    },
    {
        "variable": "covid_benefit_recipients_2020_100pct_per_capita",
        "original_sovi_code": "",
        "description": "COVID-19 emergency and recovery benefit recipients per total population",
        "source_dataset": "2021 Census Profile, census divisions",
        "source_characteristic_id": "124",
        "source_characteristic_name": "Number of COVID-19 emergency and recovery benefits recipients aged 15 years and over in private households in 2020 - 100% data",
        "source_column": "C1_COUNT_TOTAL",
        "denominator": "population_total_2021",
        "unit": "recipients_per_person",
        "derivation": "covid_benefit_recipients_2020_100pct_count / population_total_2021",
        "coverage": "98/98",
        "status": "audit_pandemic_context_variable",
        "notes": "Retained to document pandemic-transfer context. Not used as default for SSBENPC90.",
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
    {"metric": "raw_rows", "value": len(raw)},
    {"metric": "quebec_cd_rows_scanned", "value": len(qc_rows)},
    {"metric": "clean_rows", "value": len(clean)},
    {"metric": "unique_census_divisions", "value": clean["census_division_dguid"].nunique()},
    {"metric": "variables_cleaned", "value": "social_security_recipients_per_capita"},
    {"metric": "source_characteristic_ids", "value": ", ".join(config["id"] for config in CHARACTERISTICS.values())},
    {"metric": "main_default_characteristic_id", "value": "213"},
    {"metric": "main_default_characteristic_name", "value": CHARACTERISTICS["government_transfer_recipients_2019_100pct"]["expected_name_contains"]},
    {"metric": "all_required_numeric_columns_complete", "value": bool(clean[required_numeric_cols].notna().all().all())},
    {"metric": "social_security_recipients_per_capita_formula_max_abs_difference", "value": formula_diff},
    {"metric": "social_security_recipients_per_1000_formula_max_abs_difference", "value": per_1000_formula_diff},
    {"metric": "base_names_with_mojibake", "value": base_names_with_mojibake},
    {"metric": "clean_names_with_mojibake", "value": clean_names_with_mojibake},
    {"metric": "raw_characteristic_names_with_mojibake", "value": raw_characteristic_names_with_mojibake},
]

for _, row in inventory.iterrows():
    summary_rows.append({"metric": f"{row['alias']}_characteristic_id", "value": row["characteristic_id"]})
    summary_rows.append({"metric": f"{row['alias']}_characteristic_name", "value": row["characteristic_name"]})
    summary_rows.append({"metric": f"{row['alias']}_coverage_is_98_cds", "value": row["coverage_is_98_cds"]})
    summary_rows.append({"metric": f"{row['alias']}_count_non_missing", "value": row["count_non_missing"]})
    summary_rows.append({"metric": f"{row['alias']}_rate_non_missing", "value": row["rate_non_missing"]})

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
            "If this summary shows 98 rows, full numeric coverage, zero formula differences, and no mojibake, "
            "generate the README and add a SoVI YAML mapping for SSBENPC90 -> social_security_recipients_per_capita."
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
        "social_security_recipients_per_capita",
        "social_security_recipients_per_1000",
        "government_transfer_recipients_2019_100pct_count",
        "government_transfer_recipients_2020_100pct_count",
        "government_transfer_recipients_2019_25pct_count",
        "government_transfer_recipients_2020_25pct_count",
        "employment_insurance_recipients_2020_100pct_count",
        "covid_benefit_recipients_2020_100pct_count",
        "government_transfer_recipients_2020_100pct_per_capita",
        "government_transfer_recipients_2019_25pct_per_capita",
        "government_transfer_recipients_2020_25pct_per_capita",
        "employment_insurance_recipients_2020_100pct_per_capita",
        "covid_benefit_recipients_2020_100pct_per_capita",
        "government_transfer_recipients_2020_minus_2019_count",
        "government_transfer_recipients_2020_minus_2019_per_capita",
    ]
)

# Keep source characteristic IDs/names for reproducibility.
for alias in CHARACTERISTICS:
    ordered_cols += [
        f"{alias}_characteristic_id",
        f"{alias}_characteristic_name",
        f"{alias}_source_rate",
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
print("CLEAN CENSUS DIVISION SOCIAL SECURITY RECIPIENTS 2021")
print("=" * 72)

print("\nClean output:")
print("Rows:", len(clean))
print("Unique census divisions:", clean["census_division_dguid"].nunique())
print("Variables cleaned: social_security_recipients_per_capita")

print("\nFormula checks:")
print("social_security_recipients_per_capita formula max abs difference:", formula_diff)
print("social_security_recipients_per_1000 formula max abs difference:", per_1000_formula_diff)

print("\nMojibake check:")
print("Base names with mojibake:", base_names_with_mojibake)
print("Clean names with mojibake:", clean_names_with_mojibake)
print("Raw characteristic names with mojibake:", raw_characteristic_names_with_mojibake)

print("\nMain summaries:")
for variable in [
    "social_security_recipients_per_capita",
    "social_security_recipients_per_1000",
    "government_transfer_recipients_2019_100pct_count",
    "government_transfer_recipients_2020_100pct_count",
    "government_transfer_recipients_2020_minus_2019_per_capita",
    "employment_insurance_recipients_2020_100pct_per_capita",
    "covid_benefit_recipients_2020_100pct_per_capita",
]:
    print("\n", variable)
    print(clean[variable].describe())

print("\nSource characteristic inventory:")
print(inventory.to_string(index=False))

print("\nPreview:")
preview_cols = [
    "census_division_code",
    "census_division_name",
    "social_security_recipients_per_capita",
    "social_security_recipients_per_1000",
    "government_transfer_recipients_2019_100pct_count",
    "government_transfer_recipients_2020_100pct_count",
    "government_transfer_recipients_2020_minus_2019_per_capita",
]
preview_cols = [col for col in preview_cols if col in clean.columns]
print(clean[preview_cols].head(20).to_string(index=False))

print("\nSaved:")
print(OUTPUT_CLEAN)
print(OUTPUT_SOURCE_ROWS)
print(OUTPUT_METADATA)
print(OUTPUT_SUMMARY)

print("\nDone.")