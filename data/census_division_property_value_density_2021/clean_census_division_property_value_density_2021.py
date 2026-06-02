from pathlib import Path
import codecs
import re
import pandas as pd


# ============================================================
# Clean Census Division Property Value Density 2021
# ============================================================
#
# Purpose:
#   Clean a documented fallback proxy for:
#
#       RPROPDEN92 -> property_value_density
#
# Original SoVI concept:
#   Value of all property and farm products sold per unit area.
#
# Canadian fallback used here:
#   A weak residential owner-occupied dwelling-value-density proxy:
#
#       property_value_density =
#           median_owner_occupied_housing_value
#           *
#           owner_households_direct_count
#           /
#           land_area_km2
#
# Main source components:
#
#   Census Profile ID 1415:
#       Owner
#
#   Existing housing tenure/costs clean variable:
#       median_owner_occupied_housing_value
#
#   Census Profile ID 1488, used as source validation:
#       Median value of dwellings ($)
#
# Important:
#   This is not a true aggregate property-assessment variable. It does not
#   include rental property value, commercial property, industrial property,
#   institutional property, farm property, or total assessed land/property value.
#   It should be described as a weak residential owner-occupied proxy.
#
# Run from data/:
#
#   python census_division_property_value_density_2021/clean_census_division_property_value_density_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_property_value_density_2021"
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

HOUSING_TENURE_COSTS_CLEAN = (
    DATA_DIR
    / "census_division_housing_tenure_costs_2021"
    / "output"
    / "clean_census_division_housing_tenure_costs_2021.csv"
)

OUTPUT_CLEAN = OUTPUT_DIR / "clean_census_division_property_value_density_2021.csv"
OUTPUT_SOURCE_ROWS = OUTPUT_DIR / "clean_census_division_property_value_density_source_rows_2021.csv"
OUTPUT_METADATA = OUTPUT_DIR / "clean_census_division_property_value_density_variable_metadata_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "clean_census_division_property_value_density_summary_2021.csv"
OUTPUT_FORMULA_AUDIT = OUTPUT_DIR / "clean_census_division_property_value_density_formula_audit_2021.csv"
OUTPUT_UNMATCHED_AUDIT = OUTPUT_DIR / "clean_census_division_property_value_density_unmatched_audit_2021.csv"


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
    "total_tenure_households": {
        "id": "1414",
        "expected_name_contains": "Total - Private households by tenure - 25% sample data",
        "role": "total_tenure_denominator_component",
    },
    "owner_households": {
        "id": "1415",
        "expected_name_contains": "Owner",
        "role": "main_owner_household_count_component",
    },
    "renter_households": {
        "id": "1416",
        "expected_name_contains": "Renter",
        "role": "renter_household_count_and_rate_audit_component",
    },
    "median_value_of_dwellings_profile": {
        "id": "1488",
        "expected_name_contains": "Median value of dwellings ($)",
        "role": "median_dwelling_value_profile_validation_component",
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


def add_summary_stats(summary_rows: list[dict], df: pd.DataFrame, variable: str) -> None:
    stats = numeric_summary(df[variable])
    for key, value in stats.items():
        summary_rows.append(
            {
                "metric": f"{variable}_{key}",
                "value": value,
            }
        )


def extract_characteristic(
    qc_rows: pd.DataFrame,
    alias: str,
    config: dict,
    dguid_col: str,
    geo_col: str | None,
    char_id_col: str,
    char_name_col: str,
    count_col: str,
    rate_col: str | None,
    count_symbol_col: str | None,
    rate_symbol_col: str | None,
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

    source_cols += [char_id_col, char_name_col, count_col]

    if rate_col and rate_col in rows.columns:
        source_cols.append(rate_col)
    if count_symbol_col and count_symbol_col in rows.columns:
        source_cols.append(count_symbol_col)
    if rate_symbol_col and rate_symbol_col in rows.columns:
        source_cols.append(rate_symbol_col)

    source_long = rows[source_cols].copy()
    source_long.insert(0, "source_alias", alias)
    source_long.insert(1, "source_role", config["role"])
    source_long["source_count_numeric"] = clean_numeric(source_long[count_col])

    if rate_col and rate_col in source_long.columns:
        source_long["source_rate_numeric"] = clean_numeric(source_long[rate_col])

    extract_cols = [dguid_col, char_id_col, char_name_col, count_col]
    if rate_col and rate_col in rows.columns:
        extract_cols.append(rate_col)

    out = rows[extract_cols].copy()
    out[dguid_col] = out[dguid_col].astype("string").str.strip()
    out[count_col] = clean_numeric(out[count_col])

    if rate_col and rate_col in out.columns:
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
            count_col: f"{alias}_count_value",
        }
    )

    if rate_col and rate_col in rows.columns:
        out = out.rename(columns={rate_col: f"{alias}_rate_value"})

    count_values = clean_numeric(out[f"{alias}_count_value"])
    rate_values = (
        clean_numeric(out[f"{alias}_rate_value"])
        if f"{alias}_rate_value" in out.columns
        else pd.Series(dtype=float)
    )

    inventory = {
        "alias": alias,
        "characteristic_id": char_id,
        "characteristic_name": characteristic_name,
        "role": config["role"],
        "rows_extracted": len(out),
        "unique_census_divisions": int(out["census_division_dguid"].nunique()),
        "count_non_missing": int(count_values.notna().sum()),
        "count_missing": int(count_values.isna().sum()),
        "count_min": count_values.min(skipna=True),
        "count_max": count_values.max(skipna=True),
        "count_mean": count_values.mean(skipna=True),
        "count_median": count_values.median(skipna=True),
        "rate_non_missing": int(rate_values.notna().sum()) if not rate_values.empty else 0,
        "rate_missing": int(rate_values.isna().sum()) if not rate_values.empty else len(out),
        "rate_min": rate_values.min(skipna=True) if not rate_values.empty else None,
        "rate_max": rate_values.max(skipna=True) if not rate_values.empty else None,
        "rate_mean": rate_values.mean(skipna=True) if not rate_values.empty else None,
        "rate_median": rate_values.median(skipna=True) if not rate_values.empty else None,
        "coverage_is_98_cds": out["census_division_dguid"].nunique() == EXPECTED_QC_CD_COUNT,
    }

    return out, source_long, inventory


# -----------------------------
# Validate inputs
# -----------------------------

if not RAW_PROFILE.exists():
    raise FileNotFoundError(f"Missing raw Census Profile CSV:\n{RAW_PROFILE}")

if not BASE_CD_FRAME.exists():
    raise FileNotFoundError(f"Missing base CD frame:\n{BASE_CD_FRAME}")

if not HOUSING_TENURE_COSTS_CLEAN.exists():
    raise FileNotFoundError(f"Missing housing tenure/costs clean file:\n{HOUSING_TENURE_COSTS_CLEAN}")


# -----------------------------
# Load base
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
base["census_division_code"] = base["census_division_code"].astype("string").str.strip()
base["census_division_dguid"] = base["census_division_dguid"].astype("string").str.strip()
base["land_area_km2"] = clean_numeric(base["land_area_km2"])

if len(base) != EXPECTED_QC_CD_COUNT:
    raise ValueError(f"Expected {EXPECTED_QC_CD_COUNT} Québec CDs in base frame, got {len(base)}.")

if base["census_division_dguid"].duplicated().any():
    raise ValueError("Duplicate census_division_dguid values in base frame.")

if base["land_area_km2"].isna().any() or (base["land_area_km2"] <= 0).any():
    raise ValueError("Missing or non-positive land_area_km2 values in base frame.")

base_dguid_set = set(base["census_division_dguid"].dropna().astype(str))


# -----------------------------
# Load housing tenure/costs clean output
# -----------------------------

housing_encoding = detect_file_encoding_strict(HOUSING_TENURE_COSTS_CLEAN, ENCODING_CANDIDATES)

housing = pd.read_csv(
    HOUSING_TENURE_COSTS_CLEAN,
    encoding=housing_encoding,
    dtype=str,
    low_memory=False,
)

housing.columns = [str(col).strip() for col in housing.columns]

required_housing_cols = [
    "census_division_dguid",
    "median_owner_occupied_housing_value",
    "pct_renter_occupied",
]

missing_housing_cols = [col for col in required_housing_cols if col not in housing.columns]
if missing_housing_cols:
    raise ValueError(
        "Housing tenure/costs clean file is missing required columns:\n"
        + "\n".join(missing_housing_cols)
        + "\n\nAvailable columns:\n"
        + "\n".join(housing.columns)
    )

housing = housing.copy()
housing["census_division_dguid"] = housing["census_division_dguid"].astype("string").str.strip()
housing["median_owner_occupied_housing_value"] = clean_numeric(
    housing["median_owner_occupied_housing_value"]
)
housing["pct_renter_occupied"] = clean_numeric(housing["pct_renter_occupied"])

if len(housing) != EXPECTED_QC_CD_COUNT:
    raise ValueError(f"Expected {EXPECTED_QC_CD_COUNT} housing rows, got {len(housing)}.")

if housing["census_division_dguid"].duplicated().any():
    raise ValueError("Duplicate census_division_dguid values in housing tenure/costs clean file.")

if housing["median_owner_occupied_housing_value"].isna().any():
    raise ValueError("Missing median_owner_occupied_housing_value values.")

if housing["pct_renter_occupied"].isna().any():
    raise ValueError("Missing pct_renter_occupied values.")


# -----------------------------
# Load raw Census Profile
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
geo_col = find_column(columns, ["GEO", "GEO_NAME", "Geography", "Geography name", "ALT_GEO_CODE"])
char_id_col = require_column(columns, ["CHARACTERISTIC_ID", "Characteristic ID"], "characteristic ID")
char_name_col = require_column(columns, ["CHARACTERISTIC_NAME", "Characteristic Name"], "characteristic name")
count_col = require_column(columns, ["C1_COUNT_TOTAL", "Count total", "Total - Count"], "count/value")
rate_col = find_column(columns, ["C10_RATE_TOTAL", "Rate total", "Total - Rate"])
count_symbol_col = find_column(columns, ["SYMBOL.1", "C1_SYMBOL_TOTAL", "SYMBOL"])
rate_symbol_col = find_column(columns, ["SYMBOL.3", "C10_SYMBOL_TOTAL", "SYMBOL"])

raw[dguid_col] = raw[dguid_col].astype("string").str.strip()
qc_rows = raw[raw[dguid_col].isin(base_dguid_set)].copy()

if qc_rows.empty:
    raise ValueError("No Québec CD rows found in Census Profile after DGUID filtering.")


# -----------------------------
# Extract source characteristics
# -----------------------------

source_frames = []
inventory_rows = []
component_tables = {}

print("\nCleaning Census Division Property Value Density 2021")
print("Raw profile:", safe_relative(RAW_PROFILE))
print("Raw encoding:", raw_encoding)
print("Base frame:", safe_relative(BASE_CD_FRAME))
print("Housing clean:", safe_relative(HOUSING_TENURE_COSTS_CLEAN))

for alias, config in CHARACTERISTICS.items():
    extracted, source_long, inventory = extract_characteristic(
        qc_rows=qc_rows,
        alias=alias,
        config=config,
        dguid_col=dguid_col,
        geo_col=geo_col,
        char_id_col=char_id_col,
        char_name_col=char_name_col,
        count_col=count_col,
        rate_col=rate_col,
        count_symbol_col=count_symbol_col,
        rate_symbol_col=rate_symbol_col,
    )

    component_tables[alias] = extracted
    source_frames.append(source_long)
    inventory_rows.append(inventory)

    print(
        f"  {alias}: ID {inventory['characteristic_id']}, "
        f"rows={inventory['rows_extracted']}, "
        f"count_non_missing={inventory['count_non_missing']}"
    )

source_rows = pd.concat(source_frames, ignore_index=True, sort=False)
source_rows.to_csv(OUTPUT_SOURCE_ROWS, index=False, encoding="utf-8")

component_audit = pd.DataFrame(inventory_rows)


# -----------------------------
# Build clean table
# -----------------------------

identity_cols = [col for col in IDENTITY_COLUMNS if col in base.columns]

clean = base[identity_cols].copy()

clean = clean.merge(
    housing[
        [
            "census_division_dguid",
            "median_owner_occupied_housing_value",
            "pct_renter_occupied",
        ]
    ],
    on="census_division_dguid",
    how="left",
    validate="one_to_one",
)

for alias, table in component_tables.items():
    clean = clean.merge(
        table,
        on="census_division_dguid",
        how="left",
        validate="one_to_one",
    )

clean["total_tenure_households_count"] = clean["total_tenure_households_count_value"]
clean["owner_households_direct_count"] = clean["owner_households_count_value"]
clean["renter_households_direct_count"] = clean["renter_households_count_value"]
clean["median_value_of_dwellings_profile_value"] = clean[
    "median_value_of_dwellings_profile_count_value"
]

if "owner_households_rate_value" in clean.columns:
    clean["owner_households_rate_pct"] = clean["owner_households_rate_value"]

if "renter_households_rate_value" in clean.columns:
    clean["renter_households_rate_pct"] = clean["renter_households_rate_value"]

# Main numerator and density.
clean["estimated_owner_occupied_residential_property_value"] = (
    clean["median_owner_occupied_housing_value"]
    * clean["owner_households_direct_count"]
)

clean["property_value_density"] = (
    clean["estimated_owner_occupied_residential_property_value"]
    / clean["land_area_km2"]
)

# Audit: estimate owner households from total tenure count and renter rate.
clean["estimated_owner_households_from_pct_renter"] = (
    clean["total_tenure_households_count"]
    * (1 - clean["pct_renter_occupied"] / 100)
)

clean["estimated_owner_occupied_residential_property_value_from_pct_renter"] = (
    clean["median_owner_occupied_housing_value"]
    * clean["estimated_owner_households_from_pct_renter"]
)

clean["property_value_density_estimated_owner_from_pct_renter"] = (
    clean["estimated_owner_occupied_residential_property_value_from_pct_renter"]
    / clean["land_area_km2"]
)

# Audit: owner households from total minus renter count.
clean["estimated_owner_households_from_total_minus_renter"] = (
    clean["total_tenure_households_count"]
    - clean["renter_households_direct_count"]
)

clean["estimated_owner_occupied_residential_property_value_from_total_minus_renter"] = (
    clean["median_owner_occupied_housing_value"]
    * clean["estimated_owner_households_from_total_minus_renter"]
)

clean["property_value_density_estimated_owner_from_total_minus_renter"] = (
    clean["estimated_owner_occupied_residential_property_value_from_total_minus_renter"]
    / clean["land_area_km2"]
)

# Consistency diagnostics.
clean["owner_plus_renter_minus_total_tenure_count"] = (
    clean["owner_households_direct_count"]
    + clean["renter_households_direct_count"]
    - clean["total_tenure_households_count"]
)

clean["owner_direct_minus_owner_estimated_from_pct_renter"] = (
    clean["owner_households_direct_count"]
    - clean["estimated_owner_households_from_pct_renter"]
)

clean["median_value_profile_minus_housing_clean"] = (
    clean["median_value_of_dwellings_profile_value"]
    - clean["median_owner_occupied_housing_value"]
)

clean["source_file"] = safe_relative(RAW_PROFILE)
clean["source_housing_clean_file"] = safe_relative(HOUSING_TENURE_COSTS_CLEAN)
clean["source_section"] = "2021 Census Profile housing tenure/value rows"
clean["source_encoding"] = raw_encoding
clean["proxy_quality"] = "weak_residential_owner_occupied_property_value_density_proxy"
clean["method_note"] = (
    "property_value_density maps RPROPDEN92 to a weak residential owner-occupied dwelling-value-density proxy. "
    "The numerator is reconstructed as Census Profile characteristic ID 1415, Owner household count, multiplied by "
    "median_owner_occupied_housing_value from the cleaned housing tenure/costs file, corresponding to Census Profile "
    "characteristic ID 1488, Median value of dwellings ($). The result is divided by land_area_km2. This is not a "
    "true total property-assessment value and does not include rental, commercial, industrial, institutional, farm, "
    "or land-assessment values."
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
    "land_area_km2",
    "median_owner_occupied_housing_value",
    "pct_renter_occupied",
    "median_value_of_dwellings_profile_value",
    "total_tenure_households_count",
    "owner_households_direct_count",
    "renter_households_direct_count",
    "estimated_owner_occupied_residential_property_value",
    "property_value_density",
    "estimated_owner_households_from_pct_renter",
    "estimated_owner_occupied_residential_property_value_from_pct_renter",
    "property_value_density_estimated_owner_from_pct_renter",
    "estimated_owner_households_from_total_minus_renter",
    "estimated_owner_occupied_residential_property_value_from_total_minus_renter",
    "property_value_density_estimated_owner_from_total_minus_renter",
    "owner_plus_renter_minus_total_tenure_count",
    "owner_direct_minus_owner_estimated_from_pct_renter",
    "median_value_profile_minus_housing_clean",
]

for col in required_numeric_cols:
    clean[col] = clean_numeric(clean[col])
    missing = int(clean[col].isna().sum())
    if missing != 0:
        raise ValueError(f"Unexpected missing values in {col}: {missing}")

nonnegative_cols = [
    "land_area_km2",
    "median_owner_occupied_housing_value",
    "pct_renter_occupied",
    "median_value_of_dwellings_profile_value",
    "total_tenure_households_count",
    "owner_households_direct_count",
    "renter_households_direct_count",
    "estimated_owner_occupied_residential_property_value",
    "property_value_density",
    "estimated_owner_households_from_pct_renter",
    "estimated_owner_occupied_residential_property_value_from_pct_renter",
    "property_value_density_estimated_owner_from_pct_renter",
    "estimated_owner_households_from_total_minus_renter",
    "estimated_owner_occupied_residential_property_value_from_total_minus_renter",
    "property_value_density_estimated_owner_from_total_minus_renter",
]

for col in nonnegative_cols:
    if (clean[col] < 0).any():
        raise ValueError(f"Negative values found in {col}.")

count_cols = [
    "total_tenure_households_count",
    "owner_households_direct_count",
    "renter_households_direct_count",
]

for col in count_cols:
    max_fractional = ((clean[col] - clean[col].round()).abs()).max(skipna=True)
    if pd.notna(max_fractional) and max_fractional > 1e-9:
        raise ValueError(f"Non-integer-like household counts found in {col}: max fractional {max_fractional}")

main_formula_diff = (
    clean["property_value_density"]
    - (
        clean["median_owner_occupied_housing_value"]
        * clean["owner_households_direct_count"]
        / clean["land_area_km2"]
    )
).abs().max(skipna=True)

pct_formula_diff = (
    clean["property_value_density_estimated_owner_from_pct_renter"]
    - (
        clean["median_owner_occupied_housing_value"]
        * clean["total_tenure_households_count"]
        * (1 - clean["pct_renter_occupied"] / 100)
        / clean["land_area_km2"]
    )
).abs().max(skipna=True)

minus_renter_formula_diff = (
    clean["property_value_density_estimated_owner_from_total_minus_renter"]
    - (
        clean["median_owner_occupied_housing_value"]
        * (clean["total_tenure_households_count"] - clean["renter_households_direct_count"])
        / clean["land_area_km2"]
    )
).abs().max(skipna=True)

estimated_value_formula_diff = (
    clean["estimated_owner_occupied_residential_property_value"]
    - clean["median_owner_occupied_housing_value"] * clean["owner_households_direct_count"]
).abs().max(skipna=True)

median_profile_housing_max_abs_diff = clean["median_value_profile_minus_housing_clean"].abs().max(skipna=True)

formula_diffs = {
    "main_formula_diff": main_formula_diff,
    "pct_formula_diff": pct_formula_diff,
    "minus_renter_formula_diff": minus_renter_formula_diff,
    "estimated_value_formula_diff": estimated_value_formula_diff,
    "median_profile_housing_max_abs_diff": median_profile_housing_max_abs_diff,
}

tolerance = 1e-6

for name, diff in formula_diffs.items():
    if pd.notna(diff) and diff > tolerance:
        raise ValueError(f"{name} exceeded tolerance {tolerance}: {diff}")

base_names_with_mojibake = contains_mojibake(base["census_division_name"])
clean_names_with_mojibake = contains_mojibake(clean["census_division_name"])
raw_characteristic_names_with_mojibake = contains_mojibake(raw[char_name_col])

if raw_characteristic_names_with_mojibake != 0:
    raise ValueError("Mojibake detected in raw characteristic names after decoding.")


# -----------------------------
# Formula audit
# -----------------------------

formula_audit_rows = []

formula_specs = [
    {
        "candidate_alias": "property_value_density",
        "candidate_formula": "median_owner_occupied_housing_value * owner_households_direct_count / land_area_km2",
        "estimated_value_column": "estimated_owner_occupied_residential_property_value",
        "density_column": "property_value_density",
        "status": "ready_full_coverage_weak_proxy",
        "recommended_default": True,
        "interpretation": (
            "Default weak fallback proxy using direct Census Profile owner household count times median dwelling value."
        ),
    },
    {
        "candidate_alias": "property_value_density_estimated_owner_from_pct_renter",
        "candidate_formula": "median_owner_occupied_housing_value * total_tenure_households_count * (1 - pct_renter_occupied / 100) / land_area_km2",
        "estimated_value_column": "estimated_owner_occupied_residential_property_value_from_pct_renter",
        "density_column": "property_value_density_estimated_owner_from_pct_renter",
        "status": "audit_full_coverage_weak_proxy",
        "recommended_default": False,
        "interpretation": (
            "Audit proxy estimating owner households from total tenure households and renter percentage."
        ),
    },
    {
        "candidate_alias": "property_value_density_estimated_owner_from_total_minus_renter",
        "candidate_formula": "median_owner_occupied_housing_value * (total_tenure_households_count - renter_households_direct_count) / land_area_km2",
        "estimated_value_column": "estimated_owner_occupied_residential_property_value_from_total_minus_renter",
        "density_column": "property_value_density_estimated_owner_from_total_minus_renter",
        "status": "audit_full_coverage_weak_proxy",
        "recommended_default": False,
        "interpretation": (
            "Audit proxy estimating owner households as total tenure households minus renter households."
        ),
    },
]

for spec in formula_specs:
    estimated_value = clean_numeric(clean[spec["estimated_value_column"]])
    density = clean_numeric(clean[spec["density_column"]])

    formula_audit_rows.append(
        {
            "candidate_alias": spec["candidate_alias"],
            "canonical_variable": "property_value_density",
            "original_sovi_code": "RPROPDEN92",
            "candidate_formula": spec["candidate_formula"],
            "non_missing": int(density.notna().sum()),
            "missing": int(density.isna().sum()),
            "coverage_is_98_cds": int(density.notna().sum()) == EXPECTED_QC_CD_COUNT,
            "estimated_value_min": estimated_value.min(skipna=True),
            "estimated_value_max": estimated_value.max(skipna=True),
            "estimated_value_mean": estimated_value.mean(skipna=True),
            "estimated_value_median": estimated_value.median(skipna=True),
            "density_min": density.min(skipna=True),
            "density_max": density.max(skipna=True),
            "density_mean": density.mean(skipna=True),
            "density_median": density.median(skipna=True),
            "status": spec["status"],
            "recommended_default": spec["recommended_default"],
            "proxy_quality": "weak_residential_owner_occupied_property_value_density_proxy",
            "interpretation": spec["interpretation"],
        }
    )

formula_audit = pd.DataFrame(formula_audit_rows)
formula_audit.to_csv(OUTPUT_FORMULA_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Unmatched audit
# -----------------------------

unmatched = clean[
    [
        "census_division_code",
        "census_division_dguid",
        "census_division_name",
        "property_value_density",
        "owner_households_direct_count",
        "median_owner_occupied_housing_value",
        "land_area_km2",
    ]
].copy()

missing_mask = unmatched[
    [
        "property_value_density",
        "owner_households_direct_count",
        "median_owner_occupied_housing_value",
        "land_area_km2",
    ]
].isna().any(axis=1)

unmatched = unmatched[missing_mask].copy()
unmatched.to_csv(OUTPUT_UNMATCHED_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Metadata
# -----------------------------

metadata_rows = [
    {
        "variable": "property_value_density",
        "original_sovi_code": "RPROPDEN92",
        "description": "Weak residential owner-occupied dwelling-value-density proxy",
        "source_dataset": "2021 Census Profile, census divisions; cleaned housing tenure/costs output",
        "source_file": safe_relative(RAW_PROFILE),
        "source_housing_clean_file": safe_relative(HOUSING_TENURE_COSTS_CLEAN),
        "source_characteristic_ids": "1415 * 1488",
        "source_characteristic_names": "Owner; Median value of dwellings ($)",
        "denominator": "land_area_km2 from cleaned census-division base frame",
        "unit": "estimated_dollars_per_square_kilometre",
        "derivation": "median_owner_occupied_housing_value * owner_households_direct_count / land_area_km2",
        "coverage": "98/98",
        "status": "ready_full_coverage_weak_proxy",
        "proxy_quality": "weak_residential_owner_occupied_property_value_density_proxy",
        "notes": (
            "This is not a total property-assessment value. It approximates residential owner-occupied dwelling-value "
            "density using median dwelling value times owner household count. It excludes rental, commercial, industrial, "
            "institutional, farm, and land-assessment values."
        ),
    },
    {
        "variable": "estimated_owner_occupied_residential_property_value",
        "original_sovi_code": "",
        "description": "Estimated residential owner-occupied dwelling-value numerator",
        "source_dataset": "2021 Census Profile, census divisions; cleaned housing tenure/costs output",
        "source_file": safe_relative(RAW_PROFILE),
        "source_housing_clean_file": safe_relative(HOUSING_TENURE_COSTS_CLEAN),
        "source_characteristic_ids": "1415 * 1488",
        "source_characteristic_names": "Owner; Median value of dwellings ($)",
        "denominator": "",
        "unit": "estimated_dollars",
        "derivation": "median_owner_occupied_housing_value * owner_households_direct_count",
        "coverage": "98/98",
        "status": "component_audit_variable",
        "proxy_quality": "weak_residential_owner_occupied_property_value_density_proxy",
        "notes": "Estimated numerator used for property_value_density.",
    },
    {
        "variable": "property_value_density_estimated_owner_from_pct_renter",
        "original_sovi_code": "",
        "description": "Audit proxy using owner households estimated from total tenure households and renter percentage",
        "source_dataset": "2021 Census Profile, census divisions; cleaned housing tenure/costs output",
        "source_file": safe_relative(RAW_PROFILE),
        "source_housing_clean_file": safe_relative(HOUSING_TENURE_COSTS_CLEAN),
        "source_characteristic_ids": "1414, 1416, 1488",
        "source_characteristic_names": "Total - Private households by tenure; Renter; Median value of dwellings ($)",
        "denominator": "land_area_km2",
        "unit": "estimated_dollars_per_square_kilometre",
        "derivation": "median_owner_occupied_housing_value * total_tenure_households_count * (1 - pct_renter_occupied / 100) / land_area_km2",
        "coverage": "98/98",
        "status": "audit_sensitivity_variable",
        "proxy_quality": "weak_residential_owner_occupied_property_value_density_proxy",
        "notes": "Retained as a sensitivity check, not the default SoVI input.",
    },
    {
        "variable": "property_value_density_estimated_owner_from_total_minus_renter",
        "original_sovi_code": "",
        "description": "Audit proxy using owner households estimated as total tenure households minus renter households",
        "source_dataset": "2021 Census Profile, census divisions; cleaned housing tenure/costs output",
        "source_file": safe_relative(RAW_PROFILE),
        "source_housing_clean_file": safe_relative(HOUSING_TENURE_COSTS_CLEAN),
        "source_characteristic_ids": "1414, 1416, 1488",
        "source_characteristic_names": "Total - Private households by tenure; Renter; Median value of dwellings ($)",
        "denominator": "land_area_km2",
        "unit": "estimated_dollars_per_square_kilometre",
        "derivation": "median_owner_occupied_housing_value * (total_tenure_households_count - renter_households_direct_count) / land_area_km2",
        "coverage": "98/98",
        "status": "audit_sensitivity_variable",
        "proxy_quality": "weak_residential_owner_occupied_property_value_density_proxy",
        "notes": "Retained as a sensitivity check, not the default SoVI input.",
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
    {"metric": "housing_tenure_costs_clean", "value": safe_relative(HOUSING_TENURE_COSTS_CLEAN)},
    {"metric": "raw_encoding", "value": raw_encoding},
    {"metric": "base_encoding", "value": base_encoding},
    {"metric": "housing_encoding", "value": housing_encoding},
    {"metric": "raw_rows", "value": len(raw)},
    {"metric": "quebec_cd_rows_scanned", "value": len(qc_rows)},
    {"metric": "base_rows", "value": len(base)},
    {"metric": "housing_rows", "value": len(housing)},
    {"metric": "clean_rows", "value": len(clean)},
    {"metric": "unique_census_divisions", "value": clean["census_division_dguid"].nunique()},
    {"metric": "variables_cleaned", "value": "property_value_density"},
    {"metric": "original_sovi_code", "value": "RPROPDEN92"},
    {"metric": "proxy_quality", "value": "weak_residential_owner_occupied_property_value_density_proxy"},
    {"metric": "main_formula", "value": "median_owner_occupied_housing_value * owner_households_direct_count / land_area_km2"},
    {"metric": "all_required_numeric_columns_complete", "value": bool(clean[required_numeric_cols].notna().all().all())},
    {"metric": "main_formula_max_abs_difference", "value": main_formula_diff},
    {"metric": "pct_formula_max_abs_difference", "value": pct_formula_diff},
    {"metric": "minus_renter_formula_max_abs_difference", "value": minus_renter_formula_diff},
    {"metric": "estimated_value_formula_max_abs_difference", "value": estimated_value_formula_diff},
    {"metric": "median_profile_housing_max_abs_difference", "value": median_profile_housing_max_abs_diff},
    {"metric": "owner_plus_renter_minus_total_abs_max", "value": clean["owner_plus_renter_minus_total_tenure_count"].abs().max(skipna=True)},
    {"metric": "owner_direct_minus_owner_estimated_from_pct_renter_abs_max", "value": clean["owner_direct_minus_owner_estimated_from_pct_renter"].abs().max(skipna=True)},
    {"metric": "base_names_with_mojibake", "value": base_names_with_mojibake},
    {"metric": "clean_names_with_mojibake", "value": clean_names_with_mojibake},
    {"metric": "raw_characteristic_names_with_mojibake", "value": raw_characteristic_names_with_mojibake},
]

for _, row in component_audit.iterrows():
    alias = row["alias"]
    summary_rows.append({"metric": f"{alias}_characteristic_id", "value": row["characteristic_id"]})
    summary_rows.append({"metric": f"{alias}_characteristic_name", "value": row["characteristic_name"]})
    summary_rows.append({"metric": f"{alias}_coverage_is_98_cds", "value": row["coverage_is_98_cds"]})
    summary_rows.append({"metric": f"{alias}_count_non_missing", "value": row["count_non_missing"]})
    summary_rows.append({"metric": f"{alias}_rate_non_missing", "value": row["rate_non_missing"]})

for variable in required_numeric_cols + [
    "property_value_density",
    "estimated_owner_occupied_residential_property_value",
]:
    if variable in clean.columns:
        add_summary_stats(summary_rows, clean, variable)

summary_rows.append(
    {
        "metric": "important_method_note",
        "value": (
            "RPROPDEN92 is not directly reproduced. The cleaned variable is a weak residential owner-occupied "
            "dwelling-value-density proxy based on median dwelling value times owner household count divided by land area. "
            "It should be used only as a documented fallback if no stronger total assessed property-value source is available."
        ),
    }
)

summary_rows.append(
    {
        "metric": "recommended_next_step",
        "value": (
            "If this summary shows 98 rows, full numeric coverage, zero formula differences, and no mojibake, "
            "generate the README and add a SoVI YAML mapping for RPROPDEN92 -> property_value_density."
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
        "property_value_density",
        "estimated_owner_occupied_residential_property_value",
        "median_owner_occupied_housing_value",
        "owner_households_direct_count",
        "owner_households_rate_pct",
        "total_tenure_households_count",
        "renter_households_direct_count",
        "renter_households_rate_pct",
        "pct_renter_occupied",
        "median_value_of_dwellings_profile_value",
        "property_value_density_estimated_owner_from_pct_renter",
        "estimated_owner_households_from_pct_renter",
        "estimated_owner_occupied_residential_property_value_from_pct_renter",
        "property_value_density_estimated_owner_from_total_minus_renter",
        "estimated_owner_households_from_total_minus_renter",
        "estimated_owner_occupied_residential_property_value_from_total_minus_renter",
        "owner_plus_renter_minus_total_tenure_count",
        "owner_direct_minus_owner_estimated_from_pct_renter",
        "median_value_profile_minus_housing_clean",
    ]
)

for alias in CHARACTERISTICS:
    ordered_cols += [
        f"{alias}_characteristic_id",
        f"{alias}_characteristic_name",
    ]

ordered_cols += [
    "source_file",
    "source_housing_clean_file",
    "source_section",
    "source_encoding",
    "proxy_quality",
    "method_note",
]

ordered_cols = [col for col in ordered_cols if col in clean.columns]
clean = clean[ordered_cols].copy()

clean.to_csv(OUTPUT_CLEAN, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CLEAN CENSUS DIVISION PROPERTY VALUE DENSITY 2021")
print("=" * 72)

print("\nClean output:")
print("Rows:", len(clean))
print("Unique census divisions:", clean["census_division_dguid"].nunique())
print("Variables cleaned: property_value_density")
print("Original SoVI code: RPROPDEN92")
print("Proxy quality: weak_residential_owner_occupied_property_value_density_proxy")

print("\nFormula checks:")
for name, diff in formula_diffs.items():
    print(f"{name}: {diff}")

print("\nConsistency checks:")
print(
    "owner_plus_renter_minus_total_abs_max:",
    clean["owner_plus_renter_minus_total_tenure_count"].abs().max(skipna=True),
)
print(
    "owner_direct_minus_owner_estimated_from_pct_renter_abs_max:",
    clean["owner_direct_minus_owner_estimated_from_pct_renter"].abs().max(skipna=True),
)

print("\nMojibake check:")
print("Base names with mojibake:", base_names_with_mojibake)
print("Clean names with mojibake:", clean_names_with_mojibake)
print("Raw characteristic names with mojibake:", raw_characteristic_names_with_mojibake)

print("\nSource characteristic inventory:")
print(component_audit.to_string(index=False))

print("\nMain summaries:")
for variable in [
    "property_value_density",
    "estimated_owner_occupied_residential_property_value",
    "median_owner_occupied_housing_value",
    "owner_households_direct_count",
    "property_value_density_estimated_owner_from_pct_renter",
    "property_value_density_estimated_owner_from_total_minus_renter",
]:
    print("\n", variable)
    print(clean[variable].describe())

print("\nPreview:")
preview_cols = [
    "census_division_code",
    "census_division_name",
    "property_value_density",
    "estimated_owner_occupied_residential_property_value",
    "median_owner_occupied_housing_value",
    "owner_households_direct_count",
    "land_area_km2",
]
preview_cols = [col for col in preview_cols if col in clean.columns]
print(clean[preview_cols].head(20).to_string(index=False))

print("\nSaved:")
print(OUTPUT_CLEAN)
print(OUTPUT_SOURCE_ROWS)
print(OUTPUT_METADATA)
print(OUTPUT_FORMULA_AUDIT)
print(OUTPUT_UNMATCHED_AUDIT)
print(OUTPUT_SUMMARY)

print("\nDone.")