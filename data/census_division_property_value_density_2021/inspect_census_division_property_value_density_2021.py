from pathlib import Path
import codecs
import re
import pandas as pd


# ============================================================
# Inspect Census Division Property Value Density 2021
# ============================================================
#
# Purpose:
#   Inspect whether existing Census Profile housing-tenure/value data can
#   support a fallback proxy for:
#
#       RPROPDEN92 -> property_value_density
#
# Original SoVI concept:
#   Value of all property and farm products sold per square mile.
#
# Canadian fallback inspected here:
#   A weak residential owner-occupied dwelling-value-density proxy:
#
#       property_value_density_proxy =
#           median_owner_occupied_housing_value
#           * owner_household_count
#           / land_area_km2
#
# Important:
#   This is not a true aggregate property-assessment value. It does not include
#   rental property value, commercial property, industrial property, institutional
#   property, farm property, or land-assessment values. It should only be used
#   as a documented fallback proxy if no stronger municipal/property-assessment
#   source is available.
#
# Inputs:
#   - Existing clean housing tenure/costs output
#   - Raw 2021 Census Profile, to search for owner/tenure count rows
#   - Base Québec CD spatial frame, for land_area_km2
#
# Run from data/:
#
#   python census_division_property_value_density_2021/inspect_census_division_property_value_density_2021.py
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

OUTPUT_SUMMARY = OUTPUT_DIR / "property_value_density_inspection_summary_2021.csv"
OUTPUT_CHARACTERISTIC_SUMMARY = OUTPUT_DIR / "property_value_density_characteristic_summary_2021.csv"
OUTPUT_CANDIDATE_SOURCE_ROWS = OUTPUT_DIR / "property_value_density_candidate_source_rows_2021.csv"
OUTPUT_FORMULA_AUDIT = OUTPUT_DIR / "property_value_density_formula_audit_2021.csv"
OUTPUT_TARGET_SUMMARY = OUTPUT_DIR / "property_value_density_target_summary_2021.csv"
OUTPUT_UNMATCHED_AUDIT = OUTPUT_DIR / "property_value_density_unmatched_cd_audit_2021.csv"


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

KEYWORDS = [
    "tenure",
    "owner",
    "renter",
    "private households by tenure",
    "median value of dwellings",
    "value of dwellings",
    "owner households",
    "occupied private dwellings",
    "private dwellings",
]

# Known / expected Census Profile IDs from the housing tenure-costs cleaning.
KNOWN_IDS = {
    "renter_rate": "1416",
    "median_value_of_dwellings": "1488",
}

# Likely nearby tenure rows. The script validates names and coverage before use.
PREFERRED_OWNER_COUNT_NAME_TERMS = [
    "owner",
]

PREFERRED_TOTAL_TENURE_NAME_TERMS = [
    "tenure",
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
    Strict full-file encoding detection.

    Do not use nrows-based detection for StatCan Census Profile files,
    because early rows may decode as UTF-8 while later accented bytes require cp1252.
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


def classify_characteristic(name: object, char_id: object) -> dict:
    text = normalize_lower(name)
    char_id_text = normalize_text(char_id)

    is_known_renter = char_id_text == KNOWN_IDS["renter_rate"]
    is_known_median_value = char_id_text == KNOWN_IDS["median_value_of_dwellings"]

    is_owner = text == "owner" or text.endswith(" owner") or "owner" in text
    is_renter = text == "renter" or text.endswith(" renter") or "renter" in text
    is_tenure = "tenure" in text
    is_median_value = "median value of dwellings" in text or "value of dwellings" in text
    is_private_household = "private household" in text or "private households" in text
    is_occupied_private_dwelling = "occupied private dwelling" in text or "occupied private dwellings" in text

    if is_known_median_value or ("median value of dwellings" in text):
        role = "median_dwelling_value_component"
        score = 100
    elif is_owner and not is_renter:
        role = "owner_household_count_candidate"
        score = 95
    elif is_known_renter or is_renter:
        role = "renter_tenure_component"
        score = 80
    elif is_tenure:
        role = "total_tenure_denominator_candidate"
        score = 70
    elif is_occupied_private_dwelling:
        role = "occupied_private_dwellings_context"
        score = 45
    elif is_private_household:
        role = "private_households_context"
        score = 35
    else:
        role = "keyword_context"
        score = 10

    return {
        "role": role,
        "relevance_score": score,
        "is_known_renter_rate_id_1416": is_known_renter,
        "is_known_median_value_id_1488": is_known_median_value,
        "contains_owner": is_owner,
        "contains_renter": is_renter,
        "contains_tenure": is_tenure,
        "contains_median_value": is_median_value,
        "contains_private_household": is_private_household,
        "contains_occupied_private_dwelling": is_occupied_private_dwelling,
    }


def summarize_characteristic(
    group: pd.DataFrame,
    char_id_col: str,
    char_name_col: str,
    dguid_col: str,
    count_col: str,
    rate_col: str | None,
    count_symbol_col: str | None,
    rate_symbol_col: str | None,
) -> dict:
    char_id = normalize_text(group[char_id_col].iloc[0])
    char_name = normalize_text(group[char_name_col].iloc[0])
    class_info = classify_characteristic(char_name, char_id)

    count_stats = numeric_summary(group[count_col])

    row = {
        "CHARACTERISTIC_ID": char_id,
        "CHARACTERISTIC_NAME": char_name,
        **class_info,
        "n_rows": len(group),
        "n_unique_quebec_cds": int(group[dguid_col].astype(str).nunique()),
        "coverage_is_98_cds": int(group[dguid_col].astype(str).nunique()) == EXPECTED_QC_CD_COUNT,
        "count_column": count_col,
        "count_non_missing": count_stats["non_missing"],
        "count_missing": count_stats["missing"],
        "count_min": count_stats["min"],
        "count_max": count_stats["max"],
        "count_mean": count_stats["mean"],
        "count_median": count_stats["median"],
    }

    if rate_col and rate_col in group.columns:
        rate_stats = numeric_summary(group[rate_col])
        row.update(
            {
                "rate_column": rate_col,
                "rate_non_missing": rate_stats["non_missing"],
                "rate_missing": rate_stats["missing"],
                "rate_min": rate_stats["min"],
                "rate_max": rate_stats["max"],
                "rate_mean": rate_stats["mean"],
                "rate_median": rate_stats["median"],
            }
        )
    else:
        row.update(
            {
                "rate_column": "",
                "rate_non_missing": 0,
                "rate_missing": len(group),
                "rate_min": None,
                "rate_max": None,
                "rate_mean": None,
                "rate_median": None,
            }
        )

    if count_symbol_col and count_symbol_col in group.columns:
        row["count_symbols"] = " | ".join(sorted(group[count_symbol_col].dropna().astype(str).unique())[:20])
    else:
        row["count_symbols"] = ""

    if rate_symbol_col and rate_symbol_col in group.columns:
        row["rate_symbols"] = " | ".join(sorted(group[rate_symbol_col].dropna().astype(str).unique())[:20])
    else:
        row["rate_symbols"] = ""

    return row


def extract_profile_characteristic(
    qc_rows: pd.DataFrame,
    char_id: object,
    alias: str,
    dguid_col: str,
    char_id_col: str,
    char_name_col: str,
    count_col: str,
    rate_col: str | None,
    geo_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    char_id_str = str(char_id)
    rows = qc_rows[qc_rows[char_id_col].astype(str).str.strip() == char_id_str].copy()

    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    source_cols = [dguid_col]
    if geo_col and geo_col in rows.columns:
        source_cols.append(geo_col)
    source_cols += [char_id_col, char_name_col, count_col]
    if rate_col and rate_col in rows.columns:
        source_cols.append(rate_col)

    source_rows = rows[source_cols].copy()
    source_rows.insert(0, "source_alias", alias)

    out_cols = [dguid_col, count_col]
    rename_map = {
        dguid_col: "census_division_dguid",
        count_col: f"{alias}_count_value",
    }

    if rate_col and rate_col in rows.columns:
        out_cols.append(rate_col)
        rename_map[rate_col] = f"{alias}_rate_value"

    extracted = rows[out_cols].copy()
    extracted[dguid_col] = extracted[dguid_col].astype("string").str.strip()
    extracted[count_col] = clean_numeric(extracted[count_col])

    if rate_col and rate_col in extracted.columns:
        extracted[rate_col] = clean_numeric(extracted[rate_col])

    extracted = extracted.drop_duplicates(subset=[dguid_col], keep="first")
    extracted = extracted.rename(columns=rename_map)

    return extracted, source_rows


def choose_best_characteristic(
    characteristic_summary: pd.DataFrame,
    role: str,
    preferred_exact_name: str | None = None,
) -> pd.Series | None:
    if characteristic_summary.empty:
        return None

    subset = characteristic_summary[
        (characteristic_summary["role"] == role)
        & (characteristic_summary["coverage_is_98_cds"] == True)
        & (characteristic_summary["count_non_missing"] == EXPECTED_QC_CD_COUNT)
    ].copy()

    if subset.empty:
        return None

    if preferred_exact_name:
        exact = subset[
            subset["CHARACTERISTIC_NAME"].map(normalize_lower) == normalize_lower(preferred_exact_name)
        ].copy()
        if not exact.empty:
            return exact.sort_values(["CHARACTERISTIC_ID"]).iloc[0]

    subset["_name_len"] = subset["CHARACTERISTIC_NAME"].astype(str).str.len()
    subset = subset.sort_values(
        ["relevance_score", "_name_len", "CHARACTERISTIC_ID"],
        ascending=[False, True, True],
    )

    return subset.iloc[0]


def make_formula_row(alias: str, formula: str, joined: pd.DataFrame, status: str, interpretation: str) -> dict:
    density = clean_numeric(joined[alias])
    numerator_col = alias.replace("_density", "_estimated_value")
    numerator = clean_numeric(joined[numerator_col]) if numerator_col in joined.columns else pd.Series(dtype=float)

    return {
        "candidate_alias": alias,
        "candidate_formula": formula,
        "non_missing": int(density.notna().sum()),
        "missing": int(density.isna().sum()),
        "coverage_is_98_cds": int(density.notna().sum()) == EXPECTED_QC_CD_COUNT,
        "estimated_value_min": numerator.min(skipna=True) if not numerator.empty else None,
        "estimated_value_max": numerator.max(skipna=True) if not numerator.empty else None,
        "estimated_value_mean": numerator.mean(skipna=True) if not numerator.empty else None,
        "estimated_value_median": numerator.median(skipna=True) if not numerator.empty else None,
        "density_min": density.min(skipna=True),
        "density_max": density.max(skipna=True),
        "density_mean": density.mean(skipna=True),
        "density_median": density.median(skipna=True),
        "status": status,
        "recommended_default_without_review": False,
        "proxy_quality": "weak_residential_owner_occupied_property_value_density_proxy",
        "interpretation": interpretation,
    }


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
base = pd.read_csv(BASE_CD_FRAME, encoding=base_encoding, dtype=str, low_memory=False)
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
        "Base frame missing required columns:\n"
        + "\n".join(missing_base_cols)
        + "\n\nAvailable columns:\n"
        + "\n".join(base.columns)
    )

base = base.copy()
base["census_division_code"] = base["census_division_code"].astype("string").str.strip()
base["census_division_dguid"] = base["census_division_dguid"].astype("string").str.strip()
base["land_area_km2"] = clean_numeric(base["land_area_km2"])

if len(base) != EXPECTED_QC_CD_COUNT:
    raise ValueError(f"Expected {EXPECTED_QC_CD_COUNT} base rows, got {len(base)}.")

if base["census_division_dguid"].duplicated().any():
    raise ValueError("Duplicate census_division_dguid values in base frame.")

if base["land_area_km2"].isna().any() or (base["land_area_km2"] <= 0).any():
    raise ValueError("Missing or non-positive land_area_km2 values in base frame.")

base_dguid_set = set(base["census_division_dguid"].dropna().astype(str))


# -----------------------------
# Load housing tenure/costs clean file
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
housing["median_owner_occupied_housing_value"] = clean_numeric(housing["median_owner_occupied_housing_value"])
housing["pct_renter_occupied"] = clean_numeric(housing["pct_renter_occupied"])

if len(housing) != EXPECTED_QC_CD_COUNT:
    raise ValueError(f"Expected {EXPECTED_QC_CD_COUNT} housing rows, got {len(housing)}.")

if housing["census_division_dguid"].duplicated().any():
    raise ValueError("Duplicate census_division_dguid values in housing clean file.")

if housing["median_owner_occupied_housing_value"].isna().any():
    raise ValueError("Missing median_owner_occupied_housing_value in housing clean file.")

if housing["pct_renter_occupied"].isna().any():
    raise ValueError("Missing pct_renter_occupied in housing clean file.")


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
geo_col = find_column(columns, ["GEO", "Geography", "Geography name"])
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
# Keyword scan and characteristic summary
# -----------------------------

name_lower = qc_rows[char_name_col].map(normalize_lower)
keyword_mask = name_lower.apply(lambda text: any(keyword in text for keyword in KEYWORDS))

# Also force known IDs into the inspection.
known_id_mask = qc_rows[char_id_col].astype(str).str.strip().isin(set(KNOWN_IDS.values()))
keyword_rows = qc_rows[keyword_mask | known_id_mask].copy()

summary_rows = []

if not keyword_rows.empty:
    grouped = keyword_rows.groupby([char_id_col, char_name_col], dropna=False)

    for _, group in grouped:
        summary_rows.append(
            summarize_characteristic(
                group=group,
                char_id_col=char_id_col,
                char_name_col=char_name_col,
                dguid_col=dguid_col,
                count_col=count_col,
                rate_col=rate_col,
                count_symbol_col=count_symbol_col,
                rate_symbol_col=rate_symbol_col,
            )
        )

characteristic_summary = pd.DataFrame(summary_rows)

if not characteristic_summary.empty:
    characteristic_summary = characteristic_summary.sort_values(
        ["relevance_score", "CHARACTERISTIC_ID"],
        ascending=[False, True],
    )

characteristic_summary.to_csv(OUTPUT_CHARACTERISTIC_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Select source components
# -----------------------------

owner_row = choose_best_characteristic(
    characteristic_summary,
    role="owner_household_count_candidate",
    preferred_exact_name="Owner",
)

total_tenure_row = choose_best_characteristic(
    characteristic_summary,
    role="total_tenure_denominator_candidate",
)

median_value_row = characteristic_summary[
    characteristic_summary["CHARACTERISTIC_ID"].astype(str) == KNOWN_IDS["median_value_of_dwellings"]
].copy()

if median_value_row.empty:
    median_value_row = characteristic_summary[
        characteristic_summary["role"] == "median_dwelling_value_component"
    ].copy()

if not median_value_row.empty:
    median_value_row = median_value_row.sort_values(["relevance_score", "CHARACTERISTIC_ID"], ascending=[False, True]).iloc[0]
else:
    median_value_row = None

renter_row = characteristic_summary[
    characteristic_summary["CHARACTERISTIC_ID"].astype(str) == KNOWN_IDS["renter_rate"]
].copy()

if renter_row.empty:
    renter_row = characteristic_summary[characteristic_summary["role"] == "renter_tenure_component"].copy()

if not renter_row.empty:
    renter_row = renter_row.sort_values(["relevance_score", "CHARACTERISTIC_ID"], ascending=[False, True]).iloc[0]
else:
    renter_row = None


# -----------------------------
# Extract candidate source rows
# -----------------------------

source_frames = []
component_tables = {}

selected_components = []

if owner_row is not None:
    selected_components.append(("owner_households", owner_row["CHARACTERISTIC_ID"]))

if total_tenure_row is not None:
    selected_components.append(("total_tenure_households", total_tenure_row["CHARACTERISTIC_ID"]))

if median_value_row is not None:
    selected_components.append(("median_value_of_dwellings", median_value_row["CHARACTERISTIC_ID"]))

if renter_row is not None:
    selected_components.append(("renter_tenure", renter_row["CHARACTERISTIC_ID"]))

for alias, char_id in selected_components:
    extracted, source_rows = extract_profile_characteristic(
        qc_rows=qc_rows,
        char_id=char_id,
        alias=alias,
        dguid_col=dguid_col,
        char_id_col=char_id_col,
        char_name_col=char_name_col,
        count_col=count_col,
        rate_col=rate_col,
        geo_col=geo_col,
    )

    if not extracted.empty:
        component_tables[alias] = extracted
    if not source_rows.empty:
        source_frames.append(source_rows)

if source_frames:
    candidate_source_rows = pd.concat(source_frames, ignore_index=True, sort=False)
else:
    candidate_source_rows = pd.DataFrame()

candidate_source_rows.to_csv(OUTPUT_CANDIDATE_SOURCE_ROWS, index=False, encoding="utf-8")


# -----------------------------
# Formula audit
# -----------------------------

joined = base[
    [
        "census_division_code",
        "census_division_dguid",
        "census_division_name",
        "land_area_km2",
    ]
].merge(
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
    value_cols = [col for col in table.columns if col != "census_division_dguid"]
    joined = joined.merge(
        table[["census_division_dguid"] + value_cols],
        on="census_division_dguid",
        how="left",
        validate="one_to_one",
    )

# Direct owner-count formula.
formula_rows = []

if "owner_households" in component_tables:
    joined["owner_households_direct_count"] = clean_numeric(joined["owner_households_count_value"])

    joined["property_value_density_direct_owner_count_estimated_value"] = (
        joined["median_owner_occupied_housing_value"]
        * joined["owner_households_direct_count"]
    )

    joined["property_value_density_direct_owner_count_density"] = (
        joined["property_value_density_direct_owner_count_estimated_value"]
        / joined["land_area_km2"]
    )

    formula_rows.append(
        make_formula_row(
            alias="property_value_density_direct_owner_count_density",
            formula="median_owner_occupied_housing_value * owner_households_direct_count / land_area_km2",
            joined=joined,
            status=(
                "candidate_available_full_coverage"
                if joined["property_value_density_direct_owner_count_density"].notna().sum() == EXPECTED_QC_CD_COUNT
                else "candidate_partial_or_missing"
            ),
            interpretation=(
                "Weak fallback proxy using Census Profile owner household count times median dwelling value, divided by land area."
            ),
        )
    )

# Estimate owner count from total tenure count and renter percentage.
if "total_tenure_households" in component_tables:
    joined["total_tenure_households_count"] = clean_numeric(joined["total_tenure_households_count_value"])

    joined["owner_households_estimated_from_total_and_pct_renter"] = (
        joined["total_tenure_households_count"]
        * (1 - joined["pct_renter_occupied"] / 100)
    )

    joined["property_value_density_estimated_owner_from_pct_renter_estimated_value"] = (
        joined["median_owner_occupied_housing_value"]
        * joined["owner_households_estimated_from_total_and_pct_renter"]
    )

    joined["property_value_density_estimated_owner_from_pct_renter_density"] = (
        joined["property_value_density_estimated_owner_from_pct_renter_estimated_value"]
        / joined["land_area_km2"]
    )

    formula_rows.append(
        make_formula_row(
            alias="property_value_density_estimated_owner_from_pct_renter_density",
            formula="median_owner_occupied_housing_value * total_tenure_households_count * (1 - pct_renter_occupied / 100) / land_area_km2",
            joined=joined,
            status=(
                "candidate_available_full_coverage"
                if joined["property_value_density_estimated_owner_from_pct_renter_density"].notna().sum() == EXPECTED_QC_CD_COUNT
                else "candidate_partial_or_missing"
            ),
            interpretation=(
                "Fallback proxy estimating owner households from total tenure households and renter percentage."
            ),
        )
    )

# Context-only density from median value alone.
joined["median_dwelling_value_per_km2_context_density"] = (
    joined["median_owner_occupied_housing_value"] / joined["land_area_km2"]
)

joined["median_dwelling_value_per_km2_context_estimated_value"] = (
    joined["median_owner_occupied_housing_value"]
)

formula_rows.append(
    make_formula_row(
        alias="median_dwelling_value_per_km2_context_density",
        formula="median_owner_occupied_housing_value / land_area_km2",
        joined=joined,
        status="context_only_not_recommended",
        interpretation=(
            "Context-only variable. This is not an aggregate property-value density and should not be used as the default SoVI proxy."
        ),
    )
)

formula_audit = pd.DataFrame(formula_rows)
formula_audit.to_csv(OUTPUT_FORMULA_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Unmatched audit
# -----------------------------

unmatched_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
]

for col in [
    "owner_households_direct_count",
    "total_tenure_households_count",
    "property_value_density_direct_owner_count_density",
    "property_value_density_estimated_owner_from_pct_renter_density",
]:
    if col in joined.columns:
        unmatched_cols.append(col)

unmatched = joined[unmatched_cols].copy()

density_cols = [col for col in unmatched.columns if col.endswith("_density")]
if density_cols:
    missing_mask = unmatched[density_cols].isna().any(axis=1)
    unmatched = unmatched[missing_mask].copy()
else:
    unmatched = unmatched.iloc[0:0].copy()

unmatched.to_csv(OUTPUT_UNMATCHED_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Target summary
# -----------------------------

full_coverage_rows = formula_audit[
    (formula_audit["coverage_is_98_cds"] == True)
    & (formula_audit["status"] != "context_only_not_recommended")
].copy()

if full_coverage_rows.empty:
    target_summary = pd.DataFrame(
        [
            {
                "canonical_variable": "property_value_density",
                "original_sovi_code": "RPROPDEN92",
                "candidate_found": False,
                "best_candidate_alias": "",
                "best_candidate_formula": "",
                "coverage_is_98_cds": False,
                "status": "no_full_coverage_proxy_candidate",
                "proxy_quality": "",
                "interpretation": (
                    "No full-coverage residential property-value-density proxy could be constructed from the inspected components."
                ),
            }
        ]
    )
else:
    preferred_order = [
        "property_value_density_direct_owner_count_density",
        "property_value_density_estimated_owner_from_pct_renter_density",
    ]

    full_coverage_rows["_rank"] = full_coverage_rows["candidate_alias"].apply(
        lambda x: preferred_order.index(x) if x in preferred_order else 999
    )

    best = full_coverage_rows.sort_values(["_rank", "candidate_alias"]).iloc[0]

    target_summary = pd.DataFrame(
        [
            {
                "canonical_variable": "property_value_density",
                "original_sovi_code": "RPROPDEN92",
                "candidate_found": True,
                "best_candidate_alias": best["candidate_alias"],
                "best_candidate_formula": best["candidate_formula"],
                "coverage_is_98_cds": True,
                "status": "candidate_available_needs_methodological_decision",
                "proxy_quality": best["proxy_quality"],
                "interpretation": (
                    "A full-coverage weak residential owner-occupied property-value-density proxy is available. "
                    "Use only if no stronger total assessed property-value source is available."
                ),
            }
        ]
    )

target_summary.to_csv(OUTPUT_TARGET_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Inspection summary
# -----------------------------

raw_names_with_mojibake = contains_mojibake(raw[char_name_col])
base_names_with_mojibake = contains_mojibake(base["census_division_name"])
housing_names_with_mojibake = contains_mojibake(housing["census_division_name"]) if "census_division_name" in housing.columns else 0

owner_candidate_found = owner_row is not None
total_tenure_candidate_found = total_tenure_row is not None
median_value_component_found = median_value_row is not None
renter_component_found = renter_row is not None

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
    {"metric": "detected_dguid_column", "value": dguid_col},
    {"metric": "detected_geo_column", "value": geo_col or ""},
    {"metric": "detected_characteristic_id_column", "value": char_id_col},
    {"metric": "detected_characteristic_name_column", "value": char_name_col},
    {"metric": "detected_count_column", "value": count_col},
    {"metric": "detected_rate_column", "value": rate_col or ""},
    {"metric": "keyword_match_rows", "value": len(keyword_rows)},
    {"metric": "candidate_characteristics_found", "value": len(characteristic_summary)},
    {"metric": "owner_count_candidate_found", "value": owner_candidate_found},
    {"metric": "owner_count_candidate_id", "value": owner_row["CHARACTERISTIC_ID"] if owner_row is not None else ""},
    {"metric": "owner_count_candidate_name", "value": owner_row["CHARACTERISTIC_NAME"] if owner_row is not None else ""},
    {"metric": "total_tenure_candidate_found", "value": total_tenure_candidate_found},
    {"metric": "total_tenure_candidate_id", "value": total_tenure_row["CHARACTERISTIC_ID"] if total_tenure_row is not None else ""},
    {"metric": "total_tenure_candidate_name", "value": total_tenure_row["CHARACTERISTIC_NAME"] if total_tenure_row is not None else ""},
    {"metric": "median_value_component_found", "value": median_value_component_found},
    {"metric": "median_value_component_id", "value": median_value_row["CHARACTERISTIC_ID"] if median_value_row is not None else ""},
    {"metric": "median_value_component_name", "value": median_value_row["CHARACTERISTIC_NAME"] if median_value_row is not None else ""},
    {"metric": "renter_component_found", "value": renter_component_found},
    {"metric": "renter_component_id", "value": renter_row["CHARACTERISTIC_ID"] if renter_row is not None else ""},
    {"metric": "renter_component_name", "value": renter_row["CHARACTERISTIC_NAME"] if renter_row is not None else ""},
    {"metric": "formula_candidates_tested", "value": len(formula_audit)},
    {"metric": "full_coverage_proxy_candidates", "value": len(full_coverage_rows)},
    {"metric": "property_value_density_proxy_ready", "value": not full_coverage_rows.empty},
    {"metric": "raw_names_with_mojibake", "value": raw_names_with_mojibake},
    {"metric": "base_names_with_mojibake", "value": base_names_with_mojibake},
    {"metric": "housing_names_with_mojibake", "value": housing_names_with_mojibake},
    {
        "metric": "important_method_note",
        "value": (
            "RPROPDEN92 is not directly reproduced here. The inspected formula is a weak residential owner-occupied "
            "dwelling-value-density proxy based on median dwelling value times owner household count divided by land area. "
            "It is not a total property-assessment value and should be documented as a fallback proxy."
        ),
    },
    {
        "metric": "recommended_next_step",
        "value": (
            "Review property_value_density_formula_audit_2021.csv and property_value_density_target_summary_2021.csv. "
            "If the direct owner-count candidate has 98/98 coverage and the weak proxy is acceptable, generate the cleaner. "
            "Otherwise, search for a municipal property-assessment or richesse foncière uniformisée source."
        ),
    },
]

for col in [
    "median_owner_occupied_housing_value",
    "pct_renter_occupied",
    "land_area_km2",
]:
    source_df = joined if col in joined.columns else housing
    stats = numeric_summary(source_df[col])
    for key, value in stats.items():
        summary_rows.append({"metric": f"{col}_{key}", "value": value})

for col in [
    "owner_households_direct_count",
    "total_tenure_households_count",
    "property_value_density_direct_owner_count_density",
    "property_value_density_estimated_owner_from_pct_renter_density",
]:
    if col in joined.columns:
        stats = numeric_summary(joined[col])
        for key, value in stats.items():
            summary_rows.append({"metric": f"{col}_{key}", "value": value})

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CENSUS DIVISION PROPERTY VALUE DENSITY INSPECTION 2021")
print("=" * 72)

print("\nInputs:")
print("Raw profile:", safe_relative(RAW_PROFILE))
print("Base frame:", safe_relative(BASE_CD_FRAME))
print("Housing clean:", safe_relative(HOUSING_TENURE_COSTS_CLEAN))
print("Raw encoding:", raw_encoding)
print("Housing encoding:", housing_encoding)

print("\nDetected components:")
print("Owner count candidate:", owner_row["CHARACTERISTIC_ID"] if owner_row is not None else "[none]",
      "-", owner_row["CHARACTERISTIC_NAME"] if owner_row is not None else "")
print("Total tenure candidate:", total_tenure_row["CHARACTERISTIC_ID"] if total_tenure_row is not None else "[none]",
      "-", total_tenure_row["CHARACTERISTIC_NAME"] if total_tenure_row is not None else "")
print("Median value component:", median_value_row["CHARACTERISTIC_ID"] if median_value_row is not None else "[none]",
      "-", median_value_row["CHARACTERISTIC_NAME"] if median_value_row is not None else "")
print("Renter component:", renter_row["CHARACTERISTIC_ID"] if renter_row is not None else "[none]",
      "-", renter_row["CHARACTERISTIC_NAME"] if renter_row is not None else "")

print("\nTarget summary:")
print(target_summary.to_string(index=False))

print("\nFormula audit:")
print(formula_audit.to_string(index=False))

print("\nMojibake check:")
print("Raw names with mojibake:", raw_names_with_mojibake)
print("Base names with mojibake:", base_names_with_mojibake)
print("Housing names with mojibake:", housing_names_with_mojibake)

print("\nSaved:")
print(OUTPUT_SUMMARY)
print(OUTPUT_CHARACTERISTIC_SUMMARY)
print(OUTPUT_CANDIDATE_SOURCE_ROWS)
print(OUTPUT_FORMULA_AUDIT)
print(OUTPUT_TARGET_SUMMARY)
print(OUTPUT_UNMATCHED_AUDIT)

print("\nDone.")