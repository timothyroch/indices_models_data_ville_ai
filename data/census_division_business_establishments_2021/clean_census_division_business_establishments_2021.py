from pathlib import Path
import codecs
import re
import pandas as pd


# ============================================================
# Clean Census Division Business Establishments 2021
# ============================================================
#
# Purpose:
#   Clean Québec census-division business-establishment-density proxies for:
#
#       MAESDEN92   -> manufacturing_density
#       COMDEVDN92  -> commercial_density
#
# Source:
#   Business Counts in Rural and Small Town Canada dashboard extract.
#
# Raw file expected:
#
#   census_division_business_establishments_2021/raw/
#       canada_rural_business_counts_dashboard.csv
#
# Source structure:
#   The dashboard extract is at census subdivision geography, with 2021
#   CSD DGUIDs, semester time periods, employment-size classes, NAICS
#   industry classes, and observed business counts.
#
# Main period:
#   2022-01
#
# Why 2022-01:
#   The dashboard extract inspected for Québec contains periods from
#   2022-01 onward. 2022-01 is the earliest available period and is the
#   closest available business-count period to the 2021 SoVI-like table.
#
# Main formulas:
#
#   manufacturing_density =
#       manufacturing_business_count / land_area_km2
#
#   commercial_density =
#       (
#           wholesale_trade_business_count
#           + retail_trade_business_count
#           + accommodation_food_services_business_count
#       ) / land_area_km2
#
# NAICS components:
#
#   Manufacturing:
#       31-33 Manufacturing
#
#   Commercial default:
#       41 Wholesale trade
#       44-45 Retail trade
#       72 Accommodation and food services
#
# Audit commercial alternative:
#       41 Wholesale trade
#       44-45 Retail trade
#
# Run from data/:
#
#   python census_division_business_establishments_2021/clean_census_division_business_establishments_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_business_establishments_2021"
RAW_DIR = SECTION_DIR / "raw"
OUTPUT_DIR = SECTION_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = RAW_DIR / "canada_rural_business_counts_dashboard.csv"

BASE_CD_FRAME = (
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.csv"
)

OUTPUT_CLEAN = OUTPUT_DIR / "clean_census_division_business_establishments_2021.csv"
OUTPUT_SOURCE_ROWS = OUTPUT_DIR / "clean_census_division_business_establishments_source_rows_2021.csv"
OUTPUT_METADATA = OUTPUT_DIR / "clean_census_division_business_establishments_variable_metadata_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "clean_census_division_business_establishments_summary_2021.csv"
OUTPUT_COMPONENT_AUDIT = OUTPUT_DIR / "clean_census_division_business_establishments_component_audit_2021.csv"
OUTPUT_UNMATCHED_AUDIT = OUTPUT_DIR / "clean_census_division_business_establishments_unmatched_audit_2021.csv"
OUTPUT_TIME_PERIOD_AUDIT = OUTPUT_DIR / "clean_census_division_business_establishments_time_period_audit_2021.csv"


# -----------------------------
# Config
# -----------------------------

EXPECTED_QC_CD_COUNT = 98
SELECTED_TIME_PERIOD = "2022-01"
CHUNK_SIZE = 250_000

ENCODING_CANDIDATES = [
    "utf-8-sig",
    "utf-8",
    "cp1252",
    "latin1",
]

COMPONENTS = {
    "manufacturing": {
        "industry_code": "31-33",
        "industry_label": "Manufacturing",
        "original_sovi_code": "MAESDEN92",
        "target_variable": "manufacturing_density",
        "role": "main_manufacturing_component",
    },
    "wholesale_trade": {
        "industry_code": "41",
        "industry_label": "Wholesale trade",
        "original_sovi_code": "COMDEVDN92",
        "target_variable": "commercial_density",
        "role": "commercial_component",
    },
    "retail_trade": {
        "industry_code": "44-45",
        "industry_label": "Retail trade",
        "original_sovi_code": "COMDEVDN92",
        "target_variable": "commercial_density",
        "role": "commercial_component",
    },
    "accommodation_food_services": {
        "industry_code": "72",
        "industry_label": "Accommodation and food services",
        "original_sovi_code": "COMDEVDN92",
        "target_variable": "commercial_density",
        "role": "commercial_component",
    },
}

COMMERCIAL_TRADE_ONLY_COMPONENTS = [
    "wholesale_trade",
    "retail_trade",
]

COMMERCIAL_DEFAULT_COMPONENTS = [
    "wholesale_trade",
    "retail_trade",
    "accommodation_food_services",
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


def parse_prefixed_code(value: object) -> str:
    text = normalize_text(value)
    if ":" in text:
        return text.split(":", 1)[0].strip()
    return text


def parse_prefixed_label(value: object) -> str:
    text = normalize_text(value)
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text


def parse_industry_code(value: object) -> str:
    return parse_prefixed_code(value)


def parse_component_alias(industry_value: object) -> str:
    code = parse_industry_code(industry_value)
    label = normalize_lower(parse_prefixed_label(industry_value))

    if code == "31-33" or label == "manufacturing":
        return "manufacturing"
    if code == "41" or label == "wholesale trade":
        return "wholesale_trade"
    if code == "44-45" or label == "retail trade":
        return "retail_trade"
    if code == "72" or label == "accommodation and food services":
        return "accommodation_food_services"

    return ""


def extract_csd_sgc_from_dguid(value: object) -> str:
    text = normalize_text(value)

    # Example:
    #   2021A00052453065 -> 2453065
    match = re.search(r"A0005(\d{7})", text)
    if match:
        return match.group(1)

    return ""


def extract_csd_sgc_from_ref_area(value: object) -> str:
    code = parse_prefixed_code(value)
    if re.fullmatch(r"\d{7}", code):
        return code
    return ""


def extract_cd_code_from_csd_sgc(csd_sgc: object) -> str:
    text = normalize_text(csd_sgc)
    if re.fullmatch(r"\d{7}", text):
        return text[:4]
    return ""


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


def add_numeric_summary(summary_rows: list[dict], df: pd.DataFrame, variable: str) -> None:
    stats = numeric_summary(df[variable])
    for key, value in stats.items():
        summary_rows.append(
            {
                "metric": f"{variable}_{key}",
                "value": value,
            }
        )


# -----------------------------
# Validate inputs
# -----------------------------

if not RAW_CSV.exists():
    raise FileNotFoundError(f"Missing raw dashboard CSV:\n{RAW_CSV}")

if not BASE_CD_FRAME.exists():
    raise FileNotFoundError(f"Missing base CD frame:\n{BASE_CD_FRAME}")


# -----------------------------
# Load base frame
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

if base["census_division_code"].duplicated().any():
    raise ValueError("Duplicate census_division_code values in base frame.")

if base["land_area_km2"].isna().any() or (base["land_area_km2"] <= 0).any():
    raise ValueError("Missing or non-positive land_area_km2 values in base frame.")

base_cd_codes = set(base["census_division_code"].dropna().astype(str))


# -----------------------------
# Detect raw columns
# -----------------------------

raw_encoding = detect_file_encoding_strict(RAW_CSV, ENCODING_CANDIDATES)

sample = pd.read_csv(RAW_CSV, encoding=raw_encoding, dtype=str, nrows=50_000, low_memory=False)
sample.columns = [str(col).strip() for col in sample.columns]
columns = list(sample.columns)

ref_area_col = require_column(columns, ["REF_AREA", "Reference area"], "reference area")
emp_size_col = require_column(columns, ["EMP_SIZE", "Employment size"], "employment size")
industry_col = require_column(columns, ["INDUSTRY", "Industry"], "industry")
time_col = require_column(columns, ["TIME_PERIOD", "Reference period"], "time period")
value_col = require_column(columns, ["OBS_VALUE", "VALUE", "Value"], "observed value")
dguid_col = require_column(columns, ["DGUID"], "DGUID")
unit_col = find_column(columns, ["UNIT_MEASURE", "Unit of measure"])
conf_col = find_column(columns, ["CONF_STATUS", "Confidentiality status"])
obs_status_col = find_column(columns, ["OBS_STATUS", "Observation status"])
rural_flag_col = find_column(columns, ["RURAL_FLAG", "Rural area flag"])
prov_col = require_column(columns, ["PROV_TERR", "Province or territory"], "province or territory")


# -----------------------------
# Chunked extraction
# -----------------------------

needed_cols = [
    col for col in [
        ref_area_col,
        emp_size_col,
        industry_col,
        time_col,
        value_col,
        dguid_col,
        unit_col,
        conf_col,
        obs_status_col,
        rural_flag_col,
        prov_col,
    ]
    if col is not None
]

source_chunks = []
period_count_chunks = []

raw_rows_scanned = 0
qc_rows_seen = 0
qc_total_with_employees_rows_seen = 0
selected_total_with_employees_rows_seen = 0
selected_total_csd_codes = set()
selected_total_cd_codes = set()

reader = pd.read_csv(
    RAW_CSV,
    encoding=raw_encoding,
    dtype=str,
    usecols=needed_cols,
    chunksize=CHUNK_SIZE,
    low_memory=False,
)

for chunk in reader:
    chunk.columns = [str(col).strip() for col in chunk.columns]
    raw_rows_scanned += len(chunk)

    chunk[value_col] = clean_numeric(chunk[value_col])

    chunk["_province_code"] = chunk[prov_col].map(parse_prefixed_code)
    qc = chunk[chunk["_province_code"] == "24"].copy()
    qc_rows_seen += len(qc)

    if qc.empty:
        continue

    qc["_emp_size_code"] = qc[emp_size_col].map(parse_prefixed_code)
    qc["_emp_size_label"] = qc[emp_size_col].map(parse_prefixed_label)

    qc["_is_total_with_employees"] = (
        (qc["_emp_size_code"] == "_T")
        | qc["_emp_size_label"].str.lower().eq("total, with employees")
    )

    qc_total = qc[qc["_is_total_with_employees"]].copy()
    qc_total_with_employees_rows_seen += len(qc_total)

    if qc_total.empty:
        continue

    period_count_chunks.append(
        qc_total.groupby(time_col, dropna=False).size().reset_index(name="n_rows")
    )

    selected = qc_total[qc_total[time_col].astype(str) == SELECTED_TIME_PERIOD].copy()
    selected_total_with_employees_rows_seen += len(selected)

    if selected.empty:
        continue

    selected["_csd_sgc_code_from_dguid"] = selected[dguid_col].map(extract_csd_sgc_from_dguid)
    selected["_csd_sgc_code_from_ref_area"] = selected[ref_area_col].map(extract_csd_sgc_from_ref_area)
    selected["_csd_sgc_code"] = selected["_csd_sgc_code_from_dguid"]

    selected.loc[selected["_csd_sgc_code"] == "", "_csd_sgc_code"] = selected.loc[
        selected["_csd_sgc_code"] == "", "_csd_sgc_code_from_ref_area"
    ]

    selected["_is_quebec_csd"] = selected["_csd_sgc_code"].astype(str).str.fullmatch(r"24\d{5}", na=False)
    selected["_cd_code"] = selected["_csd_sgc_code"].map(extract_cd_code_from_csd_sgc)

    selected_csd = selected[selected["_is_quebec_csd"]].copy()

    selected_total_csd_codes.update(selected_csd["_csd_sgc_code"].dropna().astype(str).tolist())
    selected_total_cd_codes.update(selected_csd["_cd_code"].dropna().astype(str).tolist())

    if selected_csd.empty:
        continue

    selected_csd["_industry_code"] = selected_csd[industry_col].map(parse_industry_code)
    selected_csd["_industry_label"] = selected_csd[industry_col].map(parse_prefixed_label)
    selected_csd["_component_alias"] = selected_csd[industry_col].map(parse_component_alias)

    target_rows = selected_csd[selected_csd["_component_alias"] != ""].copy()

    if not target_rows.empty:
        source_chunks.append(target_rows)


if period_count_chunks:
    time_period_audit = (
        pd.concat(period_count_chunks, ignore_index=True)
        .groupby(time_col, dropna=False)["n_rows"]
        .sum()
        .reset_index()
        .rename(columns={time_col: "time_period"})
        .sort_values("time_period")
    )
else:
    time_period_audit = pd.DataFrame(columns=["time_period", "n_rows"])

time_period_audit.to_csv(OUTPUT_TIME_PERIOD_AUDIT, index=False, encoding="utf-8")

available_time_periods = time_period_audit["time_period"].dropna().astype(str).tolist()

if SELECTED_TIME_PERIOD not in available_time_periods:
    raise ValueError(
        f"Selected time period {SELECTED_TIME_PERIOD} was not found. "
        f"Available periods: {available_time_periods}"
    )

if not source_chunks:
    raise ValueError("No selected Québec source rows were extracted for the target NAICS components.")

source_rows = pd.concat(source_chunks, ignore_index=True, sort=False)

source_rows.to_csv(OUTPUT_SOURCE_ROWS, index=False, encoding="utf-8")


# -----------------------------
# Aggregate components from CSD to CD
# -----------------------------

def aggregate_component(component: str) -> tuple[pd.DataFrame, dict]:
    subset = source_rows[source_rows["_component_alias"] == component].copy()

    if subset.empty:
        raise ValueError(f"No source rows found for component: {component}")

    duplicate_rows = int(
        subset.duplicated(subset=["_csd_sgc_code", "_component_alias"]).sum()
    )

    if duplicate_rows != 0:
        duplicate_preview = subset[
            subset.duplicated(subset=["_csd_sgc_code", "_component_alias"], keep=False)
        ]
        raise ValueError(
            f"Duplicate selected-period CSD/component rows for {component}: {duplicate_rows}\n"
            + duplicate_preview[
                [ref_area_col, dguid_col, "_csd_sgc_code", "_cd_code", industry_col, time_col, value_col]
            ].head(30).to_string(index=False)
        )

    aggregated = (
        subset.groupby("_cd_code", dropna=False)[value_col]
        .sum(min_count=1)
        .reset_index()
        .rename(
            columns={
                "_cd_code": "census_division_code",
                value_col: f"{component}_business_count",
            }
        )
    )

    joined = base[
        ["census_division_code", "census_division_dguid", "census_division_name", "land_area_km2"]
    ].merge(
        aggregated,
        on="census_division_code",
        how="left",
        validate="one_to_one",
    )

    joined[f"{component}_density_per_km2"] = (
        joined[f"{component}_business_count"] / joined["land_area_km2"]
    )

    counts = clean_numeric(joined[f"{component}_business_count"])
    density = clean_numeric(joined[f"{component}_density_per_km2"])

    audit = {
        "component": component,
        "industry_code": COMPONENTS[component]["industry_code"],
        "industry_label": COMPONENTS[component]["industry_label"],
        "original_sovi_code": COMPONENTS[component]["original_sovi_code"],
        "target_variable": COMPONENTS[component]["target_variable"],
        "selected_time_period": SELECTED_TIME_PERIOD,
        "source_rows": len(subset),
        "unique_csd_rows": int(subset["_csd_sgc_code"].nunique()),
        "unique_cd_rows_before_join": int(subset["_cd_code"].nunique()),
        "duplicate_component_rows_before_cd_aggregation": duplicate_rows,
        "cd_rows_non_missing": int(counts.notna().sum()),
        "cd_rows_missing": int(counts.isna().sum()),
        "coverage_is_98_cds": int(counts.notna().sum()) == EXPECTED_QC_CD_COUNT,
        "business_count_min": counts.min(skipna=True),
        "business_count_max": counts.max(skipna=True),
        "business_count_mean": counts.mean(skipna=True),
        "business_count_median": counts.median(skipna=True),
        "density_per_km2_min": density.min(skipna=True),
        "density_per_km2_max": density.max(skipna=True),
        "density_per_km2_mean": density.mean(skipna=True),
        "density_per_km2_median": density.median(skipna=True),
    }

    return joined, audit


component_tables = {}
component_audit_rows = []

for component in COMPONENTS:
    table, audit = aggregate_component(component)
    component_tables[component] = table
    component_audit_rows.append(audit)


# -----------------------------
# Build clean table
# -----------------------------

identity_cols = [col for col in IDENTITY_COLUMNS if col in base.columns]
clean = base[identity_cols].copy()

for component, table in component_tables.items():
    clean = clean.merge(
        table[
            [
                "census_division_code",
                f"{component}_business_count",
                f"{component}_density_per_km2",
            ]
        ],
        on="census_division_code",
        how="left",
        validate="one_to_one",
    )

clean["commercial_trade_only_business_count"] = (
    clean["wholesale_trade_business_count"]
    + clean["retail_trade_business_count"]
)

clean["commercial_trade_accommodation_food_business_count"] = (
    clean["wholesale_trade_business_count"]
    + clean["retail_trade_business_count"]
    + clean["accommodation_food_services_business_count"]
)

clean["commercial_trade_only_density_per_km2"] = (
    clean["commercial_trade_only_business_count"] / clean["land_area_km2"]
)

clean["commercial_trade_accommodation_food_density_per_km2"] = (
    clean["commercial_trade_accommodation_food_business_count"] / clean["land_area_km2"]
)

clean["manufacturing_density"] = clean["manufacturing_density_per_km2"]

clean["commercial_density"] = clean["commercial_trade_accommodation_food_density_per_km2"]

clean["source_time_period"] = SELECTED_TIME_PERIOD
clean["source_employment_size"] = "Total, with employees"
clean["source_geography_level"] = "2021 census subdivisions aggregated to 2021 census divisions"
clean["source_file"] = safe_relative(RAW_CSV)
clean["commercial_density_definition"] = (
    "NAICS 41 Wholesale trade + NAICS 44-45 Retail trade + "
    "NAICS 72 Accommodation and food services"
)
clean["method_note"] = (
    "manufacturing_density maps MAESDEN92 to NAICS 31-33 Manufacturing business counts with employees "
    "from the Business Counts in Rural and Small Town Canada dashboard extract, aggregated from 2021 Québec "
    "census subdivisions to census divisions and divided by land_area_km2. commercial_density maps COMDEVDN92 "
    "to a constructed commercial business-count density using NAICS 41 Wholesale trade, NAICS 44-45 Retail trade, "
    "and NAICS 72 Accommodation and food services, aggregated from census subdivisions to census divisions and "
    "divided by land_area_km2. The selected time period is 2022-01 because it is the earliest available period "
    "in the inspected dashboard extract."
)


# -----------------------------
# Validation
# -----------------------------

if len(clean) != EXPECTED_QC_CD_COUNT:
    raise ValueError(f"Clean output has {len(clean)} rows; expected {EXPECTED_QC_CD_COUNT}.")

if clean["census_division_code"].duplicated().any():
    dupes = clean[clean["census_division_code"].duplicated(keep=False)]
    raise ValueError(
        "Duplicate census_division_code values in clean output:\n"
        + dupes[["census_division_code", "census_division_dguid", "census_division_name"]].to_string(index=False)
    )

required_numeric_cols = [
    "land_area_km2",
    "manufacturing_business_count",
    "wholesale_trade_business_count",
    "retail_trade_business_count",
    "accommodation_food_services_business_count",
    "commercial_trade_only_business_count",
    "commercial_trade_accommodation_food_business_count",
    "manufacturing_density_per_km2",
    "wholesale_trade_density_per_km2",
    "retail_trade_density_per_km2",
    "accommodation_food_services_density_per_km2",
    "commercial_trade_only_density_per_km2",
    "commercial_trade_accommodation_food_density_per_km2",
    "manufacturing_density",
    "commercial_density",
]

for col in required_numeric_cols:
    clean[col] = clean_numeric(clean[col])
    missing = int(clean[col].isna().sum())
    if missing != 0:
        raise ValueError(f"Unexpected missing values in {col}: {missing}")

nonnegative_cols = [col for col in required_numeric_cols if col != "land_area_km2"]

for col in nonnegative_cols:
    if (clean[col] < 0).any():
        raise ValueError(f"Negative values found in {col}.")

count_cols = [
    "manufacturing_business_count",
    "wholesale_trade_business_count",
    "retail_trade_business_count",
    "accommodation_food_services_business_count",
    "commercial_trade_only_business_count",
    "commercial_trade_accommodation_food_business_count",
]

for col in count_cols:
    max_fractional = ((clean[col] - clean[col].round()).abs()).max(skipna=True)
    if pd.notna(max_fractional) and max_fractional > 1e-9:
        raise ValueError(f"Non-integer-like business counts found in {col}: max fractional {max_fractional}")

manufacturing_formula_diff = (
    clean["manufacturing_density"]
    - clean["manufacturing_business_count"] / clean["land_area_km2"]
).abs().max(skipna=True)

wholesale_formula_diff = (
    clean["wholesale_trade_density_per_km2"]
    - clean["wholesale_trade_business_count"] / clean["land_area_km2"]
).abs().max(skipna=True)

retail_formula_diff = (
    clean["retail_trade_density_per_km2"]
    - clean["retail_trade_business_count"] / clean["land_area_km2"]
).abs().max(skipna=True)

accommodation_formula_diff = (
    clean["accommodation_food_services_density_per_km2"]
    - clean["accommodation_food_services_business_count"] / clean["land_area_km2"]
).abs().max(skipna=True)

commercial_trade_count_diff = (
    clean["commercial_trade_only_business_count"]
    - clean["wholesale_trade_business_count"]
    - clean["retail_trade_business_count"]
).abs().max(skipna=True)

commercial_broad_count_diff = (
    clean["commercial_trade_accommodation_food_business_count"]
    - clean["wholesale_trade_business_count"]
    - clean["retail_trade_business_count"]
    - clean["accommodation_food_services_business_count"]
).abs().max(skipna=True)

commercial_trade_density_diff = (
    clean["commercial_trade_only_density_per_km2"]
    - clean["commercial_trade_only_business_count"] / clean["land_area_km2"]
).abs().max(skipna=True)

commercial_density_formula_diff = (
    clean["commercial_density"]
    - clean["commercial_trade_accommodation_food_business_count"] / clean["land_area_km2"]
).abs().max(skipna=True)

commercial_alias_diff = (
    clean["commercial_density"]
    - clean["commercial_trade_accommodation_food_density_per_km2"]
).abs().max(skipna=True)

formula_diffs = {
    "manufacturing_formula_diff": manufacturing_formula_diff,
    "wholesale_formula_diff": wholesale_formula_diff,
    "retail_formula_diff": retail_formula_diff,
    "accommodation_formula_diff": accommodation_formula_diff,
    "commercial_trade_count_diff": commercial_trade_count_diff,
    "commercial_broad_count_diff": commercial_broad_count_diff,
    "commercial_trade_density_diff": commercial_trade_density_diff,
    "commercial_density_formula_diff": commercial_density_formula_diff,
    "commercial_alias_diff": commercial_alias_diff,
}

tolerance = 1e-12

for name, diff in formula_diffs.items():
    if pd.notna(diff) and diff > tolerance:
        raise ValueError(f"{name} exceeded tolerance {tolerance}: {diff}")

base_names_with_mojibake = contains_mojibake(base["census_division_name"])
clean_names_with_mojibake = contains_mojibake(clean["census_division_name"])
source_ref_area_with_mojibake = contains_mojibake(source_rows[ref_area_col])
source_industry_with_mojibake = contains_mojibake(source_rows[industry_col])

if source_ref_area_with_mojibake != 0 or source_industry_with_mojibake != 0:
    raise ValueError("Mojibake detected in selected source rows.")


# -----------------------------
# Component audit
# -----------------------------

component_audit = pd.DataFrame(component_audit_rows)

component_audit.to_csv(OUTPUT_COMPONENT_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Unmatched audit
# -----------------------------

unmatched = clean[
    [
        "census_division_code",
        "census_division_dguid",
        "census_division_name",
        "manufacturing_business_count",
        "wholesale_trade_business_count",
        "retail_trade_business_count",
        "accommodation_food_services_business_count",
        "commercial_trade_only_business_count",
        "commercial_trade_accommodation_food_business_count",
    ]
].copy()

missing_mask = unmatched[
    [
        "manufacturing_business_count",
        "wholesale_trade_business_count",
        "retail_trade_business_count",
        "accommodation_food_services_business_count",
        "commercial_trade_only_business_count",
        "commercial_trade_accommodation_food_business_count",
    ]
].isna().any(axis=1)

unmatched = unmatched[missing_mask].copy()
unmatched.to_csv(OUTPUT_UNMATCHED_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Metadata
# -----------------------------

metadata_rows = [
    {
        "variable": "manufacturing_density",
        "original_sovi_code": "MAESDEN92",
        "description": "Manufacturing business-count density",
        "source_dataset": "Business Counts in Rural and Small Town Canada dashboard extract",
        "source_file": safe_relative(RAW_CSV),
        "source_time_period": SELECTED_TIME_PERIOD,
        "source_geography": "2021 census subdivisions aggregated to 2021 census divisions",
        "source_employment_size": "Total, with employees",
        "source_industry_codes": "31-33",
        "source_industry_labels": "Manufacturing",
        "denominator": "land_area_km2 from cleaned census-division base frame",
        "unit": "businesses_with_employees_per_square_kilometre",
        "derivation": "manufacturing_business_count / land_area_km2",
        "coverage": "98/98",
        "status": "ready_full_coverage",
        "notes": "Canadian proxy for SoVI manufacturing establishments density. Uses earliest available dashboard period, 2022-01.",
    },
    {
        "variable": "commercial_density",
        "original_sovi_code": "COMDEVDN92",
        "description": "Broad commercial business-count density",
        "source_dataset": "Business Counts in Rural and Small Town Canada dashboard extract",
        "source_file": safe_relative(RAW_CSV),
        "source_time_period": SELECTED_TIME_PERIOD,
        "source_geography": "2021 census subdivisions aggregated to 2021 census divisions",
        "source_employment_size": "Total, with employees",
        "source_industry_codes": "41 + 44-45 + 72",
        "source_industry_labels": "Wholesale trade + Retail trade + Accommodation and food services",
        "denominator": "land_area_km2 from cleaned census-division base frame",
        "unit": "businesses_with_employees_per_square_kilometre",
        "derivation": "(wholesale_trade_business_count + retail_trade_business_count + accommodation_food_services_business_count) / land_area_km2",
        "coverage": "98/98",
        "status": "ready_full_coverage_constructed_proxy",
        "notes": "Constructed Canadian proxy for SoVI commercial establishments density. Trade-only density is retained as an audit variable.",
    },
    {
        "variable": "commercial_trade_only_density_per_km2",
        "original_sovi_code": "",
        "description": "Trade-only commercial business-count density audit variable",
        "source_dataset": "Business Counts in Rural and Small Town Canada dashboard extract",
        "source_file": safe_relative(RAW_CSV),
        "source_time_period": SELECTED_TIME_PERIOD,
        "source_geography": "2021 census subdivisions aggregated to 2021 census divisions",
        "source_employment_size": "Total, with employees",
        "source_industry_codes": "41 + 44-45",
        "source_industry_labels": "Wholesale trade + Retail trade",
        "denominator": "land_area_km2",
        "unit": "businesses_with_employees_per_square_kilometre",
        "derivation": "(wholesale_trade_business_count + retail_trade_business_count) / land_area_km2",
        "coverage": "98/98",
        "status": "audit_sensitivity_variable",
        "notes": "Retained to document a narrower commercial-density definition.",
    },
]

for component in COMPONENTS:
    metadata_rows.append(
        {
            "variable": f"{component}_business_count",
            "original_sovi_code": COMPONENTS[component]["original_sovi_code"],
            "description": f"{COMPONENTS[component]['industry_label']} business count with employees",
            "source_dataset": "Business Counts in Rural and Small Town Canada dashboard extract",
            "source_file": safe_relative(RAW_CSV),
            "source_time_period": SELECTED_TIME_PERIOD,
            "source_geography": "2021 census subdivisions aggregated to 2021 census divisions",
            "source_employment_size": "Total, with employees",
            "source_industry_codes": COMPONENTS[component]["industry_code"],
            "source_industry_labels": COMPONENTS[component]["industry_label"],
            "denominator": "",
            "unit": "businesses_with_employees",
            "derivation": "Sum of CSD OBS_VALUE within each census division",
            "coverage": "98/98",
            "status": "component_audit_variable",
            "notes": "Retained as a component used to construct business-density variables.",
        }
    )

metadata = pd.DataFrame(metadata_rows)
metadata.to_csv(OUTPUT_METADATA, index=False, encoding="utf-8")


# -----------------------------
# Summary
# -----------------------------

summary_rows = [
    {"metric": "raw_csv", "value": safe_relative(RAW_CSV)},
    {"metric": "base_cd_frame", "value": safe_relative(BASE_CD_FRAME)},
    {"metric": "raw_encoding", "value": raw_encoding},
    {"metric": "base_encoding", "value": base_encoding},
    {"metric": "raw_rows_scanned", "value": raw_rows_scanned},
    {"metric": "raw_columns", "value": len(columns)},
    {"metric": "base_rows", "value": len(base)},
    {"metric": "clean_rows", "value": len(clean)},
    {"metric": "unique_census_divisions", "value": clean["census_division_code"].nunique()},
    {"metric": "selected_time_period", "value": SELECTED_TIME_PERIOD},
    {"metric": "available_time_periods", "value": " | ".join(available_time_periods)},
    {"metric": "qc_rows_seen", "value": qc_rows_seen},
    {"metric": "qc_total_with_employees_rows_seen", "value": qc_total_with_employees_rows_seen},
    {"metric": "selected_total_with_employees_rows_seen", "value": selected_total_with_employees_rows_seen},
    {"metric": "selected_unique_quebec_csd_codes", "value": len(selected_total_csd_codes)},
    {"metric": "selected_unique_quebec_cd_codes", "value": len(selected_total_cd_codes)},
    {"metric": "source_rows_selected_target_components", "value": len(source_rows)},
    {"metric": "variables_cleaned", "value": "manufacturing_density, commercial_density"},
    {"metric": "manufacturing_source_industry", "value": "31-33 Manufacturing"},
    {"metric": "commercial_source_industries", "value": "41 Wholesale trade + 44-45 Retail trade + 72 Accommodation and food services"},
    {"metric": "commercial_trade_only_audit_industries", "value": "41 Wholesale trade + 44-45 Retail trade"},
    {"metric": "all_required_numeric_columns_complete", "value": bool(clean[required_numeric_cols].notna().all().all())},
    {"metric": "manufacturing_formula_max_abs_difference", "value": manufacturing_formula_diff},
    {"metric": "wholesale_formula_max_abs_difference", "value": wholesale_formula_diff},
    {"metric": "retail_formula_max_abs_difference", "value": retail_formula_diff},
    {"metric": "accommodation_formula_max_abs_difference", "value": accommodation_formula_diff},
    {"metric": "commercial_trade_count_max_abs_difference", "value": commercial_trade_count_diff},
    {"metric": "commercial_broad_count_max_abs_difference", "value": commercial_broad_count_diff},
    {"metric": "commercial_trade_density_max_abs_difference", "value": commercial_trade_density_diff},
    {"metric": "commercial_density_formula_max_abs_difference", "value": commercial_density_formula_diff},
    {"metric": "commercial_alias_max_abs_difference", "value": commercial_alias_diff},
    {"metric": "base_names_with_mojibake", "value": base_names_with_mojibake},
    {"metric": "clean_names_with_mojibake", "value": clean_names_with_mojibake},
    {"metric": "source_ref_area_with_mojibake", "value": source_ref_area_with_mojibake},
    {"metric": "source_industry_with_mojibake", "value": source_industry_with_mojibake},
]

for _, row in component_audit.iterrows():
    component = row["component"]
    summary_rows.append({"metric": f"{component}_source_rows", "value": row["source_rows"]})
    summary_rows.append({"metric": f"{component}_unique_csd_rows", "value": row["unique_csd_rows"]})
    summary_rows.append({"metric": f"{component}_unique_cd_rows_before_join", "value": row["unique_cd_rows_before_join"]})
    summary_rows.append({"metric": f"{component}_coverage_is_98_cds", "value": row["coverage_is_98_cds"]})
    summary_rows.append({"metric": f"{component}_cd_rows_non_missing", "value": row["cd_rows_non_missing"]})
    summary_rows.append({"metric": f"{component}_cd_rows_missing", "value": row["cd_rows_missing"]})

for variable in required_numeric_cols:
    add_numeric_summary(summary_rows, clean, variable)

summary_rows.append(
    {
        "metric": "recommended_next_step",
        "value": (
            "If this summary shows 98 rows, full numeric coverage, zero formula differences, and no mojibake, "
            "generate the README and add SoVI YAML mappings for MAESDEN92 -> manufacturing_density and "
            "COMDEVDN92 -> commercial_density."
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
        "manufacturing_density",
        "commercial_density",
        "manufacturing_business_count",
        "wholesale_trade_business_count",
        "retail_trade_business_count",
        "accommodation_food_services_business_count",
        "commercial_trade_only_business_count",
        "commercial_trade_accommodation_food_business_count",
        "manufacturing_density_per_km2",
        "wholesale_trade_density_per_km2",
        "retail_trade_density_per_km2",
        "accommodation_food_services_density_per_km2",
        "commercial_trade_only_density_per_km2",
        "commercial_trade_accommodation_food_density_per_km2",
        "source_time_period",
        "source_employment_size",
        "source_geography_level",
        "commercial_density_definition",
        "source_file",
        "method_note",
    ]
)

ordered_cols = [col for col in ordered_cols if col in clean.columns]
clean = clean[ordered_cols].copy()

clean.to_csv(OUTPUT_CLEAN, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CLEAN CENSUS DIVISION BUSINESS ESTABLISHMENTS 2021")
print("=" * 72)

print("\nInputs:")
print("Raw CSV:", safe_relative(RAW_CSV))
print("Raw encoding:", raw_encoding)
print("Base frame:", safe_relative(BASE_CD_FRAME))
print("Base encoding:", base_encoding)

print("\nClean output:")
print("Rows:", len(clean))
print("Unique census divisions:", clean["census_division_code"].nunique())
print("Variables cleaned: manufacturing_density, commercial_density")
print("Selected time period:", SELECTED_TIME_PERIOD)

print("\nCoverage:")
print("Selected unique Québec CSD codes:", len(selected_total_csd_codes))
print("Selected unique Québec CD codes:", len(selected_total_cd_codes))
print("Base CD rows:", len(base))

print("\nFormula checks:")
for name, diff in formula_diffs.items():
    print(f"{name}: {diff}")

print("\nMojibake check:")
print("Base names with mojibake:", base_names_with_mojibake)
print("Clean names with mojibake:", clean_names_with_mojibake)
print("Source REF_AREA with mojibake:", source_ref_area_with_mojibake)
print("Source INDUSTRY with mojibake:", source_industry_with_mojibake)

print("\nComponent audit:")
print(component_audit.to_string(index=False))

print("\nMain summaries:")
for variable in [
    "manufacturing_density",
    "commercial_density",
    "manufacturing_business_count",
    "commercial_trade_accommodation_food_business_count",
    "commercial_trade_only_density_per_km2",
]:
    print("\n", variable)
    print(clean[variable].describe())

print("\nPreview:")
preview_cols = [
    "census_division_code",
    "census_division_name",
    "manufacturing_density",
    "commercial_density",
    "manufacturing_business_count",
    "commercial_trade_accommodation_food_business_count",
    "source_time_period",
]
preview_cols = [col for col in preview_cols if col in clean.columns]
print(clean[preview_cols].head(20).to_string(index=False))

print("\nSaved:")
print(OUTPUT_CLEAN)
print(OUTPUT_SOURCE_ROWS)
print(OUTPUT_METADATA)
print(OUTPUT_COMPONENT_AUDIT)
print(OUTPUT_UNMATCHED_AUDIT)
print(OUTPUT_TIME_PERIOD_AUDIT)
print(OUTPUT_SUMMARY)

print("\nDone.")