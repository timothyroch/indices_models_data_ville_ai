from pathlib import Path
import codecs
import re
import pandas as pd


# ============================================================
# Inspect Census Division Business Establishments 2021
# Dashboard Dataset Version
# ============================================================
#
# Purpose:
#   Inspect whether the Business Counts in Rural and Small Town Canada
#   dashboard extract can support:
#
#       MAESDEN92   -> manufacturing_density
#       COMDEVDN92  -> commercial_density
#
# Source file expected:
#
#   census_division_business_establishments_2021/raw/
#       canada_rural_business_counts_dashboard.csv
#
# Source structure observed:
#
#   DATAFLOW
#   FREQ
#   REF_AREA
#   EMP_SIZE
#   INDUSTRY
#   TIME_PERIOD
#   OBS_VALUE
#   VECTOR
#   DGUID
#   DECIMALS
#   UNIT_MULT
#   UNIT_MEASURE
#   CONF_STATUS
#   OBS_STATUS
#   RURAL_FLAG
#   PROV_TERR
#
# Method:
#   1. Filter to Québec rows.
#   2. Filter to total employment size: "_T: Total, with employees".
#   3. Parse CSD SGC codes from DGUID or REF_AREA.
#   4. Aggregate CSD business counts to census divisions.
#   5. Divide by census-division land_area_km2.
#
# Candidate formulas:
#
#   manufacturing_density =
#       NAICS 31-33 Manufacturing business count / land_area_km2
#
#   commercial_density candidates:
#       41 Wholesale trade + 44-45 Retail trade
#       41 Wholesale trade + 44-45 Retail trade + 72 Accommodation and food services
#
# This is inspection-only. It does not clean final variables.
#
# Run from data/:
#
#   python census_division_business_establishments_2021/inspect_census_division_business_establishments_2021.py
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

OUTPUT_SUMMARY = OUTPUT_DIR / "business_establishments_inspection_summary_2021.csv"
OUTPUT_RAW_COLUMN_PROFILE = OUTPUT_DIR / "business_establishments_raw_column_profile_2021.csv"
OUTPUT_DIMENSION_INVENTORY = OUTPUT_DIR / "business_establishments_dimension_inventory_2021.csv"
OUTPUT_GEOGRAPHY_AUDIT = OUTPUT_DIR / "business_establishments_geography_audit_2021.csv"
OUTPUT_TIME_PERIOD_AUDIT = OUTPUT_DIR / "business_establishments_time_period_audit_2021.csv"
OUTPUT_INDUSTRY_INVENTORY = OUTPUT_DIR / "business_establishments_industry_inventory_2021.csv"
OUTPUT_CANDIDATE_SOURCE_ROWS = OUTPUT_DIR / "business_establishments_candidate_source_rows_2021.csv"
OUTPUT_FORMULA_AUDIT = OUTPUT_DIR / "business_establishments_formula_audit_2021.csv"
OUTPUT_TARGET_SUMMARY = OUTPUT_DIR / "business_establishments_target_summary_2021.csv"
OUTPUT_UNMATCHED_AUDIT = OUTPUT_DIR / "business_establishments_unmatched_cd_audit_2021.csv"


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

CHUNK_SIZE = 250_000

# The dashboard extract appears to use semester periods such as 2022-01,
# 2022-07, 2023-01, etc. For a 2021 SoVI table, choose the closest available
# period after the 2021 Census reference, but the inspection will audit all periods.
PREFERRED_PERIOD_ORDER = [
    "2021-01",
    "2021-07",
    "2022-01",
    "2022-07",
    "2023-01",
    "2023-07",
    "2024-01",
    "2024-07",
    "2025-01",
    "2025-07",
]

TARGET_COMPONENTS = {
    "manufacturing": {
        "industry_code": "31-33",
        "industry_name_contains": "manufacturing",
        "original_sovi_code": "MAESDEN92",
        "target_variable": "manufacturing_density",
        "interpretation": "NAICS 31-33 Manufacturing business counts with employees.",
    },
    "wholesale_trade": {
        "industry_code": "41",
        "industry_name_contains": "wholesale trade",
        "original_sovi_code": "COMDEVDN92",
        "target_variable": "commercial_density",
        "interpretation": "NAICS 41 Wholesale trade business counts with employees.",
    },
    "retail_trade": {
        "industry_code": "44-45",
        "industry_name_contains": "retail trade",
        "original_sovi_code": "COMDEVDN92",
        "target_variable": "commercial_density",
        "interpretation": "NAICS 44-45 Retail trade business counts with employees.",
    },
    "accommodation_food_services": {
        "industry_code": "72",
        "industry_name_contains": "accommodation and food services",
        "original_sovi_code": "COMDEVDN92",
        "target_variable": "commercial_density",
        "interpretation": "NAICS 72 Accommodation and food services business counts with employees.",
    },
}


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
    """
    Parses values like:
      '24: Quebec' -> '24'
      '31-33: Manufacturing' -> '31-33'
      '3542029: Hanover' -> '3542029'
      '_T: Total, with employees' -> '_T'
    """
    text = normalize_text(value)
    if ":" in text:
        return text.split(":", 1)[0].strip()
    return text.strip()


def parse_prefixed_label(value: object) -> str:
    text = normalize_text(value)
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text.strip()


def parse_industry_code(value: object) -> str:
    text = normalize_text(value)

    if ":" in text:
        return text.split(":", 1)[0].strip()

    bracket = re.search(r"\[(.*?)\]", text)
    if bracket:
        return bracket.group(1).strip()

    match = re.match(r"^\s*([A-Z_]+|\d{2}(?:-\d{2})?|\d{3,4})\b", text)
    if match:
        return match.group(1).strip()

    return ""


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

    # 2021A00052453065 -> 2453065
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


def period_rank(period: str) -> int:
    if period in PREFERRED_PERIOD_ORDER:
        return PREFERRED_PERIOD_ORDER.index(period)
    return 10_000


def select_preferred_period(periods: list[str]) -> str:
    periods = sorted([p for p in periods if p], key=lambda p: (period_rank(p), p))
    return periods[0] if periods else ""


def summarize_numeric(series: pd.Series) -> dict:
    values = clean_numeric(series)
    return {
        "non_missing": int(values.notna().sum()),
        "missing": int(values.isna().sum()),
        "min": values.min(skipna=True),
        "max": values.max(skipna=True),
        "mean": values.mean(skipna=True),
        "median": values.median(skipna=True),
    }


def make_raw_column_profile(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in sample.columns:
        numeric = clean_numeric(sample[col])
        rows.append(
            {
                "column": col,
                "dtype_as_loaded": str(sample[col].dtype),
                "sample_non_missing": int(sample[col].notna().sum()),
                "sample_missing": int(sample[col].isna().sum()),
                "sample_unique_values": int(sample[col].astype("string").nunique(dropna=True)),
                "sample_values": " | ".join(sample[col].dropna().astype(str).drop_duplicates().head(15).tolist()),
                "sample_numeric_non_missing": int(numeric.notna().sum()),
                "sample_numeric_min": numeric.min(skipna=True),
                "sample_numeric_max": numeric.max(skipna=True),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# Validate inputs
# -----------------------------

if not RAW_CSV.exists():
    raise FileNotFoundError(f"Missing raw dashboard CSV:\n{RAW_CSV}")

if not BASE_CD_FRAME.exists():
    raise FileNotFoundError(f"Missing base CD frame:\n{BASE_CD_FRAME}")


# -----------------------------
# Load base CD frame
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
# Read header/sample and detect columns
# -----------------------------

raw_encoding = detect_file_encoding_strict(RAW_CSV, ENCODING_CANDIDATES)

sample = pd.read_csv(RAW_CSV, encoding=raw_encoding, dtype=str, nrows=50_000, low_memory=False)
sample.columns = [str(col).strip() for col in sample.columns]
columns = list(sample.columns)

dataflow_col = find_column(columns, ["DATAFLOW"])
freq_col = find_column(columns, ["FREQ"])
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

raw_column_profile = make_raw_column_profile(sample)
raw_column_profile.to_csv(OUTPUT_RAW_COLUMN_PROFILE, index=False, encoding="utf-8")

dimension_rows = []
for col in sample.columns:
    values = sample[col].dropna().astype(str)
    dimension_rows.append(
        {
            "column": col,
            "sample_unique_values": int(values.nunique(dropna=True)),
            "sample_values": " | ".join(values.drop_duplicates().head(25).tolist()),
            "sample_contains_quebec": values.str.contains("Quebec|Québec", case=False, regex=True, na=False).any(),
            "sample_contains_total_with_employees": values.str.contains("Total, with employees", case=False, regex=False, na=False).any(),
            "sample_contains_manufacturing": values.str.contains("Manufacturing", case=False, regex=False, na=False).any(),
            "sample_contains_retail": values.str.contains("Retail trade", case=False, regex=False, na=False).any(),
            "sample_contains_wholesale": values.str.contains("Wholesale trade", case=False, regex=False, na=False).any(),
            "sample_contains_accommodation_food": values.str.contains("Accommodation and food services", case=False, regex=False, na=False).any(),
        }
    )

dimension_inventory = pd.DataFrame(dimension_rows)
dimension_inventory.to_csv(OUTPUT_DIMENSION_INVENTORY, index=False, encoding="utf-8")


# -----------------------------
# Chunked extraction
# -----------------------------

needed_cols = [
    col for col in [
        dataflow_col,
        freq_col,
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

candidate_chunks = []
qc_total_chunks = []
period_counts = []
industry_inventory_rows = []
geography_rows = []

raw_rows = 0

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
    raw_rows += len(chunk)

    chunk[value_col] = clean_numeric(chunk[value_col])

    chunk["_province_code"] = chunk[prov_col].map(parse_prefixed_code)
    chunk["_is_quebec"] = chunk["_province_code"] == "24"

    # Québec rows only.
    qc = chunk[chunk["_is_quebec"]].copy()
    if qc.empty:
        continue

    qc["_emp_size_code"] = qc[emp_size_col].map(parse_prefixed_code)
    qc["_emp_size_label"] = qc[emp_size_col].map(parse_prefixed_label)
    qc["_is_total_with_employees"] = (
        (qc["_emp_size_code"] == "_T")
        | qc["_emp_size_label"].str.lower().eq("total, with employees")
    )

    qc["_csd_sgc_code_from_dguid"] = qc[dguid_col].map(extract_csd_sgc_from_dguid)
    qc["_csd_sgc_code_from_ref_area"] = qc[ref_area_col].map(extract_csd_sgc_from_ref_area)
    qc["_csd_sgc_code"] = qc["_csd_sgc_code_from_dguid"]
    qc.loc[qc["_csd_sgc_code"] == "", "_csd_sgc_code"] = qc.loc[
        qc["_csd_sgc_code"] == "", "_csd_sgc_code_from_ref_area"
    ]
    qc["_is_quebec_csd"] = qc["_csd_sgc_code"].astype(str).str.fullmatch(r"24\d{5}", na=False)
    qc["_cd_code"] = qc["_csd_sgc_code"].map(extract_cd_code_from_csd_sgc)

    geography_rows.append(
        {
            "chunk_rows": len(chunk),
            "quebec_rows": len(qc),
            "quebec_total_with_employees_rows": int(qc["_is_total_with_employees"].sum()),
            "quebec_csd_rows": int(qc["_is_quebec_csd"].sum()),
            "unique_quebec_csd_codes": qc.loc[qc["_is_quebec_csd"], "_csd_sgc_code"].nunique(),
            "unique_quebec_cd_codes": qc.loc[qc["_is_quebec_csd"], "_cd_code"].nunique(),
        }
    )

    qc_total = qc[qc["_is_total_with_employees"] & qc["_is_quebec_csd"]].copy()
    if qc_total.empty:
        continue

    qc_total["_industry_code"] = qc_total[industry_col].map(parse_industry_code)
    qc_total["_industry_label"] = qc_total[industry_col].map(parse_prefixed_label)
    qc_total["_component_alias"] = qc_total[industry_col].map(parse_component_alias)

    qc_total_chunks.append(
        qc_total[
            [
                ref_area_col,
                dguid_col,
                "_csd_sgc_code",
                "_cd_code",
                emp_size_col,
                industry_col,
                "_industry_code",
                "_industry_label",
                "_component_alias",
                time_col,
                value_col,
                prov_col,
            ]
            + ([rural_flag_col] if rural_flag_col else [])
            + ([conf_col] if conf_col else [])
            + ([obs_status_col] if obs_status_col else [])
            + ([unit_col] if unit_col else [])
        ].copy()
    )

    target_rows = qc_total[qc_total["_component_alias"] != ""].copy()
    if not target_rows.empty:
        candidate_chunks.append(target_rows.copy())

    period_counts.append(
        qc_total.groupby(time_col, dropna=False).size().reset_index(name="n_rows")
    )

    inv = (
        qc_total.groupby([industry_col, "_industry_code", "_industry_label", "_component_alias"], dropna=False)
        .agg(
            n_rows=(value_col, "size"),
            value_non_missing=(value_col, lambda s: int(clean_numeric(s).notna().sum())),
            unique_csd_codes=("_csd_sgc_code", "nunique"),
            unique_cd_codes=("_cd_code", "nunique"),
        )
        .reset_index()
    )
    industry_inventory_rows.append(inv)


# -----------------------------
# Combine extracted rows
# -----------------------------

if qc_total_chunks:
    qc_total_all = pd.concat(qc_total_chunks, ignore_index=True, sort=False)
else:
    qc_total_all = pd.DataFrame()

if candidate_chunks:
    candidate_source_rows = pd.concat(candidate_chunks, ignore_index=True, sort=False)
else:
    candidate_source_rows = pd.DataFrame()

candidate_source_rows.to_csv(OUTPUT_CANDIDATE_SOURCE_ROWS, index=False, encoding="utf-8")

if period_counts:
    time_period_audit = (
        pd.concat(period_counts, ignore_index=True)
        .groupby(time_col, dropna=False)["n_rows"]
        .sum()
        .reset_index()
        .rename(columns={time_col: "time_period"})
        .sort_values("time_period")
    )
else:
    time_period_audit = pd.DataFrame(columns=["time_period", "n_rows"])

time_period_audit.to_csv(OUTPUT_TIME_PERIOD_AUDIT, index=False, encoding="utf-8")

available_periods = time_period_audit["time_period"].dropna().astype(str).tolist()
selected_time_period = select_preferred_period(available_periods)

if industry_inventory_rows:
    industry_inventory = pd.concat(industry_inventory_rows, ignore_index=True, sort=False)
    industry_inventory = (
        industry_inventory
        .groupby([industry_col, "_industry_code", "_industry_label", "_component_alias"], dropna=False)
        .agg(
            n_rows=("n_rows", "sum"),
            value_non_missing=("value_non_missing", "sum"),
            unique_csd_codes=("unique_csd_codes", "max"),
            unique_cd_codes=("unique_cd_codes", "max"),
        )
        .reset_index()
        .rename(
            columns={
                industry_col: "industry_raw",
                "_industry_code": "industry_code",
                "_industry_label": "industry_label",
                "_component_alias": "component_alias",
            }
        )
        .sort_values(["component_alias", "industry_code", "industry_label"], na_position="last")
    )
else:
    industry_inventory = pd.DataFrame()

industry_inventory.to_csv(OUTPUT_INDUSTRY_INVENTORY, index=False, encoding="utf-8")


# -----------------------------
# Geography audit
# -----------------------------

if not qc_total_all.empty:
    cd_codes_from_table = set(qc_total_all["_cd_code"].dropna().astype(str))
    csd_codes_from_table = set(qc_total_all["_csd_sgc_code"].dropna().astype(str))
else:
    cd_codes_from_table = set()
    csd_codes_from_table = set()

geography_chunk_audit = pd.DataFrame(geography_rows)

geography_summary_rows = [
    {"metric": "raw_rows_scanned", "value": raw_rows},
    {"metric": "qc_total_with_employees_rows_extracted", "value": len(qc_total_all)},
    {"metric": "target_candidate_rows_extracted", "value": len(candidate_source_rows)},
    {"metric": "unique_quebec_csd_codes_total_with_employees", "value": len(csd_codes_from_table)},
    {"metric": "unique_quebec_cd_codes_total_with_employees", "value": len(cd_codes_from_table)},
    {"metric": "base_cd_rows", "value": len(base)},
    {"metric": "base_cd_codes_missing_from_dashboard", "value": " | ".join(sorted(base_cd_codes - cd_codes_from_table))},
    {"metric": "dashboard_cd_codes_not_in_base", "value": " | ".join(sorted(cd_codes_from_table - base_cd_codes))},
]

geography_audit = pd.concat(
    [
        pd.DataFrame(geography_summary_rows),
        pd.DataFrame([{"metric": "chunk_level_audit_rows", "value": len(geography_chunk_audit)}]),
    ],
    ignore_index=True,
)

geography_audit.to_csv(OUTPUT_GEOGRAPHY_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Formula audit helpers
# -----------------------------

def aggregate_component_for_period(
    data: pd.DataFrame,
    component: str,
    period: str,
) -> tuple[pd.DataFrame, dict]:
    subset = data[
        (data["_component_alias"] == component)
        & (data[time_col].astype(str) == str(period))
    ].copy()

    duplicate_component_rows = 0
    if not subset.empty:
        duplicate_component_rows = int(
            subset.duplicated(subset=["_csd_sgc_code", time_col, "_component_alias"]).sum()
        )

    aggregated = (
        subset.groupby("_cd_code", dropna=False)[value_col]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"_cd_code": "census_division_code", value_col: f"{component}_business_count"})
    )

    joined = base[
        ["census_division_code", "census_division_dguid", "census_division_name", "land_area_km2"]
    ].merge(
        aggregated,
        on="census_division_code",
        how="left",
        validate="one_to_one",
    )

    joined[f"{component}_density_per_km2"] = joined[f"{component}_business_count"] / joined["land_area_km2"]

    count_values = clean_numeric(joined[f"{component}_business_count"])
    density_values = clean_numeric(joined[f"{component}_density_per_km2"])

    audit = {
        "time_period": period,
        "candidate_variable": f"{component}_density_per_km2",
        "component": component,
        "components": component,
        "candidate_type": "single_naics_sector_density",
        "formula": f"{component}_business_count / land_area_km2",
        "source_rows": len(subset),
        "unique_csd_rows": int(subset["_csd_sgc_code"].nunique()) if not subset.empty else 0,
        "unique_cd_rows_before_join": int(subset["_cd_code"].nunique()) if not subset.empty else 0,
        "duplicate_component_rows_before_cd_aggregation": duplicate_component_rows,
        "non_missing": int(density_values.notna().sum()),
        "missing": int(density_values.isna().sum()),
        "coverage_is_98_cds": int(density_values.notna().sum()) == EXPECTED_QC_CD_COUNT,
        "business_count_min": count_values.min(skipna=True),
        "business_count_max": count_values.max(skipna=True),
        "business_count_mean": count_values.mean(skipna=True),
        "business_count_median": count_values.median(skipna=True),
        "density_per_km2_min": density_values.min(skipna=True),
        "density_per_km2_max": density_values.max(skipna=True),
        "density_per_km2_mean": density_values.mean(skipna=True),
        "density_per_km2_median": density_values.median(skipna=True),
    }

    return joined, audit


def combine_components_for_period(
    component_tables: dict[str, pd.DataFrame],
    components: list[str],
    alias: str,
    period: str,
) -> tuple[pd.DataFrame, dict]:
    combined = base[
        ["census_division_code", "census_division_dguid", "census_division_name", "land_area_km2"]
    ].copy()

    component_count_cols = []

    for component in components:
        table = component_tables[component][["census_division_code", f"{component}_business_count"]].copy()
        combined = combined.merge(table, on="census_division_code", how="left", validate="one_to_one")
        component_count_cols.append(f"{component}_business_count")

    combined[f"{alias}_business_count"] = combined[component_count_cols].sum(axis=1, min_count=len(component_count_cols))
    combined[f"{alias}_density_per_km2"] = combined[f"{alias}_business_count"] / combined["land_area_km2"]

    count_values = clean_numeric(combined[f"{alias}_business_count"])
    density_values = clean_numeric(combined[f"{alias}_density_per_km2"])

    audit = {
        "time_period": period,
        "candidate_variable": f"{alias}_density_per_km2",
        "component": "",
        "components": " + ".join(components),
        "candidate_type": "combined_naics_sector_density",
        "formula": f"({' + '.join(components)}) / land_area_km2",
        "source_rows": "",
        "unique_csd_rows": "",
        "unique_cd_rows_before_join": "",
        "duplicate_component_rows_before_cd_aggregation": "",
        "non_missing": int(density_values.notna().sum()),
        "missing": int(density_values.isna().sum()),
        "coverage_is_98_cds": int(density_values.notna().sum()) == EXPECTED_QC_CD_COUNT,
        "business_count_min": count_values.min(skipna=True),
        "business_count_max": count_values.max(skipna=True),
        "business_count_mean": count_values.mean(skipna=True),
        "business_count_median": count_values.median(skipna=True),
        "density_per_km2_min": density_values.min(skipna=True),
        "density_per_km2_max": density_values.max(skipna=True),
        "density_per_km2_mean": density_values.mean(skipna=True),
        "density_per_km2_median": density_values.median(skipna=True),
    }

    return combined, audit


# -----------------------------
# Formula audit across periods
# -----------------------------

formula_rows = []
period_component_tables = {}

for period in available_periods:
    component_tables = {}

    for component in TARGET_COMPONENTS:
        table, audit = aggregate_component_for_period(candidate_source_rows, component, period)
        component_tables[component] = table

        formula_rows.append(
            {
                "original_sovi_code": TARGET_COMPONENTS[component]["original_sovi_code"],
                "target_variable": TARGET_COMPONENTS[component]["target_variable"],
                "interpretation": TARGET_COMPONENTS[component]["interpretation"],
                "recommended_default_without_review": component == "manufacturing",
                **audit,
            }
        )

    commercial_trade_only, audit_trade_only = combine_components_for_period(
        component_tables=component_tables,
        components=["wholesale_trade", "retail_trade"],
        alias="commercial_trade_only",
        period=period,
    )

    formula_rows.append(
        {
            "original_sovi_code": "COMDEVDN92",
            "target_variable": "commercial_density",
            "interpretation": "Commercial density candidate using wholesale and retail trade business counts with employees.",
            "recommended_default_without_review": False,
            **audit_trade_only,
        }
    )

    commercial_trade_food, audit_trade_food = combine_components_for_period(
        component_tables=component_tables,
        components=["wholesale_trade", "retail_trade", "accommodation_food_services"],
        alias="commercial_trade_accommodation_food",
        period=period,
    )

    formula_rows.append(
        {
            "original_sovi_code": "COMDEVDN92",
            "target_variable": "commercial_density",
            "interpretation": "Broader commercial density candidate using wholesale, retail, accommodation, and food services business counts with employees.",
            "recommended_default_without_review": False,
            **audit_trade_food,
        }
    )

    period_component_tables[period] = {
        **component_tables,
        "commercial_trade_only": commercial_trade_only,
        "commercial_trade_accommodation_food": commercial_trade_food,
    }

formula_audit = pd.DataFrame(formula_rows)

if not formula_audit.empty:
    formula_audit["_preferred_period_rank"] = formula_audit["time_period"].map(period_rank)
    formula_audit = formula_audit.sort_values(
        ["_preferred_period_rank", "candidate_variable"]
    ).drop(columns=["_preferred_period_rank"])

formula_audit.to_csv(OUTPUT_FORMULA_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Target summary
# -----------------------------

def best_candidate_for_variable(candidate_variable: str) -> pd.Series | None:
    if formula_audit.empty:
        return None

    subset = formula_audit[
        (formula_audit["candidate_variable"] == candidate_variable)
        & (formula_audit["coverage_is_98_cds"] == True)
    ].copy()

    if subset.empty:
        return None

    subset["_period_rank"] = subset["time_period"].map(period_rank)
    subset = subset.sort_values(["_period_rank", "time_period"])
    return subset.iloc[0]


manufacturing_best = best_candidate_for_variable("manufacturing_density_per_km2")
commercial_trade_best = best_candidate_for_variable("commercial_trade_only_density_per_km2")
commercial_broad_best = best_candidate_for_variable("commercial_trade_accommodation_food_density_per_km2")

manufacturing_ready = manufacturing_best is not None
commercial_trade_ready = commercial_trade_best is not None
commercial_broad_ready = commercial_broad_best is not None

commercial_best = commercial_broad_best if commercial_broad_ready else commercial_trade_best
commercial_ready = commercial_best is not None

target_summary_rows = [
    {
        "canonical_variable": "manufacturing_density",
        "original_sovi_code": "MAESDEN92",
        "candidate_found": manufacturing_ready,
        "best_candidate_variable": "manufacturing_density_per_km2" if manufacturing_ready else "",
        "best_time_period": manufacturing_best["time_period"] if manufacturing_ready else "",
        "candidate_formula": "manufacturing_business_count / land_area_km2" if manufacturing_ready else "",
        "coverage_is_98_cds": manufacturing_ready,
        "status": "candidate_available_needs_review" if manufacturing_ready else "candidate_not_ready_or_missing",
        "interpretation": (
            "Manufacturing business-count density from NAICS 31-33, aggregated from Québec CSDs to CDs."
            if manufacturing_ready
            else "No full-coverage manufacturing sector candidate was confirmed."
        ),
    },
    {
        "canonical_variable": "commercial_density",
        "original_sovi_code": "COMDEVDN92",
        "candidate_found": commercial_ready,
        "best_candidate_variable": commercial_best["candidate_variable"] if commercial_ready else "",
        "best_time_period": commercial_best["time_period"] if commercial_ready else "",
        "candidate_formula": commercial_best["formula"] if commercial_ready else "",
        "coverage_is_98_cds": commercial_ready,
        "status": (
            "candidate_available_needs_conceptual_choice"
            if commercial_ready
            else "candidate_not_ready_or_missing"
        ),
        "interpretation": (
            "Commercial density is a constructed NAICS proxy. Review whether final mapping should use trade only or trade plus accommodation/food services."
            if commercial_ready
            else "No full-coverage commercial-sector combination was confirmed."
        ),
    },
]

target_summary = pd.DataFrame(target_summary_rows)
target_summary.to_csv(OUTPUT_TARGET_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Unmatched audit for selected / best period
# -----------------------------

unmatched = base[["census_division_code", "census_division_dguid", "census_division_name"]].copy()

selected_for_unmatched = selected_time_period
if selected_for_unmatched and selected_for_unmatched in period_component_tables:
    for name, table in period_component_tables[selected_for_unmatched].items():
        count_cols = [col for col in table.columns if col.endswith("_business_count")]
        for col in count_cols:
            unmatched = unmatched.merge(
                table[["census_division_code", col]],
                on="census_division_code",
                how="left",
                validate="one_to_one",
            )

    count_cols = [col for col in unmatched.columns if col.endswith("_business_count")]
    if count_cols:
        missing_mask = unmatched[count_cols].isna().any(axis=1)
        unmatched = unmatched[missing_mask].copy()

unmatched.to_csv(OUTPUT_UNMATCHED_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Summary
# -----------------------------

raw_names_with_mojibake = 0
if not sample.empty:
    raw_names_with_mojibake += contains_mojibake(sample[ref_area_col])
    raw_names_with_mojibake += contains_mojibake(sample[industry_col])

base_names_with_mojibake = contains_mojibake(base["census_division_name"])

summary_rows = [
    {"metric": "raw_csv", "value": safe_relative(RAW_CSV)},
    {"metric": "base_cd_frame", "value": safe_relative(BASE_CD_FRAME)},
    {"metric": "raw_encoding", "value": raw_encoding},
    {"metric": "base_encoding", "value": base_encoding},
    {"metric": "raw_rows_scanned", "value": raw_rows},
    {"metric": "raw_columns", "value": len(columns)},
    {"metric": "base_rows", "value": len(base)},
    {"metric": "detected_ref_area_column", "value": ref_area_col},
    {"metric": "detected_emp_size_column", "value": emp_size_col},
    {"metric": "detected_industry_column", "value": industry_col},
    {"metric": "detected_time_period_column", "value": time_col},
    {"metric": "detected_value_column", "value": value_col},
    {"metric": "detected_dguid_column", "value": dguid_col},
    {"metric": "detected_unit_column", "value": unit_col or ""},
    {"metric": "detected_conf_status_column", "value": conf_col or ""},
    {"metric": "detected_obs_status_column", "value": obs_status_col or ""},
    {"metric": "detected_rural_flag_column", "value": rural_flag_col or ""},
    {"metric": "detected_province_column", "value": prov_col},
    {"metric": "available_time_periods", "value": " | ".join(available_periods)},
    {"metric": "selected_time_period_for_review", "value": selected_time_period},
    {"metric": "qc_total_with_employees_rows_extracted", "value": len(qc_total_all)},
    {"metric": "candidate_source_rows", "value": len(candidate_source_rows)},
    {"metric": "unique_quebec_csd_codes_total_with_employees", "value": len(csd_codes_from_table)},
    {"metric": "unique_quebec_cd_codes_total_with_employees", "value": len(cd_codes_from_table)},
    {"metric": "manufacturing_candidate_ready", "value": manufacturing_ready},
    {"metric": "manufacturing_best_time_period", "value": manufacturing_best["time_period"] if manufacturing_ready else ""},
    {"metric": "commercial_trade_only_candidate_ready", "value": commercial_trade_ready},
    {"metric": "commercial_trade_accommodation_food_candidate_ready", "value": commercial_broad_ready},
    {"metric": "commercial_best_time_period", "value": commercial_best["time_period"] if commercial_ready else ""},
    {"metric": "raw_names_with_mojibake_in_sample", "value": raw_names_with_mojibake},
    {"metric": "base_names_with_mojibake", "value": base_names_with_mojibake},
    {
        "metric": "important_method_note",
        "value": (
            "This dashboard extract is at CSD geography with 2021 DGUIDs and semester time periods. "
            "The inspection filters to Québec, total with employees, target NAICS sectors, aggregates CSD counts to CDs, "
            "and divides by land_area_km2."
        ),
    },
    {
        "metric": "recommended_next_step",
        "value": (
            "Review business_establishments_target_summary_2021.csv and business_establishments_formula_audit_2021.csv. "
            "If manufacturing and at least one commercial candidate have 98/98 coverage, choose the commercial proxy and generate the cleaner."
        ),
    },
]

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CENSUS DIVISION BUSINESS ESTABLISHMENTS DASHBOARD INSPECTION 2021")
print("=" * 72)

print("\nInputs:")
print("Raw CSV:", safe_relative(RAW_CSV))
print("Raw encoding:", raw_encoding)
print("Base frame:", safe_relative(BASE_CD_FRAME))
print("Base encoding:", base_encoding)

print("\nDetected columns:")
print("REF_AREA:", ref_area_col)
print("EMP_SIZE:", emp_size_col)
print("INDUSTRY:", industry_col)
print("TIME_PERIOD:", time_col)
print("OBS_VALUE:", value_col)
print("DGUID:", dguid_col)
print("PROV_TERR:", prov_col)
print("RURAL_FLAG:", rural_flag_col)

print("\nExtraction:")
print("Raw rows scanned:", raw_rows)
print("Québec total-with-employees rows extracted:", len(qc_total_all))
print("Candidate source rows:", len(candidate_source_rows))
print("Available periods:", " | ".join(available_periods))
print("Selected period for review:", selected_time_period)

print("\nGeography:")
print("Unique Québec CSD codes:", len(csd_codes_from_table))
print("Unique Québec CD codes:", len(cd_codes_from_table))
print("Base CD rows:", len(base))

print("\nTarget summary:")
print(target_summary.to_string(index=False))

print("\nFormula audit preview:")
if formula_audit.empty:
    print("[none]")
else:
    print(formula_audit.head(80).to_string(index=False))

print("\nMojibake check:")
print("Raw names with mojibake in sample:", raw_names_with_mojibake)
print("Base names with mojibake:", base_names_with_mojibake)

print("\nSaved:")
print(OUTPUT_SUMMARY)
print(OUTPUT_RAW_COLUMN_PROFILE)
print(OUTPUT_DIMENSION_INVENTORY)
print(OUTPUT_GEOGRAPHY_AUDIT)
print(OUTPUT_TIME_PERIOD_AUDIT)
print(OUTPUT_INDUSTRY_INVENTORY)
print(OUTPUT_CANDIDATE_SOURCE_ROWS)
print(OUTPUT_FORMULA_AUDIT)
print(OUTPUT_TARGET_SUMMARY)
print(OUTPUT_UNMATCHED_AUDIT)

print("\nDone.")