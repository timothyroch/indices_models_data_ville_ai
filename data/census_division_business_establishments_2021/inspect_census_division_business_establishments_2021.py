from pathlib import Path
import codecs
import re
import pandas as pd


# ============================================================
# Inspect Census Division Business Establishments 2021
# ============================================================
#
# Purpose:
#   Inspect whether Canadian Business Counts table 33-10-0397-01
#   can support two remaining SoVI variables:
#
#       MAESDEN92  -> manufacturing_density
#       COMDEVDN92 -> commercial_density
#
# Source:
#   Canadian Business Counts, with employees, census metropolitan
#   areas and census subdivisions, June 2021
#
# Raw files expected:
#   census_division_business_establishments_2021/raw/
#       canadian_business_counts_june_2021.csv
#       canadian_business_counts_june_2021_MetaData.csv
#
# Method idea:
#   The table is at census subdivision geography. We inspect whether
#   Québec CSD rows can be aggregated to Québec census divisions, then
#   divided by census-division land area.
#
# Candidate formulas:
#
#   manufacturing_density =
#       NAICS 31-33 Manufacturing business location count / land_area_km2
#
#   commercial_density candidates:
#       retail trade only
#       wholesale + retail trade
#       wholesale + retail + accommodation/food services
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

RAW_CSV = RAW_DIR / "canadian_business_counts_june_2021.csv"
METADATA_CSV = RAW_DIR / "canadian_business_counts_june_2021_MetaData.csv"

BASE_CD_FRAME = (
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.csv"
)

OUTPUT_SUMMARY = OUTPUT_DIR / "business_establishments_inspection_summary_2021.csv"
OUTPUT_RAW_COLUMN_PROFILE = OUTPUT_DIR / "business_establishments_raw_column_profile_2021.csv"
OUTPUT_METADATA_PREVIEW = OUTPUT_DIR / "business_establishments_metadata_preview_2021.csv"
OUTPUT_DIMENSION_INVENTORY = OUTPUT_DIR / "business_establishments_dimension_inventory_2021.csv"
OUTPUT_GEOGRAPHY_AUDIT = OUTPUT_DIR / "business_establishments_geography_audit_2021.csv"
OUTPUT_NAICS_INVENTORY = OUTPUT_DIR / "business_establishments_naics_inventory_2021.csv"
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

STANDARD_STATCAN_COLUMNS = {
    "REF_DATE",
    "GEO",
    "DGUID",
    "UOM",
    "UOM_ID",
    "SCALAR_FACTOR",
    "SCALAR_ID",
    "VECTOR",
    "COORDINATE",
    "VALUE",
    "STATUS",
    "SYMBOL",
    "TERMINATED",
    "DECIMALS",
}

COMPONENTS = {
    "manufacturing": {
        "target": "manufacturing_density",
        "original_sovi_code": "MAESDEN92",
        "naics_codes": {"31-33"},
        "name_terms": ["manufacturing"],
        "interpretation": "NAICS 31-33 Manufacturing business locations with employees.",
    },
    "wholesale_trade": {
        "target": "commercial_density",
        "original_sovi_code": "COMDEVDN92",
        "naics_codes": {"41"},
        "name_terms": ["wholesale trade"],
        "interpretation": "NAICS 41 Wholesale trade business locations with employees.",
    },
    "retail_trade": {
        "target": "commercial_density",
        "original_sovi_code": "COMDEVDN92",
        "naics_codes": {"44-45"},
        "name_terms": ["retail trade"],
        "interpretation": "NAICS 44-45 Retail trade business locations with employees.",
    },
    "accommodation_food_services": {
        "target": "commercial_density",
        "original_sovi_code": "COMDEVDN92",
        "naics_codes": {"72"},
        "name_terms": ["accommodation and food services"],
        "interpretation": "NAICS 72 Accommodation and food services business locations with employees.",
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


def extract_csd_sgc_from_dguid(value: object) -> str:
    text = normalize_text(value)
    match = re.search(r"A0005(\d{7})", text)
    if match:
        return match.group(1)
    return ""


def extract_csd_sgc_from_geo(value: object) -> str:
    text = normalize_text(value)

    # Common StatCan pattern: [CSD2466023]
    match = re.search(r"\[CSD\s*(\d{7})\]", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    # Fallback: any bracketed CSD-like 7-digit Québec SGC code.
    match = re.search(r"\[(?:.*?)(24\d{5})(?:.*?)\]", text)
    if match:
        return match.group(1)

    return ""


def extract_cd_code_from_csd_sgc(csd_sgc: object) -> str:
    text = normalize_text(csd_sgc)
    if re.fullmatch(r"\d{7}", text):
        return text[:4]
    return ""


def parse_naics_code(label: object) -> str:
    text = normalize_text(label)

    # Examples that this catches:
    #   "31-33 Manufacturing"
    #   "31-33 - Manufacturing"
    #   "Manufacturing [31-33]"
    #   "[31-33]"
    leading = re.match(r"^\s*(\d{2}(?:-\d{2})?)\b", text)
    if leading:
        return leading.group(1)

    bracket = re.search(r"\[(\d{2}(?:-\d{2})?)\]", text)
    if bracket:
        return bracket.group(1)

    # Some StatCan files include "code - name" in unexpected positions.
    anywhere = re.search(r"\b(\d{2}(?:-\d{2})?)\b", text)
    if anywhere:
        return anywhere.group(1)

    return ""


def classify_naics_component(label: object) -> str:
    text = normalize_lower(label)
    code = parse_naics_code(label)

    if code == "31-33":
        return "manufacturing"
    if code == "41":
        return "wholesale_trade"
    if code == "44-45":
        return "retail_trade"
    if code == "72":
        return "accommodation_food_services"

    # Conservative fallback for files where the code is not visible.
    stripped = re.sub(r"^\d{2}(?:-\d{2})?\s*[-:]?\s*", "", text).strip()

    if stripped == "manufacturing":
        return "manufacturing"
    if stripped == "wholesale trade":
        return "wholesale_trade"
    if stripped == "retail trade":
        return "retail_trade"
    if stripped == "accommodation and food services":
        return "accommodation_food_services"

    return ""


def make_dimension_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in df.columns:
        values = df[col].dropna().astype(str)
        unique_values = values.nunique(dropna=True)
        sample_values = " | ".join(values.drop_duplicates().head(20).tolist())

        lower_samples = sample_values.lower()

        rows.append(
            {
                "column": col,
                "unique_values": int(unique_values),
                "sample_values": sample_values,
                "is_standard_statcan_column": col in STANDARD_STATCAN_COLUMNS,
                "contains_naics_term": "naics" in col.lower() or "classification" in col.lower(),
                "contains_geography_term": col.lower() in ["geo", "dguid"] or "geography" in col.lower(),
                "sample_contains_manufacturing": "manufacturing" in lower_samples,
                "sample_contains_retail": "retail" in lower_samples,
                "sample_contains_wholesale": "wholesale" in lower_samples,
                "sample_contains_accommodation_food": "accommodation" in lower_samples or "food services" in lower_samples,
            }
        )

    return pd.DataFrame(rows)


def aggregate_component_to_cd(
    data: pd.DataFrame,
    component: str,
    base: pd.DataFrame,
    cd_code_col: str,
    value_col: str,
) -> tuple[pd.DataFrame, dict]:
    subset = data[data["_component_alias"] == component].copy()

    duplicate_csd_rows = 0
    if not subset.empty:
        duplicate_csd_rows = int(subset.duplicated(subset=["_csd_sgc_code"]).sum())

    aggregated = (
        subset.groupby(cd_code_col, dropna=False)[value_col]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={cd_code_col: "census_division_code", value_col: f"{component}_business_count"})
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

    values = clean_numeric(joined[f"{component}_business_count"])
    density = clean_numeric(joined[f"{component}_density_per_km2"])

    audit = {
        "component": component,
        "source_rows": len(subset),
        "unique_csd_rows": int(subset["_csd_sgc_code"].nunique()) if not subset.empty else 0,
        "duplicate_csd_rows_before_cd_aggregation": duplicate_csd_rows,
        "cd_rows_non_missing": int(values.notna().sum()),
        "cd_rows_missing": int(values.isna().sum()),
        "coverage_is_98_cds": int(values.notna().sum()) == EXPECTED_QC_CD_COUNT,
        "business_count_min": values.min(skipna=True),
        "business_count_max": values.max(skipna=True),
        "business_count_mean": values.mean(skipna=True),
        "business_count_median": values.median(skipna=True),
        "density_per_km2_min": density.min(skipna=True),
        "density_per_km2_max": density.max(skipna=True),
        "density_per_km2_mean": density.mean(skipna=True),
        "density_per_km2_median": density.median(skipna=True),
    }

    return joined, audit


def combine_components(
    component_tables: dict[str, pd.DataFrame],
    components: list[str],
    alias: str,
    base: pd.DataFrame,
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
        "candidate_variable": alias,
        "components": " + ".join(components),
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
# Validate inputs
# -----------------------------

if not RAW_CSV.exists():
    raise FileNotFoundError(f"Missing raw CSV:\n{RAW_CSV}")

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


# -----------------------------
# Load raw table
# -----------------------------

raw_encoding = detect_file_encoding_strict(RAW_CSV, ENCODING_CANDIDATES)

raw = pd.read_csv(RAW_CSV, encoding=raw_encoding, dtype=str, low_memory=False)
raw.columns = [str(col).strip() for col in raw.columns]

columns = list(raw.columns)

dguid_col = require_column(columns, ["DGUID", "dguid"], "DGUID")
geo_col = require_column(columns, ["GEO", "Geography", "Geography name"], "geography")
value_col = require_column(columns, ["VALUE", "Value"], "value")
naics_col = require_column(
    columns,
    [
        "North American Industry Classification System (NAICS)",
        "NAICS",
        "Industry",
        "North American Industry Classification System",
    ],
    "NAICS / industry",
)

status_col = find_column(columns, ["STATUS", "Status"])
uom_col = find_column(columns, ["UOM", "Unit of measure"])
ref_date_col = find_column(columns, ["REF_DATE", "Reference period", "Reference date"])
employment_size_col = find_column(columns, ["Employment size", "Employment size ranges", "Size"])

raw[value_col] = clean_numeric(raw[value_col])

# Metadata preview
if METADATA_CSV.exists():
    metadata_encoding = detect_file_encoding_strict(METADATA_CSV, ENCODING_CANDIDATES)
    metadata = pd.read_csv(METADATA_CSV, encoding=metadata_encoding, dtype=str, low_memory=False)
    metadata.head(200).to_csv(OUTPUT_METADATA_PREVIEW, index=False, encoding="utf-8")
else:
    metadata_encoding = ""
    pd.DataFrame().to_csv(OUTPUT_METADATA_PREVIEW, index=False, encoding="utf-8")


# -----------------------------
# Raw profiles and dimensions
# -----------------------------

raw_profile_rows = []

for col in raw.columns:
    numeric = clean_numeric(raw[col])
    raw_profile_rows.append(
        {
            "column": col,
            "dtype_as_loaded": str(raw[col].dtype),
            "non_missing": int(raw[col].notna().sum()),
            "missing": int(raw[col].isna().sum()),
            "unique_values": int(raw[col].astype("string").nunique(dropna=True)),
            "sample_values": " | ".join(raw[col].dropna().astype(str).drop_duplicates().head(12).tolist()),
            "numeric_non_missing": int(numeric.notna().sum()),
            "numeric_min": numeric.min(skipna=True),
            "numeric_max": numeric.max(skipna=True),
            "numeric_mean": numeric.mean(skipna=True),
        }
    )

raw_column_profile = pd.DataFrame(raw_profile_rows)
raw_column_profile.to_csv(OUTPUT_RAW_COLUMN_PROFILE, index=False, encoding="utf-8")

dimension_inventory = make_dimension_inventory(raw)
dimension_inventory.to_csv(OUTPUT_DIMENSION_INVENTORY, index=False, encoding="utf-8")


# -----------------------------
# Select reference period and employment-size total if needed
# -----------------------------

working = raw.copy()

if ref_date_col and ref_date_col in working.columns:
    ref_values = working[ref_date_col].dropna().astype(str).drop_duplicates().tolist()

    if len(ref_values) == 1:
        selected_ref_date = ref_values[0]
    else:
        june_2021 = [v for v in ref_values if "2021" in v and "june" in v.lower()]
        any_2021 = [v for v in ref_values if "2021" in v]
        selected_ref_date = june_2021[0] if june_2021 else (any_2021[0] if any_2021 else ref_values[0])

    working = working[working[ref_date_col].astype(str) == selected_ref_date].copy()
else:
    selected_ref_date = ""

if employment_size_col and employment_size_col in working.columns:
    size_values = working[employment_size_col].dropna().astype(str).drop_duplicates().tolist()
    total_size_values = [
        v for v in size_values
        if "all" in v.lower() and ("size" in v.lower() or "employees" in v.lower())
    ] + [
        v for v in size_values
        if "total" in v.lower()
    ]

    if total_size_values:
        selected_employment_size = total_size_values[0]
        working = working[working[employment_size_col].astype(str) == selected_employment_size].copy()
    else:
        selected_employment_size = ""
else:
    selected_employment_size = ""


# -----------------------------
# Geography parsing: CSD -> CD
# -----------------------------

working["_csd_sgc_code_from_dguid"] = working[dguid_col].map(extract_csd_sgc_from_dguid)
working["_csd_sgc_code_from_geo"] = working[geo_col].map(extract_csd_sgc_from_geo)
working["_csd_sgc_code"] = working["_csd_sgc_code_from_dguid"]
working.loc[working["_csd_sgc_code"] == "", "_csd_sgc_code"] = working.loc[
    working["_csd_sgc_code"] == "", "_csd_sgc_code_from_geo"
]

working["_is_csd_row"] = working["_csd_sgc_code"].astype(str).str.fullmatch(r"\d{7}", na=False)
working["_is_quebec_csd_row"] = working["_is_csd_row"] & working["_csd_sgc_code"].astype(str).str.startswith("24")
working["_cd_code"] = working["_csd_sgc_code"].map(extract_cd_code_from_csd_sgc)

qc_csd_rows = working[working["_is_quebec_csd_row"]].copy()
qc_csd_rows["_naics_code"] = qc_csd_rows[naics_col].map(parse_naics_code)
qc_csd_rows["_component_alias"] = qc_csd_rows[naics_col].map(classify_naics_component)

geography_audit = pd.DataFrame(
    [
        {
            "metric": "working_rows_after_ref_date_and_size_filter",
            "value": len(working),
        },
        {
            "metric": "rows_with_csd_sgc_code",
            "value": int(working["_is_csd_row"].sum()),
        },
        {
            "metric": "quebec_csd_rows",
            "value": len(qc_csd_rows),
        },
        {
            "metric": "unique_quebec_csd_codes",
            "value": qc_csd_rows["_csd_sgc_code"].nunique(),
        },
        {
            "metric": "unique_quebec_cd_codes_from_csd",
            "value": qc_csd_rows["_cd_code"].nunique(),
        },
        {
            "metric": "base_cd_rows",
            "value": len(base),
        },
        {
            "metric": "base_cd_codes_missing_from_business_table",
            "value": " | ".join(sorted(set(base["census_division_code"]) - set(qc_csd_rows["_cd_code"]))),
        },
        {
            "metric": "business_table_cd_codes_not_in_base",
            "value": " | ".join(sorted(set(qc_csd_rows["_cd_code"]) - set(base["census_division_code"]))),
        },
    ]
)

geography_audit.to_csv(OUTPUT_GEOGRAPHY_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# NAICS inventory
# -----------------------------

naics_inventory_rows = []

if not qc_csd_rows.empty:
    group_cols = [naics_col, "_naics_code", "_component_alias"]
    grouped = qc_csd_rows.groupby(group_cols, dropna=False)

    for keys, group in grouped:
        label, naics_code, component_alias = keys
        values = clean_numeric(group[value_col])

        row = {
            "naics_label": normalize_text(label),
            "parsed_naics_code": normalize_text(naics_code),
            "component_alias": normalize_text(component_alias),
            "n_rows": len(group),
            "unique_csd_codes": group["_csd_sgc_code"].nunique(),
            "unique_cd_codes": group["_cd_code"].nunique(),
            "value_non_missing": int(values.notna().sum()),
            "value_missing": int(values.isna().sum()),
            "value_min": values.min(skipna=True),
            "value_max": values.max(skipna=True),
            "value_mean": values.mean(skipna=True),
            "value_median": values.median(skipna=True),
            "coverage_has_all_base_cds": set(base["census_division_code"]).issubset(set(group["_cd_code"])),
        }

        if status_col and status_col in group.columns:
            row["status_values"] = " | ".join(sorted(group[status_col].dropna().astype(str).unique())[:20])
        else:
            row["status_values"] = ""

        if uom_col and uom_col in group.columns:
            row["unit_values"] = " | ".join(sorted(group[uom_col].dropna().astype(str).unique())[:20])
        else:
            row["unit_values"] = ""

        naics_inventory_rows.append(row)

naics_inventory = pd.DataFrame(naics_inventory_rows)

if not naics_inventory.empty:
    naics_inventory = naics_inventory.sort_values(
        ["component_alias", "parsed_naics_code", "naics_label"],
        na_position="last",
    )

naics_inventory.to_csv(OUTPUT_NAICS_INVENTORY, index=False, encoding="utf-8")


# -----------------------------
# Candidate source rows
# -----------------------------

candidate_source_rows = qc_csd_rows[qc_csd_rows["_component_alias"] != ""].copy()
candidate_source_rows.to_csv(OUTPUT_CANDIDATE_SOURCE_ROWS, index=False, encoding="utf-8")


# -----------------------------
# Component aggregation and formula audit
# -----------------------------

component_tables = {}
component_audit_rows = []

for component in COMPONENTS:
    table, audit = aggregate_component_to_cd(
        data=qc_csd_rows,
        component=component,
        base=base,
        cd_code_col="_cd_code",
        value_col=value_col,
    )

    component_tables[component] = table
    component_audit_rows.append(
        {
            "candidate_variable": f"{component}_density_per_km2",
            "original_sovi_code": COMPONENTS[component]["original_sovi_code"],
            "candidate_type": "single_naics_sector_density",
            "components": component,
            "formula": f"{component}_business_count / land_area_km2",
            "interpretation": COMPONENTS[component]["interpretation"],
            **audit,
            "recommended_default_without_review": component == "manufacturing",
        }
    )

combined_audit_rows = []

commercial_trade_only, audit_trade_only = combine_components(
    component_tables=component_tables,
    components=["wholesale_trade", "retail_trade"],
    alias="commercial_trade_only",
    base=base,
)

combined_audit_rows.append(
    {
        "candidate_variable": "commercial_trade_only_density_per_km2",
        "original_sovi_code": "COMDEVDN92",
        "candidate_type": "combined_naics_sector_density",
        "formula": "(wholesale_trade + retail_trade) / land_area_km2",
        "interpretation": "Commercial density candidate using wholesale and retail trade business locations with employees.",
        **audit_trade_only,
        "recommended_default_without_review": False,
    }
)

commercial_trade_food, audit_trade_food = combine_components(
    component_tables=component_tables,
    components=["wholesale_trade", "retail_trade", "accommodation_food_services"],
    alias="commercial_trade_accommodation_food",
    base=base,
)

combined_audit_rows.append(
    {
        "candidate_variable": "commercial_trade_accommodation_food_density_per_km2",
        "original_sovi_code": "COMDEVDN92",
        "candidate_type": "combined_naics_sector_density",
        "formula": "(wholesale_trade + retail_trade + accommodation_food_services) / land_area_km2",
        "interpretation": "Broader commercial density candidate using wholesale, retail, accommodation, and food services business locations with employees.",
        **audit_trade_food,
        "recommended_default_without_review": False,
    }
)

formula_audit = pd.DataFrame(component_audit_rows + combined_audit_rows)
formula_audit.to_csv(OUTPUT_FORMULA_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Unmatched audit
# -----------------------------

missing_any = base[
    ["census_division_code", "census_division_dguid", "census_division_name"]
].copy()

for component, table in component_tables.items():
    missing_any = missing_any.merge(
        table[["census_division_code", f"{component}_business_count"]],
        on="census_division_code",
        how="left",
        validate="one_to_one",
    )

missing_mask = False
for component in component_tables:
    missing_mask = missing_mask | missing_any[f"{component}_business_count"].isna()

unmatched_audit = missing_any[missing_mask].copy()
unmatched_audit.to_csv(OUTPUT_UNMATCHED_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Target summary
# -----------------------------

manufacturing_ready = bool(
    (
        (formula_audit["candidate_variable"] == "manufacturing_density_per_km2")
        & (formula_audit["coverage_is_98_cds"] == True)
        & (formula_audit["duplicate_csd_rows_before_cd_aggregation"] == 0)
    ).any()
)

commercial_trade_ready = bool(
    (
        (formula_audit["candidate_variable"] == "commercial_trade_only_density_per_km2")
        & (formula_audit["coverage_is_98_cds"] == True)
    ).any()
)

commercial_broad_ready = bool(
    (
        (formula_audit["candidate_variable"] == "commercial_trade_accommodation_food_density_per_km2")
        & (formula_audit["coverage_is_98_cds"] == True)
    ).any()
)

target_summary_rows = [
    {
        "canonical_variable": "manufacturing_density",
        "original_sovi_code": "MAESDEN92",
        "candidate_found": manufacturing_ready,
        "best_candidate_variable": "manufacturing_density_per_km2" if manufacturing_ready else "",
        "candidate_formula": "manufacturing_business_count / land_area_km2" if manufacturing_ready else "",
        "coverage_is_98_cds": manufacturing_ready,
        "status": "candidate_available_needs_review" if manufacturing_ready else "candidate_not_ready_or_missing",
        "interpretation": (
            "Manufacturing business-location density from NAICS 31-33, aggregated from Québec CSDs to CDs."
            if manufacturing_ready
            else "No full-coverage manufacturing sector candidate was confirmed."
        ),
    },
    {
        "canonical_variable": "commercial_density",
        "original_sovi_code": "COMDEVDN92",
        "candidate_found": commercial_trade_ready or commercial_broad_ready,
        "best_candidate_variable": (
            "commercial_trade_accommodation_food_density_per_km2"
            if commercial_broad_ready
            else ("commercial_trade_only_density_per_km2" if commercial_trade_ready else "")
        ),
        "candidate_formula": (
            "(wholesale_trade + retail_trade + accommodation_food_services) / land_area_km2"
            if commercial_broad_ready
            else ("(wholesale_trade + retail_trade) / land_area_km2" if commercial_trade_ready else "")
        ),
        "coverage_is_98_cds": commercial_trade_ready or commercial_broad_ready,
        "status": (
            "candidate_available_needs_conceptual_choice"
            if commercial_trade_ready or commercial_broad_ready
            else "candidate_not_ready_or_missing"
        ),
        "interpretation": (
            "Commercial density is not a single NAICS sector. Review whether the final proxy should use trade only "
            "or trade plus accommodation/food services."
            if commercial_trade_ready or commercial_broad_ready
            else "No full-coverage commercial-sector combination was confirmed."
        ),
    },
]

target_summary = pd.DataFrame(target_summary_rows)
target_summary.to_csv(OUTPUT_TARGET_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Summary
# -----------------------------

raw_names_with_mojibake = contains_mojibake(raw[geo_col]) + contains_mojibake(raw[naics_col])
base_names_with_mojibake = contains_mojibake(base["census_division_name"])

summary_rows = [
    {"metric": "raw_csv", "value": safe_relative(RAW_CSV)},
    {"metric": "metadata_csv", "value": safe_relative(METADATA_CSV) if METADATA_CSV.exists() else ""},
    {"metric": "base_cd_frame", "value": safe_relative(BASE_CD_FRAME)},
    {"metric": "raw_encoding", "value": raw_encoding},
    {"metric": "metadata_encoding", "value": metadata_encoding},
    {"metric": "base_encoding", "value": base_encoding},
    {"metric": "raw_rows", "value": len(raw)},
    {"metric": "raw_columns", "value": len(raw.columns)},
    {"metric": "working_rows_after_ref_date_and_size_filter", "value": len(working)},
    {"metric": "selected_ref_date", "value": selected_ref_date},
    {"metric": "selected_employment_size", "value": selected_employment_size},
    {"metric": "base_rows", "value": len(base)},
    {"metric": "detected_dguid_column", "value": dguid_col},
    {"metric": "detected_geo_column", "value": geo_col},
    {"metric": "detected_naics_column", "value": naics_col},
    {"metric": "detected_value_column", "value": value_col},
    {"metric": "detected_status_column", "value": status_col or ""},
    {"metric": "detected_uom_column", "value": uom_col or ""},
    {"metric": "detected_ref_date_column", "value": ref_date_col or ""},
    {"metric": "detected_employment_size_column", "value": employment_size_col or ""},
    {"metric": "quebec_csd_rows", "value": len(qc_csd_rows)},
    {"metric": "unique_quebec_csd_codes", "value": qc_csd_rows["_csd_sgc_code"].nunique()},
    {"metric": "unique_quebec_cd_codes_from_csd", "value": qc_csd_rows["_cd_code"].nunique()},
    {"metric": "candidate_source_rows", "value": len(candidate_source_rows)},
    {"metric": "manufacturing_candidate_ready", "value": manufacturing_ready},
    {"metric": "commercial_trade_only_candidate_ready", "value": commercial_trade_ready},
    {"metric": "commercial_trade_accommodation_food_candidate_ready", "value": commercial_broad_ready},
    {"metric": "raw_names_with_mojibake", "value": raw_names_with_mojibake},
    {"metric": "base_names_with_mojibake", "value": base_names_with_mojibake},
    {
        "metric": "important_method_note",
        "value": (
            "This table is expected to be at census subdivision geography. The inspection parses Québec CSD SGC codes, "
            "aggregates business counts to Québec census divisions, and divides by census-division land area."
        ),
    },
    {
        "metric": "recommended_next_step",
        "value": (
            "Review business_establishments_naics_inventory_2021.csv, business_establishments_formula_audit_2021.csv, "
            "and business_establishments_target_summary_2021.csv. If manufacturing has 98/98 coverage and commercial "
            "components are conceptually acceptable, generate the cleaner."
        ),
    },
]

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CENSUS DIVISION BUSINESS ESTABLISHMENTS INSPECTION 2021")
print("=" * 72)

print("\nInputs:")
print("Raw CSV:", safe_relative(RAW_CSV))
print("Raw encoding:", raw_encoding)
print("Metadata CSV:", safe_relative(METADATA_CSV) if METADATA_CSV.exists() else "[missing]")
print("Base frame:", safe_relative(BASE_CD_FRAME))

print("\nDetected columns:")
print("DGUID:", dguid_col)
print("GEO:", geo_col)
print("NAICS:", naics_col)
print("VALUE:", value_col)
print("STATUS:", status_col)
print("UOM:", uom_col)
print("REF_DATE:", ref_date_col)
print("Employment size:", employment_size_col)

print("\nFilters:")
print("Selected REF_DATE:", selected_ref_date)
print("Selected employment size:", selected_employment_size)

print("\nGeography:")
print("Working rows:", len(working))
print("Québec CSD rows:", len(qc_csd_rows))
print("Unique Québec CSD codes:", qc_csd_rows["_csd_sgc_code"].nunique())
print("Unique Québec CD codes from CSD:", qc_csd_rows["_cd_code"].nunique())
print("Base CD rows:", len(base))

print("\nTarget summary:")
print(target_summary.to_string(index=False))

print("\nFormula audit:")
print(formula_audit.to_string(index=False))

print("\nNAICS candidates found:")
if candidate_source_rows.empty:
    print("[none]")
else:
    print(
        candidate_source_rows[
            [geo_col, "_csd_sgc_code", "_cd_code", naics_col, "_naics_code", "_component_alias", value_col]
            + ([status_col] if status_col else [])
        ].head(40).to_string(index=False)
    )

print("\nMojibake check:")
print("Raw names with mojibake:", raw_names_with_mojibake)
print("Base names with mojibake:", base_names_with_mojibake)

print("\nSaved:")
print(OUTPUT_SUMMARY)
print(OUTPUT_RAW_COLUMN_PROFILE)
print(OUTPUT_METADATA_PREVIEW)
print(OUTPUT_DIMENSION_INVENTORY)
print(OUTPUT_GEOGRAPHY_AUDIT)
print(OUTPUT_NAICS_INVENTORY)
print(OUTPUT_CANDIDATE_SOURCE_ROWS)
print(OUTPUT_FORMULA_AUDIT)
print(OUTPUT_TARGET_SUMMARY)
print(OUTPUT_UNMATCHED_AUDIT)

print("\nDone.")