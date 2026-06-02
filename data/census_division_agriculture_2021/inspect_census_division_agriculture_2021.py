from pathlib import Path
import re
import pandas as pd


# ============================================================
# Inspect Census Division Agriculture Land Use 2021
# ============================================================
#
# Purpose:
#   Inspect Statistics Canada Census of Agriculture table 32-10-0249-01:
#
#       Land use, Census of Agriculture, 2021
#
#   for:
#
#       PCTFARMS92 -> pct_land_farms
#       PCTRFRM90  -> pct_rural_farm
#
# Key correction:
#   Use the human-readable column `Unit of measure`, not the coded column `UOM`.
#
# Expected pct_land_farms formula:
#
#   pct_land_farms =
#       100 * total_farm_area_km2 / land_area_km2
#
# Preferred source row:
#
#   Land use = Total farm area
#   Unit of measure = Hectares
#
# Run from data/:
#
#   python census_division_agriculture_2021/inspect_census_division_agriculture_2021.py
#
# ============================================================


from pathlib import Path
import re
import pandas as pd


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_agriculture_2021"
RAW_DIR = SECTION_DIR / "raw"
OUTPUT_DIR = SECTION_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_LAND_USE_CSV = RAW_DIR / "land_use_32100249_2021.csv"
RAW_LAND_USE_METADATA_CSV = RAW_DIR / "land_use_32100249_2021_MetaData.csv"

BASE_CD_FRAME = (
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.csv"
)

OUTPUT_RAW_COLUMN_PROFILE = OUTPUT_DIR / "agriculture_land_use_raw_column_profile_2021.csv"
OUTPUT_METADATA_PREVIEW = OUTPUT_DIR / "agriculture_land_use_metadata_preview_2021.csv"
OUTPUT_QUEBEC_CD_ROWS = OUTPUT_DIR / "agriculture_land_use_quebec_cd_rows_2021.csv"
OUTPUT_DIMENSION_INVENTORY = OUTPUT_DIR / "agriculture_land_use_dimension_inventory_2021.csv"
OUTPUT_CANDIDATE_ROWS = OUTPUT_DIR / "agriculture_land_use_candidate_rows_2021.csv"
OUTPUT_SELECTED_AREA_ROWS = OUTPUT_DIR / "agriculture_land_use_selected_total_farm_area_rows_2021.csv"
OUTPUT_TARGET_SUMMARY = OUTPUT_DIR / "agriculture_target_summary_2021.csv"
OUTPUT_FORMULA_AUDIT = OUTPUT_DIR / "agriculture_pct_land_farms_formula_audit_2021.csv"
OUTPUT_UNMATCHED_AUDIT = OUTPUT_DIR / "agriculture_land_use_unmatched_audit_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "agriculture_inspection_summary_2021.csv"


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

AREA_UNIT_TO_KM2 = {
    "hectares": 0.01,
    "hectare": 0.01,
    "acres": 0.0040468564224,
    "acre": 0.0040468564224,
    "square kilometres": 1.0,
    "square kilometre": 1.0,
    "square kilometers": 1.0,
    "square kilometer": 1.0,
    "km2": 1.0,
    "km²": 1.0,
}

PREFERRED_AREA_UNITS = [
    "hectares",
    "acres",
    "square kilometres",
    "square kilometers",
    "km2",
    "km²",
]

TOTAL_FARM_AREA_VALUE = "total farm area"

RURAL_FARM_POPULATION_TERMS = [
    "rural farm population",
    "farm population",
    "population in farm households",
    "farm household population",
    "farm operators",
    "operators",
    "agricultural workers",
    "paid labour",
]


# -----------------------------
# Helpers
# -----------------------------

def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        return str(path)


def select_encoding(path: Path) -> str:
    for encoding in ENCODING_CANDIDATES:
        try:
            pd.read_csv(path, encoding=encoding, nrows=20, low_memory=False)
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


def row_text(df: pd.DataFrame) -> pd.Series:
    return df.astype("string").fillna("").agg(" | ".join, axis=1).str.lower()


def find_exact_column(columns: list[str], target: str) -> str | None:
    for col in columns:
        if col.strip().lower() == target.strip().lower():
            return col
    return None


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        exact = find_exact_column(columns, candidate)
        if exact:
            return exact

    for candidate in candidates:
        candidate_lower = candidate.lower()
        for col in columns:
            if candidate_lower in col.lower():
                return col

    return None


def detect_dguid_column(columns: list[str]) -> str | None:
    return find_column(columns, ["DGUID"])


def detect_geo_column(columns: list[str]) -> str | None:
    return find_column(columns, ["GEO", "Geography", "Geography name"])


def detect_value_column(columns: list[str]) -> str | None:
    return find_column(columns, ["VALUE", "Value"])


def detect_unit_of_measure_column(columns: list[str]) -> str | None:
    # Critical: prefer the human-readable unit label over the coded UOM column.
    exact = find_exact_column(columns, "Unit of measure")
    if exact:
        return exact

    exact = find_exact_column(columns, "Units")
    if exact:
        return exact

    # Fallback only after human-readable labels are unavailable.
    return find_column(columns, ["Unit of measure", "Units"])


def detect_uom_code_column(columns: list[str]) -> str | None:
    return find_exact_column(columns, "UOM")


def detect_status_column(columns: list[str]) -> str | None:
    return find_column(columns, ["STATUS", "Symbol", "SYMBOL"])


def detect_land_use_column(columns: list[str]) -> str | None:
    return find_column(columns, ["Land use"])


def infer_area_unit_multiplier(unit_value: object) -> tuple[float | None, str]:
    unit_text = normalize_lower(unit_value)

    for key, multiplier in AREA_UNIT_TO_KM2.items():
        if key in unit_text:
            return multiplier, key

    return None, ""


def is_area_unit(value: object) -> bool:
    return infer_area_unit_multiplier(value)[0] is not None


def preferred_unit_rank(unit_value: object) -> int:
    unit_text = normalize_lower(unit_value)

    for idx, preferred in enumerate(PREFERRED_AREA_UNITS):
        if preferred in unit_text:
            return idx

    return 999


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in df.columns:
        series = df[col]
        numeric = clean_numeric(series)

        rows.append(
            {
                "column": col,
                "dtype_as_loaded": str(series.dtype),
                "non_missing": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "unique_values": int(series.astype("string").nunique(dropna=True)),
                "sample_values": " | ".join(series.dropna().astype(str).head(12).tolist()),
                "numeric_non_missing": int(numeric.notna().sum()),
                "numeric_min": numeric.min(skipna=True),
                "numeric_max": numeric.max(skipna=True),
                "numeric_mean": numeric.mean(skipna=True),
            }
        )

    return pd.DataFrame(rows)


def summarize_candidate_set(
    canonical_variable: str,
    original_sovi_code: str,
    candidate_name: str,
    rows: pd.DataFrame,
    value_col: str | None,
    dguid_col: str | None,
    geo_col: str | None,
    unit_col: str | None,
    note: str,
) -> dict:
    if rows.empty or value_col is None:
        return {
            "canonical_variable": canonical_variable,
            "original_sovi_code": original_sovi_code,
            "candidate_name": candidate_name,
            "n_rows": 0,
            "n_unique_dguid": 0,
            "n_unique_geo": 0,
            "value_non_missing": 0,
            "value_missing": EXPECTED_QC_CD_COUNT,
            "value_min": None,
            "value_max": None,
            "value_mean": None,
            "value_median": None,
            "unique_units": "",
            "coverage_is_98_cds": False,
            "status": "candidate_not_found",
            "note": note,
        }

    values = clean_numeric(rows[value_col])

    n_unique_dguid = int(rows[dguid_col].nunique()) if dguid_col and dguid_col in rows.columns else 0
    n_unique_geo = int(rows[geo_col].nunique()) if geo_col and geo_col in rows.columns else 0

    unique_units = ""
    if unit_col and unit_col in rows.columns:
        unique_units = " | ".join(sorted(rows[unit_col].dropna().astype(str).unique())[:20])

    coverage_is_98 = max(n_unique_dguid, n_unique_geo) == EXPECTED_QC_CD_COUNT

    if coverage_is_98 and int(values.notna().sum()) >= EXPECTED_QC_CD_COUNT:
        status = "candidate_found_full_or_multirow_coverage"
    elif not rows.empty:
        status = "candidate_needs_review"
    else:
        status = "candidate_not_found"

    return {
        "canonical_variable": canonical_variable,
        "original_sovi_code": original_sovi_code,
        "candidate_name": candidate_name,
        "n_rows": len(rows),
        "n_unique_dguid": n_unique_dguid,
        "n_unique_geo": n_unique_geo,
        "value_non_missing": int(values.notna().sum()),
        "value_missing": int(values.isna().sum()),
        "value_min": values.min(skipna=True),
        "value_max": values.max(skipna=True),
        "value_mean": values.mean(skipna=True),
        "value_median": values.median(skipna=True),
        "unique_units": unique_units,
        "coverage_is_98_cds": coverage_is_98,
        "status": status,
        "note": note,
    }


# -----------------------------
# Validate inputs
# -----------------------------

if not RAW_LAND_USE_CSV.exists():
    raise FileNotFoundError(f"Missing raw land-use CSV:\n{RAW_LAND_USE_CSV}")

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
base["census_division_name"] = base["census_division_name"].astype("string").str.strip()
base["land_area_km2"] = clean_numeric(base["land_area_km2"])

if len(base) != EXPECTED_QC_CD_COUNT:
    raise ValueError(f"Expected {EXPECTED_QC_CD_COUNT} Québec CDs in base frame, got {len(base)}.")

if base["census_division_dguid"].duplicated().any():
    raise ValueError("Duplicate census_division_dguid values in base frame.")

base_dguid_set = set(base["census_division_dguid"].dropna().astype(str))


# -----------------------------
# Load land-use table
# -----------------------------

encoding = select_encoding(RAW_LAND_USE_CSV)

raw = pd.read_csv(
    RAW_LAND_USE_CSV,
    encoding=encoding,
    dtype=str,
    low_memory=False,
)

raw.columns = [str(col).strip() for col in raw.columns]

print("\nInspecting Census Division Agriculture Land Use 2021")
print("Raw CSV:", safe_relative(RAW_LAND_USE_CSV))
print("Raw encoding:", encoding)
print("Raw rows:", len(raw))
print("Raw columns:", len(raw.columns))
print("Base CD rows:", len(base))


# -----------------------------
# Metadata preview
# -----------------------------

if RAW_LAND_USE_METADATA_CSV.exists():
    try:
        metadata_encoding = select_encoding(RAW_LAND_USE_METADATA_CSV)
        metadata = pd.read_csv(
            RAW_LAND_USE_METADATA_CSV,
            encoding=metadata_encoding,
            dtype=str,
            low_memory=False,
        )
        metadata.columns = [str(col).strip() for col in metadata.columns]
        metadata.head(1000).to_csv(OUTPUT_METADATA_PREVIEW, index=False, encoding="utf-8")
    except Exception as exc:
        metadata = pd.DataFrame([{"metadata_read_error": str(exc)}])
        metadata.to_csv(OUTPUT_METADATA_PREVIEW, index=False, encoding="utf-8")
else:
    metadata = pd.DataFrame([{"metadata_file_found": False}])
    metadata.to_csv(OUTPUT_METADATA_PREVIEW, index=False, encoding="utf-8")


# -----------------------------
# Detect key columns
# -----------------------------

columns = list(raw.columns)

dguid_col = detect_dguid_column(columns)
geo_col = detect_geo_column(columns)
value_col = detect_value_column(columns)
unit_col = detect_unit_of_measure_column(columns)
uom_code_col = detect_uom_code_column(columns)
status_col = detect_status_column(columns)
land_use_col = detect_land_use_column(columns)

raw_column_profile = profile_columns(raw)
raw_column_profile.to_csv(OUTPUT_RAW_COLUMN_PROFILE, index=False, encoding="utf-8")

if dguid_col is None:
    raise ValueError("Could not detect DGUID column in land-use table.")

if value_col is None:
    raise ValueError("Could not detect VALUE column in land-use table.")

if unit_col is None:
    raise ValueError(
        "Could not detect human-readable unit column. Expected a column named 'Unit of measure'."
    )

if land_use_col is None:
    raise ValueError("Could not detect Land use column in land-use table.")


# -----------------------------
# Filter Québec census-division rows
# -----------------------------

raw["_source_dguid_key"] = raw[dguid_col].astype("string").str.strip()
qc_rows = raw[raw["_source_dguid_key"].isin(base_dguid_set)].copy()

qc_rows.to_csv(OUTPUT_QUEBEC_CD_ROWS, index=False, encoding="utf-8")


# -----------------------------
# Dimension inventory
# -----------------------------

dimension_inventory_rows = []

candidate_dimension_cols = [
    col for col in raw.columns
    if col not in {
        dguid_col,
        geo_col,
        value_col,
        unit_col,
        uom_code_col,
        status_col,
        "REF_DATE",
        "SCALAR_FACTOR",
        "SCALAR_ID",
        "VECTOR",
        "COORDINATE",
        "SYMBOL",
        "TERMINATED",
        "DECIMALS",
        "_source_dguid_key",
    }
]

for col in candidate_dimension_cols + [unit_col]:
    if col not in qc_rows.columns:
        continue

    values = qc_rows[col].dropna().astype(str).sort_values().unique()

    dimension_inventory_rows.append(
        {
            "column": col,
            "unique_values": len(values),
            "sample_values": " | ".join(values[:80]),
            "contains_total_farm_area_term": any(
                TOTAL_FARM_AREA_VALUE in normalize_lower(value)
                for value in values
            ),
            "contains_rural_farm_population_term": any(
                any(term in normalize_lower(value) for term in RURAL_FARM_POPULATION_TERMS)
                for value in values
            ),
        }
    )

dimension_inventory = pd.DataFrame(dimension_inventory_rows)
dimension_inventory.to_csv(OUTPUT_DIMENSION_INVENTORY, index=False, encoding="utf-8")


# -----------------------------
# Candidate rows
# -----------------------------

if qc_rows.empty:
    total_farm_area_candidates = pd.DataFrame()
    rural_farm_population_candidates = pd.DataFrame()
else:
    land_use_normalized = qc_rows[land_use_col].map(normalize_lower)
    unit_normalized = qc_rows[unit_col].map(normalize_lower)

    total_area_mask = land_use_normalized == TOTAL_FARM_AREA_VALUE
    area_unit_mask = qc_rows[unit_col].map(is_area_unit)
    numeric_value_mask = clean_numeric(qc_rows[value_col]).notna()

    total_farm_area_candidates = qc_rows[
        total_area_mask & area_unit_mask & numeric_value_mask
    ].copy()

    text = row_text(qc_rows)
    rural_farm_population_candidates = qc_rows[
        text.apply(lambda x: any(term in x for term in RURAL_FARM_POPULATION_TERMS))
    ].copy()


candidate_rows = []

for label, df in [
    ("total_farm_area_candidate", total_farm_area_candidates),
    ("rural_farm_population_or_operator_candidate", rural_farm_population_candidates),
]:
    if not df.empty:
        temp = df.copy()
        temp.insert(0, "candidate_group", label)
        candidate_rows.append(temp)

if candidate_rows:
    candidate_rows_out = pd.concat(candidate_rows, ignore_index=True, sort=False)
else:
    candidate_rows_out = pd.DataFrame()

candidate_rows_out.to_csv(OUTPUT_CANDIDATE_ROWS, index=False, encoding="utf-8")


# -----------------------------
# Select preferred total farm area row
# -----------------------------

selected_total_farm_area = pd.DataFrame()

if not total_farm_area_candidates.empty:
    candidates = total_farm_area_candidates.copy()

    candidates["_area_multiplier_to_km2"] = candidates[unit_col].apply(
        lambda x: infer_area_unit_multiplier(x)[0]
    )
    candidates["_area_unit_detected"] = candidates[unit_col].apply(
        lambda x: infer_area_unit_multiplier(x)[1]
    )
    candidates["_preferred_unit_rank"] = candidates[unit_col].apply(preferred_unit_rank)
    candidates["_farm_area_value"] = clean_numeric(candidates[value_col])
    candidates["_farm_area_km2_candidate"] = (
        candidates["_farm_area_value"] * candidates["_area_multiplier_to_km2"]
    )

    candidates = candidates[
        candidates["_farm_area_km2_candidate"].notna()
        & candidates["_source_dguid_key"].isin(base_dguid_set)
    ].copy()

    selected_total_farm_area = (
        candidates.sort_values(
            ["_source_dguid_key", "_preferred_unit_rank"],
            ascending=[True, True],
        )
        .drop_duplicates("_source_dguid_key", keep="first")
        .copy()
    )

selected_total_farm_area.to_csv(OUTPUT_SELECTED_AREA_ROWS, index=False, encoding="utf-8")


# -----------------------------
# Target summaries
# -----------------------------

target_summary_rows = [
    summarize_candidate_set(
        canonical_variable="pct_land_farms",
        original_sovi_code="PCTFARMS92",
        candidate_name="total_farm_area_candidate",
        rows=total_farm_area_candidates,
        value_col=value_col,
        dguid_col=dguid_col,
        geo_col=geo_col,
        unit_col=unit_col,
        note=(
            "Source candidate for percent land in farms. Preferred row: "
            "Land use = Total farm area and Unit of measure = Hectares."
        ),
    ),
    summarize_candidate_set(
        canonical_variable="pct_rural_farm",
        original_sovi_code="PCTRFRM90",
        candidate_name="rural_farm_population_or_operator_candidate",
        rows=rural_farm_population_candidates,
        value_col=value_col,
        dguid_col=dguid_col,
        geo_col=geo_col,
        unit_col=unit_col,
        note=(
            "PCTRFRM90 is a population-share variable. Land-use data is not expected "
            "to provide a direct rural farm population measure."
        ),
    ),
]

target_summary = pd.DataFrame(target_summary_rows)
target_summary.to_csv(OUTPUT_TARGET_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Formula audit for pct_land_farms
# -----------------------------

formula_rows = []
pct_land_farms_possible = False

if not selected_total_farm_area.empty:
    duplicate_selected = int(selected_total_farm_area["_source_dguid_key"].duplicated().sum())

    joined = base.merge(
        selected_total_farm_area[
            [
                "_source_dguid_key",
                geo_col,
                land_use_col,
                unit_col,
                value_col,
                "_area_unit_detected",
                "_farm_area_value",
                "_farm_area_km2_candidate",
            ]
        ],
        left_on="census_division_dguid",
        right_on="_source_dguid_key",
        how="left",
        validate="one_to_one",
    )

    joined["pct_land_farms_candidate"] = (
        100 * joined["_farm_area_km2_candidate"] / joined["land_area_km2"]
    )

    values = clean_numeric(joined["pct_land_farms_candidate"])
    farm_area_values = clean_numeric(joined["_farm_area_km2_candidate"])

    non_missing = int(values.notna().sum())
    pct_land_farms_possible = non_missing == EXPECTED_QC_CD_COUNT

    formula_status = (
        "formula_candidate_available_full_coverage"
        if pct_land_farms_possible
        else "formula_candidate_partial_or_needs_review"
    )

    formula_rows.append(
        {
            "candidate_formula": "100 * total_farm_area_km2 / land_area_km2",
            "candidate_source_table": "32-10-0249-01",
            "candidate_source_file": safe_relative(RAW_LAND_USE_CSV),
            "selected_land_use_value": TOTAL_FARM_AREA_VALUE,
            "preferred_unit_rule": "prefer Hectares, then Acres, then km2",
            "selected_unit_values": " | ".join(sorted(joined[unit_col].dropna().astype(str).unique())),
            "duplicate_selected_rows_by_dguid": duplicate_selected,
            "non_missing": non_missing,
            "missing": int(values.isna().sum()),
            "farm_area_km2_min": farm_area_values.min(skipna=True),
            "farm_area_km2_max": farm_area_values.max(skipna=True),
            "pct_land_farms_min": values.min(skipna=True),
            "pct_land_farms_max": values.max(skipna=True),
            "pct_land_farms_mean": values.mean(skipna=True),
            "pct_land_farms_median": values.median(skipna=True),
            "pct_land_farms_values_over_100": int((values > 100).sum()),
            "formula_status": formula_status,
            "recommended_default_without_review": bool(
                pct_land_farms_possible
                and duplicate_selected == 0
                and int((values > 100).sum()) == 0
            ),
            "interpretation": (
                "Uses Census of Agriculture Total farm area with area units. "
                "If full coverage and values are plausible, this is suitable for pct_land_farms."
            ),
        }
    )

    unmatched = joined[joined["_farm_area_km2_candidate"].isna()][
        [
            "census_division_code",
            "census_division_dguid",
            "census_division_name",
        ]
    ].copy()

else:
    formula_rows.append(
        {
            "candidate_formula": "100 * total_farm_area_km2 / land_area_km2",
            "candidate_source_table": "32-10-0249-01",
            "candidate_source_file": safe_relative(RAW_LAND_USE_CSV),
            "selected_land_use_value": TOTAL_FARM_AREA_VALUE,
            "preferred_unit_rule": "prefer Hectares, then Acres, then km2",
            "selected_unit_values": "",
            "duplicate_selected_rows_by_dguid": "",
            "non_missing": 0,
            "missing": EXPECTED_QC_CD_COUNT,
            "farm_area_km2_min": None,
            "farm_area_km2_max": None,
            "pct_land_farms_min": None,
            "pct_land_farms_max": None,
            "pct_land_farms_mean": None,
            "pct_land_farms_median": None,
            "pct_land_farms_values_over_100": None,
            "formula_status": "no_total_farm_area_area_candidate_found",
            "recommended_default_without_review": False,
            "interpretation": "No total farm area candidate could be constructed from this file.",
        }
    )

    unmatched = base[
        [
            "census_division_code",
            "census_division_dguid",
            "census_division_name",
        ]
    ].copy()

formula_audit = pd.DataFrame(formula_rows)
formula_audit.to_csv(OUTPUT_FORMULA_AUDIT, index=False, encoding="utf-8")
unmatched.to_csv(OUTPUT_UNMATCHED_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Summary
# -----------------------------

pct_rural_farm_candidate_found = not rural_farm_population_candidates.empty

summary_rows = [
    {"metric": "raw_csv", "value": safe_relative(RAW_LAND_USE_CSV)},
    {
        "metric": "metadata_csv",
        "value": safe_relative(RAW_LAND_USE_METADATA_CSV) if RAW_LAND_USE_METADATA_CSV.exists() else "",
    },
    {"metric": "base_cd_frame", "value": safe_relative(BASE_CD_FRAME)},
    {"metric": "raw_encoding", "value": encoding},
    {"metric": "raw_rows", "value": len(raw)},
    {"metric": "raw_columns", "value": len(raw.columns)},
    {"metric": "base_rows", "value": len(base)},
    {"metric": "detected_dguid_column", "value": dguid_col or ""},
    {"metric": "detected_geo_column", "value": geo_col or ""},
    {"metric": "detected_land_use_column", "value": land_use_col or ""},
    {"metric": "detected_unit_of_measure_column", "value": unit_col or ""},
    {"metric": "detected_uom_code_column", "value": uom_code_col or ""},
    {"metric": "detected_value_column", "value": value_col or ""},
    {"metric": "detected_status_column", "value": status_col or ""},
    {"metric": "quebec_cd_rows_detected", "value": len(qc_rows)},
    {"metric": "total_farm_area_candidate_rows", "value": len(total_farm_area_candidates)},
    {"metric": "selected_total_farm_area_rows", "value": len(selected_total_farm_area)},
    {"metric": "rural_farm_population_or_operator_candidate_rows", "value": len(rural_farm_population_candidates)},
    {"metric": "pct_land_farms_formula_candidate_available", "value": pct_land_farms_possible},
    {"metric": "pct_rural_farm_candidate_found_in_this_file", "value": pct_rural_farm_candidate_found},
    {
        "metric": "important_method_note_pct_land_farms",
        "value": (
            "pct_land_farms is computed from Land use = Total farm area and the human-readable "
            "Unit of measure column. Hectares are preferred over acres because they convert directly to km2."
        ),
    },
    {
        "metric": "important_method_note_pct_rural_farm",
        "value": (
            "PCTRFRM90 is a rural farm population share. This land-use table should not be used as a direct proxy."
        ),
    },
    {
        "metric": "recommended_next_step",
        "value": (
            "Review agriculture_pct_land_farms_formula_audit_2021.csv. If it shows full coverage, selected unit "
            "Hectares, no values over 100, and recommended_default_without_review=True, generate the cleaner for "
            "PCTFARMS92 -> pct_land_farms. Keep PCTRFRM90 unresolved unless a true farm-population source is found."
        ),
    },
]

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CENSUS DIVISION AGRICULTURE LAND USE INSPECTION 2021")
print("=" * 72)

print("\nDetected columns:")
print("DGUID:", dguid_col)
print("GEO:", geo_col)
print("Land use:", land_use_col)
print("Unit of measure:", unit_col)
print("UOM code:", uom_code_col)
print("VALUE:", value_col)
print("STATUS:", status_col)

print("\nRow counts:")
print("Raw rows:", len(raw))
print("Québec CD rows detected:", len(qc_rows))
print("Total farm area candidate rows:", len(total_farm_area_candidates))
print("Selected total farm area rows:", len(selected_total_farm_area))
print("Rural farm population/operator candidate rows:", len(rural_farm_population_candidates))

print("\nTarget summary:")
print(target_summary.to_string(index=False))

print("\nFormula audit:")
print(formula_audit.to_string(index=False))

print("\nSummary:")
print(summary.to_string(index=False))

print("\nSaved:")
print(OUTPUT_RAW_COLUMN_PROFILE)
print(OUTPUT_METADATA_PREVIEW)
print(OUTPUT_QUEBEC_CD_ROWS)
print(OUTPUT_DIMENSION_INVENTORY)
print(OUTPUT_CANDIDATE_ROWS)
print(OUTPUT_SELECTED_AREA_ROWS)
print(OUTPUT_TARGET_SUMMARY)
print(OUTPUT_FORMULA_AUDIT)
print(OUTPUT_UNMATCHED_AUDIT)
print(OUTPUT_SUMMARY)

print("\nDone.")