from pathlib import Path
import codecs
import os
import re
import time
from collections import defaultdict

import pandas as pd


# ============================================================
# Optimized Inspect Census Division Housing Permits 2021
# ============================================================
#
# This version is designed to do the same methodological inspection as the
# original script, but without loading the 8GB raw CSV fully into memory.
#
# Main optimizations:
#   - Reads the raw StatCan CSV in chunks.
#   - Builds geography inventory incrementally.
#   - Builds candidate source rows incrementally.
#   - Builds the raw column profile incrementally.
#   - Prints progress during the long raw-file scan.
#
# Run from data/ after placing this file in:
#
#   data/census_division_housing_permits_2021/
#
# Example:
#
#   python census_division_housing_permits_2021/inspect_census_division_housing_permits_2021_optimized.py
#
# Optional tuning:
#
#   HOUSING_PERMITS_CHUNK_SIZE=100000 python census_division_housing_permits_2021/inspect_census_division_housing_permits_2021_optimized.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_housing_permits_2021"
RAW_DIR = SECTION_DIR / "raw"
OUTPUT_DIR = SECTION_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = RAW_DIR / "building_permits_by_type_structure_work_3410029201.csv"
METADATA_CSV = RAW_DIR / "building_permits_by_type_structure_work_3410029201_MetaData.csv"

BASE_CD_FRAME = (
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.csv"
)

OUTPUT_SUMMARY = OUTPUT_DIR / "housing_permits_inspection_summary_2021.csv"
OUTPUT_RAW_COLUMN_PROFILE = OUTPUT_DIR / "housing_permits_raw_column_profile_2021.csv"
OUTPUT_METADATA_PREVIEW = OUTPUT_DIR / "housing_permits_metadata_preview_2021.csv"
OUTPUT_DIMENSION_INVENTORY = OUTPUT_DIR / "housing_permits_dimension_inventory_2021.csv"
OUTPUT_GEOGRAPHY_INVENTORY = OUTPUT_DIR / "housing_permits_geography_inventory_2021.csv"
OUTPUT_CANDIDATE_SOURCE_ROWS = OUTPUT_DIR / "housing_permits_candidate_source_rows_2021.csv"
OUTPUT_FORMULA_AUDIT = OUTPUT_DIR / "housing_permits_formula_audit_2021.csv"
OUTPUT_TARGET_SUMMARY = OUTPUT_DIR / "housing_permits_target_summary_2021.csv"
OUTPUT_UNMATCHED_AUDIT = OUTPUT_DIR / "housing_permits_unmatched_cd_audit_2021.csv"


# -----------------------------
# Config
# -----------------------------

EXPECTED_QC_CD_COUNT = 98
TARGET_YEAR = "2021"

CHUNK_SIZE = int(os.environ.get("HOUSING_PERMITS_CHUNK_SIZE", "250000"))
PROGRESS_EVERY_CHUNKS = int(os.environ.get("HOUSING_PERMITS_PROGRESS_EVERY_CHUNKS", "5"))

ENCODING_CANDIDATES = [
    "utf-8-sig",
    "utf-8",
    "cp1252",
    "latin1",
]

CANDIDATES = [
    {
        "candidate_alias": "dwelling_units_created_total_residential_new_units",
        "canonical_variable": "housing_permit_density",
        "original_sovi_code": "HUPTDEN90",
        "type_of_building": "Total residential",
        "type_of_work": "New dwelling units total",
        "variable": "Number of dwelling-units created",
        "seasonal_adjustment_value_type": "Unadjusted, current",
        "candidate_formula": "annual_2021_dwelling_units_created / land_area_km2",
        "interpretation": (
            "Preferred conceptual candidate if geography is usable: number of residential dwelling units "
            "created by building permits in 2021 per square kilometre."
        ),
    },
    {
        "candidate_alias": "permits_total_residential_types_work_total",
        "canonical_variable": "housing_permit_density",
        "original_sovi_code": "HUPTDEN90",
        "type_of_building": "Total residential",
        "type_of_work": "Types of work, total",
        "variable": "Number of permits",
        "seasonal_adjustment_value_type": "Unadjusted, current",
        "candidate_formula": "annual_2021_residential_permits / land_area_km2",
        "interpretation": (
            "Residential permit-count candidate using all residential work types. Broader than new construction."
        ),
    },
    {
        "candidate_alias": "permits_total_residential_new_construction",
        "canonical_variable": "housing_permit_density",
        "original_sovi_code": "HUPTDEN90",
        "type_of_building": "Total residential",
        "type_of_work": "New construction",
        "variable": "Number of permits",
        "seasonal_adjustment_value_type": "Unadjusted, current",
        "candidate_formula": "annual_2021_new_residential_construction_permits / land_area_km2",
        "interpretation": (
            "Narrower permit-count candidate for new residential construction."
        ),
    },
    {
        "candidate_alias": "dwelling_units_created_total_residential_types_work_total",
        "canonical_variable": "housing_permit_density",
        "original_sovi_code": "HUPTDEN90",
        "type_of_building": "Total residential",
        "type_of_work": "Types of work, total",
        "variable": "Number of dwelling-units created",
        "seasonal_adjustment_value_type": "Unadjusted, current",
        "candidate_formula": "annual_2021_dwelling_units_created_all_work / land_area_km2",
        "interpretation": (
            "Broader dwelling-units-created candidate using all residential work types."
        ),
    },
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
    """Full-file strict detection. Used only for smaller files."""
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


def detect_file_encoding_sampled(path: Path, encodings: list[str]) -> str:
    """
    Sample-based encoding detection for huge files.

    The original script scans the whole 8GB file before pandas starts.
    This version samples the beginning and end of the file, which is much faster.
    For StatCan CSVs, utf-8-sig/utf-8 is normally sufficient, and latin1 is a safe fallback.
    """
    sample_head_bytes = int(os.environ.get("HOUSING_PERMITS_ENCODING_HEAD_BYTES", str(64 * 1024 * 1024)))
    sample_tail_bytes = int(os.environ.get("HOUSING_PERMITS_ENCODING_TAIL_BYTES", str(8 * 1024 * 1024)))

    size = path.stat().st_size
    with path.open("rb") as f:
        head = f.read(min(sample_head_bytes, size))

    tail = b""
    if size > len(head):
        with path.open("rb") as f:
            f.seek(max(0, size - sample_tail_bytes))
            tail = f.read(sample_tail_bytes)

    sample = head + tail

    for encoding in encodings:
        try:
            sample.decode(encoding, errors="strict")
            return encoding
        except UnicodeDecodeError:
            continue

    return "latin1"


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


def extract_cd_code_from_cd_dguid(value: object) -> str:
    text = normalize_text(value)
    match = re.search(r"A0003(\d{4})", text)
    if match:
        return match.group(1)
    return ""


def extract_csd_sgc_from_dguid(value: object) -> str:
    text = normalize_text(value)
    match = re.search(r"A0005(\d{7})", text)
    if match:
        return match.group(1)
    return ""


def extract_cd_code_from_csd_sgc(value: object) -> str:
    text = normalize_text(value)
    if re.fullmatch(r"\d{7}", text):
        return text[:4]
    return ""


def guess_geography_level(dguid: object, geo: object) -> str:
    dguid_text = normalize_text(dguid)
    geo_text = normalize_lower(geo)

    if re.search(r"A0003\d{4}", dguid_text):
        return "census_division"
    if re.search(r"A0005\d{7}", dguid_text):
        return "census_subdivision"
    if re.search(r"A0002\d{2}", dguid_text):
        return "province_or_territory"
    if "census metropolitan area" in geo_text:
        return "cma_or_cma_total"
    if "census agglomeration" in geo_text:
        return "census_agglomeration"
    if "quebec - total census metropolitan area" in geo_text:
        return "province_total_cma"
    if geo_text == "quebec":
        return "province_or_territory"
    if "canada" == geo_text:
        return "canada"
    return "other_or_unknown"


def is_quebec_relevant(dguid: object, geo: object) -> bool:
    dguid_text = normalize_text(dguid)
    geo_text = normalize_lower(geo)

    if re.search(r"A000224\b", dguid_text):
        return True
    if re.search(r"A000324\d{2}", dguid_text):
        return True
    if re.search(r"A000524\d{5}", dguid_text):
        return True
    if "quebec" in geo_text or "québec" in geo_text:
        return True
    if "québec part" in geo_text or "quebec part" in geo_text:
        return True

    return False


def exact_normalized_match(series: pd.Series, target: str) -> pd.Series:
    return series.map(normalize_lower) == normalize_lower(target)


def new_raw_profile_state() -> dict:
    return {
        "dtype_as_loaded": "",
        "non_missing": 0,
        "missing": 0,
        "unique_values": set(),
        "sample_values": [],
        "sample_values_seen": set(),
        "numeric_non_missing": 0,
        "numeric_min": None,
        "numeric_max": None,
        "numeric_sum": 0.0,
    }


def update_raw_profile_state(state: dict, series: pd.Series) -> None:
    if not state["dtype_as_loaded"]:
        state["dtype_as_loaded"] = str(series.dtype)

    state["non_missing"] += int(series.notna().sum())
    state["missing"] += int(series.isna().sum())

    values = series.dropna().astype(str)

    if not values.empty:
        unique_chunk_values = values.drop_duplicates()
        state["unique_values"].update(unique_chunk_values.tolist())

        if len(state["sample_values"]) < 15:
            for value in unique_chunk_values.tolist():
                if value not in state["sample_values_seen"]:
                    state["sample_values"].append(value)
                    state["sample_values_seen"].add(value)
                if len(state["sample_values"]) >= 15:
                    break

    numeric = clean_numeric(series)
    numeric_values = numeric.dropna()

    if not numeric_values.empty:
        n = int(numeric_values.shape[0])
        current_min = numeric_values.min(skipna=True)
        current_max = numeric_values.max(skipna=True)

        state["numeric_non_missing"] += n
        state["numeric_sum"] += float(numeric_values.sum(skipna=True))

        if state["numeric_min"] is None or current_min < state["numeric_min"]:
            state["numeric_min"] = current_min

        if state["numeric_max"] is None or current_max > state["numeric_max"]:
            state["numeric_max"] = current_max


def new_dimension_state() -> dict:
    return {
        "unique_values": set(),
        "sample_values": [],
        "sample_values_seen": set(),
    }


def update_dimension_state(state: dict, series: pd.Series) -> None:
    values = series.dropna().astype(str)

    if values.empty:
        return

    unique_chunk_values = values.drop_duplicates()
    state["unique_values"].update(unique_chunk_values.tolist())

    if len(state["sample_values"]) < 40:
        for value in unique_chunk_values.tolist():
            if value not in state["sample_values_seen"]:
                state["sample_values"].append(value)
                state["sample_values_seen"].add(value)
            if len(state["sample_values"]) >= 40:
                break


def min_text(a: object, b: object) -> object:
    if pd.isna(a) or a == "":
        return b
    if pd.isna(b) or b == "":
        return a
    return min(str(a), str(b))


def max_text(a: object, b: object) -> object:
    if pd.isna(a) or a == "":
        return b
    if pd.isna(b) or b == "":
        return a
    return max(str(a), str(b))


def update_geography_inventory_state(
    state: dict,
    chunk: pd.DataFrame,
    geo_col: str,
    dguid_col: str,
    ref_date_col: str,
    value_col: str,
) -> None:
    grouped = (
        chunk.groupby([geo_col, dguid_col], dropna=False)
        .agg(
            n_rows=(value_col, "size"),
            first_ref_date=(ref_date_col, "min"),
            last_ref_date=(ref_date_col, "max"),
        )
        .reset_index()
    )

    for _, row in grouped.iterrows():
        geo = row[geo_col]
        dguid = row[dguid_col]
        key = (
            "" if pd.isna(geo) else str(geo),
            "" if pd.isna(dguid) else str(dguid),
        )

        if key not in state:
            state[key] = {
                "n_rows": 0,
                "first_ref_date": "",
                "last_ref_date": "",
            }

        state[key]["n_rows"] += int(row["n_rows"])
        state[key]["first_ref_date"] = min_text(state[key]["first_ref_date"], row["first_ref_date"])
        state[key]["last_ref_date"] = max_text(state[key]["last_ref_date"], row["last_ref_date"])


def build_candidate_rows(
    raw_2021: pd.DataFrame,
    candidate: dict,
    building_col: str,
    work_col: str,
    variables_col: str,
    seasonal_col: str,
) -> pd.DataFrame:
    mask = (
        exact_normalized_match(raw_2021[building_col], candidate["type_of_building"])
        & exact_normalized_match(raw_2021[work_col], candidate["type_of_work"])
        & exact_normalized_match(raw_2021[variables_col], candidate["variable"])
        & exact_normalized_match(raw_2021[seasonal_col], candidate["seasonal_adjustment_value_type"])
    )

    out = raw_2021[mask].copy()
    out.insert(0, "candidate_alias", candidate["candidate_alias"])
    out.insert(1, "candidate_formula", candidate["candidate_formula"])
    out.insert(2, "candidate_interpretation", candidate["interpretation"])
    return out


def make_geo_inventory_dataframe(
    geo_state: dict,
    geo_col: str,
    dguid_col: str,
) -> pd.DataFrame:
    rows = []
    for (geo, dguid), values in geo_state.items():
        rows.append(
            {
                geo_col: geo,
                dguid_col: dguid,
                "n_rows": values["n_rows"],
                "first_ref_date": values["first_ref_date"],
                "last_ref_date": values["last_ref_date"],
            }
        )

    geo_inventory = pd.DataFrame(rows)

    if geo_inventory.empty:
        geo_inventory = pd.DataFrame(
            columns=[
                geo_col,
                dguid_col,
                "n_rows",
                "first_ref_date",
                "last_ref_date",
                "_geography_level_guess",
                "_is_quebec_relevant",
                "_cd_code_from_cd_dguid",
                "_csd_sgc_from_dguid",
                "_cd_code_from_csd_dguid",
            ]
        )
        return geo_inventory

    geo_inventory["_geography_level_guess"] = geo_inventory.apply(
        lambda row: guess_geography_level(row[dguid_col], row[geo_col]),
        axis=1,
    )
    geo_inventory["_is_quebec_relevant"] = geo_inventory.apply(
        lambda row: is_quebec_relevant(row[dguid_col], row[geo_col]),
        axis=1,
    )
    geo_inventory["_cd_code_from_cd_dguid"] = geo_inventory[dguid_col].map(extract_cd_code_from_cd_dguid)
    geo_inventory["_csd_sgc_from_dguid"] = geo_inventory[dguid_col].map(extract_csd_sgc_from_dguid)
    geo_inventory["_cd_code_from_csd_dguid"] = geo_inventory["_csd_sgc_from_dguid"].map(extract_cd_code_from_csd_sgc)

    geo_inventory = geo_inventory.sort_values(
        ["_is_quebec_relevant", "_geography_level_guess", geo_col],
        ascending=[False, True, True],
    )

    return geo_inventory


def make_raw_column_profile_dataframe(raw_profile: dict, columns: list[str]) -> pd.DataFrame:
    rows = []

    for col in columns:
        state = raw_profile[col]
        numeric_mean = (
            state["numeric_sum"] / state["numeric_non_missing"]
            if state["numeric_non_missing"] > 0
            else pd.NA
        )

        rows.append(
            {
                "column": col,
                "dtype_as_loaded": state["dtype_as_loaded"],
                "non_missing": state["non_missing"],
                "missing": state["missing"],
                "unique_values": len(state["unique_values"]),
                "sample_values": " | ".join(state["sample_values"]),
                "numeric_non_missing": state["numeric_non_missing"],
                "numeric_min": state["numeric_min"] if state["numeric_non_missing"] > 0 else pd.NA,
                "numeric_max": state["numeric_max"] if state["numeric_non_missing"] > 0 else pd.NA,
                "numeric_mean": numeric_mean,
            }
        )

    return pd.DataFrame(rows)


def make_dimension_inventory_dataframe(dimension_states: dict, dimension_cols: list[str]) -> pd.DataFrame:
    rows = []

    for col in dimension_cols:
        state = dimension_states[col]
        sample_values = " | ".join(state["sample_values"])
        lower_sample = sample_values.lower()

        rows.append(
            {
                "column": col,
                "unique_values": len(state["unique_values"]),
                "sample_values": sample_values,
                "sample_contains_quebec": "quebec" in lower_sample or "québec" in lower_sample,
                "sample_contains_census_division_hint": "census division" in lower_sample or "a0003" in lower_sample,
                "sample_contains_census_subdivision_hint": "census subdivision" in lower_sample or "a0005" in lower_sample,
                "sample_contains_total_residential": "total residential" in lower_sample,
                "sample_contains_new_dwelling_units": "new dwelling units" in lower_sample,
                "sample_contains_number_of_permits": "number of permits" in lower_sample,
                "sample_contains_number_dwelling_units_created": "number of dwelling-units created" in lower_sample,
            }
        )

    return pd.DataFrame(rows)


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

if base["census_division_dguid"].duplicated().any():
    raise ValueError("Duplicate census_division_dguid values in base frame.")

if base["land_area_km2"].isna().any() or (base["land_area_km2"] <= 0).any():
    raise ValueError("Missing or non-positive land_area_km2 values in base frame.")

base_cd_codes = set(base["census_division_code"].dropna().astype(str))
base_cd_dguids = set(base["census_division_dguid"].dropna().astype(str))


# -----------------------------
# Metadata preview
# -----------------------------

if METADATA_CSV.exists():
    metadata_encoding = detect_file_encoding_strict(METADATA_CSV, ENCODING_CANDIDATES)
    metadata = pd.read_csv(METADATA_CSV, encoding=metadata_encoding, dtype=str, low_memory=False)
    metadata.head(250).to_csv(OUTPUT_METADATA_PREVIEW, index=False, encoding="utf-8")
else:
    metadata_encoding = ""
    pd.DataFrame().to_csv(OUTPUT_METADATA_PREVIEW, index=False, encoding="utf-8")


# -----------------------------
# Stream raw table
# -----------------------------

raw_encoding = detect_file_encoding_sampled(RAW_CSV, ENCODING_CANDIDATES)

print("\n" + "=" * 72, flush=True)
print("OPTIMIZED CENSUS DIVISION HOUSING PERMITS INSPECTION 2021", flush=True)
print("=" * 72, flush=True)
print(f"Raw CSV: {safe_relative(RAW_CSV)}", flush=True)
print(f"Raw encoding: {raw_encoding}", flush=True)
print(f"Chunk size: {CHUNK_SIZE:,}", flush=True)
print("Streaming raw CSV...", flush=True)

reader = pd.read_csv(
    RAW_CSV,
    encoding=raw_encoding,
    dtype=str,
    low_memory=False,
    chunksize=CHUNK_SIZE,
)

try:
    first_chunk = next(reader)
except StopIteration:
    raise ValueError(f"Raw CSV is empty: {RAW_CSV}")

first_chunk.columns = [str(col).strip() for col in first_chunk.columns]
columns = list(first_chunk.columns)

ref_date_col = require_column(columns, ["REF_DATE", "Reference period", "Reference date"], "reference period")
geo_col = require_column(columns, ["GEO", "Geography"], "geography")
dguid_col = require_column(columns, ["DGUID", "DGUID: DGUID"], "DGUID")
building_col = require_column(columns, ["Type of building", "Type of structure"], "type of building")
work_col = require_column(columns, ["Type of work", "Types of work"], "type of work")
variables_col = require_column(columns, ["Variables", "Variable"], "variables")
seasonal_col = require_column(
    columns,
    ["Seasonal adjustment, value type", "Seasonal adjustment", "Value type"],
    "seasonal adjustment / value type",
)
value_col = require_column(columns, ["VALUE", "Value"], "value")

status_col = find_column(columns, ["STATUS", "Status"])
symbol_col = find_column(columns, ["SYMBOL", "Symbol"])
uom_col = find_column(columns, ["UOM", "Unit of measure"])

dimension_cols = [
    ref_date_col,
    geo_col,
    dguid_col,
    building_col,
    work_col,
    variables_col,
    seasonal_col,
]

if status_col:
    dimension_cols.append(status_col)
if symbol_col:
    dimension_cols.append(symbol_col)
if uom_col:
    dimension_cols.append(uom_col)

raw_profile = defaultdict(new_raw_profile_state)
dimension_states = defaultdict(new_dimension_state)
geo_state = {}

candidate_frames = []

raw_rows = 0
raw_columns = len(columns)
target_year_rows = 0
direct_cd_rows_total = 0
csd_rows_total = 0
raw_names_with_mojibake = 0

start_time = time.time()


def process_chunk(chunk: pd.DataFrame, chunk_number: int) -> None:
    global raw_rows
    global target_year_rows
    global direct_cd_rows_total
    global csd_rows_total
    global raw_names_with_mojibake

    chunk.columns = [str(col).strip() for col in chunk.columns]

    if list(chunk.columns) != columns:
        raise ValueError(
            f"Chunk {chunk_number} has unexpected columns.\n"
            f"Expected: {columns}\n"
            f"Got: {list(chunk.columns)}"
        )

    # Match the original script: VALUE is converted to numeric before profiling/inventories.
    chunk[value_col] = clean_numeric(chunk[value_col])

    raw_rows += len(chunk)

    direct_cd_rows_total += int(chunk[dguid_col].astype(str).str.contains(r"A0003\d{4}", regex=True, na=False).sum())
    csd_rows_total += int(chunk[dguid_col].astype(str).str.contains(r"A0005\d{7}", regex=True, na=False).sum())

    raw_names_with_mojibake += (
        contains_mojibake(chunk[geo_col])
        + contains_mojibake(chunk[building_col])
        + contains_mojibake(chunk[work_col])
        + contains_mojibake(chunk[variables_col])
    )

    for col in columns:
        update_raw_profile_state(raw_profile[col], chunk[col])

    for col in dimension_cols:
        update_dimension_state(dimension_states[col], chunk[col])

    update_geography_inventory_state(
        state=geo_state,
        chunk=chunk,
        geo_col=geo_col,
        dguid_col=dguid_col,
        ref_date_col=ref_date_col,
        value_col=value_col,
    )

    raw_2021 = chunk[chunk[ref_date_col].astype(str).str.startswith(TARGET_YEAR)].copy()
    target_year_rows += len(raw_2021)

    if not raw_2021.empty:
        for candidate in CANDIDATES:
            candidate_rows = build_candidate_rows(
                raw_2021=raw_2021,
                candidate=candidate,
                building_col=building_col,
                work_col=work_col,
                variables_col=variables_col,
                seasonal_col=seasonal_col,
            )
            if not candidate_rows.empty:
                candidate_frames.append(candidate_rows)


process_chunk(first_chunk, 1)

for chunk_number, chunk in enumerate(reader, start=2):
    process_chunk(chunk, chunk_number)

    if chunk_number % PROGRESS_EVERY_CHUNKS == 0:
        elapsed = time.time() - start_time
        rows_per_second = raw_rows / elapsed if elapsed > 0 else 0
        print(
            f"[progress] chunks={chunk_number:,} rows={raw_rows:,} "
            f"geographies={len(geo_state):,} candidate_rows={sum(len(x) for x in candidate_frames):,} "
            f"elapsed={elapsed/60:.1f} min speed={rows_per_second:,.0f} rows/s",
            flush=True,
        )

elapsed = time.time() - start_time
print(
    f"[progress] finished raw stream: rows={raw_rows:,}, elapsed={elapsed/60:.1f} min",
    flush=True,
)


# -----------------------------
# Materialize streamed outputs
# -----------------------------

raw_column_profile = make_raw_column_profile_dataframe(raw_profile, columns)
raw_column_profile.to_csv(OUTPUT_RAW_COLUMN_PROFILE, index=False, encoding="utf-8")

dimension_inventory = make_dimension_inventory_dataframe(dimension_states, dimension_cols)
dimension_inventory.to_csv(OUTPUT_DIMENSION_INVENTORY, index=False, encoding="utf-8")

geo_inventory = make_geo_inventory_dataframe(geo_state, geo_col, dguid_col)
geo_inventory.to_csv(OUTPUT_GEOGRAPHY_INVENTORY, index=False, encoding="utf-8")

candidate_output_columns = ["candidate_alias", "candidate_formula", "candidate_interpretation"] + columns

if candidate_frames:
    candidate_source_rows = pd.concat(candidate_frames, ignore_index=True, sort=False)
else:
    candidate_source_rows = pd.DataFrame(columns=candidate_output_columns)

if not candidate_source_rows.empty:
    candidate_source_rows["_geography_level_guess"] = candidate_source_rows.apply(
        lambda row: guess_geography_level(row[dguid_col], row[geo_col]),
        axis=1,
    )
    candidate_source_rows["_is_quebec_relevant"] = candidate_source_rows.apply(
        lambda row: is_quebec_relevant(row[dguid_col], row[geo_col]),
        axis=1,
    )
    candidate_source_rows["_cd_code_from_cd_dguid"] = candidate_source_rows[dguid_col].map(extract_cd_code_from_cd_dguid)
    candidate_source_rows["_csd_sgc_from_dguid"] = candidate_source_rows[dguid_col].map(extract_csd_sgc_from_dguid)
    candidate_source_rows["_cd_code_from_csd_dguid"] = candidate_source_rows["_csd_sgc_from_dguid"].map(extract_cd_code_from_csd_sgc)

candidate_source_rows.to_csv(OUTPUT_CANDIDATE_SOURCE_ROWS, index=False, encoding="utf-8")


# -----------------------------
# Formula audit
# -----------------------------

formula_rows = []
unmatched_frames = []

for candidate in CANDIDATES:
    alias = candidate["candidate_alias"]
    subset = candidate_source_rows[candidate_source_rows["candidate_alias"] == alias].copy()

    if subset.empty:
        formula_rows.append(
            {
                "candidate_alias": alias,
                "canonical_variable": candidate["canonical_variable"],
                "original_sovi_code": candidate["original_sovi_code"],
                "candidate_formula": candidate["candidate_formula"],
                "candidate_source_rows": 0,
                "annual_2021_source_geographies": 0,
                "annual_2021_quebec_relevant_geographies": 0,
                "direct_cd_rows_available": 0,
                "direct_cd_join_non_missing": 0,
                "csd_rows_available": 0,
                "csd_to_cd_join_non_missing": 0,
                "best_join_method": "",
                "non_missing_after_join": 0,
                "missing_after_join": EXPECTED_QC_CD_COUNT,
                "coverage_is_98_cds": False,
                "density_min": None,
                "density_max": None,
                "density_mean": None,
                "density_median": None,
                "recommended_default_without_review": False,
                "status": "candidate_rows_not_found",
                "interpretation": candidate["interpretation"],
            }
        )
        continue

    annual = (
        subset.groupby([geo_col, dguid_col], dropna=False)[value_col]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={value_col: "annual_2021_value"})
    )

    annual["_geography_level_guess"] = annual.apply(
        lambda row: guess_geography_level(row[dguid_col], row[geo_col]),
        axis=1,
    )
    annual["_is_quebec_relevant"] = annual.apply(
        lambda row: is_quebec_relevant(row[dguid_col], row[geo_col]),
        axis=1,
    )
    annual["_cd_code_from_cd_dguid"] = annual[dguid_col].map(extract_cd_code_from_cd_dguid)
    annual["_csd_sgc_from_dguid"] = annual[dguid_col].map(extract_csd_sgc_from_dguid)
    annual["_cd_code_from_csd_dguid"] = annual["_csd_sgc_from_dguid"].map(extract_cd_code_from_csd_sgc)

    # Direct CD join, if direct census-division DGUID rows exist.
    direct_cd = annual[annual[dguid_col].isin(base_cd_dguids)].copy()

    direct_join = base[
        ["census_division_code", "census_division_dguid", "census_division_name", "land_area_km2"]
    ].merge(
        direct_cd[[dguid_col, "annual_2021_value"]].rename(columns={dguid_col: "census_division_dguid"}),
        on="census_division_dguid",
        how="left",
        validate="one_to_one",
    )

    direct_join["candidate_density_per_km2"] = direct_join["annual_2021_value"] / direct_join["land_area_km2"]

    direct_non_missing = int(direct_join["candidate_density_per_km2"].notna().sum())

    # CSD-to-CD aggregation, if CSD rows exist.
    csd_rows = annual[
        annual["_csd_sgc_from_dguid"].astype(str).str.fullmatch(r"24\d{5}", na=False)
    ].copy()

    if not csd_rows.empty:
        csd_aggregated = (
            csd_rows.groupby("_cd_code_from_csd_dguid", dropna=False)["annual_2021_value"]
            .sum(min_count=1)
            .reset_index()
            .rename(
                columns={
                    "_cd_code_from_csd_dguid": "census_division_code",
                    "annual_2021_value": "annual_2021_value",
                }
            )
        )
    else:
        csd_aggregated = pd.DataFrame(columns=["census_division_code", "annual_2021_value"])

    csd_join = base[
        ["census_division_code", "census_division_dguid", "census_division_name", "land_area_km2"]
    ].merge(
        csd_aggregated,
        on="census_division_code",
        how="left",
        validate="one_to_one",
    )

    csd_join["candidate_density_per_km2"] = csd_join["annual_2021_value"] / csd_join["land_area_km2"]

    csd_non_missing = int(csd_join["candidate_density_per_km2"].notna().sum())

    if direct_non_missing >= csd_non_missing:
        best_join = direct_join
        best_join_method = "direct_cd_dguid"
        best_non_missing = direct_non_missing
    else:
        best_join = csd_join
        best_join_method = "csd_to_cd_aggregation"
        best_non_missing = csd_non_missing

    density = clean_numeric(best_join["candidate_density_per_km2"])
    coverage_is_98 = best_non_missing == EXPECTED_QC_CD_COUNT

    status = "ready_for_cleaner_candidate_full_coverage" if coverage_is_98 else "not_full_cd_coverage"

    unmatched = best_join[best_join["candidate_density_per_km2"].isna()][
        ["census_division_code", "census_division_dguid", "census_division_name"]
    ].copy()
    unmatched.insert(0, "candidate_alias", alias)
    unmatched.insert(1, "best_join_method", best_join_method)
    unmatched_frames.append(unmatched)

    formula_rows.append(
        {
            "candidate_alias": alias,
            "canonical_variable": candidate["canonical_variable"],
            "original_sovi_code": candidate["original_sovi_code"],
            "candidate_formula": candidate["candidate_formula"],
            "type_of_building": candidate["type_of_building"],
            "type_of_work": candidate["type_of_work"],
            "variable": candidate["variable"],
            "seasonal_adjustment_value_type": candidate["seasonal_adjustment_value_type"],
            "candidate_source_rows": len(subset),
            "annual_2021_source_geographies": annual[[geo_col, dguid_col]].drop_duplicates().shape[0],
            "annual_2021_quebec_relevant_geographies": int(annual["_is_quebec_relevant"].sum()),
            "direct_cd_rows_available": len(direct_cd),
            "direct_cd_join_non_missing": direct_non_missing,
            "csd_rows_available": len(csd_rows),
            "csd_to_cd_join_non_missing": csd_non_missing,
            "best_join_method": best_join_method,
            "non_missing_after_join": best_non_missing,
            "missing_after_join": EXPECTED_QC_CD_COUNT - best_non_missing,
            "coverage_is_98_cds": coverage_is_98,
            "annual_value_min": clean_numeric(best_join["annual_2021_value"]).min(skipna=True),
            "annual_value_max": clean_numeric(best_join["annual_2021_value"]).max(skipna=True),
            "annual_value_mean": clean_numeric(best_join["annual_2021_value"]).mean(skipna=True),
            "annual_value_median": clean_numeric(best_join["annual_2021_value"]).median(skipna=True),
            "density_min": density.min(skipna=True),
            "density_max": density.max(skipna=True),
            "density_mean": density.mean(skipna=True),
            "density_median": density.median(skipna=True),
            "recommended_default_without_review": False,
            "status": status,
            "interpretation": candidate["interpretation"],
        }
    )

formula_audit = pd.DataFrame(formula_rows)
formula_audit.to_csv(OUTPUT_FORMULA_AUDIT, index=False, encoding="utf-8")

if unmatched_frames:
    unmatched_audit = pd.concat(unmatched_frames, ignore_index=True, sort=False)
else:
    unmatched_audit = pd.DataFrame(
        columns=[
            "candidate_alias",
            "best_join_method",
            "census_division_code",
            "census_division_dguid",
            "census_division_name",
        ]
    )

unmatched_audit.to_csv(OUTPUT_UNMATCHED_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Target summary
# -----------------------------

ready_candidates = formula_audit[formula_audit["coverage_is_98_cds"] == True].copy()

if ready_candidates.empty:
    target_summary = pd.DataFrame(
        [
            {
                "canonical_variable": "housing_permit_density",
                "original_sovi_code": "HUPTDEN90",
                "candidate_found": False,
                "best_candidate_alias": "",
                "best_candidate_formula": "",
                "coverage_is_98_cds": False,
                "status": "no_full_cd_coverage_candidate",
                "interpretation": (
                    "No full-coverage census-division candidate was confirmed. The public table likely provides "
                    "province/CMA/CMA-part geography, not direct CD or aggregable CSD geography."
                ),
            }
        ]
    )
else:
    preferred_order = [
        "dwelling_units_created_total_residential_new_units",
        "permits_total_residential_new_construction",
        "permits_total_residential_types_work_total",
        "dwelling_units_created_total_residential_types_work_total",
    ]

    ready_candidates["_preferred_rank"] = ready_candidates["candidate_alias"].apply(
        lambda x: preferred_order.index(x) if x in preferred_order else 999
    )
    best = ready_candidates.sort_values(["_preferred_rank", "candidate_alias"]).iloc[0]

    target_summary = pd.DataFrame(
        [
            {
                "canonical_variable": "housing_permit_density",
                "original_sovi_code": "HUPTDEN90",
                "candidate_found": True,
                "best_candidate_alias": best["candidate_alias"],
                "best_candidate_formula": best["candidate_formula"],
                "coverage_is_98_cds": True,
                "status": "candidate_available_needs_review",
                "interpretation": (
                    "At least one full-coverage CD candidate exists. Review whether the selected measure should "
                    "be dwelling units created or number of permits before generating the cleaner."
                ),
            }
        ]
    )

target_summary.to_csv(OUTPUT_TARGET_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Inspection summary
# -----------------------------

quebec_relevant_geo_count = int(geo_inventory["_is_quebec_relevant"].sum())
quebec_direct_cd_geo_count = int(
    geo_inventory["_cd_code_from_cd_dguid"].astype(str).str.fullmatch(r"24\d{2}", na=False).sum()
)
quebec_csd_geo_count = int(
    geo_inventory["_csd_sgc_from_dguid"].astype(str).str.fullmatch(r"24\d{5}", na=False).sum()
)

base_names_with_mojibake = contains_mojibake(base["census_division_name"])

summary_rows = [
    {"metric": "raw_csv", "value": safe_relative(RAW_CSV)},
    {"metric": "metadata_csv", "value": safe_relative(METADATA_CSV) if METADATA_CSV.exists() else ""},
    {"metric": "base_cd_frame", "value": safe_relative(BASE_CD_FRAME)},
    {"metric": "raw_encoding", "value": raw_encoding},
    {"metric": "metadata_encoding", "value": metadata_encoding},
    {"metric": "base_encoding", "value": base_encoding},
    {"metric": "raw_rows", "value": raw_rows},
    {"metric": "raw_columns", "value": raw_columns},
    {"metric": "base_rows", "value": len(base)},
    {"metric": "detected_ref_date_column", "value": ref_date_col},
    {"metric": "detected_geo_column", "value": geo_col},
    {"metric": "detected_dguid_column", "value": dguid_col},
    {"metric": "detected_type_of_building_column", "value": building_col},
    {"metric": "detected_type_of_work_column", "value": work_col},
    {"metric": "detected_variables_column", "value": variables_col},
    {"metric": "detected_seasonal_column", "value": seasonal_col},
    {"metric": "detected_value_column", "value": value_col},
    {"metric": "detected_status_column", "value": status_col or ""},
    {"metric": "detected_symbol_column", "value": symbol_col or ""},
    {"metric": "detected_uom_column", "value": uom_col or ""},
    {"metric": "target_year", "value": TARGET_YEAR},
    {"metric": "target_year_rows", "value": target_year_rows},
    {"metric": "unique_geographies_total", "value": geo_inventory[[geo_col, dguid_col]].drop_duplicates().shape[0]},
    {"metric": "quebec_relevant_geographies", "value": quebec_relevant_geo_count},
    {"metric": "direct_cd_rows_total", "value": direct_cd_rows_total},
    {"metric": "csd_rows_total", "value": csd_rows_total},
    {"metric": "quebec_direct_cd_geographies", "value": quebec_direct_cd_geo_count},
    {"metric": "quebec_csd_geographies", "value": quebec_csd_geo_count},
    {"metric": "candidate_source_rows", "value": len(candidate_source_rows)},
    {"metric": "full_coverage_candidate_count", "value": len(ready_candidates)},
    {"metric": "housing_permit_density_candidate_ready", "value": not ready_candidates.empty},
    {"metric": "raw_names_with_mojibake", "value": raw_names_with_mojibake},
    {"metric": "base_names_with_mojibake", "value": base_names_with_mojibake},
    {
        "metric": "important_method_note",
        "value": (
            "HUPTDEN90 is inspected using residential permit and dwelling-unit-created measures from StatCan "
            "table 34-10-0292-01. This script only recommends cleaning if direct CD rows or aggregable CSD rows "
            "give 98/98 Québec census-division coverage."
        ),
    },
    {
        "metric": "recommended_next_step",
        "value": (
            "Review housing_permits_target_summary_2021.csv and housing_permits_formula_audit_2021.csv. "
            "If no full CD coverage exists, document this table as a public-source dead end for HUPTDEN90 and "
            "search for a municipality-level building-permits source or leave the variable unresolved."
        ),
    },
]

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CENSUS DIVISION HOUSING PERMITS INSPECTION 2021")
print("=" * 72)

print("\nInputs:")
print("Raw CSV:", safe_relative(RAW_CSV))
print("Raw encoding:", raw_encoding)
print("Metadata CSV:", safe_relative(METADATA_CSV) if METADATA_CSV.exists() else "[missing]")
print("Base frame:", safe_relative(BASE_CD_FRAME))
print("Base encoding:", base_encoding)

print("\nDetected columns:")
print("REF_DATE:", ref_date_col)
print("GEO:", geo_col)
print("DGUID:", dguid_col)
print("Type of building:", building_col)
print("Type of work:", work_col)
print("Variables:", variables_col)
print("Seasonal adjustment/value type:", seasonal_col)
print("VALUE:", value_col)

print("\nGeography:")
print("Unique geographies:", geo_inventory[[geo_col, dguid_col]].drop_duplicates().shape[0])
print("Québec-relevant geographies:", quebec_relevant_geo_count)
print("Québec direct CD geographies:", quebec_direct_cd_geo_count)
print("Québec CSD geographies:", quebec_csd_geo_count)
print("Base CD rows:", len(base))

print("\nCandidate source rows:", len(candidate_source_rows))

print("\nTarget summary:")
print(target_summary.to_string(index=False))

print("\nFormula audit:")
print(formula_audit.to_string(index=False))

print("\nMojibake check:")
print("Raw names with mojibake:", raw_names_with_mojibake)
print("Base names with mojibake:", base_names_with_mojibake)

print("\nSaved:")
print(OUTPUT_SUMMARY)
print(OUTPUT_RAW_COLUMN_PROFILE)
print(OUTPUT_METADATA_PREVIEW)
print(OUTPUT_DIMENSION_INVENTORY)
print(OUTPUT_GEOGRAPHY_INVENTORY)
print(OUTPUT_CANDIDATE_SOURCE_ROWS)
print(OUTPUT_FORMULA_AUDIT)
print(OUTPUT_TARGET_SUMMARY)
print(OUTPUT_UNMATCHED_AUDIT)

print("\nDone.")