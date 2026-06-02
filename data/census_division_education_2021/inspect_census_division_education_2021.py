from pathlib import Path
import re
import pandas as pd


# ============================================================
# Inspect Census Division Education Features 2021
# ============================================================
#
# Purpose:
#   Inspect the 2021 Census Profile at census-division level for education
#   variables needed by the SoVI-like census-division table.
#
# SoVI target variable inspected here:
#
#   PCTNOHSDP90 -> pct_no_high_school_diploma
#
# Likely local proxy:
#
#   No certificate, diploma or degree
#
# This script does NOT create the final clean education table.
# It verifies candidate characteristic IDs, value columns, coverage, symbols,
# and compatibility with the Québec census-division base frame.
#
# Expected raw source:
#   census_profile_census_division_2021/raw/98-401-X2021004_English_CSV_data.csv
#
# Run from data/:
#   python census_division_education_2021/inspect_census_division_education_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_education_2021"
OUTPUT_DIR = SECTION_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = (
    DATA_DIR
    / "census_profile_census_division_2021"
    / "raw"
    / "98-401-X2021004_English_CSV_data.csv"
)

BASE_CD_CANDIDATES = [
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.csv",
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.parquet",
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.geojson",
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.gpkg",
]

OUTPUT_FILE_INVENTORY = OUTPUT_DIR / "education_file_inventory_2021.csv"
OUTPUT_CD_INVENTORY = OUTPUT_DIR / "education_quebec_cd_inventory_2021.csv"
OUTPUT_BASE_JOIN = OUTPUT_DIR / "education_quebec_cd_base_join_check_2021.csv"
OUTPUT_CANDIDATE_CHARACTERISTICS = OUTPUT_DIR / "education_candidate_characteristics_2021.csv"
OUTPUT_TARGET_SUMMARY = OUTPUT_DIR / "education_target_characteristic_summary_2021.csv"
OUTPUT_TARGET_VALUES_LONG = OUTPUT_DIR / "education_target_values_long_2021.csv"
OUTPUT_TARGET_VALUES_WIDE = OUTPUT_DIR / "education_target_values_wide_2021.csv"
OUTPUT_SYMBOL_COUNTS = OUTPUT_DIR / "education_symbol_counts_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "education_inspection_summary_2021.csv"


# -----------------------------
# Encoding config
# -----------------------------

RAW_ENCODING_CANDIDATES = [
    "cp1252",
    "latin1",
    "utf-8-sig",
    "utf-8",
]

CLEANED_FILE_ENCODING_CANDIDATES = [
    "utf-8",
    "utf-8-sig",
    "cp1252",
    "latin1",
]


# -----------------------------
# Other config
# -----------------------------

CHUNK_SIZE = 200_000
QUEBEC_CD_DGUID_PREFIX = "2021A000324"

REQUIRED_COLUMNS = [
    "CENSUS_YEAR",
    "DGUID",
    "ALT_GEO_CODE",
    "GEO_LEVEL",
    "GEO_NAME",
    "CHARACTERISTIC_ID",
    "CHARACTERISTIC_NAME",
    "CHARACTERISTIC_NOTE",
    "C1_COUNT_TOTAL",
    "SYMBOL",
    "C2_COUNT_MEN+",
    "SYMBOL.1",
    "C3_COUNT_WOMEN+",
    "SYMBOL.2",
    "C10_RATE_TOTAL",
    "SYMBOL.3",
    "C11_RATE_MEN+",
    "SYMBOL.4",
    "C12_RATE_WOMEN+",
    "SYMBOL.5",
    "TNR_SF",
    "TNR_LF",
    "DATA_QUALITY_FLAG",
]

VALUE_TO_SYMBOL_COLUMN = {
    "C1_COUNT_TOTAL": "SYMBOL",
    "C2_COUNT_MEN+": "SYMBOL.1",
    "C3_COUNT_WOMEN+": "SYMBOL.2",
    "C10_RATE_TOTAL": "SYMBOL.3",
    "C11_RATE_MEN+": "SYMBOL.4",
    "C12_RATE_WOMEN+": "SYMBOL.5",
}

# Broad range. Education rows usually appear after language/mobility sections
# in Census Profile files, but exact IDs should be confirmed by inspection.
EDUCATION_CONTEXT_MIN_ID = 1800
EDUCATION_CONTEXT_MAX_ID = 2250

EDUCATION_KEYWORDS_ANY = [
    "education",
    "educational",
    "certificate",
    "diploma",
    "degree",
    "high school",
    "secondary",
    "postsecondary",
    "bachelor",
    "university",
    "college",
    "apprenticeship",
    "trades",
    "highest certificate",
]

EDUCATION_KEYWORDS_EXCLUDE = [
    "language",
    "income",
    "employment income",
    "journey to work",
    "commuting",
    "mobility",
    "place of work",
    "occupation",
    "industry",
]

EDUCATION_TARGETS = [
    {
        "original_code": "PCTNOHSDP90",
        "canonical_variable": "pct_no_high_school_diploma",
        "candidate_output_alias": "pct_no_certificate_diploma_or_degree",
        "description": "Percent population with no high school diploma / no certificate, diploma or degree",
        "target_family": "education_attainment",
        "preferred_name_contains_any": [
            "no certificate, diploma or degree",
            "no certificate diploma or degree",
            "no high school diploma",
            "without high school diploma",
        ],
        "required_name_contains_all": [],
        "avoid_name_contains_any": [
            "major field of study",
            "location of study",
            "school attendance",
            "income",
            "employment",
            "occupation",
            "industry",
            "language",
        ],
        "value_column": "C10_RATE_TOTAL",
        "symbol_column": "SYMBOL.3",
        "unit": "percent",
        "sovi_role": "no_high_school_or_no_certificate_proxy",
        "notes": (
            "Original SoVI uses percent persons with no high school diploma. "
            "The Canadian Census Profile may expose the closest broad proxy as "
            "'No certificate, diploma or degree' within highest certificate/diploma/degree."
        ),
    },
]


# -----------------------------
# Helpers
# -----------------------------

def detect_encoding(path: Path, encodings: list[str]) -> str:
    last_error = None

    for encoding in encodings:
        try:
            pd.read_csv(
                path,
                nrows=5000,
                dtype=str,
                encoding=encoding,
                low_memory=False,
            )
            return encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not decode {path} with encodings {encodings}. Last error: {last_error}",
    )


def read_csv_with_encodings(path: Path, encodings: list[str], **kwargs) -> pd.DataFrame:
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                dtype=str,
                encoding=encoding,
                low_memory=False,
                **kwargs,
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error


def read_cleaned_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return read_csv_with_encodings(path, CLEANED_FILE_ENCODING_CANDIDATES)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix in [".geojson", ".gpkg", ".shp"]:
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError(f"geopandas is required to read spatial file {path}") from exc
        return gpd.read_file(path)

    raise ValueError(f"Unsupported file type: {path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_quebec_cd_dguid(series: pd.Series) -> pd.Series:
    return clean_text(series).str.startswith(QUEBEC_CD_DGUID_PREFIX, na=False)


def find_base_cd_frame() -> Path | None:
    for path in BASE_CD_CANDIDATES:
        if path.exists():
            return path
    return None


def require_columns(columns: list[str], required: list[str], label: str) -> None:
    missing = [col for col in required if col not in columns]

    if missing:
        raise ValueError(
            f"Missing required columns in {label}:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(columns)
        )


def is_education_candidate(row: pd.Series) -> bool:
    char_id_raw = str(row.get("CHARACTERISTIC_ID", "")).strip()

    try:
        char_id = int(char_id_raw)
    except ValueError:
        char_id = None

    name = normalize_text(row.get("CHARACTERISTIC_NAME", ""))
    note = normalize_text(row.get("CHARACTERISTIC_NOTE", ""))
    text = f"{name} {note}"

    in_context_range = (
        char_id is not None
        and EDUCATION_CONTEXT_MIN_ID <= char_id <= EDUCATION_CONTEXT_MAX_ID
    )

    keyword_match = any(keyword.lower() in text for keyword in EDUCATION_KEYWORDS_ANY)
    excluded = any(keyword.lower() in text for keyword in EDUCATION_KEYWORDS_EXCLUDE)

    return (in_context_range or keyword_match) and not excluded


def target_matches_row(target: dict, row: pd.Series) -> tuple[bool, str]:
    name = normalize_text(row.get("CHARACTERISTIC_NAME", ""))
    note = normalize_text(row.get("CHARACTERISTIC_NOTE", ""))
    text = f"{name} {note}"

    avoid_terms = [term.lower() for term in target.get("avoid_name_contains_any", [])]
    if any(term in text for term in avoid_terms):
        return False, ""

    required_terms = [term.lower() for term in target.get("required_name_contains_all", [])]
    if required_terms and not all(term in text for term in required_terms):
        return False, ""

    preferred_terms = [term.lower() for term in target.get("preferred_name_contains_any", [])]

    for term in preferred_terms:
        if term in text:
            return True, f"preferred_name_contains:{term}"

    return False, ""


def summarize_symbol_counts(series: pd.Series) -> str:
    counts = series.fillna("").astype(str).value_counts(dropna=False)

    if counts.empty:
        return ""

    return "; ".join(
        f"{symbol}:{int(count)}"
        for symbol, count in counts.items()
    )


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


def build_file_inventory(raw_encoding: str) -> pd.DataFrame:
    rows = []

    rows.append(
        {
            "relative_path": str(RAW_CSV.relative_to(DATA_DIR)),
            "exists": RAW_CSV.exists(),
            "size_mb": round(RAW_CSV.stat().st_size / (1024 * 1024), 3) if RAW_CSV.exists() else None,
            "encoding_selected": raw_encoding if RAW_CSV.exists() else "",
            "role": "raw_census_profile_cd",
        }
    )

    base_path = find_base_cd_frame()
    rows.append(
        {
            "relative_path": str(base_path.relative_to(DATA_DIR)) if base_path is not None else "",
            "exists": base_path is not None,
            "size_mb": round(base_path.stat().st_size / (1024 * 1024), 3) if base_path is not None else None,
            "encoding_selected": "",
            "role": "clean_cd_spatial_population_base",
        }
    )

    return pd.DataFrame(rows)


# -----------------------------
# Initial checks
# -----------------------------

if not RAW_CSV.exists():
    raise FileNotFoundError(f"Raw Census Profile CD CSV not found:\n{RAW_CSV}")

raw_encoding = detect_encoding(RAW_CSV, RAW_ENCODING_CANDIDATES)

header = pd.read_csv(
    RAW_CSV,
    nrows=0,
    dtype=str,
    encoding=raw_encoding,
    low_memory=False,
)

columns = list(header.columns)
require_columns(columns, REQUIRED_COLUMNS, "Census Profile CD raw CSV")

usecols = REQUIRED_COLUMNS

print("\nCensus Profile CD education inspection")
print("Raw file:", RAW_CSV.relative_to(DATA_DIR))
print("Raw encoding selected:", raw_encoding)
print("Education context ID range:", f"{EDUCATION_CONTEXT_MIN_ID}–{EDUCATION_CONTEXT_MAX_ID}")

file_inventory = build_file_inventory(raw_encoding)
file_inventory.to_csv(OUTPUT_FILE_INVENTORY, index=False)


# -----------------------------
# Load base CD frame
# -----------------------------

base_path = find_base_cd_frame()
base = None

if base_path is not None:
    base = read_cleaned_table(base_path)
    base = normalize_columns(base)

    if "census_division_dguid" not in base.columns:
        print("\nWARNING: Base CD frame exists but lacks census_division_dguid.")
        base = None
    else:
        base = base.copy()
        base["census_division_dguid"] = clean_text(base["census_division_dguid"])

        print("\nLoaded existing CD base frame")
        print("Path:", base_path.relative_to(DATA_DIR))
        print("Rows:", len(base))
else:
    print("\nWARNING: No existing CD base frame found.")


# -----------------------------
# Chunked scan
# -----------------------------

total_rows = 0
quebec_cd_rows = 0
quebec_cd_inventory = {}
candidate_rows = []
target_candidate_rows = []

print("\nScanning raw Census Profile file...")

for chunk_idx, chunk in enumerate(
    pd.read_csv(
        RAW_CSV,
        dtype=str,
        encoding=raw_encoding,
        low_memory=False,
        chunksize=CHUNK_SIZE,
        usecols=usecols,
    ),
    start=1,
):
    total_rows += len(chunk)
    chunk = normalize_columns(chunk)

    chunk["GEO_LEVEL_NORM"] = clean_text(chunk["GEO_LEVEL"]).str.lower()

    is_cd = chunk["GEO_LEVEL_NORM"].eq("census division")
    is_qc_cd = is_cd & is_quebec_cd_dguid(chunk["DGUID"])

    qc = chunk.loc[is_qc_cd].copy()
    quebec_cd_rows += len(qc)

    if qc.empty:
        if chunk_idx % 10 == 0:
            print(f"  Processed chunks: {chunk_idx}, rows so far: {total_rows}")
        continue

    qc["DGUID"] = clean_text(qc["DGUID"])
    qc["ALT_GEO_CODE"] = clean_text(qc["ALT_GEO_CODE"])
    qc["GEO_NAME"] = clean_text(qc["GEO_NAME"])
    qc["CHARACTERISTIC_ID"] = clean_text(qc["CHARACTERISTIC_ID"])

    geo_cols = [
        "DGUID",
        "ALT_GEO_CODE",
        "GEO_NAME",
        "GEO_LEVEL",
        "TNR_SF",
        "TNR_LF",
        "DATA_QUALITY_FLAG",
    ]

    for _, row in qc[geo_cols].drop_duplicates(subset=["DGUID"]).iterrows():
        dguid = str(row["DGUID"])
        quebec_cd_inventory[dguid] = {
            "census_division_dguid": dguid,
            "census_division_code": row.get("ALT_GEO_CODE", ""),
            "census_division_name_profile": row.get("GEO_NAME", ""),
            "GEO_LEVEL": row.get("GEO_LEVEL", ""),
            "TNR_SF": row.get("TNR_SF", ""),
            "TNR_LF": row.get("TNR_LF", ""),
            "DATA_QUALITY_FLAG": row.get("DATA_QUALITY_FLAG", ""),
        }

    candidate_mask = qc.apply(is_education_candidate, axis=1)
    candidates = qc.loc[candidate_mask].copy()

    if not candidates.empty:
        candidate_rows.append(candidates)

        characteristic_candidates = candidates[
            [
                "CHARACTERISTIC_ID",
                "CHARACTERISTIC_NAME",
                "CHARACTERISTIC_NOTE",
            ]
        ].drop_duplicates()

        for _, char_row in characteristic_candidates.iterrows():
            for target in EDUCATION_TARGETS:
                matched, match_reason = target_matches_row(target, char_row)
                if not matched:
                    continue

                char_id = str(char_row["CHARACTERISTIC_ID"]).strip()
                rows = qc[qc["CHARACTERISTIC_ID"].eq(char_id)].copy()

                if rows.empty:
                    continue

                rows["target_original_code"] = target["original_code"]
                rows["target_canonical_variable"] = target["canonical_variable"]
                rows["target_candidate_output_alias"] = target["candidate_output_alias"]
                rows["target_family"] = target["target_family"]
                rows["target_match_reason"] = match_reason
                rows["target_value_column"] = target["value_column"]
                rows["target_symbol_column"] = target["symbol_column"]
                rows["target_unit"] = target["unit"]
                rows["target_sovi_role"] = target["sovi_role"]
                rows["target_notes"] = target["notes"]

                target_candidate_rows.append(rows)

    if chunk_idx % 10 == 0:
        print(f"  Processed chunks: {chunk_idx}, rows so far: {total_rows}")


# -----------------------------
# Build dataframes
# -----------------------------

cd_inventory = (
    pd.DataFrame(quebec_cd_inventory.values())
    .sort_values("census_division_dguid")
    .reset_index(drop=True)
    if quebec_cd_inventory
    else pd.DataFrame(
        columns=[
            "census_division_dguid",
            "census_division_code",
            "census_division_name_profile",
            "GEO_LEVEL",
            "TNR_SF",
            "TNR_LF",
            "DATA_QUALITY_FLAG",
        ]
    )
)

candidate_df = (
    pd.concat(candidate_rows, ignore_index=True)
    if candidate_rows
    else pd.DataFrame(columns=usecols)
)

target_raw_df = (
    pd.concat(target_candidate_rows, ignore_index=True)
    if target_candidate_rows
    else pd.DataFrame(columns=usecols)
)

cd_inventory.to_csv(OUTPUT_CD_INVENTORY, index=False)


# -----------------------------
# Base join check
# -----------------------------

if base is not None and not cd_inventory.empty:
    base_join = base[
        [
            col for col in [
                "census_division_code",
                "census_division_dguid",
                "census_division_name",
                "population_total_2021",
                "land_area_km2",
            ]
            if col in base.columns
        ]
    ].copy()

    base_join = base_join.merge(
        cd_inventory[
            [
                "census_division_dguid",
                "census_division_code",
                "census_division_name_profile",
                "TNR_SF",
                "TNR_LF",
                "DATA_QUALITY_FLAG",
            ]
        ].rename(
            columns={
                "census_division_code": "profile_census_division_code",
            }
        ),
        on="census_division_dguid",
        how="outer",
        indicator=True,
    )

    base_join["join_status"] = base_join["_merge"]
    base_join = base_join.drop(columns="_merge")

else:
    base_join = pd.DataFrame()

base_join.to_csv(OUTPUT_BASE_JOIN, index=False)


# -----------------------------
# Candidate characteristic diagnostics
# -----------------------------

candidate_characteristic_rows = []

if not candidate_df.empty:
    candidate_df["CHARACTERISTIC_ID"] = clean_text(candidate_df["CHARACTERISTIC_ID"])

    group_cols = [
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "CHARACTERISTIC_NOTE",
    ]

    for keys, group in candidate_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        key_map = dict(zip(group_cols, keys))

        row = {
            "CHARACTERISTIC_ID": key_map.get("CHARACTERISTIC_ID", ""),
            "CHARACTERISTIC_NAME": key_map.get("CHARACTERISTIC_NAME", ""),
            "CHARACTERISTIC_NOTE": key_map.get("CHARACTERISTIC_NOTE", ""),
            "n_rows": len(group),
            "n_unique_quebec_cds": group["DGUID"].nunique(),
        }

        for value_col in [
            "C1_COUNT_TOTAL",
            "C2_COUNT_MEN+",
            "C3_COUNT_WOMEN+",
            "C10_RATE_TOTAL",
            "C11_RATE_MEN+",
            "C12_RATE_WOMEN+",
        ]:
            summary = numeric_summary(group[value_col])
            prefix = (
                value_col
                .lower()
                .replace("+", "plus")
                .replace(" ", "_")
                .replace("-", "_")
            )

            row[f"{prefix}_non_missing"] = summary["non_missing"]
            row[f"{prefix}_missing"] = summary["missing"]
            row[f"{prefix}_min"] = summary["min"]
            row[f"{prefix}_max"] = summary["max"]
            row[f"{prefix}_mean"] = summary["mean"]
            row[f"{prefix}_median"] = summary["median"]

            symbol_col = VALUE_TO_SYMBOL_COLUMN.get(value_col)
            if symbol_col in group.columns:
                row[f"{prefix}_symbol_counts"] = summarize_symbol_counts(group[symbol_col])

        candidate_characteristic_rows.append(row)

candidate_characteristics = (
    pd.DataFrame(candidate_characteristic_rows)
    .sort_values("CHARACTERISTIC_ID", key=lambda s: pd.to_numeric(s, errors="coerce"))
    .reset_index(drop=True)
    if candidate_characteristic_rows
    else pd.DataFrame()
)

candidate_characteristics.to_csv(OUTPUT_CANDIDATE_CHARACTERISTICS, index=False)


# -----------------------------
# Target summaries and values
# -----------------------------

target_summary_rows = []
target_values_long_rows = []
symbol_count_rows = []

if not target_raw_df.empty:
    target_raw_df["CHARACTERISTIC_ID"] = clean_text(target_raw_df["CHARACTERISTIC_ID"])

    group_cols = [
        "target_original_code",
        "target_canonical_variable",
        "target_candidate_output_alias",
        "target_family",
        "target_match_reason",
        "target_value_column",
        "target_symbol_column",
        "target_unit",
        "target_sovi_role",
        "target_notes",
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "CHARACTERISTIC_NOTE",
    ]

    for keys, rows in target_raw_df.groupby(group_cols, dropna=False):
        key_map = dict(zip(group_cols, keys))

        value_col = key_map["target_value_column"]
        symbol_col = key_map["target_symbol_column"]

        values = clean_numeric(rows[value_col])
        coverage = rows["DGUID"].nunique()

        target_summary_rows.append(
            {
                "original_code": key_map["target_original_code"],
                "canonical_variable": key_map["target_canonical_variable"],
                "candidate_output_alias": key_map["target_candidate_output_alias"],
                "target_family": key_map["target_family"],
                "match_reason": key_map["target_match_reason"],
                "CHARACTERISTIC_ID": key_map["CHARACTERISTIC_ID"],
                "CHARACTERISTIC_NAME": key_map["CHARACTERISTIC_NAME"],
                "CHARACTERISTIC_NOTE": key_map["CHARACTERISTIC_NOTE"],
                "value_column": value_col,
                "symbol_column": symbol_col,
                "unit": key_map["target_unit"],
                "sovi_role": key_map["target_sovi_role"],
                "n_rows": len(rows),
                "n_unique_quebec_cds": coverage,
                "value_non_missing": int(values.notna().sum()),
                "value_missing": int(values.isna().sum()),
                "value_min": values.min(skipna=True),
                "value_max": values.max(skipna=True),
                "value_mean": values.mean(skipna=True),
                "value_median": values.median(skipna=True),
                "coverage_is_98_cds": coverage == 98,
                "status": (
                    "candidate_found_full_coverage"
                    if coverage == 98 and values.notna().sum() == 98
                    else "candidate_needs_review"
                ),
                "notes": key_map["target_notes"],
            }
        )

        for _, row in rows.iterrows():
            value_numeric = clean_numeric(pd.Series([row.get(value_col, "")])).iloc[0]
            selected_symbol = row.get(symbol_col, "")

            target_values_long_rows.append(
                {
                    "original_code": key_map["target_original_code"],
                    "canonical_variable": key_map["target_canonical_variable"],
                    "candidate_output_alias": key_map["target_candidate_output_alias"],
                    "target_family": key_map["target_family"],
                    "match_reason": key_map["target_match_reason"],
                    "census_division_dguid": row.get("DGUID", ""),
                    "census_division_code": row.get("ALT_GEO_CODE", ""),
                    "census_division_name_profile": row.get("GEO_NAME", ""),
                    "CHARACTERISTIC_ID": key_map["CHARACTERISTIC_ID"],
                    "CHARACTERISTIC_NAME": key_map["CHARACTERISTIC_NAME"],
                    "value_column": value_col,
                    "value_numeric": value_numeric,
                    "symbol": selected_symbol,
                    "unit": key_map["target_unit"],
                    "sovi_role": key_map["target_sovi_role"],
                    "TNR_SF": row.get("TNR_SF", ""),
                    "TNR_LF": row.get("TNR_LF", ""),
                    "DATA_QUALITY_FLAG": row.get("DATA_QUALITY_FLAG", ""),
                }
            )

        if symbol_col in rows.columns:
            for symbol, count in rows[symbol_col].fillna("").astype(str).value_counts().items():
                symbol_count_rows.append(
                    {
                        "original_code": key_map["target_original_code"],
                        "canonical_variable": key_map["target_canonical_variable"],
                        "candidate_output_alias": key_map["target_candidate_output_alias"],
                        "CHARACTERISTIC_ID": key_map["CHARACTERISTIC_ID"],
                        "CHARACTERISTIC_NAME": key_map["CHARACTERISTIC_NAME"],
                        "value_column": value_col,
                        "symbol_column": symbol_col,
                        "symbol": symbol,
                        "row_count": int(count),
                    }
                )

target_summary = (
    pd.DataFrame(target_summary_rows)
    .sort_values(
        [
            "original_code",
            "target_family",
            "CHARACTERISTIC_ID",
        ],
        key=lambda s: s.map(str),
    )
    .reset_index(drop=True)
    if target_summary_rows
    else pd.DataFrame()
)

target_values_long = pd.DataFrame(target_values_long_rows)
symbol_counts = pd.DataFrame(symbol_count_rows)

target_summary.to_csv(OUTPUT_TARGET_SUMMARY, index=False)
target_values_long.to_csv(OUTPUT_TARGET_VALUES_LONG, index=False)
symbol_counts.to_csv(OUTPUT_SYMBOL_COUNTS, index=False)

if not target_values_long.empty:
    target_values_long["wide_column"] = (
        target_values_long["candidate_output_alias"].astype(str)
        + "__char_"
        + target_values_long["CHARACTERISTIC_ID"].astype(str)
    )

    target_values_wide = target_values_long.pivot_table(
        index=[
            "census_division_dguid",
            "census_division_code",
            "census_division_name_profile",
        ],
        columns="wide_column",
        values="value_numeric",
        aggfunc="first",
    ).reset_index()

    target_values_wide.columns.name = None
    target_values_wide.to_csv(OUTPUT_TARGET_VALUES_WIDE, index=False)
else:
    target_values_wide = pd.DataFrame()
    target_values_wide.to_csv(OUTPUT_TARGET_VALUES_WIDE, index=False)


# -----------------------------
# Summary
# -----------------------------

unique_qc_cd_count = len(cd_inventory)
has_98_qc_cds = unique_qc_cd_count == 98

if not base_join.empty:
    matched_base_rows = int((base_join["join_status"] == "both").sum())
    base_only_rows = int((base_join["join_status"] == "left_only").sum())
    profile_only_rows = int((base_join["join_status"] == "right_only").sum())
else:
    matched_base_rows = None
    base_only_rows = None
    profile_only_rows = None

if not target_summary.empty:
    targets_with_full_candidates = (
        target_summary[target_summary["status"] == "candidate_found_full_coverage"]
        ["canonical_variable"]
        .nunique()
    )
else:
    targets_with_full_candidates = 0

summary = pd.DataFrame(
    [
        {
            "metric": "raw_csv",
            "value": str(RAW_CSV.relative_to(DATA_DIR)),
        },
        {
            "metric": "raw_encoding",
            "value": raw_encoding,
        },
        {
            "metric": "total_rows_scanned",
            "value": total_rows,
        },
        {
            "metric": "quebec_cd_rows_scanned",
            "value": quebec_cd_rows,
        },
        {
            "metric": "unique_quebec_census_divisions",
            "value": unique_qc_cd_count,
        },
        {
            "metric": "has_98_quebec_census_divisions",
            "value": has_98_qc_cds,
        },
        {
            "metric": "base_cd_frame_path",
            "value": str(base_path.relative_to(DATA_DIR)) if base_path is not None else "",
        },
        {
            "metric": "matched_base_cd_rows",
            "value": matched_base_rows,
        },
        {
            "metric": "base_only_cd_rows",
            "value": base_only_rows,
        },
        {
            "metric": "profile_only_cd_rows",
            "value": profile_only_rows,
        },
        {
            "metric": "candidate_characteristics_found",
            "value": len(candidate_characteristics),
        },
        {
            "metric": "target_candidate_rows_found",
            "value": len(target_summary),
        },
        {
            "metric": "targets_with_full_coverage_candidate",
            "value": targets_with_full_candidates,
        },
        {
            "metric": "recommended_next_step",
            "value": (
                "Review education_target_characteristic_summary_2021.csv before generating cleaner. "
                "Confirm whether the chosen proxy should be 'No certificate, diploma or degree' "
                "or a narrower no-high-school-diploma construction."
            ),
        },
    ]
)

summary.to_csv(OUTPUT_SUMMARY, index=False)


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CENSUS DIVISION EDUCATION INSPECTION SUMMARY")
print("=" * 72)

print("\nRaw file:")
print("Path:", RAW_CSV.relative_to(DATA_DIR))
print("Encoding:", raw_encoding)
print("Rows scanned:", total_rows)

print("\nQuébec census divisions:")
print("Québec CD rows scanned:", quebec_cd_rows)
print("Unique Québec CD DGUIDs:", unique_qc_cd_count)
print("Has 98 Québec CDs:", has_98_qc_cds)

if not base_join.empty:
    print("\nJoin to existing CD base frame:")
    print(base_join["join_status"].value_counts(dropna=False).to_string())

print("\nEducation candidate characteristics:")
if candidate_characteristics.empty:
    print("[none]")
else:
    display_cols = [
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "n_rows",
        "n_unique_quebec_cds",
        "c1_count_total_non_missing",
        "c10_rate_total_non_missing",
    ]
    display_cols = [col for col in display_cols if col in candidate_characteristics.columns]
    print(candidate_characteristics[display_cols].to_string(index=False))

print("\nTarget candidate summary:")
if target_summary.empty:
    print("[none]")
else:
    display_cols = [
        "original_code",
        "canonical_variable",
        "candidate_output_alias",
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "value_column",
        "n_unique_quebec_cds",
        "value_non_missing",
        "value_min",
        "value_max",
        "value_mean",
        "coverage_is_98_cds",
        "status",
    ]
    print(target_summary[display_cols].to_string(index=False))

print("\nTarget values wide preview:")
if target_values_wide.empty:
    print("[none]")
else:
    print(target_values_wide.head(10).to_string(index=False))

print("\nRecommended next step:")
print(summary.loc[summary["metric"] == "recommended_next_step", "value"].iloc[0])

print("\nSaved:")
print(OUTPUT_FILE_INVENTORY)
print(OUTPUT_CD_INVENTORY)
print(OUTPUT_BASE_JOIN)
print(OUTPUT_CANDIDATE_CHARACTERISTICS)
print(OUTPUT_TARGET_SUMMARY)
print(OUTPUT_TARGET_VALUES_LONG)
print(OUTPUT_TARGET_VALUES_WIDE)
print(OUTPUT_SYMBOL_COUNTS)
print(OUTPUT_SUMMARY)

print("\nDone.")