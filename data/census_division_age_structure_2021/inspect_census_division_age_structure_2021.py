from pathlib import Path
import pandas as pd


# ============================================================
# Inspect Census Division Age Structure Features 2021
# ============================================================
#
# Purpose:
#   Inspect the 2021 Census Profile at census-division level for the
#   age/sex variables needed by the SoVI-like census-division table.
#
# SoVI target variables inspected here:
#
#   MED_AGE90  -> median_age
#   PCTKIDS90  -> pct_under_5
#   PCTOLD90   -> pct_over_65
#   PCTFEM90   -> pct_female
#
# This script does NOT create the final clean age-structure table.
# It verifies characteristic IDs, value columns, coverage, symbols,
# derived formulas, and compatibility with the Québec census-division base frame.
#
# Expected raw source:
#   census_profile_census_division_2021/raw/98-401-X2021004_English_CSV_data.csv
#
# Run from data/:
#   python census_division_age_structure_2021/inspect_census_division_age_structure_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_age_structure_2021"
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

OUTPUT_FILE_INVENTORY = OUTPUT_DIR / "age_structure_file_inventory_2021.csv"
OUTPUT_CD_INVENTORY = OUTPUT_DIR / "age_structure_quebec_cd_inventory_2021.csv"
OUTPUT_BASE_JOIN = OUTPUT_DIR / "age_structure_quebec_cd_base_join_check_2021.csv"
OUTPUT_CANDIDATE_CHARACTERISTICS = OUTPUT_DIR / "age_structure_candidate_characteristics_2021.csv"
OUTPUT_TARGET_SUMMARY = OUTPUT_DIR / "age_structure_target_characteristic_summary_2021.csv"
OUTPUT_TARGET_VALUES_LONG = OUTPUT_DIR / "age_structure_target_values_long_2021.csv"
OUTPUT_TARGET_VALUES_WIDE = OUTPUT_DIR / "age_structure_target_values_wide_2021.csv"
OUTPUT_SYMBOL_COUNTS = OUTPUT_DIR / "age_structure_symbol_counts_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "age_structure_inspection_summary_2021.csv"


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

AGE_CONTEXT_IDS = {
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
}

AGE_TARGETS = [
    {
        "original_code": "MED_AGE90",
        "canonical_variable": "median_age",
        "description": "Median age of the population",
        "expected_characteristic_id": "40",
        "expected_characteristic_name_contains": "median age",
        "value_mode": "direct_column",
        "value_column": "C1_COUNT_TOTAL",
        "symbol_column": "SYMBOL",
        "unit": "years",
        "sovi_role": "direct_or_strong_proxy",
        "notes": "Uses Median age of the population from the 100% age/sex Census Profile section.",
    },
    {
        "original_code": "PCTKIDS90",
        "canonical_variable": "pct_under_5",
        "description": "Percent population under 5 years old",
        "expected_characteristic_id": "10",
        "expected_characteristic_name_contains": "0 to 4 years",
        "value_mode": "direct_column",
        "value_column": "C10_RATE_TOTAL",
        "symbol_column": "SYMBOL.3",
        "unit": "percent",
        "sovi_role": "direct_or_strong_proxy",
        "notes": "Uses the 0 to 4 years age group as the under-5 population share.",
    },
    {
        "original_code": "PCTOLD90",
        "canonical_variable": "pct_over_65",
        "description": "Percent population 65 years and over",
        "expected_characteristic_id": "24",
        "expected_characteristic_name_contains": "65 years and over",
        "value_mode": "direct_column",
        "value_column": "C10_RATE_TOTAL",
        "symbol_column": "SYMBOL.3",
        "unit": "percent",
        "sovi_role": "direct_or_strong_proxy",
        "notes": "Uses the 65 years and over age group as the older-adult population share.",
    },
    {
        "original_code": "PCTFEM90",
        "canonical_variable": "pct_female",
        "description": "Percent women/female population",
        "expected_characteristic_id": "8",
        "expected_characteristic_name_contains": "total - age groups of the population",
        "value_mode": "derived_ratio",
        "numerator_column": "C3_COUNT_WOMEN+",
        "denominator_column": "C1_COUNT_TOTAL",
        "symbol_column": "SYMBOL.2",
        "unit": "percent",
        "sovi_role": "derived_from_women_count_over_total_population",
        "notes": (
            "Computes 100 * C3_COUNT_WOMEN+ / C1_COUNT_TOTAL from the total age-groups row. "
            "This is inspected because Census Profile may not provide a direct total female-rate row."
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


def compute_target_value(rows: pd.DataFrame, target: dict) -> pd.Series:
    if target["value_mode"] == "direct_column":
        return clean_numeric(rows[target["value_column"]])

    if target["value_mode"] == "derived_ratio":
        numerator = clean_numeric(rows[target["numerator_column"]])
        denominator = clean_numeric(rows[target["denominator_column"]])

        value = 100.0 * numerator / denominator
        value = value.where(denominator.ne(0))

        return value

    raise ValueError(f"Unsupported value_mode: {target['value_mode']}")


def get_target_symbol(rows: pd.DataFrame, target: dict) -> pd.Series:
    symbol_col = target.get("symbol_column")
    if symbol_col and symbol_col in rows.columns:
        return rows[symbol_col]
    return pd.Series([""] * len(rows), index=rows.index)


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

print("\nCensus Profile CD age-structure inspection")
print("Raw file:", RAW_CSV.relative_to(DATA_DIR))
print("Raw encoding selected:", raw_encoding)
print("Target characteristic IDs:", sorted({target["expected_characteristic_id"] for target in AGE_TARGETS}))
print("Context characteristic IDs:", f"{min(map(int, AGE_CONTEXT_IDS))}–{max(map(int, AGE_CONTEXT_IDS))}")

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
target_value_rows = []

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

    context_rows = qc[qc["CHARACTERISTIC_ID"].isin(AGE_CONTEXT_IDS)].copy()
    if not context_rows.empty:
        candidate_rows.append(context_rows)

    target_ids = {target["expected_characteristic_id"] for target in AGE_TARGETS}
    target_rows = qc[qc["CHARACTERISTIC_ID"].isin(target_ids)].copy()
    if not target_rows.empty:
        target_value_rows.append(target_rows)

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
    pd.concat(target_value_rows, ignore_index=True)
    if target_value_rows
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

for target in AGE_TARGETS:
    char_id = target["expected_characteristic_id"]

    rows = target_raw_df[
        clean_text(target_raw_df["CHARACTERISTIC_ID"]).eq(char_id)
    ].copy()

    if rows.empty:
        target_summary_rows.append(
            {
                "original_code": target["original_code"],
                "canonical_variable": target["canonical_variable"],
                "expected_characteristic_id": char_id,
                "selected_characteristic_name": "",
                "value_mode": target["value_mode"],
                "value_column_or_formula": "",
                "symbol_column": target.get("symbol_column", ""),
                "unit": target["unit"],
                "sovi_role": target["sovi_role"],
                "n_rows": 0,
                "n_unique_quebec_cds": 0,
                "value_non_missing": 0,
                "value_missing": 0,
                "value_min": None,
                "value_max": None,
                "value_mean": None,
                "value_median": None,
                "coverage_is_98_cds": False,
                "name_contains_expected_text": False,
                "status": "missing_expected_characteristic",
                "notes": target["notes"],
            }
        )
        continue

    selected_name = (
        rows["CHARACTERISTIC_NAME"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .iloc[0]
        if rows["CHARACTERISTIC_NAME"].notna().any()
        else ""
    )

    values = compute_target_value(rows, target)
    symbols = get_target_symbol(rows, target)

    coverage = rows["DGUID"].nunique()
    name_ok = target["expected_characteristic_name_contains"].lower() in selected_name.lower()

    if target["value_mode"] == "direct_column":
        value_column_or_formula = target["value_column"]
    else:
        value_column_or_formula = (
            f"100 * {target['numerator_column']} / {target['denominator_column']}"
        )

    target_summary_rows.append(
        {
            "original_code": target["original_code"],
            "canonical_variable": target["canonical_variable"],
            "description": target["description"],
            "expected_characteristic_id": char_id,
            "selected_characteristic_name": selected_name,
            "value_mode": target["value_mode"],
            "value_column_or_formula": value_column_or_formula,
            "symbol_column": target.get("symbol_column", ""),
            "unit": target["unit"],
            "sovi_role": target["sovi_role"],
            "n_rows": len(rows),
            "n_unique_quebec_cds": coverage,
            "value_non_missing": int(values.notna().sum()),
            "value_missing": int(values.isna().sum()),
            "value_min": values.min(skipna=True),
            "value_max": values.max(skipna=True),
            "value_mean": values.mean(skipna=True),
            "value_median": values.median(skipna=True),
            "coverage_is_98_cds": coverage == 98,
            "name_contains_expected_text": name_ok,
            "status": (
                "ready_for_cleaner"
                if coverage == 98 and values.notna().sum() == 98 and name_ok
                else "needs_review"
            ),
            "notes": target["notes"],
        }
    )

    for idx, row in rows.iterrows():
        value_numeric = values.loc[idx]
        selected_symbol = symbols.loc[idx] if idx in symbols.index else ""

        target_values_long_rows.append(
            {
                "original_code": target["original_code"],
                "canonical_variable": target["canonical_variable"],
                "census_division_dguid": row.get("DGUID", ""),
                "census_division_code": row.get("ALT_GEO_CODE", ""),
                "census_division_name_profile": row.get("GEO_NAME", ""),
                "CHARACTERISTIC_ID": char_id,
                "CHARACTERISTIC_NAME": selected_name,
                "value_mode": target["value_mode"],
                "value_column_or_formula": value_column_or_formula,
                "value_numeric": value_numeric,
                "symbol": selected_symbol,
                "TNR_SF": row.get("TNR_SF", ""),
                "TNR_LF": row.get("TNR_LF", ""),
                "DATA_QUALITY_FLAG": row.get("DATA_QUALITY_FLAG", ""),
            }
        )

    symbol_col = target.get("symbol_column")
    if symbol_col and symbol_col in rows.columns:
        for symbol, count in rows[symbol_col].fillna("").astype(str).value_counts().items():
            symbol_count_rows.append(
                {
                    "original_code": target["original_code"],
                    "canonical_variable": target["canonical_variable"],
                    "CHARACTERISTIC_ID": char_id,
                    "CHARACTERISTIC_NAME": selected_name,
                    "value_mode": target["value_mode"],
                    "symbol_column": symbol_col,
                    "symbol": symbol,
                    "row_count": int(count),
                }
            )

target_summary = pd.DataFrame(target_summary_rows)
target_values_long = pd.DataFrame(target_values_long_rows)
symbol_counts = pd.DataFrame(symbol_count_rows)

target_summary.to_csv(OUTPUT_TARGET_SUMMARY, index=False)
target_values_long.to_csv(OUTPUT_TARGET_VALUES_LONG, index=False)
symbol_counts.to_csv(OUTPUT_SYMBOL_COUNTS, index=False)

if not target_values_long.empty:
    target_values_wide = target_values_long.pivot_table(
        index=[
            "census_division_dguid",
            "census_division_code",
            "census_division_name_profile",
        ],
        columns="canonical_variable",
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

ready_target_count = int((target_summary["status"] == "ready_for_cleaner").sum())

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
            "metric": "age_targets_total",
            "value": len(AGE_TARGETS),
        },
        {
            "metric": "age_targets_ready_for_cleaner",
            "value": ready_target_count,
        },
        {
            "metric": "recommended_next_step",
            "value": (
                "Generate clean_census_division_age_structure_2021.py."
                if ready_target_count == len(AGE_TARGETS)
                else "Review target summary before generating age-structure cleaner."
            ),
        },
    ]
)

summary.to_csv(OUTPUT_SUMMARY, index=False)


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CENSUS DIVISION AGE STRUCTURE INSPECTION SUMMARY")
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

print("\nCandidate age/sex characteristics:")
if candidate_characteristics.empty:
    print("[none]")
else:
    display_cols = [
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "n_rows",
        "n_unique_quebec_cds",
        "c1_count_total_non_missing",
        "c3_count_womenplus_non_missing",
        "c10_rate_total_non_missing",
    ]
    display_cols = [col for col in display_cols if col in candidate_characteristics.columns]
    print(candidate_characteristics[display_cols].to_string(index=False))

print("\nTarget variable summary:")
print(
    target_summary[
        [
            "original_code",
            "canonical_variable",
            "expected_characteristic_id",
            "selected_characteristic_name",
            "value_mode",
            "value_column_or_formula",
            "n_unique_quebec_cds",
            "value_non_missing",
            "value_min",
            "value_max",
            "value_mean",
            "coverage_is_98_cds",
            "name_contains_expected_text",
            "status",
        ]
    ].to_string(index=False)
)

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