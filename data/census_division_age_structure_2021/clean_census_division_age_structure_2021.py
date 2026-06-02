from pathlib import Path
import pandas as pd


# ============================================================
# Clean Census Division Age Structure Features 2021
# ============================================================
#
# Purpose:
#   Build a clean Québec census-division age/sex feature table from the
#   2021 Census Profile at census-division level.
#
# SoVI variables cleaned here:
#
#   MED_AGE90  -> median_age
#   PCTKIDS90  -> pct_under_5
#   PCTOLD90   -> pct_over_65
#   PCTFEM90   -> pct_female
#
# Validated by:
#   census_division_age_structure_2021/inspect_census_division_age_structure_2021.py
#
# Run from data/:
#   python census_division_age_structure_2021/clean_census_division_age_structure_2021.py
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

OUTPUT_CSV = OUTPUT_DIR / "clean_census_division_age_structure_2021.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_census_division_age_structure_2021.parquet"
OUTPUT_SOURCE_LONG = OUTPUT_DIR / "clean_census_division_age_structure_source_long_2021.csv"
OUTPUT_VARIABLE_METADATA = OUTPUT_DIR / "clean_census_division_age_structure_variable_metadata_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "clean_census_division_age_structure_summary_2021.csv"


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

RAW_USECOLS = [
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

AGE_TARGETS = [
    {
        "original_code": "MED_AGE90",
        "canonical_variable": "median_age",
        "description": "Median age of the population",
        "characteristic_id": "40",
        "expected_characteristic_name": "Median age of the population",
        "value_mode": "direct_column",
        "value_column": "C1_COUNT_TOTAL",
        "symbol_column": "SYMBOL",
        "unit": "years",
        "source_value_type": "total_value",
        "sovi_role": "direct_or_strong_proxy",
        "notes": "Uses Median age of the population from the 100% age/sex Census Profile section.",
    },
    {
        "original_code": "PCTKIDS90",
        "canonical_variable": "pct_under_5",
        "description": "Percent population under 5 years old",
        "characteristic_id": "10",
        "expected_characteristic_name": "0 to 4 years",
        "value_mode": "direct_column",
        "value_column": "C10_RATE_TOTAL",
        "symbol_column": "SYMBOL.3",
        "unit": "percent",
        "source_value_type": "total_rate",
        "sovi_role": "direct_or_strong_proxy",
        "notes": "Uses the 0 to 4 years age group as the under-5 population share.",
    },
    {
        "original_code": "PCTOLD90",
        "canonical_variable": "pct_over_65",
        "description": "Percent population 65 years and over",
        "characteristic_id": "24",
        "expected_characteristic_name": "65 years and over",
        "value_mode": "direct_column",
        "value_column": "C10_RATE_TOTAL",
        "symbol_column": "SYMBOL.3",
        "unit": "percent",
        "source_value_type": "total_rate",
        "sovi_role": "direct_or_strong_proxy",
        "notes": "Uses the 65 years and over age group as the older-adult population share.",
    },
    {
        "original_code": "PCTFEM90",
        "canonical_variable": "pct_female",
        "description": "Percent women/female population",
        "characteristic_id": "8",
        "expected_characteristic_name": "Total - Age groups of the population",
        "value_mode": "derived_ratio",
        "numerator_column": "C3_COUNT_WOMEN+",
        "denominator_column": "C1_COUNT_TOTAL",
        "symbol_column": "SYMBOL.2",
        "unit": "percent",
        "source_value_type": "derived_women_plus_share",
        "sovi_role": "derived_from_women_count_over_total_population",
        "notes": "Computes 100 * C3_COUNT_WOMEN+ / C1_COUNT_TOTAL from the total age-groups row.",
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


def find_base_cd_frame() -> Path:
    for path in BASE_CD_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find cleaned census-division spatial/population base frame.\n"
        "Expected one of:\n"
        + "\n".join(str(path) for path in BASE_CD_CANDIDATES)
    )


def require_columns(columns: list[str], required: list[str], label: str) -> None:
    missing = [col for col in required if col not in columns]

    if missing:
        raise ValueError(
            f"Missing required columns in {label}:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(columns)
        )


def is_quebec_cd_dguid(series: pd.Series) -> pd.Series:
    return clean_text(series).str.startswith(QUEBEC_CD_DGUID_PREFIX, na=False)


def looks_like_mojibake(text: object) -> bool:
    if pd.isna(text):
        return False

    s = str(text)
    suspicious_tokens = ["Ã", "Â", "�"]

    return any(token in s for token in suspicious_tokens)


def choose_display_name(row: pd.Series) -> str:
    base_name = row.get("census_division_name", "")
    profile_name = row.get("profile_census_division_name", "")

    if looks_like_mojibake(base_name) and not looks_like_mojibake(profile_name):
        return profile_name

    if pd.isna(base_name) or str(base_name).strip() == "":
        return profile_name

    return base_name


def compute_target_value(row: pd.Series, target: dict) -> float:
    if target["value_mode"] == "direct_column":
        return clean_numeric(pd.Series([row.get(target["value_column"], "")])).iloc[0]

    if target["value_mode"] == "derived_ratio":
        numerator = clean_numeric(pd.Series([row.get(target["numerator_column"], "")])).iloc[0]
        denominator = clean_numeric(pd.Series([row.get(target["denominator_column"], "")])).iloc[0]

        if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
            return pd.NA

        return 100.0 * numerator / denominator

    raise ValueError(f"Unsupported value_mode: {target['value_mode']}")


def get_target_value_descriptor(target: dict) -> str:
    if target["value_mode"] == "direct_column":
        return target["value_column"]

    if target["value_mode"] == "derived_ratio":
        return f"100 * {target['numerator_column']} / {target['denominator_column']}"

    raise ValueError(f"Unsupported value_mode: {target['value_mode']}")


def summarize_series(series: pd.Series) -> dict:
    numeric = pd.to_numeric(series, errors="coerce")

    return {
        "non_missing": int(numeric.notna().sum()),
        "missing": int(numeric.isna().sum()),
        "min": numeric.min(skipna=True),
        "max": numeric.max(skipna=True),
        "mean": numeric.mean(skipna=True),
        "median": numeric.median(skipna=True),
    }


# -----------------------------
# Initial validation
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

available_raw_columns = list(header.columns)

require_columns(
    available_raw_columns,
    RAW_USECOLS,
    "Census Profile CD raw CSV",
)

target_characteristic_ids = sorted({target["characteristic_id"] for target in AGE_TARGETS})

print("\nCleaning Census Division Age Structure Features 2021")
print("Raw source:", RAW_CSV.relative_to(DATA_DIR))
print("Raw encoding selected:", raw_encoding)
print("Target characteristic IDs:", target_characteristic_ids)


# -----------------------------
# Load base CD frame
# -----------------------------

base_path = find_base_cd_frame()
base = read_cleaned_table(base_path)
base = normalize_columns(base)

require_columns(
    list(base.columns),
    [
        "census_division_code",
        "census_division_dguid",
        "census_division_name",
        "population_total_2021",
        "land_area_km2",
    ],
    "CD spatial/population base frame",
)

base = base.copy()
base["census_division_code"] = clean_text(base["census_division_code"])
base["census_division_dguid"] = clean_text(base["census_division_dguid"])
base["census_division_name"] = clean_text(base["census_division_name"])

if base["census_division_dguid"].duplicated().any():
    duplicates = base[base["census_division_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated census_division_dguid values in base frame:\n"
        + duplicates[
            [
                "census_division_code",
                "census_division_dguid",
                "census_division_name",
            ]
        ].head(30).to_string(index=False)
    )

print("\nLoaded CD base frame")
print("Path:", base_path.relative_to(DATA_DIR))
print("Rows:", len(base))
print("Base names with possible mojibake:", int(base["census_division_name"].apply(looks_like_mojibake).sum()))


# -----------------------------
# Extract target rows from Census Profile
# -----------------------------

long_rows = []
profile_cd_inventory = {}

total_rows_scanned = 0
quebec_cd_rows_scanned = 0

print("\nScanning raw Census Profile file...")

for chunk_idx, chunk in enumerate(
    pd.read_csv(
        RAW_CSV,
        dtype=str,
        encoding=raw_encoding,
        low_memory=False,
        chunksize=CHUNK_SIZE,
        usecols=RAW_USECOLS,
    ),
    start=1,
):
    total_rows_scanned += len(chunk)
    chunk = normalize_columns(chunk)

    chunk["GEO_LEVEL_NORM"] = clean_text(chunk["GEO_LEVEL"]).str.lower()

    is_cd = chunk["GEO_LEVEL_NORM"].eq("census division")
    is_qc_cd = is_cd & is_quebec_cd_dguid(chunk["DGUID"])

    qc = chunk.loc[is_qc_cd].copy()
    quebec_cd_rows_scanned += len(qc)

    if qc.empty:
        continue

    qc["DGUID"] = clean_text(qc["DGUID"])
    qc["ALT_GEO_CODE"] = clean_text(qc["ALT_GEO_CODE"])
    qc["GEO_NAME"] = clean_text(qc["GEO_NAME"])
    qc["CHARACTERISTIC_ID"] = clean_text(qc["CHARACTERISTIC_ID"])

    inventory_cols = [
        "DGUID",
        "ALT_GEO_CODE",
        "GEO_NAME",
        "GEO_LEVEL",
        "TNR_SF",
        "TNR_LF",
        "DATA_QUALITY_FLAG",
    ]

    for _, row in qc[inventory_cols].drop_duplicates(subset=["DGUID"]).iterrows():
        dguid = str(row["DGUID"])
        profile_cd_inventory[dguid] = {
            "profile_dguid": dguid,
            "profile_census_division_code": row.get("ALT_GEO_CODE", ""),
            "profile_census_division_name": row.get("GEO_NAME", ""),
            "profile_geo_level": row.get("GEO_LEVEL", ""),
            "profile_tnr_sf": row.get("TNR_SF", ""),
            "profile_tnr_lf": row.get("TNR_LF", ""),
            "profile_data_quality_flag": row.get("DATA_QUALITY_FLAG", ""),
        }

    qc_targets = qc[qc["CHARACTERISTIC_ID"].isin(target_characteristic_ids)].copy()

    if qc_targets.empty:
        continue

    for target in AGE_TARGETS:
        target_rows = qc_targets[
            qc_targets["CHARACTERISTIC_ID"].eq(target["characteristic_id"])
        ].copy()

        if target_rows.empty:
            continue

        for _, row in target_rows.iterrows():
            value_numeric = compute_target_value(row, target)
            value_descriptor = get_target_value_descriptor(target)
            symbol_col = target["symbol_column"]

            long_rows.append(
                {
                    "original_code": target["original_code"],
                    "canonical_variable": target["canonical_variable"],
                    "description": target["description"],
                    "census_division_dguid": row.get("DGUID", ""),
                    "census_division_code_profile": row.get("ALT_GEO_CODE", ""),
                    "census_division_name_profile": row.get("GEO_NAME", ""),
                    "characteristic_id": target["characteristic_id"],
                    "characteristic_name": row.get("CHARACTERISTIC_NAME", ""),
                    "characteristic_note": row.get("CHARACTERISTIC_NOTE", ""),
                    "value_mode": target["value_mode"],
                    "value_column_or_formula": value_descriptor,
                    "symbol_column": symbol_col,
                    "value_numeric": value_numeric,
                    "symbol": row.get(symbol_col, ""),
                    "unit": target["unit"],
                    "source_value_type": target["source_value_type"],
                    "sovi_role": target["sovi_role"],
                    "tnr_sf": row.get("TNR_SF", ""),
                    "tnr_lf": row.get("TNR_LF", ""),
                    "data_quality_flag": row.get("DATA_QUALITY_FLAG", ""),
                    "source_file": str(RAW_CSV.relative_to(DATA_DIR)),
                    "notes": target["notes"],
                }
            )

    if chunk_idx % 10 == 0:
        print(f"  Processed chunks: {chunk_idx}, rows so far: {total_rows_scanned}")


source_long = pd.DataFrame(long_rows)

if source_long.empty:
    raise ValueError("No target age-structure rows were extracted from the raw Census Profile file.")

source_long["census_division_dguid"] = clean_text(source_long["census_division_dguid"])
source_long["value_numeric"] = pd.to_numeric(source_long["value_numeric"], errors="coerce")


# -----------------------------
# Validate extracted source rows
# -----------------------------

expected_cd_count = len(base)

validation_rows = []

for target in AGE_TARGETS:
    var = target["canonical_variable"]

    rows = source_long[source_long["canonical_variable"].eq(var)].copy()

    n_rows = len(rows)
    n_unique_cds = rows["census_division_dguid"].nunique()
    n_non_missing = int(pd.to_numeric(rows["value_numeric"], errors="coerce").notna().sum())
    n_missing = int(pd.to_numeric(rows["value_numeric"], errors="coerce").isna().sum())

    duplicated = rows.duplicated(subset=["canonical_variable", "census_division_dguid"]).any()

    validation_rows.append(
        {
            "original_code": target["original_code"],
            "canonical_variable": var,
            "characteristic_id": target["characteristic_id"],
            "characteristic_name_expected": target["expected_characteristic_name"],
            "value_mode": target["value_mode"],
            "value_column_or_formula": get_target_value_descriptor(target),
            "symbol_column": target["symbol_column"],
            "n_rows": n_rows,
            "n_unique_census_divisions": n_unique_cds,
            "value_non_missing": n_non_missing,
            "value_missing": n_missing,
            "has_duplicate_cd_rows": duplicated,
            "coverage_is_complete": n_unique_cds == expected_cd_count,
            "status": (
                "ready"
                if (
                    n_unique_cds == expected_cd_count
                    and n_non_missing == expected_cd_count
                    and not duplicated
                )
                else "needs_review"
            ),
        }
    )

validation = pd.DataFrame(validation_rows)

not_ready = validation[validation["status"] != "ready"]

if not not_ready.empty:
    raise ValueError(
        "One or more age-structure variables failed validation:\n"
        + not_ready.to_string(index=False)
    )


# -----------------------------
# Pivot source values to wide format
# -----------------------------

duplicate_check = (
    source_long
    .groupby(["canonical_variable", "census_division_dguid"])
    .size()
    .reset_index(name="n")
)

duplicates = duplicate_check[duplicate_check["n"] > 1]

if not duplicates.empty:
    raise ValueError(
        "Duplicate rows found for canonical variable / CD key:\n"
        + duplicates.head(30).to_string(index=False)
    )

values_wide = (
    source_long
    .pivot(
        index="census_division_dguid",
        columns="canonical_variable",
        values="value_numeric",
    )
    .reset_index()
)

values_wide.columns.name = None

symbols_wide = (
    source_long
    .pivot(
        index="census_division_dguid",
        columns="canonical_variable",
        values="symbol",
    )
    .reset_index()
)

symbols_wide.columns.name = None
symbols_wide = symbols_wide.rename(
    columns={
        target["canonical_variable"]: f"{target['canonical_variable']}__symbol"
        for target in AGE_TARGETS
    }
)

profile_inventory = pd.DataFrame(profile_cd_inventory.values())
profile_inventory["profile_dguid"] = clean_text(profile_inventory["profile_dguid"])


# -----------------------------
# Build clean output
# -----------------------------

identity_cols = [
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

identity_cols = [col for col in identity_cols if col in base.columns]

clean = base[identity_cols].copy()

clean = clean.merge(
    profile_inventory,
    left_on="census_division_dguid",
    right_on="profile_dguid",
    how="left",
    validate="one_to_one",
)

# Keep original base name, profile name, and a repaired display name.
clean["census_division_name_base_original"] = clean["census_division_name"]
clean["census_division_name_display"] = clean.apply(choose_display_name, axis=1)

# For this clean output, make the main name column safe for downstream use.
clean["census_division_name"] = clean["census_division_name_display"]

clean["census_division_name_base_had_mojibake"] = clean[
    "census_division_name_base_original"
].apply(looks_like_mojibake)

clean["profile_census_division_name_had_mojibake"] = clean[
    "profile_census_division_name"
].apply(looks_like_mojibake)

clean = clean.merge(
    values_wide,
    on="census_division_dguid",
    how="left",
    validate="one_to_one",
)

clean = clean.merge(
    symbols_wide,
    on="census_division_dguid",
    how="left",
    validate="one_to_one",
)

for target in AGE_TARGETS:
    var = target["canonical_variable"]
    clean[var] = pd.to_numeric(clean[var], errors="coerce")
    clean[f"{var}__is_missing"] = clean[var].isna()
    clean[f"{var}__source_characteristic_id"] = target["characteristic_id"]
    clean[f"{var}__source_value_column_or_formula"] = get_target_value_descriptor(target)
    clean[f"{var}__unit"] = target["unit"]
    clean[f"{var}__sovi_role"] = target["sovi_role"]

clean["age_structure_feature_count"] = len(AGE_TARGETS)
clean["age_structure_missing_count"] = clean[
    [target["canonical_variable"] for target in AGE_TARGETS]
].isna().sum(axis=1)

clean["age_structure_complete"] = clean["age_structure_missing_count"].eq(0)

clean["source"] = str(RAW_CSV.relative_to(DATA_DIR))
clean["source_section"] = "2021 Census Profile, Census divisions, age/sex section"
clean["source_encoding"] = raw_encoding


# -----------------------------
# Final validation
# -----------------------------

if len(clean) != len(base):
    raise ValueError(
        f"Clean output row count changed unexpectedly: {len(clean)} vs base {len(base)}"
    )

if clean["census_division_dguid"].duplicated().any():
    duplicated = clean[clean["census_division_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated census_division_dguid values in clean output:\n"
        + duplicated[
            [
                "census_division_code",
                "census_division_dguid",
                "census_division_name",
            ]
        ].head(30).to_string(index=False)
    )

for target in AGE_TARGETS:
    var = target["canonical_variable"]

    missing_count = int(clean[var].isna().sum())
    if missing_count != 0:
        raise ValueError(f"Unexpected missing values in {var}: {missing_count}")

remaining_mojibake_display_names = int(
    clean["census_division_name"].apply(looks_like_mojibake).sum()
)

if remaining_mojibake_display_names != 0:
    print("\nWARNING: Some display names still look mojibaked.")
    print("Count:", remaining_mojibake_display_names)


# -----------------------------
# Variable metadata
# -----------------------------

metadata_rows = []

for target in AGE_TARGETS:
    var = target["canonical_variable"]
    summary = summarize_series(clean[var])

    metadata_rows.append(
        {
            "original_code": target["original_code"],
            "canonical_variable": var,
            "description": target["description"],
            "characteristic_id": target["characteristic_id"],
            "expected_characteristic_name": target["expected_characteristic_name"],
            "value_mode": target["value_mode"],
            "value_column_or_formula": get_target_value_descriptor(target),
            "symbol_column": target["symbol_column"],
            "unit": target["unit"],
            "source_value_type": target["source_value_type"],
            "sovi_role": target["sovi_role"],
            "source_file": str(RAW_CSV.relative_to(DATA_DIR)),
            "n_rows": len(clean),
            "non_missing": summary["non_missing"],
            "missing": summary["missing"],
            "min": summary["min"],
            "max": summary["max"],
            "mean": summary["mean"],
            "median": summary["median"],
            "notes": target["notes"],
        }
    )

metadata = pd.DataFrame(metadata_rows)


# -----------------------------
# Summary
# -----------------------------

summary_rows = [
    {
        "metric": "raw_csv",
        "value": str(RAW_CSV.relative_to(DATA_DIR)),
    },
    {
        "metric": "base_cd_frame",
        "value": str(base_path.relative_to(DATA_DIR)),
    },
    {
        "metric": "raw_encoding",
        "value": raw_encoding,
    },
    {
        "metric": "total_rows_scanned",
        "value": total_rows_scanned,
    },
    {
        "metric": "quebec_cd_rows_scanned",
        "value": quebec_cd_rows_scanned,
    },
    {
        "metric": "clean_rows",
        "value": len(clean),
    },
    {
        "metric": "unique_census_divisions",
        "value": clean["census_division_dguid"].nunique(),
    },
    {
        "metric": "variables_cleaned",
        "value": ", ".join(target["canonical_variable"] for target in AGE_TARGETS),
    },
    {
        "metric": "all_variables_complete",
        "value": bool(clean["age_structure_complete"].all()),
    },
    {
        "metric": "base_names_with_mojibake",
        "value": int(clean["census_division_name_base_had_mojibake"].sum()),
    },
    {
        "metric": "profile_names_with_mojibake",
        "value": int(clean["profile_census_division_name_had_mojibake"].sum()),
    },
    {
        "metric": "display_names_with_mojibake",
        "value": remaining_mojibake_display_names,
    },
]

for target in AGE_TARGETS:
    var = target["canonical_variable"]
    summary = summarize_series(clean[var])

    for metric_name, value in summary.items():
        summary_rows.append(
            {
                "metric": f"{var}_{metric_name}",
                "value": value,
            }
        )

summary = pd.DataFrame(summary_rows)


# -----------------------------
# Save outputs
# -----------------------------

clean.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

try:
    clean.to_parquet(OUTPUT_PARQUET, index=False)
    parquet_saved = True
except Exception as exc:
    parquet_saved = False
    print("\nWARNING: Could not save Parquet output.")
    print("Reason:", exc)

source_long.to_csv(OUTPUT_SOURCE_LONG, index=False, encoding="utf-8")
metadata.to_csv(OUTPUT_VARIABLE_METADATA, index=False, encoding="utf-8")
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console diagnostics
# -----------------------------

print("\n" + "=" * 72)
print("CLEAN CENSUS DIVISION AGE STRUCTURE FEATURES 2021")
print("=" * 72)

print("\nFinal clean table:")
print("Rows:", len(clean))
print("Columns:", list(clean.columns))

print("\nName encoding diagnostics:")
print("Base names with possible mojibake:", int(clean["census_division_name_base_had_mojibake"].sum()))
print("Profile names with possible mojibake:", int(clean["profile_census_division_name_had_mojibake"].sum()))
print("Display names with possible mojibake:", remaining_mojibake_display_names)

print("\nVariables cleaned:")
for target in AGE_TARGETS:
    var = target["canonical_variable"]
    print(
        f"- {var}: "
        f"non-missing={clean[var].notna().sum()}, "
        f"missing={clean[var].isna().sum()}, "
        f"min={clean[var].min()}, "
        f"max={clean[var].max()}, "
        f"mean={clean[var].mean():.3f}"
    )

print("\nValidation summary:")
print(validation.to_string(index=False))

print("\nPreview:")
preview_cols = [
    "census_division_code",
    "census_division_name",
    "profile_census_division_name",
    "median_age",
    "pct_under_5",
    "pct_over_65",
    "pct_female",
]
preview_cols = [col for col in preview_cols if col in clean.columns]
print(clean[preview_cols].head(12).to_string(index=False))

print("\nSaved:")
print(OUTPUT_CSV)
if parquet_saved:
    print(OUTPUT_PARQUET)
print(OUTPUT_SOURCE_LONG)
print(OUTPUT_VARIABLE_METADATA)
print(OUTPUT_SUMMARY)

print("\nDone.")