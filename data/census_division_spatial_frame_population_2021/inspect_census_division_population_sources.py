from pathlib import Path
import pandas as pd


# ============================================================
# Inspect Census-Division Population Sources
# ============================================================
#
# Purpose:
#   Check whether our existing Census Profile 2021 files contain
#   census-division-level population rows that can be used to build a reusable
#   census-division spatial frame with population.
#
# This version includes encoding fallback because the main Census Profile CSV
# may not be UTF-8.
#
# Run from data/:
#   python census_division_spatial_frame_population_2021/inspect_census_division_population_sources.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

CENSUS_PROFILE_DIR = DATA_DIR / "census_profile_2021"

CD_INVENTORY_PATH = (
    DATA_DIR
    / "doctors_per_100khabs"
    / "output"
    / "quebec_census_divisions_2021_inventory.csv"
)

OUTPUT_DIR = DATA_DIR / "census_division_spatial_frame_population_2021" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE_INVENTORY = OUTPUT_DIR / "census_profile_candidate_files_inventory.csv"
OUTPUT_GEO_LEVEL_COUNTS = OUTPUT_DIR / "census_profile_geo_level_counts.csv"
OUTPUT_CD_POPULATION_CANDIDATES = OUTPUT_DIR / "census_division_population_candidate_rows.csv"
OUTPUT_CD_DGUID_VALIDATION = OUTPUT_DIR / "census_division_dguid_population_validation.csv"


# -----------------------------
# Helpers
# -----------------------------

ENCODINGS_TO_TRY = [
    "utf-8",
    "utf-8-sig",
    "cp1252",
    "latin1",
    "iso-8859-1",
]


def read_csv_with_encoding_fallback(path: Path, nrows=None) -> tuple[pd.DataFrame, str]:
    """
    Read a CSV using several common encodings.

    Returns:
        (dataframe, encoding_used)
    """
    last_error = None

    for encoding in ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(
                path,
                dtype=str,
                low_memory=False,
                encoding=encoding,
                nrows=nrows,
            )
            return df, encoding
        except UnicodeDecodeError as e:
            last_error = e
            print(f"Could not read {path.name} with encoding={encoding}: {e}")

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not read {path} with any tested encoding. Last error: {last_error}",
    )


def read_preview(path: Path, nrows: int = 5) -> tuple[pd.DataFrame, str]:
    """
    Try reading a CSV/Parquet preview.
    """
    if path.suffix.lower() == ".csv":
        return read_csv_with_encoding_fallback(path, nrows=nrows)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        return df.head(nrows), "parquet"

    raise ValueError(f"Unsupported file type: {path}")


def read_full(path: Path) -> tuple[pd.DataFrame, str]:
    """
    Read a full CSV/Parquet file.
    """
    if path.suffix.lower() == ".csv":
        return read_csv_with_encoding_fallback(path, nrows=None)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        return df, "parquet"

    raise ValueError(f"Unsupported file type: {path}")


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from column names.
    """
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


# -----------------------------
# Validate inputs
# -----------------------------

if not CENSUS_PROFILE_DIR.exists():
    raise FileNotFoundError(
        f"Census Profile folder not found:\n{CENSUS_PROFILE_DIR}"
    )

if not CD_INVENTORY_PATH.exists():
    raise FileNotFoundError(
        f"Quebec census division inventory not found:\n{CD_INVENTORY_PATH}\n\n"
        "Expected from the previous census-division boundary inspection step."
    )


# -----------------------------
# Inventory candidate files
# -----------------------------

candidate_paths = []

for suffix in ["*.csv", "*.parquet"]:
    candidate_paths.extend(CENSUS_PROFILE_DIR.rglob(suffix))

candidate_paths = sorted(candidate_paths)

if not candidate_paths:
    raise ValueError(
        f"No CSV or Parquet files found under:\n{CENSUS_PROFILE_DIR}"
    )

file_inventory_rows = []

print("\nInspecting candidate Census Profile files...")

for path in candidate_paths:
    try:
        preview, encoding_used = read_preview(path, nrows=5)
        preview = normalize_columns(preview)

        file_inventory_rows.append(
            {
                "relative_path": str(path.relative_to(DATA_DIR)),
                "file_type": path.suffix.lower(),
                "readable": True,
                "encoding_used": encoding_used,
                "preview_rows": len(preview),
                "columns": ", ".join(map(str, preview.columns)),
                "has_geo_level": "GEO_LEVEL" in preview.columns,
                "has_dguid": "DGUID" in preview.columns,
                "has_geo_name": "GEO_NAME" in preview.columns,
                "has_characteristic_id": "CHARACTERISTIC_ID" in preview.columns,
                "has_characteristic_name": "CHARACTERISTIC_NAME" in preview.columns,
                "has_c1_count_total": "C1_COUNT_TOTAL" in preview.columns,
                "has_c1_symbol": "C1_SYMBOL" in preview.columns,
                "error": "",
            }
        )

        print("\nFile:", path.relative_to(DATA_DIR))
        print("Readable: True")
        print("Encoding used:", encoding_used)
        print("Columns:", list(preview.columns))

    except Exception as e:
        file_inventory_rows.append(
            {
                "relative_path": str(path.relative_to(DATA_DIR)),
                "file_type": path.suffix.lower(),
                "readable": False,
                "encoding_used": "",
                "preview_rows": 0,
                "columns": "",
                "has_geo_level": False,
                "has_dguid": False,
                "has_geo_name": False,
                "has_characteristic_id": False,
                "has_characteristic_name": False,
                "has_c1_count_total": False,
                "has_c1_symbol": False,
                "error": str(e),
            }
        )

        print("\nFile:", path.relative_to(DATA_DIR))
        print("Readable: False")
        print("Error:", e)


file_inventory = pd.DataFrame(file_inventory_rows)
file_inventory.to_csv(OUTPUT_FILE_INVENTORY, index=False)

print("\nSaved file inventory:")
print(OUTPUT_FILE_INVENTORY)


# -----------------------------
# Pick likely Census Profile file(s)
# -----------------------------

likely_files = file_inventory[
    file_inventory["readable"]
    & file_inventory["has_geo_level"]
    & file_inventory["has_dguid"]
    & file_inventory["has_geo_name"]
    & file_inventory["has_characteristic_id"]
    & file_inventory["has_characteristic_name"]
    & file_inventory["has_c1_count_total"]
].copy()

if likely_files.empty:
    print("\nNo single file has all expected Census Profile columns.")
    print("Inspect the inventory CSV manually.")
    print("\nDone.")
    raise SystemExit(0)

print("\nLikely Census Profile files:")
print(
    likely_files[
        ["relative_path", "file_type", "encoding_used"]
    ].to_string(index=False)
)


# -----------------------------
# Load likely files and inspect GEO_LEVEL
# -----------------------------

all_geo_counts = []
all_population_candidates = []

for _, likely_row in likely_files.iterrows():
    relative_path = likely_row["relative_path"]
    path = DATA_DIR / relative_path

    print("\nLoading likely file:")
    print(path.relative_to(DATA_DIR))

    df, encoding_used = read_full(path)
    df = normalize_columns(df)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("Encoding used:", encoding_used)

    required_cols = [
        "GEO_LEVEL",
        "DGUID",
        "GEO_NAME",
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "C1_COUNT_TOTAL",
    ]

    missing_required = [col for col in required_cols if col not in df.columns]

    if missing_required:
        print("Skipping file because required columns are missing:", missing_required)
        continue

    # Normalize key columns.
    df["GEO_LEVEL"] = normalize_text(df["GEO_LEVEL"])
    df["DGUID"] = normalize_text(df["DGUID"])
    df["GEO_NAME"] = normalize_text(df["GEO_NAME"])
    df["CHARACTERISTIC_ID"] = normalize_text(df["CHARACTERISTIC_ID"])
    df["CHARACTERISTIC_NAME"] = normalize_text(df["CHARACTERISTIC_NAME"])

    geo_counts = (
        df["GEO_LEVEL"]
        .fillna("MISSING")
        .value_counts(dropna=False)
        .rename_axis("GEO_LEVEL")
        .reset_index(name="row_count")
    )
    geo_counts["source_file"] = str(path.relative_to(DATA_DIR))
    geo_counts["encoding_used"] = encoding_used

    all_geo_counts.append(geo_counts)

    print("\nGEO_LEVEL counts:")
    print(geo_counts.to_string(index=False))

    # Census division rows.
    cd_rows = df[df["GEO_LEVEL"].str.lower() == "census division"].copy()

    print("\nCensus division rows found:", len(cd_rows))

    if cd_rows.empty:
        continue

    # Population-like candidates.
    pop_name = cd_rows["CHARACTERISTIC_NAME"].str.lower()

    population_candidates = cd_rows[
        (
            cd_rows["CHARACTERISTIC_ID"].isin(["1"])
            | pop_name.eq("population, 2021")
            | pop_name.eq("population, 2016")
            | pop_name.str.contains("population", na=False)
        )
    ].copy()

    population_candidates["source_file"] = str(path.relative_to(DATA_DIR))
    population_candidates["encoding_used"] = encoding_used

    useful_cols = [
        "source_file",
        "encoding_used",
        "GEO_LEVEL",
        "DGUID",
        "GEO_NAME",
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "C1_COUNT_TOTAL",
    ]

    if "C1_SYMBOL" in population_candidates.columns:
        useful_cols.append("C1_SYMBOL")

    population_candidates = population_candidates[
        [col for col in useful_cols if col in population_candidates.columns]
    ].copy()

    all_population_candidates.append(population_candidates)

    print("\nPopulation-like candidate rows:")
    print(population_candidates.head(150).to_string(index=False))


# -----------------------------
# Save GEO_LEVEL counts
# -----------------------------

if all_geo_counts:
    geo_counts_all = pd.concat(all_geo_counts, ignore_index=True)
    geo_counts_all.to_csv(OUTPUT_GEO_LEVEL_COUNTS, index=False)

    print("\nSaved GEO_LEVEL counts:")
    print(OUTPUT_GEO_LEVEL_COUNTS)
else:
    geo_counts_all = pd.DataFrame()


# -----------------------------
# Save population candidates
# -----------------------------

if all_population_candidates:
    pop_candidates_all = pd.concat(all_population_candidates, ignore_index=True)
    pop_candidates_all.to_csv(OUTPUT_CD_POPULATION_CANDIDATES, index=False)

    print("\nSaved census-division population candidates:")
    print(OUTPUT_CD_POPULATION_CANDIDATES)
else:
    pop_candidates_all = pd.DataFrame()

    print("\nNo census-division population candidates found.")
    print("We may need to download a dedicated population and dwelling counts table.")


# -----------------------------
# Validate against Quebec CD inventory
# -----------------------------

if not pop_candidates_all.empty:
    cd_inventory = pd.read_csv(CD_INVENTORY_PATH, dtype=str)

    required_cd_cols = ["CDUID", "DGUID", "CDNAME", "CDTYPE", "LANDAREA", "PRUID"]
    missing_cd_cols = [col for col in required_cd_cols if col not in cd_inventory.columns]

    if missing_cd_cols:
        raise ValueError(
            "Missing expected columns in CD inventory:\n"
            + "\n".join(missing_cd_cols)
        )

    qc_cd_inventory = cd_inventory.copy()
    qc_cd_inventory["DGUID"] = normalize_text(qc_cd_inventory["DGUID"])
    qc_cd_inventory["CDUID"] = normalize_text(qc_cd_inventory["CDUID"])
    qc_cd_inventory["CDNAME"] = normalize_text(qc_cd_inventory["CDNAME"])

    # Prefer CHARACTERISTIC_ID == 1 if available.
    pop_for_validation = pop_candidates_all.copy()
    pop_for_validation["CHARACTERISTIC_ID"] = normalize_text(
        pop_for_validation["CHARACTERISTIC_ID"]
    )

    if (pop_for_validation["CHARACTERISTIC_ID"] == "1").any():
        pop_for_validation = pop_for_validation[
            pop_for_validation["CHARACTERISTIC_ID"] == "1"
        ].copy()

    # Filter to Quebec CD DGUIDs by joining with inventory.
    validation = qc_cd_inventory.merge(
        pop_for_validation,
        on="DGUID",
        how="left",
        validate="one_to_many",
        indicator=True,
    )

    validation["has_population_candidate"] = validation["_merge"] == "both"

    validation_out_cols = [
        "CDUID",
        "DGUID",
        "CDNAME",
        "CDTYPE",
        "LANDAREA",
        "PRUID",
        "has_population_candidate",
        "source_file",
        "encoding_used",
        "GEO_LEVEL",
        "GEO_NAME",
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "C1_COUNT_TOTAL",
    ]

    if "C1_SYMBOL" in validation.columns:
        validation_out_cols.append("C1_SYMBOL")

    validation = validation[
        [col for col in validation_out_cols if col in validation.columns]
    ].copy()

    validation.to_csv(OUTPUT_CD_DGUID_VALIDATION, index=False)

    print("\nSaved Quebec CD DGUID population validation:")
    print(OUTPUT_CD_DGUID_VALIDATION)

    print("\nValidation summary:")
    print("Quebec census divisions in inventory:", len(qc_cd_inventory))
    print(
        "Quebec census divisions with population candidate:",
        int(validation["has_population_candidate"].sum()),
    )
    print(
        "Quebec census divisions without population candidate:",
        int((~validation["has_population_candidate"]).sum()),
    )

    missing_pop = validation[~validation["has_population_candidate"]]

    if not missing_pop.empty:
        print("\nQuebec CDs missing population candidate:")
        print(missing_pop[["CDUID", "DGUID", "CDNAME"]].to_string(index=False))


# -----------------------------
# Final recommendation
# -----------------------------

print("\nInterpretation:")
print("- If we found 98 Quebec census divisions with CHARACTERISTIC_ID == 1,")
print("  then we can build the reusable CD population/spatial frame from existing data.")
print("- If not, we should download a dedicated 2021 population and dwelling counts")
print("  table for census divisions.")

print("\nDone.")