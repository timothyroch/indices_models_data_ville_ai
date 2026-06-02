from pathlib import Path
import pandas as pd


# ============================================================
# Inspect DGUID Join for StatCan Table 98-10-0007-01
# ============================================================
#
# Purpose:
#   Verify that the downloaded StatCan population/dwelling table can be joined
#   cleanly to the Quebec census-division inventory using DGUID.
#
# Source table:
#   Statistics Canada Table 98-10-0007-01
#   Population and dwelling counts: Canada and census divisions
#
# Inputs:
#   census_division_spatial_frame_population_2021/raw/98100007.csv
#   doctors_per_100khabs/output/quebec_census_divisions_2021_inventory.csv
#
# Outputs:
#   census_division_spatial_frame_population_2021/output/
#       population_dwelling_98100007_dguid_join_validation.csv
#       population_dwelling_98100007_dguid_join_matched_quebec.csv
#       population_dwelling_98100007_dguid_join_unmatched_quebec.csv
#
# Run from data/:
#   python census_division_spatial_frame_population_2021/inspect_population_dwelling_counts_dguid_join.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

RAW_TABLE = (
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "raw"
    / "98100007.csv"
)

CD_INVENTORY = (
    DATA_DIR
    / "doctors_per_100khabs"
    / "output"
    / "quebec_census_divisions_2021_inventory.csv"
)

OUTPUT_DIR = DATA_DIR / "census_division_spatial_frame_population_2021" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_VALIDATION = OUTPUT_DIR / "population_dwelling_98100007_dguid_join_validation.csv"
OUTPUT_MATCHED = OUTPUT_DIR / "population_dwelling_98100007_dguid_join_matched_quebec.csv"
OUTPUT_UNMATCHED = OUTPUT_DIR / "population_dwelling_98100007_dguid_join_unmatched_quebec.csv"
OUTPUT_DUPLICATE_DGUIDS = OUTPUT_DIR / "population_dwelling_98100007_duplicate_dguids.csv"


# -----------------------------
# Helpers
# -----------------------------

def read_csv_fallback(path: Path) -> tuple[pd.DataFrame, str]:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1"]
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False, encoding=enc)
            return df, enc
        except UnicodeDecodeError as e:
            last_error = e
            print(f"Could not read {path.name} with encoding={enc}: {e}")

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not read {path} with tested encodings. Last error: {last_error}",
    )


def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def clean_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("r", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def find_column_containing(columns: list[str], required_terms: list[str]) -> str | None:
    """
    Find the first column whose lowercase name contains all required terms.
    """
    for col in columns:
        col_lower = col.lower()
        if all(term.lower() in col_lower for term in required_terms):
            return col
    return None


# -----------------------------
# Load inputs
# -----------------------------

if not RAW_TABLE.exists():
    raise FileNotFoundError(f"Raw StatCan table not found:\n{RAW_TABLE}")

if not CD_INVENTORY.exists():
    raise FileNotFoundError(f"Quebec census division inventory not found:\n{CD_INVENTORY}")

pop, pop_encoding = read_csv_fallback(RAW_TABLE)
pop = normalize_colnames(pop)

cd = pd.read_csv(CD_INVENTORY, dtype=str)
cd = normalize_colnames(cd)

print("\nLoaded StatCan population/dwelling table")
print("Rows:", len(pop))
print("Columns:", list(pop.columns))
print("Encoding used:", pop_encoding)

print("\nLoaded Quebec census division inventory")
print("Rows:", len(cd))
print("Columns:", list(cd.columns))


# -----------------------------
# Validate required columns
# -----------------------------

required_pop_cols = ["DGUID", "GEO"]
missing_pop_cols = [col for col in required_pop_cols if col not in pop.columns]

if missing_pop_cols:
    raise ValueError(
        "Population table is missing required columns:\n"
        + "\n".join(missing_pop_cols)
    )

required_cd_cols = ["CDUID", "DGUID", "CDNAME", "CDTYPE", "LANDAREA", "PRUID"]
missing_cd_cols = [col for col in required_cd_cols if col not in cd.columns]

if missing_cd_cols:
    raise ValueError(
        "CD inventory is missing required columns:\n"
        + "\n".join(missing_cd_cols)
    )


# -----------------------------
# Detect numeric columns in StatCan table
# -----------------------------

columns = list(pop.columns)

population_2021_col = find_column_containing(
    columns,
    ["Population and dwelling counts", "Population, 2021"],
)

population_2016_col = find_column_containing(
    columns,
    ["Population and dwelling counts", "Population, 2016"],
)

population_change_col = find_column_containing(
    columns,
    ["Population percentage change", "2016 to 2021"],
)

total_private_dwellings_2021_col = find_column_containing(
    columns,
    ["Total private dwellings", "2021"],
)

private_dwellings_occupied_2021_col = find_column_containing(
    columns,
    ["Private dwellings occupied by usual residents", "2021"],
)

land_area_2021_col = find_column_containing(
    columns,
    ["Land area in square kilometres", "2021"],
)

population_density_2021_col = find_column_containing(
    columns,
    ["Population density per square kilometre", "2021"],
)

detected_cols = {
    "population_2021_col": population_2021_col,
    "population_2016_col": population_2016_col,
    "population_change_col": population_change_col,
    "total_private_dwellings_2021_col": total_private_dwellings_2021_col,
    "private_dwellings_occupied_2021_col": private_dwellings_occupied_2021_col,
    "land_area_2021_col": land_area_2021_col,
    "population_density_2021_col": population_density_2021_col,
}

print("\nDetected StatCan value columns:")
for label, col in detected_cols.items():
    print(f"{label}: {col}")

required_detected = {
    "population_2021_col": population_2021_col,
    "population_2016_col": population_2016_col,
    "land_area_2021_col": land_area_2021_col,
    "population_density_2021_col": population_density_2021_col,
}

missing_detected = [
    label for label, col in required_detected.items() if col is None
]

if missing_detected:
    raise ValueError(
        "Could not detect required value columns:\n"
        + "\n".join(missing_detected)
        + "\n\nInspect the raw column names above and adjust the column detection rules."
    )


# -----------------------------
# Clean key fields
# -----------------------------

pop = pop.copy()
cd = cd.copy()

pop["DGUID_clean"] = clean_text(pop["DGUID"])
pop["GEO_clean"] = clean_text(pop["GEO"])

cd["DGUID_clean"] = clean_text(cd["DGUID"])
cd["CDUID_clean"] = clean_text(cd["CDUID"])
cd["CDNAME_clean"] = clean_text(cd["CDNAME"])
cd["CDTYPE_clean"] = clean_text(cd["CDTYPE"])


# -----------------------------
# Check duplicates
# -----------------------------

pop_dupes = pop[pop["DGUID_clean"].duplicated(keep=False)].copy()
cd_dupes = cd[cd["DGUID_clean"].duplicated(keep=False)].copy()

print("\nDuplicate checks")
print("Duplicate DGUIDs in population table:", len(pop_dupes))
print("Duplicate DGUIDs in Quebec CD inventory:", len(cd_dupes))

if not pop_dupes.empty:
    pop_dupes.to_csv(OUTPUT_DUPLICATE_DGUIDS, index=False)
    print("Saved duplicate population DGUID rows:")
    print(OUTPUT_DUPLICATE_DGUIDS)

if not cd_dupes.empty:
    raise ValueError(
        "Duplicate DGUIDs found in Quebec CD inventory:\n"
        + cd_dupes[["CDUID", "DGUID", "CDNAME", "CDTYPE"]].to_string(index=False)
    )


# -----------------------------
# Prepare population table subset
# -----------------------------

pop_subset_cols = [
    "DGUID_clean",
    "DGUID",
    "GEO",
    population_2021_col,
    population_2016_col,
    population_change_col,
    total_private_dwellings_2021_col,
    private_dwellings_occupied_2021_col,
    land_area_2021_col,
    population_density_2021_col,
]

pop_subset_cols = [col for col in pop_subset_cols if col is not None and col in pop.columns]

pop_subset = pop[pop_subset_cols].copy()

# Rename detected value columns to clean names.
rename_map = {
    population_2021_col: "population_total_2021",
    population_2016_col: "population_total_2016",
    population_change_col: "population_change_pct_2016_2021",
    total_private_dwellings_2021_col: "total_private_dwellings_2021",
    private_dwellings_occupied_2021_col: "private_dwellings_occupied_by_usual_residents_2021",
    land_area_2021_col: "land_area_km2_population_table_2021",
    population_density_2021_col: "population_density_per_km2_2021",
}

rename_map = {
    old: new for old, new in rename_map.items()
    if old is not None and old in pop_subset.columns
}

pop_subset = pop_subset.rename(columns=rename_map)

# Numeric conversions.
numeric_cols = [
    "population_total_2021",
    "population_total_2016",
    "population_change_pct_2016_2021",
    "total_private_dwellings_2021",
    "private_dwellings_occupied_by_usual_residents_2021",
    "land_area_km2_population_table_2021",
    "population_density_per_km2_2021",
]

for col in numeric_cols:
    if col in pop_subset.columns:
        pop_subset[col] = clean_number(pop_subset[col])


# -----------------------------
# Join by DGUID
# -----------------------------

validation = cd.merge(
    pop_subset,
    on="DGUID_clean",
    how="left",
    validate="one_to_one",
    indicator=True,
)

validation["matched_population_table"] = validation["_merge"] == "both"

# Reorder useful output columns.
validation_out_cols = [
    "CDUID",
    "DGUID_x",
    "CDNAME",
    "CDTYPE",
    "LANDAREA",
    "PRUID",
    "matched_population_table",
    "GEO",
    "population_total_2021",
    "population_total_2016",
    "population_change_pct_2016_2021",
    "total_private_dwellings_2021",
    "private_dwellings_occupied_by_usual_residents_2021",
    "land_area_km2_population_table_2021",
    "population_density_per_km2_2021",
]

validation_out_cols = [col for col in validation_out_cols if col in validation.columns]

validation_out = validation[validation_out_cols].copy()
validation_out = validation_out.rename(columns={"DGUID_x": "DGUID"})

matched = validation_out[validation_out["matched_population_table"]].copy()
unmatched = validation_out[~validation_out["matched_population_table"]].copy()

validation_out.to_csv(OUTPUT_VALIDATION, index=False)
matched.to_csv(OUTPUT_MATCHED, index=False)
unmatched.to_csv(OUTPUT_UNMATCHED, index=False)


# -----------------------------
# Diagnostics
# -----------------------------

print("\nDGUID join validation")
print("Quebec CD inventory rows:", len(cd))
print("Population table rows:", len(pop))
print("Matched Quebec CDs:", len(matched))
print("Unmatched Quebec CDs:", len(unmatched))

if not unmatched.empty:
    print("\nUnmatched Quebec CD rows:")
    print(unmatched[["CDUID", "DGUID", "CDNAME", "CDTYPE"]].to_string(index=False))

print("\nNumeric missing-value checks among matched rows:")

for col in numeric_cols:
    if col in matched.columns:
        n_missing = matched[col].isna().sum()
        print(f"{col}: missing={n_missing}, non_missing={matched[col].notna().sum()}")

print("\nMatched preview:")
preview_cols = [
    "CDUID",
    "DGUID",
    "CDNAME",
    "CDTYPE",
    "GEO",
    "population_total_2021",
    "population_total_2016",
    "land_area_km2_population_table_2021",
    "population_density_per_km2_2021",
]
preview_cols = [col for col in preview_cols if col in matched.columns]

print(matched[preview_cols].head(120).to_string(index=False))

print("\nSaved:")
print(OUTPUT_VALIDATION)
print(OUTPUT_MATCHED)
print(OUTPUT_UNMATCHED)

if not pop_dupes.empty:
    print(OUTPUT_DUPLICATE_DGUIDS)

print("\nInterpretation:")
print("- If Matched Quebec CDs = 98 and Unmatched Quebec CDs = 0,")
print("  the DGUID join is valid and we can generate the final reusable cleaner.")
print("- If numeric missing values are all zero for population_total_2021,")
print("  the population denominator is usable.")

print("\nDone.")