from pathlib import Path
import pandas as pd


# ============================================================
# Inspect StatCan Table 98-10-0007-01
# Population and dwelling counts: Canada and census divisions
# ============================================================
#
# Purpose:
#   Inspect the downloaded StatCan population/dwelling count table to confirm
#   whether it can provide a reusable census-division population denominator.
#
# Source:
#   Statistics Canada Table 98-10-0007-01
#   Population and dwelling counts: Canada and census divisions
#
# Inputs:
#   census_division_spatial_frame_population_2021/raw/98100007.csv
#   census_division_spatial_frame_population_2021/raw/98100007_MetaData.csv
#   doctors_per_100khabs/output/quebec_census_divisions_2021_inventory.csv
#
# Outputs:
#   census_division_spatial_frame_population_2021/output/
#       population_dwelling_98100007_columns_inventory.csv
#       population_dwelling_98100007_quebec_preview.csv
#       population_dwelling_98100007_quebec_cd_validation_by_name.csv
#
# Run from data/:
#   python census_division_spatial_frame_population_2021/inspect_population_dwelling_counts_98100007.py
#
# ============================================================


DATA_DIR = Path(__file__).resolve().parent.parent

RAW_TABLE = (
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "raw"
    / "98100007.csv"
)

RAW_METADATA = (
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "raw"
    / "98100007_MetaData.csv"
)

CD_INVENTORY = (
    DATA_DIR
    / "doctors_per_100khabs"
    / "output"
    / "quebec_census_divisions_2021_inventory.csv"
)

OUTPUT_DIR = DATA_DIR / "census_division_spatial_frame_population_2021" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_COLUMNS = OUTPUT_DIR / "population_dwelling_98100007_columns_inventory.csv"
OUTPUT_QC_PREVIEW = OUTPUT_DIR / "population_dwelling_98100007_quebec_preview.csv"
OUTPUT_QC_VALIDATION = OUTPUT_DIR / "population_dwelling_98100007_quebec_cd_validation_by_name.csv"


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


def clean_text(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def clean_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("r", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


# -----------------------------
# Load files
# -----------------------------

if not RAW_TABLE.exists():
    raise FileNotFoundError(f"Raw table not found:\n{RAW_TABLE}")

if not CD_INVENTORY.exists():
    raise FileNotFoundError(f"Quebec CD inventory not found:\n{CD_INVENTORY}")

table, table_encoding = read_csv_fallback(RAW_TABLE)
table = normalize_colnames(table)

cd_inventory = pd.read_csv(CD_INVENTORY, dtype=str)
cd_inventory = normalize_colnames(cd_inventory)

print("\nLoaded StatCan population/dwelling table")
print("Rows:", len(table))
print("Columns:", list(table.columns))
print("Encoding used:", table_encoding)

print("\nLoaded Quebec census division inventory")
print("Rows:", len(cd_inventory))
print("Columns:", list(cd_inventory.columns))

if RAW_METADATA.exists():
    metadata, metadata_encoding = read_csv_fallback(RAW_METADATA)
    metadata = normalize_colnames(metadata)
    print("\nLoaded metadata")
    print("Rows:", len(metadata))
    print("Columns:", list(metadata.columns))
    print("Encoding used:", metadata_encoding)
else:
    metadata = pd.DataFrame()
    print("\nMetadata file not found; continuing without it.")


# -----------------------------
# Column inventory
# -----------------------------

columns_inventory = pd.DataFrame(
    {
        "column_name": table.columns,
        "non_missing_count": [table[c].notna().sum() for c in table.columns],
        "missing_count": [table[c].isna().sum() for c in table.columns],
        "example_values": [
            ", ".join(table[c].dropna().astype(str).head(8).tolist())
            for c in table.columns
        ],
    }
)

columns_inventory.to_csv(OUTPUT_COLUMNS, index=False)

print("\nColumn inventory:")
print(columns_inventory.to_string(index=False))


# -----------------------------
# Identify likely columns
# -----------------------------

print("\nUnique values by likely geography columns:")

for col in table.columns:
    col_lower = col.lower()
    if (
        "geo" in col_lower
        or "province" in col_lower
        or "territory" in col_lower
        or "type" in col_lower
    ):
        n_unique = table[col].nunique(dropna=True)
        print(f"\n{col} — unique values: {n_unique}")
        print(table[col].dropna().astype(str).drop_duplicates().head(40).to_string(index=False))


# -----------------------------
# Expected columns from display table
# -----------------------------

expected_display_cols = [
    "Geographic name",
    "Geographic area type abbreviation",
    "Province or territory abbreviation",
    "Population, 2021",
    "Population, 2016",
    "Population percentage change, 2016 to 2021",
    "Total private dwellings, 2021",
    "Total private dwellings, 2016",
    "Private dwellings occupied by usual residents, 2021",
    "Private dwellings occupied by usual residents, 2016",
    "Land area in square kilometres, 2021",
    "Population density per square kilometre, 2021",
]

missing_display_cols = [c for c in expected_display_cols if c not in table.columns]

if missing_display_cols:
    print("\nWARNING: Missing expected display-style columns:")
    for col in missing_display_cols:
        print(" ", col)
else:
    print("\nAll expected display-style columns found.")


# -----------------------------
# Filter Quebec census divisions
# -----------------------------

required_cols = [
    "Geographic name",
    "Geographic area type abbreviation",
    "Province or territory abbreviation",
    "Population, 2021",
]

missing_required = [c for c in required_cols if c not in table.columns]

if missing_required:
    raise ValueError(
        "Cannot continue because required columns are missing:\n"
        + "\n".join(missing_required)
        + "\n\nInspect the column inventory to adjust the script."
    )

df = table.copy()

df["geographic_name_clean"] = (
    clean_text(df["Geographic name"])
    .str.replace(r"\s*\(map\)\s*$", "", regex=True)
    .str.replace(r"\d+$", "", regex=True)
    .str.strip()
)

df["geo_type_clean"] = clean_text(df["Geographic area type abbreviation"])
df["province_abbrev_clean"] = clean_text(df["Province or territory abbreviation"])

qc = df[
    df["province_abbrev_clean"].isin(["Que.", "QC", "Qc", "Quebec", "Québec"])
].copy()

print("\nQuebec rows found:", len(qc))

qc_cd = qc[
    qc["geo_type_clean"].isin(["MRC", "TÉ", "CDR"])
].copy()

print("Quebec census-division-like rows found:", len(qc_cd))

qc_cd["population_2021_numeric"] = clean_number(qc_cd["Population, 2021"])

if "Population, 2016" in qc_cd.columns:
    qc_cd["population_2016_numeric"] = clean_number(qc_cd["Population, 2016"])

if "Land area in square kilometres, 2021" in qc_cd.columns:
    qc_cd["land_area_km2_table_numeric"] = clean_number(
        qc_cd["Land area in square kilometres, 2021"]
    )

if "Population density per square kilometre, 2021" in qc_cd.columns:
    qc_cd["population_density_2021_numeric"] = clean_number(
        qc_cd["Population density per square kilometre, 2021"]
    )

qc_cd.to_csv(OUTPUT_QC_PREVIEW, index=False)

print("\nQuebec CD preview:")
preview_cols = [
    "geographic_name_clean",
    "geo_type_clean",
    "province_abbrev_clean",
    "Population, 2021",
    "population_2021_numeric",
    "Population, 2016",
    "Land area in square kilometres, 2021",
    "Population density per square kilometre, 2021",
]
preview_cols = [c for c in preview_cols if c in qc_cd.columns]
print(qc_cd[preview_cols].head(120).to_string(index=False))


# -----------------------------
# Validate against CD inventory by name/type
# -----------------------------

required_cd_cols = ["CDUID", "DGUID", "CDNAME", "CDTYPE", "LANDAREA", "PRUID"]
missing_cd_cols = [c for c in required_cd_cols if c not in cd_inventory.columns]

if missing_cd_cols:
    raise ValueError(
        "CD inventory missing required columns:\n"
        + "\n".join(missing_cd_cols)
    )

cd_inv = cd_inventory.copy()
cd_inv["cdname_clean"] = clean_text(cd_inv["CDNAME"])
cd_inv["cdtype_clean"] = clean_text(cd_inv["CDTYPE"])

validation = cd_inv.merge(
    qc_cd,
    left_on=["cdname_clean", "cdtype_clean"],
    right_on=["geographic_name_clean", "geo_type_clean"],
    how="left",
    indicator=True,
    validate="one_to_one",
)

validation["matched_population_table"] = validation["_merge"] == "both"

validation_cols = [
    "CDUID",
    "DGUID",
    "CDNAME",
    "CDTYPE",
    "LANDAREA",
    "PRUID",
    "matched_population_table",
    "geographic_name_clean",
    "geo_type_clean",
    "Population, 2021",
    "population_2021_numeric",
    "Population, 2016",
    "population_2016_numeric",
    "Land area in square kilometres, 2021",
    "land_area_km2_table_numeric",
    "Population density per square kilometre, 2021",
    "population_density_2021_numeric",
]
validation_cols = [c for c in validation_cols if c in validation.columns]

validation[validation_cols].to_csv(OUTPUT_QC_VALIDATION, index=False)

print("\nValidation summary")
print("Quebec CD inventory rows:", len(cd_inv))
print("Quebec population table CD-like rows:", len(qc_cd))
print(
    "Inventory CDs matched to population table:",
    int(validation["matched_population_table"].sum()),
)
print(
    "Inventory CDs NOT matched to population table:",
    int((~validation["matched_population_table"]).sum()),
)

unmatched = validation[~validation["matched_population_table"]]

if not unmatched.empty:
    print("\nUnmatched CD inventory rows:")
    print(unmatched[["CDUID", "CDNAME", "CDTYPE"]].to_string(index=False))


# -----------------------------
# Save paths
# -----------------------------

print("\nSaved:")
print(OUTPUT_COLUMNS)
print(OUTPUT_QC_PREVIEW)
print(OUTPUT_QC_VALIDATION)

print("\nInterpretation:")
print("- If 98 inventory CDs matched, this table is ready for the final reusable cleaner.")
print("- If some rows are unmatched, we may need a small manual name/type harmonization map.")
print("\nDone.")