from pathlib import Path
import pandas as pd


# ============================================================
# Inspect ODHF Hospitals for SoVI Hospital per Capita Variable
# ============================================================
#
# Purpose:
#   Inspect the Open Database of Healthcare Facilities (ODHF) v1.1 file
#   to understand whether it can support a census-division-level hospital
#   count / hospital-per-capita proxy for SoVI.
#
# Source:
#   Statistics Canada — Open Database of Healthcare Facilities (ODHF)
#   Catalogue number: 13260001
#   Version: 1.1
#
# Important:
#   This script does NOT clean the final SoVI variable yet.
#   It only inspects:
#     - columns
#     - facility type values
#     - Quebec records
#     - Quebec hospital records
#     - CSD identifiers
#     - missing coordinates / missing CSDUID
#
# Encoding note:
#   The ODHF CSV may not be UTF-8. This script tries several common encodings.
#
# Run from data/:
#   python hospitals_per_capita/inspect_odhf_hospitals.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = DATA_DIR / "hospitals_per_capita" / "raw" / "odhf_v1.1.csv"

OUTPUT_DIR = DATA_DIR / "hospitals_per_capita" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_COLUMNS_CSV = OUTPUT_DIR / "odhf_columns_inventory.csv"
OUTPUT_FACILITY_TYPES_CSV = OUTPUT_DIR / "odhf_facility_type_counts.csv"
OUTPUT_PROVINCE_COUNTS_CSV = OUTPUT_DIR / "odhf_province_counts.csv"
OUTPUT_QUEBEC_HOSPITALS_CSV = OUTPUT_DIR / "odhf_quebec_hospitals_preview.csv"
OUTPUT_QUEBEC_HOSPITALS_BY_CSD_CSV = OUTPUT_DIR / "odhf_quebec_hospital_counts_by_csd.csv"
OUTPUT_MISSING_CSD_CSV = OUTPUT_DIR / "odhf_quebec_hospitals_missing_csd.csv"
OUTPUT_MISSING_COORDS_CSV = OUTPUT_DIR / "odhf_quebec_hospitals_missing_coordinates.csv"


# -----------------------------
# Helpers
# -----------------------------

def read_csv_with_encoding_fallback(path: Path) -> tuple[pd.DataFrame, str]:
    """
    Try reading a CSV using several common encodings.

    ODHF v1.1 can contain non-UTF-8 characters. The byte 0x96 often indicates
    Windows-1252 punctuation, such as an en dash.
    """
    encodings_to_try = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1",
        "iso-8859-1",
    ]

    last_error = None

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(
                path,
                dtype=str,
                low_memory=False,
                encoding=encoding,
            )
            return df, encoding
        except UnicodeDecodeError as e:
            last_error = e
            print(f"Could not read with encoding={encoding}: {e}")

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not read {path} with any tested encoding. Last error: {last_error}",
    )


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the first column name that exists in df.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_text_series(series: pd.Series) -> pd.Series:
    """
    Normalize strings for filtering.
    """
    return (
        series
        .astype("string")
        .str.strip()
    )


# -----------------------------
# Load raw ODHF file
# -----------------------------

if not RAW_PATH.exists():
    raise FileNotFoundError(f"Raw ODHF file not found:\n{RAW_PATH}")

df, detected_encoding = read_csv_with_encoding_fallback(RAW_PATH)

print("\nLoaded ODHF file")
print("Rows:", len(df))
print("Columns:", list(df.columns))
print("Detected encoding used:", detected_encoding)


# -----------------------------
# Normalize column names lightly
# -----------------------------
# The metadata says expected names include:
#   index, facility_name, source_facility_type, odhf_facility_type,
#   provider, unit, street_no, street_name, postal_code, city, province,
#   source_format_str_address, CSDname, CSDuid, PRuid, latitude, longitude.
#
# We preserve original names, but strip whitespace just in case.

df.columns = [str(col).strip() for col in df.columns]


# -----------------------------
# Column inventory
# -----------------------------

columns_inventory = pd.DataFrame(
    {
        "column_name": df.columns,
        "non_missing_count": [df[col].notna().sum() for col in df.columns],
        "missing_count": [df[col].isna().sum() for col in df.columns],
        "example_values": [
            ", ".join(df[col].dropna().astype(str).head(5).tolist())
            for col in df.columns
        ],
    }
)

columns_inventory.to_csv(OUTPUT_COLUMNS_CSV, index=False)

print("\nColumn inventory:")
print(columns_inventory.to_string(index=False))


# -----------------------------
# Identify important columns robustly
# -----------------------------

facility_type_col = first_existing_column(
    df,
    [
        "odhf_facility_type",
        "ODHF Facility Type",
        "ODHF_Facility_Type",
        "facility_type",
    ],
)

province_col = first_existing_column(
    df,
    [
        "province",
        "Province",
        "Province/Territory",
        "Province or Territory",
    ],
)

csd_uid_col = first_existing_column(
    df,
    [
        "CSDuid",
        "CSDUID",
        "CSD uid",
        "Census Subdivision Unique Identifier",
    ],
)

csd_name_col = first_existing_column(
    df,
    [
        "CSDname",
        "CSDNAME",
        "CSD name",
        "Census Subdivision Name",
    ],
)

pr_uid_col = first_existing_column(
    df,
    [
        "PRuid",
        "PRUID",
        "Province or Territory Unique Identifier",
    ],
)

latitude_col = first_existing_column(
    df,
    [
        "latitude",
        "Latitude",
        "LATITUDE",
    ],
)

longitude_col = first_existing_column(
    df,
    [
        "longitude",
        "Longitude",
        "LONGITUDE",
    ],
)

facility_name_col = first_existing_column(
    df,
    [
        "facility_name",
        "Facility Name",
        "facility",
        "name",
    ],
)

source_facility_type_col = first_existing_column(
    df,
    [
        "source_facility_type",
        "Source Facility Type",
    ],
)

provider_col = first_existing_column(
    df,
    [
        "provider",
        "Provider",
    ],
)

city_col = first_existing_column(
    df,
    [
        "city",
        "City",
    ],
)

address_col = first_existing_column(
    df,
    [
        "source_format_str_address",
        "Source-Format Street Address",
        "source_format_street_address",
    ],
)

index_col = first_existing_column(
    df,
    [
        "index",
        "Index",
    ],
)


print("\nDetected important columns:")
print("facility_type_col:", facility_type_col)
print("province_col:", province_col)
print("csd_uid_col:", csd_uid_col)
print("csd_name_col:", csd_name_col)
print("pr_uid_col:", pr_uid_col)
print("latitude_col:", latitude_col)
print("longitude_col:", longitude_col)
print("facility_name_col:", facility_name_col)
print("source_facility_type_col:", source_facility_type_col)
print("provider_col:", provider_col)
print("city_col:", city_col)
print("address_col:", address_col)
print("index_col:", index_col)


if facility_type_col is None:
    raise ValueError("Could not identify the ODHF facility type column.")

if province_col is None:
    raise ValueError("Could not identify the province column.")


# -----------------------------
# Facility type inspection
# -----------------------------

df[facility_type_col] = normalize_text_series(df[facility_type_col])

facility_type_counts = (
    df[facility_type_col]
    .fillna("MISSING")
    .value_counts(dropna=False)
    .rename_axis("odhf_facility_type")
    .reset_index(name="count")
)

facility_type_counts.to_csv(OUTPUT_FACILITY_TYPES_CSV, index=False)

print("\nODHF facility type counts:")
print(facility_type_counts.to_string(index=False))


# -----------------------------
# Province inspection
# -----------------------------

df[province_col] = normalize_text_series(df[province_col])

province_counts = (
    df[province_col]
    .fillna("MISSING")
    .value_counts(dropna=False)
    .rename_axis("province")
    .reset_index(name="count")
)

province_counts.to_csv(OUTPUT_PROVINCE_COUNTS_CSV, index=False)

print("\nProvince counts:")
print(province_counts.to_string(index=False))


# -----------------------------
# Filter to Quebec
# -----------------------------
# The ODHF metadata says province is stored as a province/territory value.
# It may appear as "QC", "Quebec", or "Québec" depending on the file.

province_normalized = (
    df[province_col]
    .astype("string")
    .str.strip()
    .str.lower()
)

quebec_mask = province_normalized.isin(
    [
        "qc",
        "que",
        "que.",
        "quebec",
        "québec",
    ]
)

qc = df[quebec_mask].copy()

print("\nQuebec records found:", len(qc))

if qc.empty:
    print("\nWARNING: No Quebec records found using expected province values.")
    print("Unique province values:")
    print(sorted(df[province_col].dropna().unique()))


# -----------------------------
# Filter Quebec hospitals
# -----------------------------

facility_type_normalized = (
    qc[facility_type_col]
    .astype("string")
    .str.strip()
    .str.lower()
)

hospital_mask = facility_type_normalized.isin(
    [
        "hospitals",
        "hospital",
    ]
)

qc_hospitals = qc[hospital_mask].copy()

print("\nQuebec hospital records found:", len(qc_hospitals))

if qc_hospitals.empty and not qc.empty:
    print("\nWARNING: No Quebec hospital records found using expected hospital type values.")
    print("Quebec facility type values:")
    print(qc[facility_type_col].fillna("MISSING").value_counts().to_string())


# -----------------------------
# Clean CSD / province / coordinate fields
# -----------------------------

if csd_uid_col is not None:
    qc_hospitals["CSDuid_clean"] = (
        qc_hospitals[csd_uid_col]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

if pr_uid_col is not None:
    qc_hospitals["PRuid_clean"] = (
        qc_hospitals[pr_uid_col]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

if latitude_col is not None:
    qc_hospitals["latitude_numeric"] = pd.to_numeric(
        qc_hospitals[latitude_col],
        errors="coerce",
    )

if longitude_col is not None:
    qc_hospitals["longitude_numeric"] = pd.to_numeric(
        qc_hospitals[longitude_col],
        errors="coerce",
    )


# -----------------------------
# Preview Quebec hospitals
# -----------------------------

preview_cols = [
    col for col in [
        index_col,
        facility_name_col,
        source_facility_type_col,
        facility_type_col,
        provider_col,
        city_col,
        province_col,
        csd_name_col,
        csd_uid_col,
        pr_uid_col,
        latitude_col,
        longitude_col,
        address_col,
    ]
    if col is not None and col in qc_hospitals.columns
]

print("\nQuebec hospital preview:")
if qc_hospitals.empty:
    print("No Quebec hospitals to preview.")
else:
    print(qc_hospitals[preview_cols].head(50).to_string(index=False))

qc_hospitals[preview_cols].to_csv(OUTPUT_QUEBEC_HOSPITALS_CSV, index=False)


# -----------------------------
# Hospital counts by CSD
# -----------------------------

if not qc_hospitals.empty and "CSDuid_clean" in qc_hospitals.columns:
    group_cols = ["CSDuid_clean"]

    if csd_name_col is not None and csd_name_col in qc_hospitals.columns:
        group_cols.append(csd_name_col)

    if "PRuid_clean" in qc_hospitals.columns:
        group_cols.append("PRuid_clean")

    hospitals_by_csd = (
        qc_hospitals
        .groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="hospital_count")
        .sort_values("hospital_count", ascending=False)
    )

    hospitals_by_csd.to_csv(OUTPUT_QUEBEC_HOSPITALS_BY_CSD_CSV, index=False)

    print("\nQuebec hospital counts by CSD:")
    print(hospitals_by_csd.to_string(index=False))
else:
    hospitals_by_csd = pd.DataFrame()
    print("\nNo hospital-by-CSD table generated.")
    print("Reason: no Quebec hospitals found or no CSDuid column detected.")


# -----------------------------
# Missing CSD diagnostics
# -----------------------------

missing_csd = pd.DataFrame()

if not qc_hospitals.empty and "CSDuid_clean" in qc_hospitals.columns:
    missing_csd = qc_hospitals[
        qc_hospitals["CSDuid_clean"].isna()
        | (qc_hospitals["CSDuid_clean"].astype(str).str.strip() == "")
    ].copy()

    print("\nQuebec hospitals missing CSDuid:", len(missing_csd))

    if not missing_csd.empty:
        print(missing_csd[preview_cols].to_string(index=False))
        missing_csd[preview_cols].to_csv(OUTPUT_MISSING_CSD_CSV, index=False)


# -----------------------------
# Missing coordinate diagnostics
# -----------------------------

missing_coords = pd.DataFrame()

if (
    not qc_hospitals.empty
    and "latitude_numeric" in qc_hospitals.columns
    and "longitude_numeric" in qc_hospitals.columns
):
    missing_coords = qc_hospitals[
        qc_hospitals["latitude_numeric"].isna()
        | qc_hospitals["longitude_numeric"].isna()
    ].copy()

    print("\nQuebec hospitals missing coordinates:", len(missing_coords))

    if not missing_coords.empty:
        print(missing_coords[preview_cols].to_string(index=False))
        missing_coords[preview_cols].to_csv(OUTPUT_MISSING_COORDS_CSV, index=False)


# -----------------------------
# Final notes
# -----------------------------

print("\nSaved outputs:")
print(OUTPUT_COLUMNS_CSV)
print(OUTPUT_FACILITY_TYPES_CSV)
print(OUTPUT_PROVINCE_COUNTS_CSV)
print(OUTPUT_QUEBEC_HOSPITALS_CSV)

if not hospitals_by_csd.empty:
    print(OUTPUT_QUEBEC_HOSPITALS_BY_CSD_CSV)

if not missing_csd.empty:
    print(OUTPUT_MISSING_CSD_CSV)

if not missing_coords.empty:
    print(OUTPUT_MISSING_COORDS_CSV)

print("\nInterpretation reminder:")
print("- This inspection only counts ODHF records classified as Hospitals.")
print("- ODHF is open and useful, but not guaranteed exhaustive.")
print("- ODHF CSDuid can let us aggregate facilities from CSD to census division.")
print("- Next step after inspection: build CSD → CD linkage, then count hospitals by CD.")

print("\nDone.")