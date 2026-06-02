from pathlib import Path
import pandas as pd


# ============================================================
# Inspect ODHF Residential Care Facilities for SoVI Proxy
# ============================================================
#
# Purpose:
#   Inspect the Open Database of Healthcare Facilities (ODHF) v1.1 file
#   to determine whether it can support a census-division-level proxy for
#   the SoVI nursing-home / residential-care variable.
#
# Source:
#   Statistics Canada — Open Database of Healthcare Facilities (ODHF)
#   Catalogue number: 13260001
#   Version: 1.1
#
# Important:
#   This script does NOT clean the final SoVI variable yet.
#   It only inspects:
#     - Quebec residential-care records
#     - CSD identifiers
#     - missing CSDuid
#     - missing coordinates
#     - counts by CSD
#
# Methodological note:
#   ODHF provides facility records, not resident counts or bed counts.
#   Therefore this supports a facility-count proxy, not exact
#   "CHSLD residents per capita".
#
# Shared raw data:
#   This script reuses:
#       hospitals_per_capita/raw/odhf_v1.1.csv
#
# Run from data/:
#   python residential_care_per_capita/inspect_odhf_residential_care.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = DATA_DIR / "hospitals_per_capita" / "raw" / "odhf_v1.1.csv"

OUTPUT_DIR = DATA_DIR / "residential_care_per_capita" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_COLUMNS_CSV = OUTPUT_DIR / "odhf_columns_inventory.csv"
OUTPUT_FACILITY_TYPES_CSV = OUTPUT_DIR / "odhf_facility_type_counts.csv"
OUTPUT_PROVINCE_COUNTS_CSV = OUTPUT_DIR / "odhf_province_counts.csv"
OUTPUT_QUEBEC_RESIDENTIAL_CARE_CSV = (
    OUTPUT_DIR / "odhf_quebec_residential_care_preview.csv"
)
OUTPUT_QUEBEC_RESIDENTIAL_CARE_BY_CSD_CSV = (
    OUTPUT_DIR / "odhf_quebec_residential_care_counts_by_csd.csv"
)
OUTPUT_MISSING_CSD_CSV = (
    OUTPUT_DIR / "odhf_quebec_residential_care_missing_csd.csv"
)
OUTPUT_MISSING_COORDS_CSV = (
    OUTPUT_DIR / "odhf_quebec_residential_care_missing_coordinates.csv"
)


# -----------------------------
# Helpers
# -----------------------------

def read_csv_with_encoding_fallback(path: Path) -> tuple[pd.DataFrame, str]:
    """
    Try reading a CSV using several common encodings.

    ODHF v1.1 may contain non-UTF-8 characters. In previous inspection,
    cp1252 was the encoding that correctly loaded the file.
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
    raise FileNotFoundError(
        f"Raw ODHF file not found:\n{RAW_PATH}\n\n"
        "Expected shared source file:\n"
        "hospitals_per_capita/raw/odhf_v1.1.csv"
    )

df, detected_encoding = read_csv_with_encoding_fallback(RAW_PATH)

df.columns = [str(col).strip() for col in df.columns]

print("\nLoaded ODHF file")
print("Rows:", len(df))
print("Columns:", list(df.columns))
print("Detected encoding used:", detected_encoding)


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
        "Pruid",
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
# Filter Quebec residential-care facilities
# -----------------------------
# ODHF has a standard category:
#   "Nursing and residential care facilities"
#
# Previous hospital inspection also showed a lowercase variant:
#   "nursing and residential care facilities"
#
# We normalize to lowercase before filtering.

facility_type_normalized = (
    qc[facility_type_col]
    .astype("string")
    .str.strip()
    .str.lower()
)

residential_care_mask = facility_type_normalized.isin(
    [
        "nursing and residential care facilities",
    ]
)

qc_residential = qc[residential_care_mask].copy()

print("\nQuebec residential-care records found:", len(qc_residential))

if qc_residential.empty and not qc.empty:
    print("\nWARNING: No Quebec residential-care records found.")
    print("Quebec facility type values:")
    print(qc[facility_type_col].fillna("MISSING").value_counts().to_string())


# -----------------------------
# Clean CSD / province / coordinate fields
# -----------------------------

if csd_uid_col is not None:
    qc_residential["CSDuid_clean"] = (
        qc_residential[csd_uid_col]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

if pr_uid_col is not None:
    qc_residential["PRuid_clean"] = (
        qc_residential[pr_uid_col]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

if latitude_col is not None:
    qc_residential["latitude_numeric"] = pd.to_numeric(
        qc_residential[latitude_col],
        errors="coerce",
    )

if longitude_col is not None:
    qc_residential["longitude_numeric"] = pd.to_numeric(
        qc_residential[longitude_col],
        errors="coerce",
    )


# -----------------------------
# Preview Quebec residential-care records
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
    if col is not None and col in qc_residential.columns
]

print("\nQuebec residential-care preview:")
if qc_residential.empty:
    print("No Quebec residential-care records to preview.")
else:
    print(qc_residential[preview_cols].head(50).to_string(index=False))

qc_residential[preview_cols].to_csv(OUTPUT_QUEBEC_RESIDENTIAL_CARE_CSV, index=False)


# -----------------------------
# Residential-care counts by CSD
# -----------------------------

if not qc_residential.empty and "CSDuid_clean" in qc_residential.columns:
    group_cols = ["CSDuid_clean"]

    if csd_name_col is not None and csd_name_col in qc_residential.columns:
        group_cols.append(csd_name_col)

    if "PRuid_clean" in qc_residential.columns:
        group_cols.append("PRuid_clean")

    residential_by_csd = (
        qc_residential
        .groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="residential_care_facility_count")
        .sort_values(
            "residential_care_facility_count",
            ascending=False,
        )
    )

    residential_by_csd.to_csv(
        OUTPUT_QUEBEC_RESIDENTIAL_CARE_BY_CSD_CSV,
        index=False,
    )

    print("\nQuebec residential-care counts by CSD:")
    print(residential_by_csd.to_string(index=False))
else:
    residential_by_csd = pd.DataFrame()
    print("\nNo residential-care-by-CSD table generated.")
    print("Reason: no Quebec residential-care records found or no CSDuid column detected.")


# -----------------------------
# Missing CSD diagnostics
# -----------------------------

missing_csd = pd.DataFrame()

if not qc_residential.empty and "CSDuid_clean" in qc_residential.columns:
    missing_csd = qc_residential[
        qc_residential["CSDuid_clean"].isna()
        | (qc_residential["CSDuid_clean"].astype(str).str.strip() == "")
        | (qc_residential["CSDuid_clean"].astype(str).str.lower() == "nan")
    ].copy()

    print("\nQuebec residential-care records missing CSDuid:", len(missing_csd))

    if not missing_csd.empty:
        print(missing_csd[preview_cols].to_string(index=False))
        missing_csd[preview_cols].to_csv(OUTPUT_MISSING_CSD_CSV, index=False)


# -----------------------------
# Missing coordinate diagnostics
# -----------------------------

missing_coords = pd.DataFrame()

if (
    not qc_residential.empty
    and "latitude_numeric" in qc_residential.columns
    and "longitude_numeric" in qc_residential.columns
):
    missing_coords = qc_residential[
        qc_residential["latitude_numeric"].isna()
        | qc_residential["longitude_numeric"].isna()
    ].copy()

    print("\nQuebec residential-care records missing coordinates:", len(missing_coords))

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
print(OUTPUT_QUEBEC_RESIDENTIAL_CARE_CSV)

if not residential_by_csd.empty:
    print(OUTPUT_QUEBEC_RESIDENTIAL_CARE_BY_CSD_CSV)

if not missing_csd.empty:
    print(OUTPUT_MISSING_CSD_CSV)

if not missing_coords.empty:
    print(OUTPUT_MISSING_COORDS_CSV)

print("\nInterpretation reminder:")
print("- This inspection counts ODHF records classified as Nursing and residential care facilities.")
print("- ODHF gives facility records, not CHSLD resident counts or bed counts.")
print("- This supports a residential-care facility-count proxy for the SoVI nursing-home variable.")
print("- ODHF is open and useful, but not guaranteed exhaustive.")
print("- Next step after inspection: derive CDUID from CSDuid and count residential-care records by CD.")

print("\nDone.")