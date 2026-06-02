from pathlib import Path
import pandas as pd


# ============================================================
# Clean Census Division Hospital Counts from ODHF
# ============================================================
#
# Purpose:
#   Build a clean census-division-level hospital count table from the
#   Open Database of Healthcare Facilities (ODHF).
#
# This script combines:
#   1. ODHF Quebec hospital records automatically assigned to census divisions
#      using CSDuid -> first 4 digits = CDUID.
#   2. ODHF Quebec hospital records missing CSDuid that were manually repaired.
#
# Output:
#   hospitals_per_capita/output/clean_census_division_hospital_counts_odhf_2021.csv
#   hospitals_per_capita/output/clean_census_division_hospital_counts_odhf_2021.parquet
#
# It also saves a hospital-level audit table:
#   hospitals_per_capita/output/clean_odhf_quebec_hospitals_assigned_to_census_divisions.csv
#
# Important:
#   This script computes hospital counts only.
#   It does NOT compute hospitals per capita or hospitals per 100k.
#   Population denominators should be handled later in a reusable
#   census-division population folder.
#
# Run from data/:
#   python hospitals_per_capita/clean_census_division_hospital_counts_odhf_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

AUTO_ASSIGNED_PATH = (
    DATA_DIR
    / "hospitals_per_capita"
    / "output"
    / "odhf_quebec_hospitals_with_derived_cd.csv"
)

MANUAL_REPAIR_PATH = (
    DATA_DIR
    / "hospitals_per_capita"
    / "lookup"
    / "odhf_quebec_hospitals_missing_csd_manual_repair_filled.csv"
)

CD_INVENTORY_PATH = (
    DATA_DIR
    / "doctors_per_100khabs"
    / "output"
    / "quebec_census_divisions_2021_inventory.csv"
)

OUTPUT_DIR = DATA_DIR / "hospitals_per_capita" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_COUNTS_CSV = OUTPUT_DIR / "clean_census_division_hospital_counts_odhf_2021.csv"
OUTPUT_COUNTS_PARQUET = OUTPUT_DIR / "clean_census_division_hospital_counts_odhf_2021.parquet"

OUTPUT_ASSIGNED_HOSPITALS_CSV = (
    OUTPUT_DIR / "clean_odhf_quebec_hospitals_assigned_to_census_divisions.csv"
)

OUTPUT_ASSIGNED_HOSPITALS_PARQUET = (
    OUTPUT_DIR / "clean_odhf_quebec_hospitals_assigned_to_census_divisions.parquet"
)


# -----------------------------
# Constants
# -----------------------------

SOURCE_NAME = "Statistics Canada Open Database of Healthcare Facilities (ODHF), version 1.1"
SOURCE_CATALOGUE = "13260001"
SOURCE_ODHF_VERSION = "1.1"

# ODHF v1.1 was released in 2020. We use 2021 census division geography.
ODHF_SOURCE_YEAR = 2020
GEOGRAPHY_YEAR = 2021


# -----------------------------
# Helpers
# -----------------------------

def normalize_bool_series(series: pd.Series) -> pd.Series:
    """
    Convert a mixed string/bool series to boolean.
    """
    return (
        series
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def require_columns(df: pd.DataFrame, required_cols: list[str], label: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {label}:\n"
            + "\n".join(missing)
        )


# -----------------------------
# Load inputs
# -----------------------------

for path in [AUTO_ASSIGNED_PATH, MANUAL_REPAIR_PATH, CD_INVENTORY_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found:\n{path}")

auto_raw = pd.read_csv(AUTO_ASSIGNED_PATH, dtype=str)
manual_raw = pd.read_csv(MANUAL_REPAIR_PATH, dtype=str)
cd_inventory = pd.read_csv(CD_INVENTORY_PATH, dtype=str)

print("\nLoaded automatically assigned ODHF hospital records")
print("Rows:", len(auto_raw))
print("Columns:", list(auto_raw.columns))

print("\nLoaded manually repaired ODHF hospital records")
print("Rows:", len(manual_raw))
print("Columns:", list(manual_raw.columns))

print("\nLoaded Quebec census division inventory")
print("Rows:", len(cd_inventory))
print("Columns:", list(cd_inventory.columns))


# -----------------------------
# Validate columns
# -----------------------------

required_auto_cols = [
    "index",
    "facility_name",
    "source_facility_type",
    "odhf_facility_type",
    "provider",
    "city",
    "province",
    "CSDname",
    "CSDuid",
    "derived_cd_code",
    "census_division_dguid",
    "census_division_name",
    "census_division_type",
    "census_division_land_area_km2",
    "province_code",
    "matched_to_cd_inventory",
]

required_manual_cols = [
    "index",
    "facility_name",
    "source_facility_type",
    "odhf_facility_type",
    "provider",
    "city",
    "province",
    "CSDname",
    "CSDuid",
    "manual_cd_code",
    "manual_census_division_name",
    "manual_census_division_dguid",
    "manual_repair_method",
    "manual_repair_note",
]

required_cd_cols = [
    "CDUID",
    "DGUID",
    "CDNAME",
    "CDTYPE",
    "LANDAREA",
    "PRUID",
]

require_columns(auto_raw, required_auto_cols, "automatically assigned hospital file")
require_columns(manual_raw, required_manual_cols, "manual repair file")
require_columns(cd_inventory, required_cd_cols, "census division inventory")


# -----------------------------
# Prepare census division lookup
# -----------------------------

cd_lookup = cd_inventory[
    [
        "CDUID",
        "DGUID",
        "CDNAME",
        "CDTYPE",
        "LANDAREA",
        "PRUID",
    ]
].copy()

cd_lookup = cd_lookup.rename(
    columns={
        "CDUID": "census_division_code",
        "DGUID": "census_division_dguid",
        "CDNAME": "census_division_name",
        "CDTYPE": "census_division_type",
        "LANDAREA": "census_division_land_area_km2",
        "PRUID": "province_code",
    }
)

for col in [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "census_division_type",
    "province_code",
]:
    cd_lookup[col] = cd_lookup[col].astype("string").str.strip()

cd_lookup["census_division_land_area_km2"] = pd.to_numeric(
    cd_lookup["census_division_land_area_km2"],
    errors="coerce",
)

if cd_lookup["census_division_code"].duplicated().any():
    duplicated = cd_lookup[cd_lookup["census_division_code"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated census division codes in CD inventory:\n"
        + duplicated.to_string(index=False)
    )


# -----------------------------
# Build automatically assigned hospital table
# -----------------------------

auto = auto_raw.copy()
auto["matched_to_cd_inventory_bool"] = normalize_bool_series(
    auto["matched_to_cd_inventory"]
)

auto_matched = auto[auto["matched_to_cd_inventory_bool"]].copy()

print("\nAutomatically matched hospitals:", len(auto_matched))

auto_assigned = pd.DataFrame(
    {
        "odhf_index": auto_matched["index"].astype(str).str.strip(),
        "facility_name": auto_matched["facility_name"],
        "source_facility_type": auto_matched["source_facility_type"],
        "odhf_facility_type": auto_matched["odhf_facility_type"],
        "provider": auto_matched["provider"],
        "city": auto_matched["city"],
        "province": auto_matched["province"],
        "csd_name": auto_matched["CSDname"],
        "csd_uid": auto_matched["CSDuid"],
        "census_division_code": auto_matched["derived_cd_code"].astype(str).str.strip(),
        "census_division_dguid": auto_matched["census_division_dguid"],
        "census_division_name": auto_matched["census_division_name"],
        "census_division_type": auto_matched["census_division_type"],
        "census_division_land_area_km2": auto_matched[
            "census_division_land_area_km2"
        ],
        "province_code": auto_matched["province_code"],
        "assignment_source": "automatic_csd_uid_prefix",
        "assignment_method": "automatic_csd_uid_to_cd_first_4_digits",
        "assignment_note": (
            "Census division code derived from first 4 digits of ODHF CSDuid; "
            "derived code validated against 2021 Quebec census division inventory."
        ),
    }
)


# -----------------------------
# Build manually repaired hospital table
# -----------------------------

manual = manual_raw.copy()

manual_assigned_base = pd.DataFrame(
    {
        "odhf_index": manual["index"].astype(str).str.strip(),
        "facility_name": manual["facility_name"],
        "source_facility_type": manual["source_facility_type"],
        "odhf_facility_type": manual["odhf_facility_type"],
        "provider": manual["provider"],
        "city": manual["city"],
        "province": manual["province"],
        "csd_name": manual["CSDname"],
        "csd_uid": manual["CSDuid"],
        "census_division_code": manual["manual_cd_code"].astype(str).str.strip(),
        "census_division_dguid": manual["manual_census_division_dguid"],
        "census_division_name": manual["manual_census_division_name"],
        "assignment_source": "manual_missing_csd_repair",
        "assignment_method": manual["manual_repair_method"],
        "assignment_note": manual["manual_repair_note"],
    }
)

manual_assigned = manual_assigned_base.merge(
    cd_lookup[
        [
            "census_division_code",
            "census_division_type",
            "census_division_land_area_km2",
            "province_code",
        ]
    ],
    on="census_division_code",
    how="left",
    validate="many_to_one",
)

print("Manually repaired hospitals:", len(manual_assigned))


# -----------------------------
# Combine assigned hospital records
# -----------------------------

assigned = pd.concat(
    [auto_assigned, manual_assigned],
    ignore_index=True,
)

assigned["census_division_land_area_km2"] = pd.to_numeric(
    assigned["census_division_land_area_km2"],
    errors="coerce",
)

# Add source metadata.
assigned["source_hospitals"] = SOURCE_NAME
assigned["source_catalogue"] = SOURCE_CATALOGUE
assigned["source_odhf_version"] = SOURCE_ODHF_VERSION
assigned["odhf_source_year"] = ODHF_SOURCE_YEAR
assigned["geography_year"] = GEOGRAPHY_YEAR


# -----------------------------
# Validate combined hospital-level table
# -----------------------------

if assigned["odhf_index"].duplicated().any():
    duplicated = assigned[assigned["odhf_index"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated ODHF hospital indexes after combining automatic and manual records:\n"
        + duplicated.sort_values("odhf_index").to_string(index=False)
    )

required_assigned_non_missing = [
    "odhf_index",
    "facility_name",
    "odhf_facility_type",
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "assignment_source",
    "assignment_method",
]

for col in required_assigned_non_missing:
    missing_mask = (
        assigned[col].isna()
        | (assigned[col].astype(str).str.strip() == "")
        | (assigned[col].astype(str).str.lower() == "nan")
    )
    if missing_mask.any():
        raise ValueError(
            f"Missing values found in assigned hospital column {col}:\n"
            + assigned.loc[
                missing_mask,
                ["odhf_index", "facility_name", "city", "csd_name", col],
            ].to_string(index=False)
        )

unknown_cd_codes = sorted(
    set(assigned["census_division_code"].astype(str).str.strip())
    - set(cd_lookup["census_division_code"].astype(str).str.strip())
)

if unknown_cd_codes:
    raise ValueError(
        "Assigned hospital records contain CD codes not found in CD inventory:\n"
        + "\n".join(unknown_cd_codes)
    )

expected_total_records = len(auto_raw)
if len(assigned) != expected_total_records:
    raise ValueError(
        f"Expected {expected_total_records} assigned hospitals, but got {len(assigned)}. "
        "This should equal the full Quebec ODHF hospital count."
    )


# -----------------------------
# Count hospitals by census division
# -----------------------------

counts = (
    assigned
    .groupby("census_division_code", dropna=False)
    .agg(
        hospital_count_odhf=("odhf_index", "count"),
        hospital_count_automatic_csd_uid=("assignment_source", lambda x: (x == "automatic_csd_uid_prefix").sum()),
        hospital_count_manual_repair=("assignment_source", lambda x: (x == "manual_missing_csd_repair").sum()),
    )
    .reset_index()
)

clean_counts = cd_lookup.merge(
    counts,
    on="census_division_code",
    how="left",
    validate="one_to_one",
)

for col in [
    "hospital_count_odhf",
    "hospital_count_automatic_csd_uid",
    "hospital_count_manual_repair",
]:
    clean_counts[col] = clean_counts[col].fillna(0).astype(int)

clean_counts["unit_type"] = "census_division"
clean_counts["geography_year"] = GEOGRAPHY_YEAR
clean_counts["odhf_source_year"] = ODHF_SOURCE_YEAR
clean_counts["source_hospitals"] = SOURCE_NAME
clean_counts["source_catalogue"] = SOURCE_CATALOGUE
clean_counts["source_odhf_version"] = SOURCE_ODHF_VERSION

clean_counts["feature_description"] = (
    "Count of ODHF records classified as Hospitals assigned to 2021 Quebec census divisions. "
    "Most records are assigned automatically from CSDuid to CDUID; records missing CSDuid are assigned "
    "using a documented manual repair lookup. This is a facility-record count proxy, not necessarily an "
    "official count of unique hospital systems."
)

clean_counts = clean_counts[
    [
        "census_division_code",
        "census_division_dguid",
        "census_division_name",
        "census_division_type",
        "census_division_land_area_km2",
        "province_code",
        "unit_type",
        "geography_year",
        "odhf_source_year",
        "hospital_count_odhf",
        "hospital_count_automatic_csd_uid",
        "hospital_count_manual_repair",
        "source_hospitals",
        "source_catalogue",
        "source_odhf_version",
        "feature_description",
    ]
].sort_values("census_division_code")


# -----------------------------
# Final validation
# -----------------------------

if len(clean_counts) != len(cd_lookup):
    raise ValueError(
        f"Expected one row per Quebec census division: {len(cd_lookup)}. "
        f"Got {len(clean_counts)}."
    )

if clean_counts["census_division_code"].duplicated().any():
    duplicated = clean_counts[
        clean_counts["census_division_code"].duplicated(keep=False)
    ]
    raise ValueError(
        "Duplicated census_division_code values in final count table:\n"
        + duplicated.to_string(index=False)
    )

if clean_counts["hospital_count_odhf"].sum() != len(assigned):
    raise ValueError(
        "Hospital count sum does not equal number of assigned hospital records. "
        f"Sum={clean_counts['hospital_count_odhf'].sum()}, assigned={len(assigned)}"
    )

if (
    clean_counts["hospital_count_automatic_csd_uid"].sum()
    + clean_counts["hospital_count_manual_repair"].sum()
    != clean_counts["hospital_count_odhf"].sum()
):
    raise ValueError(
        "Automatic + manual hospital counts do not equal total hospital counts."
    )


# -----------------------------
# Save outputs
# -----------------------------

assigned.to_csv(OUTPUT_ASSIGNED_HOSPITALS_CSV, index=False)
assigned.to_parquet(OUTPUT_ASSIGNED_HOSPITALS_PARQUET, index=False)

clean_counts.to_csv(OUTPUT_COUNTS_CSV, index=False)
clean_counts.to_parquet(OUTPUT_COUNTS_PARQUET, index=False)


# -----------------------------
# Diagnostics
# -----------------------------

print("\nFinal assigned hospital-level table")
print("Rows:", len(assigned))
print("Automatic assignments:", int((assigned["assignment_source"] == "automatic_csd_uid_prefix").sum()))
print("Manual repairs:", int((assigned["assignment_source"] == "manual_missing_csd_repair").sum()))

print("\nFinal census-division hospital count table")
print("Rows:", len(clean_counts))
print("Total hospital_count_odhf:", int(clean_counts["hospital_count_odhf"].sum()))
print("Census divisions with at least one hospital:", int((clean_counts["hospital_count_odhf"] > 0).sum()))
print("Census divisions with zero hospitals:", int((clean_counts["hospital_count_odhf"] == 0).sum()))

print("\nTop census divisions by ODHF hospital count:")
print(
    clean_counts[
        [
            "census_division_code",
            "census_division_name",
            "hospital_count_odhf",
            "hospital_count_automatic_csd_uid",
            "hospital_count_manual_repair",
        ]
    ]
    .sort_values("hospital_count_odhf", ascending=False)
    .head(20)
    .to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_ASSIGNED_HOSPITALS_CSV)
print(OUTPUT_ASSIGNED_HOSPITALS_PARQUET)
print(OUTPUT_COUNTS_CSV)
print(OUTPUT_COUNTS_PARQUET)

print("\nDone.")