from pathlib import Path
import pandas as pd


# ============================================================
# Clean Census Division Residential-Care Counts from ODHF
# ============================================================
#
# Purpose:
#   Build a clean census-division-level residential-care facility count table
#   from the Open Database of Healthcare Facilities (ODHF).
#
# This script uses Quebec ODHF records classified as:
#   Nursing and residential care facilities
#
# It derives census division codes from ODHF CSDuid values:
#   CSDuid example: 2466023
#   census_division_code = first 4 digits = 2466
#
# Output:
#   residential_care_per_capita/output/
#       clean_census_division_residential_care_counts_odhf_2021.csv
#       clean_census_division_residential_care_counts_odhf_2021.parquet
#
# It also saves a facility-level audit table:
#   residential_care_per_capita/output/
#       clean_odhf_quebec_residential_care_assigned_to_census_divisions.csv
#
# Important:
#   This script computes facility counts only.
#   It does NOT compute residential-care facilities per capita or per 100k.
#   Population denominators should be handled later in a reusable
#   census-division population folder.
#
# Methodological note:
#   ODHF gives facility records, not CHSLD resident counts or bed counts.
#   This is therefore a residential-care facility-count proxy for the
#   SoVI nursing-home / institutional-care variable.
#
# Run from data/:
#   python residential_care_per_capita/clean_census_division_residential_care_counts_odhf_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

RESIDENTIAL_PREVIEW_PATH = (
    DATA_DIR
    / "residential_care_per_capita"
    / "output"
    / "odhf_quebec_residential_care_preview.csv"
)

CD_INVENTORY_PATH = (
    DATA_DIR
    / "doctors_per_100khabs"
    / "output"
    / "quebec_census_divisions_2021_inventory.csv"
)

OUTPUT_DIR = DATA_DIR / "residential_care_per_capita" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_COUNTS_CSV = (
    OUTPUT_DIR / "clean_census_division_residential_care_counts_odhf_2021.csv"
)

OUTPUT_COUNTS_PARQUET = (
    OUTPUT_DIR / "clean_census_division_residential_care_counts_odhf_2021.parquet"
)

OUTPUT_ASSIGNED_FACILITIES_CSV = (
    OUTPUT_DIR
    / "clean_odhf_quebec_residential_care_assigned_to_census_divisions.csv"
)

OUTPUT_ASSIGNED_FACILITIES_PARQUET = (
    OUTPUT_DIR
    / "clean_odhf_quebec_residential_care_assigned_to_census_divisions.parquet"
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

def require_columns(df: pd.DataFrame, required_cols: list[str], label: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {label}:\n"
            + "\n".join(missing)
        )


def clean_identifier(series: pd.Series) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


# -----------------------------
# Load inputs
# -----------------------------

for path in [RESIDENTIAL_PREVIEW_PATH, CD_INVENTORY_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found:\n{path}")

residential_raw = pd.read_csv(RESIDENTIAL_PREVIEW_PATH, dtype=str)
cd_inventory = pd.read_csv(CD_INVENTORY_PATH, dtype=str)

print("\nLoaded Quebec ODHF residential-care records")
print("Rows:", len(residential_raw))
print("Columns:", list(residential_raw.columns))

print("\nLoaded Quebec census division inventory")
print("Rows:", len(cd_inventory))
print("Columns:", list(cd_inventory.columns))


# -----------------------------
# Validate columns
# -----------------------------

required_residential_cols = [
    "index",
    "facility_name",
    "source_facility_type",
    "odhf_facility_type",
    "provider",
    "city",
    "province",
    "CSDname",
    "CSDuid",
    "latitude",
    "longitude",
]

required_cd_cols = [
    "CDUID",
    "DGUID",
    "CDNAME",
    "CDTYPE",
    "LANDAREA",
    "PRUID",
]

require_columns(
    residential_raw,
    required_residential_cols,
    "Quebec residential-care preview file",
)

require_columns(
    cd_inventory,
    required_cd_cols,
    "census division inventory",
)


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
# Derive census division from CSDuid
# -----------------------------

residential = residential_raw.copy()

residential["odhf_index"] = residential["index"].astype(str).str.strip()
residential["csd_uid_clean"] = clean_identifier(residential["CSDuid"])

residential["has_csd_uid"] = (
    residential["csd_uid_clean"].notna()
    & (residential["csd_uid_clean"].astype(str).str.strip() != "")
    & (residential["csd_uid_clean"].astype(str).str.lower() != "nan")
)

if (~residential["has_csd_uid"]).any():
    missing = residential[~residential["has_csd_uid"]]
    raise ValueError(
        "Some residential-care records are missing CSDuid. "
        "This cleaner assumes no manual repair is needed.\n"
        + missing[
            [
                "odhf_index",
                "facility_name",
                "city",
                "CSDname",
                "CSDuid",
            ]
        ].to_string(index=False)
    )

residential["census_division_code"] = (
    residential["csd_uid_clean"]
    .astype(str)
    .str.slice(0, 4)
)

residential["derived_cd_code_valid_shape"] = (
    residential["census_division_code"].astype(str).str.match(r"^24\d{2}$")
)

if (~residential["derived_cd_code_valid_shape"]).any():
    invalid = residential[~residential["derived_cd_code_valid_shape"]]
    raise ValueError(
        "Some derived census division codes have invalid shape:\n"
        + invalid[
            [
                "odhf_index",
                "facility_name",
                "city",
                "CSDname",
                "CSDuid",
                "csd_uid_clean",
                "census_division_code",
            ]
        ].to_string(index=False)
    )


# -----------------------------
# Merge with census division inventory
# -----------------------------

assigned = residential.merge(
    cd_lookup,
    on="census_division_code",
    how="left",
    validate="many_to_one",
)

assigned["matched_to_cd_inventory"] = assigned["census_division_name"].notna()

if (~assigned["matched_to_cd_inventory"]).any():
    unmatched = assigned[~assigned["matched_to_cd_inventory"]]
    raise ValueError(
        "Some derived census division codes were not found in the CD inventory:\n"
        + unmatched[
            [
                "odhf_index",
                "facility_name",
                "city",
                "CSDname",
                "CSDuid",
                "csd_uid_clean",
                "census_division_code",
            ]
        ].to_string(index=False)
    )


# -----------------------------
# Build assigned facility-level audit table
# -----------------------------

assigned_facilities = pd.DataFrame(
    {
        "odhf_index": assigned["odhf_index"],
        "facility_name": assigned["facility_name"],
        "source_facility_type": assigned["source_facility_type"],
        "odhf_facility_type": assigned["odhf_facility_type"],
        "provider": assigned["provider"],
        "city": assigned["city"],
        "province": assigned["province"],
        "csd_name": assigned["CSDname"],
        "csd_uid": assigned["CSDuid"],
        "csd_uid_clean": assigned["csd_uid_clean"],
        "latitude": assigned["latitude"],
        "longitude": assigned["longitude"],
        "census_division_code": assigned["census_division_code"],
        "census_division_dguid": assigned["census_division_dguid"],
        "census_division_name": assigned["census_division_name"],
        "census_division_type": assigned["census_division_type"],
        "census_division_land_area_km2": assigned[
            "census_division_land_area_km2"
        ],
        "province_code": assigned["province_code"],
        "assignment_source": "automatic_csd_uid_prefix",
        "assignment_method": "automatic_csd_uid_to_cd_first_4_digits",
        "assignment_note": (
            "Census division code derived from first 4 digits of ODHF CSDuid; "
            "derived code validated against 2021 Quebec census division inventory."
        ),
        "source_residential_care": SOURCE_NAME,
        "source_catalogue": SOURCE_CATALOGUE,
        "source_odhf_version": SOURCE_ODHF_VERSION,
        "odhf_source_year": ODHF_SOURCE_YEAR,
        "geography_year": GEOGRAPHY_YEAR,
    }
)

assigned_facilities["census_division_land_area_km2"] = pd.to_numeric(
    assigned_facilities["census_division_land_area_km2"],
    errors="coerce",
)


# -----------------------------
# Validate assigned facility-level table
# -----------------------------

if assigned_facilities["odhf_index"].duplicated().any():
    duplicated = assigned_facilities[
        assigned_facilities["odhf_index"].duplicated(keep=False)
    ]
    raise ValueError(
        "Duplicated ODHF indexes in assigned residential-care table:\n"
        + duplicated.sort_values("odhf_index").to_string(index=False)
    )

required_assigned_non_missing = [
    "odhf_index",
    "facility_name",
    "odhf_facility_type",
    "csd_uid_clean",
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "assignment_source",
    "assignment_method",
]

for col in required_assigned_non_missing:
    missing_mask = (
        assigned_facilities[col].isna()
        | (assigned_facilities[col].astype(str).str.strip() == "")
        | (assigned_facilities[col].astype(str).str.lower() == "nan")
    )
    if missing_mask.any():
        raise ValueError(
            f"Missing values found in assigned facility column {col}:\n"
            + assigned_facilities.loc[
                missing_mask,
                [
                    "odhf_index",
                    "facility_name",
                    "city",
                    "csd_name",
                    col,
                ],
            ].to_string(index=False)
        )

if len(assigned_facilities) != len(residential_raw):
    raise ValueError(
        f"Expected {len(residential_raw)} assigned residential-care records, "
        f"but got {len(assigned_facilities)}."
    )


# -----------------------------
# Count residential-care facilities by census division
# -----------------------------

counts = (
    assigned_facilities
    .groupby("census_division_code", dropna=False)
    .agg(
        residential_care_facility_count_odhf=("odhf_index", "count"),
        residential_care_facility_count_automatic_csd_uid=(
            "assignment_source",
            lambda x: (x == "automatic_csd_uid_prefix").sum(),
        ),
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
    "residential_care_facility_count_odhf",
    "residential_care_facility_count_automatic_csd_uid",
]:
    clean_counts[col] = clean_counts[col].fillna(0).astype(int)

clean_counts["residential_care_facility_count_manual_repair"] = 0

clean_counts["unit_type"] = "census_division"
clean_counts["geography_year"] = GEOGRAPHY_YEAR
clean_counts["odhf_source_year"] = ODHF_SOURCE_YEAR
clean_counts["source_residential_care"] = SOURCE_NAME
clean_counts["source_catalogue"] = SOURCE_CATALOGUE
clean_counts["source_odhf_version"] = SOURCE_ODHF_VERSION

clean_counts["feature_description"] = (
    "Count of ODHF records classified as Nursing and residential care facilities "
    "assigned to 2021 Quebec census divisions. Records are assigned automatically "
    "from ODHF CSDuid to CDUID using the first 4 digits. This is a facility-record "
    "count proxy for residential/institutional care infrastructure, not a count of "
    "CHSLD residents or long-term-care beds."
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
        "residential_care_facility_count_odhf",
        "residential_care_facility_count_automatic_csd_uid",
        "residential_care_facility_count_manual_repair",
        "source_residential_care",
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

if clean_counts["residential_care_facility_count_odhf"].sum() != len(
    assigned_facilities
):
    raise ValueError(
        "Residential-care count sum does not equal number of assigned records. "
        f"Sum={clean_counts['residential_care_facility_count_odhf'].sum()}, "
        f"assigned={len(assigned_facilities)}"
    )

if (
    clean_counts["residential_care_facility_count_automatic_csd_uid"].sum()
    + clean_counts["residential_care_facility_count_manual_repair"].sum()
    != clean_counts["residential_care_facility_count_odhf"].sum()
):
    raise ValueError(
        "Automatic + manual residential-care counts do not equal total counts."
    )


# -----------------------------
# Save outputs
# -----------------------------

assigned_facilities.to_csv(OUTPUT_ASSIGNED_FACILITIES_CSV, index=False)
assigned_facilities.to_parquet(OUTPUT_ASSIGNED_FACILITIES_PARQUET, index=False)

clean_counts.to_csv(OUTPUT_COUNTS_CSV, index=False)
clean_counts.to_parquet(OUTPUT_COUNTS_PARQUET, index=False)


# -----------------------------
# Diagnostics
# -----------------------------

print("\nFinal assigned residential-care facility-level table")
print("Rows:", len(assigned_facilities))
print(
    "Automatic assignments:",
    int(
        (
            assigned_facilities["assignment_source"]
            == "automatic_csd_uid_prefix"
        ).sum()
    ),
)
print("Manual repairs:", 0)

print("\nFinal census-division residential-care count table")
print("Rows:", len(clean_counts))
print(
    "Total residential_care_facility_count_odhf:",
    int(clean_counts["residential_care_facility_count_odhf"].sum()),
)
print(
    "Census divisions with at least one residential-care facility:",
    int((clean_counts["residential_care_facility_count_odhf"] > 0).sum()),
)
print(
    "Census divisions with zero residential-care facilities:",
    int((clean_counts["residential_care_facility_count_odhf"] == 0).sum()),
)

print("\nTop census divisions by ODHF residential-care facility count:")
print(
    clean_counts[
        [
            "census_division_code",
            "census_division_name",
            "residential_care_facility_count_odhf",
            "residential_care_facility_count_automatic_csd_uid",
            "residential_care_facility_count_manual_repair",
        ]
    ]
    .sort_values("residential_care_facility_count_odhf", ascending=False)
    .head(20)
    .to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_ASSIGNED_FACILITIES_CSV)
print(OUTPUT_ASSIGNED_FACILITIES_PARQUET)
print(OUTPUT_COUNTS_CSV)
print(OUTPUT_COUNTS_PARQUET)

print("\nDone.")