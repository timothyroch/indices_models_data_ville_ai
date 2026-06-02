from pathlib import Path
import pandas as pd


# ============================================================
# Inspect ODHF CSD → Census Division Bridge
# ============================================================
#
# Purpose:
#   Test whether ODHF Quebec hospital records can be assigned to census
#   divisions by deriving CDUID from CSDuid.
#
# Logic:
#   For Quebec 2021 geographic identifiers:
#       CSDuid example: 2423027
#       CDUID example: 2423
#
#   Therefore:
#       derived_cd_code = first 4 characters of CSDuid_clean
#
# Inputs:
#   hospitals_per_capita/output/odhf_quebec_hospitals_preview.csv
#   doctors_per_100khabs/output/quebec_census_divisions_2021_inventory.csv
#
# Outputs:
#   hospitals_per_capita/output/odhf_quebec_hospitals_with_derived_cd.csv
#   hospitals_per_capita/output/odhf_quebec_hospital_counts_by_derived_cd.csv
#   hospitals_per_capita/output/odhf_quebec_hospitals_missing_csd_for_manual_review.csv
#   hospitals_per_capita/output/odhf_derived_cd_validation_unmatched.csv
#
# Run from data/:
#   python hospitals_per_capita/inspect_odhf_csd_to_cd_bridge.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

INPUT_HOSPITALS = (
    DATA_DIR
    / "hospitals_per_capita"
    / "output"
    / "odhf_quebec_hospitals_preview.csv"
)

INPUT_CD_INVENTORY = (
    DATA_DIR
    / "doctors_per_100khabs"
    / "output"
    / "quebec_census_divisions_2021_inventory.csv"
)

OUTPUT_DIR = DATA_DIR / "hospitals_per_capita" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_HOSPITALS_WITH_CD = OUTPUT_DIR / "odhf_quebec_hospitals_with_derived_cd.csv"
OUTPUT_COUNTS_BY_CD = OUTPUT_DIR / "odhf_quebec_hospital_counts_by_derived_cd.csv"
OUTPUT_MISSING_CSD = OUTPUT_DIR / "odhf_quebec_hospitals_missing_csd_for_manual_review.csv"
OUTPUT_UNMATCHED_CD = OUTPUT_DIR / "odhf_derived_cd_validation_unmatched.csv"


# -----------------------------
# Load inputs
# -----------------------------

if not INPUT_HOSPITALS.exists():
    raise FileNotFoundError(f"Hospital preview file not found:\n{INPUT_HOSPITALS}")

if not INPUT_CD_INVENTORY.exists():
    raise FileNotFoundError(f"Census division inventory not found:\n{INPUT_CD_INVENTORY}")

hospitals = pd.read_csv(INPUT_HOSPITALS, dtype=str)
cd_inventory = pd.read_csv(INPUT_CD_INVENTORY, dtype=str)

print("\nLoaded Quebec hospital records")
print("Rows:", len(hospitals))
print("Columns:", list(hospitals.columns))

print("\nLoaded Quebec census division inventory")
print("Rows:", len(cd_inventory))
print("Columns:", list(cd_inventory.columns))


# -----------------------------
# Validate expected columns
# -----------------------------

required_hospital_cols = [
    "index",
    "facility_name",
    "odhf_facility_type",
    "city",
    "province",
    "CSDname",
    "CSDuid",
]

missing_hospital_cols = [
    col for col in required_hospital_cols if col not in hospitals.columns
]

if missing_hospital_cols:
    raise ValueError(
        "Missing expected columns in hospital file:\n"
        + "\n".join(missing_hospital_cols)
    )

required_cd_cols = [
    "CDUID",
    "DGUID",
    "CDNAME",
    "CDTYPE",
    "LANDAREA",
    "PRUID",
]

missing_cd_cols = [col for col in required_cd_cols if col not in cd_inventory.columns]

if missing_cd_cols:
    raise ValueError(
        "Missing expected columns in census division inventory:\n"
        + "\n".join(missing_cd_cols)
    )


# -----------------------------
# Clean CSDuid and derive CDUID
# -----------------------------

hospitals = hospitals.copy()

hospitals["CSDuid_clean"] = (
    hospitals["CSDuid"]
    .astype("string")
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)

hospitals["has_csd_uid"] = (
    hospitals["CSDuid_clean"].notna()
    & (hospitals["CSDuid_clean"].astype(str).str.strip() != "")
    & (hospitals["CSDuid_clean"].astype(str).str.lower() != "nan")
)

hospitals["derived_cd_code"] = ""

hospitals.loc[hospitals["has_csd_uid"], "derived_cd_code"] = (
    hospitals.loc[hospitals["has_csd_uid"], "CSDuid_clean"]
    .astype(str)
    .str.slice(0, 4)
)

# For Quebec, derived CD codes should start with 24 and have 4 digits.
hospitals["derived_cd_code_valid_shape"] = (
    hospitals["derived_cd_code"].astype(str).str.match(r"^24\d{2}$")
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
        "CDUID": "derived_cd_code",
        "DGUID": "census_division_dguid",
        "CDNAME": "census_division_name",
        "CDTYPE": "census_division_type",
        "LANDAREA": "census_division_land_area_km2",
        "PRUID": "province_code",
    }
)

cd_lookup["derived_cd_code"] = cd_lookup["derived_cd_code"].astype(str).str.strip()


# -----------------------------
# Merge hospital records with CD inventory
# -----------------------------

merged = hospitals.merge(
    cd_lookup,
    on="derived_cd_code",
    how="left",
    validate="many_to_one",
)

merged["matched_to_cd_inventory"] = merged["census_division_name"].notna()


# -----------------------------
# Diagnostics
# -----------------------------

missing_csd = merged[~merged["has_csd_uid"]].copy()

invalid_shape = merged[
    merged["has_csd_uid"]
    & ~merged["derived_cd_code_valid_shape"]
].copy()

unmatched_cd = merged[
    merged["has_csd_uid"]
    & merged["derived_cd_code_valid_shape"]
    & ~merged["matched_to_cd_inventory"]
].copy()

print("\nBridge diagnostics")
print("Total Quebec hospital records:", len(merged))
print("Records with CSDuid:", int(merged["has_csd_uid"].sum()))
print("Records missing CSDuid:", len(missing_csd))
print("Records with invalid derived CD code shape:", len(invalid_shape))
print("Records with derived CD code not found in CD inventory:", len(unmatched_cd))
print("Records successfully matched to CD inventory:", int(merged["matched_to_cd_inventory"].sum()))


if not missing_csd.empty:
    print("\nHospitals missing CSDuid:")
    display_cols = [
        "index",
        "facility_name",
        "city",
        "CSDname",
        "CSDuid",
        "derived_cd_code",
    ]
    print(missing_csd[display_cols].to_string(index=False))

if not unmatched_cd.empty:
    print("\nDerived CD codes not found in census division inventory:")
    display_cols = [
        "index",
        "facility_name",
        "city",
        "CSDname",
        "CSDuid",
        "CSDuid_clean",
        "derived_cd_code",
    ]
    print(unmatched_cd[display_cols].to_string(index=False))


# -----------------------------
# Count hospitals by census division
# -----------------------------

matched = merged[merged["matched_to_cd_inventory"]].copy()

counts_by_cd = (
    matched
    .groupby(
        [
            "derived_cd_code",
            "census_division_dguid",
            "census_division_name",
            "census_division_type",
            "census_division_land_area_km2",
            "province_code",
        ],
        dropna=False,
    )
    .size()
    .reset_index(name="hospital_count_odhf")
    .sort_values(["hospital_count_odhf", "derived_cd_code"], ascending=[False, True])
)

print("\nHospital counts by derived census division:")
print(counts_by_cd.to_string(index=False))


# -----------------------------
# Save outputs
# -----------------------------

merged.to_csv(OUTPUT_HOSPITALS_WITH_CD, index=False)
counts_by_cd.to_csv(OUTPUT_COUNTS_BY_CD, index=False)

if not missing_csd.empty:
    missing_csd.to_csv(OUTPUT_MISSING_CSD, index=False)

if not unmatched_cd.empty:
    unmatched_cd.to_csv(OUTPUT_UNMATCHED_CD, index=False)


print("\nSaved outputs:")
print(OUTPUT_HOSPITALS_WITH_CD)
print(OUTPUT_COUNTS_BY_CD)

if not missing_csd.empty:
    print(OUTPUT_MISSING_CSD)

if not unmatched_cd.empty:
    print(OUTPUT_UNMATCHED_CD)

print("\nInterpretation:")
print("- If most records match successfully, we do not need a CSD boundary file yet.")
print("- Missing CSDuid records can be repaired manually later using city/CSDname.")
print("- The next full cleaner can aggregate hospital counts by CD and divide by CD population.")

print("\nDone.")