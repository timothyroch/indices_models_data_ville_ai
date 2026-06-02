from pathlib import Path
import pandas as pd


# ============================================================
# Create Manual Repair Template for ODHF Hospitals Missing CSDuid
# ============================================================
#
# Purpose:
#   Create an auditable manual repair template for Quebec ODHF hospital
#   records that are missing CSDuid.
#
# Why this exists:
#   Most Quebec hospital records can be assigned to census divisions by taking
#   the first 4 digits of CSDuid. However, 22 Quebec hospital records are
#   missing CSDuid. This script creates a small lookup file so those records
#   can be repaired manually before final aggregation.
#
# Input:
#   hospitals_per_capita/output/odhf_quebec_hospitals_missing_csd_for_manual_review.csv
#
# Output:
#   hospitals_per_capita/lookup/odhf_quebec_hospitals_missing_csd_manual_repair.csv
#
# Run from data/:
#   python hospitals_per_capita/create_missing_csd_manual_repair_template.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

INPUT_MISSING_CSD = (
    DATA_DIR
    / "hospitals_per_capita"
    / "output"
    / "odhf_quebec_hospitals_missing_csd_for_manual_review.csv"
)

CD_INVENTORY_PATH = (
    DATA_DIR
    / "doctors_per_100khabs"
    / "output"
    / "quebec_census_divisions_2021_inventory.csv"
)

LOOKUP_DIR = DATA_DIR / "hospitals_per_capita" / "lookup"
LOOKUP_DIR.mkdir(exist_ok=True)

OUTPUT_REPAIR_TEMPLATE = (
    LOOKUP_DIR / "odhf_quebec_hospitals_missing_csd_manual_repair.csv"
)

OUTPUT_CD_REFERENCE = (
    LOOKUP_DIR / "quebec_census_division_reference_for_manual_repair.csv"
)


# -----------------------------
# Load inputs
# -----------------------------

if not INPUT_MISSING_CSD.exists():
    raise FileNotFoundError(
        f"Missing-CSD hospital file not found:\n{INPUT_MISSING_CSD}\n\n"
        "Run first:\n"
        "python hospitals_per_capita/inspect_odhf_csd_to_cd_bridge.py"
    )

if not CD_INVENTORY_PATH.exists():
    raise FileNotFoundError(
        f"Census division inventory not found:\n{CD_INVENTORY_PATH}\n\n"
        "Run first:\n"
        "python doctors_per_100khabs/inspect_census_division_boundary.py"
    )

missing = pd.read_csv(INPUT_MISSING_CSD, dtype=str)
cd_inventory = pd.read_csv(CD_INVENTORY_PATH, dtype=str)

print("\nLoaded hospitals missing CSDuid")
print("Rows:", len(missing))
print("Columns:", list(missing.columns))

print("\nLoaded Quebec census division inventory")
print("Rows:", len(cd_inventory))
print("Columns:", list(cd_inventory.columns))


# -----------------------------
# Validate expected columns
# -----------------------------

required_missing_cols = [
    "index",
    "facility_name",
    "source_facility_type",
    "odhf_facility_type",
    "provider",
    "city",
    "province",
    "CSDname",
    "CSDuid",
]

missing_required_cols = [
    col for col in required_missing_cols if col not in missing.columns
]

if missing_required_cols:
    raise ValueError(
        "Missing expected columns in missing-CSD hospital file:\n"
        + "\n".join(missing_required_cols)
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
# Build manual repair template
# -----------------------------

repair = missing.copy()

# Keep only useful audit columns plus empty manual repair columns.
base_cols = [
    "index",
    "facility_name",
    "source_facility_type",
    "odhf_facility_type",
    "provider",
    "city",
    "province",
    "CSDname",
    "CSDuid",
]

optional_cols = [
    "latitude",
    "longitude",
    "source_format_str_address",
]

selected_cols = base_cols + [col for col in optional_cols if col in repair.columns]

repair = repair[selected_cols].copy()

repair["manual_cd_code"] = ""
repair["manual_census_division_name"] = ""
repair["manual_census_division_dguid"] = ""
repair["manual_repair_method"] = ""
repair["manual_repair_note"] = ""

repair = repair[
    selected_cols
    + [
        "manual_cd_code",
        "manual_census_division_name",
        "manual_census_division_dguid",
        "manual_repair_method",
        "manual_repair_note",
    ]
].sort_values(["city", "facility_name"], na_position="last")


# -----------------------------
# Build CD reference table
# -----------------------------

cd_reference = cd_inventory[
    [
        "CDUID",
        "DGUID",
        "CDNAME",
        "CDTYPE",
        "LANDAREA",
        "PRUID",
    ]
].copy()

cd_reference = cd_reference.rename(
    columns={
        "CDUID": "manual_cd_code",
        "DGUID": "manual_census_division_dguid",
        "CDNAME": "manual_census_division_name",
        "CDTYPE": "census_division_type",
        "LANDAREA": "census_division_land_area_km2",
        "PRUID": "province_code",
    }
).sort_values("manual_cd_code")


# -----------------------------
# Save outputs
# -----------------------------

repair.to_csv(OUTPUT_REPAIR_TEMPLATE, index=False)
cd_reference.to_csv(OUTPUT_CD_REFERENCE, index=False)


# -----------------------------
# Print diagnostics
# -----------------------------

print("\nCreated manual repair template:")
print(OUTPUT_REPAIR_TEMPLATE)
print("Rows:", len(repair))

print("\nCreated census division reference:")
print(OUTPUT_CD_REFERENCE)
print("Rows:", len(cd_reference))

print("\nManual repair columns to fill:")
print("- manual_cd_code")
print("- manual_census_division_name")
print("- manual_census_division_dguid")
print("- manual_repair_method")
print("- manual_repair_note")

print("\nRecommended manual_repair_method values:")
print("- manual_city_to_cd_exact")
print("- manual_csdname_to_cd_exact")
print("- manual_facility_name_to_cd_inferred")
print("- unresolved")

print("\nPreview of repair template:")
print(
    repair[
        [
            "index",
            "facility_name",
            "city",
            "CSDname",
            "manual_cd_code",
            "manual_census_division_name",
            "manual_repair_method",
        ]
    ].to_string(index=False)
)

print("\nDone.")