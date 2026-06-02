from pathlib import Path
import pandas as pd


# ============================================================
# Create Census Division → Health Region Crosswalk Template
# ============================================================
#
# Purpose:
#   Create an auditable template linking Quebec census divisions to CIHI
#   health regions for the doctors_per_100khabs proxy.
#
# Input:
#   doctors_per_100khabs/output/quebec_census_divisions_2021_inventory.csv
#
# Output:
#   doctors_per_100khabs/lookup/quebec_census_division_to_health_region_crosswalk.csv
#
# Run from data/:
#   python doctors_per_100khabs/create_health_region_crosswalk_template.py
#
# ============================================================


DATA_DIR = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    DATA_DIR
    / "doctors_per_100khabs"
    / "output"
    / "quebec_census_divisions_2021_inventory.csv"
)

LOOKUP_DIR = DATA_DIR / "doctors_per_100khabs" / "lookup"
LOOKUP_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = LOOKUP_DIR / "quebec_census_division_to_health_region_crosswalk.csv"


if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input census division inventory not found:\n{INPUT_CSV}")

cd = pd.read_csv(INPUT_CSV, dtype=str)

required_columns = ["CDUID", "DGUID", "CDNAME", "CDTYPE", "LANDAREA", "PRUID"]

missing_columns = [col for col in required_columns if col not in cd.columns]

if missing_columns:
    raise ValueError(
        "Missing required columns in census division inventory:\n"
        + "\n".join(missing_columns)
    )


template = cd[required_columns].copy()

template = template.rename(
    columns={
        "CDUID": "census_division_code",
        "DGUID": "census_division_dguid",
        "CDNAME": "census_division_name",
        "CDTYPE": "census_division_type",
        "LANDAREA": "census_division_land_area_km2",
        "PRUID": "province_code",
    }
)

template["health_region_name"] = ""
template["crosswalk_method"] = ""
template["crosswalk_note"] = ""

template = template[
    [
        "census_division_code",
        "census_division_dguid",
        "census_division_name",
        "census_division_type",
        "census_division_land_area_km2",
        "province_code",
        "health_region_name",
        "crosswalk_method",
        "crosswalk_note",
    ]
].sort_values("census_division_code")

template.to_csv(OUTPUT_CSV, index=False)

print("\nCreated crosswalk template:")
print(OUTPUT_CSV)

print("\nRows:", len(template))
print("\nPreview:")
print(template.head(20).to_string(index=False))

print("\nNext step:")
print("Fill health_region_name, crosswalk_method, and crosswalk_note.")
print("Use health_region_name values exactly as they appear in the cleaned CIHI table.")