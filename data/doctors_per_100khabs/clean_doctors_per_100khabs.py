from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# Clean Doctors per 100k Inhabitants — Quebec Health Regions
# ============================================================
#
# Purpose:
#   Clean the CIHI physician availability file into a health-region-level
#   feature table for Quebec.
#
# Important:
#   This script does NOT yet associate health regions with census divisions.
#   It only creates the clean health-region source table.
#
# Source:
#   CIHI — Physicians per 100,000 Population, by Specialty
#
# Current raw file expected:
#   doctors_per_100khabs/raw/cihi_family_medicine_physicians_per_100k_quebec_2024.csv
#
# Output:
#   doctors_per_100khabs/output/clean_health_region_doctors_per_100k_2024.csv
#   doctors_per_100khabs/output/clean_health_region_doctors_per_100k_2024.parquet
#
# Run from data/:
#   python doctors_per_100khabs/clean_doctors_per_100khabs.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

THIS_DIR = Path(__file__).resolve().parent

RAW_PATH = (
    THIS_DIR
    / "raw"
    / "cihi_family_medicine_physicians_per_100k_quebec_2024.csv"
)

OUTPUT_DIR = THIS_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "clean_health_region_doctors_per_100k_2024.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_health_region_doctors_per_100k_2024.parquet"


# -----------------------------
# Constants
# -----------------------------

SOURCE_NAME = "CIHI — Physicians per 100,000 Population, by Specialty"

EXPECTED_INDICATOR = "Family Medicine Physicians per 100,000 Population"
EXPECTED_PROVINCE = "Quebec"
EXPECTED_YEAR = 2024
EXPECTED_REPORTING_LEVEL = "Health region"


# -----------------------------
# Load raw file
# -----------------------------

if not RAW_PATH.exists():
    raise FileNotFoundError(
        f"Raw file not found:\n{RAW_PATH}\n\n"
        "Expected location:\n"
        "doctors_per_100khabs/raw/"
        "cihi_family_medicine_physicians_per_100k_quebec_2024.csv"
    )

raw = pd.read_csv(RAW_PATH)

print("\nLoaded raw CIHI file")
print("Rows:", len(raw))
print("Columns:", list(raw.columns))


# -----------------------------
# Basic column validation
# -----------------------------

required_columns = [
    "Province/territory",
    "Reporting level",
    "Region",
    "Place or organization",
    "Time scale",
    "Time frame",
    "Indicator",
    "Crude rate",
    "Unit of measure",
    "Refresh date",
]

missing_columns = [col for col in required_columns if col not in raw.columns]

if missing_columns:
    raise ValueError(
        "Missing required columns in raw file:\n"
        + "\n".join(missing_columns)
    )


# -----------------------------
# Normalize text columns
# -----------------------------

clean_raw = raw.copy()

for col in required_columns:
    if col in clean_raw.columns:
        clean_raw[col] = clean_raw[col].astype("string").str.strip()


# -----------------------------
# Filter to Quebec health-region rows
# -----------------------------

health_region_rows = clean_raw[
    (clean_raw["Province/territory"] == EXPECTED_PROVINCE)
    & (clean_raw["Reporting level"] == EXPECTED_REPORTING_LEVEL)
    & (clean_raw["Indicator"] == EXPECTED_INDICATOR)
    & (pd.to_numeric(clean_raw["Time frame"], errors="coerce") == EXPECTED_YEAR)
].copy()

print("\nQuebec health-region rows found:", len(health_region_rows))

if health_region_rows.empty:
    raise ValueError(
        "No Quebec health-region rows found. Check the raw file values for "
        "Province/territory, Reporting level, Indicator, and Time frame."
    )


# -----------------------------
# Clean numeric fields
# -----------------------------

health_region_rows["physicians_per_100k_health_region"] = pd.to_numeric(
    health_region_rows["Crude rate"],
    errors="coerce",
)

health_region_rows["year"] = pd.to_numeric(
    health_region_rows["Time frame"],
    errors="coerce",
).astype("Int64")


# -----------------------------
# Build clean table
# -----------------------------

clean = pd.DataFrame(
    {
        "province": health_region_rows["Province/territory"],
        "health_region_name": health_region_rows["Region"],
        "place_or_organization": health_region_rows["Place or organization"],
        "unit_type": "health_region",
        "year": health_region_rows["year"],
        "indicator": health_region_rows["Indicator"],
        "physicians_per_100k_health_region": health_region_rows[
            "physicians_per_100k_health_region"
        ],
        "unit_of_measure": health_region_rows["Unit of measure"],
        "time_scale": health_region_rows["Time scale"],
        "refresh_date": health_region_rows["Refresh date"],
        "source_doctors_per_100k": SOURCE_NAME,
    }
).copy()


# -----------------------------
# Add methodological description
# -----------------------------

clean["feature_description"] = (
    "Family medicine physicians per 100,000 population at the CIHI health-region level. "
    "This is a health-region-native measure and has not yet been assigned to census divisions."
)


# -----------------------------
# Sort rows
# -----------------------------

clean = clean.sort_values("health_region_name").reset_index(drop=True)


# -----------------------------
# Validation
# -----------------------------

if clean["health_region_name"].duplicated().any():
    duplicated = clean[clean["health_region_name"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated health_region_name values found:\n"
        + duplicated.to_string(index=False)
    )

if clean["physicians_per_100k_health_region"].isna().any():
    missing_rate = clean[clean["physicians_per_100k_health_region"].isna()]
    raise ValueError(
        "Missing physicians_per_100k_health_region values found:\n"
        + missing_rate.to_string(index=False)
    )

if (clean["physicians_per_100k_health_region"] <= 0).any():
    invalid_rate = clean[clean["physicians_per_100k_health_region"] <= 0]
    raise ValueError(
        "Non-positive physicians_per_100k_health_region values found:\n"
        + invalid_rate.to_string(index=False)
    )


# -----------------------------
# Print diagnostics
# -----------------------------

print("\nClean health-region doctors-per-100k table")
print("Rows:", len(clean))
print("Columns:", list(clean.columns))

print("\nHealth regions:")
print(clean[["health_region_name", "physicians_per_100k_health_region"]].to_string(index=False))

print("\nSummary:")
print(clean["physicians_per_100k_health_region"].describe())

print("\nMissing values by column:")
print(clean.isna().sum())


# -----------------------------
# Save outputs
# -----------------------------

clean.to_csv(OUTPUT_CSV, index=False)
clean.to_parquet(OUTPUT_PARQUET, index=False)

print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_PARQUET)

print("\nDone.")