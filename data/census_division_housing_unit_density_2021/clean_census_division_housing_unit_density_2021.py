from pathlib import Path
import pandas as pd


# ============================================================
# Clean Census Division Housing Unit Density 2021
# ============================================================
#
# Purpose:
#   Create a census-division-level housing-unit density feature for the
#   SoVI-like variable:
#
#       housing_unit_density
#
# Original SoVI variable:
#
#       HODENUT90 = housing units per square mile
#
# Canadian / Québec adaptation:
#   Uses the already-cleaned census-division spatial/population frame:
#
#       total_private_dwellings_2021
#       land_area_km2
#
#   and derives:
#
#       housing_unit_density_per_km2
#       housing_unit_density_per_sq_mile
#
# The SoVI mapping should use:
#
#       housing_unit_density_per_sq_mile
#
# because the original SoVI variable is defined per square mile.
#
# Run from data/:
#   python census_division_housing_unit_density_2021/clean_census_division_housing_unit_density_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_housing_unit_density_2021"
OUTPUT_DIR = SECTION_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_CD_FRAME = (
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.csv"
)

OUTPUT_CLEAN = OUTPUT_DIR / "clean_census_division_housing_unit_density_2021.csv"
OUTPUT_METADATA = OUTPUT_DIR / "clean_census_division_housing_unit_density_variable_metadata_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "clean_census_division_housing_unit_density_summary_2021.csv"


# -----------------------------
# Config
# -----------------------------

EXPECTED_QC_CD_COUNT = 98

# 1 square kilometre = 0.3861021585424458 square miles.
SQ_KM_TO_SQ_MI = 0.3861021585424458

IDENTITY_COLUMNS = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "census_division_type",
    "province_code",
    "province_name",
    "geography_level",
    "census_year",
    "population_total_2021",
    "total_private_dwellings_2021",
    "land_area_km2",
    "land_area_km2_boundary",
    "land_area_km2_population_table_2021",
    "population_density_per_km2_2021",
    "has_positive_population",
]


# -----------------------------
# Helpers
# -----------------------------

def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        return str(path)


def clean_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def contains_mojibake(series: pd.Series) -> int:
    text = series.astype("string")
    return int(text.str.contains("Ã|Â|�", regex=True, na=False).sum())


def numeric_summary(series: pd.Series) -> dict:
    numeric = clean_numeric(series)

    return {
        "non_missing": int(numeric.notna().sum()),
        "missing": int(numeric.isna().sum()),
        "min": numeric.min(skipna=True),
        "max": numeric.max(skipna=True),
        "mean": numeric.mean(skipna=True),
        "median": numeric.median(skipna=True),
    }


# -----------------------------
# Validate input
# -----------------------------

if not BASE_CD_FRAME.exists():
    raise FileNotFoundError(f"Missing base census-division frame:\n{BASE_CD_FRAME}")


# -----------------------------
# Load base frame
# -----------------------------

base = pd.read_csv(BASE_CD_FRAME, dtype=str, low_memory=False)
base.columns = [str(col).strip() for col in base.columns]

print("\nCleaning Census Division Housing Unit Density 2021")
print("Base CD frame:", safe_relative(BASE_CD_FRAME))
print("Base rows:", len(base))


required_columns = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "total_private_dwellings_2021",
    "land_area_km2",
]

missing_columns = [col for col in required_columns if col not in base.columns]

if missing_columns:
    raise ValueError(
        "Base frame is missing required columns:\n"
        + "\n".join(missing_columns)
        + "\n\nAvailable columns:\n"
        + "\n".join(base.columns)
    )

base["census_division_dguid"] = clean_key(base["census_division_dguid"])

if len(base) != EXPECTED_QC_CD_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_QC_CD_COUNT} Québec census divisions, got {len(base)}."
    )

if base["census_division_dguid"].duplicated().any():
    dupes = base[base["census_division_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicate census_division_dguid values in base frame:\n"
        + dupes[
            [
                "census_division_code",
                "census_division_dguid",
                "census_division_name",
            ]
        ].to_string(index=False)
    )


# -----------------------------
# Clean numeric inputs
# -----------------------------

base["total_private_dwellings_2021"] = clean_numeric(base["total_private_dwellings_2021"])
base["land_area_km2"] = clean_numeric(base["land_area_km2"])

if "population_total_2021" in base.columns:
    base["population_total_2021"] = clean_numeric(base["population_total_2021"])

if "population_density_per_km2_2021" in base.columns:
    base["population_density_per_km2_2021"] = clean_numeric(base["population_density_per_km2_2021"])

if "land_area_km2_boundary" in base.columns:
    base["land_area_km2_boundary"] = clean_numeric(base["land_area_km2_boundary"])

if "land_area_km2_population_table_2021" in base.columns:
    base["land_area_km2_population_table_2021"] = clean_numeric(base["land_area_km2_population_table_2021"])


missing_dwellings = int(base["total_private_dwellings_2021"].isna().sum())
missing_land_area = int(base["land_area_km2"].isna().sum())

if missing_dwellings != 0:
    raise ValueError(f"Missing total_private_dwellings_2021 values: {missing_dwellings}")

if missing_land_area != 0:
    raise ValueError(f"Missing land_area_km2 values: {missing_land_area}")

non_positive_area = int((base["land_area_km2"] <= 0).sum())

if non_positive_area != 0:
    bad = base[base["land_area_km2"] <= 0][
        [
            "census_division_code",
            "census_division_dguid",
            "census_division_name",
            "land_area_km2",
        ]
    ]
    raise ValueError(
        f"Found {non_positive_area} census divisions with non-positive land_area_km2:\n"
        + bad.to_string(index=False)
    )


# -----------------------------
# Compute housing-unit density
# -----------------------------

clean = base[[col for col in IDENTITY_COLUMNS if col in base.columns]].copy()

clean["housing_unit_density_per_km2"] = (
    clean["total_private_dwellings_2021"] / clean["land_area_km2"]
)

clean["land_area_sq_mile"] = clean["land_area_km2"] * SQ_KM_TO_SQ_MI

clean["housing_unit_density_per_sq_mile"] = (
    clean["total_private_dwellings_2021"] / clean["land_area_sq_mile"]
)

# Main SoVI alias. Original SoVI HODENUT90 is housing units per square mile.
clean["housing_unit_density"] = clean["housing_unit_density_per_sq_mile"]

clean["source_file"] = safe_relative(BASE_CD_FRAME)
clean["source_numerator_column"] = "total_private_dwellings_2021"
clean["source_denominator_column"] = "land_area_km2"
clean["allocation_or_derivation_method"] = "total_private_dwellings_2021 / land_area"
clean["method_note"] = (
    "Derived from the cleaned Québec census-division spatial/population frame. "
    "housing_unit_density_per_km2 uses total_private_dwellings_2021 divided by land_area_km2. "
    "housing_unit_density_per_sq_mile converts land area to square miles before division. "
    "The SoVI alias housing_unit_density uses the square-mile version for consistency with HODENUT90."
)


# -----------------------------
# Validation
# -----------------------------

if len(clean) != EXPECTED_QC_CD_COUNT:
    raise ValueError(
        f"Clean output has {len(clean)} rows; expected {EXPECTED_QC_CD_COUNT}."
    )

if clean["census_division_dguid"].duplicated().any():
    dupes = clean[clean["census_division_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicate census_division_dguid values in clean output:\n"
        + dupes[
            [
                "census_division_code",
                "census_division_dguid",
                "census_division_name",
            ]
        ].to_string(index=False)
    )

required_output_features = [
    "housing_unit_density",
    "housing_unit_density_per_km2",
    "housing_unit_density_per_sq_mile",
    "land_area_sq_mile",
]

for col in required_output_features:
    missing = int(clean[col].isna().sum())
    if missing != 0:
        raise ValueError(f"Unexpected missing values in {col}: {missing}")

    negative = int((clean[col] < 0).sum())
    if negative != 0:
        raise ValueError(f"Unexpected negative values in {col}: {negative}")

formula_diff = (
    clean["housing_unit_density"]
    - clean["housing_unit_density_per_sq_mile"]
).abs().max(skipna=True)

if formula_diff != 0:
    raise ValueError(
        f"housing_unit_density alias formula check failed. Max difference: {formula_diff}"
    )

base_names_with_mojibake = contains_mojibake(base["census_division_name"])
clean_names_with_mojibake = contains_mojibake(clean["census_division_name"])


# -----------------------------
# Metadata
# -----------------------------

metadata = pd.DataFrame(
    [
        {
            "variable": "housing_unit_density",
            "original_sovi_code": "HODENUT90",
            "description": "Housing units per square mile",
            "source_numerator": "total_private_dwellings_2021",
            "source_denominator": "land_area_km2 converted to square miles",
            "unit": "housing_units_per_square_mile",
            "derivation": "total_private_dwellings_2021 / (land_area_km2 * 0.3861021585424458)",
            "role": "recommended_sovi_housing_unit_density_variable",
            "notes": (
                "Main SoVI alias. Uses square-mile denominator because the original SoVI variable "
                "is housing units per square mile."
            ),
        },
        {
            "variable": "housing_unit_density_per_sq_mile",
            "original_sovi_code": "HODENUT90",
            "description": "Housing units per square mile",
            "source_numerator": "total_private_dwellings_2021",
            "source_denominator": "land_area_km2 converted to square miles",
            "unit": "housing_units_per_square_mile",
            "derivation": "total_private_dwellings_2021 / (land_area_km2 * 0.3861021585424458)",
            "role": "explicit_square_mile_density_audit_variable",
            "notes": "Same value as housing_unit_density.",
        },
        {
            "variable": "housing_unit_density_per_km2",
            "original_sovi_code": "",
            "description": "Housing units per square kilometre",
            "source_numerator": "total_private_dwellings_2021",
            "source_denominator": "land_area_km2",
            "unit": "housing_units_per_square_kilometre",
            "derivation": "total_private_dwellings_2021 / land_area_km2",
            "role": "metric_density_audit_variable",
            "notes": "Retained for Canadian metric-unit interpretation and audit.",
        },
        {
            "variable": "land_area_sq_mile",
            "original_sovi_code": "",
            "description": "Census-division land area converted from square kilometres to square miles",
            "source_numerator": "land_area_km2",
            "source_denominator": "",
            "unit": "square_miles",
            "derivation": "land_area_km2 * 0.3861021585424458",
            "role": "denominator_audit_variable",
            "notes": "Used to compute housing_unit_density_per_sq_mile.",
        },
    ]
)

metadata.to_csv(OUTPUT_METADATA, index=False, encoding="utf-8")


# -----------------------------
# Summary
# -----------------------------

summary_rows = [
    {"metric": "base_cd_frame", "value": safe_relative(BASE_CD_FRAME)},
    {"metric": "clean_rows", "value": len(clean)},
    {"metric": "unique_census_divisions", "value": clean["census_division_dguid"].nunique()},
    {"metric": "variables_cleaned", "value": "housing_unit_density, housing_unit_density_per_km2, housing_unit_density_per_sq_mile"},
    {"metric": "source_numerator_column", "value": "total_private_dwellings_2021"},
    {"metric": "source_denominator_column", "value": "land_area_km2"},
    {"metric": "square_km_to_square_mile_factor", "value": SQ_KM_TO_SQ_MI},
    {"metric": "missing_total_private_dwellings_2021", "value": missing_dwellings},
    {"metric": "missing_land_area_km2", "value": missing_land_area},
    {"metric": "non_positive_land_area_km2", "value": non_positive_area},
    {"metric": "all_main_variables_complete", "value": bool(clean["housing_unit_density"].notna().all())},
    {"metric": "housing_unit_density_alias_max_abs_difference", "value": formula_diff},
    {"metric": "base_names_with_mojibake", "value": base_names_with_mojibake},
    {"metric": "clean_names_with_mojibake", "value": clean_names_with_mojibake},
]

for variable in [
    "total_private_dwellings_2021",
    "land_area_km2",
    "land_area_sq_mile",
    "housing_unit_density",
    "housing_unit_density_per_km2",
    "housing_unit_density_per_sq_mile",
]:
    if variable in clean.columns:
        stats = numeric_summary(clean[variable])

        for key, value in stats.items():
            summary_rows.append(
                {
                    "metric": f"{variable}_{key}",
                    "value": value,
                }
            )

summary_rows.append(
    {
        "metric": "recommended_next_step",
        "value": (
            "If this summary shows 98 rows, complete housing_unit_density values, "
            "and no mojibake, create a SoVI YAML mapping from housing_unit_density "
            "to census_division_housing_unit_density_2021/output/"
            "clean_census_division_housing_unit_density_2021.csv."
        ),
    }
)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Save clean output
# -----------------------------

clean.to_csv(OUTPUT_CLEAN, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CLEAN CENSUS DIVISION HOUSING UNIT DENSITY 2021")
print("=" * 72)

print("\nClean output:")
print("Rows:", len(clean))
print("Unique census divisions:", clean["census_division_dguid"].nunique())
print("All housing_unit_density values complete:", bool(clean["housing_unit_density"].notna().all()))
print("housing_unit_density alias max abs difference:", formula_diff)

print("\nMojibake check:")
print("Base names with mojibake:", base_names_with_mojibake)
print("Clean names with mojibake:", clean_names_with_mojibake)

print("\nMain variable summaries:")
for variable in [
    "housing_unit_density",
    "housing_unit_density_per_km2",
    "housing_unit_density_per_sq_mile",
]:
    print("\n", variable)
    print(clean[variable].describe())

print("\nPreview:")
preview_cols = [
    "census_division_code",
    "census_division_name",
    "total_private_dwellings_2021",
    "land_area_km2",
    "housing_unit_density_per_km2",
    "housing_unit_density_per_sq_mile",
    "housing_unit_density",
]
print(clean[preview_cols].head(20).to_string(index=False))

print("\nSaved:")
print(OUTPUT_CLEAN)
print(OUTPUT_METADATA)
print(OUTPUT_SUMMARY)

print("\nDone.")