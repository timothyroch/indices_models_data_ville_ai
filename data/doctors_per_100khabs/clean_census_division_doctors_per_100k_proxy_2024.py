from pathlib import Path
import pandas as pd


# ============================================================
# Clean Census Division Doctors per 100k Proxy — Quebec 2024
# ============================================================
#
# Purpose:
#   Assign CIHI health-region-level physicians-per-100k rates to Quebec
#   census divisions using the manually filled census-division-to-health-region
#   crosswalk.
#
# Important:
#   The source physician rate is health-region-native, not census-division-native.
#   This script creates a census-division proxy by assigning each census division
#   the rate of its corresponding health region.
#
# Expected inputs:
#
#   doctors_per_100khabs/output/clean_health_region_doctors_per_100k_2024.csv
#   doctors_per_100khabs/lookup/quebec_census_division_to_health_region_crosswalk_filled.csv
#   doctors_per_100khabs/lookup/quebec_census_division_to_health_region_crosswalk_unresolved.csv
#   census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.csv
#
# Expected unresolved case:
#
#   Nord-du-Québec may remain unresolved because the census division is too
#   coarse relative to CIHI's northern health-region split.
#
# Output:
#
#   doctors_per_100khabs/output/clean_census_division_doctors_per_100k_proxy_2024.csv
#   doctors_per_100khabs/output/clean_census_division_doctors_per_100k_proxy_2024.parquet
#   doctors_per_100khabs/output/clean_census_division_doctors_per_100k_proxy_summary_2024.csv
#   doctors_per_100khabs/output/clean_census_division_doctors_per_100k_proxy_unresolved_2024.csv
#
# Run from data/:
#
#   python doctors_per_100khabs/clean_census_division_doctors_per_100k_proxy_2024.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent

OUTPUT_DIR = THIS_DIR / "output"
LOOKUP_DIR = THIS_DIR / "lookup"
OUTPUT_DIR.mkdir(exist_ok=True)

HEALTH_REGION_RATES_CANDIDATES = [
    OUTPUT_DIR / "clean_health_region_doctors_per_100k_2024.csv",
    OUTPUT_DIR / "clean_health_region_doctors_per_100k_2024.parquet",
]

CROSSWALK_FILLED_CANDIDATES = [
    LOOKUP_DIR / "quebec_census_division_to_health_region_crosswalk_filled.csv",
    LOOKUP_DIR / "quebec_census_division_to_health_region_crosswalk_filled.parquet",
]

CROSSWALK_UNRESOLVED_CANDIDATES = [
    LOOKUP_DIR / "quebec_census_division_to_health_region_crosswalk_unresolved.csv",
    LOOKUP_DIR / "quebec_census_division_to_health_region_crosswalk_unresolved.parquet",
]

BASE_CD_CANDIDATES = [
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.csv",
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.parquet",
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.geojson",
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.gpkg",
]

OUTPUT_CSV = OUTPUT_DIR / "clean_census_division_doctors_per_100k_proxy_2024.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_census_division_doctors_per_100k_proxy_2024.parquet"
OUTPUT_SUMMARY = OUTPUT_DIR / "clean_census_division_doctors_per_100k_proxy_summary_2024.csv"
OUTPUT_UNRESOLVED = OUTPUT_DIR / "clean_census_division_doctors_per_100k_proxy_unresolved_2024.csv"
OUTPUT_AUDIT = OUTPUT_DIR / "clean_census_division_doctors_per_100k_proxy_audit_2024.csv"


# -----------------------------
# Constants
# -----------------------------

SOURCE_NAME = "CIHI — Physicians per 100,000 Population, by Specialty"
SOURCE_FEATURE = "Family medicine physicians per 100,000 population"
SOURCE_YEAR = 2024

CSV_ENCODING_CANDIDATES = [
    "utf-8",
    "utf-8-sig",
    "cp1252",
    "latin1",
]

HEALTH_REGION_KEY_CANDIDATES = [
    "health_region_name",
    "health_region",
    "region_name",
    "Region",
    "region",
]

HEALTH_REGION_RATE_CANDIDATES = [
    "physicians_per_100k_health_region",
    "physicians_per_100k",
    "family_medicine_physicians_per_100k",
    "indicator_value",
    "value",
]

CD_KEY_CANDIDATES = [
    "census_division_dguid",
    "manual_census_division_dguid",
    "statcan_dguid",
    "DGUID",
    "dguid",
    "cd_dguid",
]

BASE_IDENTITY_COLUMNS = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "census_division_type",
    "province_code",
    "province_name",
    "geography_level",
    "census_year",
    "population_total_2021",
    "land_area_km2",
    "has_positive_population",
]


# -----------------------------
# Helpers
# -----------------------------

def read_csv_with_encodings(path: Path, **kwargs) -> pd.DataFrame:
    last_error = None

    for encoding in CSV_ENCODING_CANDIDATES:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return read_csv_with_encodings(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix in [".geojson", ".gpkg", ".shp"]:
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError(f"geopandas is required to read {path}") from exc
        return gpd.read_file(path)

    raise ValueError(f"Unsupported file type: {path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def safe_relative(path: Path | None) -> str:
    if path is None:
        return ""

    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        return str(path)


def looks_like_mojibake(text: object) -> bool:
    if pd.isna(text):
        return False

    s = str(text)
    suspicious_tokens = ["Ã", "Â", "�"]
    return any(token in s for token in suspicious_tokens)


def choose_display_name(row: pd.Series) -> str:
    base_name = row.get("census_division_name", "")
    crosswalk_name = row.get("crosswalk_census_division_name", "")

    if looks_like_mojibake(base_name) and not looks_like_mojibake(crosswalk_name):
        return crosswalk_name

    if pd.isna(base_name) or str(base_name).strip() == "":
        return crosswalk_name

    return base_name


def summarize_numeric(series: pd.Series) -> dict:
    numeric = clean_numeric(series)

    return {
        "non_missing": int(numeric.notna().sum()),
        "missing": int(numeric.isna().sum()),
        "min": numeric.min(skipna=True),
        "max": numeric.max(skipna=True),
        "mean": numeric.mean(skipna=True),
        "median": numeric.median(skipna=True),
    }


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"{label} is missing required columns:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(df.columns)
        )


# -----------------------------
# Resolve input paths
# -----------------------------

health_region_path = first_existing_path(HEALTH_REGION_RATES_CANDIDATES)
crosswalk_filled_path = first_existing_path(CROSSWALK_FILLED_CANDIDATES)
crosswalk_unresolved_path = first_existing_path(CROSSWALK_UNRESOLVED_CANDIDATES)
base_cd_path = first_existing_path(BASE_CD_CANDIDATES)

if health_region_path is None:
    raise FileNotFoundError(
        "Could not find clean health-region doctors-per-100k table. Expected one of:\n"
        + "\n".join(str(path) for path in HEALTH_REGION_RATES_CANDIDATES)
    )

if crosswalk_filled_path is None:
    raise FileNotFoundError(
        "Could not find filled CD-to-health-region crosswalk. Expected one of:\n"
        + "\n".join(str(path) for path in CROSSWALK_FILLED_CANDIDATES)
    )

if base_cd_path is None:
    raise FileNotFoundError(
        "Could not find cleaned Québec census-division base frame. Expected one of:\n"
        + "\n".join(str(path) for path in BASE_CD_CANDIDATES)
    )


# -----------------------------
# Load inputs
# -----------------------------

health_region = normalize_columns(read_table(health_region_path))
crosswalk = normalize_columns(read_table(crosswalk_filled_path))
base = normalize_columns(read_table(base_cd_path))

if "geometry" in base.columns:
    base = pd.DataFrame(base.drop(columns=["geometry"]))
else:
    base = pd.DataFrame(base)

if crosswalk_unresolved_path is not None:
    unresolved_input = normalize_columns(read_table(crosswalk_unresolved_path))
else:
    unresolved_input = pd.DataFrame()

print("\nLoaded inputs")
print("Health-region rates:", safe_relative(health_region_path), "rows:", len(health_region))
print("Filled crosswalk:", safe_relative(crosswalk_filled_path), "rows:", len(crosswalk))
print("Unresolved crosswalk:", safe_relative(crosswalk_unresolved_path), "rows:", len(unresolved_input))
print("Base CD frame:", safe_relative(base_cd_path), "rows:", len(base))


# -----------------------------
# Detect columns
# -----------------------------

health_region_key_col = first_existing_column(
    list(health_region.columns),
    HEALTH_REGION_KEY_CANDIDATES,
)

health_region_rate_col = first_existing_column(
    list(health_region.columns),
    HEALTH_REGION_RATE_CANDIDATES,
)

crosswalk_cd_key_col = first_existing_column(
    list(crosswalk.columns),
    CD_KEY_CANDIDATES,
)

crosswalk_health_region_col = first_existing_column(
    list(crosswalk.columns),
    HEALTH_REGION_KEY_CANDIDATES,
)

if health_region_key_col is None:
    raise ValueError(
        "Could not identify health-region key column in health-region table.\n"
        "Available columns:\n"
        + "\n".join(health_region.columns)
    )

if health_region_rate_col is None:
    raise ValueError(
        "Could not identify physicians-per-100k rate column in health-region table.\n"
        "Available columns:\n"
        + "\n".join(health_region.columns)
    )

if crosswalk_cd_key_col is None:
    raise ValueError(
        "Could not identify census-division DGUID column in filled crosswalk.\n"
        "Available columns:\n"
        + "\n".join(crosswalk.columns)
    )

if crosswalk_health_region_col is None:
    raise ValueError(
        "Could not identify health-region key column in filled crosswalk.\n"
        "Available columns:\n"
        + "\n".join(crosswalk.columns)
    )

require_columns(
    base,
    [
        "census_division_code",
        "census_division_dguid",
        "census_division_name",
        "population_total_2021",
        "land_area_km2",
    ],
    "Census-division base frame",
)


# -----------------------------
# Normalize inputs
# -----------------------------

health_region = health_region.copy()
crosswalk = crosswalk.copy()
base = base.copy()

health_region[health_region_key_col] = clean_text(health_region[health_region_key_col])
health_region[health_region_rate_col] = clean_numeric(health_region[health_region_rate_col])

crosswalk[crosswalk_cd_key_col] = clean_text(crosswalk[crosswalk_cd_key_col])
crosswalk[crosswalk_health_region_col] = clean_text(crosswalk[crosswalk_health_region_col])

base["census_division_dguid"] = clean_text(base["census_division_dguid"])
base["census_division_code"] = clean_text(base["census_division_code"])
base["census_division_name"] = clean_text(base["census_division_name"])

if base["census_division_dguid"].duplicated().any():
    duplicates = base[base["census_division_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated census_division_dguid values in base frame:\n"
        + duplicates[
            [
                "census_division_code",
                "census_division_dguid",
                "census_division_name",
            ]
        ].head(40).to_string(index=False)
    )

if health_region[health_region_key_col].duplicated().any():
    duplicates = health_region[health_region[health_region_key_col].duplicated(keep=False)]
    raise ValueError(
        "Duplicated health-region keys found in health-region rate table:\n"
        + duplicates[[health_region_key_col, health_region_rate_col]].to_string(index=False)
    )

if crosswalk[crosswalk_cd_key_col].duplicated().any():
    duplicates = crosswalk[crosswalk[crosswalk_cd_key_col].duplicated(keep=False)]
    raise ValueError(
        "Duplicated census-division keys found in filled crosswalk:\n"
        + duplicates.head(40).to_string(index=False)
    )


# -----------------------------
# Build crosswalk source table
# -----------------------------

crosswalk_keep_cols = [
    col
    for col in [
        crosswalk_cd_key_col,
        "census_division_code",
        "census_division_name",
        crosswalk_health_region_col,
        "crosswalk_method",
        "crosswalk_note",
        "manual_assignment",
        "manual_assignment_note",
        "source",
    ]
    if col in crosswalk.columns
]

crosswalk_small = crosswalk[crosswalk_keep_cols].copy()

rename_map = {
    crosswalk_cd_key_col: "census_division_dguid",
    crosswalk_health_region_col: "health_region_name",
}

if "census_division_name" in crosswalk_small.columns:
    rename_map["census_division_name"] = "crosswalk_census_division_name"

if "census_division_code" in crosswalk_small.columns:
    rename_map["census_division_code"] = "crosswalk_census_division_code"

crosswalk_small = crosswalk_small.rename(columns=rename_map)

if "crosswalk_method" not in crosswalk_small.columns:
    crosswalk_small["crosswalk_method"] = "manual_region_membership"

if "crosswalk_note" not in crosswalk_small.columns:
    crosswalk_small["crosswalk_note"] = (
        "Assigned from Quebec MRC / territoire équivalent regional membership "
        "to the matching CIHI health-region label."
    )


# -----------------------------
# Join health-region rates to crosswalk
# -----------------------------

health_region_small = health_region[
    [
        col
        for col in [
            health_region_key_col,
            health_region_rate_col,
            "province",
            "year",
            "indicator",
            "unit_of_measure",
            "source_doctors_per_100k",
            "feature_description",
        ]
        if col in health_region.columns
    ]
].copy()

health_region_small = health_region_small.rename(
    columns={
        health_region_key_col: "health_region_name",
        health_region_rate_col: "physicians_per_100k",
    }
)

crosswalk_with_rates = crosswalk_small.merge(
    health_region_small,
    on="health_region_name",
    how="left",
    validate="many_to_one",
)

crosswalk_with_rates["physicians_per_100k"] = clean_numeric(
    crosswalk_with_rates["physicians_per_100k"]
)


# -----------------------------
# Join to base CD frame
# -----------------------------

identity_cols = [col for col in BASE_IDENTITY_COLUMNS if col in base.columns]

clean = base[identity_cols].copy()

clean = clean.merge(
    crosswalk_with_rates,
    on="census_division_dguid",
    how="left",
    validate="one_to_one",
)

clean["census_division_name_base_original"] = clean["census_division_name"]
clean["census_division_name_display"] = clean.apply(choose_display_name, axis=1)
clean["census_division_name"] = clean["census_division_name_display"]

clean["census_division_name_base_had_mojibake"] = clean[
    "census_division_name_base_original"
].apply(looks_like_mojibake)

if "crosswalk_census_division_name" in clean.columns:
    clean["crosswalk_census_division_name_had_mojibake"] = clean[
        "crosswalk_census_division_name"
    ].apply(looks_like_mojibake)
else:
    clean["crosswalk_census_division_name_had_mojibake"] = False

clean["physicians_per_100k"] = clean_numeric(clean["physicians_per_100k"])

# Explicit aliases / source-audit fields.
clean["physicians_per_100k_health_region_proxy"] = clean["physicians_per_100k"]
clean["physicians_per_100k_health_region"] = clean["physicians_per_100k"]

clean["physicians_per_100k__is_missing"] = clean["physicians_per_100k"].isna()
clean["physicians_per_100k__source"] = (
    "CIHI health-region rate assigned by census-division-to-health-region crosswalk"
)
clean["physicians_per_100k__source_name"] = SOURCE_NAME
clean["physicians_per_100k__source_feature"] = SOURCE_FEATURE
clean["physicians_per_100k__source_year"] = SOURCE_YEAR
clean["physicians_per_100k__source_file"] = safe_relative(health_region_path)
clean["physicians_per_100k__crosswalk_file"] = safe_relative(crosswalk_filled_path)
clean["physicians_per_100k__unresolved_crosswalk_file"] = safe_relative(crosswalk_unresolved_path)
clean["physicians_per_100k__source_geography"] = "CIHI health region"
clean["physicians_per_100k__target_geography"] = "Statistics Canada census division"
clean["physicians_per_100k__method_note"] = (
    "The physician rate is health-region-native. Each census division receives "
    "the physicians-per-100k value of its assigned CIHI health region. This is a "
    "proxy and not a direct census-division measurement."
)
clean["physicians_per_100k__proxy_quality"] = (
    "health_region_rate_assigned_to_census_division"
)

clean["doctors_proxy_year"] = SOURCE_YEAR
clean["doctors_proxy_complete"] = clean["physicians_per_100k"].notna()


# -----------------------------
# Unresolved output
# -----------------------------

unresolved_from_clean = clean[clean["physicians_per_100k"].isna()].copy()

if not unresolved_from_clean.empty:
    unresolved_from_clean["unresolved_reason"] = (
        "No health-region physician rate could be assigned through the filled crosswalk."
    )

if not unresolved_input.empty:
    unresolved_input_copy = unresolved_input.copy()
    unresolved_input_copy.columns = [str(col).strip() for col in unresolved_input_copy.columns]
else:
    unresolved_input_copy = pd.DataFrame()

unresolved_output = unresolved_from_clean.copy()

# Keep output compact but useful.
unresolved_keep_cols = [
    col
    for col in [
        "census_division_code",
        "census_division_dguid",
        "census_division_name",
        "census_division_type",
        "population_total_2021",
        "land_area_km2",
        "health_region_name",
        "crosswalk_method",
        "crosswalk_note",
        "physicians_per_100k",
        "unresolved_reason",
        "physicians_per_100k__method_note",
    ]
    if col in unresolved_output.columns
]

unresolved_output = unresolved_output[unresolved_keep_cols].copy()


# -----------------------------
# Audit output
# -----------------------------

audit_rows = [
    {
        "metric": "health_region_rate_file",
        "value": safe_relative(health_region_path),
    },
    {
        "metric": "crosswalk_filled_file",
        "value": safe_relative(crosswalk_filled_path),
    },
    {
        "metric": "crosswalk_unresolved_file",
        "value": safe_relative(crosswalk_unresolved_path),
    },
    {
        "metric": "base_cd_frame",
        "value": safe_relative(base_cd_path),
    },
    {
        "metric": "health_region_rows",
        "value": len(health_region),
    },
    {
        "metric": "filled_crosswalk_rows",
        "value": len(crosswalk),
    },
    {
        "metric": "base_cd_rows",
        "value": len(base),
    },
    {
        "metric": "clean_rows",
        "value": len(clean),
    },
    {
        "metric": "unique_census_divisions",
        "value": clean["census_division_dguid"].nunique(),
    },
    {
        "metric": "health_region_key_column",
        "value": health_region_key_col,
    },
    {
        "metric": "health_region_rate_column",
        "value": health_region_rate_col,
    },
    {
        "metric": "crosswalk_cd_key_column",
        "value": crosswalk_cd_key_col,
    },
    {
        "metric": "crosswalk_health_region_column",
        "value": crosswalk_health_region_col,
    },
    {
        "metric": "physicians_per_100k_non_missing",
        "value": int(clean["physicians_per_100k"].notna().sum()),
    },
    {
        "metric": "physicians_per_100k_missing",
        "value": int(clean["physicians_per_100k"].isna().sum()),
    },
    {
        "metric": "doctors_proxy_full_coverage",
        "value": bool(clean["physicians_per_100k"].notna().sum() == len(clean)),
    },
    {
        "metric": "base_names_with_mojibake",
        "value": int(clean["census_division_name_base_had_mojibake"].sum()),
    },
    {
        "metric": "crosswalk_names_with_mojibake",
        "value": int(clean["crosswalk_census_division_name_had_mojibake"].sum()),
    },
    {
        "metric": "display_names_with_mojibake",
        "value": int(clean["census_division_name"].apply(looks_like_mojibake).sum()),
    },
]

rate_summary = summarize_numeric(clean["physicians_per_100k"])

for metric_name, value in rate_summary.items():
    audit_rows.append(
        {
            "metric": f"physicians_per_100k_{metric_name}",
            "value": value,
        }
    )

if not unresolved_output.empty:
    audit_rows.append(
        {
            "metric": "unresolved_census_divisions",
            "value": ", ".join(unresolved_output["census_division_name"].astype(str)),
        }
    )

audit = pd.DataFrame(audit_rows)

summary = audit.copy()


# -----------------------------
# Final validation
# -----------------------------

if len(clean) != len(base):
    raise ValueError(
        f"Clean output row count changed unexpectedly: {len(clean)} vs base {len(base)}"
    )

if clean["census_division_dguid"].duplicated().any():
    duplicates = clean[clean["census_division_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated census_division_dguid values in clean output:\n"
        + duplicates[
            [
                "census_division_code",
                "census_division_dguid",
                "census_division_name",
            ]
        ].head(40).to_string(index=False)
    )

if clean["physicians_per_100k"].notna().sum() == 0:
    raise ValueError("No physicians_per_100k values were assigned.")

# We intentionally do NOT fail on missing values, because Nord-du-Québec can
# legitimately remain unresolved pending a separate allocation decision.


# -----------------------------
# Save outputs
# -----------------------------

clean.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

try:
    clean.to_parquet(OUTPUT_PARQUET, index=False)
    parquet_saved = True
except Exception as exc:
    parquet_saved = False
    print("\nWARNING: Could not save Parquet output.")
    print("Reason:", exc)

summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")
audit.to_csv(OUTPUT_AUDIT, index=False, encoding="utf-8")
unresolved_output.to_csv(OUTPUT_UNRESOLVED, index=False, encoding="utf-8")


# -----------------------------
# Console diagnostics
# -----------------------------

print("\n" + "=" * 72)
print("CLEAN CENSUS DIVISION DOCTORS PER 100K PROXY 2024")
print("=" * 72)

print("\nFinal clean table:")
print("Rows:", len(clean))
print("Unique census divisions:", clean["census_division_dguid"].nunique())

print("\nInput sources:")
print("Health-region rates:", safe_relative(health_region_path))
print("Filled crosswalk:", safe_relative(crosswalk_filled_path))
print("Unresolved crosswalk:", safe_relative(crosswalk_unresolved_path))
print("Base CD frame:", safe_relative(base_cd_path))

print("\nDetected columns:")
print("Health-region key:", health_region_key_col)
print("Health-region rate:", health_region_rate_col)
print("Crosswalk CD key:", crosswalk_cd_key_col)
print("Crosswalk health-region key:", crosswalk_health_region_col)

print("\nPhysician proxy coverage:")
print("Non-missing:", int(clean["physicians_per_100k"].notna().sum()))
print("Missing:", int(clean["physicians_per_100k"].isna().sum()))
print("Full coverage:", bool(clean["physicians_per_100k"].notna().sum() == len(clean)))

print("\nPhysician proxy summary:")
print(clean["physicians_per_100k"].describe())

print("\nUnresolved census divisions:")
if unresolved_output.empty:
    print("[none]")
else:
    print(unresolved_output.to_string(index=False))

print("\nHealth-region assignment counts:")
if "health_region_name" in clean.columns:
    print(clean["health_region_name"].fillna("[missing]").value_counts().sort_index().to_string())

print("\nPreview:")
preview_cols = [
    "census_division_code",
    "census_division_name",
    "health_region_name",
    "physicians_per_100k",
    "crosswalk_method",
    "crosswalk_note",
]
preview_cols = [col for col in preview_cols if col in clean.columns]
print(clean[preview_cols].head(20).to_string(index=False))

print("\nSaved:")
print(OUTPUT_CSV)
if parquet_saved:
    print(OUTPUT_PARQUET)
print(OUTPUT_SUMMARY)
print(OUTPUT_AUDIT)
print(OUTPUT_UNRESOLVED)

print("\nDone.")