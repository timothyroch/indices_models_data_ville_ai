from pathlib import Path
import re
import pandas as pd


# ============================================================
# Clean Census Division Agriculture 2021
# ============================================================
#
# Purpose:
#   Clean the Census of Agriculture land-use table for:
#
#       PCTFARMS92 -> pct_land_farms
#
# Source:
#   Statistics Canada Census of Agriculture, 2021
#   Table 32-10-0249-01
#   Land use, Census of Agriculture, 2021
#
# Main formula:
#
#   pct_land_farms =
#       100 * total_farm_area_km2 / land_area_km2
#
# Preferred source row:
#
#   Land use = Total farm area
#   Unit of measure = Hectares
#
# Important:
#   Four Québec census divisions have positive farm counts but suppressed /
#   unavailable area values. These are left missing, not set to zero.
#
#   PCTRFRM90 / pct_rural_farm is not cleaned here because this table does
#   not contain a rural farm population measure.
#
# Run from data/:
#
#   python census_division_agriculture_2021/clean_census_division_agriculture_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_agriculture_2021"
RAW_DIR = SECTION_DIR / "raw"
OUTPUT_DIR = SECTION_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_LAND_USE_CSV = RAW_DIR / "land_use_32100249_2021.csv"
RAW_LAND_USE_METADATA_CSV = RAW_DIR / "land_use_32100249_2021_MetaData.csv"

BASE_CD_FRAME = (
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.csv"
)

OUTPUT_CLEAN = OUTPUT_DIR / "clean_census_division_agriculture_2021.csv"
OUTPUT_SOURCE_ROWS = OUTPUT_DIR / "clean_census_division_agriculture_total_farm_area_source_rows_2021.csv"
OUTPUT_MISSING_AUDIT = OUTPUT_DIR / "clean_census_division_agriculture_missing_area_audit_2021.csv"
OUTPUT_METADATA = OUTPUT_DIR / "clean_census_division_agriculture_variable_metadata_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "clean_census_division_agriculture_summary_2021.csv"


# -----------------------------
# Config
# -----------------------------

EXPECTED_QC_CD_COUNT = 98

ENCODING_CANDIDATES = [
    "utf-8-sig",
    "utf-8",
    "cp1252",
    "latin1",
]

ACRES_TO_KM2 = 0.0040468564224
HECTARES_TO_KM2 = 0.01

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
    "land_area_km2",
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


def select_encoding(path: Path) -> str:
    for encoding in ENCODING_CANDIDATES:
        try:
            pd.read_csv(path, encoding=encoding, nrows=20, low_memory=False)
            return encoding
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not decode {path} with candidates {ENCODING_CANDIDATES}",
    )


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = text.replace("\u00a0", " ")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_lower(value: object) -> str:
    return normalize_text(value).lower()


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
    return int(series.astype("string").str.contains("Ã|Â|�", regex=True, na=False).sum())


def find_exact_column(columns: list[str], target: str) -> str | None:
    for col in columns:
        if col.strip().lower() == target.strip().lower():
            return col
    return None


def require_column(columns: list[str], target: str) -> str:
    col = find_exact_column(columns, target)
    if col is None:
        raise ValueError(
            f"Required column not found: {target}\n\nAvailable columns:\n"
            + "\n".join(columns)
        )
    return col


def extract_unit_rows(
    source: pd.DataFrame,
    unit_col: str,
    unit_value: str,
    value_col: str,
    status_col: str,
    symbol_col: str | None,
    prefix: str,
) -> pd.DataFrame:
    subset = source[source[unit_col].map(normalize_lower) == unit_value.lower()].copy()

    keep_cols = ["DGUID", "GEO", unit_col, value_col, status_col]
    if symbol_col and symbol_col in subset.columns:
        keep_cols.append(symbol_col)

    subset = subset[keep_cols].copy()
    subset = subset.drop_duplicates(subset=["DGUID"], keep="first")

    rename = {
        "DGUID": "source_dguid",
        "GEO": f"{prefix}_source_geo",
        unit_col: f"{prefix}_unit",
        value_col: f"{prefix}_value_raw",
        status_col: f"{prefix}_status",
    }

    if symbol_col and symbol_col in subset.columns:
        rename[symbol_col] = f"{prefix}_symbol"

    subset = subset.rename(columns=rename)
    subset[f"{prefix}_value"] = clean_numeric(subset[f"{prefix}_value_raw"])

    return subset


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


def missing_reason(row: pd.Series) -> str:
    if pd.notna(row["total_farm_area_km2"]):
        return "available"

    farms = row.get("farms_reporting_total_farm_area", pd.NA)

    hectares_status = normalize_text(row.get("total_farm_area_hectares_status", ""))
    acres_status = normalize_text(row.get("total_farm_area_acres_status", ""))

    if pd.notna(farms) and farms == 0:
        return "zero_farms"

    if pd.notna(farms) and farms > 0:
        return (
            "area_value_missing_positive_farm_count"
            f"_hectares_status={hectares_status or 'missing'}"
            f"_acres_status={acres_status or 'missing'}"
        )

    return "area_value_missing_unknown_farm_count"


# -----------------------------
# Validate inputs
# -----------------------------

if not RAW_LAND_USE_CSV.exists():
    raise FileNotFoundError(f"Missing raw land-use CSV:\n{RAW_LAND_USE_CSV}")

if not BASE_CD_FRAME.exists():
    raise FileNotFoundError(f"Missing base CD frame:\n{BASE_CD_FRAME}")


# -----------------------------
# Load base frame
# -----------------------------

base = pd.read_csv(BASE_CD_FRAME, dtype=str, low_memory=False)
base.columns = [str(col).strip() for col in base.columns]

required_base_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "land_area_km2",
]

missing_base_cols = [col for col in required_base_cols if col not in base.columns]
if missing_base_cols:
    raise ValueError(
        "Base frame is missing required columns:\n"
        + "\n".join(missing_base_cols)
        + "\n\nAvailable columns:\n"
        + "\n".join(base.columns)
    )

base = base.copy()
base["census_division_code"] = base["census_division_code"].astype("string").str.strip()
base["census_division_dguid"] = base["census_division_dguid"].astype("string").str.strip()
base["census_division_name"] = base["census_division_name"].astype("string").str.strip()
base["land_area_km2"] = clean_numeric(base["land_area_km2"])

if len(base) != EXPECTED_QC_CD_COUNT:
    raise ValueError(f"Expected {EXPECTED_QC_CD_COUNT} Québec CDs in base frame, got {len(base)}.")

if base["census_division_dguid"].duplicated().any():
    raise ValueError("Duplicate census_division_dguid values in base frame.")

if base["land_area_km2"].isna().any():
    raise ValueError("Missing land_area_km2 values in base frame.")

base_dguid_set = set(base["census_division_dguid"].dropna().astype(str))


# -----------------------------
# Load raw land-use table
# -----------------------------

encoding = select_encoding(RAW_LAND_USE_CSV)

raw = pd.read_csv(
    RAW_LAND_USE_CSV,
    encoding=encoding,
    dtype=str,
    low_memory=False,
)

raw.columns = [str(col).strip() for col in raw.columns]

dguid_col = require_column(list(raw.columns), "DGUID")
geo_col = require_column(list(raw.columns), "GEO")
land_use_col = require_column(list(raw.columns), "Land use")
unit_col = require_column(list(raw.columns), "Unit of measure")
value_col = require_column(list(raw.columns), "VALUE")
status_col = require_column(list(raw.columns), "STATUS")
symbol_col = find_exact_column(list(raw.columns), "SYMBOL")

raw["DGUID"] = raw[dguid_col].astype("string").str.strip()
raw["GEO"] = raw[geo_col].astype("string").str.strip()
raw["_land_use_norm"] = raw[land_use_col].map(normalize_lower)
raw["_unit_norm"] = raw[unit_col].map(normalize_lower)

qc_total_farm_area_rows = raw[
    raw["DGUID"].isin(base_dguid_set)
    & (raw["_land_use_norm"] == "total farm area")
].copy()

if qc_total_farm_area_rows.empty:
    raise ValueError("No Québec CD rows found for Land use = Total farm area.")

qc_total_farm_area_rows.to_csv(OUTPUT_SOURCE_ROWS, index=False, encoding="utf-8")


# -----------------------------
# Extract unit-specific rows
# -----------------------------

farms_reporting = extract_unit_rows(
    source=qc_total_farm_area_rows,
    unit_col=unit_col,
    unit_value="Number of farms reporting",
    value_col=value_col,
    status_col=status_col,
    symbol_col=symbol_col,
    prefix="farms_reporting_total_farm_area",
)

hectares = extract_unit_rows(
    source=qc_total_farm_area_rows,
    unit_col=unit_col,
    unit_value="Hectares",
    value_col=value_col,
    status_col=status_col,
    symbol_col=symbol_col,
    prefix="total_farm_area_hectares",
)

acres = extract_unit_rows(
    source=qc_total_farm_area_rows,
    unit_col=unit_col,
    unit_value="Acres",
    value_col=value_col,
    status_col=status_col,
    symbol_col=symbol_col,
    prefix="total_farm_area_acres",
)


# -----------------------------
# Build clean table
# -----------------------------

identity_cols = [col for col in IDENTITY_COLUMNS if col in base.columns]

clean = base[identity_cols].copy()

clean = clean.merge(
    farms_reporting.drop(columns=["farms_reporting_total_farm_area_source_geo"], errors="ignore"),
    left_on="census_division_dguid",
    right_on="source_dguid",
    how="left",
    validate="one_to_one",
)

clean = clean.drop(columns=["source_dguid"], errors="ignore")

clean = clean.merge(
    hectares.drop(columns=["total_farm_area_hectares_source_geo"], errors="ignore"),
    left_on="census_division_dguid",
    right_on="source_dguid",
    how="left",
    validate="one_to_one",
)

clean = clean.drop(columns=["source_dguid"], errors="ignore")

clean = clean.merge(
    acres.drop(columns=["total_farm_area_acres_source_geo"], errors="ignore"),
    left_on="census_division_dguid",
    right_on="source_dguid",
    how="left",
    validate="one_to_one",
)

clean = clean.drop(columns=["source_dguid"], errors="ignore")

clean = clean.rename(
    columns={
        "farms_reporting_total_farm_area_value": "farms_reporting_total_farm_area",
        "farms_reporting_total_farm_area_value_raw": "farms_reporting_total_farm_area_raw",
        "farms_reporting_total_farm_area_status": "farms_reporting_total_farm_area_status",
        "total_farm_area_hectares_value": "total_farm_area_hectares",
        "total_farm_area_hectares_value_raw": "total_farm_area_hectares_raw",
        "total_farm_area_hectares_status": "total_farm_area_hectares_status",
        "total_farm_area_acres_value": "total_farm_area_acres",
        "total_farm_area_acres_value_raw": "total_farm_area_acres_raw",
        "total_farm_area_acres_status": "total_farm_area_acres_status",
    }
)

for col in [
    "farms_reporting_total_farm_area",
    "total_farm_area_hectares",
    "total_farm_area_acres",
]:
    if col in clean.columns:
        clean[col] = clean_numeric(clean[col])


# -----------------------------
# Derived area and pct_land_farms
# -----------------------------

clean["total_farm_area_km2_from_hectares"] = clean["total_farm_area_hectares"] * HECTARES_TO_KM2
clean["total_farm_area_km2_from_acres"] = clean["total_farm_area_acres"] * ACRES_TO_KM2

clean["total_farm_area_km2"] = clean["total_farm_area_km2_from_hectares"].combine_first(
    clean["total_farm_area_km2_from_acres"]
)

clean["total_farm_area_source_unit"] = pd.NA
clean.loc[clean["total_farm_area_km2_from_hectares"].notna(), "total_farm_area_source_unit"] = "Hectares"
clean.loc[
    clean["total_farm_area_km2_from_hectares"].isna()
    & clean["total_farm_area_km2_from_acres"].notna(),
    "total_farm_area_source_unit",
] = "Acres"

# Only set true zeros if the table explicitly reports zero farms.
zero_farm_mask = (
    clean["total_farm_area_km2"].isna()
    & clean["farms_reporting_total_farm_area"].notna()
    & (clean["farms_reporting_total_farm_area"] == 0)
)

clean.loc[zero_farm_mask, "total_farm_area_km2"] = 0
clean.loc[zero_farm_mask, "total_farm_area_source_unit"] = "zero_farms"

clean["pct_land_farms"] = 100 * clean["total_farm_area_km2"] / clean["land_area_km2"]

clean["total_farm_area_missing_reason"] = clean.apply(missing_reason, axis=1)

clean["total_farm_area_suppressed_or_unavailable"] = (
    clean["total_farm_area_km2"].isna()
    & clean["farms_reporting_total_farm_area"].notna()
    & (clean["farms_reporting_total_farm_area"] > 0)
)

clean["pct_land_farms_is_missing"] = clean["pct_land_farms"].isna()

clean["source_file"] = safe_relative(RAW_LAND_USE_CSV)
clean["method_note"] = (
    "pct_land_farms is computed as 100 * total_farm_area_km2 / land_area_km2. "
    "Total farm area is taken from Census of Agriculture table 32-10-0249-01, "
    "Land use = Total farm area, Unit of measure = Hectares when available, with acres retained as fallback. "
    "Suppressed or unavailable farm-area values are left missing, not set to zero."
)


# -----------------------------
# Validation
# -----------------------------

if len(clean) != EXPECTED_QC_CD_COUNT:
    raise ValueError(f"Clean output has {len(clean)} rows; expected {EXPECTED_QC_CD_COUNT}.")

if clean["census_division_dguid"].duplicated().any():
    dupes = clean[clean["census_division_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicate census_division_dguid values in clean output:\n"
        + dupes[["census_division_code", "census_division_dguid", "census_division_name"]].to_string(index=False)
    )

if clean["farms_reporting_total_farm_area"].isna().any():
    missing_farm_count = clean[clean["farms_reporting_total_farm_area"].isna()]
    raise ValueError(
        "Missing farms_reporting_total_farm_area values. This suggests source-row extraction failed:\n"
        + missing_farm_count[["census_division_code", "census_division_dguid", "census_division_name"]].to_string(index=False)
    )

if (clean["pct_land_farms"].dropna() < 0).any():
    raise ValueError("Negative pct_land_farms values found.")

if (clean["pct_land_farms"].dropna() > 100).any():
    over = clean[clean["pct_land_farms"] > 100]
    raise ValueError(
        "pct_land_farms values over 100 found:\n"
        + over[["census_division_code", "census_division_name", "pct_land_farms"]].to_string(index=False)
    )

pct_non_missing = int(clean["pct_land_farms"].notna().sum())
pct_missing = int(clean["pct_land_farms"].isna().sum())

base_names_with_mojibake = contains_mojibake(base["census_division_name"])
clean_names_with_mojibake = contains_mojibake(clean["census_division_name"])


# -----------------------------
# Missing audit
# -----------------------------

missing_audit = clean[clean["pct_land_farms"].isna()][
    [
        "census_division_code",
        "census_division_dguid",
        "census_division_name",
        "farms_reporting_total_farm_area",
        "farms_reporting_total_farm_area_status",
        "total_farm_area_hectares",
        "total_farm_area_hectares_status",
        "total_farm_area_acres",
        "total_farm_area_acres_status",
        "total_farm_area_missing_reason",
        "total_farm_area_suppressed_or_unavailable",
    ]
].copy()

missing_audit.to_csv(OUTPUT_MISSING_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Metadata
# -----------------------------

metadata_rows = [
    {
        "variable": "pct_land_farms",
        "original_sovi_code": "PCTFARMS92",
        "description": "Total farm area as a percentage of census-division land area",
        "source_table": "32-10-0249-01",
        "source_dataset": "Land use, Census of Agriculture, 2021",
        "source_rows": "Land use = Total farm area; Unit of measure = Hectares preferred, Acres fallback",
        "unit": "percent",
        "derivation": "100 * total_farm_area_km2 / land_area_km2",
        "coverage": f"{pct_non_missing}/{EXPECTED_QC_CD_COUNT} numeric values",
        "status": "partial_documented_suppression_or_unavailability",
        "notes": (
            "Some census divisions have positive numbers of farms reporting, but area values are unavailable "
            "with STATUS = F. These values are left missing rather than imputed to zero."
        ),
    },
    {
        "variable": "farms_reporting_total_farm_area",
        "original_sovi_code": "",
        "description": "Number of farms reporting total farm area",
        "source_table": "32-10-0249-01",
        "source_dataset": "Land use, Census of Agriculture, 2021",
        "source_rows": "Land use = Total farm area; Unit of measure = Number of farms reporting",
        "unit": "farms",
        "derivation": "direct table value",
        "coverage": f"{int(clean['farms_reporting_total_farm_area'].notna().sum())}/{EXPECTED_QC_CD_COUNT}",
        "status": "audit_variable",
        "notes": "Retained to distinguish zero-farm areas from suppressed or unavailable farm-area values.",
    },
    {
        "variable": "total_farm_area_km2",
        "original_sovi_code": "",
        "description": "Total farm area converted to square kilometres",
        "source_table": "32-10-0249-01",
        "source_dataset": "Land use, Census of Agriculture, 2021",
        "source_rows": "Land use = Total farm area; Unit of measure = Hectares preferred, Acres fallback",
        "unit": "square_kilometres",
        "derivation": "hectares * 0.01, or acres * 0.0040468564224 if hectares unavailable",
        "coverage": f"{int(clean['total_farm_area_km2'].notna().sum())}/{EXPECTED_QC_CD_COUNT}",
        "status": "component_audit_variable",
        "notes": "Used as numerator for pct_land_farms.",
    },
    {
        "variable": "pct_rural_farm",
        "original_sovi_code": "PCTRFRM90",
        "description": "Percent rural farm population",
        "source_table": "",
        "source_dataset": "",
        "source_rows": "",
        "unit": "percent",
        "derivation": "",
        "coverage": "0/98",
        "status": "unresolved_not_cleaned",
        "notes": (
            "The Census of Agriculture land-use table does not contain rural farm population. "
            "This variable requires a separate farm-population or Agriculture-Population Linkage source."
        ),
    },
]

metadata = pd.DataFrame(metadata_rows)
metadata.to_csv(OUTPUT_METADATA, index=False, encoding="utf-8")


# -----------------------------
# Summary
# -----------------------------

summary_rows = [
    {"metric": "raw_csv", "value": safe_relative(RAW_LAND_USE_CSV)},
    {
        "metric": "metadata_csv",
        "value": safe_relative(RAW_LAND_USE_METADATA_CSV) if RAW_LAND_USE_METADATA_CSV.exists() else "",
    },
    {"metric": "base_cd_frame", "value": safe_relative(BASE_CD_FRAME)},
    {"metric": "raw_encoding", "value": encoding},
    {"metric": "clean_rows", "value": len(clean)},
    {"metric": "unique_census_divisions", "value": clean["census_division_dguid"].nunique()},
    {"metric": "variables_cleaned", "value": "pct_land_farms"},
    {"metric": "variables_unresolved", "value": "pct_rural_farm"},
    {"metric": "pct_land_farms_non_missing", "value": pct_non_missing},
    {"metric": "pct_land_farms_missing", "value": pct_missing},
    {
        "metric": "pct_land_farms_missing_census_divisions",
        "value": " | ".join(clean.loc[clean["pct_land_farms"].isna(), "census_division_name"].astype(str)),
    },
    {
        "metric": "pct_land_farms_missing_reason",
        "value": "positive farms reporting but total farm area suppressed or unavailable",
    },
    {
        "metric": "total_farm_area_suppressed_or_unavailable_count",
        "value": int(clean["total_farm_area_suppressed_or_unavailable"].sum()),
    },
    {
        "metric": "pct_land_farms_values_over_100",
        "value": int((clean["pct_land_farms"].dropna() > 100).sum()),
    },
    {"metric": "base_names_with_mojibake", "value": base_names_with_mojibake},
    {"metric": "clean_names_with_mojibake", "value": clean_names_with_mojibake},
]

for variable in [
    "farms_reporting_total_farm_area",
    "total_farm_area_hectares",
    "total_farm_area_acres",
    "total_farm_area_km2",
    "pct_land_farms",
]:
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
            "If this summary shows 98 rows, 94 non-missing pct_land_farms values, 4 documented missing "
            "values due to unavailable/suppressed farm area, no values over 100, and no mojibake, generate "
            "the README and add a SoVI YAML mapping for PCTFARMS92 with partial documented coverage."
        ),
    }
)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Save clean output
# -----------------------------

ordered_cols = (
    identity_cols
    + [
        "pct_land_farms",
        "total_farm_area_km2",
        "total_farm_area_hectares",
        "total_farm_area_acres",
        "total_farm_area_source_unit",
        "farms_reporting_total_farm_area",
        "farms_reporting_total_farm_area_status",
        "total_farm_area_hectares_status",
        "total_farm_area_acres_status",
        "total_farm_area_missing_reason",
        "total_farm_area_suppressed_or_unavailable",
        "pct_land_farms_is_missing",
        "source_file",
        "method_note",
    ]
)

ordered_cols = [col for col in ordered_cols if col in clean.columns]
clean = clean[ordered_cols].copy()
clean.to_csv(OUTPUT_CLEAN, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CLEAN CENSUS DIVISION AGRICULTURE 2021")
print("=" * 72)

print("\nClean output:")
print("Rows:", len(clean))
print("Unique census divisions:", clean["census_division_dguid"].nunique())
print("Variables cleaned: pct_land_farms")
print("Variables unresolved: pct_rural_farm")

print("\npct_land_farms coverage:")
print("Non-missing:", pct_non_missing)
print("Missing:", pct_missing)

print("\nMissing pct_land_farms audit:")
if missing_audit.empty:
    print("[none]")
else:
    print(missing_audit.to_string(index=False))

print("\nMain summaries:")
for variable in [
    "farms_reporting_total_farm_area",
    "total_farm_area_km2",
    "pct_land_farms",
]:
    print("\n", variable)
    print(clean[variable].describe())

print("\nMojibake check:")
print("Base names with mojibake:", base_names_with_mojibake)
print("Clean names with mojibake:", clean_names_with_mojibake)

print("\nSaved:")
print(OUTPUT_CLEAN)
print(OUTPUT_SOURCE_ROWS)
print(OUTPUT_MISSING_AUDIT)
print(OUTPUT_METADATA)
print(OUTPUT_SUMMARY)

print("\nDone.")