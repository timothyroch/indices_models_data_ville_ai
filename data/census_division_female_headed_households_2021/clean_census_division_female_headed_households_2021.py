from pathlib import Path
import re
import pandas as pd
import codecs

# ============================================================
# Clean Census Division Female-Headed Households 2021
# ============================================================
#
# Purpose:
#   Clean a Québec census-division proxy for the original SoVI variable:
#
#       PCTF_HH90 -> pct_female_headed_households
#
# Canadian Census Profile adaptation:
#
#   Main target:
#       CHARACTERISTIC_ID = 87
#       CHARACTERISTIC_NAME = "in which the parent is a woman+"
#       Value column = C10_RATE_TOTAL
#
#   Context rows:
#       CHARACTERISTIC_ID = 86
#       CHARACTERISTIC_NAME = "Total one-parent families"
#
#       CHARACTERISTIC_ID = 78
#       CHARACTERISTIC_NAME = "Total number of census families in private households - 100% data"
#
# Interpretation:
#   The main variable is the rate of female-parent one-parent census families
#   in the Census Profile family structure universe. This is not a literal
#   all-private-households female-headed-household measure, but it is a strong
#   Canadian adaptation of the original SoVI concept: female-headed households,
#   no spouse present.
#
# Run from data/:
#
#   python census_division_female_headed_households_2021/clean_census_division_female_headed_households_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_female_headed_households_2021"
OUTPUT_DIR = SECTION_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_PROFILE = (
    DATA_DIR
    / "census_profile_census_division_2021"
    / "raw"
    / "98-401-X2021004_English_CSV_data.csv"
)

BASE_CD_FRAME = (
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.csv"
)

OUTPUT_CLEAN = OUTPUT_DIR / "clean_census_division_female_headed_households_2021.csv"
OUTPUT_SOURCE_ROWS = OUTPUT_DIR / "clean_census_division_female_headed_households_source_rows_2021.csv"
OUTPUT_METADATA = OUTPUT_DIR / "clean_census_division_female_headed_households_variable_metadata_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "clean_census_division_female_headed_households_summary_2021.csv"


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

CHARACTERISTICS = {
    "female_one_parent_families": {
        "id": "87",
        "expected_name_contains": "in which the parent is a woman",
        "role": "main_target",
    },
    "total_one_parent_families": {
        "id": "86",
        "expected_name_contains": "total one-parent families",
        "role": "context_denominator_for_female_share_of_one_parent_families",
    },
    "total_census_families": {
        "id": "78",
        "expected_name_contains": "total number of census families",
        "role": "census_family_denominator_audit",
    },
}

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



def can_decode_file(path: Path, encoding: str, chunk_size: int = 1024 * 1024) -> bool:
    decoder = codecs.getincrementaldecoder(encoding)()

    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                decoder.decode(chunk)

        decoder.decode(b"", final=True)
        return True

    except UnicodeDecodeError:
        return False


def select_encoding(path: Path) -> str:
    for encoding in ENCODING_CANDIDATES:
        if can_decode_file(path, encoding):
            return encoding

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


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {col.lower(): col for col in columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for candidate in candidates:
        candidate_lower = candidate.lower()
        for col in columns:
            if candidate_lower in col.lower():
                return col

    return None


def require_column(columns: list[str], candidates: list[str], label: str) -> str:
    col = find_column(columns, candidates)
    if col is None:
        raise ValueError(
            f"Could not detect {label} column.\nCandidates: {candidates}\n\nAvailable columns:\n"
            + "\n".join(columns)
        )
    return col


def numeric_summary(series: pd.Series) -> dict:
    values = clean_numeric(series)

    return {
        "non_missing": int(values.notna().sum()),
        "missing": int(values.isna().sum()),
        "min": values.min(skipna=True),
        "max": values.max(skipna=True),
        "mean": values.mean(skipna=True),
        "median": values.median(skipna=True),
    }


def extract_characteristic(
    qc_rows: pd.DataFrame,
    alias: str,
    config: dict,
    dguid_col: str,
    char_id_col: str,
    char_name_col: str,
    count_col: str,
    rate_col: str,
    count_symbol_col: str | None,
    rate_symbol_col: str | None,
) -> tuple[pd.DataFrame, dict]:
    char_id = str(config["id"])

    rows = qc_rows[qc_rows[char_id_col].astype(str).str.strip() == char_id].copy()

    if rows.empty:
        raise ValueError(f"No rows found for characteristic ID {char_id} ({alias}).")

    rows[char_name_col] = rows[char_name_col].map(normalize_text)
    unique_names = sorted(rows[char_name_col].dropna().astype(str).unique())

    if len(unique_names) != 1:
        raise ValueError(
            f"Expected one characteristic name for ID {char_id}, got:\n"
            + "\n".join(unique_names)
        )

    characteristic_name = unique_names[0]
    expected_fragment = config["expected_name_contains"].lower()

    if expected_fragment not in characteristic_name.lower():
        raise ValueError(
            f"Characteristic ID {char_id} name does not match expectation.\n"
            f"Expected to contain: {expected_fragment}\n"
            f"Actual: {characteristic_name}"
        )

    out_cols = [
        dguid_col,
        char_id_col,
        char_name_col,
        count_col,
        rate_col,
    ]

    if count_symbol_col and count_symbol_col in rows.columns:
        out_cols.append(count_symbol_col)

    if rate_symbol_col and rate_symbol_col in rows.columns and rate_symbol_col not in out_cols:
        out_cols.append(rate_symbol_col)

    out = rows[out_cols].copy()
    out[dguid_col] = out[dguid_col].astype("string").str.strip()
    out[count_col] = clean_numeric(out[count_col])
    out[rate_col] = clean_numeric(out[rate_col])

    duplicate_dguid_count = int(out[dguid_col].duplicated().sum())
    if duplicate_dguid_count != 0:
        raise ValueError(
            f"Duplicate DGUID rows for characteristic ID {char_id} ({alias}): {duplicate_dguid_count}"
        )

    out = out.rename(
        columns={
            dguid_col: "census_division_dguid",
            char_id_col: f"{alias}_characteristic_id",
            char_name_col: f"{alias}_characteristic_name",
            count_col: f"{alias}_count",
            rate_col: f"{alias}_rate",
        }
    )

    if count_symbol_col and count_symbol_col in rows.columns:
        out = out.rename(columns={count_symbol_col: f"{alias}_count_symbol"})

    if rate_symbol_col and rate_symbol_col in rows.columns:
        out = out.rename(columns={rate_symbol_col: f"{alias}_rate_symbol"})

    count_values = clean_numeric(out[f"{alias}_count"])
    rate_values = clean_numeric(out[f"{alias}_rate"])

    inventory = {
        "alias": alias,
        "characteristic_id": char_id,
        "characteristic_name": characteristic_name,
        "role": config["role"],
        "rows_extracted": len(out),
        "unique_census_divisions": out["census_division_dguid"].nunique(),
        "count_non_missing": int(count_values.notna().sum()),
        "count_missing": int(count_values.isna().sum()),
        "count_min": count_values.min(skipna=True),
        "count_max": count_values.max(skipna=True),
        "count_mean": count_values.mean(skipna=True),
        "count_median": count_values.median(skipna=True),
        "rate_non_missing": int(rate_values.notna().sum()),
        "rate_missing": int(rate_values.isna().sum()),
        "rate_min": rate_values.min(skipna=True),
        "rate_max": rate_values.max(skipna=True),
        "rate_mean": rate_values.mean(skipna=True),
        "rate_median": rate_values.median(skipna=True),
        "coverage_is_98_cds": out["census_division_dguid"].nunique() == EXPECTED_QC_CD_COUNT,
    }

    return out, inventory


# -----------------------------
# Validate inputs
# -----------------------------

if not RAW_PROFILE.exists():
    raise FileNotFoundError(f"Missing Census Profile raw CSV:\n{RAW_PROFILE}")

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
base["census_division_dguid"] = base["census_division_dguid"].astype("string").str.strip()

if len(base) != EXPECTED_QC_CD_COUNT:
    raise ValueError(f"Expected {EXPECTED_QC_CD_COUNT} Québec CDs, got {len(base)}.")

if base["census_division_dguid"].duplicated().any():
    raise ValueError("Duplicate census_division_dguid values in base frame.")

base_dguid_set = set(base["census_division_dguid"].dropna().astype(str))


# -----------------------------
# Load Census Profile
# -----------------------------

raw_encoding = select_encoding(RAW_PROFILE)

raw = pd.read_csv(
    RAW_PROFILE,
    encoding=raw_encoding,
    dtype=str,
    low_memory=False,
)

raw.columns = [str(col).strip() for col in raw.columns]

columns = list(raw.columns)

dguid_col = require_column(columns, ["DGUID", "dguid"], "DGUID")
char_id_col = require_column(columns, ["CHARACTERISTIC_ID", "Characteristic ID"], "characteristic ID")
char_name_col = require_column(columns, ["CHARACTERISTIC_NAME", "Characteristic Name"], "characteristic name")
count_col = require_column(columns, ["C1_COUNT_TOTAL", "Count total", "Total - Count"], "count")
rate_col = require_column(columns, ["C10_RATE_TOTAL", "Rate total", "Total - Rate"], "rate")

count_symbol_col = find_column(columns, ["SYMBOL.1", "C1_SYMBOL_TOTAL", "SYMBOL"])
rate_symbol_col = find_column(columns, ["SYMBOL.3", "C10_SYMBOL_TOTAL", "SYMBOL"])

raw[dguid_col] = raw[dguid_col].astype("string").str.strip()
qc_rows = raw[raw[dguid_col].isin(base_dguid_set)].copy()

if qc_rows.empty:
    raise ValueError("No Québec CD rows found in Census Profile raw file after DGUID filtering.")


print("\nCleaning Census Division Female-Headed Households 2021")
print("Raw profile:", safe_relative(RAW_PROFILE))
print("Raw encoding:", raw_encoding)
print("Base frame:", safe_relative(BASE_CD_FRAME))
print("Base rows:", len(base))


# -----------------------------
# Extract forced characteristics
# -----------------------------

source_frames = []
inventory_rows = []

for alias, config in CHARACTERISTICS.items():
    extracted, inventory = extract_characteristic(
        qc_rows=qc_rows,
        alias=alias,
        config=config,
        dguid_col=dguid_col,
        char_id_col=char_id_col,
        char_name_col=char_name_col,
        count_col=count_col,
        rate_col=rate_col,
        count_symbol_col=count_symbol_col,
        rate_symbol_col=rate_symbol_col,
    )

    source_frames.append(extracted)
    inventory_rows.append(inventory)

    print(
        f"  {alias}: ID {inventory['characteristic_id']}, "
        f"rows={inventory['rows_extracted']}, "
        f"count_non_missing={inventory['count_non_missing']}, "
        f"rate_non_missing={inventory['rate_non_missing']}"
    )

source_rows = pd.concat(source_frames, ignore_index=False, sort=False)
source_rows.to_csv(OUTPUT_SOURCE_ROWS, index=False, encoding="utf-8")

inventory = pd.DataFrame(inventory_rows)


# -----------------------------
# Build clean table
# -----------------------------

identity_cols = [col for col in IDENTITY_COLUMNS if col in base.columns]

clean = base[identity_cols].copy()

for extracted in source_frames:
    clean = clean.merge(
        extracted,
        on="census_division_dguid",
        how="left",
        validate="one_to_one",
    )

# Main SoVI variable.
clean["pct_female_headed_households"] = clean["female_one_parent_families_rate"]

# Audit / context variables.
clean["female_one_parent_families_count"] = clean["female_one_parent_families_count"]
clean["pct_female_one_parent_families_of_census_families"] = clean["female_one_parent_families_rate"]

clean["total_one_parent_families_count"] = clean["total_one_parent_families_count"]
clean["pct_one_parent_families"] = clean["total_one_parent_families_rate"]

clean["total_census_families_count"] = clean["total_census_families_count"]

clean["female_share_of_one_parent_families"] = (
    100
    * clean["female_one_parent_families_count"]
    / clean["total_one_parent_families_count"]
)

clean["source_file"] = safe_relative(RAW_PROFILE)
clean["method_note"] = (
    "pct_female_headed_households maps PCTF_HH90 to Census Profile characteristic ID 87, "
    "'in which the parent is a woman+', using C10_RATE_TOTAL. This is a Canadian census-family "
    "proxy for female-headed/no-spouse-present households. It captures female-parent one-parent "
    "census families as a percentage of the Census Profile family universe, not all private households."
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

required_numeric_cols = [
    "pct_female_headed_households",
    "female_one_parent_families_count",
    "pct_female_one_parent_families_of_census_families",
    "total_one_parent_families_count",
    "pct_one_parent_families",
    "total_census_families_count",
    "female_share_of_one_parent_families",
]

for col in required_numeric_cols:
    clean[col] = clean_numeric(clean[col])
    missing = int(clean[col].isna().sum())
    if missing != 0:
        raise ValueError(f"Unexpected missing values in {col}: {missing}")

if (clean["pct_female_headed_households"] < 0).any() or (clean["pct_female_headed_households"] > 100).any():
    raise ValueError("pct_female_headed_households has values outside [0, 100].")

if (clean["female_share_of_one_parent_families"] < 0).any() or (clean["female_share_of_one_parent_families"] > 100).any():
    raise ValueError("female_share_of_one_parent_families has values outside [0, 100].")

female_share_formula_diff = (
    clean["female_share_of_one_parent_families"]
    - 100 * clean["female_one_parent_families_count"] / clean["total_one_parent_families_count"]
).abs().max(skipna=True)

target_alias_diff = (
    clean["pct_female_headed_households"]
    - clean["pct_female_one_parent_families_of_census_families"]
).abs().max(skipna=True)

if female_share_formula_diff != 0:
    raise ValueError(f"female_share_of_one_parent_families formula check failed: {female_share_formula_diff}")

if target_alias_diff != 0:
    raise ValueError(f"pct_female_headed_households alias check failed: {target_alias_diff}")

base_names_with_mojibake = contains_mojibake(base["census_division_name"])
clean_names_with_mojibake = contains_mojibake(clean["census_division_name"])


# -----------------------------
# Metadata
# -----------------------------

metadata_rows = [
    {
        "variable": "pct_female_headed_households",
        "original_sovi_code": "PCTF_HH90",
        "description": "Female-parent one-parent census families as a percentage of the Census Profile family universe",
        "source_dataset": "2021 Census Profile, census divisions",
        "source_characteristic_id": "87",
        "source_characteristic_name": "in which the parent is a woman+",
        "source_column": "C10_RATE_TOTAL",
        "unit": "percent",
        "derivation": "direct Census Profile rate",
        "coverage": "98/98",
        "status": "ready_full_coverage",
        "notes": (
            "Canadian census-family proxy for the original female-headed households/no-spouse-present SoVI concept. "
            "Not a literal all-private-households female household maintainer measure."
        ),
    },
    {
        "variable": "female_one_parent_families_count",
        "original_sovi_code": "",
        "description": "Count of one-parent census families in which the parent is a woman",
        "source_dataset": "2021 Census Profile, census divisions",
        "source_characteristic_id": "87",
        "source_characteristic_name": "in which the parent is a woman+",
        "source_column": "C1_COUNT_TOTAL",
        "unit": "families",
        "derivation": "direct Census Profile count",
        "coverage": "98/98",
        "status": "audit_variable",
        "notes": "Retained for reproducibility and interpretation.",
    },
    {
        "variable": "pct_one_parent_families",
        "original_sovi_code": "",
        "description": "Total one-parent families as a percentage of the Census Profile family universe",
        "source_dataset": "2021 Census Profile, census divisions",
        "source_characteristic_id": "86",
        "source_characteristic_name": "Total one-parent families",
        "source_column": "C10_RATE_TOTAL",
        "unit": "percent",
        "derivation": "direct Census Profile rate",
        "coverage": "98/98",
        "status": "context_variable",
        "notes": "Used to contextualize the main female-parent one-parent-family rate.",
    },
    {
        "variable": "female_share_of_one_parent_families",
        "original_sovi_code": "",
        "description": "Share of one-parent families in which the parent is a woman",
        "source_dataset": "2021 Census Profile, census divisions",
        "source_characteristic_id": "87 / 86",
        "source_characteristic_name": "in which the parent is a woman+ / Total one-parent families",
        "source_column": "C1_COUNT_TOTAL",
        "unit": "percent",
        "derivation": "100 * female_one_parent_families_count / total_one_parent_families_count",
        "coverage": "98/98",
        "status": "audit_variable",
        "notes": "This is not the default SoVI variable, but is useful for checking the composition of one-parent families.",
    },
    {
        "variable": "total_census_families_count",
        "original_sovi_code": "",
        "description": "Total number of census families in private households",
        "source_dataset": "2021 Census Profile, census divisions",
        "source_characteristic_id": "78",
        "source_characteristic_name": "Total number of census families in private households - 100% data",
        "source_column": "C1_COUNT_TOTAL",
        "unit": "families",
        "derivation": "direct Census Profile count",
        "coverage": "98/98",
        "status": "denominator_audit_variable",
        "notes": "Retained as a denominator/context audit variable.",
    },
]

metadata = pd.DataFrame(metadata_rows)
metadata.to_csv(OUTPUT_METADATA, index=False, encoding="utf-8")


# -----------------------------
# Summary
# -----------------------------

summary_rows = [
    {"metric": "raw_profile_csv", "value": safe_relative(RAW_PROFILE)},
    {"metric": "base_cd_frame", "value": safe_relative(BASE_CD_FRAME)},
    {"metric": "raw_encoding", "value": raw_encoding},
    {"metric": "raw_rows", "value": len(raw)},
    {"metric": "quebec_cd_rows_scanned", "value": len(qc_rows)},
    {"metric": "clean_rows", "value": len(clean)},
    {"metric": "unique_census_divisions", "value": clean["census_division_dguid"].nunique()},
    {"metric": "variables_cleaned", "value": "pct_female_headed_households"},
    {"metric": "source_characteristic_ids", "value": "87, 86, 78"},
    {"metric": "all_required_numeric_columns_complete", "value": bool(clean[required_numeric_cols].notna().all().all())},
    {"metric": "pct_female_headed_households_source", "value": "CHARACTERISTIC_ID 87, C10_RATE_TOTAL"},
    {"metric": "female_share_of_one_parent_families_formula_max_abs_difference", "value": female_share_formula_diff},
    {"metric": "pct_female_headed_households_alias_max_abs_difference", "value": target_alias_diff},
    {"metric": "base_names_with_mojibake", "value": base_names_with_mojibake},
    {"metric": "clean_names_with_mojibake", "value": clean_names_with_mojibake},
]

for _, row in inventory.iterrows():
    summary_rows.append(
        {
            "metric": f"{row['alias']}_characteristic_id",
            "value": row["characteristic_id"],
        }
    )
    summary_rows.append(
        {
            "metric": f"{row['alias']}_characteristic_name",
            "value": row["characteristic_name"],
        }
    )
    summary_rows.append(
        {
            "metric": f"{row['alias']}_coverage_is_98_cds",
            "value": row["coverage_is_98_cds"],
        }
    )

for variable in required_numeric_cols:
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
            "If this summary shows 98 rows, complete numeric coverage, zero formula differences, "
            "and no mojibake, generate the README and add a SoVI YAML mapping for "
            "PCTF_HH90 -> pct_female_headed_households."
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
        "pct_female_headed_households",
        "female_one_parent_families_count",
        "pct_female_one_parent_families_of_census_families",
        "total_one_parent_families_count",
        "pct_one_parent_families",
        "total_census_families_count",
        "female_share_of_one_parent_families",
        "female_one_parent_families_characteristic_id",
        "female_one_parent_families_characteristic_name",
        "total_one_parent_families_characteristic_id",
        "total_one_parent_families_characteristic_name",
        "total_census_families_characteristic_id",
        "total_census_families_characteristic_name",
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
print("CLEAN CENSUS DIVISION FEMALE-HEADED HOUSEHOLDS 2021")
print("=" * 72)

print("\nClean output:")
print("Rows:", len(clean))
print("Unique census divisions:", clean["census_division_dguid"].nunique())
print("Variables cleaned: pct_female_headed_households")

print("\nFormula checks:")
print("female_share_of_one_parent_families formula max abs difference:", female_share_formula_diff)
print("pct_female_headed_households alias max abs difference:", target_alias_diff)

print("\nMojibake check:")
print("Base names with mojibake:", base_names_with_mojibake)
print("Clean names with mojibake:", clean_names_with_mojibake)

print("\nMain summaries:")
for variable in [
    "pct_female_headed_households",
    "female_one_parent_families_count",
    "pct_one_parent_families",
    "female_share_of_one_parent_families",
]:
    print("\n", variable)
    print(clean[variable].describe())

print("\nSource characteristic inventory:")
print(inventory.to_string(index=False))

print("\nPreview:")
preview_cols = [
    "census_division_code",
    "census_division_name",
    "pct_female_headed_households",
    "female_one_parent_families_count",
    "total_one_parent_families_count",
    "pct_one_parent_families",
    "female_share_of_one_parent_families",
    "total_census_families_count",
]
preview_cols = [col for col in preview_cols if col in clean.columns]
print(clean[preview_cols].head(20).to_string(index=False))

print("\nSaved:")
print(OUTPUT_CLEAN)
print(OUTPUT_SOURCE_ROWS)
print(OUTPUT_METADATA)
print(OUTPUT_SUMMARY)

print("\nDone.")