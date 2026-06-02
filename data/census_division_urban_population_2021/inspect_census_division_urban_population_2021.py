from pathlib import Path
import re
import pandas as pd


# ============================================================
# Inspect Census Division Urban Population 2021
# ============================================================
#
# Purpose:
#   Inspect whether a defensible census-division-level proxy exists for:
#
#       pct_urban
#
# Original SoVI variable:
#
#       PCTURB90 = Percent urban population, 1990
#
# Methodological issue:
#   The original SoVI variable is a population share: percent of residents
#   living in urban areas. At Canadian census-division geography, this may
#   require one of the following:
#
#       1. A direct Census Profile urban / rural / population-centre row.
#       2. A population-centre overlay or aggregation from smaller geography.
#       3. A documented density-based proxy.
#       4. Leaving the variable unresolved.
#
# This script inspects the raw Census Profile and the existing base frame.
# It does not create a clean final variable.
#
# Run from data/:
#   python census_division_urban_population_2021/inspect_census_division_urban_population_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_urban_population_2021"
OUTPUT_DIR = SECTION_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CENSUS_PROFILE = (
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

OUTPUT_KEYWORD_MATCHES = OUTPUT_DIR / "urban_population_keyword_matches_2021.csv"
OUTPUT_CHARACTERISTIC_INVENTORY = OUTPUT_DIR / "urban_population_candidate_characteristic_inventory_2021.csv"
OUTPUT_TARGET_SUMMARY = OUTPUT_DIR / "urban_population_target_characteristic_summary_2021.csv"
OUTPUT_BASE_FRAME_PROXY_AUDIT = OUTPUT_DIR / "urban_population_base_frame_proxy_audit_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "urban_population_inspection_summary_2021.csv"


# -----------------------------
# Config
# -----------------------------

ENCODING_CANDIDATES = [
    "cp1252",
    "utf-8-sig",
    "utf-8",
    "latin1",
]

CHUNKSIZE = 200_000

QUEBEC_CD_DGUID_PREFIX = "2021A000324"
EXPECTED_QC_CD_COUNT = 98

VALUE_RATE_COLUMN = "C10_RATE_TOTAL"
VALUE_COUNT_COLUMN = "C1_COUNT_TOTAL"
SYMBOL_RATE_COLUMN = "SYMBOL.3"
SYMBOL_COUNT_COLUMN = "SYMBOL"

RAW_COLUMNS_NEEDED = [
    "CENSUS_YEAR",
    "DGUID",
    "ALT_GEO_CODE",
    "GEO_LEVEL",
    "GEO_NAME",
    "TNR_SF",
    "TNR_LF",
    "DATA_QUALITY_FLAG",
    "CHARACTERISTIC_ID",
    "CHARACTERISTIC_NAME",
    "CHARACTERISTIC_NOTE",
    "C1_COUNT_TOTAL",
    "SYMBOL",
    "C10_RATE_TOTAL",
    "SYMBOL.3",
]

SEARCH_KEYWORDS = sorted(
    {
        "urban",
        "rural",
        "population centre",
        "population center",
        "small population centre",
        "medium population centre",
        "large urban population centre",
        "rural area",
        "rural areas",
        "population density",
    }
)

TARGET_CANDIDATES = {
    "direct_pct_urban": {
        "canonical_variable": "pct_urban",
        "original_code": "PCTURB90",
        "description": "Percent urban population",
        "positive_terms": [
            "urban",
            "population centre",
            "population center",
        ],
        "exclude_terms": [
            "language",
            "commuting",
            "place of work",
            "income",
            "education",
            "major field of study",
            "occupation",
            "industry",
        ],
        "value_mode": "direct_rate_candidate",
        "unit": "percent",
        "notes": (
            "Candidate direct Census Profile row for urban or population-centre population share. "
            "Needs review because not every keyword match is a valid urban-population denominator."
        ),
    },
    "direct_pct_rural_inverse": {
        "canonical_variable": "pct_urban",
        "original_code": "PCTURB90",
        "description": "Percent urban population from 100 - rural share",
        "positive_terms": [
            "rural",
            "rural area",
            "rural areas",
        ],
        "exclude_terms": [
            "language",
            "commuting",
            "place of work",
            "income",
            "education",
            "major field of study",
            "occupation",
            "industry",
        ],
        "value_mode": "inverse_rate_candidate",
        "unit": "percent",
        "notes": (
            "Candidate inverse formula: pct_urban = 100 - pct_rural, if a valid rural-population "
            "share row exists with the correct denominator."
        ),
    },
}


# -----------------------------
# Helpers
# -----------------------------

def select_encoding(path: Path) -> str:
    for encoding in ENCODING_CANDIDATES:
        try:
            pd.read_csv(path, encoding=encoding, nrows=5, low_memory=False)
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

    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def contains_any(text: object, patterns: list[str]) -> bool:
    text_norm = normalize_text(text)
    return any(pattern in text_norm for pattern in patterns)


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        return str(path)


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


def summarize_candidate(
    rows: pd.DataFrame,
    candidate_alias: str,
    candidate_config: dict,
    match_reason: str,
    preferred: bool,
) -> dict:
    if rows.empty:
        return {
            "canonical_variable": candidate_config["canonical_variable"],
            "original_code": candidate_config["original_code"],
            "candidate_alias": candidate_alias,
            "description": candidate_config["description"],
            "match_reason": match_reason,
            "preferred": preferred,
            "CHARACTERISTIC_ID": "",
            "CHARACTERISTIC_NAME": "",
            "CHARACTERISTIC_NOTE": "",
            "value_mode": candidate_config["value_mode"],
            "value_column": VALUE_RATE_COLUMN,
            "symbol_column": SYMBOL_RATE_COLUMN,
            "unit": candidate_config["unit"],
            "n_rows": 0,
            "n_unique_quebec_cds": 0,
            "value_non_missing": 0,
            "value_missing": EXPECTED_QC_CD_COUNT,
            "value_min": None,
            "value_max": None,
            "value_mean": None,
            "value_median": None,
            "coverage_is_98_cds": False,
            "status": "candidate_not_found",
            "notes": candidate_config["notes"],
        }

    values = clean_numeric(rows[VALUE_RATE_COLUMN])

    characteristic_ids = sorted(rows["CHARACTERISTIC_ID"].astype(str).str.strip().unique())
    characteristic_names = sorted(rows["CHARACTERISTIC_NAME"].astype(str).str.strip().unique())
    characteristic_notes = sorted(rows["CHARACTERISTIC_NOTE"].astype(str).str.strip().unique())

    one_characteristic = len(characteristic_ids) == 1
    coverage_is_98 = int(rows["DGUID"].nunique()) == EXPECTED_QC_CD_COUNT
    values_complete = int(values.notna().sum()) == EXPECTED_QC_CD_COUNT

    if one_characteristic and coverage_is_98 and values_complete:
        status = "candidate_found_full_coverage"
    elif coverage_is_98 and values_complete:
        status = "candidate_found_full_coverage_multiple_characteristics"
    else:
        status = "candidate_needs_review"

    return {
        "canonical_variable": candidate_config["canonical_variable"],
        "original_code": candidate_config["original_code"],
        "candidate_alias": candidate_alias,
        "description": candidate_config["description"],
        "match_reason": match_reason,
        "preferred": preferred,
        "CHARACTERISTIC_ID": characteristic_ids[0] if one_characteristic else "; ".join(characteristic_ids),
        "CHARACTERISTIC_NAME": characteristic_names[0] if len(characteristic_names) == 1 else "; ".join(characteristic_names),
        "CHARACTERISTIC_NOTE": characteristic_notes[0] if len(characteristic_notes) == 1 else "; ".join(characteristic_notes),
        "value_mode": candidate_config["value_mode"],
        "value_column": VALUE_RATE_COLUMN,
        "symbol_column": SYMBOL_RATE_COLUMN,
        "unit": candidate_config["unit"],
        "n_rows": len(rows),
        "n_unique_quebec_cds": int(rows["DGUID"].nunique()),
        "value_non_missing": int(values.notna().sum()),
        "value_missing": int(values.isna().sum()),
        "value_min": values.min(skipna=True),
        "value_max": values.max(skipna=True),
        "value_mean": values.mean(skipna=True),
        "value_median": values.median(skipna=True),
        "coverage_is_98_cds": coverage_is_98,
        "status": status,
        "notes": candidate_config["notes"],
    }


def rank_inventory_for_candidate(inventory: pd.DataFrame, config: dict) -> pd.DataFrame:
    if inventory.empty:
        return inventory.copy()

    out = inventory.copy()
    out["_name_norm"] = out["CHARACTERISTIC_NAME"].map(normalize_text)

    out["_excluded"] = out["_name_norm"].apply(lambda x: contains_any(x, config["exclude_terms"]))
    out["_positive_term_count"] = out["_name_norm"].apply(
        lambda x: sum(term in x for term in config["positive_terms"])
    )
    out["_coverage_score"] = out["n_unique_quebec_cds"].fillna(0)
    out["_non_missing_score"] = out["rate_non_missing"].fillna(0)

    # Prefer specific population-centre / urban/rural rows with full coverage.
    out = out.sort_values(
        [
            "_excluded",
            "_positive_term_count",
            "_coverage_score",
            "_non_missing_score",
            "CHARACTERISTIC_ID",
        ],
        ascending=[True, False, False, False, True],
    )

    return out.drop(
        columns=[
            "_name_norm",
            "_excluded",
            "_positive_term_count",
            "_coverage_score",
            "_non_missing_score",
        ]
    )


# -----------------------------
# Validate inputs
# -----------------------------

if not RAW_CENSUS_PROFILE.exists():
    raise FileNotFoundError(f"Missing raw Census Profile CD file:\n{RAW_CENSUS_PROFILE}")

if not BASE_CD_FRAME.exists():
    raise FileNotFoundError(f"Missing base CD frame:\n{BASE_CD_FRAME}")


# -----------------------------
# Load base frame and inspect possible proxy columns
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
        "Base CD frame is missing required columns:\n"
        + "\n".join(missing_base_cols)
        + "\n\nAvailable columns:\n"
        + "\n".join(base.columns)
    )

base["census_division_dguid"] = base["census_division_dguid"].astype("string").str.strip()

print("\nInspecting Census Division Urban Population 2021")
print("Raw Census Profile:", safe_relative(RAW_CENSUS_PROFILE))
print("Base CD frame:", safe_relative(BASE_CD_FRAME))
print("Base rows:", len(base))


# -----------------------------
# Base-frame proxy audit
# -----------------------------

base_proxy_rows = []

base_proxy_candidates = [
    {
        "candidate_name": "population_density_per_km2_2021",
        "column": "population_density_per_km2_2021",
        "interpretation": (
            "Density proxy only. This is not percent urban population, but may be used "
            "as a weak urbanization/intensity proxy if no direct urban/rural source exists."
        ),
    },
    {
        "candidate_name": "population_total_2021",
        "column": "population_total_2021",
        "interpretation": "Not an urban share. Useful only for checking scale and possible later weighting.",
    },
    {
        "candidate_name": "land_area_km2",
        "column": "land_area_km2",
        "interpretation": "Not an urban share. Useful as denominator for density proxies.",
    },
]

for candidate in base_proxy_candidates:
    col = candidate["column"]

    if col in base.columns:
        stats = numeric_summary(base[col])
        source_column_found = True
    else:
        stats = {
            "non_missing": 0,
            "missing": len(base),
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
        source_column_found = False

    base_proxy_rows.append(
        {
            "candidate_name": candidate["candidate_name"],
            "source_column": col,
            "source_column_found": source_column_found,
            "non_missing": stats["non_missing"],
            "missing": stats["missing"],
            "min": stats["min"],
            "max": stats["max"],
            "mean": stats["mean"],
            "median": stats["median"],
            "coverage_is_98_cds": stats["non_missing"] == EXPECTED_QC_CD_COUNT,
            "proxy_status": (
                "available_weak_proxy_or_auxiliary"
                if source_column_found and stats["non_missing"] == EXPECTED_QC_CD_COUNT
                else "not_available"
            ),
            "interpretation": candidate["interpretation"],
        }
    )

base_proxy_audit = pd.DataFrame(base_proxy_rows)
base_proxy_audit.to_csv(OUTPUT_BASE_FRAME_PROXY_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Scan raw Census Profile
# -----------------------------

encoding = select_encoding(RAW_CENSUS_PROFILE)
print("Raw encoding selected:", encoding)

keyword_match_chunks = []
total_rows_scanned = 0
quebec_cd_rows_scanned = 0

print("\nScanning raw Census Profile file...")

for chunk_idx, chunk in enumerate(
    pd.read_csv(
        RAW_CENSUS_PROFILE,
        encoding=encoding,
        dtype=str,
        usecols=lambda col: col in RAW_COLUMNS_NEEDED,
        chunksize=CHUNKSIZE,
        low_memory=False,
    ),
    start=1,
):
    chunk.columns = [str(col).strip() for col in chunk.columns]
    total_rows_scanned += len(chunk)

    if "GEO_LEVEL" not in chunk.columns or "DGUID" not in chunk.columns:
        raise ValueError("Raw file is missing GEO_LEVEL or DGUID columns.")

    qc_cd = chunk[
        (chunk["GEO_LEVEL"].astype("string").str.strip() == "Census division")
        & (chunk["DGUID"].astype("string").str.startswith(QUEBEC_CD_DGUID_PREFIX, na=False))
    ].copy()

    quebec_cd_rows_scanned += len(qc_cd)

    if qc_cd.empty:
        continue

    qc_cd["CHARACTERISTIC_NAME_NORM"] = qc_cd["CHARACTERISTIC_NAME"].map(normalize_text)

    keyword_mask = False
    for keyword in SEARCH_KEYWORDS:
        keyword_mask = keyword_mask | qc_cd["CHARACTERISTIC_NAME_NORM"].str.contains(
            re.escape(keyword),
            na=False,
        )

    matches = qc_cd[keyword_mask].copy()

    if not matches.empty:
        keyword_match_chunks.append(matches)

    if chunk_idx % 10 == 0:
        print(f"  Processed chunks: {chunk_idx}, rows so far: {total_rows_scanned}")


if keyword_match_chunks:
    keyword_matches = pd.concat(keyword_match_chunks, ignore_index=True)
else:
    keyword_matches = pd.DataFrame(columns=RAW_COLUMNS_NEEDED + ["CHARACTERISTIC_NAME_NORM"])

keyword_matches.to_csv(OUTPUT_KEYWORD_MATCHES, index=False, encoding="utf-8")

print("\nKeyword match rows:", len(keyword_matches))


# -----------------------------
# Characteristic inventory
# -----------------------------

inventory_rows = []

if not keyword_matches.empty:
    grouped = keyword_matches.groupby(
        ["CHARACTERISTIC_ID", "CHARACTERISTIC_NAME", "CHARACTERISTIC_NOTE"],
        dropna=False,
    )

    for (characteristic_id, characteristic_name, characteristic_note), group in grouped:
        values_rate = clean_numeric(group[VALUE_RATE_COLUMN]) if VALUE_RATE_COLUMN in group.columns else pd.Series(dtype="float64")
        values_count = clean_numeric(group[VALUE_COUNT_COLUMN]) if VALUE_COUNT_COLUMN in group.columns else pd.Series(dtype="float64")

        inventory_rows.append(
            {
                "CHARACTERISTIC_ID": str(characteristic_id).strip(),
                "CHARACTERISTIC_NAME": characteristic_name,
                "CHARACTERISTIC_NOTE": characteristic_note,
                "n_rows": len(group),
                "n_unique_quebec_cds": int(group["DGUID"].nunique()),
                "rate_non_missing": int(values_rate.notna().sum()),
                "rate_missing": int(values_rate.isna().sum()),
                "rate_min": values_rate.min(skipna=True),
                "rate_max": values_rate.max(skipna=True),
                "rate_mean": values_rate.mean(skipna=True),
                "rate_median": values_rate.median(skipna=True),
                "count_non_missing": int(values_count.notna().sum()),
                "count_missing": int(values_count.isna().sum()),
                "count_min": values_count.min(skipna=True),
                "count_max": values_count.max(skipna=True),
                "count_mean": values_count.mean(skipna=True),
                "name_norm": normalize_text(characteristic_name),
            }
        )

inventory = pd.DataFrame(inventory_rows)

if not inventory.empty:
    inventory = inventory.sort_values(
        ["CHARACTERISTIC_ID", "CHARACTERISTIC_NAME"],
        ascending=[True, True],
    )

inventory.to_csv(OUTPUT_CHARACTERISTIC_INVENTORY, index=False, encoding="utf-8")


# -----------------------------
# Target-level candidate summary
# -----------------------------

target_summary_rows = []

for candidate_alias, config in TARGET_CANDIDATES.items():
    if inventory.empty:
        target_summary_rows.append(
            summarize_candidate(
                rows=pd.DataFrame(),
                candidate_alias=candidate_alias,
                candidate_config=config,
                match_reason="inventory_empty",
                preferred=True,
            )
        )
        continue

    candidate_inventory = inventory[
        inventory["name_norm"].apply(lambda x: contains_any(x, config["positive_terms"]))
    ].copy()

    if candidate_inventory.empty:
        target_summary_rows.append(
            summarize_candidate(
                rows=pd.DataFrame(),
                candidate_alias=candidate_alias,
                candidate_config=config,
                match_reason=f"not_found: {'|'.join(config['positive_terms'])}",
                preferred=True,
            )
        )
        continue

    ranked = rank_inventory_for_candidate(candidate_inventory, config)

    # Keep top preferred candidate and several alternatives for review.
    selected_id = str(ranked.iloc[0]["CHARACTERISTIC_ID"]).strip()
    selected_rows = keyword_matches[
        keyword_matches["CHARACTERISTIC_ID"].astype(str).str.strip() == selected_id
    ].copy()

    target_summary_rows.append(
        summarize_candidate(
            rows=selected_rows,
            candidate_alias=candidate_alias,
            candidate_config=config,
            match_reason=f"top_ranked_keyword_match: {'|'.join(config['positive_terms'])}",
            preferred=True,
        )
    )

    for _, alt in ranked.iloc[1:].head(20).iterrows():
        alt_id = str(alt["CHARACTERISTIC_ID"]).strip()
        alt_rows = keyword_matches[
            keyword_matches["CHARACTERISTIC_ID"].astype(str).str.strip() == alt_id
        ].copy()

        target_summary_rows.append(
            summarize_candidate(
                rows=alt_rows,
                candidate_alias=f"{candidate_alias}_alternative",
                candidate_config=config,
                match_reason="alternative_keyword_match",
                preferred=False,
            )
        )


target_summary = pd.DataFrame(target_summary_rows)

if not target_summary.empty:
    target_summary = target_summary.sort_values(
        ["preferred", "candidate_alias", "CHARACTERISTIC_ID"],
        ascending=[False, True, True],
    )

target_summary.to_csv(OUTPUT_TARGET_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Summary
# -----------------------------

preferred_full_coverage = target_summary[
    (target_summary["preferred"] == True)
    & (target_summary["status"] == "candidate_found_full_coverage")
].copy()

direct_candidate_available = not preferred_full_coverage.empty

density_proxy_available = bool(
    base_proxy_audit[
        (base_proxy_audit["candidate_name"] == "population_density_per_km2_2021")
        & (base_proxy_audit["coverage_is_98_cds"] == True)
    ].shape[0]
)

if direct_candidate_available:
    recommended_status = "direct_or_inverse_candidate_needs_review"
elif density_proxy_available:
    recommended_status = "no_direct_candidate_yet_density_proxy_available"
else:
    recommended_status = "no_candidate_found"

summary_rows = [
    {"metric": "raw_csv", "value": safe_relative(RAW_CENSUS_PROFILE)},
    {"metric": "base_cd_frame", "value": safe_relative(BASE_CD_FRAME)},
    {"metric": "raw_encoding", "value": encoding},
    {"metric": "total_rows_scanned", "value": total_rows_scanned},
    {"metric": "quebec_cd_rows_scanned", "value": quebec_cd_rows_scanned},
    {"metric": "base_rows", "value": len(base)},
    {"metric": "keyword_match_rows", "value": len(keyword_matches)},
    {"metric": "candidate_characteristics_found", "value": len(inventory)},
    {"metric": "target_variable_inspected", "value": "pct_urban"},
    {"metric": "preferred_full_coverage_candidates_found", "value": len(preferred_full_coverage)},
    {
        "metric": "preferred_full_coverage_candidate_ids",
        "value": ", ".join(preferred_full_coverage["CHARACTERISTIC_ID"].astype(str)),
    },
    {
        "metric": "preferred_full_coverage_candidate_names",
        "value": " | ".join(preferred_full_coverage["CHARACTERISTIC_NAME"].astype(str)),
    },
    {"metric": "density_proxy_available_in_base_frame", "value": density_proxy_available},
    {"metric": "recommended_status", "value": recommended_status},
    {
        "metric": "important_method_note",
        "value": (
            "PCTURB90 is a percent-urban-population variable. A density proxy should not be used "
            "as equivalent unless no defensible direct urban/rural population-share source exists."
        ),
    },
    {
        "metric": "recommended_next_step",
        "value": (
            "Review urban_population_target_characteristic_summary_2021.csv. If a valid urban or rural "
            "population-share Census Profile row exists, generate a cleaner. If only density is available, "
            "decide whether to use population_density_per_km2_2021 as a weak proxy or leave pct_urban unresolved."
        ),
    },
]

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CENSUS DIVISION URBAN POPULATION INSPECTION 2021")
print("=" * 72)

print("\nScan summary:")
print(summary.to_string(index=False))

print("\nBase-frame proxy audit:")
print(base_proxy_audit.to_string(index=False))

print("\nTarget candidate summary:")
if target_summary.empty:
    print("[none]")
else:
    display_cols = [
        "canonical_variable",
        "original_code",
        "candidate_alias",
        "preferred",
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "value_mode",
        "n_unique_quebec_cds",
        "value_non_missing",
        "value_missing",
        "value_min",
        "value_max",
        "value_mean",
        "coverage_is_98_cds",
        "status",
    ]
    display_cols = [col for col in display_cols if col in target_summary.columns]
    print(target_summary[display_cols].to_string(index=False))

print("\nSaved:")
print(OUTPUT_KEYWORD_MATCHES)
print(OUTPUT_CHARACTERISTIC_INVENTORY)
print(OUTPUT_TARGET_SUMMARY)
print(OUTPUT_BASE_FRAME_PROXY_AUDIT)
print(OUTPUT_SUMMARY)

print("\nDone.")