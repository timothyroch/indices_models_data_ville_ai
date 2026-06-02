from pathlib import Path
import re
import pandas as pd


# ============================================================
# Inspect Census Division Employment Sector 2021
# ============================================================
#
# Purpose:
#   Inspect the 2021 Census Profile at census-division geography for
#   SoVI-like employment-sector variables:
#
#       pct_extractive_employment
#       pct_transport_utility_employment
#       pct_service_employment
#
# Original SoVI variables:
#
#       AGRIPC90  -> pct_extractive_employment
#       TRANPC90  -> pct_transport_utility_employment
#       SERVPC90  -> pct_service_employment
#
# Important:
#   This script is inspection-only. It does not clean the final variables.
#
# Methodological issue:
#   The original SoVI employment variables may mix industry and occupation
#   concepts. The Canadian Census Profile contains both NAICS industry rows
#   and NOC occupation rows. This inspection therefore keeps candidate rows
#   broad and auditable rather than assuming the first keyword match is correct.
#
# Run from data/:
#   python census_division_employment_sector_2021/inspect_census_division_employment_sector_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_employment_sector_2021"
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

OUTPUT_KEYWORD_MATCHES = OUTPUT_DIR / "employment_sector_keyword_matches_2021.csv"
OUTPUT_CHARACTERISTIC_INVENTORY = OUTPUT_DIR / "employment_sector_candidate_characteristic_inventory_2021.csv"
OUTPUT_TARGET_SUMMARY = OUTPUT_DIR / "employment_sector_target_characteristic_summary_2021.csv"
OUTPUT_DERIVED_FORMULA_AUDIT = OUTPUT_DIR / "employment_sector_derived_formula_audit_2021.csv"
OUTPUT_QC_PREVIEW = OUTPUT_DIR / "employment_sector_quebec_cd_preview_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "employment_sector_inspection_summary_2021.csv"


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

# Target concepts. These are deliberately not forced IDs yet.
# The point of this script is to discover the right Canadian Census Profile rows.
TARGETS = {
    "pct_extractive_employment": {
        "original_code": "AGRIPC90",
        "description": "Employment in agriculture / extractive sectors",
        "preferred_candidate_aliases": {
            "agriculture_forestry_fishing_hunting": [
                "agriculture, forestry, fishing and hunting",
            ],
            "mining_quarrying_oil_gas": [
                "mining, quarrying, and oil and gas extraction",
                "mining, quarrying and oil and gas extraction",
            ],
        },
        "secondary_terms": [
            "agriculture",
            "forestry",
            "fishing",
            "hunting",
            "mining",
            "quarrying",
            "oil and gas",
            "extraction",
            "natural resources",
        ],
        "exclude_terms": [
            "language",
            "place of work",
            "commuting",
            "major field of study",
            "certificate",
            "income",
        ],
        "expected_formula": (
            "Potential derived industry proxy: Agriculture, forestry, fishing and hunting "
            "+ Mining, quarrying, and oil and gas extraction."
        ),
        "unit": "percent",
        "sovi_role": "extractive_or_primary_industry_employment_proxy",
    },
    "pct_transport_utility_employment": {
        "original_code": "TRANPC90",
        "description": "Employment in transportation / communications / utilities",
        "preferred_candidate_aliases": {
            "transportation_warehousing": [
                "transportation and warehousing",
            ],
            "utilities": [
                "utilities",
            ],
            "information_cultural_industries": [
                "information and cultural industries",
            ],
        },
        "secondary_terms": [
            "transportation",
            "warehousing",
            "utilities",
            "information and cultural",
            "communications",
            "communication",
        ],
        "exclude_terms": [
            "language",
            "place of work",
            "commuting",
            "journey to work",
            "major field of study",
            "certificate",
            "income",
        ],
        "expected_formula": (
            "Potential derived industry proxy: Transportation and warehousing + Utilities "
            "+ possibly Information and cultural industries as the closest Canadian proxy "
            "for the communications part of the original SoVI concept."
        ),
        "unit": "percent",
        "sovi_role": "transportation_utility_communication_industry_proxy",
    },
    "pct_service_employment": {
        "original_code": "SERVPC90",
        "description": "Employment in service industries or service occupations",
        "preferred_candidate_aliases": {
            "service_producing_sector": [
                "service-producing sector",
                "service producing sector",
            ],
            "sales_service_occupations": [
                "sales and service occupations",
            ],
            "health_care_social_assistance": [
                "health care and social assistance",
            ],
            "educational_services": [
                "educational services",
            ],
            "accommodation_food_services": [
                "accommodation and food services",
            ],
            "other_services": [
                "other services",
            ],
            "public_administration": [
                "public administration",
            ],
            "retail_trade": [
                "retail trade",
            ],
            "finance_insurance": [
                "finance and insurance",
            ],
            "professional_scientific_technical": [
                "professional, scientific and technical services",
            ],
        },
        "secondary_terms": [
            "service-producing",
            "service producing",
            "sales and service",
            "services",
            "health care",
            "social assistance",
            "educational services",
            "accommodation",
            "food services",
            "public administration",
            "retail trade",
            "finance",
            "insurance",
            "professional, scientific",
            "technical services",
        ],
        "exclude_terms": [
            "language",
            "place of work",
            "commuting",
            "journey to work",
            "major field of study",
            "certificate",
            "income",
        ],
        "expected_formula": (
            "Needs review. Canadian options may include a broad service-producing-sector row, "
            "a sales-and-service occupation row, or a derived sum of service-industry NAICS sectors. "
            "The cleaner should not choose automatically without reviewing the inspection output."
        ),
        "unit": "percent",
        "sovi_role": "service_employment_proxy_needs_industry_vs_occupation_review",
    },
}

SEARCH_KEYWORDS = sorted(
    {
        "industry",
        "occupation",
        "naics",
        "noc",
        "agriculture",
        "forestry",
        "fishing",
        "hunting",
        "mining",
        "quarrying",
        "oil and gas",
        "extraction",
        "natural resources",
        "transportation",
        "warehousing",
        "utilities",
        "information and cultural",
        "communication",
        "communications",
        "service-producing",
        "service producing",
        "services",
        "sales and service",
        "retail trade",
        "health care",
        "social assistance",
        "educational services",
        "accommodation",
        "food services",
        "public administration",
        "finance and insurance",
        "professional, scientific",
        "technical services",
        "other services",
    }
)


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


def summarize_rows(
    rows: pd.DataFrame,
    target_key: str,
    candidate_alias: str,
    target_family: str,
    match_reason: str,
    preferred: bool,
    notes: str,
) -> dict:
    target = TARGETS[target_key]

    if rows.empty:
        return {
            "original_code": target["original_code"],
            "canonical_variable": target_key,
            "candidate_output_alias": candidate_alias,
            "target_family": target_family,
            "match_reason": match_reason,
            "preferred": preferred,
            "CHARACTERISTIC_ID": "",
            "CHARACTERISTIC_NAME": "",
            "CHARACTERISTIC_NOTE": "",
            "value_column": VALUE_RATE_COLUMN,
            "symbol_column": SYMBOL_RATE_COLUMN,
            "unit": target["unit"],
            "sovi_role": target["sovi_role"],
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
            "notes": notes,
        }

    values = clean_numeric(rows[VALUE_RATE_COLUMN])

    characteristic_ids = sorted(rows["CHARACTERISTIC_ID"].astype(str).str.strip().unique())
    characteristic_names = sorted(rows["CHARACTERISTIC_NAME"].astype(str).str.strip().unique())
    characteristic_notes = sorted(rows["CHARACTERISTIC_NOTE"].astype(str).str.strip().unique())

    one_characteristic = len(characteristic_ids) == 1

    coverage_is_98 = int(rows["DGUID"].nunique()) == EXPECTED_QC_CD_COUNT
    values_complete = int(values.notna().sum()) == EXPECTED_QC_CD_COUNT

    if coverage_is_98 and values_complete and one_characteristic:
        status = "candidate_found_full_coverage"
    elif coverage_is_98 and values_complete:
        status = "candidate_found_full_coverage_multiple_characteristics"
    else:
        status = "candidate_needs_review"

    return {
        "original_code": target["original_code"],
        "canonical_variable": target_key,
        "candidate_output_alias": candidate_alias,
        "target_family": target_family,
        "match_reason": match_reason,
        "preferred": preferred,
        "CHARACTERISTIC_ID": characteristic_ids[0] if one_characteristic else "; ".join(characteristic_ids),
        "CHARACTERISTIC_NAME": characteristic_names[0] if len(characteristic_names) == 1 else "; ".join(characteristic_names),
        "CHARACTERISTIC_NOTE": characteristic_notes[0] if len(characteristic_notes) == 1 else "; ".join(characteristic_notes),
        "value_column": VALUE_RATE_COLUMN,
        "symbol_column": SYMBOL_RATE_COLUMN,
        "unit": target["unit"],
        "sovi_role": target["sovi_role"],
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
        "notes": notes,
    }


def rank_inventory_for_terms(
    inventory: pd.DataFrame,
    positive_terms: list[str],
    exclude_terms: list[str],
) -> pd.DataFrame:
    if inventory.empty:
        return inventory.copy()

    out = inventory.copy()
    out["_name_norm"] = out["CHARACTERISTIC_NAME"].map(normalize_text)

    out["_excluded"] = out["_name_norm"].apply(lambda x: contains_any(x, exclude_terms))
    out["_positive_term_count"] = out["_name_norm"].apply(
        lambda x: sum(term in x for term in positive_terms)
    )

    # Prefer exact-ish content matches, non-excluded rows, full coverage, complete rates.
    out["_coverage_score"] = out["n_unique_quebec_cds"].fillna(0)
    out["_non_missing_score"] = out["rate_non_missing"].fillna(0)

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
        "Base CD frame is missing required columns:\n"
        + "\n".join(missing_base_cols)
        + "\n\nAvailable columns:\n"
        + "\n".join(base.columns)
    )

base["census_division_dguid"] = base["census_division_dguid"].astype("string").str.strip()

print("\nInspecting Census Division Employment Sector 2021")
print("Raw Census Profile:", safe_relative(RAW_CENSUS_PROFILE))
print("Base CD frame:", safe_relative(BASE_CD_FRAME))
print("Base rows:", len(base))


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

for target_key, target in TARGETS.items():
    if inventory.empty:
        for alias in target["preferred_candidate_aliases"]:
            target_summary_rows.append(
                summarize_rows(
                    rows=pd.DataFrame(),
                    target_key=target_key,
                    candidate_alias=alias,
                    target_family="preferred_candidate",
                    match_reason="inventory_empty",
                    preferred=True,
                    notes=target["expected_formula"],
                )
            )
        continue

    # Preferred named candidates.
    for alias, terms in target["preferred_candidate_aliases"].items():
        candidate_inventory = inventory[
            inventory["name_norm"].apply(lambda x: contains_any(x, terms))
        ].copy()

        if not candidate_inventory.empty:
            ranked = rank_inventory_for_terms(
                candidate_inventory,
                positive_terms=terms,
                exclude_terms=target["exclude_terms"],
            )
            selected_id = str(ranked.iloc[0]["CHARACTERISTIC_ID"]).strip()

            selected_rows = keyword_matches[
                keyword_matches["CHARACTERISTIC_ID"].astype(str).str.strip() == selected_id
            ].copy()

            target_summary_rows.append(
                summarize_rows(
                    rows=selected_rows,
                    target_key=target_key,
                    candidate_alias=alias,
                    target_family="preferred_candidate",
                    match_reason=f"name_contains:{'|'.join(terms)}",
                    preferred=True,
                    notes=target["expected_formula"],
                )
            )

            # Keep a few alternatives for audit.
            for _, alt in ranked.iloc[1:].head(5).iterrows():
                alt_id = str(alt["CHARACTERISTIC_ID"]).strip()
                alt_rows = keyword_matches[
                    keyword_matches["CHARACTERISTIC_ID"].astype(str).str.strip() == alt_id
                ].copy()

                target_summary_rows.append(
                    summarize_rows(
                        rows=alt_rows,
                        target_key=target_key,
                        candidate_alias=f"{alias}_alternative",
                        target_family="alternative_candidate",
                        match_reason=f"alternative_name_contains:{'|'.join(terms)}",
                        preferred=False,
                        notes="Alternative candidate retained for review.",
                    )
                )
        else:
            target_summary_rows.append(
                summarize_rows(
                    rows=pd.DataFrame(),
                    target_key=target_key,
                    candidate_alias=alias,
                    target_family="preferred_candidate",
                    match_reason=f"not_found:name_contains:{'|'.join(terms)}",
                    preferred=True,
                    notes=target["expected_formula"],
                )
            )

    # Broader secondary candidates. These help catch rows that do not match
    # the preferred labels exactly.
    secondary_inventory = inventory[
        inventory["name_norm"].apply(lambda x: contains_any(x, target["secondary_terms"]))
    ].copy()

    if not secondary_inventory.empty:
        ranked_secondary = rank_inventory_for_terms(
            secondary_inventory,
            positive_terms=target["secondary_terms"],
            exclude_terms=target["exclude_terms"],
        )

        already_ids = {
            str(row["CHARACTERISTIC_ID"]).strip()
            for row in target_summary_rows
            if row["canonical_variable"] == target_key and row["CHARACTERISTIC_ID"]
        }

        kept = 0
        for _, candidate in ranked_secondary.iterrows():
            candidate_id = str(candidate["CHARACTERISTIC_ID"]).strip()
            if candidate_id in already_ids:
                continue

            rows = keyword_matches[
                keyword_matches["CHARACTERISTIC_ID"].astype(str).str.strip() == candidate_id
            ].copy()

            target_summary_rows.append(
                summarize_rows(
                    rows=rows,
                    target_key=target_key,
                    candidate_alias=f"{target_key}_secondary_candidate",
                    target_family="secondary_candidate",
                    match_reason="secondary_keyword_match",
                    preferred=False,
                    notes="Secondary candidate retained for review because it matches the broader target concept.",
                )
            )

            kept += 1
            if kept >= 15:
                break


target_summary = pd.DataFrame(target_summary_rows)

if not target_summary.empty:
    target_summary = target_summary.sort_values(
        [
            "canonical_variable",
            "preferred",
            "target_family",
            "candidate_output_alias",
            "CHARACTERISTIC_ID",
        ],
        ascending=[True, False, True, True, True],
    )

target_summary.to_csv(OUTPUT_TARGET_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Derived formula audit
# -----------------------------

derived_rows = []

for target_key, target in TARGETS.items():
    preferred_rows = target_summary[
        (target_summary["canonical_variable"] == target_key)
        & (target_summary["preferred"] == True)
        & (target_summary["status"] == "candidate_found_full_coverage")
    ].copy()

    preferred_aliases_expected = set(target["preferred_candidate_aliases"].keys())
    preferred_aliases_found = set(preferred_rows["candidate_output_alias"])

    all_preferred_found = preferred_aliases_expected.issubset(preferred_aliases_found)

    derived_rows.append(
        {
            "canonical_variable": target_key,
            "original_code": target["original_code"],
            "candidate_formula": target["expected_formula"],
            "preferred_components_expected": ", ".join(sorted(preferred_aliases_expected)),
            "preferred_components_found": ", ".join(sorted(preferred_aliases_found)),
            "n_preferred_components_expected": len(preferred_aliases_expected),
            "n_preferred_components_found_full_coverage": len(preferred_aliases_found),
            "all_preferred_components_found_full_coverage": all_preferred_found,
            "recommended_default_without_review": False,
            "interpretation": (
                "Inspection candidate only. Review industry-versus-occupation meaning and chosen components "
                "before generating a cleaner."
            ),
        }
    )

derived_formula_audit = pd.DataFrame(derived_rows)
derived_formula_audit.to_csv(OUTPUT_DERIVED_FORMULA_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Preview selected rows
# -----------------------------

preview_ids = []

if not target_summary.empty:
    preview_ids = [
        str(x).strip()
        for x in target_summary.loc[
            (target_summary["preferred"] == True)
            & (target_summary["CHARACTERISTIC_ID"].astype(str).str.strip() != ""),
            "CHARACTERISTIC_ID",
        ].dropna().unique()
    ]

if preview_ids and not keyword_matches.empty:
    preview = keyword_matches[
        keyword_matches["CHARACTERISTIC_ID"].astype(str).str.strip().isin(preview_ids)
    ].copy()
else:
    preview = pd.DataFrame()

preview.to_csv(OUTPUT_QC_PREVIEW, index=False, encoding="utf-8")


# -----------------------------
# Summary
# -----------------------------

ready_candidate_targets = []

for target_key in TARGETS:
    target_rows = target_summary[
        (target_summary["canonical_variable"] == target_key)
        & (target_summary["preferred"] == True)
        & (target_summary["status"] == "candidate_found_full_coverage")
    ].copy()

    if not target_rows.empty:
        ready_candidate_targets.append(target_key)

review_needed_targets = [
    target_key for target_key in TARGETS
    if target_key not in ready_candidate_targets
]

summary_rows = [
    {"metric": "raw_csv", "value": safe_relative(RAW_CENSUS_PROFILE)},
    {"metric": "base_cd_frame", "value": safe_relative(BASE_CD_FRAME)},
    {"metric": "raw_encoding", "value": encoding},
    {"metric": "total_rows_scanned", "value": total_rows_scanned},
    {"metric": "quebec_cd_rows_scanned", "value": quebec_cd_rows_scanned},
    {"metric": "base_rows", "value": len(base)},
    {"metric": "keyword_match_rows", "value": len(keyword_matches)},
    {"metric": "candidate_characteristics_found", "value": len(inventory)},
    {"metric": "target_variables_inspected", "value": ", ".join(TARGETS.keys())},
    {"metric": "targets_with_at_least_one_preferred_full_coverage_candidate", "value": len(ready_candidate_targets)},
    {"metric": "targets_with_preferred_full_coverage_candidate", "value": ", ".join(ready_candidate_targets)},
    {"metric": "targets_needing_review_or_missing_preferred_candidate", "value": ", ".join(review_needed_targets)},
    {
        "metric": "important_method_note",
        "value": (
            "This inspection deliberately does not choose final variables automatically. "
            "The original SoVI employment variables may correspond to Canadian industry rows, "
            "occupation rows, or derived component sums. Review the target summary before cleaning."
        ),
    },
    {
        "metric": "recommended_next_step",
        "value": (
            "Review employment_sector_target_characteristic_summary_2021.csv and "
            "employment_sector_derived_formula_audit_2021.csv. Decide whether each target should use "
            "NAICS industry components, NOC occupation rows, or remain unresolved before generating the cleaner."
        ),
    },
]

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CENSUS DIVISION EMPLOYMENT SECTOR INSPECTION 2021")
print("=" * 72)

print("\nScan summary:")
print(summary.to_string(index=False))

print("\nTarget characteristic summary:")
if target_summary.empty:
    print("[none]")
else:
    display_cols = [
        "original_code",
        "canonical_variable",
        "candidate_output_alias",
        "target_family",
        "preferred",
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "value_column",
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

print("\nDerived formula audit:")
print(derived_formula_audit.to_string(index=False))

print("\nSaved:")
print(OUTPUT_KEYWORD_MATCHES)
print(OUTPUT_CHARACTERISTIC_INVENTORY)
print(OUTPUT_TARGET_SUMMARY)
print(OUTPUT_DERIVED_FORMULA_AUDIT)
print(OUTPUT_QC_PREVIEW)
print(OUTPUT_SUMMARY)

print("\nDone.")