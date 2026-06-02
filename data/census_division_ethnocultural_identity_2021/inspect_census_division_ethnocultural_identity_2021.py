from pathlib import Path
import re
import pandas as pd


# ============================================================
# Inspect Census Division Ethnocultural / Indigenous Identity 2021
# ============================================================
#
# Purpose:
#   Inspect the 2021 Census Profile at census-division geography for
#   SoVI-like ethnocultural / Indigenous identity variables:
#
#       pct_black
#       pct_indigenous
#       pct_asian
#       pct_hispanic
#
# Original SoVI variables:
#
#       PCTBLACK90      -> pct_black
#       PCTINDIAN90     -> pct_indigenous
#       PCTASIAN90      -> pct_asian
#       PCTHISPANIC90   -> pct_hispanic
#
# Important correction:
#   The first version matched keywords too broadly and accidentally selected
#   language / ethnolinguistic rows such as:
#
#       Blackfoot
#       Chinese languages
#       Tagalog
#
#   This regenerated version uses explicit expected Census Profile
#   CHARACTERISTIC_ID values for the preferred visible-minority / identity rows.
#
# Run from data/:
#
#   python census_division_ethnocultural_identity_2021/inspect_census_division_ethnocultural_identity_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_ethnocultural_identity_2021"
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

OUTPUT_KEYWORD_MATCHES = OUTPUT_DIR / "ethnocultural_identity_keyword_matches_2021.csv"
OUTPUT_FORCED_ID_ROWS = OUTPUT_DIR / "ethnocultural_identity_forced_characteristic_rows_2021.csv"
OUTPUT_CHARACTERISTIC_INVENTORY = OUTPUT_DIR / "ethnocultural_identity_candidate_characteristic_inventory_2021.csv"
OUTPUT_TARGET_SUMMARY = OUTPUT_DIR / "ethnocultural_identity_target_characteristic_summary_2021.csv"
OUTPUT_DERIVED_FORMULA_AUDIT = OUTPUT_DIR / "ethnocultural_identity_derived_formula_audit_2021.csv"
OUTPUT_QC_PREVIEW = OUTPUT_DIR / "ethnocultural_identity_quebec_cd_preview_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "ethnocultural_identity_inspection_summary_2021.csv"


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


# These are the corrected preferred characteristic IDs observed in the
# Census Profile visible-minority / Indigenous identity block.
#
# pct_asian is derived from visible-minority Asian population-group components.
FORCED_TARGETS = {
    "pct_black": {
        "original_code": "PCTBLACK90",
        "description": "Percent Black population / visible minority proxy",
        "preferred_characteristic_id": "1687",
        "expected_name_contains": ["black"],
        "candidate_output_alias": "pct_black",
        "value_mode": "direct_rate",
        "value_column": VALUE_RATE_COLUMN,
        "symbol_column": SYMBOL_RATE_COLUMN,
        "unit": "percent",
        "sovi_role": "canadian_visible_minority_proxy_for_black",
        "notes": (
            "Uses the Census Profile visible-minority / population-group row 'Black'. "
            "This explicitly avoids Blackfoot language/ethnocultural rows."
        ),
    },
    "pct_hispanic": {
        "original_code": "PCTHISPANIC90",
        "description": "Percent Hispanic population / Latin American proxy",
        "preferred_characteristic_id": "1690",
        "expected_name_contains": ["latin american"],
        "candidate_output_alias": "pct_hispanic",
        "value_mode": "direct_rate",
        "value_column": VALUE_RATE_COLUMN,
        "symbol_column": SYMBOL_RATE_COLUMN,
        "unit": "percent",
        "sovi_role": "latin_american_proxy_for_hispanic",
        "notes": (
            "Canadian Census Profile does not use the U.S. Hispanic category directly. "
            "Uses Latin American visible minority / population group as the closest proxy."
        ),
    },
    "pct_indigenous": {
        "original_code": "PCTINDIAN90",
        "description": "Percent Indigenous identity population",
        "preferred_characteristic_id": "1403",
        "expected_name_contains": ["indigenous identity"],
        "candidate_output_alias": "pct_indigenous",
        "value_mode": "direct_rate",
        "value_column": VALUE_RATE_COLUMN,
        "symbol_column": SYMBOL_RATE_COLUMN,
        "unit": "percent",
        "sovi_role": "indigenous_identity_proxy",
        "notes": (
            "Uses broad Indigenous identity rate. This avoids total-denominator rows, "
            "non-Indigenous rows, and narrower First Nations / Métis / Inuit component rows."
        ),
    },
}

ASIAN_COMPONENTS = [
    {
        "component_key": "south_asian",
        "candidate_output_alias": "pct_asian_component_south_asian",
        "characteristic_id": "1685",
        "expected_name_contains": ["south asian"],
    },
    {
        "component_key": "chinese",
        "candidate_output_alias": "pct_asian_component_chinese",
        "characteristic_id": "1686",
        "expected_name_contains": ["chinese"],
    },
    {
        "component_key": "filipino",
        "candidate_output_alias": "pct_asian_component_filipino",
        "characteristic_id": "1688",
        "expected_name_contains": ["filipino"],
    },
    {
        "component_key": "southeast_asian",
        "candidate_output_alias": "pct_asian_component_southeast_asian",
        "characteristic_id": "1691",
        "expected_name_contains": ["southeast asian"],
    },
    {
        "component_key": "west_asian",
        "candidate_output_alias": "pct_asian_component_west_asian",
        "characteristic_id": "1692",
        "expected_name_contains": ["west asian"],
    },
    {
        "component_key": "korean",
        "candidate_output_alias": "pct_asian_component_korean",
        "characteristic_id": "1693",
        "expected_name_contains": ["korean"],
    },
    {
        "component_key": "japanese",
        "candidate_output_alias": "pct_asian_component_japanese",
        "characteristic_id": "1694",
        "expected_name_contains": ["japanese"],
    },
]

FORCED_IDS = (
    [x["preferred_characteristic_id"] for x in FORCED_TARGETS.values()]
    + [x["characteristic_id"] for x in ASIAN_COMPONENTS]
)

SEARCH_KEYWORDS = sorted(
    {
        "black",
        "latin american",
        "hispanic",
        "indigenous",
        "aboriginal",
        "first nations",
        "métis",
        "metis",
        "inuk",
        "inuit",
        "visible minority",
        "population group",
        "south asian",
        "chinese",
        "filipino",
        "southeast asian",
        "west asian",
        "korean",
        "japanese",
        "asian",
        "blackfoot",
        "tagalog",
        "chinese languages",
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


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        return str(path)


def name_contains_expected(name: object, expected_terms: list[str]) -> bool:
    name_norm = normalize_text(name)
    return all(term in name_norm for term in expected_terms)


def summarize_rows(
    df: pd.DataFrame,
    original_code: str,
    canonical_variable: str,
    candidate_output_alias: str,
    preferred: bool,
    value_mode: str,
    value_column: str,
    symbol_column: str,
    unit: str,
    sovi_role: str,
    notes: str,
    expected_name_contains: list[str],
) -> dict:
    if df.empty:
        return {
            "original_code": original_code,
            "canonical_variable": canonical_variable,
            "candidate_output_alias": candidate_output_alias,
            "preferred": preferred,
            "CHARACTERISTIC_ID": "",
            "CHARACTERISTIC_NAME": "",
            "value_mode": value_mode,
            "value_column": value_column,
            "symbol_column": symbol_column,
            "unit": unit,
            "sovi_role": sovi_role,
            "n_rows": 0,
            "n_unique_quebec_cds": 0,
            "value_non_missing": 0,
            "value_missing": EXPECTED_QC_CD_COUNT,
            "value_min": None,
            "value_max": None,
            "value_mean": None,
            "value_median": None,
            "coverage_is_98_cds": False,
            "name_contains_expected_text": False,
            "status": "forced_characteristic_id_not_found",
            "notes": notes,
        }

    values = clean_numeric(df[value_column])
    characteristic_ids = sorted(df["CHARACTERISTIC_ID"].astype(str).unique())
    characteristic_names = sorted(df["CHARACTERISTIC_NAME"].astype(str).unique())

    one_characteristic = len(characteristic_ids) == 1
    characteristic_id = characteristic_ids[0] if one_characteristic else "; ".join(characteristic_ids)
    characteristic_name = characteristic_names[0] if len(characteristic_names) == 1 else "; ".join(characteristic_names)

    coverage_is_98 = int(df["DGUID"].nunique()) == EXPECTED_QC_CD_COUNT
    values_complete = int(values.notna().sum()) == EXPECTED_QC_CD_COUNT
    name_ok = name_contains_expected(characteristic_name, expected_name_contains)

    if coverage_is_98 and values_complete and name_ok:
        status = "ready_for_cleaner"
    elif coverage_is_98 and values_complete and not name_ok:
        status = "candidate_needs_name_review"
    else:
        status = "candidate_needs_coverage_review"

    return {
        "original_code": original_code,
        "canonical_variable": canonical_variable,
        "candidate_output_alias": candidate_output_alias,
        "preferred": preferred,
        "CHARACTERISTIC_ID": characteristic_id,
        "CHARACTERISTIC_NAME": characteristic_name,
        "value_mode": value_mode,
        "value_column": value_column,
        "symbol_column": symbol_column,
        "unit": unit,
        "sovi_role": sovi_role,
        "n_rows": len(df),
        "n_unique_quebec_cds": int(df["DGUID"].nunique()),
        "value_non_missing": int(values.notna().sum()),
        "value_missing": int(values.isna().sum()),
        "value_min": values.min(skipna=True),
        "value_max": values.max(skipna=True),
        "value_mean": values.mean(skipna=True),
        "value_median": values.median(skipna=True),
        "coverage_is_98_cds": coverage_is_98,
        "name_contains_expected_text": name_ok,
        "status": status,
        "notes": notes,
    }


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

print("\nInspecting Census Division Ethnocultural / Indigenous Identity 2021")
print("Raw Census Profile:", safe_relative(RAW_CENSUS_PROFILE))
print("Base CD frame:", safe_relative(BASE_CD_FRAME))
print("Base rows:", len(base))


# -----------------------------
# Scan raw Census Profile
# -----------------------------

encoding = select_encoding(RAW_CENSUS_PROFILE)
print("Raw encoding selected:", encoding)

keyword_match_chunks = []
forced_id_chunks = []

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
    qc_cd["CHARACTERISTIC_ID_STR"] = qc_cd["CHARACTERISTIC_ID"].astype(str).str.strip()

    forced_rows = qc_cd[qc_cd["CHARACTERISTIC_ID_STR"].isin(FORCED_IDS)].copy()
    if not forced_rows.empty:
        forced_id_chunks.append(forced_rows)

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
    keyword_matches = pd.DataFrame(columns=RAW_COLUMNS_NEEDED + ["CHARACTERISTIC_NAME_NORM", "CHARACTERISTIC_ID_STR"])

if forced_id_chunks:
    forced_rows_all = pd.concat(forced_id_chunks, ignore_index=True)
else:
    forced_rows_all = pd.DataFrame(columns=RAW_COLUMNS_NEEDED + ["CHARACTERISTIC_NAME_NORM", "CHARACTERISTIC_ID_STR"])

keyword_matches.to_csv(OUTPUT_KEYWORD_MATCHES, index=False, encoding="utf-8")
forced_rows_all.to_csv(OUTPUT_FORCED_ID_ROWS, index=False, encoding="utf-8")

print("\nKeyword match rows:", len(keyword_matches))
print("Forced ID rows:", len(forced_rows_all))


# -----------------------------
# Characteristic inventory for keyword matches
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
                "CHARACTERISTIC_ID": characteristic_id,
                "CHARACTERISTIC_NAME": characteristic_name,
                "CHARACTERISTIC_NOTE": characteristic_note,
                "n_rows": len(group),
                "n_unique_quebec_cds": int(group["DGUID"].nunique()),
                "rate_non_missing": int(values_rate.notna().sum()),
                "rate_missing": int(values_rate.isna().sum()),
                "rate_min": values_rate.min(skipna=True),
                "rate_max": values_rate.max(skipna=True),
                "rate_mean": values_rate.mean(skipna=True),
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
# Target-level forced-ID inspection
# -----------------------------

target_summary_rows = []

for canonical_variable, target in FORCED_TARGETS.items():
    selected_rows = forced_rows_all[
        forced_rows_all["CHARACTERISTIC_ID_STR"] == str(target["preferred_characteristic_id"])
    ].copy()

    target_summary_rows.append(
        summarize_rows(
            df=selected_rows,
            original_code=target["original_code"],
            canonical_variable=canonical_variable,
            candidate_output_alias=target["candidate_output_alias"],
            preferred=True,
            value_mode=target["value_mode"],
            value_column=target["value_column"],
            symbol_column=target["symbol_column"],
            unit=target["unit"],
            sovi_role=target["sovi_role"],
            notes=target["notes"],
            expected_name_contains=target["expected_name_contains"],
        )
    )

# Asian components.
for component in ASIAN_COMPONENTS:
    selected_rows = forced_rows_all[
        forced_rows_all["CHARACTERISTIC_ID_STR"] == str(component["characteristic_id"])
    ].copy()

    target_summary_rows.append(
        summarize_rows(
            df=selected_rows,
            original_code="PCTASIAN90",
            canonical_variable="pct_asian",
            candidate_output_alias=component["candidate_output_alias"],
            preferred=True,
            value_mode="derived_component_sum_component",
            value_column=VALUE_RATE_COLUMN,
            symbol_column=SYMBOL_RATE_COLUMN,
            unit="percent",
            sovi_role="derived_asian_visible_minority_proxy",
            notes=f"Forced visible-minority Asian component for derived pct_asian: {component['component_key']}.",
            expected_name_contains=component["expected_name_contains"],
        )
    )

target_summary = pd.DataFrame(target_summary_rows)

if not target_summary.empty:
    target_summary = target_summary.sort_values(
        ["canonical_variable", "candidate_output_alias", "CHARACTERISTIC_ID"],
        ascending=[True, True, True],
    )

target_summary.to_csv(OUTPUT_TARGET_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Derived formula audit
# -----------------------------

asian_component_rows = target_summary[
    (target_summary["canonical_variable"] == "pct_asian")
    & (target_summary["value_mode"] == "derived_component_sum_component")
].copy()

asian_components_ready = (
    (asian_component_rows["status"] == "ready_for_cleaner")
    & (asian_component_rows["coverage_is_98_cds"] == True)
    & (asian_component_rows["value_non_missing"] == EXPECTED_QC_CD_COUNT)
)

asian_formula_available = (
    len(asian_component_rows) == len(ASIAN_COMPONENTS)
    and bool(asian_components_ready.all())
)

derived_formula_audit = pd.DataFrame(
    [
        {
            "canonical_variable": "pct_asian",
            "candidate_formula": (
                "sum of rate columns for South Asian + Chinese + Filipino + "
                "Southeast Asian + West Asian + Korean + Japanese"
            ),
            "available": asian_formula_available,
            "n_components_expected": len(ASIAN_COMPONENTS),
            "n_components_found": len(asian_component_rows),
            "components_found": "; ".join(
                f"{row['candidate_output_alias']}|{row['CHARACTERISTIC_ID']}|{row['CHARACTERISTIC_NAME']}"
                for _, row in asian_component_rows.iterrows()
            ),
            "recommended_default_without_review": asian_formula_available,
            "interpretation": (
                "Derived Canadian visible-minority Asian-group proxy using forced visible-minority "
                "component IDs. This avoids language rows such as Chinese languages and Tagalog."
            ),
        }
    ]
)

derived_formula_audit.to_csv(OUTPUT_DERIVED_FORMULA_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Preview selected rows
# -----------------------------

preview_ids = [
    str(x)
    for x in target_summary["CHARACTERISTIC_ID"].dropna().unique()
    if str(x).strip()
]

if preview_ids and not forced_rows_all.empty:
    preview = forced_rows_all[
        forced_rows_all["CHARACTERISTIC_ID_STR"].isin(preview_ids)
    ].copy()
else:
    preview = pd.DataFrame()

preview.to_csv(OUTPUT_QC_PREVIEW, index=False, encoding="utf-8")


# -----------------------------
# Summary
# -----------------------------

ready_targets = []
unresolved_or_review_needed = []

for target_key in ["pct_black", "pct_hispanic", "pct_indigenous"]:
    rows = target_summary[
        (target_summary["canonical_variable"] == target_key)
        & (target_summary["status"] == "ready_for_cleaner")
    ]
    if not rows.empty:
        ready_targets.append(target_key)
    else:
        unresolved_or_review_needed.append(target_key)

if asian_formula_available:
    ready_targets.append("pct_asian")
else:
    unresolved_or_review_needed.append("pct_asian")

summary_rows = [
    {"metric": "raw_csv", "value": safe_relative(RAW_CENSUS_PROFILE)},
    {"metric": "base_cd_frame", "value": safe_relative(BASE_CD_FRAME)},
    {"metric": "raw_encoding", "value": encoding},
    {"metric": "total_rows_scanned", "value": total_rows_scanned},
    {"metric": "quebec_cd_rows_scanned", "value": quebec_cd_rows_scanned},
    {"metric": "base_rows", "value": len(base)},
    {"metric": "keyword_match_rows", "value": len(keyword_matches)},
    {"metric": "forced_id_rows", "value": len(forced_rows_all)},
    {"metric": "candidate_characteristics_found_keyword_inventory", "value": len(inventory)},
    {"metric": "target_variables_inspected", "value": "pct_black, pct_hispanic, pct_indigenous, pct_asian"},
    {"metric": "forced_ids_used", "value": ", ".join(FORCED_IDS)},
    {"metric": "target_variables_ready_or_formula_available", "value": len(ready_targets)},
    {"metric": "ready_targets", "value": ", ".join(ready_targets)},
    {"metric": "asian_components_ready", "value": int(asian_components_ready.sum())},
    {"metric": "asian_formula_available", "value": asian_formula_available},
    {"metric": "unresolved_or_review_needed_targets", "value": ", ".join(unresolved_or_review_needed)},
    {
        "metric": "important_method_note",
        "value": (
            "This inspection uses forced visible-minority / Indigenous identity characteristic IDs "
            "to avoid false keyword matches such as Blackfoot, Chinese languages, and Tagalog."
        ),
    },
    {
        "metric": "recommended_next_step",
        "value": (
            "Review ethnocultural_identity_target_characteristic_summary_2021.csv. "
            "If pct_black uses 1687 Black, pct_hispanic uses 1690 Latin American, "
            "pct_indigenous uses 1403 Indigenous identity, and pct_asian uses 1685/1686/1688/1691/1692/1693/1694, "
            "generate the cleaner."
        ),
    },
]

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CENSUS DIVISION ETHNOCULTURAL / INDIGENOUS IDENTITY INSPECTION 2021")
print("=" * 72)

print("\nScan summary:")
print(summary.to_string(index=False))

print("\nTarget characteristic summary:")
display_cols = [
    "original_code",
    "canonical_variable",
    "candidate_output_alias",
    "preferred",
    "CHARACTERISTIC_ID",
    "CHARACTERISTIC_NAME",
    "value_mode",
    "value_column",
    "n_unique_quebec_cds",
    "value_non_missing",
    "value_missing",
    "value_min",
    "value_max",
    "value_mean",
    "coverage_is_98_cds",
    "name_contains_expected_text",
    "status",
]
display_cols = [col for col in display_cols if col in target_summary.columns]
print(target_summary[display_cols].to_string(index=False))

print("\nDerived formula audit:")
print(derived_formula_audit.to_string(index=False))

print("\nSaved:")
print(OUTPUT_KEYWORD_MATCHES)
print(OUTPUT_FORCED_ID_ROWS)
print(OUTPUT_CHARACTERISTIC_INVENTORY)
print(OUTPUT_TARGET_SUMMARY)
print(OUTPUT_DERIVED_FORMULA_AUDIT)
print(OUTPUT_QC_PREVIEW)
print(OUTPUT_SUMMARY)

print("\nDone.")