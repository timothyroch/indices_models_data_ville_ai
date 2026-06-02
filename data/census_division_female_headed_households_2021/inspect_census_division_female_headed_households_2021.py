from pathlib import Path
import codecs
import re
import pandas as pd


# ============================================================
# Inspect Census Division Female-Headed Households 2021
# ============================================================
#
# Purpose:
#   Inspect whether the 2021 Census Profile and/or the existing
#   household-family cleaned block can support:
#
#       PCTF_HH90 -> pct_female_headed_households
#
# Original SoVI concept:
#   Percent female-headed households, no spouse present.
#
# Canadian adaptation candidates:
#   - Female lone-parent families / census families
#   - Female lone-parent families / private households
#   - Direct Census Profile rate row, if available
#
# This is inspection-only. It does not clean the final variable.
#
# Run from data/:
#
#   python census_division_female_headed_households_2021/inspect_census_division_female_headed_households_2021.py
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

EXISTING_HOUSEHOLD_FAMILY_CLEAN = (
    DATA_DIR
    / "census_division_household_family_2021"
    / "output"
    / "clean_census_division_household_family_2021.csv"
)

OUTPUT_EXISTING_COLUMN_PROFILE = OUTPUT_DIR / "female_headed_households_existing_household_family_column_profile_2021.csv"
OUTPUT_EXISTING_CANDIDATE_COLUMNS = OUTPUT_DIR / "female_headed_households_existing_candidate_columns_2021.csv"
OUTPUT_KEYWORD_ROWS = OUTPUT_DIR / "female_headed_households_keyword_rows_2021.csv"
OUTPUT_CHARACTERISTIC_SUMMARY = OUTPUT_DIR / "female_headed_households_characteristic_summary_2021.csv"
OUTPUT_TARGET_SUMMARY = OUTPUT_DIR / "female_headed_households_target_summary_2021.csv"
OUTPUT_FORMULA_AUDIT = OUTPUT_DIR / "female_headed_households_formula_audit_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "female_headed_households_inspection_summary_2021.csv"


# -----------------------------
# Config
# -----------------------------

EXPECTED_QC_CD_COUNT = 98

ENCODING_CANDIDATES = [
    # StatCan CSVs are often UTF-8, UTF-8 with BOM, or Windows-1252.
    # latin1 is last because it can decode almost any byte sequence and
    # therefore should only be used as a fallback.
    "utf-8-sig",
    "utf-8",
    "cp1252",
    "latin1",
]

CSV_READ_KWARGS = {
    "dtype": str,
    "low_memory": False,
}

KEYWORDS = [
    "female",
    "woman",
    "women",
    "lone-parent",
    "lone parent",
    "one-parent",
    "one parent",
    "single-parent",
    "single parent",
    "no spouse",
    "without spouse",
    "spouse not present",
    "household maintainer",
    "census family",
    "private household",
    "household type",
    "family type",
]

EXISTING_COLUMN_KEYWORDS = [
    "female",
    "woman",
    "women",
    "lone",
    "single",
    "parent",
    "spouse",
    "household",
    "family",
]


# -----------------------------
# Helpers
# -----------------------------

def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        return str(path)


def can_decode_entire_file(path: Path, encoding: str, chunk_size: int = 1024 * 1024) -> bool:
    """Return True only if the whole file can be decoded with the encoding.

    Important: checking only the first few rows is unsafe for large Census Profile
    files. A file can look UTF-8 at the top but contain Windows-1252 accented
    characters later. This is exactly the failure mode that produced:

        UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9 ...

    The incremental decoder avoids loading the entire raw byte file into memory.
    """
    decoder = codecs.getincrementaldecoder(encoding)()

    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                decoder.decode(chunk)
            decoder.decode(b"", final=True)
        return True
    except UnicodeDecodeError:
        return False


def select_encoding(path: Path) -> str:
    """Select the first candidate encoding that decodes the full file."""
    for encoding in ENCODING_CANDIDATES:
        if can_decode_entire_file(path, encoding):
            return encoding

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not decode {path} with candidates {ENCODING_CANDIDATES}",
    )


def read_csv_robust(path: Path, source_label: str) -> tuple[pd.DataFrame, str]:
    """Read a CSV with full-file encoding detection and useful diagnostics.

    Returns:
        (dataframe, encoding_used)

    This intentionally does not use encoding_errors='ignore' or 'replace', because
    silently corrupting characteristic names would make keyword-based inspection
    unreliable.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {source_label}:\n{path}")

    if path.stat().st_size == 0:
        raise ValueError(f"{source_label} exists but is empty:\n{path}")

    encoding = select_encoding(path)

    try:
        df = pd.read_csv(path, encoding=encoding, **CSV_READ_KWARGS)
    except UnicodeDecodeError as exc:
        raise UnicodeDecodeError(
            exc.encoding,
            exc.object,
            exc.start,
            exc.end,
            f"Failed while reading {source_label} with selected encoding {encoding}: {exc.reason}",
        ) from exc
    except pd.errors.ParserError as exc:
        raise pd.errors.ParserError(
            f"Failed to parse {source_label} as CSV with encoding {encoding}. "
            f"Path: {path}\nOriginal parser error: {exc}"
        ) from exc

    df.columns = [str(col).strip() for col in df.columns]
    return df, encoding


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


def profile_columns(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    rows = []

    for col in df.columns:
        series = df[col]
        numeric = clean_numeric(series)

        rows.append(
            {
                "source_label": source_label,
                "column": col,
                "dtype_as_loaded": str(series.dtype),
                "non_missing": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "unique_values": int(series.astype("string").nunique(dropna=True)),
                "sample_values": " | ".join(series.dropna().astype(str).head(10).tolist()),
                "numeric_non_missing": int(numeric.notna().sum()),
                "numeric_min": numeric.min(skipna=True),
                "numeric_max": numeric.max(skipna=True),
                "numeric_mean": numeric.mean(skipna=True),
            }
        )

    return pd.DataFrame(rows)


def classify_characteristic_name(name: object) -> dict:
    text = normalize_lower(name)

    # Use word boundaries so "male" is not accidentally detected inside "female".
    female_word = bool(re.search(r"\b(female|woman|women)\b", text))
    male_word = bool(re.search(r"\b(male|man|men)\b", text))

    lone_parent = any(
        term in text
        for term in [
            "lone-parent",
            "lone parent",
            "one-parent",
            "one parent",
            "single-parent",
            "single parent",
        ]
    )

    no_spouse = any(
        term in text
        for term in [
            "no spouse",
            "without spouse",
            "spouse not present",
            "without a spouse",
        ]
    )

    household = "household" in text
    family = "family" in text or "families" in text
    maintainer = "maintainer" in text
    census_family = "census family" in text or "census families" in text
    private_household = "private household" in text or "private households" in text

    total_like = (
        text.startswith("total")
        or "total -" in text
        or text.strip() in ["total", "total households", "total families"]
    )

    target_like = female_word and (lone_parent or no_spouse or maintainer)

    denominator_census_family = total_like and census_family
    denominator_private_household = total_like and private_household
    denominator_lone_parent_total = lone_parent and not female_word and not male_word and not no_spouse

    if target_like and lone_parent:
        role = "preferred_target_female_lone_parent_candidate"
    elif target_like and no_spouse:
        role = "possible_target_female_no_spouse_candidate"
    elif target_like and maintainer:
        role = "possible_target_female_maintainer_candidate"
    elif denominator_census_family:
        role = "possible_denominator_total_census_families"
    elif denominator_private_household:
        role = "possible_denominator_total_private_households"
    elif denominator_lone_parent_total:
        role = "possible_denominator_or_context_total_lone_parent"
    elif female_word and family:
        role = "female_family_context_candidate"
    elif female_word and household:
        role = "female_household_context_candidate"
    else:
        role = "other_keyword_match"

    return {
        "role": role,
        "female_word": female_word,
        "male_word": male_word,
        "lone_parent": lone_parent,
        "no_spouse": no_spouse,
        "household": household,
        "family": family,
        "maintainer": maintainer,
        "census_family": census_family,
        "private_household": private_household,
        "total_like": total_like,
        "target_like": target_like,
    }


def summarize_characteristic(
    group: pd.DataFrame,
    char_id_col: str,
    char_name_col: str,
    dguid_col: str,
    count_col: str | None,
    rate_col: str | None,
    count_symbol_col: str | None,
    rate_symbol_col: str | None,
) -> dict:
    char_id = normalize_text(group[char_id_col].iloc[0])
    char_name = normalize_text(group[char_name_col].iloc[0])

    role_info = classify_characteristic_name(char_name)

    row = {
        "CHARACTERISTIC_ID": char_id,
        "CHARACTERISTIC_NAME": char_name,
        **role_info,
        "n_rows": len(group),
        "n_unique_quebec_cds": int(group[dguid_col].astype(str).nunique()),
        "coverage_is_98_cds": int(group[dguid_col].astype(str).nunique()) == EXPECTED_QC_CD_COUNT,
    }

    if count_col and count_col in group.columns:
        values = clean_numeric(group[count_col])
        row.update(
            {
                "count_column": count_col,
                "count_non_missing": int(values.notna().sum()),
                "count_missing": int(values.isna().sum()),
                "count_min": values.min(skipna=True),
                "count_max": values.max(skipna=True),
                "count_mean": values.mean(skipna=True),
                "count_median": values.median(skipna=True),
            }
        )
    else:
        row.update(
            {
                "count_column": "",
                "count_non_missing": 0,
                "count_missing": len(group),
                "count_min": None,
                "count_max": None,
                "count_mean": None,
                "count_median": None,
            }
        )

    if rate_col and rate_col in group.columns:
        values = clean_numeric(group[rate_col])
        row.update(
            {
                "rate_column": rate_col,
                "rate_non_missing": int(values.notna().sum()),
                "rate_missing": int(values.isna().sum()),
                "rate_min": values.min(skipna=True),
                "rate_max": values.max(skipna=True),
                "rate_mean": values.mean(skipna=True),
                "rate_median": values.median(skipna=True),
            }
        )
    else:
        row.update(
            {
                "rate_column": "",
                "rate_non_missing": 0,
                "rate_missing": len(group),
                "rate_min": None,
                "rate_max": None,
                "rate_mean": None,
                "rate_median": None,
            }
        )

    if count_symbol_col and count_symbol_col in group.columns:
        row["count_symbols"] = " | ".join(sorted(group[count_symbol_col].dropna().astype(str).unique())[:20])
    else:
        row["count_symbols"] = ""

    if rate_symbol_col and rate_symbol_col in group.columns:
        row["rate_symbols"] = " | ".join(sorted(group[rate_symbol_col].dropna().astype(str).unique())[:20])
    else:
        row["rate_symbols"] = ""

    return row


def extract_characteristic_series(
    qc_rows: pd.DataFrame,
    char_id: object,
    char_id_col: str,
    dguid_col: str,
    value_col: str,
) -> pd.DataFrame:
    out = qc_rows[qc_rows[char_id_col].astype(str) == str(char_id)][
        [dguid_col, value_col]
    ].copy()

    out[dguid_col] = out[dguid_col].astype("string").str.strip()
    out[value_col] = clean_numeric(out[value_col])
    out = out.drop_duplicates(subset=[dguid_col], keep="first")

    return out.rename(columns={dguid_col: "census_division_dguid", value_col: "value"})


def numeric_stats_for_series(series: pd.Series) -> dict:
    values = clean_numeric(series)
    return {
        "non_missing": int(values.notna().sum()),
        "missing": int(values.isna().sum()),
        "min": values.min(skipna=True),
        "max": values.max(skipna=True),
        "mean": values.mean(skipna=True),
        "median": values.median(skipna=True),
    }


# -----------------------------
# Validate inputs
# -----------------------------

if not RAW_PROFILE.exists():
    raise FileNotFoundError(f"Missing Census Profile raw CSV:\n{RAW_PROFILE}")

if not BASE_CD_FRAME.exists():
    raise FileNotFoundError(f"Missing base CD frame:\n{BASE_CD_FRAME}")


# -----------------------------
# Load base
# -----------------------------

base, base_encoding = read_csv_robust(BASE_CD_FRAME, "base CD frame")

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
base_dguid_set = set(base["census_division_dguid"].dropna().astype(str))

if len(base) != EXPECTED_QC_CD_COUNT:
    raise ValueError(f"Expected {EXPECTED_QC_CD_COUNT} Québec CDs, got {len(base)}.")


# -----------------------------
# Inspect existing household/family clean output
# -----------------------------

existing_source_found = EXISTING_HOUSEHOLD_FAMILY_CLEAN.exists()

if existing_source_found:
    existing, existing_encoding = read_csv_robust(EXISTING_HOUSEHOLD_FAMILY_CLEAN, "existing household/family clean output")

    existing_profile = profile_columns(existing, "existing_household_family_clean")
    existing_profile.to_csv(OUTPUT_EXISTING_COLUMN_PROFILE, index=False, encoding="utf-8")

    candidate_existing_cols = [
        col for col in existing.columns
        if any(term in col.lower() for term in EXISTING_COLUMN_KEYWORDS)
    ]

    if candidate_existing_cols:
        existing_candidates = existing[
            [col for col in ["census_division_dguid", "census_division_name"] if col in existing.columns]
            + candidate_existing_cols
        ].copy()
    else:
        existing_candidates = pd.DataFrame()

    existing_candidates.to_csv(OUTPUT_EXISTING_CANDIDATE_COLUMNS, index=False, encoding="utf-8")
else:
    existing_encoding = ""
    existing_profile = pd.DataFrame()
    existing_profile.to_csv(OUTPUT_EXISTING_COLUMN_PROFILE, index=False, encoding="utf-8")
    existing_candidates = pd.DataFrame()
    existing_candidates.to_csv(OUTPUT_EXISTING_CANDIDATE_COLUMNS, index=False, encoding="utf-8")


# -----------------------------
# Load raw Census Profile
# -----------------------------

raw, raw_encoding = read_csv_robust(RAW_PROFILE, "raw Census Profile CSV")

columns = list(raw.columns)

dguid_col = require_column(columns, ["DGUID", "dguid"], "DGUID")
char_id_col = require_column(columns, ["CHARACTERISTIC_ID", "Characteristic ID"], "characteristic ID")
char_name_col = require_column(columns, ["CHARACTERISTIC_NAME", "Characteristic Name"], "characteristic name")

count_col = find_column(columns, ["C1_COUNT_TOTAL", "Count total", "Total - Count"])
rate_col = find_column(columns, ["C10_RATE_TOTAL", "Rate total", "Total - Rate"])

count_symbol_col = find_column(columns, ["SYMBOL.1", "C1_SYMBOL_TOTAL", "SYMBOL"])
rate_symbol_col = find_column(columns, ["SYMBOL.3", "C10_SYMBOL_TOTAL", "SYMBOL"])

if count_col is None and rate_col is None:
    raise ValueError(
        "Could not detect either count or rate value columns.\nAvailable columns:\n"
        + "\n".join(columns)
    )

raw[dguid_col] = raw[dguid_col].astype("string").str.strip()
qc_rows = raw[raw[dguid_col].isin(base_dguid_set)].copy()

if qc_rows.empty:
    raise ValueError("No Québec CD rows found in Census Profile raw file after DGUID filtering.")


# -----------------------------
# Keyword rows and characteristic summary
# -----------------------------

name_lower = qc_rows[char_name_col].map(normalize_lower)
keyword_mask = name_lower.apply(lambda text: any(keyword in text for keyword in KEYWORDS))

keyword_rows = qc_rows[keyword_mask].copy()
keyword_rows.to_csv(OUTPUT_KEYWORD_ROWS, index=False, encoding="utf-8")

summary_rows = []

if not keyword_rows.empty:
    grouped = keyword_rows.groupby([char_id_col, char_name_col], dropna=False)

    for _, group in grouped:
        summary_rows.append(
            summarize_characteristic(
                group=group,
                char_id_col=char_id_col,
                char_name_col=char_name_col,
                dguid_col=dguid_col,
                count_col=count_col,
                rate_col=rate_col,
                count_symbol_col=count_symbol_col,
                rate_symbol_col=rate_symbol_col,
            )
        )

characteristic_summary = pd.DataFrame(summary_rows)

if not characteristic_summary.empty:
    role_order = {
        "preferred_target_female_lone_parent_candidate": 0,
        "possible_target_female_no_spouse_candidate": 1,
        "possible_target_female_maintainer_candidate": 2,
        "possible_denominator_total_census_families": 3,
        "possible_denominator_total_private_households": 4,
        "possible_denominator_or_context_total_lone_parent": 5,
        "female_family_context_candidate": 6,
        "female_household_context_candidate": 7,
        "other_keyword_match": 8,
    }
    characteristic_summary["_role_order"] = characteristic_summary["role"].map(role_order).fillna(99)
    characteristic_summary = characteristic_summary.sort_values(
        ["_role_order", "CHARACTERISTIC_ID"]
    ).drop(columns=["_role_order"])

characteristic_summary.to_csv(OUTPUT_CHARACTERISTIC_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Target summary
# -----------------------------

target_summary_rows = []

if characteristic_summary.empty:
    target_summary_rows.append(
        {
            "canonical_variable": "pct_female_headed_households",
            "original_sovi_code": "PCTF_HH90",
            "candidate_type": "none",
            "candidate_found": False,
            "n_candidates": 0,
            "best_candidate_characteristic_id": "",
            "best_candidate_characteristic_name": "",
            "best_candidate_value_mode": "",
            "coverage_is_98_cds": False,
            "status": "no_candidate_found",
            "interpretation": "No candidate Census Profile characteristic was found by keyword inspection.",
        }
    )
else:
    direct_rate_candidates = characteristic_summary[
        characteristic_summary["role"].isin(
            [
                "preferred_target_female_lone_parent_candidate",
                "possible_target_female_no_spouse_candidate",
                "possible_target_female_maintainer_candidate",
            ]
        )
        & (characteristic_summary["rate_non_missing"] == EXPECTED_QC_CD_COUNT)
    ].copy()

    count_candidates = characteristic_summary[
        characteristic_summary["role"].isin(
            [
                "preferred_target_female_lone_parent_candidate",
                "possible_target_female_no_spouse_candidate",
                "possible_target_female_maintainer_candidate",
            ]
        )
        & (characteristic_summary["count_non_missing"] == EXPECTED_QC_CD_COUNT)
    ].copy()

    denominator_candidates = characteristic_summary[
        characteristic_summary["role"].isin(
            [
                "possible_denominator_total_census_families",
                "possible_denominator_total_private_households",
            ]
        )
        & (characteristic_summary["count_non_missing"] == EXPECTED_QC_CD_COUNT)
    ].copy()

    if not direct_rate_candidates.empty:
        best = direct_rate_candidates.iloc[0]
        target_summary_rows.append(
            {
                "canonical_variable": "pct_female_headed_households",
                "original_sovi_code": "PCTF_HH90",
                "candidate_type": "direct_rate_candidate",
                "candidate_found": True,
                "n_candidates": len(direct_rate_candidates),
                "best_candidate_characteristic_id": best["CHARACTERISTIC_ID"],
                "best_candidate_characteristic_name": best["CHARACTERISTIC_NAME"],
                "best_candidate_value_mode": "rate",
                "coverage_is_98_cds": True,
                "status": "direct_rate_candidate_available_needs_review",
                "interpretation": (
                    "A direct Census Profile rate candidate exists. Review the characteristic "
                    "and denominator before cleaning because the Census rate denominator may be "
                    "census families rather than all private households."
                ),
            }
        )
    else:
        target_summary_rows.append(
            {
                "canonical_variable": "pct_female_headed_households",
                "original_sovi_code": "PCTF_HH90",
                "candidate_type": "direct_rate_candidate",
                "candidate_found": False,
                "n_candidates": 0,
                "best_candidate_characteristic_id": "",
                "best_candidate_characteristic_name": "",
                "best_candidate_value_mode": "rate",
                "coverage_is_98_cds": False,
                "status": "no_direct_rate_candidate",
                "interpretation": "No full-coverage direct rate candidate was found.",
            }
        )

    target_summary_rows.append(
        {
            "canonical_variable": "pct_female_headed_households",
            "original_sovi_code": "PCTF_HH90",
            "candidate_type": "count_formula_candidate",
            "candidate_found": bool(not count_candidates.empty and not denominator_candidates.empty),
            "n_candidates": len(count_candidates),
            "best_candidate_characteristic_id": count_candidates.iloc[0]["CHARACTERISTIC_ID"] if not count_candidates.empty else "",
            "best_candidate_characteristic_name": count_candidates.iloc[0]["CHARACTERISTIC_NAME"] if not count_candidates.empty else "",
            "best_candidate_value_mode": "count / denominator count",
            "coverage_is_98_cds": bool(not count_candidates.empty and not denominator_candidates.empty),
            "status": (
                "formula_candidate_available_needs_review"
                if not count_candidates.empty and not denominator_candidates.empty
                else "formula_candidate_not_available"
            ),
            "interpretation": (
                "A derived formula is possible if a female lone-parent/no-spouse count and a defensible "
                "denominator count are both available. Candidate denominators should be reviewed."
            ),
        }
    )

target_summary = pd.DataFrame(target_summary_rows)
target_summary.to_csv(OUTPUT_TARGET_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Formula audit
# -----------------------------

formula_rows = []

if not characteristic_summary.empty and count_col is not None:
    count_candidates = characteristic_summary[
        characteristic_summary["role"].isin(
            [
                "preferred_target_female_lone_parent_candidate",
                "possible_target_female_no_spouse_candidate",
                "possible_target_female_maintainer_candidate",
            ]
        )
        & (characteristic_summary["count_non_missing"] == EXPECTED_QC_CD_COUNT)
    ].head(10)

    denominator_candidates = characteristic_summary[
        characteristic_summary["role"].isin(
            [
                "possible_denominator_total_census_families",
                "possible_denominator_total_private_households",
            ]
        )
        & (characteristic_summary["count_non_missing"] == EXPECTED_QC_CD_COUNT)
    ].head(10)

    for _, numerator in count_candidates.iterrows():
        numerator_series = extract_characteristic_series(
            qc_rows=qc_rows,
            char_id=numerator["CHARACTERISTIC_ID"],
            char_id_col=char_id_col,
            dguid_col=dguid_col,
            value_col=count_col,
        )

        for _, denominator in denominator_candidates.iterrows():
            denominator_series = extract_characteristic_series(
                qc_rows=qc_rows,
                char_id=denominator["CHARACTERISTIC_ID"],
                char_id_col=char_id_col,
                dguid_col=dguid_col,
                value_col=count_col,
            )

            joined = base[["census_division_dguid"]].merge(
                numerator_series.rename(columns={"value": "numerator_count"}),
                on="census_division_dguid",
                how="left",
            ).merge(
                denominator_series.rename(columns={"value": "denominator_count"}),
                on="census_division_dguid",
                how="left",
            )

            joined["candidate_pct"] = (
                100 * joined["numerator_count"] / joined["denominator_count"]
            )

            values = clean_numeric(joined["candidate_pct"])

            formula_rows.append(
                {
                    "candidate_formula": "100 * numerator_count / denominator_count",
                    "numerator_characteristic_id": numerator["CHARACTERISTIC_ID"],
                    "numerator_characteristic_name": numerator["CHARACTERISTIC_NAME"],
                    "denominator_characteristic_id": denominator["CHARACTERISTIC_ID"],
                    "denominator_characteristic_name": denominator["CHARACTERISTIC_NAME"],
                    "non_missing": int(values.notna().sum()),
                    "missing": int(values.isna().sum()),
                    "min": values.min(skipna=True),
                    "max": values.max(skipna=True),
                    "mean": values.mean(skipna=True),
                    "median": values.median(skipna=True),
                    "values_over_100": int((values > 100).sum()),
                    "coverage_is_98_cds": int(values.notna().sum()) == EXPECTED_QC_CD_COUNT,
                    "recommended_default_without_review": False,
                    "interpretation": (
                        "Formula candidate only. Review whether denominator should be all census families, "
                        "all private households, or a narrower household/family universe."
                    ),
                }
            )

formula_audit = pd.DataFrame(formula_rows)
formula_audit.to_csv(OUTPUT_FORMULA_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Summary
# -----------------------------

direct_rate_available = False
formula_available = False

if not target_summary.empty:
    direct_rate_available = bool(
        (
            (target_summary["candidate_type"] == "direct_rate_candidate")
            & (target_summary["candidate_found"] == True)
        ).any()
    )
    formula_available = bool(
        (
            (target_summary["candidate_type"] == "count_formula_candidate")
            & (target_summary["candidate_found"] == True)
        ).any()
    )

preferred_target_count = 0
denominator_count = 0

if not characteristic_summary.empty:
    preferred_target_count = int(
        characteristic_summary["role"].isin(
            [
                "preferred_target_female_lone_parent_candidate",
                "possible_target_female_no_spouse_candidate",
                "possible_target_female_maintainer_candidate",
            ]
        ).sum()
    )
    denominator_count = int(
        characteristic_summary["role"].isin(
            [
                "possible_denominator_total_census_families",
                "possible_denominator_total_private_households",
            ]
        ).sum()
    )

summary_rows = [
    {"metric": "raw_profile_csv", "value": safe_relative(RAW_PROFILE)},
    {"metric": "base_cd_frame", "value": safe_relative(BASE_CD_FRAME)},
    {"metric": "existing_household_family_clean", "value": safe_relative(EXISTING_HOUSEHOLD_FAMILY_CLEAN)},
    {"metric": "existing_household_family_clean_found", "value": existing_source_found},
    {"metric": "base_encoding", "value": base_encoding},
    {"metric": "existing_household_family_encoding", "value": existing_encoding},
    {"metric": "raw_encoding", "value": raw_encoding},
    {"metric": "raw_rows", "value": len(raw)},
    {"metric": "quebec_cd_rows_scanned", "value": len(qc_rows)},
    {"metric": "base_rows", "value": len(base)},
    {"metric": "detected_dguid_column", "value": dguid_col},
    {"metric": "detected_characteristic_id_column", "value": char_id_col},
    {"metric": "detected_characteristic_name_column", "value": char_name_col},
    {"metric": "detected_count_column", "value": count_col or ""},
    {"metric": "detected_rate_column", "value": rate_col or ""},
    {"metric": "keyword_match_rows", "value": len(keyword_rows)},
    {"metric": "candidate_characteristics_found", "value": len(characteristic_summary)},
    {"metric": "target_like_candidate_characteristics", "value": preferred_target_count},
    {"metric": "denominator_candidate_characteristics", "value": denominator_count},
    {"metric": "direct_rate_candidate_available", "value": direct_rate_available},
    {"metric": "formula_candidate_available", "value": formula_available},
    {
        "metric": "recommended_next_step",
        "value": (
            "Review female_headed_households_characteristic_summary_2021.csv and "
            "female_headed_households_formula_audit_2021.csv. If a female lone-parent/no-spouse candidate "
            "and a defensible denominator have 98/98 coverage, generate the cleaner."
        ),
    },
]

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CENSUS DIVISION FEMALE-HEADED HOUSEHOLDS INSPECTION 2021")
print("=" * 72)

print("\nInputs:")
print("Raw profile:", safe_relative(RAW_PROFILE))
print("Base frame:", safe_relative(BASE_CD_FRAME))
print("Existing household/family clean found:", existing_source_found)
print("Base frame encoding:", base_encoding)
print("Existing household/family encoding:", existing_encoding or "[not loaded]")
print("Raw profile encoding:", raw_encoding)

print("\nDetected columns:")
print("DGUID:", dguid_col)
print("Characteristic ID:", char_id_col)
print("Characteristic name:", char_name_col)
print("Count:", count_col)
print("Rate:", rate_col)

print("\nInspection counts:")
print("Raw rows:", len(raw))
print("Québec CD rows scanned:", len(qc_rows))
print("Keyword match rows:", len(keyword_rows))
print("Candidate characteristics found:", len(characteristic_summary))
print("Target-like candidate characteristics:", preferred_target_count)
print("Denominator candidate characteristics:", denominator_count)
print("Direct rate candidate available:", direct_rate_available)
print("Formula candidate available:", formula_available)

print("\nTarget summary:")
print(target_summary.to_string(index=False))

print("\nTop characteristic candidates:")
if characteristic_summary.empty:
    print("[none]")
else:
    display_cols = [
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "role",
        "n_unique_quebec_cds",
        "count_non_missing",
        "rate_non_missing",
        "count_min",
        "count_max",
        "rate_min",
        "rate_max",
    ]
    display_cols = [col for col in display_cols if col in characteristic_summary.columns]
    print(characteristic_summary[display_cols].head(60).to_string(index=False))

print("\nFormula audit preview:")
if formula_audit.empty:
    print("[none]")
else:
    print(formula_audit.head(40).to_string(index=False))

print("\nSaved:")
print(OUTPUT_EXISTING_COLUMN_PROFILE)
print(OUTPUT_EXISTING_CANDIDATE_COLUMNS)
print(OUTPUT_KEYWORD_ROWS)
print(OUTPUT_CHARACTERISTIC_SUMMARY)
print(OUTPUT_TARGET_SUMMARY)
print(OUTPUT_FORMULA_AUDIT)
print(OUTPUT_SUMMARY)

print("\nDone.")