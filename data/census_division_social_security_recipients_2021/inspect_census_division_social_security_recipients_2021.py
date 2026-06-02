from pathlib import Path
import re
import pandas as pd
import codecs

# ============================================================
# Inspect Census Division Social Security Recipients 2021
# ============================================================
#
# Purpose:
#   Inspect whether the 2021 Census Profile can support:
#
#       SSBENPC90 -> social_security_recipients_per_capita
#
# Original SoVI concept:
#   Per capita Social Security recipients.
#
# Canadian adaptation candidates:
#   - Government transfer recipients
#   - Old Age Security / Guaranteed Income Supplement recipients
#   - Canada Pension Plan / Quebec Pension Plan recipients
#   - Social assistance / employment insurance / other benefit recipients
#
# This script is inspection-only. It does not choose the final proxy.
#
# Run from data/:
#
#   python census_division_social_security_recipients_2021/inspect_census_division_social_security_recipients_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_social_security_recipients_2021"
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

OUTPUT_KEYWORD_ROWS = OUTPUT_DIR / "social_security_recipients_keyword_rows_2021.csv"
OUTPUT_CHARACTERISTIC_SUMMARY = OUTPUT_DIR / "social_security_recipients_characteristic_summary_2021.csv"
OUTPUT_RANKED_CANDIDATES = OUTPUT_DIR / "social_security_recipients_ranked_candidates_2021.csv"
OUTPUT_FORMULA_AUDIT = OUTPUT_DIR / "social_security_recipients_formula_audit_2021.csv"
OUTPUT_TARGET_SUMMARY = OUTPUT_DIR / "social_security_recipients_target_summary_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "social_security_recipients_inspection_summary_2021.csv"


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

KEYWORDS = [
    "social security",
    "government transfer",
    "government transfers",
    "old age security",
    "guaranteed income supplement",
    "oas",
    "gis",
    "canada pension plan",
    "quebec pension plan",
    "québec pension plan",
    "cpp",
    "qpp",
    "public pension",
    "public pensions",
    "pension",
    "retirement income",
    "recipient",
    "recipients",
    "benefit",
    "benefits",
    "social assistance",
    "employment insurance",
    "workers' compensation",
    "workers compensation",
    "child benefits",
    "covid-19 emergency",
    "covid-19 recovery",
]

PRIMARY_RECIPIENT_TERMS = [
    "number of",
    "recipient",
]

CONTEXTUAL_EXCLUSION_TERMS = [
    "composition of",
    "median",
    "average",
    "aggregate",
    "$",
    "income groups",
    "decile",
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


def numeric_stats(series: pd.Series) -> dict:
    values = clean_numeric(series)

    return {
        "non_missing": int(values.notna().sum()),
        "missing": int(values.isna().sum()),
        "min": values.min(skipna=True),
        "max": values.max(skipna=True),
        "mean": values.mean(skipna=True),
        "median": values.median(skipna=True),
    }


def classify_characteristic_name(name: object) -> dict:
    text = normalize_lower(name)

    has_number_of = "number of" in text
    has_recipient = "recipient" in text or "recipients" in text
    is_recipient_count_like = has_number_of and has_recipient

    gov_transfer = "government transfer" in text or "government transfers" in text

    oas_gis = (
        "old age security" in text
        or "guaranteed income supplement" in text
        or "oas" in text
        or "gis" in text
    )

    cpp_qpp = (
        "canada pension plan" in text
        or "quebec pension plan" in text
        or "québec pension plan" in text
        or "cpp" in text
        or "qpp" in text
    )

    social_assistance = "social assistance" in text
    employment_insurance = "employment insurance" in text
    workers_compensation = "workers' compensation" in text or "workers compensation" in text
    child_benefits = "child benefits" in text
    covid_benefits = "covid" in text and "benefit" in text
    retirement_income = "retirement income" in text
    pension = "pension" in text

    contextual_exclusion = any(term in text for term in CONTEXTUAL_EXCLUSION_TERMS)

    if is_recipient_count_like and gov_transfer:
        role = "preferred_broad_government_transfer_recipient_candidate"
        relevance_score = 100
    elif is_recipient_count_like and (oas_gis or cpp_qpp):
        role = "strong_public_pension_recipient_candidate"
        relevance_score = 90
    elif is_recipient_count_like and social_assistance:
        role = "possible_social_assistance_recipient_candidate"
        relevance_score = 85
    elif is_recipient_count_like and employment_insurance:
        role = "possible_employment_insurance_recipient_candidate"
        relevance_score = 75
    elif is_recipient_count_like and workers_compensation:
        role = "possible_workers_compensation_recipient_candidate"
        relevance_score = 70
    elif is_recipient_count_like and child_benefits:
        role = "possible_child_benefit_recipient_candidate"
        relevance_score = 65
    elif is_recipient_count_like and covid_benefits:
        role = "possible_covid_benefit_recipient_candidate"
        relevance_score = 35
    elif is_recipient_count_like and (pension or retirement_income):
        role = "possible_retirement_or_pension_recipient_candidate"
        relevance_score = 60
    elif has_recipient:
        role = "other_recipient_context_candidate"
        relevance_score = 45
    elif gov_transfer or oas_gis or cpp_qpp or social_assistance or employment_insurance:
        role = "contextual_transfer_or_pension_candidate"
        relevance_score = 30
    else:
        role = "other_keyword_match"
        relevance_score = 10

    if contextual_exclusion and not is_recipient_count_like:
        relevance_score = min(relevance_score, 20)

    return {
        "role": role,
        "relevance_score": relevance_score,
        "has_number_of": has_number_of,
        "has_recipient": has_recipient,
        "is_recipient_count_like": is_recipient_count_like,
        "gov_transfer": gov_transfer,
        "oas_gis": oas_gis,
        "cpp_qpp": cpp_qpp,
        "social_assistance": social_assistance,
        "employment_insurance": employment_insurance,
        "workers_compensation": workers_compensation,
        "child_benefits": child_benefits,
        "covid_benefits": covid_benefits,
        "retirement_income": retirement_income,
        "pension": pension,
        "contextual_exclusion": contextual_exclusion,
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
        stats = numeric_stats(group[count_col])
        row.update(
            {
                "count_column": count_col,
                "count_non_missing": stats["non_missing"],
                "count_missing": stats["missing"],
                "count_min": stats["min"],
                "count_max": stats["max"],
                "count_mean": stats["mean"],
                "count_median": stats["median"],
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
        stats = numeric_stats(group[rate_col])
        row.update(
            {
                "rate_column": rate_col,
                "rate_non_missing": stats["non_missing"],
                "rate_missing": stats["missing"],
                "rate_min": stats["min"],
                "rate_max": stats["max"],
                "rate_mean": stats["mean"],
                "rate_median": stats["median"],
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
    out = qc_rows[qc_rows[char_id_col].astype(str).str.strip() == str(char_id)][
        [dguid_col, value_col]
    ].copy()

    out[dguid_col] = out[dguid_col].astype("string").str.strip()
    out[value_col] = clean_numeric(out[value_col])
    out = out.drop_duplicates(subset=[dguid_col], keep="first")

    return out.rename(columns={dguid_col: "census_division_dguid", value_col: "value"})


def make_candidate_alias(row: pd.Series) -> str:
    char_id = str(row["CHARACTERISTIC_ID"])
    role = str(row["role"])

    if role == "preferred_broad_government_transfer_recipient_candidate":
        return "government_transfer_recipients"
    if row.get("oas_gis", False):
        return "oas_gis_recipients"
    if row.get("cpp_qpp", False):
        return "cpp_qpp_recipients"
    if row.get("social_assistance", False):
        return "social_assistance_recipients"
    if row.get("employment_insurance", False):
        return "employment_insurance_recipients"
    if row.get("workers_compensation", False):
        return "workers_compensation_recipients"
    if row.get("child_benefits", False):
        return "child_benefit_recipients"
    if row.get("covid_benefits", False):
        return "covid_benefit_recipients"

    return f"candidate_{char_id}"


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

base = pd.read_csv(BASE_CD_FRAME, dtype=str, low_memory=False)
base.columns = [str(col).strip() for col in base.columns]

required_base_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "population_total_2021",
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
base["population_total_2021"] = clean_numeric(base["population_total_2021"])

if len(base) != EXPECTED_QC_CD_COUNT:
    raise ValueError(f"Expected {EXPECTED_QC_CD_COUNT} Québec CDs, got {len(base)}.")

if base["census_division_dguid"].duplicated().any():
    raise ValueError("Duplicate census_division_dguid values in base frame.")

if base["population_total_2021"].isna().any():
    raise ValueError("Missing population_total_2021 in base frame.")

base_dguid_set = set(base["census_division_dguid"].dropna().astype(str))


# -----------------------------
# Load raw Census Profile
# -----------------------------

raw_encoding = select_encoding(RAW_PROFILE)
raw = pd.read_csv(RAW_PROFILE, encoding=raw_encoding, dtype=str, low_memory=False)
raw.columns = [str(col).strip() for col in raw.columns]

columns = list(raw.columns)

dguid_col = require_column(columns, ["DGUID", "dguid"], "DGUID")
char_id_col = require_column(columns, ["CHARACTERISTIC_ID", "Characteristic ID"], "characteristic ID")
char_name_col = require_column(columns, ["CHARACTERISTIC_NAME", "Characteristic Name"], "characteristic name")

count_col = find_column(columns, ["C1_COUNT_TOTAL", "Count total", "Total - Count"])
rate_col = find_column(columns, ["C10_RATE_TOTAL", "Rate total", "Total - Rate"])

count_symbol_col = find_column(columns, ["SYMBOL.1", "C1_SYMBOL_TOTAL", "SYMBOL"])
rate_symbol_col = find_column(columns, ["SYMBOL.3", "C10_SYMBOL_TOTAL", "SYMBOL"])

if count_col is None:
    raise ValueError(
        "Could not detect count value column.\nAvailable columns:\n"
        + "\n".join(columns)
    )

raw[dguid_col] = raw[dguid_col].astype("string").str.strip()
qc_rows = raw[raw[dguid_col].isin(base_dguid_set)].copy()

if qc_rows.empty:
    raise ValueError("No Québec CD rows found in Census Profile raw file after DGUID filtering.")


# -----------------------------
# Keyword scan
# -----------------------------

name_lower = qc_rows[char_name_col].map(normalize_lower)
keyword_mask = name_lower.apply(lambda text: any(keyword in text for keyword in KEYWORDS))

keyword_rows = qc_rows[keyword_mask].copy()
keyword_rows.to_csv(OUTPUT_KEYWORD_ROWS, index=False, encoding="utf-8")


# -----------------------------
# Characteristic summary
# -----------------------------

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
    characteristic_summary = characteristic_summary.sort_values(
        [
            "relevance_score",
            "is_recipient_count_like",
            "coverage_is_98_cds",
            "count_non_missing",
            "CHARACTERISTIC_ID",
        ],
        ascending=[False, False, False, False, True],
    )

characteristic_summary.to_csv(OUTPUT_CHARACTERISTIC_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Ranked candidate table
# -----------------------------

if characteristic_summary.empty:
    ranked_candidates = pd.DataFrame()
else:
    ranked_candidates = characteristic_summary[
        characteristic_summary["is_recipient_count_like"]
        & (characteristic_summary["count_non_missing"] == EXPECTED_QC_CD_COUNT)
    ].copy()

    if not ranked_candidates.empty:
        ranked_candidates["candidate_alias"] = ranked_candidates.apply(make_candidate_alias, axis=1)
        ranked_candidates["candidate_per_capita_formula"] = (
            "recipient_count / population_total_2021"
        )
        ranked_candidates["recommended_default_without_review"] = False
        ranked_candidates["default_review_note"] = (
            "Review conceptual fit before cleaning. Government-transfer recipients are broad; "
            "OAS/GIS and CPP/QPP are closer to old-age public pension concepts but may double-count "
            "people if summed."
        )

ranked_candidates.to_csv(OUTPUT_RANKED_CANDIDATES, index=False, encoding="utf-8")


# -----------------------------
# Formula audit
# -----------------------------

formula_rows = []

if not ranked_candidates.empty:
    for _, candidate in ranked_candidates.head(30).iterrows():
        char_id = candidate["CHARACTERISTIC_ID"]
        alias = candidate["candidate_alias"]

        count_series = extract_characteristic_series(
            qc_rows=qc_rows,
            char_id=char_id,
            char_id_col=char_id_col,
            dguid_col=dguid_col,
            value_col=count_col,
        ).rename(columns={"value": "recipient_count"})

        joined = base[
            [
                "census_division_code",
                "census_division_dguid",
                "census_division_name",
                "population_total_2021",
            ]
        ].merge(
            count_series,
            on="census_division_dguid",
            how="left",
            validate="one_to_one",
        )

        joined["candidate_per_capita"] = joined["recipient_count"] / joined["population_total_2021"]
        joined["candidate_per_1000"] = 1000 * joined["candidate_per_capita"]
        joined["candidate_per_100k"] = 100000 * joined["candidate_per_capita"]

        count_values = clean_numeric(joined["recipient_count"])
        per_capita = clean_numeric(joined["candidate_per_capita"])
        per_1000 = clean_numeric(joined["candidate_per_1000"])

        row = {
            "candidate_alias": alias,
            "candidate_formula": "recipient_count / population_total_2021",
            "candidate_formula_per_1000": "1000 * recipient_count / population_total_2021",
            "CHARACTERISTIC_ID": char_id,
            "CHARACTERISTIC_NAME": candidate["CHARACTERISTIC_NAME"],
            "role": candidate["role"],
            "relevance_score": candidate["relevance_score"],
            "source_count_column": count_col,
            "source_rate_column": rate_col or "",
            "non_missing": int(per_capita.notna().sum()),
            "missing": int(per_capita.isna().sum()),
            "coverage_is_98_cds": int(per_capita.notna().sum()) == EXPECTED_QC_CD_COUNT,
            "recipient_count_min": count_values.min(skipna=True),
            "recipient_count_max": count_values.max(skipna=True),
            "recipient_count_mean": count_values.mean(skipna=True),
            "per_capita_min": per_capita.min(skipna=True),
            "per_capita_max": per_capita.max(skipna=True),
            "per_capita_mean": per_capita.mean(skipna=True),
            "per_capita_median": per_capita.median(skipna=True),
            "per_1000_min": per_1000.min(skipna=True),
            "per_1000_max": per_1000.max(skipna=True),
            "per_1000_mean": per_1000.mean(skipna=True),
            "per_capita_values_over_1": int((per_capita > 1).sum()),
            "recommended_default_without_review": False,
            "interpretation": (
                "Candidate only. This computes recipient count divided by total population. "
                "Review whether this source row is conceptually closest to the original SoVI Social Security recipients variable."
            ),
        }

        if rate_col and candidate.get("rate_non_missing", 0) == EXPECTED_QC_CD_COUNT:
            rate_series = extract_characteristic_series(
                qc_rows=qc_rows,
                char_id=char_id,
                char_id_col=char_id_col,
                dguid_col=dguid_col,
                value_col=rate_col,
            ).rename(columns={"value": "source_rate"})

            rate_joined = base[["census_division_dguid"]].merge(
                rate_series,
                on="census_division_dguid",
                how="left",
                validate="one_to_one",
            )
            source_rate = clean_numeric(rate_joined["source_rate"])

            row.update(
                {
                    "source_rate_non_missing": int(source_rate.notna().sum()),
                    "source_rate_min": source_rate.min(skipna=True),
                    "source_rate_max": source_rate.max(skipna=True),
                    "source_rate_mean": source_rate.mean(skipna=True),
                    "source_rate_median": source_rate.median(skipna=True),
                }
            )
        else:
            row.update(
                {
                    "source_rate_non_missing": 0,
                    "source_rate_min": None,
                    "source_rate_max": None,
                    "source_rate_mean": None,
                    "source_rate_median": None,
                }
            )

        formula_rows.append(row)

formula_audit = pd.DataFrame(formula_rows)
formula_audit.to_csv(OUTPUT_FORMULA_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Target summary
# -----------------------------

target_summary_rows = []

if ranked_candidates.empty:
    target_summary_rows.append(
        {
            "canonical_variable": "social_security_recipients_per_capita",
            "original_sovi_code": "SSBENPC90",
            "candidate_type": "recipient_count_per_capita",
            "candidate_found": False,
            "n_candidates": 0,
            "best_candidate_alias": "",
            "best_candidate_characteristic_id": "",
            "best_candidate_characteristic_name": "",
            "coverage_is_98_cds": False,
            "status": "no_full_coverage_recipient_count_candidate",
            "interpretation": (
                "No full-coverage recipient-count candidate was found from Census Profile keyword inspection."
            ),
        }
    )
else:
    preferred = ranked_candidates[
        ranked_candidates["role"] == "preferred_broad_government_transfer_recipient_candidate"
    ].copy()

    public_pension = ranked_candidates[
        ranked_candidates["role"] == "strong_public_pension_recipient_candidate"
    ].copy()

    best = ranked_candidates.iloc[0]

    target_summary_rows.append(
        {
            "canonical_variable": "social_security_recipients_per_capita",
            "original_sovi_code": "SSBENPC90",
            "candidate_type": "recipient_count_per_capita",
            "candidate_found": True,
            "n_candidates": len(ranked_candidates),
            "best_candidate_alias": best["candidate_alias"],
            "best_candidate_characteristic_id": best["CHARACTERISTIC_ID"],
            "best_candidate_characteristic_name": best["CHARACTERISTIC_NAME"],
            "coverage_is_98_cds": True,
            "status": "candidate_available_needs_review",
            "interpretation": (
                "At least one full-coverage Census Profile recipient-count candidate exists. "
                "The cleaner should choose between a broad government-transfer recipient proxy and narrower "
                "public-pension proxies such as OAS/GIS or CPP/QPP."
            ),
        }
    )

    target_summary_rows.append(
        {
            "canonical_variable": "social_security_recipients_per_capita",
            "original_sovi_code": "SSBENPC90",
            "candidate_type": "broad_government_transfer_proxy",
            "candidate_found": not preferred.empty,
            "n_candidates": len(preferred),
            "best_candidate_alias": preferred.iloc[0]["candidate_alias"] if not preferred.empty else "",
            "best_candidate_characteristic_id": preferred.iloc[0]["CHARACTERISTIC_ID"] if not preferred.empty else "",
            "best_candidate_characteristic_name": preferred.iloc[0]["CHARACTERISTIC_NAME"] if not preferred.empty else "",
            "coverage_is_98_cds": bool(not preferred.empty),
            "status": "available_needs_review" if not preferred.empty else "not_found",
            "interpretation": (
                "Broad Canadian proxy: number of government-transfer recipients divided by total population. "
                "This likely captures a broader safety-net concept than U.S. Social Security alone."
            ),
        }
    )

    target_summary_rows.append(
        {
            "canonical_variable": "social_security_recipients_per_capita",
            "original_sovi_code": "SSBENPC90",
            "candidate_type": "public_pension_proxy",
            "candidate_found": not public_pension.empty,
            "n_candidates": len(public_pension),
            "best_candidate_alias": public_pension.iloc[0]["candidate_alias"] if not public_pension.empty else "",
            "best_candidate_characteristic_id": public_pension.iloc[0]["CHARACTERISTIC_ID"] if not public_pension.empty else "",
            "best_candidate_characteristic_name": public_pension.iloc[0]["CHARACTERISTIC_NAME"] if not public_pension.empty else "",
            "coverage_is_98_cds": bool(not public_pension.empty),
            "status": "available_needs_review" if not public_pension.empty else "not_found",
            "interpretation": (
                "Narrower public-pension proxy. OAS/GIS and CPP/QPP may be closer to Social Security conceptually, "
                "but summing them may double-count recipients, so a direct single row is preferable if available."
            ),
        }
    )

target_summary = pd.DataFrame(target_summary_rows)
target_summary.to_csv(OUTPUT_TARGET_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Summary
# -----------------------------

candidate_characteristics_found = len(characteristic_summary)
ranked_candidate_count = len(ranked_candidates)

preferred_count = 0
public_pension_count = 0
social_assistance_count = 0
employment_insurance_count = 0

if not ranked_candidates.empty:
    preferred_count = int(
        (ranked_candidates["role"] == "preferred_broad_government_transfer_recipient_candidate").sum()
    )
    public_pension_count = int(
        (ranked_candidates["role"] == "strong_public_pension_recipient_candidate").sum()
    )
    social_assistance_count = int(
        (ranked_candidates["role"] == "possible_social_assistance_recipient_candidate").sum()
    )
    employment_insurance_count = int(
        (ranked_candidates["role"] == "possible_employment_insurance_recipient_candidate").sum()
    )

formula_candidate_available = (
    not formula_audit.empty
    and bool((formula_audit["coverage_is_98_cds"] == True).any())
)

summary_rows = [
    {"metric": "raw_profile_csv", "value": safe_relative(RAW_PROFILE)},
    {"metric": "base_cd_frame", "value": safe_relative(BASE_CD_FRAME)},
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
    {"metric": "candidate_characteristics_found", "value": candidate_characteristics_found},
    {"metric": "ranked_full_coverage_recipient_count_candidates", "value": ranked_candidate_count},
    {"metric": "broad_government_transfer_candidates", "value": preferred_count},
    {"metric": "public_pension_candidates", "value": public_pension_count},
    {"metric": "social_assistance_candidates", "value": social_assistance_count},
    {"metric": "employment_insurance_candidates", "value": employment_insurance_count},
    {"metric": "formula_candidate_available", "value": formula_candidate_available},
    {
        "metric": "important_method_note",
        "value": (
            "SSBENPC90 is a U.S. Social Security recipients variable. Canadian Census Profile rows may be "
            "broader government-transfer recipient counts or narrower public-pension recipient counts. "
            "The inspection does not choose the final proxy automatically."
        ),
    },
    {
        "metric": "recommended_next_step",
        "value": (
            "Review social_security_recipients_ranked_candidates_2021.csv and "
            "social_security_recipients_formula_audit_2021.csv. If a defensible full-coverage recipient-count "
            "proxy is selected, generate the cleaner for social_security_recipients_per_capita."
        ),
    },
]

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CENSUS DIVISION SOCIAL SECURITY RECIPIENTS INSPECTION 2021")
print("=" * 72)

print("\nInputs:")
print("Raw profile:", safe_relative(RAW_PROFILE))
print("Base frame:", safe_relative(BASE_CD_FRAME))

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
print("Candidate characteristics found:", candidate_characteristics_found)
print("Ranked full-coverage recipient-count candidates:", ranked_candidate_count)
print("Formula candidate available:", formula_candidate_available)

print("\nTarget summary:")
print(target_summary.to_string(index=False))

print("\nTop ranked candidates:")
if ranked_candidates.empty:
    print("[none]")
else:
    display_cols = [
        "candidate_alias",
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "role",
        "relevance_score",
        "count_non_missing",
        "count_min",
        "count_max",
        "count_mean",
        "rate_non_missing",
        "rate_min",
        "rate_max",
        "rate_mean",
    ]
    display_cols = [col for col in display_cols if col in ranked_candidates.columns]
    print(ranked_candidates[display_cols].head(40).to_string(index=False))

print("\nFormula audit preview:")
if formula_audit.empty:
    print("[none]")
else:
    print(formula_audit.head(30).to_string(index=False))

print("\nSaved:")
print(OUTPUT_KEYWORD_ROWS)
print(OUTPUT_CHARACTERISTIC_SUMMARY)
print(OUTPUT_RANKED_CANDIDATES)
print(OUTPUT_FORMULA_AUDIT)
print(OUTPUT_TARGET_SUMMARY)
print(OUTPUT_SUMMARY)

print("\nDone.")