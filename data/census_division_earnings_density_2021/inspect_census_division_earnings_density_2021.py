from pathlib import Path
import codecs
import re
import pandas as pd


# ============================================================
# Inspect Census Division Earnings Density 2021
# Targeted Income-Section Diagnostic
# ============================================================
#
# Purpose:
#   Diagnose whether the 2021 Census Profile can support:
#
#       EARNDEN90 -> earnings_density
#
# Original SoVI concept:
#   Earnings, in $1,000, in all industries per square mile.
#
# This version is more targeted than the first inspection. It:
#
#   1. Exports the Census Profile income block around characteristic IDs 100-260.
#   2. Searches for direct aggregate income rows.
#   3. Searches for derivable aggregate-income proxies using:
#
#          number of employment income recipients
#          *
#          average employment income among recipients
#
#   4. Computes candidate income-density formulas when possible:
#
#          estimated_aggregate_income / land_area_km2
#          estimated_aggregate_income / land_area_square_miles
#
# Important encoding note:
#   The raw Census Profile is read with strict full-file encoding detection.
#
# Run from data/:
#
#   python census_division_earnings_density_2021/inspect_census_division_earnings_density_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_earnings_density_2021"
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

OUTPUT_INCOME_SECTION_ROWS = OUTPUT_DIR / "earnings_density_income_section_rows_100_260_2021.csv"
OUTPUT_INCOME_SECTION_INVENTORY = OUTPUT_DIR / "earnings_density_income_section_characteristic_inventory_100_260_2021.csv"
OUTPUT_KEYWORD_ROWS = OUTPUT_DIR / "earnings_density_keyword_rows_2021.csv"
OUTPUT_CHARACTERISTIC_SUMMARY = OUTPUT_DIR / "earnings_density_characteristic_summary_2021.csv"
OUTPUT_RANKED_CANDIDATES = OUTPUT_DIR / "earnings_density_ranked_candidates_2021.csv"
OUTPUT_FORMULA_AUDIT = OUTPUT_DIR / "earnings_density_formula_audit_2021.csv"
OUTPUT_TARGET_SUMMARY = OUTPUT_DIR / "earnings_density_target_summary_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "earnings_density_inspection_summary_2021.csv"


# -----------------------------
# Config
# -----------------------------

EXPECTED_QC_CD_COUNT = 98

INCOME_SECTION_ID_MIN = 100
INCOME_SECTION_ID_MAX = 260

ENCODING_CANDIDATES = [
    "utf-8-sig",
    "utf-8",
    "cp1252",
    "latin1",
]

KM2_TO_SQUARE_MILES = 0.3861021585424458

KEYWORDS = [
    "income",
    "employment income",
    "average employment income",
    "market income",
    "total income",
    "after-tax income",
    "wages",
    "salaries",
    "commissions",
    "self-employment",
    "aggregate",
    "earnings",
    "recipient",
    "recipients",
]


# -----------------------------
# Helpers
# -----------------------------

def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        return str(path)


def detect_file_encoding_strict(path: Path, encodings: list[str]) -> str:
    for encoding in encodings:
        decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
        try:
            with path.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    decoder.decode(chunk)
                decoder.decode(b"", final=True)
            return encoding
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not strictly decode {path} with candidates {encodings}",
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


def parse_year(text: str) -> str:
    match = re.search(r"\b(2019|2020|2021)\b", text)
    return match.group(1) if match else ""


def parse_sample_class(text: str) -> str:
    if "100% data" in text:
        return "100pct"
    if "25% sample" in text or "25% sample data" in text:
        return "25pct"
    return "unknown"


def classify_income_domain(text: str) -> str:
    if "employment insurance" in text:
        return "employment_insurance_benefits"
    if "covid" in text:
        return "covid_benefits"
    if "government transfer" in text or "government transfers" in text:
        return "government_transfers"
    if "employment income" in text:
        return "employment_income"
    if "wages, salaries and commissions" in text or "wages salaries and commissions" in text:
        return "wages_salaries_commissions"
    if "self-employment" in text or "self employment" in text:
        return "self_employment_income"
    if "market income" in text:
        return "market_income"
    if "after-tax income" in text or "after tax income" in text:
        return "after_tax_income"
    if "total income" in text:
        return "total_income"
    return "other_income"


def classify_characteristic_name(name: object) -> dict:
    text = normalize_lower(name)

    domain = classify_income_domain(text)
    year = parse_year(text)
    sample_class = parse_sample_class(text)

    has_number_of = "number of" in text
    has_recipient = "recipient" in text or "recipients" in text
    has_average = "average" in text
    has_median = "median" in text
    has_aggregate = "aggregate" in text
    has_dollar_hint = "$" in text or "($)" in text
    has_percentage = "percentage" in text or "percent" in text

    is_number_recipient_row = has_number_of and has_recipient
    is_average_income_row = has_average and "income" in text and not has_number_of
    is_direct_aggregate_amount_row = (
        has_aggregate
        and not has_average
        and not has_median
        and not has_number_of
        and not has_percentage
    )

    if is_direct_aggregate_amount_row and domain == "employment_income":
        role = "direct_aggregate_employment_income_candidate"
        relevance_score = 100
    elif is_direct_aggregate_amount_row and domain == "wages_salaries_commissions":
        role = "direct_aggregate_wages_salaries_candidate"
        relevance_score = 90
    elif is_number_recipient_row and domain == "employment_income":
        role = "employment_income_recipient_count_component"
        relevance_score = 85
    elif is_average_income_row and domain == "employment_income":
        role = "average_employment_income_component"
        relevance_score = 84
    elif is_number_recipient_row and domain in ["market_income", "total_income", "after_tax_income"]:
        role = f"{domain}_recipient_count_context_component"
        relevance_score = 55
    elif is_average_income_row and domain in ["market_income", "total_income", "after_tax_income"]:
        role = f"average_{domain}_context_component"
        relevance_score = 54
    elif is_direct_aggregate_amount_row:
        role = "other_direct_aggregate_income_candidate"
        relevance_score = 45
    elif "income" in text:
        role = "income_section_context"
        relevance_score = 20
    else:
        role = "other_context"
        relevance_score = 0

    return {
        "role": role,
        "relevance_score": relevance_score,
        "income_domain": domain,
        "year": year,
        "sample_class": sample_class,
        "has_number_of": has_number_of,
        "has_recipient": has_recipient,
        "has_average": has_average,
        "has_median": has_median,
        "has_aggregate": has_aggregate,
        "has_dollar_hint": has_dollar_hint,
        "has_percentage": has_percentage,
        "is_number_recipient_row": is_number_recipient_row,
        "is_average_income_row": is_average_income_row,
        "is_direct_aggregate_amount_row": is_direct_aggregate_amount_row,
    }


def summarize_characteristic(
    group: pd.DataFrame,
    char_id_col: str,
    char_name_col: str,
    dguid_col: str,
    value_col: str,
    rate_col: str | None,
    value_symbol_col: str | None,
    rate_symbol_col: str | None,
) -> dict:
    char_id = normalize_text(group[char_id_col].iloc[0])
    char_name = normalize_text(group[char_name_col].iloc[0])
    class_info = classify_characteristic_name(char_name)

    value_stats = numeric_stats(group[value_col])

    row = {
        "CHARACTERISTIC_ID": char_id,
        "CHARACTERISTIC_ID_NUMERIC": pd.to_numeric(char_id, errors="coerce"),
        "CHARACTERISTIC_NAME": char_name,
        **class_info,
        "n_rows": len(group),
        "n_unique_quebec_cds": int(group[dguid_col].astype(str).nunique()),
        "coverage_is_98_cds": int(group[dguid_col].astype(str).nunique()) == EXPECTED_QC_CD_COUNT,
        "value_column": value_col,
        "value_non_missing": value_stats["non_missing"],
        "value_missing": value_stats["missing"],
        "value_min": value_stats["min"],
        "value_max": value_stats["max"],
        "value_mean": value_stats["mean"],
        "value_median": value_stats["median"],
    }

    if rate_col and rate_col in group.columns:
        rate_stats = numeric_stats(group[rate_col])
        row.update(
            {
                "rate_column": rate_col,
                "rate_non_missing": rate_stats["non_missing"],
                "rate_missing": rate_stats["missing"],
                "rate_min": rate_stats["min"],
                "rate_max": rate_stats["max"],
                "rate_mean": rate_stats["mean"],
                "rate_median": rate_stats["median"],
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

    if value_symbol_col and value_symbol_col in group.columns:
        row["value_symbols"] = " | ".join(sorted(group[value_symbol_col].dropna().astype(str).unique())[:20])
    else:
        row["value_symbols"] = ""

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


def candidate_alias_from_pair(domain: str, year: str, sample_class: str) -> str:
    year_part = year if year else "unknown_year"
    sample_part = sample_class if sample_class else "unknown_sample"
    return f"derived_aggregate_{domain}_{year_part}_{sample_part}"


def score_pair(domain: str, year: str, sample_class: str) -> int:
    score = 0

    if domain == "employment_income":
        score += 100
    elif domain == "wages_salaries_commissions":
        score += 90
    elif domain == "market_income":
        score += 55
    elif domain == "total_income":
        score += 45
    else:
        score += 20

    if year == "2020":
        score += 10
    elif year == "2019":
        score += 5

    if sample_class == "100pct":
        score += 10
    elif sample_class == "25pct":
        score += 5

    return score


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

base_encoding = detect_file_encoding_strict(BASE_CD_FRAME, ENCODING_CANDIDATES)

base = pd.read_csv(BASE_CD_FRAME, encoding=base_encoding, dtype=str, low_memory=False)
base.columns = [str(col).strip() for col in base.columns]

required_base_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "population_total_2021",
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
base["census_division_dguid"] = base["census_division_dguid"].astype("string").str.strip()
base["population_total_2021"] = clean_numeric(base["population_total_2021"])
base["land_area_km2"] = clean_numeric(base["land_area_km2"])

if len(base) != EXPECTED_QC_CD_COUNT:
    raise ValueError(f"Expected {EXPECTED_QC_CD_COUNT} Québec CDs, got {len(base)}.")

if base["census_division_dguid"].duplicated().any():
    raise ValueError("Duplicate census_division_dguid values in base frame.")

if base["land_area_km2"].isna().any() or (base["land_area_km2"] <= 0).any():
    raise ValueError("Missing or non-positive land_area_km2 values in base frame.")

base_dguid_set = set(base["census_division_dguid"].dropna().astype(str))


# -----------------------------
# Load raw Census Profile
# -----------------------------

raw_encoding = detect_file_encoding_strict(RAW_PROFILE, ENCODING_CANDIDATES)

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
value_col = require_column(columns, ["C1_COUNT_TOTAL", "Count total", "Total - Count"], "main value/count")
rate_col = find_column(columns, ["C10_RATE_TOTAL", "Rate total", "Total - Rate"])
value_symbol_col = find_column(columns, ["SYMBOL.1", "C1_SYMBOL_TOTAL", "SYMBOL"])
rate_symbol_col = find_column(columns, ["SYMBOL.3", "C10_SYMBOL_TOTAL", "SYMBOL"])

raw[dguid_col] = raw[dguid_col].astype("string").str.strip()
raw["_characteristic_id_numeric"] = pd.to_numeric(raw[char_id_col], errors="coerce")

qc_rows = raw[raw[dguid_col].isin(base_dguid_set)].copy()

if qc_rows.empty:
    raise ValueError("No Québec CD rows found in Census Profile raw file after DGUID filtering.")


# -----------------------------
# Income section 100-260
# -----------------------------

income_section_rows = qc_rows[
    (qc_rows["_characteristic_id_numeric"] >= INCOME_SECTION_ID_MIN)
    & (qc_rows["_characteristic_id_numeric"] <= INCOME_SECTION_ID_MAX)
].copy()

income_section_rows.to_csv(OUTPUT_INCOME_SECTION_ROWS, index=False, encoding="utf-8")

income_inventory_rows = []

if not income_section_rows.empty:
    grouped = income_section_rows.groupby([char_id_col, char_name_col], dropna=False)

    for _, group in grouped:
        income_inventory_rows.append(
            summarize_characteristic(
                group=group,
                char_id_col=char_id_col,
                char_name_col=char_name_col,
                dguid_col=dguid_col,
                value_col=value_col,
                rate_col=rate_col,
                value_symbol_col=value_symbol_col,
                rate_symbol_col=rate_symbol_col,
            )
        )

income_inventory = pd.DataFrame(income_inventory_rows)

if not income_inventory.empty:
    income_inventory = income_inventory.sort_values(
        ["CHARACTERISTIC_ID_NUMERIC", "CHARACTERISTIC_NAME"]
    )

income_inventory.to_csv(OUTPUT_INCOME_SECTION_INVENTORY, index=False, encoding="utf-8")


# -----------------------------
# Keyword rows and summary
# -----------------------------

name_lower = qc_rows[char_name_col].map(normalize_lower)
keyword_mask = name_lower.apply(lambda text: any(keyword in text for keyword in KEYWORDS))

keyword_rows = qc_rows[keyword_mask].copy()
keyword_rows.to_csv(OUTPUT_KEYWORD_ROWS, index=False, encoding="utf-8")

keyword_summary_rows = []

if not keyword_rows.empty:
    grouped = keyword_rows.groupby([char_id_col, char_name_col], dropna=False)

    for _, group in grouped:
        keyword_summary_rows.append(
            summarize_characteristic(
                group=group,
                char_id_col=char_id_col,
                char_name_col=char_name_col,
                dguid_col=dguid_col,
                value_col=value_col,
                rate_col=rate_col,
                value_symbol_col=value_symbol_col,
                rate_symbol_col=rate_symbol_col,
            )
        )

keyword_summary = pd.DataFrame(keyword_summary_rows)

if not keyword_summary.empty:
    keyword_summary = keyword_summary.sort_values(
        [
            "relevance_score",
            "CHARACTERISTIC_ID_NUMERIC",
        ],
        ascending=[False, True],
    )

keyword_summary.to_csv(OUTPUT_CHARACTERISTIC_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Candidate discovery
# -----------------------------

candidate_rows = []

# Direct aggregate amount candidates, if they exist.
direct_aggregate_candidates = income_inventory[
    (income_inventory["is_direct_aggregate_amount_row"] == True)
    & (income_inventory["coverage_is_98_cds"] == True)
    & (income_inventory["value_non_missing"] == EXPECTED_QC_CD_COUNT)
].copy()

for _, row in direct_aggregate_candidates.iterrows():
    candidate_rows.append(
        {
            "candidate_type": "direct_aggregate_amount",
            "candidate_alias": f"direct_aggregate_{row['income_domain']}_{row['year'] or 'unknown_year'}_{row['sample_class'] or 'unknown_sample'}",
            "candidate_score": int(row["relevance_score"]),
            "income_domain": row["income_domain"],
            "year": row["year"],
            "sample_class": row["sample_class"],
            "value_characteristic_id": row["CHARACTERISTIC_ID"],
            "value_characteristic_name": row["CHARACTERISTIC_NAME"],
            "number_characteristic_id": "",
            "number_characteristic_name": "",
            "average_characteristic_id": "",
            "average_characteristic_name": "",
            "candidate_formula": "aggregate_income_value / land_area",
            "coverage_is_98_cds": True,
            "recommended_default_without_review": False,
            "review_note": "Direct aggregate amount candidate. Confirm units before cleaning.",
        }
    )

# Derived candidates from count * average.
number_components = income_inventory[
    (income_inventory["is_number_recipient_row"] == True)
    & (income_inventory["coverage_is_98_cds"] == True)
    & (income_inventory["value_non_missing"] == EXPECTED_QC_CD_COUNT)
].copy()

average_components = income_inventory[
    (income_inventory["is_average_income_row"] == True)
    & (income_inventory["coverage_is_98_cds"] == True)
    & (income_inventory["value_non_missing"] == EXPECTED_QC_CD_COUNT)
].copy()

for _, nrow in number_components.iterrows():
    n_id = int(nrow["CHARACTERISTIC_ID_NUMERIC"])
    n_domain = nrow["income_domain"]
    n_year = nrow["year"]
    n_sample = nrow["sample_class"]

    possible_averages = average_components[
        (average_components["income_domain"] == n_domain)
        & (
            (average_components["year"] == n_year)
            | (average_components["year"] == "")
            | (n_year == "")
        )
    ].copy()

    for _, arow in possible_averages.iterrows():
        a_id = int(arow["CHARACTERISTIC_ID_NUMERIC"])
        a_sample = arow["sample_class"]

        # Prefer adjacent / near-adjacent Census Profile pairs, e.g. number row 118
        # followed by average row 119.
        if abs(a_id - n_id) > 3:
            continue

        if n_sample != "unknown" and a_sample != "unknown" and n_sample != a_sample:
            continue

        sample_class = n_sample if n_sample != "unknown" else a_sample
        year = n_year if n_year else arow["year"]

        candidate_rows.append(
            {
                "candidate_type": "derived_count_times_average",
                "candidate_alias": candidate_alias_from_pair(n_domain, year, sample_class),
                "candidate_score": score_pair(n_domain, year, sample_class),
                "income_domain": n_domain,
                "year": year,
                "sample_class": sample_class,
                "value_characteristic_id": "",
                "value_characteristic_name": "",
                "number_characteristic_id": nrow["CHARACTERISTIC_ID"],
                "number_characteristic_name": nrow["CHARACTERISTIC_NAME"],
                "average_characteristic_id": arow["CHARACTERISTIC_ID"],
                "average_characteristic_name": arow["CHARACTERISTIC_NAME"],
                "candidate_formula": "number_of_recipients * average_income / land_area",
                "coverage_is_98_cds": True,
                "recommended_default_without_review": False,
                "review_note": (
                    "Derived aggregate-income candidate. This reconstructs aggregate income "
                    "from recipient count times average income. Review before cleaning."
                ),
            }
        )

ranked_candidates = pd.DataFrame(candidate_rows)

if not ranked_candidates.empty:
    ranked_candidates = ranked_candidates.sort_values(
        ["candidate_score", "candidate_type", "candidate_alias"],
        ascending=[False, True, True],
    )

ranked_candidates.to_csv(OUTPUT_RANKED_CANDIDATES, index=False, encoding="utf-8")


# -----------------------------
# Formula audit
# -----------------------------

formula_rows = []

if not ranked_candidates.empty:
    for _, candidate in ranked_candidates.head(50).iterrows():
        candidate_type = candidate["candidate_type"]

        joined = base[
            [
                "census_division_code",
                "census_division_dguid",
                "census_division_name",
                "population_total_2021",
                "land_area_km2",
            ]
        ].copy()

        if candidate_type == "direct_aggregate_amount":
            value_series = extract_characteristic_series(
                qc_rows=qc_rows,
                char_id=candidate["value_characteristic_id"],
                char_id_col=char_id_col,
                dguid_col=dguid_col,
                value_col=value_col,
            ).rename(columns={"value": "aggregate_income_value"})

            joined = joined.merge(
                value_series,
                on="census_division_dguid",
                how="left",
                validate="one_to_one",
            )

            joined["component_number_value"] = pd.NA
            joined["component_average_value"] = pd.NA

        elif candidate_type == "derived_count_times_average":
            number_series = extract_characteristic_series(
                qc_rows=qc_rows,
                char_id=candidate["number_characteristic_id"],
                char_id_col=char_id_col,
                dguid_col=dguid_col,
                value_col=value_col,
            ).rename(columns={"value": "component_number_value"})

            average_series = extract_characteristic_series(
                qc_rows=qc_rows,
                char_id=candidate["average_characteristic_id"],
                char_id_col=char_id_col,
                dguid_col=dguid_col,
                value_col=value_col,
            ).rename(columns={"value": "component_average_value"})

            joined = joined.merge(
                number_series,
                on="census_division_dguid",
                how="left",
                validate="one_to_one",
            ).merge(
                average_series,
                on="census_division_dguid",
                how="left",
                validate="one_to_one",
            )

            joined["aggregate_income_value"] = (
                joined["component_number_value"] * joined["component_average_value"]
            )

        else:
            continue

        joined["land_area_square_miles"] = joined["land_area_km2"] * KM2_TO_SQUARE_MILES

        joined["earnings_density_dollars_per_km2"] = (
            joined["aggregate_income_value"] / joined["land_area_km2"]
        )

        joined["earnings_density_thousands_per_km2"] = (
            joined["aggregate_income_value"] / 1000 / joined["land_area_km2"]
        )

        joined["earnings_density_thousands_per_square_mile"] = (
            joined["aggregate_income_value"] / 1000 / joined["land_area_square_miles"]
        )

        joined["aggregate_income_per_capita"] = (
            joined["aggregate_income_value"] / joined["population_total_2021"]
        )

        aggregate_values = clean_numeric(joined["aggregate_income_value"])
        density_km2 = clean_numeric(joined["earnings_density_dollars_per_km2"])
        density_thousands_km2 = clean_numeric(joined["earnings_density_thousands_per_km2"])
        density_thousands_mile2 = clean_numeric(joined["earnings_density_thousands_per_square_mile"])
        per_capita = clean_numeric(joined["aggregate_income_per_capita"])

        formula_rows.append(
            {
                "candidate_alias": candidate["candidate_alias"],
                "candidate_type": candidate_type,
                "candidate_score": candidate["candidate_score"],
                "income_domain": candidate["income_domain"],
                "year": candidate["year"],
                "sample_class": candidate["sample_class"],
                "candidate_formula": candidate["candidate_formula"],
                "value_characteristic_id": candidate["value_characteristic_id"],
                "value_characteristic_name": candidate["value_characteristic_name"],
                "number_characteristic_id": candidate["number_characteristic_id"],
                "number_characteristic_name": candidate["number_characteristic_name"],
                "average_characteristic_id": candidate["average_characteristic_id"],
                "average_characteristic_name": candidate["average_characteristic_name"],
                "non_missing": int(density_km2.notna().sum()),
                "missing": int(density_km2.isna().sum()),
                "coverage_is_98_cds": int(density_km2.notna().sum()) == EXPECTED_QC_CD_COUNT,
                "aggregate_income_min": aggregate_values.min(skipna=True),
                "aggregate_income_max": aggregate_values.max(skipna=True),
                "aggregate_income_mean": aggregate_values.mean(skipna=True),
                "aggregate_income_median": aggregate_values.median(skipna=True),
                "density_dollars_per_km2_min": density_km2.min(skipna=True),
                "density_dollars_per_km2_max": density_km2.max(skipna=True),
                "density_dollars_per_km2_mean": density_km2.mean(skipna=True),
                "density_dollars_per_km2_median": density_km2.median(skipna=True),
                "density_thousands_per_km2_min": density_thousands_km2.min(skipna=True),
                "density_thousands_per_km2_max": density_thousands_km2.max(skipna=True),
                "density_thousands_per_km2_mean": density_thousands_km2.mean(skipna=True),
                "density_thousands_per_square_mile_min": density_thousands_mile2.min(skipna=True),
                "density_thousands_per_square_mile_max": density_thousands_mile2.max(skipna=True),
                "density_thousands_per_square_mile_mean": density_thousands_mile2.mean(skipna=True),
                "aggregate_income_per_capita_min": per_capita.min(skipna=True),
                "aggregate_income_per_capita_max": per_capita.max(skipna=True),
                "aggregate_income_per_capita_mean": per_capita.mean(skipna=True),
                "negative_aggregate_values": int((aggregate_values < 0).sum()),
                "recommended_default_without_review": False,
                "interpretation": (
                    "Candidate only. EARNDEN90 is originally in $1,000 per square mile. "
                    "The original-style candidate is density_thousands_per_square_mile."
                ),
            }
        )

formula_audit = pd.DataFrame(formula_rows)
formula_audit.to_csv(OUTPUT_FORMULA_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Target summary
# -----------------------------

target_summary_rows = []

direct_candidate_count = 0
derived_candidate_count = 0
employment_derived_candidate_count = 0

if not ranked_candidates.empty:
    direct_candidate_count = int((ranked_candidates["candidate_type"] == "direct_aggregate_amount").sum())
    derived_candidate_count = int((ranked_candidates["candidate_type"] == "derived_count_times_average").sum())
    employment_derived_candidate_count = int(
        (
            (ranked_candidates["candidate_type"] == "derived_count_times_average")
            & (ranked_candidates["income_domain"] == "employment_income")
        ).sum()
    )

if formula_audit.empty:
    target_summary_rows.append(
        {
            "canonical_variable": "earnings_density",
            "original_sovi_code": "EARNDEN90",
            "candidate_type": "none",
            "candidate_found": False,
            "n_candidates": 0,
            "best_candidate_alias": "",
            "best_candidate_description": "",
            "coverage_is_98_cds": False,
            "status": "no_candidate_found",
            "interpretation": (
                "No direct aggregate-income row or derived count-times-average income candidate was found."
            ),
        }
    )
else:
    best = formula_audit.sort_values("candidate_score", ascending=False).iloc[0]

    target_summary_rows.append(
        {
            "canonical_variable": "earnings_density",
            "original_sovi_code": "EARNDEN90",
            "candidate_type": "best_available_candidate",
            "candidate_found": True,
            "n_candidates": len(formula_audit),
            "best_candidate_alias": best["candidate_alias"],
            "best_candidate_description": best["candidate_formula"],
            "coverage_is_98_cds": bool(best["coverage_is_98_cds"]),
            "status": "candidate_available_needs_review",
            "interpretation": (
                "At least one full-coverage earnings-density candidate exists. "
                "Review whether it should be accepted as a derived proxy before cleaning."
            ),
        }
    )

    target_summary_rows.append(
        {
            "canonical_variable": "earnings_density",
            "original_sovi_code": "EARNDEN90",
            "candidate_type": "direct_aggregate_amount",
            "candidate_found": direct_candidate_count > 0,
            "n_candidates": direct_candidate_count,
            "best_candidate_alias": (
                ranked_candidates[ranked_candidates["candidate_type"] == "direct_aggregate_amount"].iloc[0]["candidate_alias"]
                if direct_candidate_count > 0 else ""
            ),
            "best_candidate_description": "direct aggregate income / land area" if direct_candidate_count > 0 else "",
            "coverage_is_98_cds": direct_candidate_count > 0,
            "status": "available_needs_review" if direct_candidate_count > 0 else "not_found",
            "interpretation": "Direct aggregate amount row, if available, is conceptually strongest.",
        }
    )

    target_summary_rows.append(
        {
            "canonical_variable": "earnings_density",
            "original_sovi_code": "EARNDEN90",
            "candidate_type": "derived_employment_income_density",
            "candidate_found": employment_derived_candidate_count > 0,
            "n_candidates": employment_derived_candidate_count,
            "best_candidate_alias": (
                ranked_candidates[
                    (ranked_candidates["candidate_type"] == "derived_count_times_average")
                    & (ranked_candidates["income_domain"] == "employment_income")
                ].iloc[0]["candidate_alias"]
                if employment_derived_candidate_count > 0 else ""
            ),
            "best_candidate_description": (
                "employment income recipients * average employment income / land area"
                if employment_derived_candidate_count > 0 else ""
            ),
            "coverage_is_98_cds": employment_derived_candidate_count > 0,
            "status": "available_needs_review" if employment_derived_candidate_count > 0 else "not_found",
            "interpretation": (
                "Derived employment-income density is a plausible Canadian proxy if no direct aggregate row exists."
            ),
        }
    )

target_summary = pd.DataFrame(target_summary_rows)
target_summary.to_csv(OUTPUT_TARGET_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Summary
# -----------------------------

formula_candidate_available = (
    not formula_audit.empty
    and bool((formula_audit["coverage_is_98_cds"] == True).any())
)

income_section_characteristics = len(income_inventory)
keyword_characteristics = len(keyword_summary)
ranked_candidate_count = len(ranked_candidates)

base_names_with_mojibake = contains_mojibake(base["census_division_name"])
raw_characteristic_names_with_mojibake = contains_mojibake(raw[char_name_col])

summary_rows = [
    {"metric": "raw_profile_csv", "value": safe_relative(RAW_PROFILE)},
    {"metric": "base_cd_frame", "value": safe_relative(BASE_CD_FRAME)},
    {"metric": "raw_encoding", "value": raw_encoding},
    {"metric": "base_encoding", "value": base_encoding},
    {"metric": "raw_rows", "value": len(raw)},
    {"metric": "quebec_cd_rows_scanned", "value": len(qc_rows)},
    {"metric": "base_rows", "value": len(base)},
    {"metric": "income_section_id_min", "value": INCOME_SECTION_ID_MIN},
    {"metric": "income_section_id_max", "value": INCOME_SECTION_ID_MAX},
    {"metric": "income_section_rows", "value": len(income_section_rows)},
    {"metric": "income_section_characteristics", "value": income_section_characteristics},
    {"metric": "keyword_match_rows", "value": len(keyword_rows)},
    {"metric": "keyword_characteristics_found", "value": keyword_characteristics},
    {"metric": "ranked_candidates", "value": ranked_candidate_count},
    {"metric": "direct_aggregate_candidates", "value": direct_candidate_count},
    {"metric": "derived_count_times_average_candidates", "value": derived_candidate_count},
    {"metric": "derived_employment_income_candidates", "value": employment_derived_candidate_count},
    {"metric": "formula_candidate_available", "value": formula_candidate_available},
    {"metric": "base_names_with_mojibake", "value": base_names_with_mojibake},
    {"metric": "raw_characteristic_names_with_mojibake", "value": raw_characteristic_names_with_mojibake},
    {
        "metric": "important_method_note",
        "value": (
            "This targeted inspection checks whether EARNDEN90 can be approximated either from a direct "
            "aggregate income row or from a derived aggregate employment income formula: number of employment "
            "income recipients times average employment income, divided by land area."
        ),
    },
    {
        "metric": "recommended_next_step",
        "value": (
            "Review earnings_density_income_section_characteristic_inventory_100_260_2021.csv and "
            "earnings_density_formula_audit_2021.csv. If the derived employment-income candidate is valid, "
            "generate the cleaner with forced characteristic IDs."
        ),
    },
]

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CENSUS DIVISION EARNINGS DENSITY TARGETED INSPECTION 2021")
print("=" * 72)

print("\nInputs:")
print("Raw profile:", safe_relative(RAW_PROFILE))
print("Base frame:", safe_relative(BASE_CD_FRAME))
print("Raw encoding:", raw_encoding)
print("Base encoding:", base_encoding)

print("\nIncome section:")
print("Rows:", len(income_section_rows))
print("Characteristics:", income_section_characteristics)
print("ID range:", INCOME_SECTION_ID_MIN, "to", INCOME_SECTION_ID_MAX)

print("\nCandidates:")
print("Ranked candidates:", ranked_candidate_count)
print("Direct aggregate candidates:", direct_candidate_count)
print("Derived count-times-average candidates:", derived_candidate_count)
print("Derived employment-income candidates:", employment_derived_candidate_count)
print("Formula candidate available:", formula_candidate_available)

print("\nTarget summary:")
print(target_summary.to_string(index=False))

print("\nTop income-section characteristics:")
if income_inventory.empty:
    print("[none]")
else:
    display_cols = [
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "role",
        "income_domain",
        "year",
        "sample_class",
        "value_non_missing",
        "value_min",
        "value_max",
        "value_mean",
        "rate_non_missing",
        "rate_min",
        "rate_max",
    ]
    display_cols = [col for col in display_cols if col in income_inventory.columns]
    print(income_inventory[display_cols].head(80).to_string(index=False))

print("\nRanked candidates:")
if ranked_candidates.empty:
    print("[none]")
else:
    print(ranked_candidates.head(40).to_string(index=False))

print("\nFormula audit preview:")
if formula_audit.empty:
    print("[none]")
else:
    print(formula_audit.head(30).to_string(index=False))

print("\nSaved:")
print(OUTPUT_INCOME_SECTION_ROWS)
print(OUTPUT_INCOME_SECTION_INVENTORY)
print(OUTPUT_KEYWORD_ROWS)
print(OUTPUT_CHARACTERISTIC_SUMMARY)
print(OUTPUT_RANKED_CANDIDATES)
print(OUTPUT_FORMULA_AUDIT)
print(OUTPUT_TARGET_SUMMARY)
print(OUTPUT_SUMMARY)

print("\nDone.")