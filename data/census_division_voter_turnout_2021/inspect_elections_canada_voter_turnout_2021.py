from pathlib import Path
import re
import unicodedata
import pandas as pd


# ============================================================
# Inspect Elections Canada Voter Turnout Sources 2021
# ============================================================
#
# Purpose:
#   Inspect downloaded Elections Canada 44th General Election CSV tables
#   to determine whether they can support a SoVI-style political/civic
#   participation proxy for:
#
#       PCTVOTE92 -> pct_vote_leading_party
#
# Sources inspected:
#
#   Table 3:
#       Number of ballots cast and voter turnout
#       Provincial/territorial validation table.
#
#   Table 11:
#       Voting results by electoral district
#       Main source for federal electoral district turnout.
#
#   Table 12:
#       List of candidates by electoral district and individual results
#       Candidate-level source for leading-candidate / leading-party vote share.
#
# Candidate output variables:
#
#   voter_turnout_pct_federal_2021
#       Published district-level turnout from Table 11.
#
#   pct_vote_leading_candidate_federal_2021
#       Leading candidate share of valid votes in each federal electoral district.
#
#   pct_vote_leading_party_federal_2021
#       Leading party share of valid votes in each federal electoral district.
#       Note: Table 12 does not include explicit party affiliation according to
#       the supplied data dictionary, so this may remain unavailable unless a
#       party column is present in the file or another source is added.
#
# Run from data/:
#
#   python census_division_voter_turnout_2021/inspect_elections_canada_voter_turnout_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_voter_turnout_2021"
RAW_DIR = SECTION_DIR / "raw"
OUTPUT_DIR = SECTION_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE3_PATH = RAW_DIR / "number_of_ballots_cast_and_voter_turnout_table3.csv"
TABLE11_PATH = RAW_DIR / "voting_result_by_electoral_district_table11.csv"
TABLE12_PATH = RAW_DIR / "list_of_candidates_by_electoral_district_and_individual_results_table12.csv"

OUTPUT_FILE_INVENTORY = OUTPUT_DIR / "elections_canada_file_inventory_2021.csv"
OUTPUT_COLUMN_INVENTORY = OUTPUT_DIR / "elections_canada_column_inventory_2021.csv"

OUTPUT_TABLE3_PREVIEW = OUTPUT_DIR / "table3_number_of_ballots_cast_and_voter_turnout_preview_2021.csv"
OUTPUT_TABLE11_PREVIEW = OUTPUT_DIR / "table11_voting_results_by_electoral_district_preview_2021.csv"
OUTPUT_TABLE12_PREVIEW = OUTPUT_DIR / "table12_candidate_results_preview_2021.csv"

OUTPUT_TABLE3_LONG_AUDIT = OUTPUT_DIR / "table3_turnout_long_audit_2021.csv"
OUTPUT_TABLE11_LONG_AUDIT = OUTPUT_DIR / "table11_voting_results_long_audit_2021.csv"
OUTPUT_TABLE12_LONG_AUDIT = OUTPUT_DIR / "table12_candidate_results_long_audit_2021.csv"

OUTPUT_QC_DISTRICT_TURNOUT_CANDIDATES = OUTPUT_DIR / "quebec_federal_district_turnout_candidates_2021.csv"
OUTPUT_QC_LEADING_CANDIDATE_CANDIDATES = OUTPUT_DIR / "quebec_federal_district_leading_candidate_vote_share_candidates_2021.csv"
OUTPUT_QC_LEADING_PARTY_CANDIDATES = OUTPUT_DIR / "quebec_federal_district_leading_party_vote_share_candidates_2021.csv"

OUTPUT_DISTRICT_COMBINED = OUTPUT_DIR / "quebec_federal_district_vote_proxy_candidates_combined_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "elections_canada_voter_turnout_inspection_summary_2021.csv"


# -----------------------------
# Config
# -----------------------------

CSV_ENCODING_CANDIDATES = [
    "utf-8-sig",
    "utf-8",
    "cp1252",
    "latin1",
]

DISTRICT_ID_CANDIDATES = [
    "electoral_district_number_numero_de_circonscription",
    "electoral_district_number",
    "electoral_district_no",
    "electoral_district_code",
    "district_number",
    "district_no",
    "district_code",
    "ed_number",
    "ed_no",
    "ed_code",
    "circonscription_number",
    "circonscription_no",
    "numero_de_circonscription",
]

DISTRICT_NAME_CANDIDATES = [
    "electoral_district_name_nom_de_circonscription",
    "electoral_district_name",
    "electoral_district",
    "district_name",
    "district",
    "ed_name",
    "name_of_electoral_district",
    "nom_de_circonscription",
    "circonscription",
]

PROVINCE_CANDIDATES = [
    "province",
    "province_territory",
    "province_or_territory",
    "prov",
    "province_territoire",
]

REGISTERED_ELECTORS_CANDIDATES = [
    "electors_electeurs",
    "registered_electors",
    "number_of_registered_electors",
    "electors",
    "number_of_electors",
    "electors_on_lists",
    "electors_on_the_lists",
    "registered_voters",
    "nombre_d_electeurs_inscrits",
    "electeurs_inscrits",
]

BALLOTS_CAST_CANDIDATES = [
    "total_ballots_cast_total_des_bulletins_deposes",
    "ballots_cast",
    "number_of_ballots_cast",
    "total_ballots_cast",
    "votes_cast",
    "number_of_votes_cast",
    "total_votes_cast",
    "bulletins_deposes",
    "bulletins_de_vote_deposes",
]

VALID_BALLOTS_CANDIDATES = [
    "valid_ballots_bulletins_valides",
    "valid_ballots",
    "valid_votes",
    "number_of_valid_ballots",
    "number_of_valid_votes",
    "total_valid_ballots",
    "total_valid_votes",
    "votes_valides",
    "bulletins_valides",
]

REJECTED_BALLOTS_CANDIDATES = [
    "rejected_ballots_bulletins_rejetes",
    "rejected_ballots",
    "rejected_votes",
    "number_of_rejected_ballots",
    "number_of_rejected_votes",
    "total_rejected_ballots",
    "total_rejected_votes",
    "bulletins_rejetes",
    "votes_rejetes",
]

TURNOUT_CANDIDATES = [
    "percentage_of_voter_turnout_pourcentage_de_la_participation_electorale",
    "voter_turnout",
    "voter_turnout_pct",
    "voter_turnout_percent",
    "turnout",
    "turnout_pct",
    "turnout_percent",
    "percentage_voter_turnout",
    "percent_voter_turnout",
    "taux_de_participation",
    "participation_electorale",
]

CANDIDATE_NAME_CANDIDATES = [
    "candidate_candidat",
    "candidate",
    "candidat",
]

CANDIDATE_RESIDENCE_CANDIDATES = [
    "candidate_residence_residence_du_candidat",
    "candidate_residence",
    "residence_du_candidat",
]

CANDIDATE_OCCUPATION_CANDIDATES = [
    "candidate_occupation_profession_du_candidat",
    "candidate_occupation",
    "profession_du_candidat",
]

CANDIDATE_VOTE_CANDIDATES = [
    "votes_obtained_votes_obtenus",
    "votes_obtained",
    "valid_votes_obtained",
    "votes",
    "number_of_votes",
    "candidate_votes",
    "valid_votes_for_candidate",
    "votes_received",
    "voix_obtenues",
    "votes_obtenus",
    "nombre_de_votes",
]

CANDIDATE_PERCENT_CANDIDATES = [
    "percentage_of_votes_obtained_pourcentage_des_votes_obtenus",
    "percentage_of_votes_obtained",
    "pourcentage_des_votes_obtenus",
    "candidate_vote_share",
    "candidate_vote_share_pct",
    "percent_votes_obtained",
]

MAJORITY_CANDIDATES = [
    "majority_majorite",
    "majority",
    "majorite",
]

MAJORITY_PERCENT_CANDIDATES = [
    "majority_percentage_pourcentage_de_majorite",
    "majority_percentage",
    "pourcentage_de_majorite",
]

PARTY_CANDIDATES = [
    "political_affiliation",
    "party",
    "party_name",
    "political_party",
    "affiliation",
    "appartenance_politique",
    "parti",
]


# -----------------------------
# Helpers
# -----------------------------

def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_column_name(value: object) -> str:
    text = strip_accents(str(value).strip().lower())
    text = text.replace("%", " pct ")
    text = text.replace("#", " number ")
    text = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_column_name(col) for col in out.columns]
    return out


def normalize_text_value(value: object) -> str:
    if pd.isna(value):
        return ""

    text = strip_accents(str(value).strip().lower())
    text = re.sub(r"\s+", " ", text)
    return text


def read_csv_flex(path: Path) -> tuple[pd.DataFrame, str, str]:
    last_error = None

    for encoding in CSV_ENCODING_CANDIDATES:
        try:
            df = pd.read_csv(
                path,
                dtype=str,
                encoding=encoding,
                low_memory=False,
            )
            return df, encoding, "comma"
        except Exception as exc:
            last_error = exc

        try:
            df = pd.read_csv(
                path,
                dtype=str,
                encoding=encoding,
                sep=None,
                engine="python",
            )
            return df, encoding, "python_sep_auto"
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not read {path}. Last error: {last_error}")


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace("\u00a0", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


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


def filter_quebec(df: pd.DataFrame, province_col: str | None) -> pd.DataFrame:
    if province_col is None:
        return df.copy()

    province_norm = df[province_col].map(normalize_text_value)

    return df[
        province_norm.isin({"quebec", "qc", "que", "24"})
        | province_norm.str.contains("quebec", na=False)
    ].copy()


def write_preview(df: pd.DataFrame, path: Path, n: int = 100) -> None:
    df.head(n).to_csv(path, index=False, encoding="utf-8")


def selected_column_audit(
    df: pd.DataFrame,
    qc_df: pd.DataFrame,
    col_type: str,
    selected_col: str | None,
) -> dict:
    if selected_col is None:
        return {
            "column_type": col_type,
            "selected_column": "",
            "found": False,
            "non_missing_all_rows": 0,
            "non_missing_quebec_rows": 0,
            "unique_quebec_values": 0,
            "numeric_non_missing_quebec_rows": 0,
            "numeric_min_quebec": None,
            "numeric_max_quebec": None,
            "numeric_mean_quebec": None,
            "notes": "No matching column detected.",
        }

    summary = numeric_summary(qc_df[selected_col]) if not qc_df.empty else {
        "non_missing": 0,
        "missing": 0,
        "min": None,
        "max": None,
        "mean": None,
        "median": None,
    }

    return {
        "column_type": col_type,
        "selected_column": selected_col,
        "found": True,
        "non_missing_all_rows": int(df[selected_col].notna().sum()),
        "non_missing_quebec_rows": int(qc_df[selected_col].notna().sum()) if not qc_df.empty else 0,
        "unique_quebec_values": int(qc_df[selected_col].nunique(dropna=True)) if not qc_df.empty else 0,
        "numeric_non_missing_quebec_rows": summary["non_missing"],
        "numeric_min_quebec": summary["min"],
        "numeric_max_quebec": summary["max"],
        "numeric_mean_quebec": summary["mean"],
        "notes": "",
    }


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file for {label}:\n{path}")


# -----------------------------
# Validate files
# -----------------------------

expected_files = [
    ("table3_ballots_turnout", TABLE3_PATH),
    ("table11_voting_results_by_electoral_district", TABLE11_PATH),
    ("table12_candidate_results", TABLE12_PATH),
]

for label, path in expected_files:
    require_file(path, label)


# -----------------------------
# Load files
# -----------------------------

loaded = {}
file_inventory_rows = []
column_inventory_rows = []

for label, path in expected_files:
    raw_df, encoding, parse_mode = read_csv_flex(path)
    df = normalize_columns(raw_df)

    loaded[label] = {
        "path": path,
        "raw": raw_df,
        "df": df,
        "encoding": encoding,
        "parse_mode": parse_mode,
    }

    file_inventory_rows.append(
        {
            "label": label,
            "relative_path": str(path.relative_to(DATA_DIR)),
            "exists": path.exists(),
            "encoding": encoding,
            "parse_mode": parse_mode,
            "rows": len(df),
            "columns": len(df.columns),
            "size_kb": round(path.stat().st_size / 1024, 2),
        }
    )

    for col in df.columns:
        summary = numeric_summary(df[col])
        column_inventory_rows.append(
            {
                "label": label,
                "relative_path": str(path.relative_to(DATA_DIR)),
                "column": col,
                "dtype": str(df[col].dtype),
                "non_missing": int(df[col].notna().sum()),
                "unique_values": int(df[col].nunique(dropna=True)),
                "numeric_non_missing": summary["non_missing"],
                "numeric_min": summary["min"],
                "numeric_max": summary["max"],
                "numeric_mean": summary["mean"],
            }
        )

file_inventory = pd.DataFrame(file_inventory_rows)
column_inventory = pd.DataFrame(column_inventory_rows)

file_inventory.to_csv(OUTPUT_FILE_INVENTORY, index=False, encoding="utf-8")
column_inventory.to_csv(OUTPUT_COLUMN_INVENTORY, index=False, encoding="utf-8")

write_preview(loaded["table3_ballots_turnout"]["df"], OUTPUT_TABLE3_PREVIEW)
write_preview(loaded["table11_voting_results_by_electoral_district"]["df"], OUTPUT_TABLE11_PREVIEW)
write_preview(loaded["table12_candidate_results"]["df"], OUTPUT_TABLE12_PREVIEW)


# -----------------------------
# Table 3 inspection
# -----------------------------

table3 = loaded["table3_ballots_turnout"]["df"].copy()
table3_cols = list(table3.columns)

table3_province_col = first_existing_column(table3_cols, PROVINCE_CANDIDATES)
table3_district_id_col = first_existing_column(table3_cols, DISTRICT_ID_CANDIDATES)
table3_district_name_col = first_existing_column(table3_cols, DISTRICT_NAME_CANDIDATES)
table3_registered_col = first_existing_column(table3_cols, REGISTERED_ELECTORS_CANDIDATES)
table3_ballots_cast_col = first_existing_column(table3_cols, BALLOTS_CAST_CANDIDATES)
table3_valid_col = first_existing_column(table3_cols, VALID_BALLOTS_CANDIDATES)
table3_rejected_col = first_existing_column(table3_cols, REJECTED_BALLOTS_CANDIDATES)
table3_turnout_col = first_existing_column(table3_cols, TURNOUT_CANDIDATES)

table3_qc = filter_quebec(table3, table3_province_col)

table3_long_audit = pd.DataFrame(
    [
        selected_column_audit(table3, table3_qc, "province", table3_province_col),
        selected_column_audit(table3, table3_qc, "district_id", table3_district_id_col),
        selected_column_audit(table3, table3_qc, "district_name", table3_district_name_col),
        selected_column_audit(table3, table3_qc, "registered_electors", table3_registered_col),
        selected_column_audit(table3, table3_qc, "ballots_cast", table3_ballots_cast_col),
        selected_column_audit(table3, table3_qc, "valid_ballots", table3_valid_col),
        selected_column_audit(table3, table3_qc, "rejected_ballots", table3_rejected_col),
        selected_column_audit(table3, table3_qc, "published_turnout", table3_turnout_col),
    ]
)
table3_long_audit.to_csv(OUTPUT_TABLE3_LONG_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Table 11 inspection
# -----------------------------

table11 = loaded["table11_voting_results_by_electoral_district"]["df"].copy()
table11_cols = list(table11.columns)

table11_province_col = first_existing_column(table11_cols, PROVINCE_CANDIDATES)
table11_district_id_col = first_existing_column(table11_cols, DISTRICT_ID_CANDIDATES)
table11_district_name_col = first_existing_column(table11_cols, DISTRICT_NAME_CANDIDATES)
table11_registered_col = first_existing_column(table11_cols, REGISTERED_ELECTORS_CANDIDATES)
table11_ballots_cast_col = first_existing_column(table11_cols, BALLOTS_CAST_CANDIDATES)
table11_valid_col = first_existing_column(table11_cols, VALID_BALLOTS_CANDIDATES)
table11_rejected_col = first_existing_column(table11_cols, REJECTED_BALLOTS_CANDIDATES)
table11_turnout_col = first_existing_column(table11_cols, TURNOUT_CANDIDATES)

table11_qc = filter_quebec(table11, table11_province_col)

table11_long_audit = pd.DataFrame(
    [
        selected_column_audit(table11, table11_qc, "province", table11_province_col),
        selected_column_audit(table11, table11_qc, "district_id", table11_district_id_col),
        selected_column_audit(table11, table11_qc, "district_name", table11_district_name_col),
        selected_column_audit(table11, table11_qc, "registered_electors", table11_registered_col),
        selected_column_audit(table11, table11_qc, "ballots_cast", table11_ballots_cast_col),
        selected_column_audit(table11, table11_qc, "valid_ballots", table11_valid_col),
        selected_column_audit(table11, table11_qc, "rejected_ballots", table11_rejected_col),
        selected_column_audit(table11, table11_qc, "published_turnout", table11_turnout_col),
    ]
)
table11_long_audit.to_csv(OUTPUT_TABLE11_LONG_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Table 12 inspection
# -----------------------------

table12 = loaded["table12_candidate_results"]["df"].copy()
table12_cols = list(table12.columns)

table12_province_col = first_existing_column(table12_cols, PROVINCE_CANDIDATES)
table12_district_id_col = first_existing_column(table12_cols, DISTRICT_ID_CANDIDATES)
table12_district_name_col = first_existing_column(table12_cols, DISTRICT_NAME_CANDIDATES)
table12_candidate_col = first_existing_column(table12_cols, CANDIDATE_NAME_CANDIDATES)
table12_candidate_residence_col = first_existing_column(table12_cols, CANDIDATE_RESIDENCE_CANDIDATES)
table12_candidate_occupation_col = first_existing_column(table12_cols, CANDIDATE_OCCUPATION_CANDIDATES)
table12_candidate_votes_col = first_existing_column(table12_cols, CANDIDATE_VOTE_CANDIDATES)
table12_candidate_pct_col = first_existing_column(table12_cols, CANDIDATE_PERCENT_CANDIDATES)
table12_majority_col = first_existing_column(table12_cols, MAJORITY_CANDIDATES)
table12_majority_pct_col = first_existing_column(table12_cols, MAJORITY_PERCENT_CANDIDATES)
table12_party_col = first_existing_column(table12_cols, PARTY_CANDIDATES)

table12_qc = filter_quebec(table12, table12_province_col)

table12_long_audit = pd.DataFrame(
    [
        selected_column_audit(table12, table12_qc, "province", table12_province_col),
        selected_column_audit(table12, table12_qc, "district_id", table12_district_id_col),
        selected_column_audit(table12, table12_qc, "district_name", table12_district_name_col),
        selected_column_audit(table12, table12_qc, "candidate", table12_candidate_col),
        selected_column_audit(table12, table12_qc, "candidate_residence", table12_candidate_residence_col),
        selected_column_audit(table12, table12_qc, "candidate_occupation", table12_candidate_occupation_col),
        selected_column_audit(table12, table12_qc, "candidate_votes", table12_candidate_votes_col),
        selected_column_audit(table12, table12_qc, "candidate_vote_pct", table12_candidate_pct_col),
        selected_column_audit(table12, table12_qc, "majority", table12_majority_col),
        selected_column_audit(table12, table12_qc, "majority_pct", table12_majority_pct_col),
        selected_column_audit(table12, table12_qc, "party", table12_party_col),
    ]
)
table12_long_audit.to_csv(OUTPUT_TABLE12_LONG_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Build Table 11 district-level turnout candidates
# -----------------------------

turnout_candidates = []

for source_label, df, province_col, district_id_col, district_name_col, registered_col, ballots_col, valid_col, rejected_col, turnout_col in [
    (
        "table3_number_of_ballots_cast_and_voter_turnout",
        table3,
        table3_province_col,
        table3_district_id_col,
        table3_district_name_col,
        table3_registered_col,
        table3_ballots_cast_col,
        table3_valid_col,
        table3_rejected_col,
        table3_turnout_col,
    ),
    (
        "table11_voting_results_by_electoral_district",
        table11,
        table11_province_col,
        table11_district_id_col,
        table11_district_name_col,
        table11_registered_col,
        table11_ballots_cast_col,
        table11_valid_col,
        table11_rejected_col,
        table11_turnout_col,
    ),
]:
    qc = filter_quebec(df, province_col)

    if qc.empty:
        continue

    temp = pd.DataFrame(index=qc.index)
    temp["source_table"] = source_label

    temp["federal_electoral_district_id"] = (
        qc[district_id_col].astype("string").str.strip()
        if district_id_col is not None
        else pd.NA
    )

    temp["federal_electoral_district_name"] = (
        qc[district_name_col].astype("string").str.strip()
        if district_name_col is not None
        else pd.NA
    )

    temp["province"] = (
        qc[province_col].astype("string").str.strip()
        if province_col is not None
        else pd.NA
    )

    temp["registered_electors"] = (
        clean_numeric(qc[registered_col])
        if registered_col is not None
        else pd.NA
    )

    temp["ballots_cast"] = (
        clean_numeric(qc[ballots_col])
        if ballots_col is not None
        else pd.NA
    )

    temp["valid_ballots"] = (
        clean_numeric(qc[valid_col])
        if valid_col is not None
        else pd.NA
    )

    temp["rejected_ballots"] = (
        clean_numeric(qc[rejected_col])
        if rejected_col is not None
        else pd.NA
    )

    temp["published_voter_turnout_pct"] = (
        clean_numeric(qc[turnout_col])
        if turnout_col is not None
        else pd.NA
    )

    valid_numeric = clean_numeric(temp["valid_ballots"])
    rejected_numeric = clean_numeric(temp["rejected_ballots"])
    registered_numeric = clean_numeric(temp["registered_electors"])
    ballots_cast_numeric = clean_numeric(temp["ballots_cast"])

    has_valid_or_rejected = valid_numeric.notna() | rejected_numeric.notna()
    temp["computed_ballots_cast_from_valid_plus_rejected"] = valid_numeric.fillna(0) + rejected_numeric.fillna(0)
    temp.loc[~has_valid_or_rejected, "computed_ballots_cast_from_valid_plus_rejected"] = pd.NA

    ballots_from_parts = clean_numeric(temp["computed_ballots_cast_from_valid_plus_rejected"])

    temp["computed_voter_turnout_pct_from_ballots_cast"] = 100 * ballots_cast_numeric / registered_numeric
    temp["computed_voter_turnout_pct_from_valid_plus_rejected"] = 100 * ballots_from_parts / registered_numeric

    temp["best_available_voter_turnout_pct"] = temp["published_voter_turnout_pct"]
    temp["best_available_voter_turnout_source"] = "published"

    missing_best = temp["best_available_voter_turnout_pct"].isna()
    temp.loc[missing_best, "best_available_voter_turnout_pct"] = temp.loc[
        missing_best,
        "computed_voter_turnout_pct_from_ballots_cast",
    ]
    temp.loc[missing_best, "best_available_voter_turnout_source"] = "computed_from_ballots_cast"

    missing_best = temp["best_available_voter_turnout_pct"].isna()
    temp.loc[missing_best, "best_available_voter_turnout_pct"] = temp.loc[
        missing_best,
        "computed_voter_turnout_pct_from_valid_plus_rejected",
    ]
    temp.loc[missing_best, "best_available_voter_turnout_source"] = "computed_from_valid_plus_rejected"

    temp.loc[temp["best_available_voter_turnout_pct"].isna(), "best_available_voter_turnout_source"] = "unavailable"

    district_like = (
        temp["federal_electoral_district_name"].notna()
        | temp["federal_electoral_district_id"].notna()
    )
    temp["is_district_level_candidate"] = district_like

    turnout_candidates.append(temp)

if turnout_candidates:
    turnout_candidates_df = pd.concat(turnout_candidates, ignore_index=True)

    dedupe_cols = [
        "source_table",
        "federal_electoral_district_id",
        "federal_electoral_district_name",
    ]

    turnout_candidates_df = turnout_candidates_df.drop_duplicates(
        subset=[col for col in dedupe_cols if col in turnout_candidates_df.columns],
        keep="first",
    )

    district_turnout_candidates_df = turnout_candidates_df[
        turnout_candidates_df["is_district_level_candidate"].fillna(False)
    ].copy()
else:
    turnout_candidates_df = pd.DataFrame()
    district_turnout_candidates_df = pd.DataFrame()

district_turnout_candidates_df.to_csv(
    OUTPUT_QC_DISTRICT_TURNOUT_CANDIDATES,
    index=False,
    encoding="utf-8",
)


# -----------------------------
# Build leading-candidate vote-share candidates from Table 12
# -----------------------------

can_attempt_leading_candidate = (
    table12_district_name_col is not None
    and table12_candidate_col is not None
    and table12_candidate_votes_col is not None
)

if can_attempt_leading_candidate:
    qc = table12_qc.copy()

    group_cols = []
    if table12_district_id_col is not None:
        group_cols.append(table12_district_id_col)
    group_cols.append(table12_district_name_col)

    working = qc.copy()
    working["_candidate_votes_numeric"] = clean_numeric(working[table12_candidate_votes_col])
    working["_candidate_vote_pct_numeric"] = (
        clean_numeric(working[table12_candidate_pct_col])
        if table12_candidate_pct_col is not None
        else pd.NA
    )

    working["_candidate"] = working[table12_candidate_col].astype("string").str.strip()

    # Preserve optional candidate metadata.
    if table12_candidate_residence_col is not None:
        working["_candidate_residence"] = working[table12_candidate_residence_col].astype("string").str.strip()
    else:
        working["_candidate_residence"] = pd.NA

    if table12_candidate_occupation_col is not None:
        working["_candidate_occupation"] = working[table12_candidate_occupation_col].astype("string").str.strip()
    else:
        working["_candidate_occupation"] = pd.NA

    if table12_majority_col is not None:
        working["_majority"] = clean_numeric(working[table12_majority_col])
    else:
        working["_majority"] = pd.NA

    if table12_majority_pct_col is not None:
        working["_majority_pct"] = clean_numeric(working[table12_majority_pct_col])
    else:
        working["_majority_pct"] = pd.NA

    total_by_district = (
        working
        .groupby(group_cols, dropna=False)["_candidate_votes_numeric"]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"_candidate_votes_numeric": "district_valid_votes_from_candidate_sum"})
    )

    working = working.merge(
        total_by_district,
        on=group_cols,
        how="left",
        validate="many_to_one",
    )

    working["computed_candidate_vote_share_pct"] = (
        100
        * working["_candidate_votes_numeric"]
        / working["district_valid_votes_from_candidate_sum"]
    )

    # Prefer the published candidate percent if present; otherwise computed.
    working["best_available_candidate_vote_share_pct"] = working["_candidate_vote_pct_numeric"]
    working["best_available_candidate_vote_share_source"] = "published"

    missing_pct = working["best_available_candidate_vote_share_pct"].isna()
    working.loc[missing_pct, "best_available_candidate_vote_share_pct"] = working.loc[
        missing_pct,
        "computed_candidate_vote_share_pct",
    ]
    working.loc[missing_pct, "best_available_candidate_vote_share_source"] = "computed_from_candidate_vote_sum"

    leading = (
        working
        .sort_values("_candidate_votes_numeric", ascending=False)
        .groupby(group_cols, dropna=False)
        .head(1)
        .copy()
    )

    if table12_district_id_col is not None:
        leading["federal_electoral_district_id"] = leading[table12_district_id_col].astype("string").str.strip()
    else:
        leading["federal_electoral_district_id"] = pd.NA

    leading["federal_electoral_district_name"] = leading[table12_district_name_col].astype("string").str.strip()
    leading["province"] = leading[table12_province_col].astype("string").str.strip() if table12_province_col is not None else pd.NA
    leading["leading_candidate"] = leading["_candidate"]
    leading["leading_candidate_residence"] = leading["_candidate_residence"]
    leading["leading_candidate_occupation"] = leading["_candidate_occupation"]
    leading["leading_candidate_valid_votes"] = leading["_candidate_votes_numeric"]
    leading["leading_candidate_published_vote_share_pct"] = leading["_candidate_vote_pct_numeric"]
    leading["pct_vote_leading_candidate_federal_2021"] = leading["best_available_candidate_vote_share_pct"]
    leading["pct_vote_leading_candidate_federal_2021_source"] = leading["best_available_candidate_vote_share_source"]
    leading["majority"] = leading["_majority"]
    leading["majority_pct"] = leading["_majority_pct"]
    leading["source_table"] = "table12_candidate_results"

    leading_candidate_rows = leading[
        [
            "source_table",
            "province",
            "federal_electoral_district_id",
            "federal_electoral_district_name",
            "leading_candidate",
            "leading_candidate_residence",
            "leading_candidate_occupation",
            "leading_candidate_valid_votes",
            "district_valid_votes_from_candidate_sum",
            "leading_candidate_published_vote_share_pct",
            "computed_candidate_vote_share_pct",
            "pct_vote_leading_candidate_federal_2021",
            "pct_vote_leading_candidate_federal_2021_source",
            "majority",
            "majority_pct",
        ]
    ].copy()
else:
    leading_candidate_rows = pd.DataFrame(
        columns=[
            "source_table",
            "province",
            "federal_electoral_district_id",
            "federal_electoral_district_name",
            "leading_candidate",
            "leading_candidate_residence",
            "leading_candidate_occupation",
            "leading_candidate_valid_votes",
            "district_valid_votes_from_candidate_sum",
            "leading_candidate_published_vote_share_pct",
            "computed_candidate_vote_share_pct",
            "pct_vote_leading_candidate_federal_2021",
            "pct_vote_leading_candidate_federal_2021_source",
            "majority",
            "majority_pct",
        ]
    )

leading_candidate_rows.to_csv(
    OUTPUT_QC_LEADING_CANDIDATE_CANDIDATES,
    index=False,
    encoding="utf-8",
)


# -----------------------------
# Build leading-party vote-share candidates from Table 12 if party exists
# -----------------------------

can_attempt_leading_party = (
    table12_district_name_col is not None
    and table12_party_col is not None
    and table12_candidate_votes_col is not None
)

if can_attempt_leading_party:
    qc = table12_qc.copy()

    group_cols = []
    if table12_district_id_col is not None:
        group_cols.append(table12_district_id_col)
    group_cols.append(table12_district_name_col)

    working = qc.copy()
    working["_candidate_votes_numeric"] = clean_numeric(working[table12_candidate_votes_col])
    working["_party"] = working[table12_party_col].astype("string").str.strip()

    party_by_district = (
        working
        .groupby(group_cols + ["_party"], dropna=False)["_candidate_votes_numeric"]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"_candidate_votes_numeric": "party_valid_votes"})
    )

    total_by_district = (
        party_by_district
        .groupby(group_cols, dropna=False)["party_valid_votes"]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"party_valid_votes": "district_valid_votes_from_party_sum"})
    )

    party_by_district = party_by_district.merge(
        total_by_district,
        on=group_cols,
        how="left",
        validate="many_to_one",
    )

    party_by_district["party_vote_share_pct"] = (
        100
        * party_by_district["party_valid_votes"]
        / party_by_district["district_valid_votes_from_party_sum"]
    )

    leading_party = (
        party_by_district
        .sort_values("party_valid_votes", ascending=False)
        .groupby(group_cols, dropna=False)
        .head(1)
        .copy()
    )

    if table12_district_id_col is not None:
        leading_party["federal_electoral_district_id"] = leading_party[table12_district_id_col].astype("string").str.strip()
    else:
        leading_party["federal_electoral_district_id"] = pd.NA

    leading_party["federal_electoral_district_name"] = leading_party[table12_district_name_col].astype("string").str.strip()
    leading_party["leading_party"] = leading_party["_party"]
    leading_party["leading_party_valid_votes"] = leading_party["party_valid_votes"]
    leading_party["pct_vote_leading_party_federal_2021"] = leading_party["party_vote_share_pct"]
    leading_party["source_table"] = "table12_candidate_results"

    leading_party_rows = leading_party[
        [
            "source_table",
            "federal_electoral_district_id",
            "federal_electoral_district_name",
            "leading_party",
            "leading_party_valid_votes",
            "district_valid_votes_from_party_sum",
            "pct_vote_leading_party_federal_2021",
        ]
    ].copy()

else:
    leading_party_rows = pd.DataFrame(
        columns=[
            "source_table",
            "federal_electoral_district_id",
            "federal_electoral_district_name",
            "leading_party",
            "leading_party_valid_votes",
            "district_valid_votes_from_party_sum",
            "pct_vote_leading_party_federal_2021",
        ]
    )

leading_party_rows.to_csv(
    OUTPUT_QC_LEADING_PARTY_CANDIDATES,
    index=False,
    encoding="utf-8",
)


# -----------------------------
# Combined district proxy table
# -----------------------------

if not district_turnout_candidates_df.empty:
    turnout_for_join = district_turnout_candidates_df[
        [
            "federal_electoral_district_id",
            "federal_electoral_district_name",
            "registered_electors",
            "ballots_cast",
            "valid_ballots",
            "rejected_ballots",
            "best_available_voter_turnout_pct",
            "best_available_voter_turnout_source",
        ]
    ].copy()
else:
    turnout_for_join = pd.DataFrame()

if not leading_candidate_rows.empty:
    leading_candidate_for_join = leading_candidate_rows[
        [
            "federal_electoral_district_id",
            "federal_electoral_district_name",
            "leading_candidate",
            "leading_candidate_valid_votes",
            "district_valid_votes_from_candidate_sum",
            "pct_vote_leading_candidate_federal_2021",
            "pct_vote_leading_candidate_federal_2021_source",
            "majority",
            "majority_pct",
        ]
    ].copy()
else:
    leading_candidate_for_join = pd.DataFrame()

if not leading_party_rows.empty:
    leading_party_for_join = leading_party_rows[
        [
            "federal_electoral_district_id",
            "federal_electoral_district_name",
            "leading_party",
            "leading_party_valid_votes",
            "district_valid_votes_from_party_sum",
            "pct_vote_leading_party_federal_2021",
        ]
    ].copy()
else:
    leading_party_for_join = pd.DataFrame()

combined = turnout_for_join.copy()

if not combined.empty and not leading_candidate_for_join.empty:
    combined = combined.merge(
        leading_candidate_for_join,
        on=["federal_electoral_district_id", "federal_electoral_district_name"],
        how="outer",
        validate="one_to_one",
    )
elif combined.empty and not leading_candidate_for_join.empty:
    combined = leading_candidate_for_join.copy()

if not combined.empty and not leading_party_for_join.empty:
    combined = combined.merge(
        leading_party_for_join,
        on=["federal_electoral_district_id", "federal_electoral_district_name"],
        how="outer",
        validate="one_to_one",
    )
elif combined.empty and not leading_party_for_join.empty:
    combined = leading_party_for_join.copy()

combined.to_csv(OUTPUT_DISTRICT_COMBINED, index=False, encoding="utf-8")


# -----------------------------
# Summary
# -----------------------------

turnout_non_missing = (
    int(district_turnout_candidates_df["best_available_voter_turnout_pct"].notna().sum())
    if not district_turnout_candidates_df.empty and "best_available_voter_turnout_pct" in district_turnout_candidates_df.columns
    else 0
)

turnout_unique_districts = (
    int(district_turnout_candidates_df["federal_electoral_district_name"].nunique(dropna=True))
    if not district_turnout_candidates_df.empty and "federal_electoral_district_name" in district_turnout_candidates_df.columns
    else 0
)

leading_candidate_non_missing = (
    int(leading_candidate_rows["pct_vote_leading_candidate_federal_2021"].notna().sum())
    if not leading_candidate_rows.empty
    else 0
)

leading_candidate_unique_districts = (
    int(leading_candidate_rows["federal_electoral_district_name"].nunique(dropna=True))
    if not leading_candidate_rows.empty
    else 0
)

leading_party_non_missing = (
    int(leading_party_rows["pct_vote_leading_party_federal_2021"].notna().sum())
    if not leading_party_rows.empty
    else 0
)

leading_party_unique_districts = (
    int(leading_party_rows["federal_electoral_district_name"].nunique(dropna=True))
    if not leading_party_rows.empty
    else 0
)

turnout_status = (
    "district_turnout_candidate_found"
    if turnout_non_missing > 0
    else "district_turnout_not_found_yet"
)

leading_candidate_status = (
    "leading_candidate_vote_share_candidate_found"
    if leading_candidate_non_missing > 0
    else "leading_candidate_vote_share_not_found_yet"
)

leading_party_status = (
    "leading_party_vote_share_candidate_found"
    if leading_party_non_missing > 0
    else "leading_party_vote_share_not_found_yet"
)

summary_rows = [
    {"metric": "table3_file", "value": str(TABLE3_PATH.relative_to(DATA_DIR))},
    {"metric": "table11_file", "value": str(TABLE11_PATH.relative_to(DATA_DIR))},
    {"metric": "table12_file", "value": str(TABLE12_PATH.relative_to(DATA_DIR))},
    {"metric": "table3_rows", "value": len(table3)},
    {"metric": "table11_rows", "value": len(table11)},
    {"metric": "table12_rows", "value": len(table12)},
    {"metric": "table3_quebec_rows_after_filter", "value": len(table3_qc)},
    {"metric": "table11_quebec_rows_after_filter", "value": len(table11_qc)},
    {"metric": "table12_quebec_rows_after_filter", "value": len(table12_qc)},

    {"metric": "table11_district_id_column", "value": table11_district_id_col or ""},
    {"metric": "table11_district_name_column", "value": table11_district_name_col or ""},
    {"metric": "table11_registered_electors_column", "value": table11_registered_col or ""},
    {"metric": "table11_ballots_cast_column", "value": table11_ballots_cast_col or ""},
    {"metric": "table11_turnout_column", "value": table11_turnout_col or ""},

    {"metric": "table12_district_id_column", "value": table12_district_id_col or ""},
    {"metric": "table12_district_name_column", "value": table12_district_name_col or ""},
    {"metric": "table12_candidate_column", "value": table12_candidate_col or ""},
    {"metric": "table12_candidate_votes_column", "value": table12_candidate_votes_col or ""},
    {"metric": "table12_candidate_vote_pct_column", "value": table12_candidate_pct_col or ""},
    {"metric": "table12_majority_column", "value": table12_majority_col or ""},
    {"metric": "table12_majority_pct_column", "value": table12_majority_pct_col or ""},
    {"metric": "table12_party_column", "value": table12_party_col or ""},

    {"metric": "turnout_status", "value": turnout_status},
    {"metric": "turnout_candidate_non_missing_rows", "value": turnout_non_missing},
    {"metric": "turnout_candidate_unique_district_names", "value": turnout_unique_districts},

    {"metric": "leading_candidate_status", "value": leading_candidate_status},
    {"metric": "leading_candidate_candidate_non_missing_rows", "value": leading_candidate_non_missing},
    {"metric": "leading_candidate_candidate_unique_district_names", "value": leading_candidate_unique_districts},

    {"metric": "leading_party_status", "value": leading_party_status},
    {"metric": "leading_party_candidate_non_missing_rows", "value": leading_party_non_missing},
    {"metric": "leading_party_candidate_unique_district_names", "value": leading_party_unique_districts},

    {
        "metric": "method_note",
        "value": (
            "Table 11 provides federal electoral district turnout. Table 12 provides candidate-level "
            "vote results and can support leading-candidate vote share. The supplied Table 12 data "
            "dictionary does not include party affiliation, so leading-party vote share may remain "
            "unavailable unless the file contains an extra party column or a separate party mapping is added."
        ),
    },
    {
        "metric": "recommended_next_step",
        "value": (
            "Review the combined district proxy candidate file. If turnout and leading-candidate "
            "shares look valid, next locate the 2013 Representation Order federal electoral district "
            "boundary file and build a spatial allocation to Québec census divisions."
        ),
    },
]

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("ELECTIONS CANADA VOTER TURNOUT / VOTE SHARE INSPECTION 2021")
print("=" * 72)

print("\nFiles loaded:")
print(file_inventory.to_string(index=False))

print("\nTable 3 detected columns:")
print(table3_long_audit.to_string(index=False))

print("\nTable 11 detected columns:")
print(table11_long_audit.to_string(index=False))

print("\nTable 12 detected columns:")
print(table12_long_audit.to_string(index=False))

print("\nQuébec federal district turnout candidates:")
if district_turnout_candidates_df.empty:
    print("[none]")
else:
    display_cols = [
        "source_table",
        "federal_electoral_district_id",
        "federal_electoral_district_name",
        "registered_electors",
        "ballots_cast",
        "valid_ballots",
        "rejected_ballots",
        "published_voter_turnout_pct",
        "best_available_voter_turnout_pct",
        "best_available_voter_turnout_source",
    ]
    display_cols = [col for col in display_cols if col in district_turnout_candidates_df.columns]
    print(district_turnout_candidates_df[display_cols].head(30).to_string(index=False))

print("\nQuébec leading-candidate vote-share candidates:")
if leading_candidate_rows.empty:
    print("[none]")
else:
    print(leading_candidate_rows.head(30).to_string(index=False))

print("\nQuébec leading-party vote-share candidates:")
if leading_party_rows.empty:
    print("[none]")
else:
    print(leading_party_rows.head(30).to_string(index=False))

print("\nCombined district proxy candidates:")
if combined.empty:
    print("[none]")
else:
    print(combined.head(30).to_string(index=False))

print("\nSummary:")
print(summary.to_string(index=False))

print("\nSaved:")
print(OUTPUT_FILE_INVENTORY)
print(OUTPUT_COLUMN_INVENTORY)
print(OUTPUT_TABLE3_PREVIEW)
print(OUTPUT_TABLE11_PREVIEW)
print(OUTPUT_TABLE12_PREVIEW)
print(OUTPUT_TABLE3_LONG_AUDIT)
print(OUTPUT_TABLE11_LONG_AUDIT)
print(OUTPUT_TABLE12_LONG_AUDIT)
print(OUTPUT_QC_DISTRICT_TURNOUT_CANDIDATES)
print(OUTPUT_QC_LEADING_CANDIDATE_CANDIDATES)
print(OUTPUT_QC_LEADING_PARTY_CANDIDATES)
print(OUTPUT_DISTRICT_COMBINED)
print(OUTPUT_SUMMARY)

print("\nDone.")