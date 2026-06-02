from pathlib import Path
import re
import unicodedata
import pandas as pd


# ============================================================
# Clean Federal District Vote Proxy 2021
# ============================================================
#
# Purpose:
#   Build a clean Québec federal electoral district vote-proxy table from
#   Elections Canada 44th General Election tables.
#
# Inputs:
#   Table 11:
#       voting_result_by_electoral_district_table11.csv
#       Used for turnout, electors, ballots cast, valid ballots, rejected ballots.
#
#   Table 12:
#       list_of_candidates_by_electoral_district_and_individual_results_table12.csv
#       Used for candidate-level vote results and leading-candidate vote share.
#
# Outputs:
#   output/clean_quebec_federal_district_vote_proxy_2021.csv
#   output/clean_quebec_federal_district_vote_proxy_summary_2021.csv
#   output/clean_quebec_federal_district_vote_proxy_candidate_long_2021.csv
#   output/clean_quebec_federal_district_vote_proxy_variable_metadata_2021.csv
#
# Run from data/:
#   python census_division_voter_turnout_2021/clean_federal_district_vote_proxy_2021.py
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

TABLE11_PATH = RAW_DIR / "voting_result_by_electoral_district_table11.csv"
TABLE12_PATH = RAW_DIR / "list_of_candidates_by_electoral_district_and_individual_results_table12.csv"

OUTPUT_CLEAN = OUTPUT_DIR / "clean_quebec_federal_district_vote_proxy_2021.csv"
OUTPUT_CANDIDATE_LONG = OUTPUT_DIR / "clean_quebec_federal_district_vote_proxy_candidate_long_2021.csv"
OUTPUT_METADATA = OUTPUT_DIR / "clean_quebec_federal_district_vote_proxy_variable_metadata_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "clean_quebec_federal_district_vote_proxy_summary_2021.csv"


# -----------------------------
# Config
# -----------------------------

CSV_ENCODING_CANDIDATES = [
    "utf-8-sig",
    "utf-8",
    "cp1252",
    "latin1",
]

PROVINCE_CANDIDATES = [
    "province",
]

DISTRICT_ID_CANDIDATES = [
    "electoral_district_number_numero_de_circonscription",
    "electoral_district_number",
    "numero_de_circonscription",
]

DISTRICT_NAME_CANDIDATES = [
    "electoral_district_name_nom_de_circonscription",
    "electoral_district_name",
    "nom_de_circonscription",
]

REGISTERED_ELECTORS_CANDIDATES = [
    "electors_electeurs",
    "registered_electors",
    "electors",
]

BALLOTS_CAST_CANDIDATES = [
    "total_ballots_cast_total_des_bulletins_deposes",
    "total_ballots_cast",
    "ballots_cast",
]

VALID_BALLOTS_CANDIDATES = [
    "valid_ballots_bulletins_valides",
    "valid_ballots",
    "valid_votes",
]

REJECTED_BALLOTS_CANDIDATES = [
    "rejected_ballots_bulletins_rejetes",
    "rejected_ballots",
    "rejected_votes",
]

TURNOUT_CANDIDATES = [
    "percentage_of_voter_turnout_pourcentage_de_la_participation_electorale",
    "voter_turnout_pct",
    "turnout_pct",
]

CANDIDATE_CANDIDATES = [
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

CANDIDATE_VOTES_CANDIDATES = [
    "votes_obtained_votes_obtenus",
    "votes_obtained",
    "votes_obtenus",
    "candidate_votes",
]

CANDIDATE_VOTE_PCT_CANDIDATES = [
    "percentage_of_votes_obtained_pourcentage_des_votes_obtenus",
    "percentage_of_votes_obtained",
    "pourcentage_des_votes_obtenus",
    "candidate_vote_share_pct",
]

MAJORITY_CANDIDATES = [
    "majority_majorite",
    "majority",
    "majorite",
]

MAJORITY_PCT_CANDIDATES = [
    "majority_percentage_pourcentage_de_majorite",
    "majority_percentage",
    "pourcentage_de_majorite",
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


def read_csv_flex(path: Path) -> tuple[pd.DataFrame, str]:
    last_error = None

    for encoding in CSV_ENCODING_CANDIDATES:
        try:
            df = pd.read_csv(
                path,
                dtype=str,
                encoding=encoding,
                low_memory=False,
            )
            return df, encoding
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


def first_existing_column(columns: list[str], candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in columns:
            return col

    raise ValueError(
        f"Could not find required column for {label}.\n"
        f"Candidates:\n{candidates}\n\n"
        f"Available columns:\n{columns}"
    )


def optional_column(columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def filter_quebec(df: pd.DataFrame, province_col: str) -> pd.DataFrame:
    province_norm = df[province_col].map(normalize_text_value)

    return df[
        province_norm.isin({"quebec", "qc", "que", "24"})
        | province_norm.str.contains("quebec", na=False)
    ].copy()


def split_candidate_party(candidate_string: object) -> dict:
    """
    Elections Canada Table 12 candidate strings appear to embed party labels, e.g.

        "Yves-François Blanchet ** Bloc Québécois/Bloc Québécois"
        "Dominique Vien Conservative/Conservateur"

    This parser extracts:
        candidate_name_clean
        elected_marker_detected
        party_label_raw
        party_label_english_or_first
        party_label_french_or_second

    It is intentionally conservative. If no party-like suffix is detected,
    it leaves party fields empty rather than fabricating a party.
    """

    if pd.isna(candidate_string):
        return {
            "candidate_name_clean": "",
            "elected_marker_detected": False,
            "party_label_raw": "",
            "party_label_english_or_first": "",
            "party_label_french_or_second": "",
            "candidate_parse_status": "missing_candidate_string",
        }

    text = str(candidate_string).strip()
    text = re.sub(r"\s+", " ", text)

    elected_marker_detected = "**" in text

    # First handle the common explicit elected-marker separator.
    if "**" in text:
        left, right = text.split("**", 1)
        candidate_name = left.strip()
        party_raw = right.strip()
    else:
        # Fallback: many rows still appear as "Name Party/Parti".
        # We look for the last segment containing a slash and treat it as party-like.
        # This is conservative but useful for non-elected candidates.
        tokens = text.split(" ")
        slash_positions = [idx for idx, token in enumerate(tokens) if "/" in token]

        if slash_positions:
            slash_idx = slash_positions[-1]
            # Walk left while tokens look like part of a party label.
            # This captures labels such as "Bloc Québécois/Bloc Québécois",
            # "Liberal/Libéral", "Conservative/Conservateur".
            start_idx = max(0, slash_idx - 2)
            candidate_name = " ".join(tokens[:start_idx]).strip()
            party_raw = " ".join(tokens[start_idx:]).strip()

            if candidate_name == "":
                candidate_name = text
                party_raw = ""
        else:
            candidate_name = text
            party_raw = ""

    party_first = ""
    party_second = ""

    if party_raw and "/" in party_raw:
        parts = [part.strip() for part in party_raw.split("/", 1)]
        party_first = parts[0]
        party_second = parts[1] if len(parts) > 1 else ""
    elif party_raw:
        party_first = party_raw
        party_second = ""

    return {
        "candidate_name_clean": candidate_name,
        "elected_marker_detected": elected_marker_detected,
        "party_label_raw": party_raw,
        "party_label_english_or_first": party_first,
        "party_label_french_or_second": party_second,
        "candidate_parse_status": "parsed" if candidate_name else "needs_review",
    }


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


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        return str(path)


# -----------------------------
# Validate inputs
# -----------------------------

if not TABLE11_PATH.exists():
    raise FileNotFoundError(f"Missing Table 11 file:\n{TABLE11_PATH}")

if not TABLE12_PATH.exists():
    raise FileNotFoundError(f"Missing Table 12 file:\n{TABLE12_PATH}")


# -----------------------------
# Load tables
# -----------------------------

table11_raw, table11_encoding = read_csv_flex(TABLE11_PATH)
table12_raw, table12_encoding = read_csv_flex(TABLE12_PATH)

table11 = normalize_columns(table11_raw)
table12 = normalize_columns(table12_raw)

print("\nLoaded Elections Canada files")
print("Table 11:", relative(TABLE11_PATH), "rows:", len(table11), "encoding:", table11_encoding)
print("Table 12:", relative(TABLE12_PATH), "rows:", len(table12), "encoding:", table12_encoding)


# -----------------------------
# Detect Table 11 columns
# -----------------------------

table11_cols = list(table11.columns)

t11_province_col = first_existing_column(table11_cols, PROVINCE_CANDIDATES, "Table 11 province")
t11_district_id_col = first_existing_column(table11_cols, DISTRICT_ID_CANDIDATES, "Table 11 district id")
t11_district_name_col = first_existing_column(table11_cols, DISTRICT_NAME_CANDIDATES, "Table 11 district name")
t11_registered_col = first_existing_column(table11_cols, REGISTERED_ELECTORS_CANDIDATES, "Table 11 registered electors")
t11_ballots_cast_col = first_existing_column(table11_cols, BALLOTS_CAST_CANDIDATES, "Table 11 ballots cast")
t11_valid_col = first_existing_column(table11_cols, VALID_BALLOTS_CANDIDATES, "Table 11 valid ballots")
t11_rejected_col = first_existing_column(table11_cols, REJECTED_BALLOTS_CANDIDATES, "Table 11 rejected ballots")
t11_turnout_col = first_existing_column(table11_cols, TURNOUT_CANDIDATES, "Table 11 turnout")


# -----------------------------
# Detect Table 12 columns
# -----------------------------

table12_cols = list(table12.columns)

t12_province_col = first_existing_column(table12_cols, PROVINCE_CANDIDATES, "Table 12 province")
t12_district_id_col = first_existing_column(table12_cols, DISTRICT_ID_CANDIDATES, "Table 12 district id")
t12_district_name_col = first_existing_column(table12_cols, DISTRICT_NAME_CANDIDATES, "Table 12 district name")
t12_candidate_col = first_existing_column(table12_cols, CANDIDATE_CANDIDATES, "Table 12 candidate")
t12_candidate_votes_col = first_existing_column(table12_cols, CANDIDATE_VOTES_CANDIDATES, "Table 12 candidate votes")
t12_candidate_pct_col = first_existing_column(table12_cols, CANDIDATE_VOTE_PCT_CANDIDATES, "Table 12 candidate vote percent")

t12_candidate_residence_col = optional_column(table12_cols, CANDIDATE_RESIDENCE_CANDIDATES)
t12_candidate_occupation_col = optional_column(table12_cols, CANDIDATE_OCCUPATION_CANDIDATES)
t12_majority_col = optional_column(table12_cols, MAJORITY_CANDIDATES)
t12_majority_pct_col = optional_column(table12_cols, MAJORITY_PCT_CANDIDATES)


# -----------------------------
# Filter Québec
# -----------------------------

table11_qc = filter_quebec(table11, t11_province_col)
table12_qc = filter_quebec(table12, t12_province_col)

print("\nQuébec rows")
print("Table 11 Québec rows:", len(table11_qc))
print("Table 12 Québec rows:", len(table12_qc))


# -----------------------------
# Build turnout table from Table 11
# -----------------------------

turnout = pd.DataFrame()

turnout["federal_electoral_district_id"] = (
    table11_qc[t11_district_id_col].astype("string").str.strip()
)

turnout["federal_electoral_district_name"] = (
    table11_qc[t11_district_name_col].astype("string").str.strip()
)

turnout["province"] = table11_qc[t11_province_col].astype("string").str.strip()

turnout["registered_electors"] = clean_numeric(table11_qc[t11_registered_col])
turnout["ballots_cast"] = clean_numeric(table11_qc[t11_ballots_cast_col])
turnout["valid_ballots"] = clean_numeric(table11_qc[t11_valid_col])
turnout["rejected_ballots"] = clean_numeric(table11_qc[t11_rejected_col])
turnout["published_voter_turnout_pct"] = clean_numeric(table11_qc[t11_turnout_col])

turnout["computed_ballots_cast_from_valid_plus_rejected"] = (
    turnout["valid_ballots"] + turnout["rejected_ballots"]
)

turnout["computed_voter_turnout_pct_from_ballots_cast"] = (
    100 * turnout["ballots_cast"] / turnout["registered_electors"]
)

turnout["computed_voter_turnout_pct_from_valid_plus_rejected"] = (
    100
    * turnout["computed_ballots_cast_from_valid_plus_rejected"]
    / turnout["registered_electors"]
)

turnout["voter_turnout_pct_federal_2021"] = turnout["published_voter_turnout_pct"]
turnout["voter_turnout_pct_federal_2021_source"] = "published_table11"

missing_turnout = turnout["voter_turnout_pct_federal_2021"].isna()
turnout.loc[missing_turnout, "voter_turnout_pct_federal_2021"] = turnout.loc[
    missing_turnout,
    "computed_voter_turnout_pct_from_ballots_cast",
]
turnout.loc[missing_turnout, "voter_turnout_pct_federal_2021_source"] = "computed_from_ballots_cast"

turnout["table11_source_file"] = relative(TABLE11_PATH)


# -----------------------------
# Build candidate long table from Table 12
# -----------------------------

candidate_long = pd.DataFrame()

candidate_long["federal_electoral_district_id"] = (
    table12_qc[t12_district_id_col].astype("string").str.strip()
)

candidate_long["federal_electoral_district_name"] = (
    table12_qc[t12_district_name_col].astype("string").str.strip()
)

candidate_long["province"] = table12_qc[t12_province_col].astype("string").str.strip()

candidate_long["candidate_raw"] = (
    table12_qc[t12_candidate_col].astype("string").str.strip()
)

candidate_long["candidate_votes"] = clean_numeric(table12_qc[t12_candidate_votes_col])
candidate_long["candidate_vote_share_pct_published"] = clean_numeric(table12_qc[t12_candidate_pct_col])

if t12_candidate_residence_col is not None:
    candidate_long["candidate_residence"] = table12_qc[t12_candidate_residence_col].astype("string").str.strip()
else:
    candidate_long["candidate_residence"] = pd.NA

if t12_candidate_occupation_col is not None:
    candidate_long["candidate_occupation"] = table12_qc[t12_candidate_occupation_col].astype("string").str.strip()
else:
    candidate_long["candidate_occupation"] = pd.NA

if t12_majority_col is not None:
    candidate_long["majority"] = clean_numeric(table12_qc[t12_majority_col])
else:
    candidate_long["majority"] = pd.NA

if t12_majority_pct_col is not None:
    candidate_long["majority_pct"] = clean_numeric(table12_qc[t12_majority_pct_col])
else:
    candidate_long["majority_pct"] = pd.NA

parsed_candidate = candidate_long["candidate_raw"].apply(split_candidate_party).apply(pd.Series)
candidate_long = pd.concat([candidate_long, parsed_candidate], axis=1)

district_vote_totals = (
    candidate_long
    .groupby(
        [
            "federal_electoral_district_id",
            "federal_electoral_district_name",
        ],
        dropna=False,
    )["candidate_votes"]
    .sum(min_count=1)
    .reset_index()
    .rename(columns={"candidate_votes": "district_valid_votes_from_candidate_sum"})
)

candidate_long = candidate_long.merge(
    district_vote_totals,
    on=[
        "federal_electoral_district_id",
        "federal_electoral_district_name",
    ],
    how="left",
    validate="many_to_one",
)

candidate_long["candidate_vote_share_pct_computed"] = (
    100
    * candidate_long["candidate_votes"]
    / candidate_long["district_valid_votes_from_candidate_sum"]
)

candidate_long["candidate_vote_share_pct_best"] = candidate_long["candidate_vote_share_pct_published"]
candidate_long["candidate_vote_share_pct_best_source"] = "published_table12"

missing_candidate_pct = candidate_long["candidate_vote_share_pct_best"].isna()
candidate_long.loc[missing_candidate_pct, "candidate_vote_share_pct_best"] = candidate_long.loc[
    missing_candidate_pct,
    "candidate_vote_share_pct_computed",
]
candidate_long.loc[missing_candidate_pct, "candidate_vote_share_pct_best_source"] = "computed_from_candidate_sum"

candidate_long["table12_source_file"] = relative(TABLE12_PATH)


# -----------------------------
# Build leading candidate table
# -----------------------------

leading_candidate = (
    candidate_long
    .sort_values(
        [
            "federal_electoral_district_id",
            "candidate_votes",
        ],
        ascending=[True, False],
    )
    .groupby(
        [
            "federal_electoral_district_id",
            "federal_electoral_district_name",
        ],
        dropna=False,
    )
    .head(1)
    .copy()
)

leading_candidate = leading_candidate.rename(
    columns={
        "candidate_raw": "leading_candidate_raw",
        "candidate_name_clean": "leading_candidate_name",
        "candidate_residence": "leading_candidate_residence",
        "candidate_occupation": "leading_candidate_occupation",
        "candidate_votes": "leading_candidate_votes",
        "candidate_vote_share_pct_published": "leading_candidate_vote_share_pct_published",
        "candidate_vote_share_pct_computed": "leading_candidate_vote_share_pct_computed",
        "candidate_vote_share_pct_best": "pct_vote_leading_candidate_federal_2021",
        "candidate_vote_share_pct_best_source": "pct_vote_leading_candidate_federal_2021_source",
        "party_label_raw": "leading_party_label_raw",
        "party_label_english_or_first": "leading_party_label_english_or_first",
        "party_label_french_or_second": "leading_party_label_french_or_second",
        "elected_marker_detected": "leading_candidate_elected_marker_detected",
        "candidate_parse_status": "leading_candidate_parse_status",
    }
)

leading_candidate_keep = [
    "federal_electoral_district_id",
    "federal_electoral_district_name",
    "leading_candidate_raw",
    "leading_candidate_name",
    "leading_candidate_elected_marker_detected",
    "leading_party_label_raw",
    "leading_party_label_english_or_first",
    "leading_party_label_french_or_second",
    "leading_candidate_residence",
    "leading_candidate_occupation",
    "leading_candidate_votes",
    "district_valid_votes_from_candidate_sum",
    "leading_candidate_vote_share_pct_published",
    "leading_candidate_vote_share_pct_computed",
    "pct_vote_leading_candidate_federal_2021",
    "pct_vote_leading_candidate_federal_2021_source",
    "majority",
    "majority_pct",
    "leading_candidate_parse_status",
    "table12_source_file",
]

leading_candidate = leading_candidate[leading_candidate_keep].copy()


# -----------------------------
# Build leading party-like proxy
# -----------------------------
#
# Table 12 does not provide a separate party column, but the party label appears
# embedded in Candidate/Candidat. We expose this as a parsed candidate-party
# label, not as an official separate party field.
#
# Since federal districts normally have one candidate per party, the winning
# candidate's parsed party label is the leading party-like label.
# -----------------------------

leading_candidate["pct_vote_leading_party_federal_2021"] = leading_candidate[
    "pct_vote_leading_candidate_federal_2021"
]

leading_candidate["pct_vote_leading_party_federal_2021_source"] = (
    "parsed_from_leading_candidate_label_table12"
)

leading_candidate["pct_vote_leading_party_federal_2021_method_note"] = (
    "Table 12 does not include a separate party column. The party label is parsed "
    "from the Candidate/Candidat field and therefore should be treated as a "
    "parsed party-label proxy."
)


# -----------------------------
# Merge turnout and leading candidate
# -----------------------------

clean = turnout.merge(
    leading_candidate,
    on=[
        "federal_electoral_district_id",
        "federal_electoral_district_name",
    ],
    how="outer",
    validate="one_to_one",
)

# Preserve province from Table 11 if possible.
if "province_x" in clean.columns and "province_y" in clean.columns:
    clean["province"] = clean["province_x"].combine_first(clean["province_y"])
    clean = clean.drop(columns=["province_x", "province_y"])

clean["election_year"] = 2021
clean["election_name"] = "44th Canadian federal general election"
clean["source_organization"] = "Elections Canada"
clean["source_geography"] = "Federal electoral district / circonscription fédérale"
clean["target_future_geography"] = "Statistics Canada census division after spatial allocation"

clean["vote_proxy_complete"] = (
    clean["voter_turnout_pct_federal_2021"].notna()
    & clean["pct_vote_leading_candidate_federal_2021"].notna()
)

clean["so_vi_variable_target"] = "PCTVOTE92"
clean["so_vi_proxy_recommended"] = "pct_vote_leading_party_federal_2021"
clean["so_vi_proxy_alternative"] = "voter_turnout_pct_federal_2021"

clean["method_note"] = (
    "Federal electoral district vote proxy. For the SoVI PCTVOTE92 adaptation, "
    "pct_vote_leading_party_federal_2021 is derived from the winning candidate's "
    "vote share and a parsed party label embedded in Elections Canada Table 12. "
    "voter_turnout_pct_federal_2021 is retained as a civic-participation alternative. "
    "This table is not yet census-division-level; spatial allocation is required."
)


# -----------------------------
# Validation
# -----------------------------

expected_qc_districts = 78

if len(clean) != expected_qc_districts:
    raise ValueError(
        f"Expected {expected_qc_districts} Québec federal districts, got {len(clean)}."
    )

if clean["federal_electoral_district_id"].duplicated().any():
    dupes = clean[clean["federal_electoral_district_id"].duplicated(keep=False)]
    raise ValueError(
        "Duplicate federal_electoral_district_id values in clean output:\n"
        + dupes[
            [
                "federal_electoral_district_id",
                "federal_electoral_district_name",
            ]
        ].to_string(index=False)
    )

required_complete_cols = [
    "voter_turnout_pct_federal_2021",
    "pct_vote_leading_candidate_federal_2021",
    "pct_vote_leading_party_federal_2021",
    "registered_electors",
    "valid_ballots",
]

for col in required_complete_cols:
    missing = int(clean[col].isna().sum())
    if missing != 0:
        raise ValueError(f"Unexpected missing values in {col}: {missing}")

# Verify Table 11 valid ballots and Table 12 candidate-vote sums agree.
clean["valid_ballots_minus_candidate_sum"] = (
    clean["valid_ballots"] - clean["district_valid_votes_from_candidate_sum"]
)

max_abs_vote_difference = clean["valid_ballots_minus_candidate_sum"].abs().max(skipna=True)

if max_abs_vote_difference != 0:
    print("\nWARNING: Some district valid-ballot totals differ from candidate-vote sums.")
    print("Maximum absolute difference:", max_abs_vote_difference)


# -----------------------------
# Metadata
# -----------------------------

metadata_rows = [
    {
        "variable": "voter_turnout_pct_federal_2021",
        "description": "Published voter turnout percentage by federal electoral district",
        "source_table": "Elections Canada Table 11",
        "source_column": t11_turnout_col,
        "unit": "percent",
        "role": "civic_participation_alternative_proxy",
        "notes": "Useful as an alternative to the leading-party/leading-candidate vote-share proxy.",
    },
    {
        "variable": "pct_vote_leading_candidate_federal_2021",
        "description": "Vote share of the leading candidate by federal electoral district",
        "source_table": "Elections Canada Table 12",
        "source_column": t12_candidate_pct_col,
        "unit": "percent",
        "role": "direct_candidate_level_vote_share_proxy",
        "notes": "Closest directly observed district-level quantity from Table 12.",
    },
    {
        "variable": "pct_vote_leading_party_federal_2021",
        "description": "Parsed leading-party vote-share proxy by federal electoral district",
        "source_table": "Elections Canada Table 12",
        "source_column": t12_candidate_col,
        "unit": "percent",
        "role": "recommended_sovi_pctvote_proxy",
        "notes": (
            "Uses the leading candidate's vote share and parses the party label from "
            "Candidate/Candidat. This is not based on a separate explicit party column."
        ),
    },
]

metadata = pd.DataFrame(metadata_rows)


# -----------------------------
# Summary
# -----------------------------

summary_rows = [
    {"metric": "table11_file", "value": relative(TABLE11_PATH)},
    {"metric": "table12_file", "value": relative(TABLE12_PATH)},
    {"metric": "table11_encoding", "value": table11_encoding},
    {"metric": "table12_encoding", "value": table12_encoding},
    {"metric": "table11_rows", "value": len(table11)},
    {"metric": "table12_rows", "value": len(table12)},
    {"metric": "table11_quebec_rows", "value": len(table11_qc)},
    {"metric": "table12_quebec_rows", "value": len(table12_qc)},
    {"metric": "clean_rows", "value": len(clean)},
    {"metric": "unique_federal_electoral_districts", "value": clean["federal_electoral_district_id"].nunique()},
    {"metric": "all_vote_proxy_rows_complete", "value": bool(clean["vote_proxy_complete"].all())},
    {"metric": "max_abs_valid_ballots_minus_candidate_sum", "value": max_abs_vote_difference},
    {"metric": "recommended_sovi_proxy", "value": "pct_vote_leading_party_federal_2021"},
    {"metric": "alternative_proxy", "value": "voter_turnout_pct_federal_2021"},
]

for variable in [
    "voter_turnout_pct_federal_2021",
    "pct_vote_leading_candidate_federal_2021",
    "pct_vote_leading_party_federal_2021",
    "majority_pct",
]:
    stats = summarize_numeric(clean[variable])
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
            "Locate or download 2013 Representation Order federal electoral district "
            "boundary polygons, then spatially allocate federal district vote proxies "
            "to Québec census divisions."
        ),
    }
)

summary = pd.DataFrame(summary_rows)


# -----------------------------
# Save
# -----------------------------

clean.to_csv(OUTPUT_CLEAN, index=False, encoding="utf-8")
candidate_long.to_csv(OUTPUT_CANDIDATE_LONG, index=False, encoding="utf-8")
metadata.to_csv(OUTPUT_METADATA, index=False, encoding="utf-8")
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CLEAN QUÉBEC FEDERAL DISTRICT VOTE PROXY 2021")
print("=" * 72)

print("\nClean table:")
print("Rows:", len(clean))
print("Unique districts:", clean["federal_electoral_district_id"].nunique())
print("Complete vote proxy rows:", bool(clean["vote_proxy_complete"].all()))

print("\nSource columns:")
print("Table 11 district id:", t11_district_id_col)
print("Table 11 district name:", t11_district_name_col)
print("Table 11 turnout:", t11_turnout_col)
print("Table 12 candidate:", t12_candidate_col)
print("Table 12 candidate votes:", t12_candidate_votes_col)
print("Table 12 candidate pct:", t12_candidate_pct_col)

print("\nVariable summaries:")
for variable in [
    "voter_turnout_pct_federal_2021",
    "pct_vote_leading_candidate_federal_2021",
    "pct_vote_leading_party_federal_2021",
]:
    print("\n", variable)
    print(clean[variable].describe())

print("\nValid-ballot consistency check:")
print("Max abs(valid_ballots - candidate_vote_sum):", max_abs_vote_difference)

print("\nPreview:")
preview_cols = [
    "federal_electoral_district_id",
    "federal_electoral_district_name",
    "voter_turnout_pct_federal_2021",
    "leading_candidate_name",
    "leading_party_label_english_or_first",
    "pct_vote_leading_candidate_federal_2021",
    "pct_vote_leading_party_federal_2021",
    "majority_pct",
]
print(clean[preview_cols].head(25).to_string(index=False))

print("\nSaved:")
print(OUTPUT_CLEAN)
print(OUTPUT_CANDIDATE_LONG)
print(OUTPUT_METADATA)
print(OUTPUT_SUMMARY)

print("\nDone.")