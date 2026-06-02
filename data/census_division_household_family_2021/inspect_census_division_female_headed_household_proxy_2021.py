from pathlib import Path
import re
import pandas as pd


# ============================================================
# Inspect Census Division Female-Headed Household Proxy 2021
# ============================================================
#
# Purpose:
#   Run a targeted inspection for possible Canadian Census Profile proxies
#   for the original SoVI variable:
#
#       PCTF_HH90 -> pct_female_headed_households
#
# Why this exists:
#   The broader household/family inspection found:
#
#       AVGPERHH -> avg_people_per_household
#
#   but did not find a usable female-headed-household proxy.
#
# This script searches broadly for rows involving:
#
#   - female
#   - women+
#   - woman
#   - household maintainer
#   - primary household maintainer
#   - lone-parent family
#   - female parent
#   - parent in a lone-parent family
#
# It also captures possible denominators:
#
#   - total private households
#   - total census families
#   - total lone-parent families
#   - total household maintainers
#
# Run from data/:
#   python census_division_household_family_2021/inspect_census_division_female_headed_household_proxy_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_household_family_2021"
OUTPUT_DIR = SECTION_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = (
    DATA_DIR
    / "census_profile_census_division_2021"
    / "raw"
    / "98-401-X2021004_English_CSV_data.csv"
)

BASE_CD_CANDIDATES = [
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.csv",
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.parquet",
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.geojson",
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.gpkg",
]

OUTPUT_FILE_INVENTORY = OUTPUT_DIR / "female_headed_household_proxy_file_inventory_2021.csv"
OUTPUT_CD_INVENTORY = OUTPUT_DIR / "female_headed_household_proxy_quebec_cd_inventory_2021.csv"
OUTPUT_BASE_JOIN = OUTPUT_DIR / "female_headed_household_proxy_base_join_check_2021.csv"
OUTPUT_ALL_MATCHES = OUTPUT_DIR / "female_headed_household_proxy_all_keyword_matches_2021.csv"
OUTPUT_CHARACTERISTIC_SUMMARY = OUTPUT_DIR / "female_headed_household_proxy_characteristic_summary_2021.csv"
OUTPUT_CANDIDATE_CLASSIFICATION = OUTPUT_DIR / "female_headed_household_proxy_candidate_classification_2021.csv"
OUTPUT_TARGET_VALUES_LONG = OUTPUT_DIR / "female_headed_household_proxy_target_values_long_2021.csv"
OUTPUT_TARGET_VALUES_WIDE = OUTPUT_DIR / "female_headed_household_proxy_target_values_wide_2021.csv"
OUTPUT_DERIVED_FORMULA_AUDIT = OUTPUT_DIR / "female_headed_household_proxy_derived_formula_audit_2021.csv"
OUTPUT_SYMBOL_COUNTS = OUTPUT_DIR / "female_headed_household_proxy_symbol_counts_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "female_headed_household_proxy_inspection_summary_2021.csv"


# -----------------------------
# Encoding config
# -----------------------------

RAW_ENCODING_CANDIDATES = [
    "cp1252",
    "latin1",
    "utf-8-sig",
    "utf-8",
]

CLEANED_FILE_ENCODING_CANDIDATES = [
    "utf-8",
    "utf-8-sig",
    "cp1252",
    "latin1",
]


# -----------------------------
# Other config
# -----------------------------

CHUNK_SIZE = 200_000
QUEBEC_CD_DGUID_PREFIX = "2021A000324"

REQUIRED_COLUMNS = [
    "CENSUS_YEAR",
    "DGUID",
    "ALT_GEO_CODE",
    "GEO_LEVEL",
    "GEO_NAME",
    "CHARACTERISTIC_ID",
    "CHARACTERISTIC_NAME",
    "CHARACTERISTIC_NOTE",
    "C1_COUNT_TOTAL",
    "SYMBOL",
    "C2_COUNT_MEN+",
    "SYMBOL.1",
    "C3_COUNT_WOMEN+",
    "SYMBOL.2",
    "C10_RATE_TOTAL",
    "SYMBOL.3",
    "C11_RATE_MEN+",
    "SYMBOL.4",
    "C12_RATE_WOMEN+",
    "SYMBOL.5",
    "TNR_SF",
    "TNR_LF",
    "DATA_QUALITY_FLAG",
]

VALUE_TO_SYMBOL_COLUMN = {
    "C1_COUNT_TOTAL": "SYMBOL",
    "C2_COUNT_MEN+": "SYMBOL.1",
    "C3_COUNT_WOMEN+": "SYMBOL.2",
    "C10_RATE_TOTAL": "SYMBOL.3",
    "C11_RATE_MEN+": "SYMBOL.4",
    "C12_RATE_WOMEN+": "SYMBOL.5",
}

# Wider than the first household-family inspection.
# This is intentional because family/maintainer rows may live far from the
# simple household-size block.
TARGET_KEYWORDS_ANY = [
    "female",
    "women+",
    "woman",
    "women",
    "maintainer",
    "household maintainer",
    "primary household maintainer",
    "lone-parent",
    "lone parent",
    "one-parent",
    "single-parent",
    "female parent",
    "male parent",
    "parent in a lone-parent",
    "parent in a lone parent",
    "census families",
    "census family",
    "private households",
    "household size",
    "household type",
    "family structure",
]

EXCLUDE_KEYWORDS_ANY = [
    "language",
    "knowledge of",
    "mother tongue",
    "immigration",
    "citizenship",
    "ethnic",
    "visible minority",
    "indigenous",
    "aboriginal",
    "mobility",
    "commuting",
    "journey to work",
    "place of work",
    "income",
    "low-income",
    "low income",
    "shelter",
    "rent",
    "owner",
    "renter",
    "tenant",
    "housing suitability",
    "dwelling condition",
    "condominium",
]

# Labels for candidate classification.
# These are not hard commitments; they help us inspect likely candidates.
CLASSIFICATION_RULES = [
    {
        "candidate_type": "female_parent_lone_parent_family_count_or_rate",
        "include_all": ["female", "parent"],
        "include_any": ["lone-parent", "lone parent", "one-parent", "census famil"],
        "exclude_any": EXCLUDE_KEYWORDS_ANY,
        "priority": 1,
    },
    {
        "candidate_type": "female_or_women_household_maintainer",
        "include_all": ["maintainer"],
        "include_any": ["female", "women+", "woman", "women"],
        "exclude_any": EXCLUDE_KEYWORDS_ANY,
        "priority": 2,
    },
    {
        "candidate_type": "total_lone_parent_family_denominator",
        "include_all": ["lone"],
        "include_any": ["total", "parent"],
        "exclude_any": EXCLUDE_KEYWORDS_ANY + ["female parent", "male parent"],
        "priority": 3,
    },
    {
        "candidate_type": "total_private_household_denominator",
        "include_all": ["private", "household"],
        "include_any": ["total"],
        "exclude_any": EXCLUDE_KEYWORDS_ANY,
        "priority": 4,
    },
    {
        "candidate_type": "total_census_family_denominator",
        "include_all": ["census famil"],
        "include_any": ["total"],
        "exclude_any": EXCLUDE_KEYWORDS_ANY + ["children", "couple", "lone-parent", "lone parent"],
        "priority": 5,
    },
]


# -----------------------------
# Helpers
# -----------------------------

def detect_encoding(path: Path, encodings: list[str]) -> str:
    last_error = None

    for encoding in encodings:
        try:
            pd.read_csv(
                path,
                nrows=5000,
                dtype=str,
                encoding=encoding,
                low_memory=False,
            )
            return encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not decode {path} with encodings {encodings}. Last error: {last_error}",
    )


def read_csv_with_encodings(path: Path, encodings: list[str], **kwargs) -> pd.DataFrame:
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                dtype=str,
                encoding=encoding,
                low_memory=False,
                **kwargs,
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error


def read_cleaned_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return read_csv_with_encodings(path, CLEANED_FILE_ENCODING_CANDIDATES)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix in [".geojson", ".gpkg", ".shp"]:
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError(f"geopandas is required to read spatial file {path}") from exc
        return gpd.read_file(path)

    raise ValueError(f"Unsupported file type: {path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_quebec_cd_dguid(series: pd.Series) -> pd.Series:
    return clean_text(series).str.startswith(QUEBEC_CD_DGUID_PREFIX, na=False)


def find_base_cd_frame() -> Path | None:
    for path in BASE_CD_CANDIDATES:
        if path.exists():
            return path
    return None


def require_columns(columns: list[str], required: list[str], label: str) -> None:
    missing = [col for col in required if col not in columns]

    if missing:
        raise ValueError(
            f"Missing required columns in {label}:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(columns)
        )


def keyword_match(row: pd.Series) -> bool:
    name = normalize_text(row.get("CHARACTERISTIC_NAME", ""))
    note = normalize_text(row.get("CHARACTERISTIC_NOTE", ""))
    text = f"{name} {note}"

    include = any(keyword.lower() in text for keyword in TARGET_KEYWORDS_ANY)
    exclude = any(keyword.lower() in text for keyword in EXCLUDE_KEYWORDS_ANY)

    return include and not exclude


def classify_characteristic(name: object) -> tuple[str, int, str]:
    normalized = normalize_text(name)

    matched_types = []

    for rule in CLASSIFICATION_RULES:
        include_all_ok = all(term.lower() in normalized for term in rule["include_all"])
        include_any_ok = any(term.lower() in normalized for term in rule["include_any"])
        exclude_ok = not any(term.lower() in normalized for term in rule["exclude_any"])

        if include_all_ok and include_any_ok and exclude_ok:
            matched_types.append(
                (
                    rule["candidate_type"],
                    rule["priority"],
                    f"matched include_all={rule['include_all']} include_any={rule['include_any']}",
                )
            )

    if not matched_types:
        return "other_keyword_match", 999, "keyword match, not classified as a primary proxy candidate"

    matched_types = sorted(matched_types, key=lambda x: x[1])
    return matched_types[0]


def summarize_symbol_counts(series: pd.Series) -> str:
    counts = series.fillna("").astype(str).value_counts(dropna=False)

    if counts.empty:
        return ""

    return "; ".join(
        f"{symbol}:{int(count)}"
        for symbol, count in counts.items()
    )


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


def build_file_inventory(raw_encoding: str) -> pd.DataFrame:
    rows = []

    rows.append(
        {
            "relative_path": str(RAW_CSV.relative_to(DATA_DIR)),
            "exists": RAW_CSV.exists(),
            "size_mb": round(RAW_CSV.stat().st_size / (1024 * 1024), 3) if RAW_CSV.exists() else None,
            "encoding_selected": raw_encoding if RAW_CSV.exists() else "",
            "role": "raw_census_profile_cd",
        }
    )

    base_path = find_base_cd_frame()
    rows.append(
        {
            "relative_path": str(base_path.relative_to(DATA_DIR)) if base_path is not None else "",
            "exists": base_path is not None,
            "size_mb": round(base_path.stat().st_size / (1024 * 1024), 3) if base_path is not None else None,
            "encoding_selected": "",
            "role": "clean_cd_spatial_population_base",
        }
    )

    return pd.DataFrame(rows)


# -----------------------------
# Initial checks
# -----------------------------

if not RAW_CSV.exists():
    raise FileNotFoundError(f"Raw Census Profile CD CSV not found:\n{RAW_CSV}")

raw_encoding = detect_encoding(RAW_CSV, RAW_ENCODING_CANDIDATES)

header = pd.read_csv(
    RAW_CSV,
    nrows=0,
    dtype=str,
    encoding=raw_encoding,
    low_memory=False,
)

columns = list(header.columns)
require_columns(columns, REQUIRED_COLUMNS, "Census Profile CD raw CSV")

print("\nTargeted female-headed-household proxy inspection")
print("Raw file:", RAW_CSV.relative_to(DATA_DIR))
print("Raw encoding selected:", raw_encoding)

file_inventory = build_file_inventory(raw_encoding)
file_inventory.to_csv(OUTPUT_FILE_INVENTORY, index=False)


# -----------------------------
# Load base CD frame
# -----------------------------

base_path = find_base_cd_frame()
base = None

if base_path is not None:
    base = read_cleaned_table(base_path)
    base = normalize_columns(base)

    if "census_division_dguid" not in base.columns:
        print("\nWARNING: Base CD frame exists but lacks census_division_dguid.")
        base = None
    else:
        base = base.copy()
        base["census_division_dguid"] = clean_text(base["census_division_dguid"])

        print("\nLoaded existing CD base frame")
        print("Path:", base_path.relative_to(DATA_DIR))
        print("Rows:", len(base))
else:
    print("\nWARNING: No existing CD base frame found.")


# -----------------------------
# Chunked scan
# -----------------------------

total_rows = 0
quebec_cd_rows = 0
quebec_cd_inventory = {}
match_rows = []

print("\nScanning raw Census Profile file...")

for chunk_idx, chunk in enumerate(
    pd.read_csv(
        RAW_CSV,
        dtype=str,
        encoding=raw_encoding,
        low_memory=False,
        chunksize=CHUNK_SIZE,
        usecols=REQUIRED_COLUMNS,
    ),
    start=1,
):
    total_rows += len(chunk)
    chunk = normalize_columns(chunk)

    chunk["GEO_LEVEL_NORM"] = clean_text(chunk["GEO_LEVEL"]).str.lower()

    is_cd = chunk["GEO_LEVEL_NORM"].eq("census division")
    is_qc_cd = is_cd & is_quebec_cd_dguid(chunk["DGUID"])

    qc = chunk.loc[is_qc_cd].copy()
    quebec_cd_rows += len(qc)

    if qc.empty:
        if chunk_idx % 10 == 0:
            print(f"  Processed chunks: {chunk_idx}, rows so far: {total_rows}")
        continue

    qc["DGUID"] = clean_text(qc["DGUID"])
    qc["ALT_GEO_CODE"] = clean_text(qc["ALT_GEO_CODE"])
    qc["GEO_NAME"] = clean_text(qc["GEO_NAME"])
    qc["CHARACTERISTIC_ID"] = clean_text(qc["CHARACTERISTIC_ID"])

    geo_cols = [
        "DGUID",
        "ALT_GEO_CODE",
        "GEO_NAME",
        "GEO_LEVEL",
        "TNR_SF",
        "TNR_LF",
        "DATA_QUALITY_FLAG",
    ]

    for _, row in qc[geo_cols].drop_duplicates(subset=["DGUID"]).iterrows():
        dguid = str(row["DGUID"])
        quebec_cd_inventory[dguid] = {
            "census_division_dguid": dguid,
            "census_division_code": row.get("ALT_GEO_CODE", ""),
            "census_division_name_profile": row.get("GEO_NAME", ""),
            "GEO_LEVEL": row.get("GEO_LEVEL", ""),
            "TNR_SF": row.get("TNR_SF", ""),
            "TNR_LF": row.get("TNR_LF", ""),
            "DATA_QUALITY_FLAG": row.get("DATA_QUALITY_FLAG", ""),
        }

    matches = qc[qc.apply(keyword_match, axis=1)].copy()

    if not matches.empty:
        match_rows.append(matches)

    if chunk_idx % 10 == 0:
        print(f"  Processed chunks: {chunk_idx}, rows so far: {total_rows}")


# -----------------------------
# Build inventory and base join
# -----------------------------

cd_inventory = (
    pd.DataFrame(quebec_cd_inventory.values())
    .sort_values("census_division_dguid")
    .reset_index(drop=True)
    if quebec_cd_inventory
    else pd.DataFrame(
        columns=[
            "census_division_dguid",
            "census_division_code",
            "census_division_name_profile",
            "GEO_LEVEL",
            "TNR_SF",
            "TNR_LF",
            "DATA_QUALITY_FLAG",
        ]
    )
)

cd_inventory.to_csv(OUTPUT_CD_INVENTORY, index=False)

if base is not None and not cd_inventory.empty:
    base_join = base[
        [
            col for col in [
                "census_division_code",
                "census_division_dguid",
                "census_division_name",
                "population_total_2021",
                "land_area_km2",
            ]
            if col in base.columns
        ]
    ].copy()

    base_join = base_join.merge(
        cd_inventory[
            [
                "census_division_dguid",
                "census_division_code",
                "census_division_name_profile",
                "TNR_SF",
                "TNR_LF",
                "DATA_QUALITY_FLAG",
            ]
        ].rename(columns={"census_division_code": "profile_census_division_code"}),
        on="census_division_dguid",
        how="outer",
        indicator=True,
    )

    base_join["join_status"] = base_join["_merge"]
    base_join = base_join.drop(columns="_merge")
else:
    base_join = pd.DataFrame()

base_join.to_csv(OUTPUT_BASE_JOIN, index=False)


# -----------------------------
# Analyze matches
# -----------------------------

all_matches = (
    pd.concat(match_rows, ignore_index=True)
    if match_rows
    else pd.DataFrame(columns=REQUIRED_COLUMNS)
)

if not all_matches.empty:
    all_matches["candidate_type"], all_matches["candidate_priority"], all_matches["candidate_reason"] = zip(
        *all_matches["CHARACTERISTIC_NAME"].map(classify_characteristic)
    )
else:
    all_matches["candidate_type"] = pd.Series(dtype=str)
    all_matches["candidate_priority"] = pd.Series(dtype=int)
    all_matches["candidate_reason"] = pd.Series(dtype=str)

all_matches.to_csv(OUTPUT_ALL_MATCHES, index=False)


# -----------------------------
# Characteristic summary
# -----------------------------

characteristic_rows = []

if not all_matches.empty:
    all_matches["CHARACTERISTIC_ID"] = clean_text(all_matches["CHARACTERISTIC_ID"])

    group_cols = [
        "candidate_type",
        "candidate_priority",
        "candidate_reason",
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "CHARACTERISTIC_NOTE",
    ]

    for keys, group in all_matches.groupby(group_cols, dropna=False):
        key_map = dict(zip(group_cols, keys))

        row = {
            "candidate_type": key_map.get("candidate_type", ""),
            "candidate_priority": key_map.get("candidate_priority", ""),
            "candidate_reason": key_map.get("candidate_reason", ""),
            "CHARACTERISTIC_ID": key_map.get("CHARACTERISTIC_ID", ""),
            "CHARACTERISTIC_NAME": key_map.get("CHARACTERISTIC_NAME", ""),
            "CHARACTERISTIC_NOTE": key_map.get("CHARACTERISTIC_NOTE", ""),
            "n_rows": len(group),
            "n_unique_quebec_cds": group["DGUID"].nunique(),
            "coverage_is_98_cds": group["DGUID"].nunique() == 98,
        }

        for value_col in [
            "C1_COUNT_TOTAL",
            "C2_COUNT_MEN+",
            "C3_COUNT_WOMEN+",
            "C10_RATE_TOTAL",
            "C11_RATE_MEN+",
            "C12_RATE_WOMEN+",
        ]:
            summary = numeric_summary(group[value_col])
            prefix = (
                value_col
                .lower()
                .replace("+", "plus")
                .replace(" ", "_")
                .replace("-", "_")
            )

            row[f"{prefix}_non_missing"] = summary["non_missing"]
            row[f"{prefix}_missing"] = summary["missing"]
            row[f"{prefix}_min"] = summary["min"]
            row[f"{prefix}_max"] = summary["max"]
            row[f"{prefix}_mean"] = summary["mean"]
            row[f"{prefix}_median"] = summary["median"]

            symbol_col = VALUE_TO_SYMBOL_COLUMN.get(value_col)
            if symbol_col in group.columns:
                row[f"{prefix}_symbol_counts"] = summarize_symbol_counts(group[symbol_col])

        characteristic_rows.append(row)

characteristic_summary = (
    pd.DataFrame(characteristic_rows)
    .sort_values(
        ["candidate_priority", "CHARACTERISTIC_ID"],
        key=lambda s: pd.to_numeric(s, errors="coerce"),
    )
    .reset_index(drop=True)
    if characteristic_rows
    else pd.DataFrame()
)

characteristic_summary.to_csv(OUTPUT_CHARACTERISTIC_SUMMARY, index=False)


# -----------------------------
# Candidate classification
# -----------------------------

if not characteristic_summary.empty:
    candidate_classification = characteristic_summary[
        [
            "candidate_type",
            "candidate_priority",
            "CHARACTERISTIC_ID",
            "CHARACTERISTIC_NAME",
            "n_unique_quebec_cds",
            "coverage_is_98_cds",
            "c1_count_total_non_missing",
            "c1_count_total_min",
            "c1_count_total_max",
            "c1_count_total_mean",
            "c10_rate_total_non_missing",
            "c10_rate_total_min",
            "c10_rate_total_max",
            "c10_rate_total_mean",
        ]
    ].copy()
else:
    candidate_classification = pd.DataFrame()

candidate_classification.to_csv(OUTPUT_CANDIDATE_CLASSIFICATION, index=False)


# -----------------------------
# Target values long / wide
# -----------------------------

# Keep only the most relevant candidate types for easier manual review.
PRIMARY_CANDIDATE_TYPES = {
    "female_parent_lone_parent_family_count_or_rate",
    "female_or_women_household_maintainer",
    "total_lone_parent_family_denominator",
    "total_private_household_denominator",
    "total_census_family_denominator",
}

target_values_long_rows = []
symbol_count_rows = []

if not all_matches.empty:
    primary = all_matches[all_matches["candidate_type"].isin(PRIMARY_CANDIDATE_TYPES)].copy()

    for _, row in primary.iterrows():
        candidate_type = row.get("candidate_type", "")
        char_id = row.get("CHARACTERISTIC_ID", "")
        char_name = row.get("CHARACTERISTIC_NAME", "")

        # Store both count and rate columns where available, because we do not
        # yet know which formula will be best.
        for value_col, symbol_col in [
            ("C1_COUNT_TOTAL", "SYMBOL"),
            ("C10_RATE_TOTAL", "SYMBOL.3"),
        ]:
            value_numeric = clean_numeric(pd.Series([row.get(value_col, "")])).iloc[0]

            if pd.isna(value_numeric):
                continue

            target_values_long_rows.append(
                {
                    "candidate_type": candidate_type,
                    "candidate_priority": row.get("candidate_priority", ""),
                    "census_division_dguid": row.get("DGUID", ""),
                    "census_division_code": row.get("ALT_GEO_CODE", ""),
                    "census_division_name_profile": row.get("GEO_NAME", ""),
                    "CHARACTERISTIC_ID": char_id,
                    "CHARACTERISTIC_NAME": char_name,
                    "value_column": value_col,
                    "value_numeric": value_numeric,
                    "symbol_column": symbol_col,
                    "symbol": row.get(symbol_col, ""),
                    "TNR_SF": row.get("TNR_SF", ""),
                    "TNR_LF": row.get("TNR_LF", ""),
                    "DATA_QUALITY_FLAG": row.get("DATA_QUALITY_FLAG", ""),
                }
            )

    for group_cols, group in primary.groupby(
        ["candidate_type", "CHARACTERISTIC_ID", "CHARACTERISTIC_NAME"], dropna=False
    ):
        candidate_type, char_id, char_name = group_cols
        for symbol_col in ["SYMBOL", "SYMBOL.3"]:
            for symbol, count in group[symbol_col].fillna("").astype(str).value_counts().items():
                symbol_count_rows.append(
                    {
                        "candidate_type": candidate_type,
                        "CHARACTERISTIC_ID": char_id,
                        "CHARACTERISTIC_NAME": char_name,
                        "symbol_column": symbol_col,
                        "symbol": symbol,
                        "row_count": int(count),
                    }
                )

target_values_long = pd.DataFrame(target_values_long_rows)
symbol_counts = pd.DataFrame(symbol_count_rows)

target_values_long.to_csv(OUTPUT_TARGET_VALUES_LONG, index=False)
symbol_counts.to_csv(OUTPUT_SYMBOL_COUNTS, index=False)

if not target_values_long.empty:
    target_values_long["wide_column"] = (
        target_values_long["candidate_type"].astype(str)
        + "__char_"
        + target_values_long["CHARACTERISTIC_ID"].astype(str)
        + "__"
        + target_values_long["value_column"].astype(str)
    )

    target_values_wide = target_values_long.pivot_table(
        index=[
            "census_division_dguid",
            "census_division_code",
            "census_division_name_profile",
        ],
        columns="wide_column",
        values="value_numeric",
        aggfunc="first",
    ).reset_index()

    target_values_wide.columns.name = None
else:
    target_values_wide = pd.DataFrame()

target_values_wide.to_csv(OUTPUT_TARGET_VALUES_WIDE, index=False)


# -----------------------------
# Derived formula audit
# -----------------------------

formula_rows = []

if not target_values_wide.empty:
    cols = list(target_values_wide.columns)

    female_cols_count = [
        col for col in cols
        if col.startswith("female_parent_lone_parent_family_count_or_rate")
        and col.endswith("__C1_COUNT_TOTAL")
    ]
    female_cols_rate = [
        col for col in cols
        if col.startswith("female_parent_lone_parent_family_count_or_rate")
        and col.endswith("__C10_RATE_TOTAL")
    ]
    maintainer_cols_rate = [
        col for col in cols
        if col.startswith("female_or_women_household_maintainer")
        and col.endswith("__C10_RATE_TOTAL")
    ]
    private_household_denominator_cols = [
        col for col in cols
        if col.startswith("total_private_household_denominator")
        and col.endswith("__C1_COUNT_TOTAL")
    ]
    lone_parent_denominator_cols = [
        col for col in cols
        if col.startswith("total_lone_parent_family_denominator")
        and col.endswith("__C1_COUNT_TOTAL")
    ]

    formula_rows.append(
        {
            "candidate_formula": "published_female_parent_lone_parent_rate",
            "available_columns": "; ".join(female_cols_rate),
            "available": len(female_cols_rate) > 0,
            "recommended_default_without_review": False,
            "interpretation": (
                "Likely percent of lone-parent families with a female parent, not percent of all households."
            ),
        }
    )

    formula_rows.append(
        {
            "candidate_formula": "published_female_or_women_household_maintainer_rate",
            "available_columns": "; ".join(maintainer_cols_rate),
            "available": len(maintainer_cols_rate) > 0,
            "recommended_default_without_review": False,
            "interpretation": (
                "Potentially closer to female-headed households if the Census row is about household maintainers."
            ),
        }
    )

    formula_rows.append(
        {
            "candidate_formula": "100 * female_parent_lone_parent_family_count / total_private_households",
            "available_columns": "; ".join(female_cols_count + private_household_denominator_cols),
            "available": len(female_cols_count) > 0 and len(private_household_denominator_cols) > 0,
            "recommended_default_without_review": False,
            "interpretation": (
                "Closer to share of households affected by female-parent lone-parent structure, "
                "but conceptually narrower than all female-headed households."
            ),
        }
    )

    formula_rows.append(
        {
            "candidate_formula": "100 * female_parent_lone_parent_family_count / total_lone_parent_families",
            "available_columns": "; ".join(female_cols_count + lone_parent_denominator_cols),
            "available": len(female_cols_count) > 0 and len(lone_parent_denominator_cols) > 0,
            "recommended_default_without_review": False,
            "interpretation": (
                "Female share among lone-parent families. Useful but not same denominator as original SoVI variable."
            ),
        }
    )
else:
    formula_rows.append(
        {
            "candidate_formula": "",
            "available_columns": "",
            "available": False,
            "recommended_default_without_review": False,
            "interpretation": "No primary candidate columns found.",
        }
    )

derived_formula_audit = pd.DataFrame(formula_rows)
derived_formula_audit.to_csv(OUTPUT_DERIVED_FORMULA_AUDIT, index=False)


# -----------------------------
# Summary
# -----------------------------

unique_qc_cd_count = len(cd_inventory)
has_98_qc_cds = unique_qc_cd_count == 98

if not base_join.empty:
    matched_base_rows = int((base_join["join_status"] == "both").sum())
    base_only_rows = int((base_join["join_status"] == "left_only").sum())
    profile_only_rows = int((base_join["join_status"] == "right_only").sum())
else:
    matched_base_rows = None
    base_only_rows = None
    profile_only_rows = None

primary_candidate_count = (
    int(characteristic_summary["candidate_type"].isin(PRIMARY_CANDIDATE_TYPES).sum())
    if not characteristic_summary.empty
    else 0
)

summary = pd.DataFrame(
    [
        {
            "metric": "raw_csv",
            "value": str(RAW_CSV.relative_to(DATA_DIR)),
        },
        {
            "metric": "raw_encoding",
            "value": raw_encoding,
        },
        {
            "metric": "total_rows_scanned",
            "value": total_rows,
        },
        {
            "metric": "quebec_cd_rows_scanned",
            "value": quebec_cd_rows,
        },
        {
            "metric": "unique_quebec_census_divisions",
            "value": unique_qc_cd_count,
        },
        {
            "metric": "has_98_quebec_census_divisions",
            "value": has_98_qc_cds,
        },
        {
            "metric": "base_cd_frame_path",
            "value": str(base_path.relative_to(DATA_DIR)) if base_path is not None else "",
        },
        {
            "metric": "matched_base_cd_rows",
            "value": matched_base_rows,
        },
        {
            "metric": "base_only_cd_rows",
            "value": base_only_rows,
        },
        {
            "metric": "profile_only_cd_rows",
            "value": profile_only_rows,
        },
        {
            "metric": "keyword_match_rows",
            "value": len(all_matches),
        },
        {
            "metric": "candidate_characteristics_found",
            "value": len(characteristic_summary),
        },
        {
            "metric": "primary_candidate_characteristics_found",
            "value": primary_candidate_count,
        },
        {
            "metric": "recommended_next_step",
            "value": (
                "Review female_headed_household_proxy_characteristic_summary_2021.csv and "
                "female_headed_household_proxy_derived_formula_audit_2021.csv. "
                "Then choose whether PCTF_HH90 should use a household-maintainer row, "
                "a female-parent lone-parent family rate, or a derived formula."
            ),
        },
    ]
)

summary.to_csv(OUTPUT_SUMMARY, index=False)


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("TARGETED FEMALE-HEADED-HOUSEHOLD PROXY INSPECTION SUMMARY")
print("=" * 72)

print("\nRaw file:")
print("Path:", RAW_CSV.relative_to(DATA_DIR))
print("Encoding:", raw_encoding)
print("Rows scanned:", total_rows)

print("\nQuébec census divisions:")
print("Québec CD rows scanned:", quebec_cd_rows)
print("Unique Québec CD DGUIDs:", unique_qc_cd_count)
print("Has 98 Québec CDs:", has_98_qc_cds)

if not base_join.empty:
    print("\nJoin to existing CD base frame:")
    print(base_join["join_status"].value_counts(dropna=False).to_string())

print("\nPrimary candidate classifications:")
if candidate_classification.empty:
    print("[none]")
else:
    print(candidate_classification.to_string(index=False))

print("\nDerived formula audit:")
print(derived_formula_audit.to_string(index=False))

print("\nTarget values wide preview:")
if target_values_wide.empty:
    print("[none]")
else:
    print(target_values_wide.head(10).to_string(index=False))

print("\nRecommended next step:")
print(summary.loc[summary["metric"] == "recommended_next_step", "value"].iloc[0])

print("\nSaved:")
print(OUTPUT_FILE_INVENTORY)
print(OUTPUT_CD_INVENTORY)
print(OUTPUT_BASE_JOIN)
print(OUTPUT_ALL_MATCHES)
print(OUTPUT_CHARACTERISTIC_SUMMARY)
print(OUTPUT_CANDIDATE_CLASSIFICATION)
print(OUTPUT_TARGET_VALUES_LONG)
print(OUTPUT_TARGET_VALUES_WIDE)
print(OUTPUT_DERIVED_FORMULA_AUDIT)
print(OUTPUT_SYMBOL_COUNTS)
print(OUTPUT_SUMMARY)

print("\nDone.")