from pathlib import Path
import re
import pandas as pd


# ============================================================
# Inspect 2021 Census Profile at Census Division Level
# ============================================================
#
# Purpose:
#   Inspect the newly downloaded Statistics Canada Census Profile file for
#   Census divisions and verify that it can support CD-level SoVI cleaning.
#
# Main file:
#   census_profile_census_division_2021/raw/98-401-X2021004_English_CSV_data.csv
#
# This script does NOT clean final SoVI variables.
# It verifies:
#   1. file readability and encoding;
#   2. GEO_LEVEL coverage;
#   3. presence of 98 Quebec census divisions;
#   4. compatibility with our CD spatial/population base frame;
#   5. characteristic coverage at Quebec CD level;
#   6. SoVI-relevant candidate CHARACTERISTIC_IDs and names;
#   7. symbol / suppression / caution counts for candidate characteristics.
#
# Run from data/:
#   python census_profile_census_division_2021/inspect_census_division_profile_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_profile_census_division_2021"
RAW_DIR = SECTION_DIR / "raw"
OUTPUT_DIR = SECTION_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = RAW_DIR / "98-401-X2021004_English_CSV_data.csv"
META_TXT = RAW_DIR / "98-401-X2021004_English_meta.txt"
GEO_STARTING_ROW = RAW_DIR / "98-401-X2021004_Geo_starting_row.CSV"

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

OUTPUT_FILE_INVENTORY = OUTPUT_DIR / "census_division_profile_file_inventory_2021.csv"
OUTPUT_GEO_LEVEL_COUNTS = OUTPUT_DIR / "census_division_profile_geo_level_counts_2021.csv"
OUTPUT_QUEBEC_CD_INVENTORY = OUTPUT_DIR / "quebec_census_division_inventory_from_profile_2021.csv"
OUTPUT_QUEBEC_CD_BASE_JOIN = OUTPUT_DIR / "quebec_census_division_profile_base_join_check_2021.csv"
OUTPUT_CHARACTERISTIC_COVERAGE = OUTPUT_DIR / "quebec_cd_characteristic_coverage_2021.csv"
OUTPUT_SOVI_CANDIDATES = OUTPUT_DIR / "quebec_cd_sovi_candidate_characteristics_2021.csv"
OUTPUT_SOVI_TARGET_SUMMARY = OUTPUT_DIR / "quebec_cd_sovi_target_characteristic_summary_2021.csv"
OUTPUT_SYMBOL_COUNTS = OUTPUT_DIR / "quebec_cd_sovi_candidate_symbol_counts_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "census_division_profile_inspection_summary_2021.csv"


# -----------------------------
# Config
# -----------------------------

ENCODING_CANDIDATES = [
    "cp1252",
    "latin1",
    "utf-8",
    "utf-8-sig",
]

CHUNK_SIZE = 200_000

QUEBEC_CD_DGUID_PREFIX = "2021A000324"

REQUIRED_PROFILE_COLUMNS = [
    "CENSUS_YEAR",
    "DGUID",
    "ALT_GEO_CODE",
    "GEO_LEVEL",
    "GEO_NAME",
    "CHARACTERISTIC_ID",
    "CHARACTERISTIC_NAME",
    "C1_COUNT_TOTAL",
    "C10_RATE_TOTAL",
]

OPTIONAL_PROFILE_COLUMNS = [
    "CHARACTERISTIC_NOTE",
    "SYMBOL",
    "SYMBOL.3",
    "TNR_SF",
    "TNR_LF",
    "DATA_QUALITY_FLAG",
]


# -----------------------------
# SoVI target candidate specs
# -----------------------------
#
# These are not final choices. They are inspection targets.
# The script will search both by approximate CHARACTERISTIC_ID when known
# and by keywords in CHARACTERISTIC_NAME.
#
# value_column_preference:
#   count_or_value = usually C1_COUNT_TOTAL
#   rate = usually C10_RATE_TOTAL
#
# For many percentage rows in Census Profile, C10_RATE_TOTAL is the safer
# column. For raw counts / dollar values / medians / averages, C1_COUNT_TOTAL
# is usually the value-bearing column.
#

SOVI_TARGETS = [
    {
        "original_code": "MED_AGE90",
        "canonical_variable": "median_age",
        "description": "Median age",
        "candidate_ids": [40],
        "keywords_any": ["median age"],
        "keywords_all": [],
        "value_column_preference": "count_or_value",
    },
    {
        "original_code": "PERCAP89",
        "canonical_variable": "per_capita_income",
        "description": "Per capita income / income proxy",
        "candidate_ids": [128, 130, 252, 253],
        "keywords_any": [
            "average total income",
            "average after-tax income",
            "median after-tax income",
            "median total income",
        ],
        "keywords_all": ["income"],
        "value_column_preference": "count_or_value",
    },
    {
        "original_code": "MVALOO90",
        "canonical_variable": "median_owner_occupied_housing_value",
        "description": "Median value of owner-occupied housing",
        "candidate_ids": [],
        "keywords_any": [
            "median value of dwellings",
            "median value of owner",
            "value of dwellings",
            "owner-occupied",
        ],
        "keywords_all": ["median"],
        "value_column_preference": "count_or_value",
    },
    {
        "original_code": "MEDRENT90",
        "canonical_variable": "median_rent",
        "description": "Median rent / shelter cost for renter households",
        "candidate_ids": [],
        "keywords_any": [
            "median monthly shelter costs for rented dwellings",
            "median monthly shelter costs for renter",
            "median shelter costs for rented",
            "median rent",
        ],
        "keywords_all": ["rented"],
        "value_column_preference": "count_or_value",
    },
    {
        "original_code": "PCTBLACK90",
        "canonical_variable": "pct_black_or_local_proxy",
        "description": "Percent Black / local visible minority proxy",
        "candidate_ids": [],
        "keywords_any": ["black"],
        "keywords_all": [],
        "value_column_preference": "rate",
    },
    {
        "original_code": "PCTINDIAN90",
        "canonical_variable": "pct_indigenous_or_local_proxy",
        "description": "Percent Indigenous identity / local proxy",
        "candidate_ids": [],
        "keywords_any": [
            "indigenous identity",
            "aboriginal identity",
            "first nations",
            "métis",
            "metis",
            "inuk",
            "inuit",
        ],
        "keywords_all": [],
        "value_column_preference": "rate",
    },
    {
        "original_code": "PCTASIAN90",
        "canonical_variable": "pct_asian_or_local_proxy",
        "description": "Percent Asian / local proxy",
        "candidate_ids": [],
        "keywords_any": [
            "south asian",
            "chinese",
            "filipino",
            "southeast asian",
            "west asian",
            "korean",
            "japanese",
        ],
        "keywords_all": [],
        "value_column_preference": "rate",
    },
    {
        "original_code": "PCTHISPANIC90",
        "canonical_variable": "pct_hispanic_or_local_proxy",
        "description": "Percent Hispanic / Latin American proxy",
        "candidate_ids": [],
        "keywords_any": [
            "latin american",
            "hispanic",
        ],
        "keywords_all": [],
        "value_column_preference": "rate",
    },
    {
        "original_code": "PCTKIDS90",
        "canonical_variable": "pct_under_5_or_child_proxy",
        "description": "Percent population under five",
        "candidate_ids": [10],
        "keywords_any": [
            "0 to 4 years",
            "0 to 5 years",
            "under 5",
        ],
        "keywords_all": [],
        "value_column_preference": "rate",
    },
    {
        "original_code": "PCTOLD90",
        "canonical_variable": "pct_over_65",
        "description": "Percent population over 65",
        "candidate_ids": [24, 37],
        "keywords_any": [
            "65 years and over",
        ],
        "keywords_all": [],
        "value_column_preference": "rate",
    },
    {
        "original_code": "PCTVLUN91",
        "canonical_variable": "pct_unemployed",
        "description": "Percent unemployed",
        "candidate_ids": [],
        "keywords_any": [
            "unemployed",
            "unemployment rate",
        ],
        "keywords_all": [],
        "value_column_preference": "rate",
    },
    {
        "original_code": "AVGPERHH",
        "canonical_variable": "avg_people_per_household",
        "description": "Average household size",
        "candidate_ids": [57],
        "keywords_any": [
            "average household size",
        ],
        "keywords_all": [],
        "value_column_preference": "count_or_value",
    },
    {
        "original_code": "PCTHH7589",
        "canonical_variable": "pct_high_income_households",
        "description": "Percent high-income households",
        "candidate_ids": [276, 277, 278, 279, 280],
        "keywords_any": [
            "$100,000 and over",
            "$125,000 to $149,999",
            "$150,000 and over",
            "$200,000 and over",
        ],
        "keywords_all": ["household"],
        "value_column_preference": "rate",
    },
    {
        "original_code": "PCTPOV90",
        "canonical_variable": "pct_poverty_or_low_income",
        "description": "Percent low income / poverty proxy",
        "candidate_ids": [345, 360],
        "keywords_any": [
            "prevalence of low income based on the low-income measure",
            "prevalence of low income based on the low-income cut-offs",
            "lim-at",
            "lico-at",
        ],
        "keywords_all": [],
        "value_column_preference": "rate",
    },
    {
        "original_code": "PCTRENTER90",
        "canonical_variable": "pct_renter_occupied",
        "description": "Percent renter households / renter-occupied units",
        "candidate_ids": [],
        "keywords_any": [
            "renter",
            "tenant",
            "rented",
        ],
        "keywords_all": [],
        "value_column_preference": "rate",
    },
    {
        "original_code": "PCTMOBL90",
        "canonical_variable": "pct_mobile_homes",
        "description": "Percent mobile homes / movable dwellings",
        "candidate_ids": [49],
        "keywords_any": [
            "movable dwelling",
            "mobile home",
        ],
        "keywords_all": [],
        "value_column_preference": "rate",
    },
    {
        "original_code": "PCTNOHS90",
        "canonical_variable": "pct_no_high_school",
        "description": "Percent with no high school diploma / no certificate",
        "candidate_ids": [],
        "keywords_any": [
            "no certificate, diploma or degree",
            "no certificate",
            "no diploma",
            "no degree",
        ],
        "keywords_all": [],
        "value_column_preference": "rate",
    },
    {
        "original_code": "CVBRPC91",
        "canonical_variable": "labor_force_participation_rate",
        "description": "Labour force participation rate",
        "candidate_ids": [],
        "keywords_any": [
            "participation rate",
            "labour force participation",
            "labor force participation",
        ],
        "keywords_all": [],
        "value_column_preference": "rate",
    },
    {
        "original_code": "FEMLBR90",
        "canonical_variable": "female_labor_force_participation_rate",
        "description": "Female labour force participation rate",
        "candidate_ids": [],
        "keywords_any": [
            "participation rate",
        ],
        "keywords_all": ["women"],
        "value_column_preference": "rate",
    },
    {
        "original_code": "AGRIPC90",
        "canonical_variable": "pct_extractive_employment",
        "description": "Percent employed in extractive / primary industries",
        "candidate_ids": [],
        "keywords_any": [
            "agriculture, forestry, fishing and hunting",
            "mining, quarrying, and oil and gas extraction",
            "agriculture",
            "forestry",
            "mining",
        ],
        "keywords_all": ["industry"],
        "value_column_preference": "rate",
    },
    {
        "original_code": "TRANPC90",
        "canonical_variable": "pct_transport_utilities_employment",
        "description": "Percent employed in transportation / utilities",
        "candidate_ids": [],
        "keywords_any": [
            "transportation and warehousing",
            "utilities",
        ],
        "keywords_all": ["industry"],
        "value_column_preference": "rate",
    },
    {
        "original_code": "SERVPC90",
        "canonical_variable": "pct_service_employment",
        "description": "Percent employed in service occupations",
        "candidate_ids": [],
        "keywords_any": [
            "sales and service occupations",
            "service occupations",
        ],
        "keywords_all": [],
        "value_column_preference": "rate",
    },
    {
        "original_code": "PCTFEM90",
        "canonical_variable": "pct_female",
        "description": "Percent female / women+",
        "candidate_ids": [],
        "keywords_any": [
            "women+",
            "female",
        ],
        "keywords_all": [],
        "value_column_preference": "rate",
    },
    {
        "original_code": "PCTF_HH90",
        "canonical_variable": "pct_female_headed_households",
        "description": "Percent female-headed households / woman+ lone parent proxy",
        "candidate_ids": [87],
        "keywords_any": [
            "in which the parent is a woman",
            "female-headed",
            "woman+",
            "one-parent",
            "lone-parent",
        ],
        "keywords_all": [],
        "value_column_preference": "rate",
    },
]


# -----------------------------
# Helpers
# -----------------------------

def detect_encoding(path: Path) -> str:
    last_error = None

    for encoding in ENCODING_CANDIDATES:
        try:
            pd.read_csv(
                path,
                nrows=5,
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
        f"Could not decode {path} with encodings {ENCODING_CANDIDATES}. "
        f"Last error: {last_error}",
    )


def read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    last_error = None

    for encoding in ENCODING_CANDIDATES:
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


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return read_csv_with_fallback(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix in [".geojson", ".gpkg", ".shp"]:
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError(
                f"geopandas is required to read spatial file {path}"
            ) from exc
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


def norm_text(value: object) -> str:
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


def target_matches_characteristic(target: dict, char_id: object, char_name: object) -> tuple[bool, str]:
    text = norm_text(char_name)

    candidate_ids = {str(x) for x in target.get("candidate_ids", [])}
    char_id_str = str(char_id)

    if char_id_str in candidate_ids:
        return True, "candidate_id"

    keywords_any = [kw.lower() for kw in target.get("keywords_any", [])]
    keywords_all = [kw.lower() for kw in target.get("keywords_all", [])]

    has_any = any(kw in text for kw in keywords_any) if keywords_any else False
    has_all = all(kw in text for kw in keywords_all) if keywords_all else True

    if has_any and has_all:
        return True, "keyword_match"

    return False, ""


def summarize_symbol(series: pd.Series) -> str:
    counts = series.fillna("").astype(str).value_counts(dropna=False)
    if counts.empty:
        return ""
    return "; ".join(f"{idx}:{int(val)}" for idx, val in counts.items())


def build_file_inventory() -> pd.DataFrame:
    rows = []

    for path in [RAW_CSV, META_TXT, GEO_STARTING_ROW, RAW_DIR / "README_meta.txt"]:
        rows.append(
            {
                "relative_path": str(path.relative_to(DATA_DIR)) if path.exists() else str(path),
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else None,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# Initial checks
# -----------------------------

if not RAW_CSV.exists():
    raise FileNotFoundError(f"Raw Census Profile CD CSV not found:\n{RAW_CSV}")

encoding = detect_encoding(RAW_CSV)

print("\nCensus Profile Census Division raw file")
print("Path:", RAW_CSV)
print("Encoding selected:", encoding)

header = pd.read_csv(
    RAW_CSV,
    nrows=0,
    dtype=str,
    encoding=encoding,
    low_memory=False,
)

columns = list(header.columns)

print("\nColumns:")
print(columns)

require_columns(columns, REQUIRED_PROFILE_COLUMNS, "Census Profile CD raw CSV")

usecols = [
    col for col in REQUIRED_PROFILE_COLUMNS + OPTIONAL_PROFILE_COLUMNS
    if col in columns
]

file_inventory = build_file_inventory()
file_inventory.to_csv(OUTPUT_FILE_INVENTORY, index=False)


# -----------------------------
# Load base CD frame if available
# -----------------------------

base_path = find_base_cd_frame()
base = None

if base_path is not None:
    base = read_table(base_path)
    base = normalize_columns(base)

    if "census_division_dguid" not in base.columns:
        print("\nWARNING: Base CD frame exists but lacks census_division_dguid.")
        base = None
    else:
        base = base.copy()
        base["census_division_dguid"] = clean_text(base["census_division_dguid"])

        print("\nLoaded existing CD spatial/population base frame")
        print("Path:", base_path.relative_to(DATA_DIR))
        print("Rows:", len(base))
else:
    print("\nWARNING: No existing CD spatial/population base frame found.")


# -----------------------------
# Chunked scan
# -----------------------------

geo_level_counts = {}
quebec_cd_inventory = {}
characteristic_coverage = {}
candidate_rows = []
symbol_rows = []

total_rows = 0
quebec_cd_rows = 0

print("\nScanning Census Profile CD file in chunks...")

for chunk_idx, chunk in enumerate(
    pd.read_csv(
        RAW_CSV,
        dtype=str,
        encoding=encoding,
        low_memory=False,
        chunksize=CHUNK_SIZE,
        usecols=usecols,
    ),
    start=1,
):
    total_rows += len(chunk)
    chunk = normalize_columns(chunk)

    chunk["GEO_LEVEL_NORM"] = clean_text(chunk["GEO_LEVEL"]).str.lower()

    # GEO_LEVEL counts
    for level, count in chunk["GEO_LEVEL"].value_counts(dropna=False).items():
        key = str(level)
        geo_level_counts[key] = geo_level_counts.get(key, 0) + int(count)

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

    # Inventory of Quebec CDs
    geo_cols = [
        "DGUID",
        "ALT_GEO_CODE",
        "GEO_NAME",
        "GEO_LEVEL",
        "TNR_SF",
        "TNR_LF",
        "DATA_QUALITY_FLAG",
    ]
    geo_cols = [col for col in geo_cols if col in qc.columns]

    for _, row in qc[geo_cols].drop_duplicates(subset=["DGUID"]).iterrows():
        dguid = str(row["DGUID"])
        quebec_cd_inventory[dguid] = {
            "DGUID": dguid,
            "ALT_GEO_CODE": row.get("ALT_GEO_CODE", ""),
            "GEO_NAME": row.get("GEO_NAME", ""),
            "GEO_LEVEL": row.get("GEO_LEVEL", ""),
            "TNR_SF": row.get("TNR_SF", ""),
            "TNR_LF": row.get("TNR_LF", ""),
            "DATA_QUALITY_FLAG": row.get("DATA_QUALITY_FLAG", ""),
        }

    # Characteristic coverage
    grouped = qc.groupby(
        ["CHARACTERISTIC_ID", "CHARACTERISTIC_NAME"],
        dropna=False,
    )

    for (char_id, char_name), g in grouped:
        char_id_str = str(char_id)

        if char_id_str not in characteristic_coverage:
            characteristic_coverage[char_id_str] = {
                "CHARACTERISTIC_ID": char_id_str,
                "CHARACTERISTIC_NAME": char_name,
                "n_quebec_cd_rows": 0,
                "n_unique_quebec_cds": set(),
                "count_total_non_missing": 0,
                "rate_total_non_missing": 0,
                "symbols_count_total": {},
                "symbols_rate_total": {},
            }

        entry = characteristic_coverage[char_id_str]
        entry["n_quebec_cd_rows"] += len(g)
        entry["n_unique_quebec_cds"].update(g["DGUID"].dropna().astype(str).tolist())

        c1_numeric = clean_numeric(g["C1_COUNT_TOTAL"])
        c10_numeric = clean_numeric(g["C10_RATE_TOTAL"])

        entry["count_total_non_missing"] += int(c1_numeric.notna().sum())
        entry["rate_total_non_missing"] += int(c10_numeric.notna().sum())

        if "SYMBOL" in g.columns:
            for sym, cnt in g["SYMBOL"].fillna("").astype(str).value_counts().items():
                entry["symbols_count_total"][sym] = entry["symbols_count_total"].get(sym, 0) + int(cnt)

        if "SYMBOL.3" in g.columns:
            for sym, cnt in g["SYMBOL.3"].fillna("").astype(str).value_counts().items():
                entry["symbols_rate_total"][sym] = entry["symbols_rate_total"].get(sym, 0) + int(cnt)

    # SoVI candidate detection
    candidate_chars = qc[
        [
            col for col in [
                "CHARACTERISTIC_ID",
                "CHARACTERISTIC_NAME",
                "CHARACTERISTIC_NOTE",
            ]
            if col in qc.columns
        ]
    ].drop_duplicates()

    for _, char_row in candidate_chars.iterrows():
        char_id = char_row.get("CHARACTERISTIC_ID", "")
        char_name = char_row.get("CHARACTERISTIC_NAME", "")

        for target in SOVI_TARGETS:
            matched, match_method = target_matches_characteristic(target, char_id, char_name)

            if not matched:
                continue

            target_rows = qc[qc["CHARACTERISTIC_ID"].astype(str).str.strip() == str(char_id)].copy()

            if target_rows.empty:
                continue

            count_numeric = clean_numeric(target_rows["C1_COUNT_TOTAL"])
            rate_numeric = clean_numeric(target_rows["C10_RATE_TOTAL"])

            candidate_rows.append(
                {
                    "original_code": target["original_code"],
                    "canonical_variable": target["canonical_variable"],
                    "target_description": target["description"],
                    "match_method": match_method,
                    "value_column_preference": target["value_column_preference"],
                    "CHARACTERISTIC_ID": char_id,
                    "CHARACTERISTIC_NAME": char_name,
                    "CHARACTERISTIC_NOTE": char_row.get("CHARACTERISTIC_NOTE", ""),
                    "n_quebec_cd_rows": len(target_rows),
                    "n_unique_quebec_cds": target_rows["DGUID"].nunique(),
                    "count_total_non_missing": int(count_numeric.notna().sum()),
                    "rate_total_non_missing": int(rate_numeric.notna().sum()),
                    "count_total_min": count_numeric.min(skipna=True),
                    "count_total_max": count_numeric.max(skipna=True),
                    "count_total_mean": count_numeric.mean(skipna=True),
                    "rate_total_min": rate_numeric.min(skipna=True),
                    "rate_total_max": rate_numeric.max(skipna=True),
                    "rate_total_mean": rate_numeric.mean(skipna=True),
                    "symbol_count_total": summarize_symbol(target_rows["SYMBOL"]) if "SYMBOL" in target_rows.columns else "",
                    "symbol_rate_total": summarize_symbol(target_rows["SYMBOL.3"]) if "SYMBOL.3" in target_rows.columns else "",
                }
            )

            if "SYMBOL" in target_rows.columns:
                for sym, cnt in target_rows["SYMBOL"].fillna("").astype(str).value_counts().items():
                    symbol_rows.append(
                        {
                            "original_code": target["original_code"],
                            "canonical_variable": target["canonical_variable"],
                            "CHARACTERISTIC_ID": char_id,
                            "CHARACTERISTIC_NAME": char_name,
                            "value_family": "C1_COUNT_TOTAL",
                            "symbol": sym,
                            "row_count": int(cnt),
                        }
                    )

            if "SYMBOL.3" in target_rows.columns:
                for sym, cnt in target_rows["SYMBOL.3"].fillna("").astype(str).value_counts().items():
                    symbol_rows.append(
                        {
                            "original_code": target["original_code"],
                            "canonical_variable": target["canonical_variable"],
                            "CHARACTERISTIC_ID": char_id,
                            "CHARACTERISTIC_NAME": char_name,
                            "value_family": "C10_RATE_TOTAL",
                            "symbol": sym,
                            "row_count": int(cnt),
                        }
                    )

    if chunk_idx % 10 == 0:
        print(f"  Processed chunks: {chunk_idx}, rows so far: {total_rows}")


# -----------------------------
# Build output dataframes
# -----------------------------

geo_level_counts_df = (
    pd.DataFrame(
        [
            {
                "GEO_LEVEL": level,
                "row_count": count,
            }
            for level, count in geo_level_counts.items()
        ]
    )
    .sort_values("row_count", ascending=False)
    .reset_index(drop=True)
)

quebec_cd_inventory_df = (
    pd.DataFrame(quebec_cd_inventory.values())
    .sort_values("DGUID")
    .reset_index(drop=True)
    if quebec_cd_inventory
    else pd.DataFrame(
        columns=[
            "DGUID",
            "ALT_GEO_CODE",
            "GEO_NAME",
            "GEO_LEVEL",
            "TNR_SF",
            "TNR_LF",
            "DATA_QUALITY_FLAG",
        ]
    )
)

coverage_rows = []
for entry in characteristic_coverage.values():
    symbols_count = entry["symbols_count_total"]
    symbols_rate = entry["symbols_rate_total"]

    coverage_rows.append(
        {
            "CHARACTERISTIC_ID": entry["CHARACTERISTIC_ID"],
            "CHARACTERISTIC_NAME": entry["CHARACTERISTIC_NAME"],
            "n_quebec_cd_rows": entry["n_quebec_cd_rows"],
            "n_unique_quebec_cds": len(entry["n_unique_quebec_cds"]),
            "count_total_non_missing": entry["count_total_non_missing"],
            "rate_total_non_missing": entry["rate_total_non_missing"],
            "symbols_count_total": "; ".join(
                f"{k}:{v}" for k, v in sorted(symbols_count.items())
            ),
            "symbols_rate_total": "; ".join(
                f"{k}:{v}" for k, v in sorted(symbols_rate.items())
            ),
        }
    )

characteristic_coverage_df = (
    pd.DataFrame(coverage_rows)
    .sort_values("CHARACTERISTIC_ID", key=lambda s: pd.to_numeric(s, errors="coerce"))
    .reset_index(drop=True)
    if coverage_rows
    else pd.DataFrame()
)

sovi_candidates_df = (
    pd.DataFrame(candidate_rows)
    .drop_duplicates()
    .sort_values(
        ["original_code", "match_method", "CHARACTERISTIC_ID"],
        key=lambda s: s.map(str),
    )
    .reset_index(drop=True)
    if candidate_rows
    else pd.DataFrame()
)

symbol_counts_df = (
    pd.DataFrame(symbol_rows)
    .drop_duplicates()
    .sort_values(["original_code", "CHARACTERISTIC_ID", "value_family", "symbol"])
    .reset_index(drop=True)
    if symbol_rows
    else pd.DataFrame()
)


# -----------------------------
# Base join check
# -----------------------------

if base is not None and not quebec_cd_inventory_df.empty:
    profile_geo = quebec_cd_inventory_df[["DGUID", "ALT_GEO_CODE", "GEO_NAME"]].copy()
    profile_geo = profile_geo.rename(
        columns={
            "DGUID": "profile_dguid",
            "ALT_GEO_CODE": "profile_cd_code",
            "GEO_NAME": "profile_geo_name",
        }
    )

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
        profile_geo,
        left_on="census_division_dguid",
        right_on="profile_dguid",
        how="outer",
        indicator=True,
    )

    base_join["join_status"] = base_join["_merge"]
    base_join = base_join.drop(columns="_merge")

else:
    base_join = pd.DataFrame()


# -----------------------------
# Target summary
# -----------------------------

target_summary_rows = []

for target in SOVI_TARGETS:
    if sovi_candidates_df.empty:
        candidates_for_target = pd.DataFrame()
    else:
        candidates_for_target = sovi_candidates_df[
            sovi_candidates_df["canonical_variable"] == target["canonical_variable"]
        ].copy()

    if candidates_for_target.empty:
        selected_id = ""
        selected_name = ""
        selected_column = ""
        best_coverage = 0
        status = "no_candidate_found"
    else:
        # Prefer exact candidate_id matches, full 98-CD coverage, then non-missing preferred column.
        temp = candidates_for_target.copy()
        temp["match_priority"] = temp["match_method"].map(
            {
                "candidate_id": 0,
                "keyword_match": 1,
            }
        ).fillna(9)

        preferred = target["value_column_preference"]
        if preferred == "rate":
            temp["preferred_non_missing"] = temp["rate_total_non_missing"]
            selected_column = "C10_RATE_TOTAL"
        else:
            temp["preferred_non_missing"] = temp["count_total_non_missing"]
            selected_column = "C1_COUNT_TOTAL"

        temp = temp.sort_values(
            [
                "match_priority",
                "n_unique_quebec_cds",
                "preferred_non_missing",
            ],
            ascending=[True, False, False],
        )

        best = temp.iloc[0]

        selected_id = best["CHARACTERISTIC_ID"]
        selected_name = best["CHARACTERISTIC_NAME"]
        best_coverage = int(best["n_unique_quebec_cds"])
        status = "candidate_found_full_coverage" if best_coverage == 98 else "candidate_found_partial_coverage"

    target_summary_rows.append(
        {
            "original_code": target["original_code"],
            "canonical_variable": target["canonical_variable"],
            "description": target["description"],
            "candidate_count": len(candidates_for_target),
            "selected_characteristic_id_for_review": selected_id,
            "selected_characteristic_name_for_review": selected_name,
            "suggested_value_column_for_review": selected_column,
            "best_unique_quebec_cd_coverage": best_coverage,
            "status": status,
            "notes": (
                "Review before using in cleaner. Candidate selection is heuristic."
                if status.startswith("candidate_found")
                else "No candidate found by current IDs/keywords."
            ),
        }
    )

target_summary_df = pd.DataFrame(target_summary_rows)


# -----------------------------
# Summary
# -----------------------------

unique_qc_cd_count = len(quebec_cd_inventory_df)
has_98_qc_cds = unique_qc_cd_count == 98

if not base_join.empty:
    matched_base_rows = int((base_join["join_status"] == "both").sum())
    profile_only_rows = int((base_join["join_status"] == "right_only").sum())
    base_only_rows = int((base_join["join_status"] == "left_only").sum())
else:
    matched_base_rows = None
    profile_only_rows = None
    base_only_rows = None

summary_df = pd.DataFrame(
    [
        {
            "metric": "raw_csv",
            "value": str(RAW_CSV.relative_to(DATA_DIR)),
        },
        {
            "metric": "encoding",
            "value": encoding,
        },
        {
            "metric": "total_rows_scanned",
            "value": total_rows,
        },
        {
            "metric": "quebec_census_division_rows",
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
            "metric": "profile_only_cd_rows",
            "value": profile_only_rows,
        },
        {
            "metric": "base_only_cd_rows",
            "value": base_only_rows,
        },
        {
            "metric": "total_characteristics_at_quebec_cd_level",
            "value": len(characteristic_coverage_df),
        },
        {
            "metric": "sovi_targets_with_candidate_found",
            "value": int(target_summary_df["status"].str.startswith("candidate_found").sum()),
        },
        {
            "metric": "sovi_targets_without_candidate_found",
            "value": int((target_summary_df["status"] == "no_candidate_found").sum()),
        },
        {
            "metric": "recommended_next_step",
            "value": (
                "Review candidate characteristic summary, then generate CD-level SoVI Census Profile cleaner."
                if has_98_qc_cds
                else "Dataset does not contain the expected 98 Quebec CDs; investigate download."
            ),
        },
    ]
)


# -----------------------------
# Save outputs
# -----------------------------

geo_level_counts_df.to_csv(OUTPUT_GEO_LEVEL_COUNTS, index=False)
quebec_cd_inventory_df.to_csv(OUTPUT_QUEBEC_CD_INVENTORY, index=False)
base_join.to_csv(OUTPUT_QUEBEC_CD_BASE_JOIN, index=False)
characteristic_coverage_df.to_csv(OUTPUT_CHARACTERISTIC_COVERAGE, index=False)
sovi_candidates_df.to_csv(OUTPUT_SOVI_CANDIDATES, index=False)
target_summary_df.to_csv(OUTPUT_SOVI_TARGET_SUMMARY, index=False)
symbol_counts_df.to_csv(OUTPUT_SYMBOL_COUNTS, index=False)
summary_df.to_csv(OUTPUT_SUMMARY, index=False)


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CENSUS DIVISION PROFILE INSPECTION SUMMARY")
print("=" * 72)

print("\nRaw file:")
print("Path:", RAW_CSV.relative_to(DATA_DIR))
print("Encoding:", encoding)
print("Rows scanned:", total_rows)

print("\nGEO_LEVEL counts:")
print(geo_level_counts_df.to_string(index=False))

print("\nQuebec census divisions:")
print("Quebec CD rows:", quebec_cd_rows)
print("Unique Quebec CD DGUIDs:", unique_qc_cd_count)
print("Has 98 Quebec CDs:", has_98_qc_cds)

if not quebec_cd_inventory_df.empty:
    print("\nQuebec CD preview:")
    print(quebec_cd_inventory_df.head(15).to_string(index=False))

if not base_join.empty:
    print("\nJoin to existing CD base frame:")
    print(base_join["join_status"].value_counts(dropna=False).to_string())

print("\nSoVI target candidate summary:")
print(
    target_summary_df[
        [
            "original_code",
            "canonical_variable",
            "candidate_count",
            "selected_characteristic_id_for_review",
            "selected_characteristic_name_for_review",
            "suggested_value_column_for_review",
            "best_unique_quebec_cd_coverage",
            "status",
        ]
    ].to_string(index=False)
)

print("\nCandidate rows saved:", len(sovi_candidates_df))
print("Characteristic coverage rows saved:", len(characteristic_coverage_df))

print("\nRecommended next step:")
print(summary_df.loc[summary_df["metric"] == "recommended_next_step", "value"].iloc[0])

print("\nSaved:")
print(OUTPUT_FILE_INVENTORY)
print(OUTPUT_GEO_LEVEL_COUNTS)
print(OUTPUT_QUEBEC_CD_INVENTORY)
print(OUTPUT_QUEBEC_CD_BASE_JOIN)
print(OUTPUT_CHARACTERISTIC_COVERAGE)
print(OUTPUT_SOVI_CANDIDATES)
print(OUTPUT_SOVI_TARGET_SUMMARY)
print(OUTPUT_SYMBOL_COUNTS)
print(OUTPUT_SUMMARY)

print("\nDone.")