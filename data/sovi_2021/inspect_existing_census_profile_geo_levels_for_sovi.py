from pathlib import Path
import pandas as pd


# ============================================================
# Inspect Existing Census Profile Geo Levels for SoVI
# ============================================================
#
# Purpose:
#   Check whether the existing Census Profile raw file already contains
#   census-division-level rows that can be reused for the SoVI CD-level table.
#
# Main question:
#   Does census_profile_2021/98-401-X2021007_English_CSV_data.csv contain
#   Quebec rows where GEO_LEVEL == "Census division"?
#
# If yes:
#   We can likely reuse the existing Census Profile raw file for many SoVI
#   census-division variables.
#
# If no:
#   We should download a Census Profile file at the Census division geography.
#
# Run from data/:
#   python sovi_2021/inspect_existing_census_profile_geo_levels_for_sovi.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = DATA_DIR / "sovi_2021" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CANDIDATES = [
    DATA_DIR / "census_profile_2021" / "98-401-X2021007_English_CSV_data.csv",
    DATA_DIR / "census_profile_2021" / "raw" / "98-401-X2021007_English_CSV_data.csv",
    DATA_DIR / "census_profile_2021" / "raw" / "98100007.csv",
    DATA_DIR / "census_profile_2021" / "98100007.csv",
]

OUTPUT_GEO_LEVEL_COUNTS = (
    OUTPUT_DIR / "existing_census_profile_geo_level_counts_for_sovi.csv"
)

OUTPUT_QUEBEC_CD_PREVIEW = (
    OUTPUT_DIR / "existing_census_profile_quebec_census_division_preview_for_sovi.csv"
)

OUTPUT_QUEBEC_CD_CHARACTERISTIC_COUNTS = (
    OUTPUT_DIR / "existing_census_profile_quebec_census_division_characteristic_counts_for_sovi.csv"
)

OUTPUT_SOVI_KEYWORD_AUDIT = (
    OUTPUT_DIR / "existing_census_profile_quebec_cd_sovi_keyword_audit.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR / "existing_census_profile_geo_level_summary_for_sovi.csv"
)


# -----------------------------
# Config
# -----------------------------

ENCODING_CANDIDATES = [
    # StatsCan Census Profile CSVs commonly contain Windows-1252 bytes
    # such as 0xC9 in names. Trying utf-8 first can pass on the header
    # / first rows but fail later during chunked scanning.
    "cp1252",
    "latin1",
    "utf-8-sig",
    "utf-8",
]

CHUNK_SIZE = 200_000

QUEBEC_CD_DGUID_PREFIX = "2021A000324"

SOVI_KEYWORD_GROUPS = {
    "median_age": [
        "median age",
    ],
    "income": [
        "income",
        "after-tax income",
        "total income",
        "employment income",
    ],
    "low_income": [
        "low-income",
        "low income",
        "lim-at",
        "lico-at",
    ],
    "unemployment": [
        "unemployed",
        "unemployment",
        "labour force status",
    ],
    "education_no_certificate": [
        "no certificate",
        "no diploma",
        "no degree",
    ],
    "housing_tenure_renter": [
        "renter",
        "tenant",
        "rented",
    ],
    "housing_type_mobile": [
        "movable dwelling",
        "mobile home",
    ],
    "household_family_single_parent": [
        "one-parent",
        "lone-parent",
        "single-parent",
    ],
    "visible_minority_ethnocultural": [
        "visible minority",
        "black",
        "south asian",
        "chinese",
        "filipino",
        "latin american",
        "arab",
        "southeast asian",
        "west asian",
        "korean",
        "japanese",
        "indigenous",
        "aboriginal",
        "first nations",
        "métis",
        "inuk",
        "inuit",
    ],
    "occupation_industry": [
        "industry",
        "occupation",
        "agriculture",
        "forestry",
        "fishing",
        "hunting",
        "mining",
        "transportation",
        "warehousing",
        "utilities",
        "sales and service",
    ],
    "sex_female": [
        "female",
        "women",
    ],
    "dwellings": [
        "private dwellings",
        "occupied by usual residents",
    ],
}


# -----------------------------
# Helpers
# -----------------------------

def find_existing_raw_file() -> Path:
    for path in RAW_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find an existing Census Profile raw CSV.\n"
        "Expected one of:\n"
        + "\n".join(str(path) for path in RAW_CANDIDATES)
    )


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


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def normalize_geo_level(series: pd.Series) -> pd.Series:
    return (
        clean_text(series)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def is_quebec_census_division_dguid(series: pd.Series) -> pd.Series:
    return clean_text(series).str.startswith(QUEBEC_CD_DGUID_PREFIX, na=False)


def require_columns(columns: list[str], required: list[str], label: str) -> None:
    missing = [col for col in required if col not in columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {label}:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(columns)
        )


def keyword_match_group(characteristic_name: str) -> list[str]:
    text = str(characteristic_name).lower()
    matched_groups = []

    for group, keywords in SOVI_KEYWORD_GROUPS.items():
        if any(keyword.lower() in text for keyword in keywords):
            matched_groups.append(group)

    return matched_groups


# -----------------------------
# Locate and inspect raw file
# -----------------------------

raw_path = find_existing_raw_file()
encoding = detect_encoding(raw_path)

print("\nExisting Census Profile raw file")
print("Path:", raw_path)
print("Encoding selected:", encoding)

header = pd.read_csv(
    raw_path,
    nrows=0,
    dtype=str,
    encoding=encoding,
    low_memory=False,
)

columns = list(header.columns)

print("\nColumns:")
print(columns)

require_columns(
    columns,
    ["GEO_LEVEL", "DGUID", "GEO_NAME"],
    "Census Profile raw file",
)

optional_columns = [
    "CHARACTERISTIC_ID",
    "CHARACTERISTIC_NAME",
    "CHARACTERISTIC_NOTE",
]

usecols = [
    col for col in [
        "GEO_LEVEL",
        "DGUID",
        "GEO_NAME",
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "CHARACTERISTIC_NOTE",
    ]
    if col in columns
]


# -----------------------------
# Chunked scan
# -----------------------------

geo_level_counts = {}
quebec_cd_unique = {}
quebec_cd_characteristic_counts = {}
quebec_cd_keyword_rows = []

total_rows = 0
quebec_cd_rows = 0

print("\nScanning file in chunks...")

for chunk_idx, chunk in enumerate(
    pd.read_csv(
        raw_path,
        dtype=str,
        encoding=encoding,
        low_memory=False,
        chunksize=CHUNK_SIZE,
        usecols=usecols,
    ),
    start=1,
):
    total_rows += len(chunk)

    chunk = chunk.copy()
    chunk.columns = [str(col).strip() for col in chunk.columns]

    chunk["GEO_LEVEL_NORM"] = normalize_geo_level(chunk["GEO_LEVEL"])

    # Count all GEO_LEVEL values.
    for level, count in chunk["GEO_LEVEL"].value_counts(dropna=False).items():
        level_key = str(level)
        geo_level_counts[level_key] = geo_level_counts.get(level_key, 0) + int(count)

    # Quebec census division rows.
    is_cd = chunk["GEO_LEVEL_NORM"].eq("census division")
    is_qc_cd = is_cd & is_quebec_census_division_dguid(chunk["DGUID"])

    qc_cd = chunk.loc[is_qc_cd].copy()
    quebec_cd_rows += len(qc_cd)

    if not qc_cd.empty:
        # Unique CD preview.
        for _, row in qc_cd[["DGUID", "GEO_NAME", "GEO_LEVEL"]].drop_duplicates().iterrows():
            dguid = str(row["DGUID"])
            quebec_cd_unique[dguid] = {
                "DGUID": dguid,
                "GEO_NAME": row["GEO_NAME"],
                "GEO_LEVEL": row["GEO_LEVEL"],
            }

        # Characteristic counts.
        if "CHARACTERISTIC_ID" in qc_cd.columns:
            for char_id, count in qc_cd["CHARACTERISTIC_ID"].value_counts(dropna=False).items():
                char_key = str(char_id)
                quebec_cd_characteristic_counts[char_key] = (
                    quebec_cd_characteristic_counts.get(char_key, 0) + int(count)
                )

        # Keyword audit on unique characteristic names.
        if "CHARACTERISTIC_NAME" in qc_cd.columns:
            candidate_chars = qc_cd[
                [
                    col for col in [
                        "CHARACTERISTIC_ID",
                        "CHARACTERISTIC_NAME",
                        "CHARACTERISTIC_NOTE",
                    ]
                    if col in qc_cd.columns
                ]
            ].drop_duplicates()

            for _, char_row in candidate_chars.iterrows():
                char_name = char_row.get("CHARACTERISTIC_NAME", "")
                matched_groups = keyword_match_group(char_name)

                if matched_groups:
                    quebec_cd_keyword_rows.append(
                        {
                            "matched_keyword_groups": "; ".join(matched_groups),
                            "CHARACTERISTIC_ID": char_row.get("CHARACTERISTIC_ID", ""),
                            "CHARACTERISTIC_NAME": char_name,
                            "CHARACTERISTIC_NOTE": char_row.get("CHARACTERISTIC_NOTE", ""),
                        }
                    )

    if chunk_idx % 10 == 0:
        print(f"  Processed chunks: {chunk_idx}, rows so far: {total_rows}")


# -----------------------------
# Build outputs
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

quebec_cd_preview_df = (
    pd.DataFrame(quebec_cd_unique.values())
    .sort_values(["DGUID"])
    .reset_index(drop=True)
    if quebec_cd_unique
    else pd.DataFrame(columns=["DGUID", "GEO_NAME", "GEO_LEVEL"])
)

quebec_cd_characteristic_counts_df = (
    pd.DataFrame(
        [
            {
                "CHARACTERISTIC_ID": char_id,
                "row_count_in_quebec_cd_rows": count,
            }
            for char_id, count in quebec_cd_characteristic_counts.items()
        ]
    )
    .sort_values("CHARACTERISTIC_ID")
    .reset_index(drop=True)
    if quebec_cd_characteristic_counts
    else pd.DataFrame(columns=["CHARACTERISTIC_ID", "row_count_in_quebec_cd_rows"])
)

keyword_audit_df = (
    pd.DataFrame(quebec_cd_keyword_rows)
    .drop_duplicates()
    .sort_values(["matched_keyword_groups", "CHARACTERISTIC_ID"])
    .reset_index(drop=True)
    if quebec_cd_keyword_rows
    else pd.DataFrame(
        columns=[
            "matched_keyword_groups",
            "CHARACTERISTIC_ID",
            "CHARACTERISTIC_NAME",
            "CHARACTERISTIC_NOTE",
        ]
    )
)

unique_quebec_cd_count = len(quebec_cd_preview_df)

has_census_division_rows = (
    geo_level_counts_df["GEO_LEVEL"]
    .astype(str)
    .str.lower()
    .str.strip()
    .eq("census division")
    .any()
)

has_98_quebec_census_divisions = unique_quebec_cd_count == 98

summary_df = pd.DataFrame(
    [
        {
            "metric": "raw_file",
            "value": str(raw_path.relative_to(DATA_DIR)),
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
            "metric": "has_any_census_division_geo_level",
            "value": has_census_division_rows,
        },
        {
            "metric": "quebec_census_division_rows",
            "value": quebec_cd_rows,
        },
        {
            "metric": "unique_quebec_census_division_dguids",
            "value": unique_quebec_cd_count,
        },
        {
            "metric": "has_98_quebec_census_divisions",
            "value": has_98_quebec_census_divisions,
        },
        {
            "metric": "recommended_next_step",
            "value": (
                "Reuse existing Census Profile raw file for CD-level SoVI extraction."
                if has_98_quebec_census_divisions
                else "Download or locate Census Profile data at Census division geography."
            ),
        },
    ]
)


# -----------------------------
# Save
# -----------------------------

geo_level_counts_df.to_csv(OUTPUT_GEO_LEVEL_COUNTS, index=False)
quebec_cd_preview_df.to_csv(OUTPUT_QUEBEC_CD_PREVIEW, index=False)
quebec_cd_characteristic_counts_df.to_csv(
    OUTPUT_QUEBEC_CD_CHARACTERISTIC_COUNTS,
    index=False,
)
keyword_audit_df.to_csv(OUTPUT_SOVI_KEYWORD_AUDIT, index=False)
summary_df.to_csv(OUTPUT_SUMMARY, index=False)


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CENSUS PROFILE GEO-LEVEL INSPECTION FOR SoVI")
print("=" * 72)

print("\nTotal rows scanned:", total_rows)

print("\nGEO_LEVEL counts:")
print(geo_level_counts_df.to_string(index=False))

print("\nQuebec Census Division detection:")
print("Rows where GEO_LEVEL == Census division and DGUID starts 2021A000324:", quebec_cd_rows)
print("Unique Quebec CD DGUIDs:", unique_quebec_cd_count)
print("Has 98 Quebec census divisions:", has_98_quebec_census_divisions)

if not quebec_cd_preview_df.empty:
    print("\nQuebec CD preview:")
    print(quebec_cd_preview_df.head(20).to_string(index=False))

if not keyword_audit_df.empty:
    print("\nPotential SoVI-relevant characteristic names found at Quebec CD level:")
    print(
        keyword_audit_df[
            [
                "matched_keyword_groups",
                "CHARACTERISTIC_ID",
                "CHARACTERISTIC_NAME",
            ]
        ]
        .head(60)
        .to_string(index=False)
    )
else:
    print("\nNo SoVI keyword matches found at Quebec CD level.")

print("\nRecommended next step:")
print(summary_df.loc[summary_df["metric"] == "recommended_next_step", "value"].iloc[0])

print("\nSaved:")
print(OUTPUT_GEO_LEVEL_COUNTS)
print(OUTPUT_QUEBEC_CD_PREVIEW)
print(OUTPUT_QUEBEC_CD_CHARACTERISTIC_COUNTS)
print(OUTPUT_SOVI_KEYWORD_AUDIT)
print(OUTPUT_SUMMARY)

print("\nDone.")