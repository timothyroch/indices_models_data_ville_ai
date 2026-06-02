from pathlib import Path
import re
import pandas as pd


# ============================================================
# Targeted Census Profile Feature-Family Inspection
# ============================================================
#
# Purpose:
#   Inspect candidate CHARACTERISTIC_ID values for remaining
#   SVI / SoVI / HGNN-relevant feature families:
#
#   - household_family
#   - language
#   - immigration_ethnocultural
#   - housing_tenure_costs
#   - commuting_transport
#   - occupation_industry
#   - sex_gender
#
# This script does NOT clean or extract final features.
# It helps us choose which characteristic IDs should be used
# in the next clean feature tables.
#
# Run from data/:
#   python census_profile_2021/inspect_remaining_feature_families.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

THIS_DIR = Path(__file__).resolve().parent

RAW_PROFILE_PATH = THIS_DIR / "98-401-X2021007_English_CSV_data.csv"

OUTPUT_DIR = THIS_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "remaining_feature_family_candidates_2021.csv"
OUTPUT_MARKDOWN = OUTPUT_DIR / "remaining_feature_family_candidates_2021.md"


# -----------------------------
# Targeted search patterns
# -----------------------------
# These are intentionally narrower than the huge discovery script.
# The goal is to reduce noise and make manual selection easier.

TARGETS = {
    "household_family": [
        r"\bcensus families\b",
        r"\bcouple families\b",
        r"\blone-parent\b",
        r"\bone-parent\b",
        r"\bfemale lone-parent\b",
        r"\bmale lone-parent\b",
        r"\bchildren in census families\b",
        r"\bfamily structure\b",
        r"\bhousehold type\b",
        r"\bprivate households by household size\b",
        r"\baverage size of households\b",
        r"\bpersons in private households\b",
    ],

    "language": [
        r"\bknowledge of official languages\b",
        r"\bEnglish only\b",
        r"\bFrench only\b",
        r"\bEnglish and French\b",
        r"\bneither English nor French\b",
        r"\bfirst official language spoken\b",
        r"\bmother tongue\b",
        r"\blanguage spoken most often at home\b",
        r"\bnon-official language\b",
        r"\bofficial language minority\b",
    ],

    "immigration_ethnocultural": [
        r"\bimmigrant status\b",
        r"\bimmigrants\b",
        r"\brecent immigrants\b",
        r"\bnon-permanent residents\b",
        r"\bcitizenship\b",
        r"\bplace of birth\b",
        r"\bgeneration status\b",
        r"\bvisible minority\b",
        r"\bnot a visible minority\b",
        r"\bpopulation group\b",
        r"\bIndigenous identity\b",
        r"\bethnic or cultural origin\b",
    ],

    "housing_tenure_costs": [
        r"\btenure\b",
        r"\bowner\b",
        r"\brenter\b",
        r"\brented\b",
        r"\bowned\b",
        r"\bshelter cost\b",
        r"\bshelter-cost-to-income\b",
        r"\bspending 30% or more\b",
        r"\bmajor repairs\b",
        r"\bcore housing need\b",
        r"\bmonthly shelter costs\b",
        r"\bvalue of dwelling\b",
        r"\bcondominium status\b",
    ],

    "commuting_transport": [
        r"\bmain mode of commuting\b",
        r"\bcar, truck or van\b",
        r"\bpublic transit\b",
        r"\bwalked\b",
        r"\bbicycle\b",
        r"\bcommuting destination\b",
        r"\bcommuting duration\b",
        r"\bplace of work\b",
        r"\bworked at home\b",
        r"\bno fixed workplace address\b",
        r"\busual place of work\b",
    ],

    "occupation_industry": [
        r"\boccupation\b",
        r"\bNational Occupational Classification\b",
        r"\bindustry\b",
        r"\bNorth American Industry Classification System\b",
        r"\bagriculture, forestry, fishing and hunting\b",
        r"\bmining, quarrying, and oil and gas extraction\b",
        r"\bmanufacturing\b",
        r"\btransportation and warehousing\b",
        r"\butilities\b",
        r"\bhealth care and social assistance\b",
        r"\baccommodation and food services\b",
        r"\bservice occupations\b",
        r"\bsales and service occupations\b",
        r"\btrades, transport and equipment operators\b",
    ],

    "sex_gender": [
        r"\bTotal - Gender\b",
        r"\bMen\+\b",
        r"\bWomen\+\b",
        r"\bTotal - Age groups of the population\b",
        r"\bmale\b",
        r"\bfemale\b",
        r"\bsex at birth\b",
        r"\bgender\b",
    ],
}


# -----------------------------
# Helpers
# -----------------------------

def compile_patterns(patterns):
    return [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").strip()


def matched_families(characteristic_name, compiled_patterns):
    if pd.isna(characteristic_name):
        return []

    name = str(characteristic_name)
    matches = []

    for family, patterns in compiled_patterns.items():
        if any(pattern.search(name) for pattern in patterns):
            matches.append(family)

    return matches


# -----------------------------
# Load Census Profile
# -----------------------------

usecols = [
    "DGUID",
    "GEO_LEVEL",
    "GEO_NAME",
    "CHARACTERISTIC_ID",
    "CHARACTERISTIC_NAME",
    "CHARACTERISTIC_NOTE",
    "C1_COUNT_TOTAL",
    "C10_RATE_TOTAL",
    "SYMBOL",
]

df = pd.read_csv(
    RAW_PROFILE_PATH,
    usecols=usecols,
    encoding="iso-8859-1",
    low_memory=False,
)

print("\nLoaded Census Profile")
print("Rows:", len(df))
print("Columns:", list(df.columns))


# -----------------------------
# Keep census tract rows only
# -----------------------------

tracts = df[df["GEO_LEVEL"].astype(str).str.strip() == "Census tract"].copy()

print("\nCensus tract rows:", len(tracts))

if tracts.empty:
    raise ValueError("No census tract rows found. Check the raw profile file.")


# -----------------------------
# Build unique characteristic table
# -----------------------------

characteristics = (
    tracts[
        [
            "CHARACTERISTIC_ID",
            "CHARACTERISTIC_NAME",
            "CHARACTERISTIC_NOTE",
        ]
    ]
    .drop_duplicates()
    .copy()
)

characteristics["CHARACTERISTIC_ID"] = pd.to_numeric(
    characteristics["CHARACTERISTIC_ID"],
    errors="coerce",
).astype("Int64")

characteristics = characteristics.sort_values("CHARACTERISTIC_ID")

print("\nUnique census tract characteristics:", len(characteristics))


# -----------------------------
# Match target families
# -----------------------------

compiled_patterns = {
    family: compile_patterns(patterns)
    for family, patterns in TARGETS.items()
}

records = []

for _, row in characteristics.iterrows():
    characteristic_id = row["CHARACTERISTIC_ID"]
    characteristic_name = clean_text(row["CHARACTERISTIC_NAME"])
    characteristic_note = clean_text(row["CHARACTERISTIC_NOTE"])

    families = matched_families(characteristic_name, compiled_patterns)

    for family in families:
        records.append(
            {
                "feature_family": family,
                "characteristic_id": characteristic_id,
                "characteristic_name": characteristic_name,
                "characteristic_note": characteristic_note,
            }
        )

candidates = pd.DataFrame(records)

if candidates.empty:
    print("\nNo candidates found.")
    raise SystemExit(0)

candidates = candidates.sort_values(
    ["feature_family", "characteristic_id"]
).reset_index(drop=True)


# -----------------------------
# Add diagnostics
# -----------------------------

candidate_ids = candidates["characteristic_id"].dropna().unique().tolist()

subset = tracts[tracts["CHARACTERISTIC_ID"].isin(candidate_ids)].copy()

subset["count_value"] = pd.to_numeric(
    subset["C1_COUNT_TOTAL"],
    errors="coerce",
)

subset["rate_value"] = pd.to_numeric(
    subset["C10_RATE_TOTAL"],
    errors="coerce",
)

diagnostics = []

for characteristic_id, group in subset.groupby("CHARACTERISTIC_ID"):
    symbols = (
        group["SYMBOL"]
        .astype("string")
        .fillna("")
        .value_counts()
        .to_dict()
    )

    diagnostics.append(
        {
            "characteristic_id": int(characteristic_id),
            "n_census_tract_rows": len(group),
            "count_non_missing": int(group["count_value"].notna().sum()),
            "count_missing": int(group["count_value"].isna().sum()),
            "rate_non_missing": int(group["rate_value"].notna().sum()),
            "rate_missing": int(group["rate_value"].isna().sum()),
            "symbol_counts": symbols,
        }
    )

diagnostics_df = pd.DataFrame(diagnostics)

candidates = candidates.merge(
    diagnostics_df,
    on="characteristic_id",
    how="left",
)


# -----------------------------
# Print readable console summary
# -----------------------------

print("\n--- CANDIDATE COUNTS BY FEATURE FAMILY ---")
print(candidates["feature_family"].value_counts().sort_index().to_string())

for family in sorted(candidates["feature_family"].unique()):
    family_df = candidates[candidates["feature_family"] == family].copy()

    print("\n" + "=" * 100)
    print(family)
    print("=" * 100)

    display_cols = [
        "characteristic_id",
        "characteristic_name",
        "n_census_tract_rows",
        "count_non_missing",
        "rate_non_missing",
        "symbol_counts",
    ]

    print(family_df[display_cols].to_string(index=False))


# -----------------------------
# Save CSV
# -----------------------------

candidates.to_csv(OUTPUT_CSV, index=False)


# -----------------------------
# Save Markdown
# -----------------------------

with open(OUTPUT_MARKDOWN, "w", encoding="utf-8") as f:
    f.write("# Remaining Feature-Family Candidate Inventory — 2021 Census Profile\n\n")
    f.write("This file was generated by `inspect_remaining_feature_families.py`.\n\n")
    f.write("It searches the 2021 Census Profile file for candidate `CHARACTERISTIC_ID` values related to remaining SVI, SoVI, and HGNN feature families.\n\n")

    f.write("## Source file\n\n")
    f.write("```text\n")
    f.write(str(RAW_PROFILE_PATH) + "\n")
    f.write("```\n\n")

    f.write("## Geographic filter\n\n")
    f.write("```text\n")
    f.write("GEO_LEVEL == Census tract\n")
    f.write("```\n\n")

    f.write("## Candidate counts by feature family\n\n")
    counts = candidates["feature_family"].value_counts().sort_index()

    f.write("| Feature family | Candidate rows |\n")
    f.write("|---|---:|\n")

    for family, count in counts.items():
        f.write(f"| `{family}` | {count} |\n")

    for family in sorted(candidates["feature_family"].unique()):
        family_df = candidates[candidates["feature_family"] == family].copy()

        f.write(f"\n## {family}\n\n")
        f.write("| Characteristic ID | Characteristic name | Count non-missing | Rate non-missing | Symbol counts |\n")
        f.write("|---:|---|---:|---:|---|\n")

        for _, row in family_df.iterrows():
            f.write(
                "| "
                + str(row["characteristic_id"])
                + " | "
                + str(row["characteristic_name"]).replace("|", "\\|")
                + " | "
                + str(row["count_non_missing"])
                + " | "
                + str(row["rate_non_missing"])
                + " | "
                + str(row["symbol_counts"]).replace("|", "\\|")
                + " |\n"
            )


print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_MARKDOWN)

print("\nDone.")