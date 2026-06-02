from pathlib import Path
import re
import pandas as pd


# ============================================================
# SVI CHARACTERISTIC DISCOVERY SCRIPT
# ============================================================
#
# Purpose:
#   Search the 2021 Statistics Canada Census Profile file for
#   candidate CHARACTERISTIC_ID values related to the SVI feature set.
#
# This script does NOT clean or extract features.
# It only discovers candidate Census Profile rows.
#
# Run from data/:
#   python census_profile_2021/inspect_svi_characteristics.py
#
# Expected raw file:
#   data/census_profile_2021/98-401-X2021007_English_CSV_data.csv
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent

RAW_PROFILE_PATH = THIS_DIR / "98-401-X2021007_English_CSV_data.csv"

OUTPUT_DIR = THIS_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "svi_characteristic_candidates_2021.csv"
OUTPUT_MARKDOWN = OUTPUT_DIR / "svi_characteristic_candidates_2021.md"


# -----------------------------
# Target search patterns
# -----------------------------

TARGETS = {
    # Already partially handled, but included for inventory completeness.
    "low_income": [
        r"\blow[- ]income\b",
        r"\bLIM[- ]AT\b",
        r"\bLICO[- ]AT\b",
        r"low-income measure",
        r"low-income cut-offs",
    ],

    "unemployment_labour_force": [
        r"\bunemployed\b",
        r"\bunemployment\b",
        r"\blabou?r force\b",
        r"\bemployment rate\b",
        r"\bparticipation rate\b",
        r"\bemployed\b",
        r"\bnot in the labou?r force\b",
    ],

    "income": [
        r"\bmedian.*income\b",
        r"\baverage.*income\b",
        r"\btotal income\b",
        r"\bafter-tax income\b",
        r"\bmarket income\b",
        r"\bemployment income\b",
        r"\bincome.*individuals\b",
        r"\bincome.*households\b",
        r"\bincome.*families\b",
    ],

    "education_no_high_school": [
        r"\bno certificate, diploma or degree\b",
        r"\bno certificate\b",
        r"\bhigh school diploma\b",
        r"\bhighest certificate\b",
        r"\beducational attainment\b",
        r"\bcertificate, diploma or degree\b",
        r"\bpopulation aged 15 years and over by highest certificate",
        r"\bpopulation aged 25 years and over by highest certificate",
    ],

    "age_children_elderly": [
        r"\b0 to 14 years\b",
        r"\b0 to 4 years\b",
        r"\b5 to 9 years\b",
        r"\b10 to 14 years\b",
        r"\b15 to 19 years\b",
        r"\b65 years and over\b",
        r"\b65 to 69 years\b",
        r"\b70 to 74 years\b",
        r"\b75 to 79 years\b",
        r"\b80 to 84 years\b",
        r"\b85 years and over\b",
        r"\bage groups of the population\b",
    ],

    "single_parent_households": [
        r"\blone-parent\b",
        r"\bsingle-parent\b",
        r"\bone-parent\b",
        r"\bfamilies with children\b",
        r"\bcensus families\b",
        r"\bfamily structure\b",
        r"\bchildren in census families\b",
    ],

    "visible_minority_immigration": [
        r"\bvisible minority\b",
        r"\bracialized\b",
        r"\bimmigrant\b",
        r"\brecent immigrant\b",
        r"\bnon-permanent resident\b",
        r"\bgeneration status\b",
        r"\bplace of birth\b",
        r"\bethnocultural\b",
        r"\bcitizenship\b",
    ],

    "language_barrier": [
        r"\bknowledge of official languages\b",
        r"\bEnglish and French\b",
        r"\bneither English nor French\b",
        r"\bEnglish only\b",
        r"\bFrench only\b",
        r"\blanguage spoken most often at home\b",
        r"\bmother tongue\b",
        r"\bnon-official language\b",
        r"\bofficial language minority\b",
        r"\bfirst official language spoken\b",
    ],

    "housing_type_multiunit_mobile": [
        r"\bstructural type of dwelling\b",
        r"\bapartment\b",
        r"\bapartment in a building\b",
        r"\bduplex\b",
        r"\brow house\b",
        r"\bsemi-detached\b",
        r"\bsingle-detached\b",
        r"\bmovable dwelling\b",
        r"\bmobile home\b",
        r"\bother movable dwelling\b",
    ],

    "crowding_housing_suitability": [
        r"\bhousing suitability\b",
        r"\bnot suitable\b",
        r"\bsuitable\b",
        r"\bpersons per room\b",
        r"\bmore than one person per room\b",
        r"\bcrowding\b",
        r"\bovercrowd",
        r"\brooms\b",
        r"\bbedrooms\b",
    ],

    "collective_dwellings_group_quarters": [
        r"\bcollective dwelling\b",
        r"\bcollective dwellings\b",
        r"\bpopulation in collective dwellings\b",
        r"\binstitutional\b",
        r"\bnon-institutional\b",
        r"\bnursing home\b",
        r"\bresidence for senior citizens\b",
        r"\bshelter\b",
        r"\blodging or rooming house\b",
    ],

    "vehicle_transport_proxy": [
        r"\bvehicle\b",
        r"\bcar\b",
        r"\bautomobile\b",
        r"\bcommuting destination\b",
        r"\bmain mode of commuting\b",
        r"\bpublic transit\b",
        r"\bwalked\b",
        r"\bcycling\b",
        r"\bcommute\b",
        r"\bplace of work\b",
    ],

    "disability_proxy": [
        r"\bdisability\b",
        r"\bactivity limitation\b",
        r"\blimitation\b",
        r"\bmobility\b",
        r"\bcare\b",
        r"\bneeds assistance\b",
    ],
}


# -----------------------------
# Helpers
# -----------------------------

def compile_patterns(patterns):
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]


def find_targets(characteristic_name, compiled_target_patterns):
    if pd.isna(characteristic_name):
        return []

    name = str(characteristic_name)

    matched_targets = []

    for target_name, patterns in compiled_target_patterns.items():
        if any(pattern.search(name) for pattern in patterns):
            matched_targets.append(target_name)

    return matched_targets


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").strip()


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
    raise ValueError("No census tract rows found. Check that the correct Census Profile file was downloaded.")


# -----------------------------
# Build characteristic dictionary
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
# Search target patterns
# -----------------------------

compiled_target_patterns = {
    target: compile_patterns(patterns)
    for target, patterns in TARGETS.items()
}

records = []

for _, row in characteristics.iterrows():
    characteristic_id = row["CHARACTERISTIC_ID"]
    characteristic_name = clean_text(row["CHARACTERISTIC_NAME"])
    characteristic_note = clean_text(row["CHARACTERISTIC_NOTE"])

    matched_targets = find_targets(
        characteristic_name,
        compiled_target_patterns,
    )

    for target in matched_targets:
        records.append(
            {
                "target_concept": target,
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
    [
        "target_concept",
        "characteristic_id",
    ]
).reset_index(drop=True)


# -----------------------------
# Add basic availability diagnostics
# -----------------------------

# For each candidate characteristic, inspect the number of census tract rows,
# missing values, symbols, and whether count/rate values exist.
diagnostics = []

candidate_ids = candidates["characteristic_id"].dropna().unique().tolist()

subset = tracts[
    tracts["CHARACTERISTIC_ID"].isin(candidate_ids)
].copy()

subset["count_value"] = pd.to_numeric(
    subset["C1_COUNT_TOTAL"],
    errors="coerce",
)

subset["rate_value"] = pd.to_numeric(
    subset["C10_RATE_TOTAL"],
    errors="coerce",
)

for characteristic_id, group in subset.groupby("CHARACTERISTIC_ID"):
    n_rows = len(group)
    count_non_missing = group["count_value"].notna().sum()
    rate_non_missing = group["rate_value"].notna().sum()
    count_missing = group["count_value"].isna().sum()
    rate_missing = group["rate_value"].isna().sum()

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
            "n_census_tract_rows": n_rows,
            "count_non_missing": int(count_non_missing),
            "count_missing": int(count_missing),
            "rate_non_missing": int(rate_non_missing),
            "rate_missing": int(rate_missing),
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
# Print summary
# -----------------------------

print("\n--- CANDIDATE COUNTS BY TARGET CONCEPT ---")
print(candidates["target_concept"].value_counts().sort_index().to_string())

print("\n--- CANDIDATES ---")
for target in sorted(candidates["target_concept"].unique()):
    target_df = candidates[candidates["target_concept"] == target].copy()

    print("\n" + "=" * 80)
    print(target)
    print("=" * 80)

    display_cols = [
        "characteristic_id",
        "characteristic_name",
        "n_census_tract_rows",
        "count_non_missing",
        "rate_non_missing",
        "symbol_counts",
    ]

    print(target_df[display_cols].to_string(index=False))


# -----------------------------
# Save CSV output
# -----------------------------

candidates.to_csv(OUTPUT_CSV, index=False)


# -----------------------------
# Save Markdown output
# -----------------------------

with open(OUTPUT_MARKDOWN, "w", encoding="utf-8") as f:
    f.write("# SVI Characteristic Candidate Inventory — 2021 Census Profile\n\n")
    f.write("This file was generated by `inspect_svi_characteristics.py`.\n\n")
    f.write("It searches the 2021 Census Profile file for candidate `CHARACTERISTIC_ID` values related to the SVI feature set.\n\n")
    f.write("The source file is:\n\n")
    f.write("```text\n")
    f.write(str(RAW_PROFILE_PATH) + "\n")
    f.write("```\n\n")
    f.write("The search is restricted to:\n\n")
    f.write("```text\n")
    f.write("GEO_LEVEL == Census tract\n")
    f.write("```\n\n")

    f.write("## Candidate counts by target concept\n\n")
    counts = candidates["target_concept"].value_counts().sort_index()
    f.write("| Target concept | Candidate rows |\n")
    f.write("|---|---:|\n")
    for target, count in counts.items():
        f.write(f"| `{target}` | {count} |\n")

    for target in sorted(candidates["target_concept"].unique()):
        target_df = candidates[candidates["target_concept"] == target].copy()

        f.write(f"\n## {target}\n\n")
        f.write("| Characteristic ID | Characteristic name | Count non-missing | Rate non-missing | Symbol counts |\n")
        f.write("|---:|---|---:|---:|---|\n")

        for _, row in target_df.iterrows():
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