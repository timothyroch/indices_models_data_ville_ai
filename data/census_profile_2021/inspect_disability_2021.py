from pathlib import Path
import re
import pandas as pd


# ============================================================
# Inspect Disability / Activity Limitation Candidates
# ============================================================
#
# Purpose:
#   Search the 2021 Census Profile for disability-like variables that could
#   potentially support an SVI-style disability feature.
#
# Why this exists:
#   The SVI requires:
#       percent persons aged 5+ with a disability
#
#   We do not yet know whether the Canadian 2021 Census Profile contains an
#   equivalent tract-level disability variable. This script searches the full
#   CHARACTERISTIC_NAME inventory for relevant terms and prints candidate
#   characteristics, plus nearby context around any matches.
#
# This is only an inspection script.
# It does not clean or extract final features.
#
# Run from data/:
#   python census_profile_2021/inspect_disability_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

THIS_DIR = Path(__file__).resolve().parent

RAW_PROFILE_PATH = THIS_DIR / "98-401-X2021007_English_CSV_data.csv"

OUTPUT_DIR = THIS_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_MATCHES_CSV = OUTPUT_DIR / "disability_2021_candidate_matches.csv"
OUTPUT_CONTEXT_CSV = OUTPUT_DIR / "disability_2021_candidate_context_windows.csv"
OUTPUT_MD = OUTPUT_DIR / "disability_2021_inspection.md"


# -----------------------------
# Keyword groups
# -----------------------------
# Some terms may produce false positives.
#
# For example:
#   "mobility" may refer to migration / residential mobility, not disability.
#   "activity" may refer to labour-market work activity, not activity limitation.
#
# That is why the script prints the exact characteristic names and context
# around matches. We should not accept a variable simply because it matched
# one keyword.

KEYWORD_GROUPS = {
    "direct_disability_terms": [
        "disability",
        "disabilities",
        "disabled",
    ],
    "activity_limitation_terms": [
        "activity limitation",
        "activity limitations",
        "limitation",
        "limitations",
    ],
    "difficulty_terms": [
        "difficulty",
        "difficulties",
    ],
    "functional_domain_terms": [
        "mobility",
        "hearing",
        "seeing",
        "vision",
        "walking",
        "cognitive",
        "mental",
        "physical",
        "learning",
        "developmental",
    ],
    "care_assistance_terms": [
        "assistance",
        "care",
        "needs help",
        "requires help",
    ],
}


# -----------------------------
# Helpers
# -----------------------------

def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def find_keyword_groups(name: str) -> list[str]:
    text = normalize_text(name).lower()
    matched_groups = []

    for group_name, keywords in KEYWORD_GROUPS.items():
        for keyword in keywords:
            if keyword.lower() in text:
                matched_groups.append(group_name)
                break

    return matched_groups


def find_keywords(name: str) -> list[str]:
    text = normalize_text(name).lower()
    matched_keywords = []

    for keywords in KEYWORD_GROUPS.values():
        for keyword in keywords:
            if keyword.lower() in text:
                matched_keywords.append(keyword)

    return matched_keywords


# -----------------------------
# Load needed columns
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


# -----------------------------
# Prepare ID column
# -----------------------------

df["CHARACTERISTIC_ID"] = pd.to_numeric(
    df["CHARACTERISTIC_ID"],
    errors="coerce",
).astype("Int64")

df["CHARACTERISTIC_NAME_CLEAN"] = df["CHARACTERISTIC_NAME"].map(normalize_text)


# -----------------------------
# Build global characteristic inventory
# -----------------------------

global_inventory = (
    df[
        [
            "CHARACTERISTIC_ID",
            "CHARACTERISTIC_NAME",
            "CHARACTERISTIC_NOTE",
        ]
    ]
    .drop_duplicates()
    .dropna(subset=["CHARACTERISTIC_ID"])
    .sort_values("CHARACTERISTIC_ID")
    .copy()
)

global_inventory["CHARACTERISTIC_NOTE"] = global_inventory[
    "CHARACTERISTIC_NOTE"
].fillna("")

global_inventory["matched_keyword_groups"] = global_inventory[
    "CHARACTERISTIC_NAME"
].apply(find_keyword_groups)

global_inventory["matched_keywords"] = global_inventory[
    "CHARACTERISTIC_NAME"
].apply(find_keywords)

matches = global_inventory[
    global_inventory["matched_keyword_groups"].apply(len) > 0
].copy()


# -----------------------------
# Add census tract diagnostics for matched IDs
# -----------------------------

ct_rows = df[df["GEO_LEVEL"].astype(str).str.strip() == "Census tract"].copy()

diagnostics = []

for characteristic_id, group in ct_rows.groupby("CHARACTERISTIC_ID"):
    count_values = pd.to_numeric(group["C1_COUNT_TOTAL"], errors="coerce")
    rate_values = pd.to_numeric(group["C10_RATE_TOTAL"], errors="coerce")

    symbol_counts = (
        group["SYMBOL"]
        .astype("string")
        .fillna("")
        .value_counts()
        .to_dict()
    )

    diagnostics.append(
        {
            "CHARACTERISTIC_ID": int(characteristic_id),
            "n_census_tract_rows": len(group),
            "count_non_missing": int(count_values.notna().sum()),
            "count_missing": int(count_values.isna().sum()),
            "rate_non_missing": int(rate_values.notna().sum()),
            "rate_missing": int(rate_values.isna().sum()),
            "symbol_counts": symbol_counts,
        }
    )

diagnostics_df = pd.DataFrame(diagnostics)

matches = matches.merge(
    diagnostics_df,
    on="CHARACTERISTIC_ID",
    how="left",
)

matches["present_at_census_tract_level"] = matches[
    "n_census_tract_rows"
].notna()


# -----------------------------
# Build context windows around matched IDs
# -----------------------------
# This helps reveal whether a match is actually part of a disability block or
# a false positive from another topic, such as migration mobility or work
# activity.

CONTEXT_RADIUS = 6

context_ids = set()

for characteristic_id in matches["CHARACTERISTIC_ID"].dropna().astype(int):
    for nearby_id in range(characteristic_id - CONTEXT_RADIUS, characteristic_id + CONTEXT_RADIUS + 1):
        context_ids.add(nearby_id)

context = global_inventory[
    global_inventory["CHARACTERISTIC_ID"].astype(int).isin(context_ids)
].copy()

context = context.merge(
    diagnostics_df,
    on="CHARACTERISTIC_ID",
    how="left",
)

context["is_direct_keyword_match"] = context["CHARACTERISTIC_ID"].isin(
    matches["CHARACTERISTIC_ID"]
)

context["present_at_census_tract_level"] = context[
    "n_census_tract_rows"
].notna()

context = context.sort_values("CHARACTERISTIC_ID")


# -----------------------------
# Print results
# -----------------------------

print("\n--- DISABILITY / ACTIVITY LIMITATION CANDIDATE MATCHES ---\n")

if matches.empty:
    print("No disability-like keyword matches were found in the Census Profile characteristic inventory.")
else:
    display_cols = [
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "matched_keyword_groups",
        "matched_keywords",
        "present_at_census_tract_level",
        "count_non_missing",
        "rate_non_missing",
        "symbol_counts",
    ]

    print(matches[display_cols].to_string(index=False))


print("\n--- CONTEXT WINDOWS AROUND MATCHES ---\n")

if context.empty:
    print("No context windows were generated because there were no matches.")
else:
    context_display_cols = [
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "is_direct_keyword_match",
        "present_at_census_tract_level",
        "count_non_missing",
        "rate_non_missing",
        "symbol_counts",
    ]

    print(context[context_display_cols].to_string(index=False))


# -----------------------------
# Simple interpretation hints
# -----------------------------

print("\n--- INTERPRETATION HINTS ---\n")

if matches.empty:
    print("Likely result: no exact disability variable exists in this Census Profile file.")
else:
    print("Review the matched names carefully.")
    print("Useful exact SVI-style candidates would look like:")
    print("  - persons aged 5 years and over with a disability")
    print("  - population with disability")
    print("  - activity limitation")
    print("  - difficulty seeing/hearing/walking/etc.")
    print("")
    print("False positives may include:")
    print("  - mobility status = residential migration, not disability")
    print("  - work activity = employment activity, not activity limitation")
    print("  - care = childcare or unpaid care, not disability")


# -----------------------------
# Save outputs
# -----------------------------

matches.to_csv(OUTPUT_MATCHES_CSV, index=False)
context.to_csv(OUTPUT_CONTEXT_CSV, index=False)

with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    f.write("# Disability 2021 Inspection\n\n")
    f.write("This file was generated by `inspect_disability_2021.py`.\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This audit searches the 2021 Census Profile characteristic inventory "
        "for disability-like variables that could potentially support an "
        "SVI-style disability feature.\n\n"
    )

    f.write("The target SVI concept is:\n\n")
    f.write("```text\n")
    f.write("percent persons aged 5+ with a disability\n")
    f.write("```\n\n")

    f.write("## Keyword groups searched\n\n")
    for group_name, keywords in KEYWORD_GROUPS.items():
        f.write(f"### {group_name}\n\n")
        f.write("```text\n")
        f.write(", ".join(keywords) + "\n")
        f.write("```\n\n")

    f.write("## Candidate matches\n\n")

    if matches.empty:
        f.write("No disability-like keyword matches were found.\n\n")
    else:
        f.write(
            "| Characteristic ID | Characteristic name | Matched groups | "
            "Matched keywords | Census tract level | Count non-missing | "
            "Rate non-missing | Symbol counts |\n"
        )
        f.write("|---:|---|---|---|---|---:|---:|---|\n")

        for _, row in matches.iterrows():
            f.write(
                "| "
                + str(row["CHARACTERISTIC_ID"])
                + " | "
                + str(row["CHARACTERISTIC_NAME"]).replace("|", "\\|")
                + " | "
                + str(row["matched_keyword_groups"]).replace("|", "\\|")
                + " | "
                + str(row["matched_keywords"]).replace("|", "\\|")
                + " | "
                + str(row["present_at_census_tract_level"])
                + " | "
                + str(row.get("count_non_missing", ""))
                + " | "
                + str(row.get("rate_non_missing", ""))
                + " | "
                + str(row.get("symbol_counts", "")).replace("|", "\\|")
                + " |\n"
            )

    f.write("\n## Context windows around matches\n\n")
    f.write(
        "The context window includes nearby characteristic IDs around every "
        "keyword match. This helps distinguish true disability variables from "
        "false positives such as residential mobility or labour-market work activity.\n\n"
    )

    if context.empty:
        f.write("No context windows were generated because there were no matches.\n\n")
    else:
        f.write(
            "| Characteristic ID | Direct keyword match | Characteristic name | "
            "Census tract level | Count non-missing | Rate non-missing | Symbol counts |\n"
        )
        f.write("|---:|---|---|---|---:|---:|---|\n")

        for _, row in context.iterrows():
            f.write(
                "| "
                + str(row["CHARACTERISTIC_ID"])
                + " | "
                + str(row["is_direct_keyword_match"])
                + " | "
                + str(row["CHARACTERISTIC_NAME"]).replace("|", "\\|")
                + " | "
                + str(row["present_at_census_tract_level"])
                + " | "
                + str(row.get("count_non_missing", ""))
                + " | "
                + str(row.get("rate_non_missing", ""))
                + " | "
                + str(row.get("symbol_counts", "")).replace("|", "\\|")
                + " |\n"
            )

    f.write("\n## Interpretation notes\n\n")
    f.write(
        "A valid SVI-style disability feature would need to measure disability "
        "or activity limitation directly. Matches involving residential mobility, "
        "labour-force work activity, or generic care terms should not be treated "
        "as disability variables without further evidence.\n\n"
    )

    f.write("If no direct disability/activity-limitation variable is found, then the Census Profile should be treated as missing the exact SVI disability variable.\n")

print("\nSaved:")
print(OUTPUT_MATCHES_CSV)
print(OUTPUT_CONTEXT_CSV)
print(OUTPUT_MD)

print("\nDone.")