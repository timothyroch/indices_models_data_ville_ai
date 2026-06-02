from pathlib import Path
import pandas as pd


# ============================================================
# Inspect No-Vehicle-Available Candidates
# ============================================================
#
# Purpose:
#   Search the 2021 Census Profile for vehicle-availability-like variables
#   that could potentially support an SVI-style "no vehicle available" feature.
#
# Why this exists:
#   The SVI requires:
#       percent households with no vehicle available
#
#   We do not yet know whether the Canadian 2021 Census Profile contains an
#   equivalent census-tract-level variable. This script searches the full
#   CHARACTERISTIC_NAME inventory for relevant terms and prints candidate
#   characteristics, plus nearby context around matches.
#
# Important:
#   Commuting mode is not the same as vehicle availability.
#   "Public transit commute" or "car/truck/van commute" should not automatically
#   be treated as household vehicle access.
#
# Run from data/:
#   python census_profile_2021/inspect_no_vehicle_available_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

THIS_DIR = Path(__file__).resolve().parent

RAW_PROFILE_PATH = THIS_DIR / "98-401-X2021007_English_CSV_data.csv"

OUTPUT_DIR = THIS_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_MATCHES_CSV = OUTPUT_DIR / "no_vehicle_available_2021_candidate_matches.csv"
OUTPUT_CONTEXT_CSV = OUTPUT_DIR / "no_vehicle_available_2021_candidate_context_windows.csv"
OUTPUT_MD = OUTPUT_DIR / "no_vehicle_available_2021_inspection.md"


# -----------------------------
# Keyword groups
# -----------------------------
# Some terms may produce false positives.
#
# For example:
#   "car, truck or van" usually appears in commuting mode, not household vehicle
#   availability.
#
# A useful exact candidate would look like:
#   - no vehicle available
#   - households with no vehicle
#   - vehicle availability
#   - number of vehicles available
#
# The script prints context so we can distinguish exact vehicle availability
# from commuting behavior.

KEYWORD_GROUPS = {
    "direct_no_vehicle_terms": [
        "no vehicle",
        "no vehicles",
        "without vehicle",
        "without vehicles",
        "no car",
        "no automobile",
        "no automobiles",
    ],
    "vehicle_availability_terms": [
        "vehicle available",
        "vehicles available",
        "vehicle availability",
        "number of vehicles",
        "vehicles per household",
        "automobiles available",
        "automobile available",
    ],
    "vehicle_general_terms": [
        "vehicle",
        "vehicles",
        "car",
        "cars",
        "automobile",
        "automobiles",
        "truck",
        "van",
    ],
    "commuting_transport_terms": [
        "commute",
        "commuting",
        "main mode",
        "mode of commuting",
        "public transit",
        "walked",
        "bicycle",
        "car, truck or van",
        "driver",
        "passenger",
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


def classify_candidate(name: str) -> str:
    """
    Lightweight classification to help interpretation.

    This does not decide final validity. It only labels obvious categories.
    """
    text = normalize_text(name).lower()

    exact_patterns = [
        "no vehicle",
        "no vehicles",
        "without vehicle",
        "without vehicles",
        "vehicle availability",
        "vehicles available",
        "number of vehicles",
    ]

    commuting_patterns = [
        "commute",
        "commuting",
        "main mode",
        "mode of commuting",
        "usual place of work",
        "car, truck or van",
        "public transit",
        "driver",
        "passenger",
        "walked",
        "bicycle",
    ]

    if any(pattern in text for pattern in exact_patterns):
        return "possible_exact_vehicle_availability_candidate"

    if any(pattern in text for pattern in commuting_patterns):
        return "likely_commuting_mode_or_workplace_false_positive"

    if "vehicle" in text or "car" in text or "truck" in text or "van" in text:
        return "vehicle_related_needs_manual_review"

    return "other_keyword_match_needs_manual_review"


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

matches["candidate_classification"] = matches["CHARACTERISTIC_NAME"].apply(
    classify_candidate
)


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

CONTEXT_RADIUS = 8

context_ids = set()

for characteristic_id in matches["CHARACTERISTIC_ID"].dropna().astype(int):
    for nearby_id in range(
        characteristic_id - CONTEXT_RADIUS,
        characteristic_id + CONTEXT_RADIUS + 1,
    ):
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

context["candidate_classification"] = context["CHARACTERISTIC_NAME"].apply(
    classify_candidate
)

context = context.sort_values("CHARACTERISTIC_ID")


# -----------------------------
# Print results
# -----------------------------

print("\n--- NO-VEHICLE-AVAILABLE CANDIDATE MATCHES ---\n")

if matches.empty:
    print("No vehicle-availability-like keyword matches were found in the Census Profile characteristic inventory.")
else:
    display_cols = [
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "matched_keyword_groups",
        "matched_keywords",
        "candidate_classification",
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
        "candidate_classification",
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
    print("Likely result: no exact no-vehicle-available variable exists in this Census Profile file.")
else:
    print("Review the matched names carefully.")
    print("")
    print("Useful exact SVI-style candidates would look like:")
    print("  - households with no vehicle available")
    print("  - no vehicle available")
    print("  - number of vehicles available")
    print("  - vehicle availability")
    print("")
    print("Likely false positives include:")
    print("  - car, truck or van as a commuting mode")
    print("  - public transit commute")
    print("  - commute destination")
    print("  - worked at home / no fixed workplace")
    print("")
    print("Commuting behavior can be useful for HGNN mobility context,")
    print("but it is not equivalent to household vehicle availability.")


# -----------------------------
# Save outputs
# -----------------------------

matches.to_csv(OUTPUT_MATCHES_CSV, index=False)
context.to_csv(OUTPUT_CONTEXT_CSV, index=False)

with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    f.write("# No Vehicle Available 2021 Inspection\n\n")
    f.write("This file was generated by `inspect_no_vehicle_available_2021.py`.\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This audit searches the 2021 Census Profile characteristic inventory "
        "for vehicle-availability-like variables that could potentially support "
        "an SVI-style no-vehicle-available feature.\n\n"
    )

    f.write("The target SVI concept is:\n\n")
    f.write("```text\n")
    f.write("percent households with no vehicle available\n")
    f.write("```\n\n")

    f.write("## Important distinction\n\n")
    f.write(
        "Commuting mode is not the same as household vehicle availability. "
        "For example, a person may commute by public transit while living in a "
        "household that owns a vehicle, or may commute as a passenger in a vehicle "
        "without their household owning one. Commuting variables should therefore "
        "not be treated as exact SVI no-vehicle variables.\n\n"
    )

    f.write("## Keyword groups searched\n\n")
    for group_name, keywords in KEYWORD_GROUPS.items():
        f.write(f"### {group_name}\n\n")
        f.write("```text\n")
        f.write(", ".join(keywords) + "\n")
        f.write("```\n\n")

    f.write("## Candidate matches\n\n")

    if matches.empty:
        f.write("No vehicle-availability-like keyword matches were found.\n\n")
    else:
        f.write(
            "| Characteristic ID | Characteristic name | Matched groups | "
            "Matched keywords | Classification | Census tract level | "
            "Count non-missing | Rate non-missing | Symbol counts |\n"
        )
        f.write("|---:|---|---|---|---|---|---:|---:|---|\n")

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
                + str(row["candidate_classification"]).replace("|", "\\|")
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
        "keyword match. This helps distinguish true household vehicle availability "
        "from commuting-mode variables.\n\n"
    )

    if context.empty:
        f.write("No context windows were generated because there were no matches.\n\n")
    else:
        f.write(
            "| Characteristic ID | Direct keyword match | Characteristic name | "
            "Classification | Census tract level | Count non-missing | "
            "Rate non-missing | Symbol counts |\n"
        )
        f.write("|---:|---|---|---|---|---:|---:|---|\n")

        for _, row in context.iterrows():
            f.write(
                "| "
                + str(row["CHARACTERISTIC_ID"])
                + " | "
                + str(row["is_direct_keyword_match"])
                + " | "
                + str(row["CHARACTERISTIC_NAME"]).replace("|", "\\|")
                + " | "
                + str(row["candidate_classification"]).replace("|", "\\|")
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
        "A valid SVI-style no-vehicle feature would need to measure household "
        "vehicle availability directly. Matches involving commuting mode, commuting "
        "destination, public transit, or car/truck/van commuting should not be "
        "treated as no-vehicle-available variables without further evidence.\n\n"
    )

    f.write(
        "If no direct vehicle-availability variable is found, then the Census Profile "
        "should be treated as missing the exact SVI no-vehicle variable.\n"
    )

print("\nSaved:")
print(OUTPUT_MATCHES_CSV)
print(OUTPUT_CONTEXT_CSV)
print(OUTPUT_MD)

print("\nDone.")