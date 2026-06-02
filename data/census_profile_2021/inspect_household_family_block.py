from pathlib import Path
import pandas as pd


# ============================================================
# Inspect Household / Family Census Profile Block
# ============================================================
#
# Purpose:
#   Print every Census Profile characteristic in the likely
#   household / family block at the census tract level.
#
# This is only an inspection script.
# It does not clean or extract final features.
#
# Why this exists:
#   The existing household_family_2021 cleaner appears coherent, but this
#   audit script verifies the exact local Census Profile block around the
#   selected household/family IDs before we treat the cleaner as research-grade.
#
# Run from data/:
#   python census_profile_2021/inspect_household_family_block.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

THIS_DIR = Path(__file__).resolve().parent

RAW_PROFILE_PATH = THIS_DIR / "98-401-X2021007_English_CSV_data.csv"

OUTPUT_DIR = THIS_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "household_family_block_40_120.csv"
OUTPUT_MD = OUTPUT_DIR / "household_family_block_40_120.md"


# -----------------------------
# Settings
# -----------------------------
# The current cleaner uses IDs:
#   50, 56, 71, 76, 77, 78, 79, 86, 89, 90, 92, 95, 96, 100, 105
#
# This range is intentionally wider to inspect the surrounding local block:
# - household size
# - census family structure
# - persons in census families
# - household type
# - non-census-family households

MIN_ID = 40
MAX_ID = 120


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
# Filter to census tracts and ID range
# -----------------------------

df["CHARACTERISTIC_ID"] = pd.to_numeric(
    df["CHARACTERISTIC_ID"],
    errors="coerce",
).astype("Int64")

block_rows = df[
    (df["GEO_LEVEL"].astype(str).str.strip() == "Census tract")
    & (df["CHARACTERISTIC_ID"] >= MIN_ID)
    & (df["CHARACTERISTIC_ID"] <= MAX_ID)
].copy()

print(f"\nRows in ID range {MIN_ID}-{MAX_ID}:", len(block_rows))

if block_rows.empty:
    raise ValueError("No rows found in requested characteristic ID range.")


# -----------------------------
# Build characteristic inventory
# -----------------------------

inventory = (
    block_rows[
        [
            "CHARACTERISTIC_ID",
            "CHARACTERISTIC_NAME",
            "CHARACTERISTIC_NOTE",
        ]
    ]
    .drop_duplicates()
    .sort_values("CHARACTERISTIC_ID")
    .copy()
)

inventory["CHARACTERISTIC_NOTE"] = inventory["CHARACTERISTIC_NOTE"].fillna("")


# -----------------------------
# Add diagnostics
# -----------------------------

diagnostics = []

for characteristic_id, group in block_rows.groupby("CHARACTERISTIC_ID"):
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

inventory = inventory.merge(
    diagnostics_df,
    on="CHARACTERISTIC_ID",
    how="left",
)


# -----------------------------
# Flag IDs used by current cleaner
# -----------------------------

CURRENT_CLEANER_IDS = {
    50,
    56,
    71,
    76,
    77,
    78,
    79,
    86,
    89,
    90,
    92,
    95,
    96,
    100,
    105,
}

inventory["used_in_current_household_family_cleaner"] = inventory[
    "CHARACTERISTIC_ID"
].isin(CURRENT_CLEANER_IDS)


# -----------------------------
# Print result
# -----------------------------

print("\n--- CHARACTERISTICS IN HOUSEHOLD / FAMILY BLOCK ---\n")

display_cols = [
    "CHARACTERISTIC_ID",
    "CHARACTERISTIC_NAME",
    "used_in_current_household_family_cleaner",
    "count_non_missing",
    "rate_non_missing",
    "symbol_counts",
]

print(inventory[display_cols].to_string(index=False))


# -----------------------------
# Print only currently used IDs
# -----------------------------

used = inventory[inventory["used_in_current_household_family_cleaner"]].copy()

print("\n--- IDS USED BY CURRENT household_family_2021 CLEANER ---\n")
print(used[display_cols].to_string(index=False))


# -----------------------------
# Save outputs
# -----------------------------

inventory.to_csv(OUTPUT_CSV, index=False)

with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    f.write("# Household / Family Block Inspection — Characteristic IDs 40–120\n\n")
    f.write("This file was generated by `inspect_household_family_block.py`.\n\n")
    f.write("The inspection is restricted to:\n\n")
    f.write("```text\n")
    f.write('GEO_LEVEL == "Census tract"\n')
    f.write(f"{MIN_ID} <= CHARACTERISTIC_ID <= {MAX_ID}\n")
    f.write("```\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This audit verifies the local Census Profile block around the "
        "`household_family_2021` cleaner. It helps confirm that the selected "
        "household, census-family, one-parent-family, and household-type IDs "
        "are not shifted or semantically misread.\n\n"
    )

    f.write("## Currently used cleaner IDs\n\n")
    f.write("```text\n")
    f.write(", ".join(str(x) for x in sorted(CURRENT_CLEANER_IDS)) + "\n")
    f.write("```\n\n")

    f.write("## Full inspected block\n\n")
    f.write(
        "| Characteristic ID | Used in current cleaner | Characteristic name | "
        "Count non-missing | Rate non-missing | Symbol counts |\n"
    )
    f.write("|---:|---|---|---:|---:|---|\n")

    for _, row in inventory.iterrows():
        f.write(
            "| "
            + str(row["CHARACTERISTIC_ID"])
            + " | "
            + str(row["used_in_current_household_family_cleaner"])
            + " | "
            + str(row["CHARACTERISTIC_NAME"]).replace("|", "\\|")
            + " | "
            + str(row["count_non_missing"])
            + " | "
            + str(row["rate_non_missing"])
            + " | "
            + str(row["symbol_counts"]).replace("|", "\\|")
            + " |\n"
        )

    f.write("\n## IDs used by current cleaner only\n\n")
    f.write(
        "| Characteristic ID | Characteristic name | Count non-missing | "
        "Rate non-missing | Symbol counts |\n"
    )
    f.write("|---:|---|---:|---:|---|\n")

    for _, row in used.iterrows():
        f.write(
            "| "
            + str(row["CHARACTERISTIC_ID"])
            + " | "
            + str(row["CHARACTERISTIC_NAME"]).replace("|", "\\|")
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
print(OUTPUT_MD)

print("\nDone.")