from pathlib import Path
import pandas as pd


# ============================================================
# Inspect Housing Type Census Profile Block
# ============================================================
#
# Purpose:
#   Print every Census Profile characteristic in the local housing-type /
#   structural-dwelling block at the census tract level.
#
# This is only an inspection/audit script.
# It does not clean or extract final features.
#
# Why this exists:
#   The current housing_type_2021 cleaner appears coherent, but this audit
#   verifies the exact local Census Profile block around the selected
#   structural-dwelling IDs before we treat the cleaner as research-grade.
#
# Run from data/:
#   python census_profile_2021/inspect_housing_type_block.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

THIS_DIR = Path(__file__).resolve().parent

RAW_PROFILE_PATH = THIS_DIR / "98-401-X2021007_English_CSV_data.csv"

OUTPUT_DIR = THIS_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "housing_type_block_35_60.csv"
OUTPUT_MD = OUTPUT_DIR / "housing_type_block_35_60.md"


# -----------------------------
# Settings
# -----------------------------
# The current cleaner uses IDs:
#   41, 42, 43, 44, 45, 46, 47, 49
#
# ID 48 is intentionally not used in the current cleaner, but this inspection
# will reveal its exact meaning. This matters because if 48 is a real
# structural-dwelling category, we may want to keep it as a raw column or
# include it in a non-single-detached / multiunit calculation.
#
# This range is intentionally wider to inspect the surrounding local block:
# - occupied private dwelling total
# - structural type of dwelling
# - transition into household size / household-family fields

MIN_ID = 35
MAX_ID = 60

CURRENT_CLEANER_IDS = {
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    49,
}


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

inventory["used_in_current_housing_type_cleaner"] = inventory[
    "CHARACTERISTIC_ID"
].isin(CURRENT_CLEANER_IDS)


# -----------------------------
# Print result
# -----------------------------

print("\n--- CHARACTERISTICS IN HOUSING TYPE BLOCK ---\n")

display_cols = [
    "CHARACTERISTIC_ID",
    "CHARACTERISTIC_NAME",
    "used_in_current_housing_type_cleaner",
    "count_non_missing",
    "rate_non_missing",
    "symbol_counts",
]

print(inventory[display_cols].to_string(index=False))


# -----------------------------
# Print only currently used IDs
# -----------------------------

used = inventory[
    inventory["used_in_current_housing_type_cleaner"]
].copy()

print("\n--- IDS USED BY CURRENT housing_type_2021 CLEANER ---\n")
print(used[display_cols].to_string(index=False))


# -----------------------------
# Highlight ID 48 separately
# -----------------------------

id_48 = inventory[inventory["CHARACTERISTIC_ID"] == 48].copy()

print("\n--- ID 48 CHECK ---\n")

if id_48.empty:
    print("ID 48 was not found in the inspected range.")
else:
    print(id_48[display_cols].to_string(index=False))


# -----------------------------
# Check whether any current cleaner IDs were not found
# -----------------------------

found_ids = set(inventory["CHARACTERISTIC_ID"].dropna().astype(int).unique())
missing_current_ids = sorted(CURRENT_CLEANER_IDS - found_ids)

if missing_current_ids:
    print("\nWARNING: Some current cleaner IDs were not found in inspected range:")
    for characteristic_id in missing_current_ids:
        print(f"  {characteristic_id}")
else:
    print("\nAll current cleaner IDs were found in the inspected range.")


# -----------------------------
# Save outputs
# -----------------------------

inventory.to_csv(OUTPUT_CSV, index=False)

with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    f.write("# Housing Type Block Inspection — Characteristic IDs 35–60\n\n")
    f.write("This file was generated by `inspect_housing_type_block.py`.\n\n")

    f.write("## Geographic filter\n\n")
    f.write("```text\n")
    f.write('GEO_LEVEL == "Census tract"\n')
    f.write("```\n\n")

    f.write("## Inspected range\n\n")
    f.write("```text\n")
    f.write(f"{MIN_ID} <= CHARACTERISTIC_ID <= {MAX_ID}\n")
    f.write("```\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This audit verifies the local Census Profile block around the "
        "`housing_type_2021` cleaner. It helps confirm that the selected "
        "structural-dwelling IDs are not shifted or semantically misread. "
        "It also checks ID 48, which is not currently used by the cleaner.\n\n"
    )

    f.write("## Currently used cleaner IDs\n\n")
    f.write("```text\n")
    f.write(", ".join(str(x) for x in sorted(CURRENT_CLEANER_IDS)) + "\n")
    f.write("```\n\n")

    if missing_current_ids:
        f.write("## Warning\n\n")
        f.write("The following current cleaner IDs were not found in the inspected range:\n\n")
        f.write("```text\n")
        f.write(", ".join(str(x) for x in missing_current_ids) + "\n")
        f.write("```\n\n")
    else:
        f.write("## Current cleaner ID coverage\n\n")
        f.write("All current cleaner IDs were found in the inspected range.\n\n")

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
            + str(row["used_in_current_housing_type_cleaner"])
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

    f.write("\n## ID 48 check\n\n")

    if id_48.empty:
        f.write("ID 48 was not found in the inspected range.\n")
    else:
        f.write(
            "| Characteristic ID | Characteristic name | Count non-missing | "
            "Rate non-missing | Symbol counts |\n"
        )
        f.write("|---:|---|---:|---:|---|\n")

        for _, row in id_48.iterrows():
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