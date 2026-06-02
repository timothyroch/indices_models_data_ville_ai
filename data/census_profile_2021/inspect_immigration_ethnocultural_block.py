from pathlib import Path
import pandas as pd


# ============================================================
# Inspect Immigration / Ethnocultural Census Profile Blocks
# ============================================================
#
# Purpose:
#   Print Census Profile characteristics in the local blocks used by
#   immigration_ethnocultural_2021 at the census tract level.
#
# This is only an inspection/audit script.
# It does not clean or extract final features.
#
# Why this exists:
#   The current immigration_ethnocultural_2021 cleaner appears coherent, but
#   this audit verifies the exact local Census Profile blocks before we treat
#   the cleaner as research-grade.
#
# Run from data/:
#   python census_profile_2021/inspect_immigration_ethnocultural_block.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

THIS_DIR = Path(__file__).resolve().parent

RAW_PROFILE_PATH = THIS_DIR / "98-401-X2021007_English_CSV_data.csv"

OUTPUT_DIR = THIS_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "immigration_ethnocultural_block_audit.csv"
OUTPUT_MD = OUTPUT_DIR / "immigration_ethnocultural_block_audit.md"


# -----------------------------
# Settings
# -----------------------------
# The current cleaner uses IDs:
#   1402, 1403, 1410,
#   1522, 1527, 1528, 1529, 1537,
#   1544, 1604, 1665,
#   1670, 1673, 1675,
#   1683, 1684, 1695, 1697, 1698
#
# The relevant blocks are not fully contiguous:
# - Indigenous identity is around 1402-1410.
# - Citizenship / immigrant status / place of birth / generation status /
#   admission category / visible minority / ethnic origin are around 1522-1698.
#
# We inspect slightly wider local windows.

INSPECTION_RANGES = [
    {
        "block_name": "indigenous_identity_local_block",
        "min_id": 1395,
        "max_id": 1412,
    },
    {
        "block_name": "citizenship_immigration_ethnocultural_local_block",
        "min_id": 1515,
        "max_id": 1705,
    },
]

CURRENT_CLEANER_IDS = {
    1402,
    1403,
    1410,

    1522,
    1527,
    1528,
    1529,
    1537,

    1544,
    1604,
    1665,

    1670,
    1673,
    1675,

    1683,
    1684,
    1695,
    1697,
    1698,
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
# Prepare ID column
# -----------------------------

df["CHARACTERISTIC_ID"] = pd.to_numeric(
    df["CHARACTERISTIC_ID"],
    errors="coerce",
).astype("Int64")


# -----------------------------
# Build audit inventory
# -----------------------------

all_inventory_parts = []

for block in INSPECTION_RANGES:
    block_name = block["block_name"]
    min_id = block["min_id"]
    max_id = block["max_id"]

    block_rows = df[
        (df["GEO_LEVEL"].astype(str).str.strip() == "Census tract")
        & (df["CHARACTERISTIC_ID"] >= min_id)
        & (df["CHARACTERISTIC_ID"] <= max_id)
    ].copy()

    print(f"\nRows in {block_name} ({min_id}-{max_id}):", len(block_rows))

    if block_rows.empty:
        print(f"Warning: no rows found for {block_name}.")
        continue

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
    inventory["block_name"] = block_name

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

    all_inventory_parts.append(inventory)


if not all_inventory_parts:
    raise ValueError("No inspection rows found in any requested block.")


inventory = pd.concat(all_inventory_parts, ignore_index=True)

inventory["used_in_current_immigration_ethnocultural_cleaner"] = inventory[
    "CHARACTERISTIC_ID"
].isin(CURRENT_CLEANER_IDS)

inventory = inventory[
    [
        "block_name",
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
        "CHARACTERISTIC_NOTE",
        "used_in_current_immigration_ethnocultural_cleaner",
        "n_census_tract_rows",
        "count_non_missing",
        "count_missing",
        "rate_non_missing",
        "rate_missing",
        "symbol_counts",
    ]
].sort_values(["block_name", "CHARACTERISTIC_ID"])


# -----------------------------
# Print result
# -----------------------------

print("\n--- FULL IMMIGRATION / ETHNOCULTURAL BLOCK AUDIT ---\n")

display_cols = [
    "block_name",
    "CHARACTERISTIC_ID",
    "CHARACTERISTIC_NAME",
    "used_in_current_immigration_ethnocultural_cleaner",
    "count_non_missing",
    "rate_non_missing",
    "symbol_counts",
]

print(inventory[display_cols].to_string(index=False))


# -----------------------------
# Print only currently used IDs
# -----------------------------

used = inventory[
    inventory["used_in_current_immigration_ethnocultural_cleaner"]
].copy()

print("\n--- IDS USED BY CURRENT immigration_ethnocultural_2021 CLEANER ---\n")
print(used[display_cols].to_string(index=False))


# -----------------------------
# Check whether any current cleaner IDs were not found
# -----------------------------

found_ids = set(inventory["CHARACTERISTIC_ID"].dropna().astype(int).unique())
missing_current_ids = sorted(CURRENT_CLEANER_IDS - found_ids)

if missing_current_ids:
    print("\nWARNING: Some current cleaner IDs were not found in inspected ranges:")
    for characteristic_id in missing_current_ids:
        print(f"  {characteristic_id}")
else:
    print("\nAll current cleaner IDs were found in the inspected ranges.")


# -----------------------------
# Save outputs
# -----------------------------

inventory.to_csv(OUTPUT_CSV, index=False)

with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    f.write("# Immigration / Ethnocultural Block Inspection\n\n")
    f.write("This file was generated by `inspect_immigration_ethnocultural_block.py`.\n\n")

    f.write("## Geographic filter\n\n")
    f.write("```text\n")
    f.write('GEO_LEVEL == "Census tract"\n')
    f.write("```\n\n")

    f.write("## Inspected ranges\n\n")
    f.write("| Block name | Min ID | Max ID |\n")
    f.write("|---|---:|---:|\n")
    for block in INSPECTION_RANGES:
        f.write(
            "| "
            + block["block_name"]
            + " | "
            + str(block["min_id"])
            + " | "
            + str(block["max_id"])
            + " |\n"
        )

    f.write("\n## Purpose\n\n")
    f.write(
        "This audit verifies the local Census Profile blocks around the "
        "`immigration_ethnocultural_2021` cleaner. It helps confirm that the "
        "selected Indigenous identity, citizenship, immigrant status, generation "
        "status, visible minority, and ethnic/cultural origin IDs are not shifted "
        "or semantically misread.\n\n"
    )

    f.write("## Currently used cleaner IDs\n\n")
    f.write("```text\n")
    f.write(", ".join(str(x) for x in sorted(CURRENT_CLEANER_IDS)) + "\n")
    f.write("```\n\n")

    if missing_current_ids:
        f.write("## Warning\n\n")
        f.write("The following current cleaner IDs were not found in the inspected ranges:\n\n")
        f.write("```text\n")
        f.write(", ".join(str(x) for x in missing_current_ids) + "\n")
        f.write("```\n\n")
    else:
        f.write("## Current cleaner ID coverage\n\n")
        f.write("All current cleaner IDs were found in the inspected ranges.\n\n")

    f.write("## Full inspected block\n\n")
    f.write(
        "| Block | Characteristic ID | Used in current cleaner | "
        "Characteristic name | Count non-missing | Rate non-missing | Symbol counts |\n"
    )
    f.write("|---|---:|---|---|---:|---:|---|\n")

    for _, row in inventory.iterrows():
        f.write(
            "| "
            + str(row["block_name"])
            + " | "
            + str(row["CHARACTERISTIC_ID"])
            + " | "
            + str(row["used_in_current_immigration_ethnocultural_cleaner"])
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
        "| Block | Characteristic ID | Characteristic name | "
        "Count non-missing | Rate non-missing | Symbol counts |\n"
    )
    f.write("|---|---:|---|---:|---:|---|\n")

    for _, row in used.iterrows():
        f.write(
            "| "
            + str(row["block_name"])
            + " | "
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