from pathlib import Path
import pandas as pd


# ============================================================
# Fill Manual Repair for ODHF Quebec Hospitals Missing CSDuid
# ============================================================
#
# Purpose:
#   Fill the manual census-division assignment for the 22 Quebec ODHF
#   hospital records that are missing CSDuid.
#
# This script does NOT create the final hospitals-per-capita variable yet.
# It only fills the repair lookup cleanly and validates the assigned
# census division codes against the official Quebec CD reference.
#
# Input:
#   hospitals_per_capita/lookup/odhf_quebec_hospitals_missing_csd_manual_repair.csv
#   hospitals_per_capita/lookup/quebec_census_division_reference_for_manual_repair.csv
#
# Output:
#   hospitals_per_capita/lookup/odhf_quebec_hospitals_missing_csd_manual_repair_filled.csv
#
# Run from data/:
#   python hospitals_per_capita/fill_missing_csd_manual_repair.py
#
# ============================================================


DATA_DIR = Path(__file__).resolve().parent.parent

LOOKUP_DIR = DATA_DIR / "hospitals_per_capita" / "lookup"

INPUT_REPAIR_TEMPLATE = (
    LOOKUP_DIR / "odhf_quebec_hospitals_missing_csd_manual_repair.csv"
)

INPUT_CD_REFERENCE = (
    LOOKUP_DIR / "quebec_census_division_reference_for_manual_repair.csv"
)

OUTPUT_FILLED_REPAIR = (
    LOOKUP_DIR / "odhf_quebec_hospitals_missing_csd_manual_repair_filled.csv"
)


# -----------------------------
# Manual repair mapping
# -----------------------------
#
# Key = ODHF record index
# Value = official Quebec census division code
#
# CD codes are validated against:
#   quebec_census_division_reference_for_manual_repair.csv

MANUAL_INDEX_TO_CD_CODE = {
    "7014": "2493",  # Alma -> Lac-Saint-Jean-Est
    "6879": "2488",  # Amos -> Abitibi
    "6942": "2407",  # Amqui -> La Matapédia
    "6919": "2467",  # Châteauguay -> Roussillon
    "6986": "2449",  # Drummondville -> Drummond
    "6893": "2497",  # Fermont -> Sept-Rivières--Caniapiscau
    "6888": "2481",  # Gatineau / Hull -> Gatineau
    "6911": "2458",  # Greenfield Park -> Longueuil
    "6935": "2401",  # Les Îles-de-la-Madeleine -> Communauté maritime des Îles-de-la-Madeleine
    "6916": "2458",  # Longueuil -> Longueuil
    "6889": "2483",  # Maniwaki -> La Vallée-de-la-Gatineau
    "6885": "2484",  # Mansfield-et-Pontefract -> Pontiac
    "6908": "2406",  # Maria -> Avignon
    "6951": "2466",  # Montréal -> Montréal
    "6991": "2466",  # Montréal -> Montréal
    "6993": "2466",  # Montréal -> Montréal
    "7007": "2466",  # Montréal -> Montréal
    "S1": "2466",    # Montréal -> Montréal
    "S2": "2469",    # Ormstown -> Le Haut-Saint-Laurent
    "6944": "2410",  # Rimouski -> Rimouski-Neigette
    "6878": "2486",  # Rouyn-Noranda -> Rouyn-Noranda
    "6912": "2456",  # Saint-Jean-sur-Richelieu -> Le Haut-Richelieu
}


# -----------------------------
# Load files
# -----------------------------

if not INPUT_REPAIR_TEMPLATE.exists():
    raise FileNotFoundError(f"Repair template not found:\n{INPUT_REPAIR_TEMPLATE}")

if not INPUT_CD_REFERENCE.exists():
    raise FileNotFoundError(f"CD reference file not found:\n{INPUT_CD_REFERENCE}")

repair = pd.read_csv(INPUT_REPAIR_TEMPLATE, dtype=str)
cd_ref = pd.read_csv(INPUT_CD_REFERENCE, dtype=str)

print("\nLoaded manual repair template")
print("Rows:", len(repair))
print("Columns:", list(repair.columns))

print("\nLoaded census division reference")
print("Rows:", len(cd_ref))
print("Columns:", list(cd_ref.columns))


# -----------------------------
# Validate columns
# -----------------------------

required_repair_cols = [
    "index",
    "facility_name",
    "city",
    "CSDname",
    "manual_cd_code",
    "manual_census_division_name",
    "manual_census_division_dguid",
    "manual_repair_method",
    "manual_repair_note",
]

missing_repair_cols = [col for col in required_repair_cols if col not in repair.columns]

if missing_repair_cols:
    raise ValueError(
        "Missing required columns in repair template:\n"
        + "\n".join(missing_repair_cols)
    )

required_cd_ref_cols = [
    "manual_cd_code",
    "manual_census_division_dguid",
    "manual_census_division_name",
]

missing_cd_ref_cols = [col for col in required_cd_ref_cols if col not in cd_ref.columns]

if missing_cd_ref_cols:
    raise ValueError(
        "Missing required columns in CD reference:\n"
        + "\n".join(missing_cd_ref_cols)
    )


# -----------------------------
# Validate mapping keys
# -----------------------------

repair["index"] = repair["index"].astype(str).str.strip()

repair_indices = set(repair["index"])
mapping_indices = set(MANUAL_INDEX_TO_CD_CODE.keys())

missing_from_mapping = sorted(repair_indices - mapping_indices)
extra_in_mapping = sorted(mapping_indices - repair_indices)

if missing_from_mapping:
    raise ValueError(
        "Some repair-template records are missing from MANUAL_INDEX_TO_CD_CODE:\n"
        + "\n".join(missing_from_mapping)
    )

if extra_in_mapping:
    raise ValueError(
        "Some MANUAL_INDEX_TO_CD_CODE entries are not present in repair template:\n"
        + "\n".join(extra_in_mapping)
    )


# -----------------------------
# Prepare CD reference lookup
# -----------------------------

cd_ref = cd_ref.copy()
cd_ref["manual_cd_code"] = cd_ref["manual_cd_code"].astype(str).str.strip()

cd_ref_lookup = cd_ref.set_index("manual_cd_code")

unknown_cd_codes = sorted(set(MANUAL_INDEX_TO_CD_CODE.values()) - set(cd_ref_lookup.index))

if unknown_cd_codes:
    raise ValueError(
        "Manual mapping contains CD codes not found in CD reference:\n"
        + "\n".join(unknown_cd_codes)
    )


# -----------------------------
# Fill manual repair columns
# -----------------------------

filled = repair.copy()

for idx, row in filled.iterrows():
    record_index = str(row["index"]).strip()
    cd_code = MANUAL_INDEX_TO_CD_CODE[record_index]

    cd_info = cd_ref_lookup.loc[cd_code]

    filled.at[idx, "manual_cd_code"] = cd_code
    filled.at[idx, "manual_census_division_name"] = cd_info[
        "manual_census_division_name"
    ]
    filled.at[idx, "manual_census_division_dguid"] = cd_info[
        "manual_census_division_dguid"
    ]

    city = str(row.get("city", "")).strip()
    csdname = str(row.get("CSDname", "")).strip()

    if csdname and csdname.lower() != "nan":
        filled.at[idx, "manual_repair_method"] = "manual_csdname_to_cd_exact"
        filled.at[idx, "manual_repair_note"] = (
            f"Assigned to census division {cd_code} "
            f"({cd_info['manual_census_division_name']}) using CSDname/city evidence: "
            f"CSDname='{csdname}', city='{city}'."
        )
    else:
        filled.at[idx, "manual_repair_method"] = "manual_city_to_cd_exact"
        filled.at[idx, "manual_repair_note"] = (
            f"Assigned to census division {cd_code} "
            f"({cd_info['manual_census_division_name']}) using city evidence: "
            f"city='{city}'."
        )


# -----------------------------
# Final validation
# -----------------------------

manual_cols = [
    "manual_cd_code",
    "manual_census_division_name",
    "manual_census_division_dguid",
    "manual_repair_method",
    "manual_repair_note",
]

for col in manual_cols:
    empty_mask = (
        filled[col].isna()
        | (filled[col].astype(str).str.strip() == "")
        | (filled[col].astype(str).str.lower() == "nan")
    )

    if empty_mask.any():
        raise ValueError(
            f"Some rows still have empty values in {col}:\n"
            + filled.loc[empty_mask, ["index", "facility_name", "city", "CSDname", col]]
            .to_string(index=False)
        )

# Validate that the filled CD code/name/DGUID combinations match the reference.
validation = filled.merge(
    cd_ref[
        [
            "manual_cd_code",
            "manual_census_division_dguid",
            "manual_census_division_name",
        ]
    ],
    on=[
        "manual_cd_code",
        "manual_census_division_dguid",
        "manual_census_division_name",
    ],
    how="left",
    indicator=True,
)

if (validation["_merge"] != "both").any():
    bad = validation[validation["_merge"] != "both"]
    raise ValueError(
        "Some filled manual repairs do not match the CD reference:\n"
        + bad.to_string(index=False)
    )


# -----------------------------
# Save output
# -----------------------------

filled.to_csv(OUTPUT_FILLED_REPAIR, index=False)

print("\nFilled manual repair file saved:")
print(OUTPUT_FILLED_REPAIR)

print("\nRows repaired:", len(filled))

print("\nAssigned records:")
print(
    filled[
        [
            "index",
            "facility_name",
            "city",
            "CSDname",
            "manual_cd_code",
            "manual_census_division_name",
            "manual_repair_method",
        ]
    ].to_string(index=False)
)

print("\nCounts by repaired census division:")
print(
    filled["manual_census_division_name"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nDone.")