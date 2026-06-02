from pathlib import Path
import pandas as pd


# ============================================================
# Fill Census Division → Health Region Crosswalk
# ============================================================
#
# Purpose:
#   Fill the Quebec census-division-to-health-region crosswalk used to assign
#   CIHI health-region-level doctors-per-100k rates to census divisions.
#
# Important:
#   This is the manual/auditable Option A crosswalk.
#   It does not perform spatial overlay.
#
# Inputs:
#   doctors_per_100khabs/lookup/quebec_census_division_to_health_region_crosswalk.csv
#   doctors_per_100khabs/output/clean_health_region_doctors_per_100k_2024.csv
#
# Output:
#   doctors_per_100khabs/lookup/quebec_census_division_to_health_region_crosswalk_filled.csv
#
# Run from data/:
#   python doctors_per_100khabs/fill_health_region_crosswalk.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

LOOKUP_DIR = DATA_DIR / "doctors_per_100khabs" / "lookup"
OUTPUT_DIR = DATA_DIR / "doctors_per_100khabs" / "output"

INPUT_TEMPLATE = LOOKUP_DIR / "quebec_census_division_to_health_region_crosswalk.csv"
INPUT_HEALTH_REGION_TABLE = OUTPUT_DIR / "clean_health_region_doctors_per_100k_2024.csv"

OUTPUT_FILLED = LOOKUP_DIR / "quebec_census_division_to_health_region_crosswalk_filled.csv"
OUTPUT_UNRESOLVED = LOOKUP_DIR / "quebec_census_division_to_health_region_crosswalk_unresolved.csv"


# -----------------------------
# Manual health-region groups
# -----------------------------
# These health_region_name values must exactly match the cleaned CIHI table.
#
# Note:
#   The original census divisions are mostly Quebec MRCs / territoires
#   équivalents. We assign each CD to the corresponding Quebec health region.
#
# Special case:
#   CDUID 2499 = Nord-du-Québec.
#   The CD covers territory that CIHI reports through multiple northern health
#   regions: Nord-du-Québec, Nunavik, and Terres-Cries-de-la-Baie-James.
#   At census-division scale, this cannot be represented as a simple one-to-one
#   assignment without an additional population-weighted or spatial split.
#   Therefore it is left unresolved for now.

HEALTH_REGION_GROUPS = {
    "Gaspésie–Îles-de-la-Madeleine Region (Que.)": [
        "2401",  # Communauté maritime des Îles-de-la-Madeleine
        "2402",  # Le Rocher-Percé
        "2403",  # La Côte-de-Gaspé
        "2404",  # La Haute-Gaspésie
        "2405",  # Bonaventure
        "2406",  # Avignon
    ],

    "Bas-Saint-Laurent Region (Que.)": [
        "2407",  # La Matapédia
        "2408",  # La Matanie
        "2409",  # La Mitis
        "2410",  # Rimouski-Neigette
        "2411",  # Les Basques
        "2412",  # Rivière-du-Loup
        "2413",  # Témiscouata
        "2414",  # Kamouraska
    ],

    "Capitale-Nationale Region (Que.)": [
        "2415",  # Charlevoix-Est
        "2416",  # Charlevoix
        "2420",  # L'Île-d'Orléans
        "2421",  # La Côte-de-Beaupré
        "2422",  # La Jacques-Cartier
        "2423",  # Québec
        "2434",  # Portneuf
    ],

    "Chaudière-Appalaches Region (Que.)": [
        "2417",  # L'Islet
        "2418",  # Montmagny
        "2419",  # Bellechasse
        "2425",  # Lévis
        "2426",  # La Nouvelle-Beauce
        "2427",  # Robert-Cliche / Beauce-Centre
        "2428",  # Les Etchemins
        "2429",  # Beauce-Sartigan
        "2431",  # Les Appalaches
        "2433",  # Lotbinière
    ],

    "Mauricie et Centre-du-Québec Region (Que.)": [
        "2432",  # L'Érable
        "2435",  # Mékinac
        "2436",  # Shawinigan
        "2437",  # Francheville
        "2438",  # Bécancour
        "2439",  # Arthabaska
        "2449",  # Drummond
        "2450",  # Nicolet-Yamaska
        "2451",  # Maskinongé
        "2490",  # La Tuque
    ],

    "Estrie Region (Que.)": [
        "2430",  # Le Granit
        "2440",  # Les Sources
        "2441",  # Le Haut-Saint-François
        "2442",  # Le Val-Saint-François
        "2443",  # Sherbrooke
        "2444",  # Coaticook
        "2445",  # Memphrémagog
        "2446",  # Brome-Missisquoi
        "2447",  # La Haute-Yamaska
    ],

    "Montérégie Region (Que.)": [
        "2448",  # Acton
        "2453",  # Pierre-De Saurel
        "2454",  # Les Maskoutains
        "2455",  # Rouville
        "2456",  # Le Haut-Richelieu
        "2457",  # La Vallée-du-Richelieu
        "2458",  # Longueuil
        "2459",  # Marguerite-D'Youville
        "2467",  # Roussillon
        "2468",  # Les Jardins-de-Napierville
        "2469",  # Le Haut-Saint-Laurent
        "2470",  # Beauharnois-Salaberry
        "2471",  # Vaudreuil-Soulanges
    ],

    "Lanaudière Region (Que.)": [
        "2452",  # D'Autray
        "2460",  # L'Assomption
        "2461",  # Joliette
        "2462",  # Matawinie
        "2463",  # Montcalm
        "2464",  # Les Moulins
    ],

    "Laval Region (Que.)": [
        "2465",  # Laval
    ],

    "Montréal Region (Que.)": [
        "2466",  # Montréal
    ],

    "Laurentides Region (Que.)": [
        "2472",  # Deux-Montagnes
        "2473",  # Thérèse-De Blainville
        "2474",  # Mirabel
        "2475",  # La Rivière-du-Nord
        "2476",  # Argenteuil
        "2477",  # Les Pays-d'en-Haut
        "2478",  # Les Laurentides
        "2479",  # Antoine-Labelle
    ],

    "Outaouais Region (Que.)": [
        "2480",  # Papineau
        "2481",  # Gatineau
        "2482",  # Les Collines-de-l'Outaouais
        "2483",  # La Vallée-de-la-Gatineau
        "2484",  # Pontiac
    ],

    "Abitibi-Témiscamingue Region (Que.)": [
        "2485",  # Témiscamingue
        "2486",  # Rouyn-Noranda
        "2487",  # Abitibi-Ouest
        "2488",  # Abitibi
        "2489",  # La Vallée-de-l'Or
    ],

    "Saguenay–Lac-Saint-Jean Region (Que.)": [
        "2491",  # Le Domaine-du-Roy
        "2492",  # Maria-Chapdelaine
        "2493",  # Lac-Saint-Jean-Est
        "2494",  # Le Saguenay-et-son-Fjord
    ],

    "Côte-Nord Region (Que.)": [
        "2495",  # La Haute-Côte-Nord
        "2496",  # Manicouagan
        "2497",  # Sept-Rivières--Caniapiscau
        "2498",  # Minganie--Le Golfe-du-Saint-Laurent
    ],
}

UNRESOLVED_CDS = {
    "2499": {
        "crosswalk_method": "unresolved_northern_split",
        "crosswalk_note": (
            "Nord-du-Québec census division contains territories associated with "
            "multiple CIHI northern health regions: Nord-du-Québec Region (Que.), "
            "Nunavik Region (Que.), and Terres-Cries-de-la-Baie-James Region (Que.). "
            "A one-to-one CD-to-health-region assignment would be misleading without "
            "additional population-weighted or spatial allocation."
        ),
    }
}


# -----------------------------
# Flatten manual mapping
# -----------------------------

manual_mapping = {}

for health_region_name, cd_codes in HEALTH_REGION_GROUPS.items():
    for cd_code in cd_codes:
        if cd_code in manual_mapping:
            raise ValueError(f"Duplicate CDUID in manual mapping: {cd_code}")
        manual_mapping[cd_code] = health_region_name


# -----------------------------
# Load files
# -----------------------------

if not INPUT_TEMPLATE.exists():
    raise FileNotFoundError(f"Crosswalk template not found:\n{INPUT_TEMPLATE}")

if not INPUT_HEALTH_REGION_TABLE.exists():
    raise FileNotFoundError(
        f"Clean health-region doctors table not found:\n{INPUT_HEALTH_REGION_TABLE}"
    )

crosswalk = pd.read_csv(INPUT_TEMPLATE, dtype=str)
health_regions = pd.read_csv(INPUT_HEALTH_REGION_TABLE, dtype=str)

print("\nLoaded crosswalk template")
print("Rows:", len(crosswalk))

print("\nLoaded clean health-region doctors table")
print("Rows:", len(health_regions))


# -----------------------------
# Validate expected columns
# -----------------------------

required_crosswalk_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "census_division_type",
    "census_division_land_area_km2",
    "province_code",
    "health_region_name",
    "crosswalk_method",
    "crosswalk_note",
]

missing_crosswalk_cols = [
    col for col in required_crosswalk_cols if col not in crosswalk.columns
]

if missing_crosswalk_cols:
    raise ValueError(
        "Missing required columns in crosswalk template:\n"
        + "\n".join(missing_crosswalk_cols)
    )

if "health_region_name" not in health_regions.columns:
    raise ValueError(
        "Missing health_region_name column in clean health-region doctors table."
    )


# -----------------------------
# Fill crosswalk
# -----------------------------

crosswalk = crosswalk.copy()

crosswalk["health_region_name"] = ""
crosswalk["crosswalk_method"] = ""
crosswalk["crosswalk_note"] = ""

for idx, row in crosswalk.iterrows():
    cd_code = str(row["census_division_code"]).strip()

    if cd_code in manual_mapping:
        crosswalk.at[idx, "health_region_name"] = manual_mapping[cd_code]
        crosswalk.at[idx, "crosswalk_method"] = "manual_region_membership"
        crosswalk.at[idx, "crosswalk_note"] = (
            "Assigned from Quebec MRC / territoire équivalent regional membership "
            "to the matching CIHI health-region label."
        )

    elif cd_code in UNRESOLVED_CDS:
        crosswalk.at[idx, "health_region_name"] = ""
        crosswalk.at[idx, "crosswalk_method"] = UNRESOLVED_CDS[cd_code][
            "crosswalk_method"
        ]
        crosswalk.at[idx, "crosswalk_note"] = UNRESOLVED_CDS[cd_code][
            "crosswalk_note"
        ]

    else:
        crosswalk.at[idx, "health_region_name"] = ""
        crosswalk.at[idx, "crosswalk_method"] = "missing_manual_assignment"
        crosswalk.at[idx, "crosswalk_note"] = (
            "No manual health-region assignment was provided for this census division."
        )


# -----------------------------
# Validation
# -----------------------------

expected_cd_codes = set(crosswalk["census_division_code"].astype(str).str.strip())
mapped_cd_codes = set(manual_mapping.keys()) | set(UNRESOLVED_CDS.keys())

extra_mapped_codes = sorted(mapped_cd_codes - expected_cd_codes)
missing_from_mapping = sorted(expected_cd_codes - mapped_cd_codes)

if extra_mapped_codes:
    raise ValueError(
        "Manual mapping contains CDUIDs not present in template:\n"
        + "\n".join(extra_mapped_codes)
    )

if missing_from_mapping:
    raise ValueError(
        "Some census divisions are missing from manual mapping:\n"
        + "\n".join(missing_from_mapping)
    )

available_health_regions = set(
    health_regions["health_region_name"].astype(str).str.strip()
)

used_health_regions = set(
    crosswalk.loc[
        crosswalk["health_region_name"].astype(str).str.strip() != "",
        "health_region_name",
    ]
    .astype(str)
    .str.strip()
)

unknown_health_regions = sorted(used_health_regions - available_health_regions)

if unknown_health_regions:
    raise ValueError(
        "Some assigned health_region_name values do not exist in the clean CIHI table:\n"
        + "\n".join(unknown_health_regions)
    )

if crosswalk["census_division_code"].duplicated().any():
    duplicated = crosswalk[
        crosswalk["census_division_code"].duplicated(keep=False)
    ]
    raise ValueError(
        "Duplicated census_division_code values found:\n"
        + duplicated.to_string(index=False)
    )


# -----------------------------
# Diagnostics
# -----------------------------

assigned = crosswalk[
    crosswalk["health_region_name"].astype(str).str.strip() != ""
].copy()

unresolved = crosswalk[
    crosswalk["health_region_name"].astype(str).str.strip() == ""
].copy()

print("\nCrosswalk status")
print("Total census divisions:", len(crosswalk))
print("Assigned census divisions:", len(assigned))
print("Unresolved census divisions:", len(unresolved))

print("\nAssigned census divisions by health region:")
print(
    assigned["health_region_name"]
    .value_counts()
    .sort_index()
    .to_string()
)

if not unresolved.empty:
    print("\nUnresolved census divisions:")
    print(
        unresolved[
            [
                "census_division_code",
                "census_division_name",
                "crosswalk_method",
                "crosswalk_note",
            ]
        ].to_string(index=False)
    )


# -----------------------------
# Save outputs
# -----------------------------

crosswalk.to_csv(OUTPUT_FILLED, index=False)

if not unresolved.empty:
    unresolved.to_csv(OUTPUT_UNRESOLVED, index=False)

print("\nSaved filled crosswalk:")
print(OUTPUT_FILLED)

if not unresolved.empty:
    print("\nSaved unresolved rows:")
    print(OUTPUT_UNRESOLVED)

print("\nDone.")