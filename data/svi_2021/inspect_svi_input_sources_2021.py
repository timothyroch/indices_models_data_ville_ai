from pathlib import Path
import pandas as pd

try:
    import geopandas as gpd
except ImportError:
    gpd = None


# ============================================================
# Inspect SVI Input Sources 2021
# ============================================================
#
# Purpose:
#   Validate the cleaned feature sources that could feed the Québec 2021
#   census-tract SVI-like input table.
#
# This script does NOT compute SVI.
# It checks:
#   1. whether each expected cleaned feature file exists;
#   2. whether the expected SVI source columns exist;
#   3. whether each file has statcan_dguid as a join key;
#   4. whether each file joins cleanly to the census-tract spatial/population frame;
#   5. missingness for each SVI candidate variable;
#   6. which variables are direct, proxies, weak proxies, or missing.
#
# Run from data/:
#   python svi_2021/inspect_svi_input_sources_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = DATA_DIR / "svi_2021" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SOURCE_INVENTORY = OUTPUT_DIR / "svi_input_source_file_inventory_2021.csv"
OUTPUT_VARIABLE_AVAILABILITY = OUTPUT_DIR / "svi_input_variable_availability_2021.csv"
OUTPUT_CANDIDATE_COLUMN_DIAGNOSTICS = OUTPUT_DIR / "svi_input_candidate_column_diagnostics_2021.csv"
OUTPUT_JOIN_DIAGNOSTICS = OUTPUT_DIR / "svi_input_join_diagnostics_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "svi_input_availability_summary_2021.csv"


# -----------------------------
# Base spatial/population frame
# -----------------------------

BASE_FRAME_CANDIDATES = [
    DATA_DIR
    / "spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_tract_spatial_frame_with_population_2021.parquet",
    DATA_DIR
    / "spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_tract_spatial_frame_with_population_2021.geojson",
    DATA_DIR
    / "spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_tract_spatial_frame_with_population_2021.gpkg",
    DATA_DIR
    / "spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_tract_spatial_frame_with_population_2021.csv",
]


# -----------------------------
# Expected SVI variable mapping
# -----------------------------
#
# status_class values:
#   direct_or_strong_proxy
#   local_adaptation_proxy
#   weak_proxy_candidate_only
#   missing_no_current_proxy
#
# include_in_main_candidate:
#   True  = should be included in the first clean SVI input table if column exists
#   False = diagnostic only, not recommended as main SVI variable yet
#
# Notes:
#   - pct_disability and pct_group_quarters are currently placeholders.
#   - pct_no_vehicle has only weak commuting-based candidates, not an exact household
#     vehicle availability measure.
#   - per_capita_income uses an income proxy and must be reversed later by the SVI
#     scoring recipe because lower income means higher vulnerability.
#

SVI_VARIABLES = [
    {
        "canonical_variable": "pct_below_poverty",
        "domain": "socioeconomic_status",
        "source_folder": "low_income_2021",
        "preferred_file": "clean_census_tract_low_income_2021.csv",
        "candidate_columns": [
            "pct_low_income_lim_at",
            "published_pct_low_income_lim_at",
            "pct_low_income_lico_at",
            "published_pct_low_income_lico_at",
        ],
        "status_class": "direct_or_strong_proxy",
        "include_in_main_candidate": True,
        "direction": "higher_more_vulnerable",
        "proxy_note": (
            "Canadian low-income proxy for the SVI poverty variable. "
            "Preferred column is LIM-AT low-income proportion."
        ),
    },
    {
        "canonical_variable": "pct_unemployed",
        "domain": "socioeconomic_status",
        "source_folder": "unemployment_2021",
        "preferred_file": "clean_census_tract_unemployment_2021.csv",
        "candidate_columns": [
            "pct_unemployed",
            "unemployment_measure_default",
            "unemployment_rate",
            "pct_civilian_unemployed",
        ],
        "status_class": "direct_or_strong_proxy",
        "include_in_main_candidate": True,
        "direction": "higher_more_vulnerable",
        "proxy_note": "Closest available unemployment rate from the cleaned unemployment table.",
    },
    {
        "canonical_variable": "per_capita_income",
        "domain": "socioeconomic_status",
        "source_folder": "income_2021",
        "preferred_file": "clean_census_tract_income_2021.csv",
        "candidate_columns": [
            "income_measure_default",
            "median_after_tax_income_15plus_2020",
            "median_total_income_15plus_2020",
            "average_after_tax_income_15plus_2020",
            "average_total_income_15plus_2020",
            "median_household_income_2020",
        ],
        "status_class": "local_adaptation_proxy",
        "include_in_main_candidate": True,
        "direction": "lower_more_vulnerable",
        "proxy_note": (
            "Original SVI uses per-capita income. Current Canadian adaptation "
            "likely uses a cleaned income proxy, with lower income meaning higher vulnerability."
        ),
    },
    {
        "canonical_variable": "pct_no_high_school",
        "domain": "socioeconomic_status",
        "source_folder": "education_2021",
        "preferred_file": "clean_census_tract_education_2021.csv",
        "candidate_columns": [
            "education_measure_default",
            "pct_no_certificate_15plus",
            "pct_no_certificate_25_64",
        ],
        "status_class": "local_adaptation_proxy",
        "include_in_main_candidate": True,
        "direction": "higher_more_vulnerable",
        "proxy_note": (
            "Canadian proxy for no high school diploma. Uses no certificate, diploma, or degree."
        ),
    },
    {
        "canonical_variable": "pct_age_65_plus",
        "domain": "household_composition_disability",
        "source_folder": "age_structure_2021",
        "preferred_file": "clean_census_tract_age_structure_2021.csv",
        "candidate_columns": [
            "pct_age_65_plus",
        ],
        "status_class": "direct_or_strong_proxy",
        "include_in_main_candidate": True,
        "direction": "higher_more_vulnerable",
        "proxy_note": "Direct age 65+ measure.",
    },
    {
        "canonical_variable": "pct_age_17_or_younger",
        "domain": "household_composition_disability",
        "source_folder": "age_structure_2021",
        "preferred_file": "clean_census_tract_age_structure_2021.csv",
        "candidate_columns": [
            "pct_age_0_14",
            "pct_age_0_17",
            "pct_age_0_19",
        ],
        "status_class": "local_adaptation_proxy",
        "include_in_main_candidate": True,
        "direction": "higher_more_vulnerable",
        "proxy_note": (
            "Original SVI uses age 17 or younger. Current available Canadian proxy is age 0-14."
        ),
    },
    {
        "canonical_variable": "pct_disability",
        "domain": "household_composition_disability",
        "source_folder": None,
        "preferred_file": None,
        "candidate_columns": [],
        "status_class": "missing_no_current_proxy",
        "include_in_main_candidate": False,
        "direction": "higher_more_vulnerable",
        "proxy_note": (
            "Missing. No acceptable current census-tract disability/activity-limitation "
            "feature has been cleaned."
        ),
    },
    {
        "canonical_variable": "pct_single_parent_households",
        "domain": "household_composition_disability",
        "source_folder": "household_family_2021",
        "preferred_file": "clean_census_tract_household_family_2021.csv",
        "candidate_columns": [
            "single_parent_measure_default",
            "pct_one_parent_family_households",
            "pct_one_parent_families_among_census_families",
            "pct_persons_in_one_parent_family",
        ],
        "status_class": "local_adaptation_proxy",
        "include_in_main_candidate": True,
        "direction": "higher_more_vulnerable",
        "proxy_note": (
            "Proxy for SVI single-parent households with children under 18. "
            "Current default may not be restricted exactly to children under 18."
        ),
    },
    {
        "canonical_variable": "pct_minority",
        "domain": "minority_status_language",
        "source_folder": "immigration_ethnocultural_2021",
        "preferred_file": "clean_census_tract_immigration_ethnocultural_2021.csv",
        "candidate_columns": [
            "ethnocultural_measure_default",
            "pct_visible_minority",
            "visible_minority_measure_default",
            "pct_racialized_population",
            "pct_immigrant",
            "pct_recent_immigrant",
        ],
        "status_class": "local_adaptation_proxy",
        "include_in_main_candidate": True,
        "direction": "higher_more_vulnerable",
        "proxy_note": (
            "Canadian/Quebec local-adaptation proxy for the U.S. SVI minority-status variable."
        ),
    },
    {
        "canonical_variable": "pct_limited_language",
        "domain": "minority_status_language",
        "source_folder": "language_2021",
        "preferred_file": "clean_census_tract_language_2021.csv",
        "candidate_columns": [
            "language_barrier_measure_default",
            "pct_limited_official_language",
            "pct_no_knowledge_official_languages",
            "pct_no_knowledge_english_french",
            "official_language_barrier_measure_default",
        ],
        "status_class": "local_adaptation_proxy",
        "include_in_main_candidate": True,
        "direction": "higher_more_vulnerable",
        "proxy_note": (
            "Canadian/Quebec local-adaptation proxy for English less than well. "
            "Should be documented carefully in the recipe."
        ),
    },
    {
        "canonical_variable": "pct_multiunit_structures",
        "domain": "housing_transportation",
        "source_folder": "housing_type_2021",
        "preferred_file": "clean_census_tract_housing_type_2021.csv",
        "candidate_columns": [
            "multiunit_measure_default",
            "pct_apartment_multiunit",
            "pct_apartment_building",
            "pct_apartment_5plus_storeys",
            "pct_apartment_less_than_5_storeys",
        ],
        "status_class": "direct_or_strong_proxy",
        "include_in_main_candidate": True,
        "direction": "higher_more_vulnerable",
        "proxy_note": "Strong Canadian housing-type proxy for SVI multi-unit structures.",
    },
    {
        "canonical_variable": "pct_mobile_homes",
        "domain": "housing_transportation",
        "source_folder": "housing_type_2021",
        "preferred_file": "clean_census_tract_housing_type_2021.csv",
        "candidate_columns": [
            "mobile_home_measure_default",
            "pct_movable_dwelling",
            "pct_mobile_home",
        ],
        "status_class": "direct_or_strong_proxy",
        "include_in_main_candidate": True,
        "direction": "higher_more_vulnerable",
        "proxy_note": "Canadian proxy for mobile homes, likely based on movable dwellings.",
    },
    {
        "canonical_variable": "pct_crowding",
        "domain": "housing_transportation",
        "source_folder": "housing_suitability_crowding_2021",
        "preferred_file": "clean_census_tract_housing_suitability_crowding_2021.csv",
        "candidate_columns": [
            "crowding_measure_default",
            "pct_more_than_one_person_per_room",
            "pct_not_suitable_housing",
            "housing_suitability_measure_default",
        ],
        "status_class": "direct_or_strong_proxy",
        "include_in_main_candidate": True,
        "direction": "higher_more_vulnerable",
        "proxy_note": (
            "Strong proxy for SVI crowding. Preferred variable is more than one person per room."
        ),
    },
    {
        "canonical_variable": "pct_no_vehicle",
        "domain": "housing_transportation",
        "source_folder": "commuting_transport_2021",
        "preferred_file": "clean_census_tract_commuting_transport_2021.csv",
        "candidate_columns": [
            "pct_commute_non_car_modes",
            "public_transit_commuting_measure_default",
            "pct_commute_public_transit",
            "pct_commute_active_transport",
        ],
        "status_class": "weak_proxy_candidate_only",
        "include_in_main_candidate": False,
        "direction": "higher_more_vulnerable",
        "proxy_note": (
            "No exact household vehicle availability variable currently cleaned. "
            "Commuting-mode variables are weak proxy candidates only and should not be "
            "treated as exact SVI no-vehicle availability."
        ),
    },
    {
        "canonical_variable": "pct_group_quarters",
        "domain": "housing_transportation",
        "source_folder": None,
        "preferred_file": None,
        "candidate_columns": [],
        "status_class": "missing_no_current_proxy",
        "include_in_main_candidate": False,
        "direction": "higher_more_vulnerable",
        "proxy_note": (
            "Missing. Need census-tract population in collective dwellings/group quarters. "
            "ODHF facility counts are not an acceptable proxy."
        ),
    },
]


# -----------------------------
# Helpers
# -----------------------------

def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, low_memory=False)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix in [".geojson", ".gpkg", ".shp"]:
        if gpd is None:
            raise ImportError(
                f"geopandas is required to read spatial file: {path}"
            )
        return gpd.read_file(path)

    raise ValueError(f"Unsupported file type: {path}")


def find_existing_base_frame() -> Path:
    for path in BASE_FRAME_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find the census-tract spatial/population frame.\n"
        "Expected one of:\n"
        + "\n".join(str(p) for p in BASE_FRAME_CANDIDATES)
    )


def find_source_file(source_folder: str, preferred_file: str | None) -> tuple[Path | None, str]:
    """
    Return:
        (path, resolution_method)
    """
    if source_folder is None:
        return None, "no_source_expected"

    output_dir = DATA_DIR / source_folder / "output"

    if preferred_file is not None:
        preferred_path = output_dir / preferred_file
        if preferred_path.exists():
            return preferred_path, "preferred_path"

    if not output_dir.exists():
        return None, "output_folder_missing"

    # Prefer clean census tract files.
    csv_candidates = sorted(output_dir.glob("clean_census_tract*.csv"))
    if len(csv_candidates) == 1:
        return csv_candidates[0], "single_clean_census_tract_csv_fallback"

    if len(csv_candidates) > 1:
        # Try to choose a file whose name contains the folder topic.
        folder_tokens = [
            token for token in source_folder.replace("_2021", "").split("_")
            if token
        ]

        scored = []
        for candidate in csv_candidates:
            name = candidate.name.lower()
            score = sum(1 for token in folder_tokens if token in name)
            scored.append((score, candidate))

        scored = sorted(scored, key=lambda x: (-x[0], str(x[1])))
        if scored[0][0] > 0:
            return scored[0][1], "best_scored_clean_census_tract_csv_fallback"

        return csv_candidates[0], "first_clean_census_tract_csv_fallback"

    # Then try any clean CSV.
    any_clean_csv = sorted(output_dir.glob("clean*.csv"))
    if len(any_clean_csv) == 1:
        return any_clean_csv[0], "single_clean_csv_fallback"

    if len(any_clean_csv) > 1:
        return any_clean_csv[0], "first_clean_csv_fallback"

    return None, "no_clean_csv_found"


def numeric_summary(series: pd.Series) -> dict:
    numeric = pd.to_numeric(series, errors="coerce")

    return {
        "numeric_non_missing": int(numeric.notna().sum()),
        "numeric_missing": int(numeric.isna().sum()),
        "min": numeric.min(skipna=True),
        "max": numeric.max(skipna=True),
        "mean": numeric.mean(skipna=True),
        "median": numeric.median(skipna=True),
    }


def safe_column_list(df: pd.DataFrame, max_chars: int = 1200) -> str:
    text = ", ".join(map(str, df.columns))
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ... [truncated]"


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


# -----------------------------
# Load base frame
# -----------------------------

base_path = find_existing_base_frame()
base = read_table(base_path)
base = normalize_columns(base)

print("\nLoaded base census-tract spatial/population frame")
print("Path:", base_path.relative_to(DATA_DIR))
print("Rows:", len(base))
print("Columns:", list(base.columns))

if "statcan_dguid" not in base.columns:
    raise ValueError(
        "Base frame must contain statcan_dguid.\n"
        f"Columns found: {list(base.columns)}"
    )

base = base.copy()
base["statcan_dguid"] = clean_text(base["statcan_dguid"])

if base["statcan_dguid"].duplicated().any():
    duplicated = base[base["statcan_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated statcan_dguid values in base frame:\n"
        + duplicated[["statcan_dguid"]].head(30).to_string(index=False)
    )

population_col = None
for candidate in ["population_total", "population_total_2021", "population"]:
    if candidate in base.columns:
        population_col = candidate
        break

if population_col is None:
    print("\nWARNING: No population column found in base frame.")
    base["__positive_population__"] = True
else:
    base[population_col] = pd.to_numeric(base[population_col], errors="coerce")
    base["__positive_population__"] = base[population_col] > 0

base_rows = len(base)
positive_population_rows = int(base["__positive_population__"].sum())

print("Base rows:", base_rows)
print("Positive-population rows:", positive_population_rows)
print("Population column:", population_col)


# -----------------------------
# Source file inventory
# -----------------------------

source_inventory_rows = []
source_cache: dict[str, pd.DataFrame] = {}
source_path_cache: dict[str, Path | None] = {}
source_resolution_cache: dict[str, str] = {}

unique_sources = sorted(
    {
        (spec["source_folder"], spec["preferred_file"])
        for spec in SVI_VARIABLES
        if spec["source_folder"] is not None
    }
)

print("\nInspecting source files...")

for source_folder, preferred_file in unique_sources:
    path, resolution_method = find_source_file(source_folder, preferred_file)

    source_path_cache[source_folder] = path
    source_resolution_cache[source_folder] = resolution_method

    if path is None:
        source_inventory_rows.append(
            {
                "source_folder": source_folder,
                "preferred_file": preferred_file,
                "resolved_file": "",
                "resolution_method": resolution_method,
                "file_exists": False,
                "readable": False,
                "row_count": 0,
                "has_statcan_dguid": False,
                "duplicate_statcan_dguid_count": None,
                "columns": "",
                "error": "No source file found",
            }
        )
        print(f"- {source_folder}: NOT FOUND ({resolution_method})")
        continue

    try:
        df = read_table(path)
        df = normalize_columns(df)

        has_key = "statcan_dguid" in df.columns
        duplicate_key_count = None

        if has_key:
            df["statcan_dguid"] = clean_text(df["statcan_dguid"])
            duplicate_key_count = int(df["statcan_dguid"].duplicated().sum())

        source_cache[source_folder] = df

        source_inventory_rows.append(
            {
                "source_folder": source_folder,
                "preferred_file": preferred_file,
                "resolved_file": str(path.relative_to(DATA_DIR)),
                "resolution_method": resolution_method,
                "file_exists": True,
                "readable": True,
                "row_count": len(df),
                "has_statcan_dguid": has_key,
                "duplicate_statcan_dguid_count": duplicate_key_count,
                "columns": safe_column_list(df),
                "error": "",
            }
        )

        print(
            f"- {source_folder}: loaded {path.relative_to(DATA_DIR)} "
            f"({len(df)} rows, key={has_key})"
        )

    except Exception as e:
        source_inventory_rows.append(
            {
                "source_folder": source_folder,
                "preferred_file": preferred_file,
                "resolved_file": str(path.relative_to(DATA_DIR)),
                "resolution_method": resolution_method,
                "file_exists": True,
                "readable": False,
                "row_count": 0,
                "has_statcan_dguid": False,
                "duplicate_statcan_dguid_count": None,
                "columns": "",
                "error": str(e),
            }
        )
        print(f"- {source_folder}: ERROR reading {path.relative_to(DATA_DIR)}: {e}")


source_inventory = pd.DataFrame(source_inventory_rows)
source_inventory.to_csv(OUTPUT_SOURCE_INVENTORY, index=False)


# -----------------------------
# Variable-level inspection
# -----------------------------

availability_rows = []
candidate_column_rows = []
join_diagnostic_rows = []

base_keys = base[["statcan_dguid", "__positive_population__"]].copy()

print("\nInspecting canonical SVI variable availability...")

for spec in SVI_VARIABLES:
    canonical = spec["canonical_variable"]
    domain = spec["domain"]
    source_folder = spec["source_folder"]
    status_class = spec["status_class"]
    include_main = spec["include_in_main_candidate"]
    direction = spec["direction"]
    proxy_note = spec["proxy_note"]
    candidates = spec["candidate_columns"]

    print(f"\nVariable: {canonical}")

    if source_folder is None:
        availability_rows.append(
            {
                "canonical_variable": canonical,
                "domain": domain,
                "status_class": status_class,
                "include_in_main_candidate": include_main,
                "direction": direction,
                "source_folder": "",
                "resolved_file": "",
                "source_file_found": False,
                "source_file_readable": False,
                "source_rows": 0,
                "has_statcan_dguid": False,
                "selected_source_column": "",
                "selected_column_found": False,
                "candidate_columns_found": "",
                "candidate_columns_missing": "",
                "base_rows": base_rows,
                "positive_population_rows": positive_population_rows,
                "matched_base_rows": 0,
                "unmatched_base_rows": base_rows,
                "duplicate_source_key_count": None,
                "missing_all_base_rows": base_rows,
                "missing_positive_population_rows": positive_population_rows,
                "numeric_non_missing": 0,
                "numeric_missing": base_rows,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "ready_for_clean_table": False,
                "proxy_note": proxy_note,
            }
        )
        print("  Missing by design:", proxy_note)
        continue

    df = source_cache.get(source_folder)
    source_path = source_path_cache.get(source_folder)

    source_file_found = source_path is not None and source_path.exists()
    source_file_readable = df is not None
    resolved_file = (
        str(source_path.relative_to(DATA_DIR))
        if source_path is not None and source_path.exists()
        else ""
    )

    if df is None:
        availability_rows.append(
            {
                "canonical_variable": canonical,
                "domain": domain,
                "status_class": status_class,
                "include_in_main_candidate": include_main,
                "direction": direction,
                "source_folder": source_folder,
                "resolved_file": resolved_file,
                "source_file_found": source_file_found,
                "source_file_readable": False,
                "source_rows": 0,
                "has_statcan_dguid": False,
                "selected_source_column": "",
                "selected_column_found": False,
                "candidate_columns_found": "",
                "candidate_columns_missing": ", ".join(candidates),
                "base_rows": base_rows,
                "positive_population_rows": positive_population_rows,
                "matched_base_rows": 0,
                "unmatched_base_rows": base_rows,
                "duplicate_source_key_count": None,
                "missing_all_base_rows": base_rows,
                "missing_positive_population_rows": positive_population_rows,
                "numeric_non_missing": 0,
                "numeric_missing": base_rows,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "ready_for_clean_table": False,
                "proxy_note": proxy_note,
            }
        )
        print("  Source file unavailable.")
        continue

    has_key = "statcan_dguid" in df.columns
    source_rows = len(df)

    if has_key:
        df = df.copy()
        df["statcan_dguid"] = clean_text(df["statcan_dguid"])

    duplicate_source_key_count = (
        int(df["statcan_dguid"].duplicated().sum())
        if has_key
        else None
    )

    found_candidates = [col for col in candidates if col in df.columns]
    missing_candidates = [col for col in candidates if col not in df.columns]
    selected_col = first_existing_column(df, candidates)

    for candidate_col in candidates:
        if candidate_col in df.columns:
            summary = numeric_summary(df[candidate_col])
            candidate_column_rows.append(
                {
                    "canonical_variable": canonical,
                    "domain": domain,
                    "source_folder": source_folder,
                    "resolved_file": resolved_file,
                    "candidate_column": candidate_col,
                    "column_found": True,
                    "status_class": status_class,
                    "include_in_main_candidate": include_main,
                    **summary,
                }
            )
        else:
            candidate_column_rows.append(
                {
                    "canonical_variable": canonical,
                    "domain": domain,
                    "source_folder": source_folder,
                    "resolved_file": resolved_file,
                    "candidate_column": candidate_col,
                    "column_found": False,
                    "status_class": status_class,
                    "include_in_main_candidate": include_main,
                    "numeric_non_missing": 0,
                    "numeric_missing": None,
                    "min": None,
                    "max": None,
                    "mean": None,
                    "median": None,
                }
            )

    if not has_key or selected_col is None:
        availability_rows.append(
            {
                "canonical_variable": canonical,
                "domain": domain,
                "status_class": status_class,
                "include_in_main_candidate": include_main,
                "direction": direction,
                "source_folder": source_folder,
                "resolved_file": resolved_file,
                "source_file_found": source_file_found,
                "source_file_readable": source_file_readable,
                "source_rows": source_rows,
                "has_statcan_dguid": has_key,
                "selected_source_column": selected_col or "",
                "selected_column_found": selected_col is not None,
                "candidate_columns_found": ", ".join(found_candidates),
                "candidate_columns_missing": ", ".join(missing_candidates),
                "base_rows": base_rows,
                "positive_population_rows": positive_population_rows,
                "matched_base_rows": 0,
                "unmatched_base_rows": base_rows,
                "duplicate_source_key_count": duplicate_source_key_count,
                "missing_all_base_rows": base_rows,
                "missing_positive_population_rows": positive_population_rows,
                "numeric_non_missing": 0,
                "numeric_missing": base_rows,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "ready_for_clean_table": False,
                "proxy_note": proxy_note,
            }
        )

        if not has_key:
            print("  Missing statcan_dguid key.")
        if selected_col is None:
            print("  No candidate column found.")
        continue

    # Join validation.
    join_df = base_keys.merge(
        df[["statcan_dguid", selected_col]].copy(),
        on="statcan_dguid",
        how="left",
        validate="one_to_one" if duplicate_source_key_count == 0 else "one_to_many",
        indicator=True,
    )

    matched_base_rows = int((join_df["_merge"] == "both").sum())
    unmatched_base_rows = int((join_df["_merge"] == "left_only").sum())

    selected_numeric = pd.to_numeric(join_df[selected_col], errors="coerce")

    missing_all = int(selected_numeric.isna().sum())
    missing_positive = int(
        selected_numeric[join_df["__positive_population__"]].isna().sum()
    )

    summary = numeric_summary(join_df[selected_col])

    ready_for_clean_table = (
        source_file_found
        and source_file_readable
        and has_key
        and selected_col is not None
        and duplicate_source_key_count == 0
        and unmatched_base_rows == 0
        and include_main
        and status_class
        in [
            "direct_or_strong_proxy",
            "local_adaptation_proxy",
        ]
    )

    availability_rows.append(
        {
            "canonical_variable": canonical,
            "domain": domain,
            "status_class": status_class,
            "include_in_main_candidate": include_main,
            "direction": direction,
            "source_folder": source_folder,
            "resolved_file": resolved_file,
            "source_file_found": source_file_found,
            "source_file_readable": source_file_readable,
            "source_rows": source_rows,
            "has_statcan_dguid": has_key,
            "selected_source_column": selected_col,
            "selected_column_found": True,
            "candidate_columns_found": ", ".join(found_candidates),
            "candidate_columns_missing": ", ".join(missing_candidates),
            "base_rows": base_rows,
            "positive_population_rows": positive_population_rows,
            "matched_base_rows": matched_base_rows,
            "unmatched_base_rows": unmatched_base_rows,
            "duplicate_source_key_count": duplicate_source_key_count,
            "missing_all_base_rows": missing_all,
            "missing_positive_population_rows": missing_positive,
            "ready_for_clean_table": ready_for_clean_table,
            "proxy_note": proxy_note,
            **summary,
        }
    )

    join_diagnostic_rows.append(
        {
            "canonical_variable": canonical,
            "source_folder": source_folder,
            "resolved_file": resolved_file,
            "selected_source_column": selected_col,
            "base_rows": base_rows,
            "source_rows": source_rows,
            "matched_base_rows": matched_base_rows,
            "unmatched_base_rows": unmatched_base_rows,
            "duplicate_source_key_count": duplicate_source_key_count,
            "missing_all_base_rows": missing_all,
            "missing_positive_population_rows": missing_positive,
        }
    )

    print(f"  Selected column: {selected_col}")
    print(f"  Matched base rows: {matched_base_rows}/{base_rows}")
    print(f"  Missing values among all base rows: {missing_all}")
    print(f"  Status: {status_class}")
    print(f"  Ready for clean table: {ready_for_clean_table}")


# -----------------------------
# Save variable diagnostics
# -----------------------------

availability = pd.DataFrame(availability_rows)
candidate_columns = pd.DataFrame(candidate_column_rows)
join_diagnostics = pd.DataFrame(join_diagnostic_rows)

availability.to_csv(OUTPUT_VARIABLE_AVAILABILITY, index=False)
candidate_columns.to_csv(OUTPUT_CANDIDATE_COLUMN_DIAGNOSTICS, index=False)
join_diagnostics.to_csv(OUTPUT_JOIN_DIAGNOSTICS, index=False)


# -----------------------------
# Summary
# -----------------------------

summary_rows = []

summary_rows.append(
    {
        "metric": "base_frame_path",
        "value": str(base_path.relative_to(DATA_DIR)),
    }
)

summary_rows.append(
    {
        "metric": "base_rows",
        "value": base_rows,
    }
)

summary_rows.append(
    {
        "metric": "positive_population_rows",
        "value": positive_population_rows,
    }
)

summary_rows.append(
    {
        "metric": "population_column",
        "value": population_col or "",
    }
)

for status, count in availability["status_class"].value_counts(dropna=False).items():
    summary_rows.append(
        {
            "metric": f"variables_status_{status}",
            "value": int(count),
        }
    )

summary_rows.append(
    {
        "metric": "variables_ready_for_clean_table",
        "value": int(availability["ready_for_clean_table"].sum()),
    }
)

summary_rows.append(
    {
        "metric": "variables_not_ready_for_clean_table",
        "value": int((~availability["ready_for_clean_table"]).sum()),
    }
)

not_ready = availability[~availability["ready_for_clean_table"]]

summary_rows.append(
    {
        "metric": "not_ready_variables",
        "value": ", ".join(not_ready["canonical_variable"].tolist()),
    }
)

missing_no_proxy = availability[
    availability["status_class"] == "missing_no_current_proxy"
]["canonical_variable"].tolist()

summary_rows.append(
    {
        "metric": "missing_no_current_proxy_variables",
        "value": ", ".join(missing_no_proxy),
    }
)

weak_proxy_only = availability[
    availability["status_class"] == "weak_proxy_candidate_only"
]["canonical_variable"].tolist()

summary_rows.append(
    {
        "metric": "weak_proxy_candidate_only_variables",
        "value": ", ".join(weak_proxy_only),
    }
)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False)


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("SVI INPUT SOURCE INSPECTION SUMMARY")
print("=" * 72)

print("\nBase frame:")
print("Path:", base_path.relative_to(DATA_DIR))
print("Rows:", base_rows)
print("Positive-population rows:", positive_population_rows)
print("Population column:", population_col)

print("\nAvailability by status:")
print(availability["status_class"].value_counts(dropna=False).to_string())

print("\nVariables ready for first clean SVI input table:")
ready_cols = [
    "canonical_variable",
    "selected_source_column",
    "source_folder",
    "missing_all_base_rows",
    "status_class",
]
print(
    availability[availability["ready_for_clean_table"]][ready_cols]
    .to_string(index=False)
)

print("\nVariables not ready / not recommended for main table:")
not_ready_cols = [
    "canonical_variable",
    "status_class",
    "selected_source_column",
    "source_folder",
    "missing_all_base_rows",
    "proxy_note",
]
print(
    availability[~availability["ready_for_clean_table"]][not_ready_cols]
    .to_string(index=False)
)

print("\nImportant interpretation:")
print("- direct_or_strong_proxy and local_adaptation_proxy variables can feed the first SVI input table.")
print("- weak_proxy_candidate_only variables should not be treated as exact SVI variables.")
print("- missing_no_current_proxy variables require new data or must remain missing placeholders.")
print("- The SVI scoring implementation currently errors on missing values, so the final scoring")
print("  recipe will need an honest reproduction level and missing-variable strategy.")

print("\nSaved:")
print(OUTPUT_SOURCE_INVENTORY)
print(OUTPUT_VARIABLE_AVAILABILITY)
print(OUTPUT_CANDIDATE_COLUMN_DIAGNOSTICS)
print(OUTPUT_JOIN_DIAGNOSTICS)
print(OUTPUT_SUMMARY)

print("\nDone.")