from pathlib import Path
import pandas as pd


# ============================================================
# Inspect Census Division SoVI Input Sources 2021
# ============================================================
#
# Purpose:
#   Build an audit inventory for the 2021 Québec census-division SoVI-like
#   input table.
#
# This script:
#   1. Defines the expected 42 SoVI-like canonical columns.
#   2. Loads source mappings from YAML files in sovi_2021/mappings/.
#   3. Checks which variables are available from completed data blocks.
#   4. Joins available variables to the Québec census-division base frame.
#   5. Produces a draft wide SoVI input table with all expected columns.
#   6. Marks missing, partial, and unresolved variables explicitly.
#
# Important:
#   This is an inspection/draft-building script, not the final cleaner.
#
# Run from data/:
#   python sovi_2021/inspect_census_division_sovi_input_sources_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "sovi_2021"
MAPPINGS_DIR = SECTION_DIR / "mappings"
OUTPUT_DIR = SECTION_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_CD_CANDIDATES = [
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.csv",
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.parquet",
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.geojson",
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.gpkg",
]

OUTPUT_EXPECTED_COLUMNS = OUTPUT_DIR / "census_division_sovi_expected_columns_2021.csv"
OUTPUT_SOURCE_AUDIT = OUTPUT_DIR / "census_division_sovi_input_source_audit_2021.csv"
OUTPUT_SOURCE_INVENTORY = OUTPUT_DIR / "census_division_sovi_source_file_inventory_2021.csv"
OUTPUT_MISSING_VARIABLES = OUTPUT_DIR / "census_division_sovi_missing_or_unresolved_variables_2021.csv"
OUTPUT_DRAFT_TABLE = OUTPUT_DIR / "draft_clean_census_division_sovi_input_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "census_division_sovi_input_source_inspection_summary_2021.csv"
OUTPUT_MAPPING_FILE_INVENTORY = OUTPUT_DIR / "census_division_sovi_mapping_file_inventory_2021.csv"


# -----------------------------
# Encoding config
# -----------------------------

CSV_ENCODING_CANDIDATES = [
    "utf-8",
    "utf-8-sig",
    "cp1252",
    "latin1",
]


# -----------------------------
# Expected SoVI variables
# -----------------------------

EXPECTED_VARIABLES = [
    ("med_age", "MED_AGE90", "Median age", "years"),
    ("per_capita_income", "PERCAP89", "Per capita income", "currency"),
    ("median_home_value", "MVALOO90", "Median dollar value of owner-occupied housing", "currency"),
    ("median_rent", "MEDRENT90", "Median rent for renter-occupied housing units", "currency"),
    ("physicians_per_100k", "PHYSICN90", "Physicians per 100,000 population", "rate_per_100k"),
    ("pct_vote_leading_party", "PCTVOTE92", "Percent vote cast for president for leading party", "proportion_or_percent"),
    ("birth_rate", "BRATE90", "Birth rate", "rate"),
    ("net_international_migration", "MIGRA_97", "Net international migration", "count_or_rate"),
    ("pct_land_farms", "PCTFARMS92", "Land in farms as percent of total land", "proportion_or_percent"),
    ("pct_black", "PCTBLACK90", "Percent African American", "proportion_or_percent"),
    ("pct_indigenous", "PCTINDIAN90", "Percent Native American / Indigenous", "proportion_or_percent"),
    ("pct_asian", "PCTASIAN90", "Percent Asian", "proportion_or_percent"),
    ("pct_hispanic", "PCTHISPANIC90", "Percent Hispanic", "proportion_or_percent"),
    ("pct_under_5", "PCTKIDS90", "Percent population under five", "proportion_or_percent"),
    ("pct_over_65", "PCTOLD90", "Percent population over 65", "proportion_or_percent"),
    ("pct_unemployed", "PCTVLUN91", "Percent civilian labour force unemployed", "proportion_or_percent"),
    ("avg_people_per_household", "AVGPERHH", "Average people per household", "persons"),
    ("pct_high_income_households", "PCTHH7589", "Percent high-income households", "proportion_or_percent"),
    ("pct_poverty", "PCTPOV90", "Percent living in poverty / low-income proxy", "proportion_or_percent"),
    ("pct_renter", "PCTRENTER90", "Percent renter-occupied housing units", "proportion_or_percent"),
    ("pct_rural_farm", "PCTRFRM90", "Percent rural farm population", "proportion_or_percent"),
    ("debt_revenue_ratio", "DEBREV92", "Local government debt-to-revenue ratio", "ratio"),
    ("pct_mobile_homes", "PCTMOBL90", "Percent housing units that are mobile homes", "proportion_or_percent"),
    ("pct_no_high_school", "PCTNOHS90", "Percent age 25+ with no high school diploma", "proportion_or_percent"),
    ("housing_unit_density", "HODENUT90", "Housing units per square mile", "density"),
    ("housing_permit_density", "HUPTDEN90", "Housing permits / residential construction density", "density"),
    ("manufacturing_density", "MAESDEN92", "Manufacturing establishments per square mile", "density"),
    ("earnings_density", "EARNDEN90", "Earnings in all industries per square mile", "density"),
    ("commercial_density", "COMDEVDN92", "Commercial establishments per square mile", "density"),
    ("property_value_density", "RPROPDEN92", "Value of property and farm products sold per square mile", "density"),
    ("labor_force_participation", "CVBRPC91", "Labour-force participation", "proportion_or_percent"),
    ("female_labor_force_participation", "FEMLBR90", "Female labour-force participation", "proportion_or_percent"),
    ("pct_extractive_employment", "AGRIPC90", "Employment in extractive industries", "proportion_or_percent"),
    ("pct_transport_utility_employment", "TRANPC90", "Employment in transportation / communications / utilities", "proportion_or_percent"),
    ("pct_service_employment", "SERVPC90", "Employment in service occupations", "proportion_or_percent"),
    ("nursing_home_residents_per_capita", "NRRESPC91", "Nursing-home residents per capita", "per_capita"),
    ("hospitals_per_capita", "HOSPTPC91", "Community hospitals per capita", "per_capita"),
    ("pct_population_change", "PCCHGPOP90", "Percent population change", "proportion_or_percent"),
    ("pct_urban", "PCTURB90", "Percent urban population", "proportion_or_percent"),
    ("pct_female", "PCTFEM90", "Percent female", "proportion_or_percent"),
    ("pct_female_headed_households", "PCTF_HH90", "Percent female-headed households, no spouse present", "proportion_or_percent"),
    ("social_security_recipients_per_capita", "SSBENPC90", "Social Security recipients per capita", "per_capita"),
]


# -----------------------------
# Helpers
# -----------------------------

def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for the YAML-driven SoVI inspection engine.\n"
            "Install it in the active environment with:\n\n"
            "    pip install pyyaml\n"
        ) from exc

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Mapping file must contain a YAML dictionary: {path}")

    return data


def load_source_mappings() -> tuple[dict, pd.DataFrame]:
    if not MAPPINGS_DIR.exists():
        raise FileNotFoundError(
            f"Missing mappings directory:\n{MAPPINGS_DIR}\n\n"
            "Create it and add YAML mapping files such as:\n"
            "  source_mappings_current_usable_2021.yaml\n"
            "  source_mappings_ethnocultural_identity_2021.yaml"
        )

    yaml_files = sorted(
        list(MAPPINGS_DIR.glob("*.yaml"))
        + list(MAPPINGS_DIR.glob("*.yml"))
    )

    if not yaml_files:
        raise FileNotFoundError(
            f"No YAML mapping files found in:\n{MAPPINGS_DIR}"
        )

    combined = {}
    inventory_rows = []

    for path in yaml_files:
        mapping_block = load_yaml(path)

        if not mapping_block:
            inventory_rows.append(
                {
                    "mapping_file": safe_relative(path),
                    "variables_in_file": 0,
                    "variables": "",
                    "status": "empty_file",
                }
            )
            continue

        duplicate_keys = sorted(set(combined).intersection(mapping_block))

        if duplicate_keys:
            raise ValueError(
                "Duplicate SoVI mapping keys found while loading YAML files.\n"
                f"File: {path}\n"
                f"Duplicate variables: {', '.join(duplicate_keys)}\n\n"
                "Each canonical variable should be mapped in only one YAML file."
            )

        for canonical_variable, mapping in mapping_block.items():
            if not isinstance(mapping, dict):
                raise ValueError(
                    f"Mapping for {canonical_variable} in {path} must be a dictionary."
                )

            mapping = mapping.copy()
            mapping["_mapping_file"] = safe_relative(path)
            combined[canonical_variable] = mapping

        inventory_rows.append(
            {
                "mapping_file": safe_relative(path),
                "variables_in_file": len(mapping_block),
                "variables": ", ".join(mapping_block.keys()),
                "status": "loaded",
            }
        )

    inventory = pd.DataFrame(inventory_rows)

    return combined, inventory


def read_csv_with_encodings(path: Path, **kwargs) -> pd.DataFrame:
    last_error = None

    for encoding in CSV_ENCODING_CANDIDATES:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return read_csv_with_encodings(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix in [".geojson", ".gpkg", ".shp"]:
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError(f"geopandas is required to read {path}") from exc
        return gpd.read_file(path)

    raise ValueError(f"Unsupported file format: {path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def find_base_cd_frame() -> Path:
    for path in BASE_CD_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find census-division base frame. Expected one of:\n"
        + "\n".join(str(path) for path in BASE_CD_CANDIDATES)
    )


def resolve_source_file(mapping: dict) -> Path | None:
    source_folder = DATA_DIR / mapping["source_folder"]

    for rel in mapping.get("file_candidates", []):
        path = source_folder / rel
        if path.exists():
            return path

    for pattern in mapping.get("glob_patterns", []):
        matches = sorted(source_folder.glob(pattern))
        if matches:
            return matches[0]

    return None


def find_key_column(columns: list[str], mapping: dict | None = None) -> str | None:
    candidates = []

    if mapping is not None:
        candidates.extend(mapping.get("key_column_candidates", []))

    candidates.extend(
        [
            "census_division_dguid",
            "statcan_dguid",
            "DGUID",
            "dguid",
            "manual_census_division_dguid",
            "profile_dguid",
            "cd_dguid",
        ]
    )

    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            ordered_candidates.append(candidate)
            seen.add(candidate)

    for col in ordered_candidates:
        if col in columns:
            return col

    return None


def clean_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def find_first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def numeric_summary(series: pd.Series) -> dict:
    numeric = clean_numeric(series)

    return {
        "numeric_non_missing": int(numeric.notna().sum()),
        "numeric_missing": int(numeric.isna().sum()),
        "min": numeric.min(skipna=True),
        "max": numeric.max(skipna=True),
        "mean": numeric.mean(skipna=True),
        "median": numeric.median(skipna=True),
    }


def safe_relative(path: Path | None) -> str:
    if path is None:
        return ""

    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        return str(path)


def classify_ready_status(
    mapping: dict,
    matched_rows: int,
    base_rows: int,
) -> tuple[bool, str]:
    status_class = mapping.get("status_class", "")

    if status_class == "unresolved_not_numeric":
        return False, "unresolved_not_numeric"

    if status_class == "unresolved_not_feasible":
        return False, "unresolved_not_feasible"

    if status_class == "not_mapped_yet":
        return False, "not_ready_unmapped"

    if matched_rows == base_rows:
        return True, "ready_full_coverage"

    if mapping.get("allow_partial_coverage", False):
        # Explicit YAML threshold wins.
        # Otherwise, any documented partial variable with at least one numeric value is allowed.
        minimum = int(mapping.get("minimum_non_missing_rows", 90))

        if matched_rows >= minimum:
            return True, "ready_partial_with_documented_missing"

    return False, "not_ready_missing_or_partial"


# -----------------------------
# Load YAML source mappings
# -----------------------------

SOURCE_MAPPINGS, mapping_file_inventory = load_source_mappings()
mapping_file_inventory.to_csv(OUTPUT_MAPPING_FILE_INVENTORY, index=False, encoding="utf-8")


# -----------------------------
# Load base frame
# -----------------------------

base_path = find_base_cd_frame()
base = read_table(base_path)
base = normalize_columns(base)

if "geometry" in base.columns:
    base_non_spatial = pd.DataFrame(base.drop(columns=["geometry"]))
else:
    base_non_spatial = pd.DataFrame(base)

required_base_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
]

missing_base_cols = [col for col in required_base_cols if col not in base_non_spatial.columns]
if missing_base_cols:
    raise ValueError(
        "Base frame is missing required columns:\n"
        + "\n".join(missing_base_cols)
        + "\n\nAvailable columns:\n"
        + "\n".join(base_non_spatial.columns)
    )

base_non_spatial["census_division_dguid"] = clean_key(base_non_spatial["census_division_dguid"])

if base_non_spatial["census_division_dguid"].duplicated().any():
    dupes = base_non_spatial[base_non_spatial["census_division_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated census_division_dguid values in base frame:\n"
        + dupes[required_base_cols].head(30).to_string(index=False)
    )

print("\nLoaded Québec census-division base frame")
print("Path:", safe_relative(base_path))
print("Rows:", len(base_non_spatial))
print("Columns:", list(base_non_spatial.columns))

print("\nLoaded SoVI mapping files")
print(mapping_file_inventory.to_string(index=False))


# -----------------------------
# Prepare expected-column table
# -----------------------------

expected_rows = []

for canonical, original, description, expected_unit in EXPECTED_VARIABLES:
    mapping = SOURCE_MAPPINGS.get(canonical)

    expected_rows.append(
        {
            "canonical_variable": canonical,
            "original_variable": original,
            "description": description,
            "expected_unit_from_recipe_or_methodology": expected_unit,
            "has_mapping_config": mapping is not None,
            "mapping_file": mapping.get("_mapping_file", "") if mapping else "",
            "mapping_status_class": mapping.get("status_class", "not_mapped_yet") if mapping else "not_mapped_yet",
            "proxy_quality": mapping.get("proxy_quality", "") if mapping else "",
            "allow_partial_coverage": mapping.get("allow_partial_coverage", False) if mapping else False,
            "minimum_non_missing_rows": mapping.get("minimum_non_missing_rows", "") if mapping else "",
            "mapping_note": mapping.get("note", "") if mapping else "",
            "expected_missing_note": mapping.get("expected_missing_note", "") if mapping else "",
        }
    )

expected_df = pd.DataFrame(expected_rows)
expected_df.to_csv(OUTPUT_EXPECTED_COLUMNS, index=False, encoding="utf-8")


# -----------------------------
# Build draft SoVI table
# -----------------------------

identity_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "census_division_type",
    "province_code",
    "province_name",
    "geography_level",
    "census_year",
    "population_total_2021",
    "land_area_km2",
    "has_positive_population",
]

identity_cols = [col for col in identity_cols if col in base_non_spatial.columns]

draft = base_non_spatial[identity_cols].copy()
draft.insert(0, "zone_id", draft["census_division_dguid"])

for canonical, _, _, _ in EXPECTED_VARIABLES:
    draft[canonical] = pd.NA


# -----------------------------
# Inspect mapped sources
# -----------------------------

audit_rows = []
inventory_rows = []

for canonical, original, description, expected_unit in EXPECTED_VARIABLES:
    mapping = SOURCE_MAPPINGS.get(canonical)

    if mapping is None:
        audit_rows.append(
            {
                "canonical_variable": canonical,
                "original_variable": original,
                "description": description,
                "expected_unit": expected_unit,
                "mapping_file": "",
                "source_folder": "",
                "resolved_file": "",
                "source_file_found": False,
                "source_file_readable": False,
                "source_rows": 0,
                "source_key_column": "",
                "selected_source_column": "",
                "selected_column_found": False,
                "candidate_columns_found": "",
                "candidate_columns_missing": "",
                "base_rows": len(draft),
                "matched_base_rows": 0,
                "unmatched_base_rows": len(draft),
                "duplicate_source_key_count": 0,
                "numeric_non_missing_after_join": 0,
                "numeric_missing_after_join": len(draft),
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "status_class": "not_mapped_yet",
                "coverage_status": "not_ready_unmapped",
                "ready_for_draft_table": False,
                "inserted_into_draft_table": False,
                "proxy_quality": "",
                "allow_partial_coverage": False,
                "expected_missing_note": "",
                "note": "No completed source mapping yet.",
            }
        )
        continue

    if "source_folder" not in mapping:
        audit_rows.append(
            {
                "canonical_variable": canonical,
                "original_variable": original,
                "description": description,
                "expected_unit": expected_unit,
                "mapping_file": mapping.get("_mapping_file", ""),
                "source_folder": "",
                "resolved_file": "",
                "source_file_found": False,
                "source_file_readable": False,
                "source_rows": 0,
                "source_key_column": "",
                "selected_source_column": "",
                "selected_column_found": False,
                "candidate_columns_found": "",
                "candidate_columns_missing": ", ".join(mapping.get("column_candidates", [])),
                "base_rows": len(draft),
                "matched_base_rows": 0,
                "unmatched_base_rows": len(draft),
                "duplicate_source_key_count": 0,
                "numeric_non_missing_after_join": 0,
                "numeric_missing_after_join": len(draft),
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "status_class": mapping.get("status_class", "mapping_missing_source_folder"),
                "coverage_status": "not_ready_mapping_missing_source_folder",
                "ready_for_draft_table": False,
                "inserted_into_draft_table": False,
                "proxy_quality": mapping.get("proxy_quality", ""),
                "allow_partial_coverage": mapping.get("allow_partial_coverage", False),
                "expected_missing_note": mapping.get("expected_missing_note", ""),
                "note": mapping.get("note", ""),
            }
        )
        continue

    source_path = resolve_source_file(mapping)

    inventory_rows.append(
        {
            "canonical_variable": canonical,
            "mapping_file": mapping.get("_mapping_file", ""),
            "source_folder": mapping["source_folder"],
            "resolved_file": safe_relative(source_path),
            "source_file_found": source_path is not None,
            "status_class": mapping.get("status_class", ""),
            "proxy_quality": mapping.get("proxy_quality", ""),
            "allow_partial_coverage": mapping.get("allow_partial_coverage", False),
            "minimum_non_missing_rows": mapping.get("minimum_non_missing_rows", ""),
            "expected_missing_note": mapping.get("expected_missing_note", ""),
            "note": mapping.get("note", ""),
        }
    )

    if source_path is None:
        audit_rows.append(
            {
                "canonical_variable": canonical,
                "original_variable": original,
                "description": description,
                "expected_unit": expected_unit,
                "mapping_file": mapping.get("_mapping_file", ""),
                "source_folder": mapping["source_folder"],
                "resolved_file": "",
                "source_file_found": False,
                "source_file_readable": False,
                "source_rows": 0,
                "source_key_column": "",
                "selected_source_column": "",
                "selected_column_found": False,
                "candidate_columns_found": "",
                "candidate_columns_missing": ", ".join(mapping.get("column_candidates", [])),
                "base_rows": len(draft),
                "matched_base_rows": 0,
                "unmatched_base_rows": len(draft),
                "duplicate_source_key_count": 0,
                "numeric_non_missing_after_join": 0,
                "numeric_missing_after_join": len(draft),
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "status_class": "source_file_missing",
                "coverage_status": "not_ready_source_file_missing",
                "ready_for_draft_table": False,
                "inserted_into_draft_table": False,
                "proxy_quality": mapping.get("proxy_quality", ""),
                "allow_partial_coverage": mapping.get("allow_partial_coverage", False),
                "expected_missing_note": mapping.get("expected_missing_note", ""),
                "note": mapping.get("note", ""),
            }
        )
        continue

    try:
        source = read_table(source_path)
        source = normalize_columns(source)
        source_readable = True
        read_error = ""
    except Exception as exc:
        source = pd.DataFrame()
        source_readable = False
        read_error = str(exc)

    if not source_readable:
        audit_rows.append(
            {
                "canonical_variable": canonical,
                "original_variable": original,
                "description": description,
                "expected_unit": expected_unit,
                "mapping_file": mapping.get("_mapping_file", ""),
                "source_folder": mapping["source_folder"],
                "resolved_file": safe_relative(source_path),
                "source_file_found": True,
                "source_file_readable": False,
                "source_rows": 0,
                "source_key_column": "",
                "selected_source_column": "",
                "selected_column_found": False,
                "candidate_columns_found": "",
                "candidate_columns_missing": ", ".join(mapping.get("column_candidates", [])),
                "base_rows": len(draft),
                "matched_base_rows": 0,
                "unmatched_base_rows": len(draft),
                "duplicate_source_key_count": 0,
                "numeric_non_missing_after_join": 0,
                "numeric_missing_after_join": len(draft),
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "status_class": "source_file_unreadable",
                "coverage_status": "not_ready_source_file_unreadable",
                "ready_for_draft_table": False,
                "inserted_into_draft_table": False,
                "proxy_quality": mapping.get("proxy_quality", ""),
                "allow_partial_coverage": mapping.get("allow_partial_coverage", False),
                "expected_missing_note": mapping.get("expected_missing_note", ""),
                "note": f"{mapping.get('note', '')} Read error: {read_error}",
            }
        )
        continue

    columns = list(source.columns)
    key_col = find_key_column(columns, mapping=mapping)
    selected_col = find_first_existing_column(columns, mapping.get("column_candidates", []))

    candidate_found = [
        col for col in mapping.get("column_candidates", [])
        if col in columns
    ]
    candidate_missing = [
        col for col in mapping.get("column_candidates", [])
        if col not in columns
    ]

    if key_col is None or selected_col is None:
        audit_rows.append(
            {
                "canonical_variable": canonical,
                "original_variable": original,
                "description": description,
                "expected_unit": expected_unit,
                "mapping_file": mapping.get("_mapping_file", ""),
                "source_folder": mapping["source_folder"],
                "resolved_file": safe_relative(source_path),
                "source_file_found": True,
                "source_file_readable": True,
                "source_rows": len(source),
                "source_key_column": key_col or "",
                "selected_source_column": selected_col or "",
                "selected_column_found": selected_col is not None,
                "candidate_columns_found": ", ".join(candidate_found),
                "candidate_columns_missing": ", ".join(candidate_missing),
                "base_rows": len(draft),
                "matched_base_rows": 0,
                "unmatched_base_rows": len(draft),
                "duplicate_source_key_count": int(source[key_col].duplicated().sum()) if key_col is not None else 0,
                "numeric_non_missing_after_join": 0,
                "numeric_missing_after_join": len(draft),
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "status_class": "key_or_column_missing",
                "coverage_status": "not_ready_key_or_column_missing",
                "ready_for_draft_table": False,
                "inserted_into_draft_table": False,
                "proxy_quality": mapping.get("proxy_quality", ""),
                "allow_partial_coverage": mapping.get("allow_partial_coverage", False),
                "expected_missing_note": mapping.get("expected_missing_note", ""),
                "note": mapping.get("note", ""),
            }
        )
        continue

    source = source.copy()
    source[key_col] = clean_key(source[key_col])
    source[selected_col] = clean_numeric(source[selected_col])

    duplicate_source_key_count = int(source[key_col].duplicated().sum())

    source_small = (
        source[[key_col, selected_col]]
        .drop_duplicates(subset=[key_col], keep="first")
        .rename(columns={key_col: "census_division_dguid", selected_col: canonical})
    )

    joined = draft[["census_division_dguid"]].merge(
        source_small,
        on="census_division_dguid",
        how="left",
        validate="one_to_one",
    )

    numeric = clean_numeric(joined[canonical])
    summary = numeric_summary(numeric)

    matched_rows = int(numeric.notna().sum())
    unmatched_rows = int(numeric.isna().sum())

    ready, coverage_status = classify_ready_status(
        mapping=mapping,
        matched_rows=matched_rows,
        base_rows=len(draft),
    )

    inserted = False
    if ready:
        draft[canonical] = numeric
        inserted = True

    audit_rows.append(
        {
            "canonical_variable": canonical,
            "original_variable": original,
            "description": description,
            "expected_unit": expected_unit,
            "mapping_file": mapping.get("_mapping_file", ""),
            "source_folder": mapping["source_folder"],
            "resolved_file": safe_relative(source_path),
            "source_file_found": True,
            "source_file_readable": True,
            "source_rows": len(source),
            "source_key_column": key_col,
            "selected_source_column": selected_col,
            "selected_column_found": True,
            "candidate_columns_found": ", ".join(candidate_found),
            "candidate_columns_missing": ", ".join(candidate_missing),
            "base_rows": len(draft),
            "matched_base_rows": matched_rows,
            "unmatched_base_rows": unmatched_rows,
            "duplicate_source_key_count": duplicate_source_key_count,
            "numeric_non_missing_after_join": summary["numeric_non_missing"],
            "numeric_missing_after_join": summary["numeric_missing"],
            "min": summary["min"],
            "max": summary["max"],
            "mean": summary["mean"],
            "median": summary["median"],
            "status_class": mapping.get("status_class", ""),
            "coverage_status": coverage_status,
            "ready_for_draft_table": ready,
            "inserted_into_draft_table": inserted,
            "proxy_quality": mapping.get("proxy_quality", ""),
            "allow_partial_coverage": mapping.get("allow_partial_coverage", False),
            "expected_missing_note": mapping.get("expected_missing_note", ""),
            "note": mapping.get("note", ""),
        }
    )


# -----------------------------
# Save outputs
# -----------------------------

source_audit = pd.DataFrame(audit_rows)
source_inventory = pd.DataFrame(inventory_rows)

missing = source_audit[~source_audit["ready_for_draft_table"]].copy()

source_audit.to_csv(OUTPUT_SOURCE_AUDIT, index=False, encoding="utf-8")
source_inventory.to_csv(OUTPUT_SOURCE_INVENTORY, index=False, encoding="utf-8")
missing.to_csv(OUTPUT_MISSING_VARIABLES, index=False, encoding="utf-8")
draft.to_csv(OUTPUT_DRAFT_TABLE, index=False, encoding="utf-8")

ready_count = int(source_audit["ready_for_draft_table"].sum())
not_ready_count = int((~source_audit["ready_for_draft_table"]).sum())
mapped_config_count = int(source_audit["mapping_file"].astype(str).ne("").sum())
mapped_source_file_found_count = int(source_audit["source_file_found"].sum())

partial_ready_count = int(
    source_audit["coverage_status"]
    .astype(str)
    .eq("ready_partial_with_documented_missing")
    .sum()
)

full_ready_count = int(
    source_audit["coverage_status"]
    .astype(str)
    .eq("ready_full_coverage")
    .sum()
)

summary = pd.DataFrame(
    [
        {"metric": "base_cd_frame", "value": safe_relative(base_path)},
        {"metric": "base_rows", "value": len(draft)},
        {"metric": "mappings_dir", "value": safe_relative(MAPPINGS_DIR)},
        {"metric": "mapping_files_loaded", "value": len(mapping_file_inventory)},
        {
            "metric": "mapping_files",
            "value": ", ".join(mapping_file_inventory["mapping_file"].astype(str)),
        },
        {"metric": "expected_sovi_variables", "value": len(EXPECTED_VARIABLES)},
        {"metric": "variables_with_mapping_config", "value": mapped_config_count},
        {"metric": "source_files_found_for_mapped_variables", "value": mapped_source_file_found_count},
        {"metric": "variables_ready_for_draft_table", "value": ready_count},
        {"metric": "variables_ready_full_coverage", "value": full_ready_count},
        {"metric": "variables_ready_partial_with_documented_missing", "value": partial_ready_count},
        {"metric": "variables_not_ready_or_unmapped", "value": not_ready_count},
        {
            "metric": "ready_variables",
            "value": ", ".join(
                source_audit.loc[source_audit["ready_for_draft_table"], "canonical_variable"]
            ),
        },
        {
            "metric": "ready_partial_variables",
            "value": ", ".join(
                source_audit.loc[
                    source_audit["coverage_status"].astype(str).eq("ready_partial_with_documented_missing"),
                    "canonical_variable",
                ]
            ),
        },
        {
            "metric": "not_ready_variables",
            "value": ", ".join(
                source_audit.loc[~source_audit["ready_for_draft_table"], "canonical_variable"]
            ),
        },
        {
            "metric": "important_method_note_mapping_architecture",
            "value": (
                "Source mappings are loaded from YAML files in sovi_2021/mappings. "
                "Adding a new completed variable should usually require adding or editing a YAML mapping file, "
                "not regenerating this inspection engine."
            ),
        },
        {
            "metric": "important_method_note_education",
            "value": (
                "The draft table uses pct_no_high_school_diploma_25_64 for pct_no_high_school "
                "because the SoVI/YAML variable PCTNOHS90 is age-25-plus-oriented. "
                "This is still a 25-64 proxy, not an exact 25+ reproduction."
            ),
        },
        {
            "metric": "important_method_note_physicians",
            "value": (
                "The draft table includes physicians_per_100k from the finalized doctors-per-100k "
                "census-division proxy table. It is a CIHI health-region proxy and has one documented "
                "missing value for Nord-du-Québec."
            ),
        },
        {
            "metric": "important_method_note_voting",
            "value": (
                "The draft table includes pct_vote_leading_party from the census-division voter proxy block. "
                "This is an area-weighted 2021 Canadian federal-election proxy based on federal electoral "
                "district leading-party/candidate vote share, not a direct reproduction of the original U.S. "
                "presidential leading-party vote variable."
            ),
        },
        {
            "metric": "important_method_note_ethnocultural_identity",
            "value": (
                "The draft table includes pct_black, pct_indigenous, pct_asian, and pct_hispanic from the "
                "ethnocultural identity block. These are Canadian Census Profile visible-minority / "
                "Indigenous identity proxies; pct_asian is a derived component sum."
            ),
        },
        {
            "metric": "important_method_note_nursing",
            "value": (
                "The draft table includes nursing_home_residents_per_capita using ODHF residential-care "
                "facilities per 100k population. This is a facility-density proxy, not literal residents per capita."
            ),
        },
        {
            "metric": "important_method_note_hospitals",
            "value": (
                "The draft table includes hospitals_per_capita using ODHF hospitals per 100k population. "
                "This is a facility-density proxy."
            ),
        },
        {
            "metric": "recommended_next_step",
            "value": (
                "Review census_division_sovi_input_source_audit_2021.csv and the draft table. "
                "If the YAML mappings loaded correctly, continue adding new variable blocks by creating "
                "additional YAML files in sovi_2021/mappings."
            ),
        },
    ]
)

summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CENSUS DIVISION SoVI INPUT SOURCE INSPECTION 2021")
print("=" * 72)

print("\nExpected SoVI variables:", len(EXPECTED_VARIABLES))
print("Base rows:", len(draft))
print("Mapping files loaded:", len(mapping_file_inventory))
print("Variables with mapping config:", mapped_config_count)
print("Variables ready for draft table:", ready_count)
print("  Full coverage:", full_ready_count)
print("  Partial documented coverage:", partial_ready_count)
print("Variables not ready / unmapped:", not_ready_count)

print("\nReady variables:")
ready_display = source_audit[source_audit["ready_for_draft_table"]][
    [
        "canonical_variable",
        "original_variable",
        "coverage_status",
        "selected_source_column",
        "resolved_file",
        "numeric_non_missing_after_join",
        "numeric_missing_after_join",
        "min",
        "max",
        "mean",
        "proxy_quality",
        "mapping_file",
    ]
]

if ready_display.empty:
    print("[none]")
else:
    print(ready_display.to_string(index=False))

print("\nNot ready / unresolved variables:")
not_ready_display = source_audit[~source_audit["ready_for_draft_table"]][
    [
        "canonical_variable",
        "original_variable",
        "status_class",
        "coverage_status",
        "source_file_found",
        "selected_source_column",
        "matched_base_rows",
        "proxy_quality",
        "note",
    ]
]

print(not_ready_display.to_string(index=False))

print("\nDraft table preview:")
preview_cols = [
    "zone_id",
    "census_division_code",
    "census_division_name",
    "med_age",
    "per_capita_income",
    "physicians_per_100k",
    "pct_vote_leading_party",
    "pct_black",
    "pct_indigenous",
    "pct_asian",
    "pct_hispanic",
    "pct_unemployed",
    "pct_poverty",
    "median_rent",
    "pct_no_high_school",
    "hospitals_per_capita",
    "nursing_home_residents_per_capita",
]
preview_cols = [col for col in preview_cols if col in draft.columns]
print(draft[preview_cols].head(12).to_string(index=False))

print("\nSaved:")
print(OUTPUT_MAPPING_FILE_INVENTORY)
print(OUTPUT_EXPECTED_COLUMNS)
print(OUTPUT_SOURCE_INVENTORY)
print(OUTPUT_SOURCE_AUDIT)
print(OUTPUT_MISSING_VARIABLES)
print(OUTPUT_DRAFT_TABLE)
print(OUTPUT_SUMMARY)

print("\nRecommended next step:")
print(summary.loc[summary["metric"] == "recommended_next_step", "value"].iloc[0])

print("\nDone.")