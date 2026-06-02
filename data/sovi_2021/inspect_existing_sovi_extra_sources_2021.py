from pathlib import Path
import pandas as pd


# ============================================================
# Inspect Existing Extra SoVI Sources 2021
# ============================================================
#
# Purpose:
#   Run a targeted inspection for SoVI variables that may already exist
#   outside the main Census Profile CD blocks.
#
# Focus variables:
#
#   1. physicians_per_100k
#      - doctors_per_100khabs/
#      - CIHI health-region proxy, possibly not yet finalized at CD level.
#
#   2. per_capita_income
#      - census_division_income_2021/
#      - inspect whether income_measure_default is truly the intended
#        SoVI-compatible PERCAP89 proxy.
#
#   3. nursing_home_residents_per_capita
#      - residential_care_per_capita/
#      - inspect ODHF residential-care per-100k proxy and all nursing-like
#        candidate columns.
#
# Optional reference:
#   4. hospitals_per_capita
#      - hospitals_per_capita/
#      - included only as a parallel ODHF facility-density sanity check.
#
# This script does NOT modify existing source sections.
# It writes inspection/audit outputs into:
#
#   sovi_2021/output/
#
# Run from data/:
#   python sovi_2021/inspect_existing_sovi_extra_sources_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "sovi_2021"
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

OUTPUT_FILE_INVENTORY = OUTPUT_DIR / "existing_sovi_extra_sources_file_inventory_2021.csv"
OUTPUT_VARIABLE_AUDIT = OUTPUT_DIR / "existing_sovi_extra_sources_variable_audit_2021.csv"
OUTPUT_COLUMN_INVENTORY = OUTPUT_DIR / "existing_sovi_extra_sources_column_inventory_2021.csv"
OUTPUT_JOIN_AUDIT = OUTPUT_DIR / "existing_sovi_extra_sources_join_audit_2021.csv"
OUTPUT_DOCTORS_PROXY_DRAFT = OUTPUT_DIR / "draft_census_division_doctors_per_100k_proxy_from_existing_sources_2024.csv"
OUTPUT_DRAFT_EXTRA_TABLE = OUTPUT_DIR / "draft_existing_sovi_extra_sources_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "existing_sovi_extra_sources_inspection_summary_2021.csv"


# -----------------------------
# Config
# -----------------------------

CSV_ENCODING_CANDIDATES = [
    "utf-8",
    "utf-8-sig",
    "cp1252",
    "latin1",
]

KEY_CANDIDATES = [
    "census_division_dguid",
    "statcan_dguid",
    "DGUID",
    "dguid",
    "profile_dguid",
    "manual_census_division_dguid",
    "cd_dguid",
]

BASE_IDENTITY_COLUMNS = [
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


TARGETS = [
    {
        "canonical_variable": "per_capita_income",
        "original_sovi_code": "PERCAP89",
        "source_folder": "census_division_income_2021",
        "primary_files": [
            "output/clean_census_division_income_2021.csv",
            "output/clean_census_division_income_2021.parquet",
        ],
        "metadata_files": [
            "output/clean_census_division_income_variable_metadata_2021.csv",
            "output/clean_census_division_income_summary_2021.csv",
        ],
        "column_candidates": [
            "per_capita_income",
            "income_measure_default",
            "median_total_income",
            "median_after_tax_income",
            "average_total_income",
            "average_after_tax_income",
        ],
        "preferred_output_column": "per_capita_income",
        "method_note": (
            "Inspect whether income_measure_default is a defensible proxy for original "
            "SoVI PERCAP89. This may not be literal per-capita income."
        ),
    },
    {
        "canonical_variable": "nursing_home_residents_per_capita",
        "original_sovi_code": "NRRESPC91",
        "source_folder": "residential_care_per_capita",
        "primary_files": [
            "output/clean_census_division_residential_care_per_100k_population_odhf_2021.csv",
            "output/clean_census_division_residential_care_per_100k_population_odhf_2021.parquet",
            "output/clean_census_division_residential_care_counts_odhf_2021.csv",
            "output/clean_census_division_residential_care_counts_odhf_2021.parquet",
        ],
        "metadata_files": [
            "output/odhf_facility_type_counts.csv",
            "output/odhf_quebec_residential_care_counts_by_csd.csv",
        ],
        "column_candidates": [
            "residential_care_facilities_per_100k_population_odhf",
            "residential_care_facilities_per_100k_population",
            "residential_care_per_100k_population_odhf",
            "residential_care_per_100k_population",
            "residential_care_count",
            "residential_care_facility_count",
            "residential_care_facilities_count",
        ],
        "preferred_output_column": "nursing_home_residents_per_capita",
        "method_note": (
            "This is likely a facility-density proxy from ODHF residential-care facilities, "
            "not literal nursing-home residents per capita."
        ),
    },
    {
        "canonical_variable": "hospitals_per_capita",
        "original_sovi_code": "HOSPTPC91",
        "source_folder": "hospitals_per_capita",
        "primary_files": [
            "output/clean_census_division_hospitals_per_100k_population_odhf_2021.csv",
            "output/clean_census_division_hospitals_per_100k_population_odhf_2021.parquet",
            "output/clean_census_division_hospital_counts_odhf_2021.csv",
            "output/clean_census_division_hospital_counts_odhf_2021.parquet",
        ],
        "metadata_files": [
            "output/odhf_facility_type_counts.csv",
            "output/odhf_quebec_hospital_counts_by_derived_cd.csv",
        ],
        "column_candidates": [
            "hospitals_per_100k_population_odhf",
            "hospitals_per_100k_population",
            "hospital_facilities_per_100k_population_odhf",
            "hospital_facilities_per_100k_population",
            "hospital_count",
            "hospitals_count",
        ],
        "preferred_output_column": "hospitals_per_capita",
        "method_note": (
            "ODHF hospital facility density proxy. Included here as a sanity check "
            "because it parallels the residential-care source."
        ),
    },
]


DOCTOR_CONFIG = {
    "canonical_variable": "physicians_per_100k",
    "original_sovi_code": "PHYSICN90",
    "source_folder": "doctors_per_100khabs",
    "final_cd_proxy_files": [
        "output/clean_census_division_doctors_per_100k_proxy_2024.csv",
        "output/clean_census_division_doctors_per_100k_proxy_2024.parquet",
    ],
    "health_region_rate_files": [
        "output/clean_health_region_doctors_per_100k_2024.csv",
        "output/clean_health_region_doctors_per_100k_2024.parquet",
    ],
    "crosswalk_files": [
        "lookup/quebec_census_division_to_health_region_crosswalk_filled.csv",
    ],
    "unresolved_crosswalk_files": [
        "lookup/quebec_census_division_to_health_region_crosswalk_unresolved.csv",
    ],
    "column_candidates": [
        "physicians_per_100k",
        "physicians_per_100k_health_region_proxy",
        "physicians_per_100k_health_region",
        "family_medicine_physicians_per_100k",
        "indicator_value",
        "value",
    ],
    "health_region_key_candidates": [
        "health_region_name",
        "health_region",
        "region_name",
        "Geography",
        "geography",
    ],
    "method_note": (
        "CIHI physician rate is health-region-native. CD proxy requires joining "
        "health-region rates to a census-division-to-health-region crosswalk."
    ),
}


# -----------------------------
# Helpers
# -----------------------------

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

    raise ValueError(f"Unsupported file type: {path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def clean_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def safe_relative(path: Path | None) -> str:
    if path is None:
        return ""

    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        return str(path)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_file(source_folder: str, rel_candidates: list[str]) -> Path | None:
    folder = DATA_DIR / source_folder

    for rel in rel_candidates:
        path = folder / rel
        if path.exists():
            return path

    return None


def find_key_column(columns: list[str]) -> str | None:
    for col in KEY_CANDIDATES:
        if col in columns:
            return col
    return None


def find_first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def numeric_summary(series: pd.Series) -> dict:
    numeric = clean_numeric(series)

    return {
        "non_missing": int(numeric.notna().sum()),
        "missing": int(numeric.isna().sum()),
        "min": numeric.min(skipna=True),
        "max": numeric.max(skipna=True),
        "mean": numeric.mean(skipna=True),
        "median": numeric.median(skipna=True),
    }


def inspect_table_columns(path: Path, source_label: str) -> list[dict]:
    rows = []

    try:
        df = read_table(path)
        df = normalize_columns(df)
    except Exception as exc:
        return [
            {
                "source_label": source_label,
                "file": safe_relative(path),
                "readable": False,
                "column": "",
                "dtype": "",
                "non_missing": "",
                "numeric_non_missing": "",
                "min": "",
                "max": "",
                "mean": "",
                "error": str(exc),
            }
        ]

    for col in df.columns:
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")

        rows.append(
            {
                "source_label": source_label,
                "file": safe_relative(path),
                "readable": True,
                "column": col,
                "dtype": str(series.dtype),
                "non_missing": int(series.notna().sum()),
                "numeric_non_missing": int(numeric.notna().sum()),
                "min": numeric.min(skipna=True),
                "max": numeric.max(skipna=True),
                "mean": numeric.mean(skipna=True),
                "error": "",
            }
        )

    return rows


def find_base_cd_frame() -> Path:
    for path in BASE_CD_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find base census-division frame. Expected one of:\n"
        + "\n".join(str(path) for path in BASE_CD_CANDIDATES)
    )


def standardize_base(base: pd.DataFrame) -> pd.DataFrame:
    base = normalize_columns(base)

    if "geometry" in base.columns:
        base = pd.DataFrame(base.drop(columns=["geometry"]))
    else:
        base = pd.DataFrame(base)

    required = [
        "census_division_code",
        "census_division_dguid",
        "census_division_name",
    ]

    missing = [col for col in required if col not in base.columns]
    if missing:
        raise ValueError(
            "Base frame missing required columns:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(base.columns)
        )

    base["census_division_dguid"] = clean_key(base["census_division_dguid"])

    if base["census_division_dguid"].duplicated().any():
        dupes = base[base["census_division_dguid"].duplicated(keep=False)]
        raise ValueError(
            "Duplicated census_division_dguid values in base frame:\n"
            + dupes[required].head(30).to_string(index=False)
        )

    return base


def join_candidate_to_base(
    base: pd.DataFrame,
    source: pd.DataFrame,
    source_key_col: str,
    source_value_col: str,
    output_col: str,
) -> tuple[pd.Series, dict]:
    source_small = source[[source_key_col, source_value_col]].copy()
    source_small[source_key_col] = clean_key(source_small[source_key_col])
    source_small[source_value_col] = clean_numeric(source_small[source_value_col])

    duplicate_source_key_count = int(source_small[source_key_col].duplicated().sum())

    source_small = source_small.drop_duplicates(subset=[source_key_col], keep="first")
    source_small = source_small.rename(
        columns={
            source_key_col: "census_division_dguid",
            source_value_col: output_col,
        }
    )

    joined = base[["census_division_dguid"]].merge(
        source_small,
        on="census_division_dguid",
        how="left",
        validate="one_to_one",
    )

    values = clean_numeric(joined[output_col])
    summary = numeric_summary(values)

    audit = {
        "duplicate_source_key_count": duplicate_source_key_count,
        "matched_base_rows": int(values.notna().sum()),
        "unmatched_base_rows": int(values.isna().sum()),
        **summary,
    }

    return values, audit


# -----------------------------
# Load base
# -----------------------------

base_path = find_base_cd_frame()
base = standardize_base(read_table(base_path))

identity_cols = [col for col in BASE_IDENTITY_COLUMNS if col in base.columns]

draft = base[identity_cols].copy()
draft.insert(0, "zone_id", draft["census_division_dguid"])

for col in [
    "physicians_per_100k",
    "per_capita_income",
    "nursing_home_residents_per_capita",
    "hospitals_per_capita",
]:
    draft[col] = pd.NA

print("\nLoaded base census-division frame")
print("Path:", safe_relative(base_path))
print("Rows:", len(base))


# -----------------------------
# File inventory
# -----------------------------

file_inventory_rows = []

for target in TARGETS:
    folder = DATA_DIR / target["source_folder"]
    all_rel = target["primary_files"] + target["metadata_files"]

    for rel in all_rel:
        path = folder / rel

        file_inventory_rows.append(
            {
                "canonical_variable": target["canonical_variable"],
                "source_folder": target["source_folder"],
                "relative_candidate_path": f"{target['source_folder']}/{rel}",
                "exists": path.exists(),
                "size_kb": round(path.stat().st_size / 1024, 2) if path.exists() else "",
                "role": "primary_or_metadata",
            }
        )

doctor_folder = DATA_DIR / DOCTOR_CONFIG["source_folder"]
for rel in (
    DOCTOR_CONFIG["final_cd_proxy_files"]
    + DOCTOR_CONFIG["health_region_rate_files"]
    + DOCTOR_CONFIG["crosswalk_files"]
    + DOCTOR_CONFIG["unresolved_crosswalk_files"]
):
    path = doctor_folder / rel
    file_inventory_rows.append(
        {
            "canonical_variable": DOCTOR_CONFIG["canonical_variable"],
            "source_folder": DOCTOR_CONFIG["source_folder"],
            "relative_candidate_path": f"{DOCTOR_CONFIG['source_folder']}/{rel}",
            "exists": path.exists(),
            "size_kb": round(path.stat().st_size / 1024, 2) if path.exists() else "",
            "role": "doctor_proxy_component",
        }
    )

file_inventory = pd.DataFrame(file_inventory_rows)
file_inventory.to_csv(OUTPUT_FILE_INVENTORY, index=False, encoding="utf-8")


# -----------------------------
# Column inventory
# -----------------------------

column_inventory_rows = []

for target in TARGETS:
    for rel in target["primary_files"] + target["metadata_files"]:
        path = DATA_DIR / target["source_folder"] / rel
        if path.exists():
            column_inventory_rows.extend(
                inspect_table_columns(
                    path=path,
                    source_label=target["canonical_variable"],
                )
            )

for rel in (
    DOCTOR_CONFIG["final_cd_proxy_files"]
    + DOCTOR_CONFIG["health_region_rate_files"]
    + DOCTOR_CONFIG["crosswalk_files"]
    + DOCTOR_CONFIG["unresolved_crosswalk_files"]
):
    path = DATA_DIR / DOCTOR_CONFIG["source_folder"] / rel
    if path.exists():
        column_inventory_rows.extend(
            inspect_table_columns(
                path=path,
                source_label=DOCTOR_CONFIG["canonical_variable"],
            )
        )

column_inventory = pd.DataFrame(column_inventory_rows)
column_inventory.to_csv(OUTPUT_COLUMN_INVENTORY, index=False, encoding="utf-8")


# -----------------------------
# Inspect ordinary CD-native targets
# -----------------------------

variable_audit_rows = []
join_audit_rows = []

for target in TARGETS:
    source_path = resolve_file(target["source_folder"], target["primary_files"])

    row_base = {
        "canonical_variable": target["canonical_variable"],
        "original_sovi_code": target["original_sovi_code"],
        "source_folder": target["source_folder"],
        "source_file": safe_relative(source_path),
        "method_note": target["method_note"],
    }

    if source_path is None:
        variable_audit_rows.append(
            {
                **row_base,
                "source_file_found": False,
                "source_file_readable": False,
                "source_rows": 0,
                "source_key_column": "",
                "selected_source_column": "",
                "ready_for_sovi_input": False,
                "proxy_status": "source_file_missing",
                "matched_base_rows": 0,
                "unmatched_base_rows": len(base),
                "non_missing": 0,
                "missing": len(base),
                "min": "",
                "max": "",
                "mean": "",
                "median": "",
            }
        )
        continue

    try:
        source = normalize_columns(read_table(source_path))
        readable = True
        read_error = ""
    except Exception as exc:
        readable = False
        read_error = str(exc)
        source = pd.DataFrame()

    if not readable:
        variable_audit_rows.append(
            {
                **row_base,
                "source_file_found": True,
                "source_file_readable": False,
                "source_rows": 0,
                "source_key_column": "",
                "selected_source_column": "",
                "ready_for_sovi_input": False,
                "proxy_status": f"source_file_unreadable: {read_error}",
                "matched_base_rows": 0,
                "unmatched_base_rows": len(base),
                "non_missing": 0,
                "missing": len(base),
                "min": "",
                "max": "",
                "mean": "",
                "median": "",
            }
        )
        continue

    key_col = find_key_column(list(source.columns))
    value_col = find_first_existing_column(list(source.columns), target["column_candidates"])

    if key_col is None or value_col is None:
        variable_audit_rows.append(
            {
                **row_base,
                "source_file_found": True,
                "source_file_readable": True,
                "source_rows": len(source),
                "source_key_column": key_col or "",
                "selected_source_column": value_col or "",
                "ready_for_sovi_input": False,
                "proxy_status": "key_or_value_column_missing",
                "matched_base_rows": 0,
                "unmatched_base_rows": len(base),
                "non_missing": 0,
                "missing": len(base),
                "min": "",
                "max": "",
                "mean": "",
                "median": "",
            }
        )
        continue

    values, join_audit = join_candidate_to_base(
        base=base,
        source=source,
        source_key_col=key_col,
        source_value_col=value_col,
        output_col=target["preferred_output_column"],
    )

    ready = join_audit["matched_base_rows"] == len(base)

    draft[target["canonical_variable"]] = values

    variable_audit_rows.append(
        {
            **row_base,
            "source_file_found": True,
            "source_file_readable": True,
            "source_rows": len(source),
            "source_key_column": key_col,
            "selected_source_column": value_col,
            "ready_for_sovi_input": ready,
            "proxy_status": (
                "ready_full_coverage"
                if ready
                else "partial_or_missing_after_join"
            ),
            "matched_base_rows": join_audit["matched_base_rows"],
            "unmatched_base_rows": join_audit["unmatched_base_rows"],
            "non_missing": join_audit["non_missing"],
            "missing": join_audit["missing"],
            "min": join_audit["min"],
            "max": join_audit["max"],
            "mean": join_audit["mean"],
            "median": join_audit["median"],
        }
    )

    join_audit_rows.append(
        {
            "canonical_variable": target["canonical_variable"],
            "source_file": safe_relative(source_path),
            "source_key_column": key_col,
            "selected_source_column": value_col,
            **join_audit,
        }
    )


# -----------------------------
# Inspect doctors / physicians
# -----------------------------

doctor_source_folder = DOCTOR_CONFIG["source_folder"]

final_doctor_path = resolve_file(
    doctor_source_folder,
    DOCTOR_CONFIG["final_cd_proxy_files"],
)

doctor_row_base = {
    "canonical_variable": DOCTOR_CONFIG["canonical_variable"],
    "original_sovi_code": DOCTOR_CONFIG["original_sovi_code"],
    "source_folder": doctor_source_folder,
    "method_note": DOCTOR_CONFIG["method_note"],
}

doctor_proxy_values = pd.Series([pd.NA] * len(base), index=base.index, dtype="object")
doctor_proxy_status = "not_started"

if final_doctor_path is not None:
    try:
        doctor_final = normalize_columns(read_table(final_doctor_path))
        key_col = find_key_column(list(doctor_final.columns))
        value_col = find_first_existing_column(
            list(doctor_final.columns),
            DOCTOR_CONFIG["column_candidates"],
        )

        if key_col is not None and value_col is not None:
            doctor_proxy_values, join_audit = join_candidate_to_base(
                base=base,
                source=doctor_final,
                source_key_col=key_col,
                source_value_col=value_col,
                output_col="physicians_per_100k",
            )
            doctor_proxy_status = "final_cd_proxy_file_found"
            ready = join_audit["matched_base_rows"] == len(base)

            draft["physicians_per_100k"] = doctor_proxy_values

            variable_audit_rows.append(
                {
                    **doctor_row_base,
                    "source_file": safe_relative(final_doctor_path),
                    "source_file_found": True,
                    "source_file_readable": True,
                    "source_rows": len(doctor_final),
                    "source_key_column": key_col,
                    "selected_source_column": value_col,
                    "ready_for_sovi_input": ready,
                    "proxy_status": (
                        "ready_full_coverage_from_final_cd_proxy"
                        if ready
                        else "final_cd_proxy_partial_or_missing"
                    ),
                    "matched_base_rows": join_audit["matched_base_rows"],
                    "unmatched_base_rows": join_audit["unmatched_base_rows"],
                    "non_missing": join_audit["non_missing"],
                    "missing": join_audit["missing"],
                    "min": join_audit["min"],
                    "max": join_audit["max"],
                    "mean": join_audit["mean"],
                    "median": join_audit["median"],
                }
            )
            join_audit_rows.append(
                {
                    "canonical_variable": "physicians_per_100k",
                    "source_file": safe_relative(final_doctor_path),
                    "source_key_column": key_col,
                    "selected_source_column": value_col,
                    **join_audit,
                }
            )
        else:
            doctor_proxy_status = "final_cd_proxy_file_found_but_key_or_value_missing"

    except Exception as exc:
        doctor_proxy_status = f"final_cd_proxy_file_unreadable: {exc}"

else:
    doctor_proxy_status = "final_cd_proxy_file_missing_try_reconstruct_from_components"


if final_doctor_path is None:
    health_region_path = resolve_file(
        doctor_source_folder,
        DOCTOR_CONFIG["health_region_rate_files"],
    )
    crosswalk_path = resolve_file(
        doctor_source_folder,
        DOCTOR_CONFIG["crosswalk_files"],
    )
    unresolved_path = resolve_file(
        doctor_source_folder,
        DOCTOR_CONFIG["unresolved_crosswalk_files"],
    )

    if health_region_path is None or crosswalk_path is None:
        variable_audit_rows.append(
            {
                **doctor_row_base,
                "source_file": "",
                "source_file_found": False,
                "source_file_readable": False,
                "source_rows": 0,
                "source_key_column": "",
                "selected_source_column": "",
                "ready_for_sovi_input": False,
                "proxy_status": "doctor_components_missing",
                "matched_base_rows": 0,
                "unmatched_base_rows": len(base),
                "non_missing": 0,
                "missing": len(base),
                "min": "",
                "max": "",
                "mean": "",
                "median": "",
            }
        )
    else:
        hr = normalize_columns(read_table(health_region_path))
        cw = normalize_columns(read_table(crosswalk_path))

        hr_key_col = find_first_existing_column(
            list(hr.columns),
            DOCTOR_CONFIG["health_region_key_candidates"],
        )
        hr_value_col = find_first_existing_column(
            list(hr.columns),
            DOCTOR_CONFIG["column_candidates"],
        )
        cw_cd_key_col = find_key_column(list(cw.columns))
        cw_hr_key_col = find_first_existing_column(
            list(cw.columns),
            DOCTOR_CONFIG["health_region_key_candidates"],
        )

        if hr_key_col is None or hr_value_col is None or cw_cd_key_col is None or cw_hr_key_col is None:
            variable_audit_rows.append(
                {
                    **doctor_row_base,
                    "source_file": f"{safe_relative(health_region_path)} + {safe_relative(crosswalk_path)}",
                    "source_file_found": True,
                    "source_file_readable": True,
                    "source_rows": f"hr={len(hr)}, crosswalk={len(cw)}",
                    "source_key_column": f"hr_key={hr_key_col}, cw_cd_key={cw_cd_key_col}, cw_hr_key={cw_hr_key_col}",
                    "selected_source_column": hr_value_col or "",
                    "ready_for_sovi_input": False,
                    "proxy_status": "doctor_component_key_or_value_missing",
                    "matched_base_rows": 0,
                    "unmatched_base_rows": len(base),
                    "non_missing": 0,
                    "missing": len(base),
                    "min": "",
                    "max": "",
                    "mean": "",
                    "median": "",
                }
            )
        else:
            hr = hr.copy()
            cw = cw.copy()

            hr[hr_key_col] = clean_key(hr[hr_key_col])
            cw[cw_hr_key_col] = clean_key(cw[cw_hr_key_col])
            cw[cw_cd_key_col] = clean_key(cw[cw_cd_key_col])
            hr[hr_value_col] = clean_numeric(hr[hr_value_col])

            reconstructed = cw.merge(
                hr[[hr_key_col, hr_value_col]].drop_duplicates(subset=[hr_key_col]),
                left_on=cw_hr_key_col,
                right_on=hr_key_col,
                how="left",
                validate="many_to_one",
            )

            reconstructed = reconstructed.rename(
                columns={
                    cw_cd_key_col: "census_division_dguid",
                    hr_value_col: "physicians_per_100k",
                    cw_hr_key_col: "health_region_name",
                }
            )

            # Join to base for audit.
            doctor_source_small = reconstructed[
                [
                    "census_division_dguid",
                    "health_region_name",
                    "physicians_per_100k",
                ]
                + [
                    col
                    for col in ["crosswalk_method", "crosswalk_note"]
                    if col in reconstructed.columns
                ]
            ].copy()

            doctor_source_small["census_division_dguid"] = clean_key(
                doctor_source_small["census_division_dguid"]
            )
            doctor_source_small["physicians_per_100k"] = clean_numeric(
                doctor_source_small["physicians_per_100k"]
            )

            doctor_join = base[identity_cols].merge(
                doctor_source_small,
                on="census_division_dguid",
                how="left",
                validate="one_to_one",
            )

            if unresolved_path is not None:
                try:
                    unresolved = normalize_columns(read_table(unresolved_path))
                    unresolved_count = len(unresolved)
                except Exception:
                    unresolved_count = ""
            else:
                unresolved_count = ""

            doctor_join["physicians_per_100k__source"] = "CIHI health-region rate assigned by CD-to-health-region crosswalk"
            doctor_join["physicians_per_100k__source_file"] = safe_relative(health_region_path)
            doctor_join["physicians_per_100k__crosswalk_file"] = safe_relative(crosswalk_path)
            doctor_join["physicians_per_100k__unresolved_crosswalk_file"] = safe_relative(unresolved_path)
            doctor_join["physicians_per_100k__method_note"] = DOCTOR_CONFIG["method_note"]

            doctor_join.to_csv(OUTPUT_DOCTORS_PROXY_DRAFT, index=False, encoding="utf-8")

            values = clean_numeric(doctor_join["physicians_per_100k"])
            summary = numeric_summary(values)

            matched = int(values.notna().sum())
            unmatched = int(values.isna().sum())

            # Partial is expected if Nord-du-Québec remains unresolved.
            ready = matched == len(base)

            draft["physicians_per_100k"] = values

            variable_audit_rows.append(
                {
                    **doctor_row_base,
                    "source_file": f"{safe_relative(health_region_path)} + {safe_relative(crosswalk_path)}",
                    "source_file_found": True,
                    "source_file_readable": True,
                    "source_rows": f"hr={len(hr)}, crosswalk={len(cw)}, unresolved={unresolved_count}",
                    "source_key_column": f"hr_key={hr_key_col}, cw_cd_key={cw_cd_key_col}, cw_hr_key={cw_hr_key_col}",
                    "selected_source_column": hr_value_col,
                    "ready_for_sovi_input": ready,
                    "proxy_status": (
                        "ready_full_coverage_reconstructed_from_components"
                        if ready
                        else "partial_reconstructed_proxy_crosswalk_unresolved"
                    ),
                    "matched_base_rows": matched,
                    "unmatched_base_rows": unmatched,
                    "non_missing": summary["non_missing"],
                    "missing": summary["missing"],
                    "min": summary["min"],
                    "max": summary["max"],
                    "mean": summary["mean"],
                    "median": summary["median"],
                }
            )

            join_audit_rows.append(
                {
                    "canonical_variable": "physicians_per_100k",
                    "source_file": f"{safe_relative(health_region_path)} + {safe_relative(crosswalk_path)}",
                    "source_key_column": f"hr_key={hr_key_col}, cw_cd_key={cw_cd_key_col}, cw_hr_key={cw_hr_key_col}",
                    "selected_source_column": hr_value_col,
                    "duplicate_source_key_count": int(doctor_source_small["census_division_dguid"].duplicated().sum()),
                    "matched_base_rows": matched,
                    "unmatched_base_rows": unmatched,
                    **summary,
                }
            )


# -----------------------------
# Save outputs
# -----------------------------

variable_audit = pd.DataFrame(variable_audit_rows)
join_audit = pd.DataFrame(join_audit_rows)

variable_audit.to_csv(OUTPUT_VARIABLE_AUDIT, index=False, encoding="utf-8")
join_audit.to_csv(OUTPUT_JOIN_AUDIT, index=False, encoding="utf-8")
draft.to_csv(OUTPUT_DRAFT_EXTRA_TABLE, index=False, encoding="utf-8")

ready_count = int(variable_audit["ready_for_sovi_input"].sum())
partial_count = int(
    variable_audit["proxy_status"]
    .astype(str)
    .str.contains("partial", case=False, na=False)
    .sum()
)

summary_rows = [
    {
        "metric": "base_cd_frame",
        "value": safe_relative(base_path),
    },
    {
        "metric": "base_rows",
        "value": len(base),
    },
    {
        "metric": "target_variables_inspected",
        "value": ", ".join(
            ["physicians_per_100k"]
            + [target["canonical_variable"] for target in TARGETS]
        ),
    },
    {
        "metric": "variables_ready_full_coverage",
        "value": ready_count,
    },
    {
        "metric": "variables_partial_or_unresolved",
        "value": partial_count,
    },
    {
        "metric": "ready_variables",
        "value": ", ".join(
            variable_audit.loc[
                variable_audit["ready_for_sovi_input"],
                "canonical_variable",
            ].astype(str)
        ),
    },
    {
        "metric": "not_fully_ready_variables",
        "value": ", ".join(
            variable_audit.loc[
                ~variable_audit["ready_for_sovi_input"],
                "canonical_variable",
            ].astype(str)
        ),
    },
]

for _, row in variable_audit.iterrows():
    canonical = row["canonical_variable"]

    summary_rows.extend(
        [
            {
                "metric": f"{canonical}_proxy_status",
                "value": row["proxy_status"],
            },
            {
                "metric": f"{canonical}_selected_source_column",
                "value": row["selected_source_column"],
            },
            {
                "metric": f"{canonical}_matched_base_rows",
                "value": row["matched_base_rows"],
            },
            {
                "metric": f"{canonical}_unmatched_base_rows",
                "value": row["unmatched_base_rows"],
            },
            {
                "metric": f"{canonical}_mean",
                "value": row["mean"],
            },
        ]
    )

summary_rows.append(
    {
        "metric": "recommended_next_step",
        "value": (
            "Review existing_sovi_extra_sources_variable_audit_2021.csv. "
            "If physicians_per_100k is partial because Nord-du-Québec remains unresolved, "
            "decide whether to leave it missing, assign a documented northern health-region proxy, "
            "or build a spatial/population-weighted allocation rule. Then update the SoVI input-source "
            "inspection mapping."
        ),
    }
)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("EXISTING EXTRA SoVI SOURCE INSPECTION 2021")
print("=" * 72)

print("\nBase rows:", len(base))
print("Variables ready full coverage:", ready_count)
print("Variables partial/unresolved:", partial_count)

print("\nVariable audit:")
display_cols = [
    "canonical_variable",
    "original_sovi_code",
    "source_file",
    "selected_source_column",
    "ready_for_sovi_input",
    "proxy_status",
    "matched_base_rows",
    "unmatched_base_rows",
    "min",
    "max",
    "mean",
    "method_note",
]
display_cols = [col for col in display_cols if col in variable_audit.columns]
print(variable_audit[display_cols].to_string(index=False))

print("\nDraft extra table preview:")
preview_cols = [
    "zone_id",
    "census_division_code",
    "census_division_name",
    "physicians_per_100k",
    "per_capita_income",
    "nursing_home_residents_per_capita",
    "hospitals_per_capita",
]
preview_cols = [col for col in preview_cols if col in draft.columns]
print(draft[preview_cols].head(12).to_string(index=False))

print("\nSaved:")
print(OUTPUT_FILE_INVENTORY)
print(OUTPUT_COLUMN_INVENTORY)
print(OUTPUT_VARIABLE_AUDIT)
print(OUTPUT_JOIN_AUDIT)
if OUTPUT_DOCTORS_PROXY_DRAFT.exists():
    print(OUTPUT_DOCTORS_PROXY_DRAFT)
print(OUTPUT_DRAFT_EXTRA_TABLE)
print(OUTPUT_SUMMARY)

print("\nRecommended next step:")
print(summary.loc[summary["metric"] == "recommended_next_step", "value"].iloc[0])

print("\nDone.")