from pathlib import Path
import re
import pandas as pd


# ============================================================
# Clean Census Division Demographic Estimates 2021
# ============================================================
#
# Purpose:
#   Create clean Québec census-division demographic-estimate variables for
#   the SoVI-like input table:
#
#       birth_rate
#       net_international_migration
#
# Original SoVI variables:
#
#       BRATE90   -> birth_rate
#       MIGRA_97  -> net_international_migration
#
# Source:
#   Statistics Canada Annual Demographic Estimates: Subprovincial Areas
#   Population and demographic components, total estimates, Canada's CDs,
#   2001 to 2021.
#
# Input workbook:
#
#   census_division_demographic_estimates_2021/raw/
#   population_estimates_for_canada_subprovincial_areas.xlsx
#
# Main formulas:
#
#   birth_rate =
#       1000 * births_2020_2021 / population_2021
#
#   net_international_migration =
#       immigrants_2020_2021
#     - emigrants_2020_2021
#     + returning_emigrants_2020_2021
#     - net_temporary_emigrants_2020_2021
#     + net_non_permanent_residents_2020_2021
#
# Notes:
#   - birth_rate is a crude birth rate per 1,000 population.
#   - net_international_migration is kept as a count for the SoVI variable.
#   - net_international_migration_per_1000 is also retained as an audit /
#     sensitivity variable.
#
# Run from data/:
#
#   python census_division_demographic_estimates_2021/clean_census_division_demographic_estimates_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_demographic_estimates_2021"
RAW_DIR = SECTION_DIR / "raw"
OUTPUT_DIR = SECTION_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_EXCEL = RAW_DIR / "population_estimates_for_canada_subprovincial_areas.xlsx"

BASE_CD_FRAME = (
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.csv"
)

OUTPUT_CLEAN = OUTPUT_DIR / "clean_census_division_demographic_estimates_2021.csv"
OUTPUT_COMPONENT_LONG = OUTPUT_DIR / "clean_census_division_demographic_estimates_component_long_2021.csv"
OUTPUT_METADATA = OUTPUT_DIR / "clean_census_division_demographic_estimates_variable_metadata_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "clean_census_division_demographic_estimates_summary_2021.csv"


# -----------------------------
# Config
# -----------------------------

EXPECTED_QC_CD_COUNT = 98

TARGET_SHEETS = {
    "population_2021_estimate": {
        "sheet_name": "Population",
        "target_kind": "single_year",
        "target_year": 2021,
        "role": "population_denominator_for_birth_rate",
        "unit": "persons",
    },
    "births_2020_2021": {
        "sheet_name": "Births~Naissances",
        "target_kind": "year_period",
        "target_start_year": 2020,
        "target_end_year": 2021,
        "role": "births_numerator_for_birth_rate",
        "unit": "births",
    },
    "immigrants_2020_2021": {
        "sheet_name": "Immigrants",
        "target_kind": "year_period",
        "target_start_year": 2020,
        "target_end_year": 2021,
        "role": "net_international_migration_component_positive",
        "unit": "persons",
    },
    "emigrants_2020_2021": {
        "sheet_name": "Emigrants~Émigrants",
        "target_kind": "year_period",
        "target_start_year": 2020,
        "target_end_year": 2021,
        "role": "net_international_migration_component_negative",
        "unit": "persons",
    },
    "returning_emigrants_2020_2021": {
        "sheet_name": "Ret.Emi~Émi de retour",
        "target_kind": "year_period",
        "target_start_year": 2020,
        "target_end_year": 2021,
        "role": "net_international_migration_component_positive",
        "unit": "persons",
    },
    "net_temporary_emigrants_2020_2021": {
        "sheet_name": "Net temp emi~Solde émig temp",
        "target_kind": "year_period",
        "target_start_year": 2020,
        "target_end_year": 2021,
        "role": "net_international_migration_component_negative",
        "unit": "persons",
    },
    "net_non_permanent_residents_2020_2021": {
        "sheet_name": "NPR(n)~RNP(s)",
        "target_kind": "year_period",
        "target_start_year": 2020,
        "target_end_year": 2021,
        "role": "net_international_migration_component_signed_stock_change",
        "unit": "persons",
    },
}

IDENTITY_COLUMNS = [
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


# -----------------------------
# Helpers
# -----------------------------

def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        return str(path)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = text.replace("\u00a0", " ")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def clean_numeric_value(value: object):
    text = normalize_text(value)

    if not text:
        return pd.NA

    text = (
        text.replace(",", "")
        .replace("$", "")
        .replace("%", "")
        .strip()
    )

    return pd.to_numeric(text, errors="coerce")


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("\u00a0", " ", regex=False)
        .str.strip(),
        errors="coerce",
    )


def normalize_cd_code(value: object) -> str:
    text = normalize_text(value)

    if not text:
        return ""

    numeric = pd.to_numeric(text, errors="coerce")
    if pd.notna(numeric):
        return f"{int(numeric):04d}"

    match = re.search(r"\b(\d{4})\b", text)
    if match:
        return match.group(1)

    return text


def extract_years(value: object) -> list[int]:
    text = normalize_text(value)
    years = re.findall(r"(?:19|20)\d{2}", text)
    return [int(year) for year in years]


def is_table_title(value: object) -> bool:
    text = normalize_text(value).lower()
    return text.startswith("table ") or text.startswith("tableau ")


def is_single_year_label(value: object) -> bool:
    text = normalize_text(value)
    years = extract_years(text)

    if is_table_title(text):
        return False

    if len(years) != 1:
        return False

    if len(text) > 40:
        return False

    return True


def is_period_label(value: object) -> bool:
    text = normalize_text(value)
    years = extract_years(text)

    if is_table_title(text):
        return False

    if len(years) != 2:
        return False

    if len(text) > 60:
        return False

    return True


def find_cd_code_column(raw: pd.DataFrame, base_cd_codes: set[str]) -> tuple[int | None, int]:
    best_col = None
    best_count = 0

    for col_idx in range(raw.shape[1]):
        values = raw.iloc[:, col_idx].map(normalize_cd_code)
        count = int(values.isin(base_cd_codes).sum())

        if count > best_count:
            best_count = count
            best_col = col_idx

    if best_count == 0:
        return None, 0

    return best_col, best_count


def find_name_column(raw: pd.DataFrame, cd_code_col: int | None, target_col: int | None) -> int | None:
    if cd_code_col is None:
        return None

    excluded = {cd_code_col}
    if target_col is not None:
        excluded.add(target_col)

    cd_rows_mask = raw.iloc[:, cd_code_col].map(normalize_cd_code).str.fullmatch(r"\d{4}", na=False)

    best_col = None
    best_score = 0

    for col_idx in range(raw.shape[1]):
        if col_idx in excluded:
            continue

        values = raw.loc[cd_rows_mask, col_idx].map(normalize_text)
        score = int(values.str.contains(r"[A-Za-zÀ-ÿ]", regex=True, na=False).sum())

        if score > best_score:
            best_score = score
            best_col = col_idx

    return best_col


def find_header_rows(raw: pd.DataFrame, target_kind: str) -> list[dict]:
    rows = []

    max_scan_rows = min(len(raw), 80)

    for row_idx in range(max_scan_rows):
        row = raw.iloc[row_idx]

        if target_kind == "single_year":
            label_cols = [
                col_idx
                for col_idx, value in enumerate(row.tolist())
                if is_single_year_label(value)
            ]
        else:
            label_cols = [
                col_idx
                for col_idx, value in enumerate(row.tolist())
                if is_period_label(value)
            ]

        if len(label_cols) >= 5:
            rows.append(
                {
                    "row_index": row_idx,
                    "label_count": len(label_cols),
                    "label_columns": label_cols,
                    "sample_labels": " | ".join(
                        normalize_text(raw.iat[row_idx, col_idx])
                        for col_idx in label_cols[:8]
                    ),
                }
            )

    rows = sorted(rows, key=lambda x: (x["label_count"], x["row_index"]), reverse=True)
    return rows


def find_target_column_strict(
    raw: pd.DataFrame,
    config: dict,
    cd_code_col: int | None,
    name_col: int | None,
    base_cd_codes: set[str],
) -> tuple[int | None, str, str, int | None, str]:
    target_kind = config["target_kind"]
    header_rows = find_header_rows(raw, target_kind=target_kind)

    for header_info in header_rows:
        row_idx = header_info["row_index"]
        label_cols = header_info["label_columns"]

        for col_idx in label_cols:
            value = raw.iat[row_idx, col_idx]
            years = extract_years(value)

            if target_kind == "single_year":
                if years == [config["target_year"]]:
                    return (
                        col_idx,
                        normalize_text(value),
                        f"strict_header_match_row_{row_idx}",
                        row_idx,
                        header_info["sample_labels"],
                    )

            else:
                if years == [config["target_start_year"], config["target_end_year"]]:
                    return (
                        col_idx,
                        normalize_text(value),
                        f"strict_header_match_row_{row_idx}",
                        row_idx,
                        header_info["sample_labels"],
                    )

    excluded_cols = set()
    if cd_code_col is not None:
        excluded_cols.add(cd_code_col)
    if name_col is not None:
        excluded_cols.add(name_col)

    if cd_code_col is None:
        return None, "", "target_column_not_found_no_cd_code_column", None, ""

    cd_rows_mask = raw.iloc[:, cd_code_col].map(normalize_cd_code).isin(base_cd_codes)

    numeric_candidates = []

    for col_idx in range(raw.shape[1]):
        if col_idx in excluded_cols:
            continue

        values = clean_numeric(raw.loc[cd_rows_mask, col_idx])
        non_missing = int(values.notna().sum())
        unique_values = int(values.nunique(dropna=True))

        if non_missing >= EXPECTED_QC_CD_COUNT * 0.9:
            numeric_candidates.append(
                {
                    "col_idx": col_idx,
                    "non_missing": non_missing,
                    "unique_values": unique_values,
                    "min": values.min(skipna=True),
                    "max": values.max(skipna=True),
                }
            )

    if numeric_candidates:
        numeric_candidates = sorted(
            numeric_candidates,
            key=lambda x: (x["col_idx"], x["unique_values"]),
            reverse=True,
        )

        selected = numeric_candidates[0]
        col_idx = selected["col_idx"]

        label = ""
        label_row = None

        for header_info in header_rows:
            row_idx = header_info["row_index"]
            possible_label = normalize_text(raw.iat[row_idx, col_idx])
            if possible_label:
                label = possible_label
                label_row = row_idx
                break

        if not label:
            label = f"column_{col_idx}"

        return (
            col_idx,
            label,
            "fallback_rightmost_numeric_data_column",
            label_row,
            " | ".join(x["sample_labels"] for x in header_rows[:2]),
        )

    return None, "", "target_column_not_found", None, " | ".join(x["sample_labels"] for x in header_rows[:2])


def contains_mojibake(series: pd.Series) -> int:
    text = series.astype("string")
    return int(text.str.contains("Ã|Â|�", regex=True, na=False).sum())


def numeric_summary(series: pd.Series) -> dict:
    numeric = clean_numeric(series)

    return {
        "non_missing": int(numeric.notna().sum()),
        "missing": int(numeric.isna().sum()),
        "unique_values": int(numeric.nunique(dropna=True)),
        "min": numeric.min(skipna=True),
        "max": numeric.max(skipna=True),
        "mean": numeric.mean(skipna=True),
        "median": numeric.median(skipna=True),
    }


def assert_plausible_component(component_alias: str, values: pd.Series) -> None:
    numeric = clean_numeric(values)
    non_missing = int(numeric.notna().sum())
    unique_count = int(numeric.nunique(dropna=True))
    min_value = numeric.min(skipna=True)
    max_value = numeric.max(skipna=True)

    if non_missing != EXPECTED_QC_CD_COUNT:
        raise ValueError(
            f"{component_alias} has {non_missing} non-missing values; expected {EXPECTED_QC_CD_COUNT}."
        )

    if unique_count <= 1:
        raise ValueError(
            f"{component_alias} appears implausible: all values are identical or only one unique value."
        )

    if component_alias == "population_2021_estimate" and min_value < 1000:
        raise ValueError(
            f"{component_alias} appears implausible: minimum population is {min_value}."
        )

    if component_alias == "births_2020_2021" and max_value <= 0:
        raise ValueError(
            f"{component_alias} appears implausible: maximum births is {max_value}."
        )


def parse_component_sheet(
    raw_excel: Path,
    component_alias: str,
    config: dict,
    base: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    sheet_name = config["sheet_name"]

    raw = pd.read_excel(
        raw_excel,
        sheet_name=sheet_name,
        header=None,
        dtype=str,
        engine="openpyxl",
    )

    raw = raw.dropna(how="all").dropna(axis=1, how="all")

    base_cd_codes = set(base["census_division_code"].astype(str).str.strip())

    cd_code_col, cd_code_matches = find_cd_code_column(raw, base_cd_codes)
    name_col_initial = find_name_column(raw, cd_code_col, None)

    target_col, target_col_label, target_col_detection, target_header_row, detected_header_sample = find_target_column_strict(
        raw=raw,
        config=config,
        cd_code_col=cd_code_col,
        name_col=name_col_initial,
        base_cd_codes=base_cd_codes,
    )

    name_col = find_name_column(raw, cd_code_col, target_col)

    if cd_code_col is None:
        raise ValueError(f"Could not find census-division code column in sheet {sheet_name}.")

    if target_col is None:
        raise ValueError(f"Could not find target year/period column in sheet {sheet_name}.")

    rows = []

    for raw_row_idx in range(len(raw)):
        cd_code = normalize_cd_code(raw.iat[raw_row_idx, cd_code_col])

        if cd_code not in base_cd_codes:
            continue

        source_geo_name = normalize_text(raw.iat[raw_row_idx, name_col]) if name_col is not None else ""
        value = clean_numeric_value(raw.iat[raw_row_idx, target_col])

        rows.append(
            {
                "component_alias": component_alias,
                "sheet_name": sheet_name,
                "raw_row_index_zero_indexed": raw_row_idx,
                "census_division_code": cd_code,
                "source_geo_name": source_geo_name,
                "target_header_row_zero_indexed": target_header_row if target_header_row is not None else "",
                "target_column_index_zero_indexed": target_col,
                "target_column_label": target_col_label,
                "target_column_detection": target_col_detection,
                "target_value": value,
                "unit": config["unit"],
                "role": config["role"],
            }
        )

    component_df = pd.DataFrame(rows)

    if component_df.empty:
        raise ValueError(f"No Québec CD rows extracted for {component_alias} from sheet {sheet_name}.")

    component_df["target_value"] = clean_numeric(component_df["target_value"])

    duplicate_cd_codes = int(component_df["census_division_code"].duplicated().sum())
    if duplicate_cd_codes != 0:
        raise ValueError(
            f"{component_alias} has duplicate census_division_code rows after extraction: {duplicate_cd_codes}"
        )

    joined = base[["census_division_code", "census_division_dguid", "census_division_name"]].merge(
        component_df[["census_division_code", "source_geo_name", "target_value"]],
        on="census_division_code",
        how="left",
        validate="one_to_one",
    )

    assert_plausible_component(component_alias, joined["target_value"])

    values = clean_numeric(joined["target_value"])

    inventory = {
        "component_alias": component_alias,
        "sheet_name": sheet_name,
        "sheet_rows_after_drop_empty": len(raw),
        "sheet_cols_after_drop_empty": raw.shape[1],
        "cd_code_column_index_zero_indexed": cd_code_col,
        "cd_code_matches_in_sheet": cd_code_matches,
        "name_column_index_zero_indexed": name_col if name_col is not None else "",
        "target_header_row_zero_indexed": target_header_row if target_header_row is not None else "",
        "target_column_index_zero_indexed": target_col,
        "target_column_label": target_col_label,
        "target_column_detection": target_col_detection,
        "detected_header_sample": detected_header_sample,
        "rows_extracted_for_base_qc_cds": len(component_df),
        "duplicate_cd_codes_extracted": duplicate_cd_codes,
        "matched_base_rows_non_missing": int(values.notna().sum()),
        "value_non_missing": int(values.notna().sum()),
        "value_missing": int(values.isna().sum()),
        "value_unique_count": int(values.nunique(dropna=True)),
        "value_min": values.min(skipna=True),
        "value_max": values.max(skipna=True),
        "value_mean": values.mean(skipna=True),
        "value_median": values.median(skipna=True),
        "coverage_is_98_cds": int(values.notna().sum()) == EXPECTED_QC_CD_COUNT,
        "coverage_status": "ready_for_cleaner_full_coverage",
    }

    return component_df, inventory


# -----------------------------
# Validate inputs
# -----------------------------

if not RAW_EXCEL.exists():
    raise FileNotFoundError(f"Missing raw Excel file:\n{RAW_EXCEL}")

if not BASE_CD_FRAME.exists():
    raise FileNotFoundError(f"Missing base CD frame:\n{BASE_CD_FRAME}")


# -----------------------------
# Load base frame
# -----------------------------

base = pd.read_csv(BASE_CD_FRAME, dtype=str, low_memory=False)
base.columns = [str(col).strip() for col in base.columns]

required_base_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
]

missing = [col for col in required_base_cols if col not in base.columns]
if missing:
    raise ValueError(
        "Base frame is missing required columns:\n"
        + "\n".join(missing)
        + "\n\nAvailable columns:\n"
        + "\n".join(base.columns)
    )

base = base.copy()
base["census_division_code"] = base["census_division_code"].astype(str).str.strip()
base["census_division_dguid"] = base["census_division_dguid"].astype(str).str.strip()
base["census_division_name"] = base["census_division_name"].astype(str).str.strip()

if len(base) != EXPECTED_QC_CD_COUNT:
    raise ValueError(f"Expected {EXPECTED_QC_CD_COUNT} Québec CDs, got {len(base)}.")

if base["census_division_code"].duplicated().any():
    raise ValueError("Duplicate census_division_code values in base frame.")

for optional_numeric_col in ["population_total_2021", "land_area_km2"]:
    if optional_numeric_col in base.columns:
        base[optional_numeric_col] = clean_numeric(base[optional_numeric_col])


# -----------------------------
# Workbook sheet check
# -----------------------------

try:
    xl = pd.ExcelFile(RAW_EXCEL)
except ImportError as exc:
    raise ImportError(
        "Reading Excel files requires openpyxl. Install it with:\n\n"
        "    pip install openpyxl\n"
    ) from exc

available_sheets = xl.sheet_names

missing_target_sheets = sorted(
    {
        config["sheet_name"]
        for config in TARGET_SHEETS.values()
        if config["sheet_name"] not in available_sheets
    }
)

if missing_target_sheets:
    raise ValueError(
        "Workbook is missing expected sheets:\n"
        + "\n".join(missing_target_sheets)
        + "\n\nAvailable sheets:\n"
        + "\n".join(available_sheets)
    )


print("\nCleaning Census Division Demographic Estimates 2021")
print("Raw Excel:", safe_relative(RAW_EXCEL))
print("Base CD frame:", safe_relative(BASE_CD_FRAME))
print("Base rows:", len(base))


# -----------------------------
# Parse components
# -----------------------------

component_frames = []
parse_inventory_rows = []

for component_alias, config in TARGET_SHEETS.items():
    print(f"\nParsing {component_alias} from sheet {config['sheet_name']}")

    component_df, inventory = parse_component_sheet(
        raw_excel=RAW_EXCEL,
        component_alias=component_alias,
        config=config,
        base=base,
    )

    component_frames.append(component_df)
    parse_inventory_rows.append(inventory)

    print(
        f"  status={inventory['coverage_status']}, "
        f"matches={inventory['matched_base_rows_non_missing']}/98, "
        f"unique={inventory['value_unique_count']}, "
        f"min={inventory['value_min']}, "
        f"max={inventory['value_max']}, "
        f"target_col={inventory['target_column_index_zero_indexed']}, "
        f"label={inventory['target_column_label']}"
    )


component_long = pd.concat(component_frames, ignore_index=True)
parse_inventory = pd.DataFrame(parse_inventory_rows)

component_long.to_csv(OUTPUT_COMPONENT_LONG, index=False, encoding="utf-8")


# -----------------------------
# Pivot to wide clean table
# -----------------------------

component_wide = (
    component_long
    .pivot_table(
        index="census_division_code",
        columns="component_alias",
        values="target_value",
        aggfunc="first",
    )
    .reset_index()
)

component_wide.columns.name = None

identity_cols = [col for col in IDENTITY_COLUMNS if col in base.columns]

clean = base[identity_cols].copy()

clean = clean.merge(
    component_wide,
    on="census_division_code",
    how="left",
    validate="one_to_one",
)

for component_alias in TARGET_SHEETS:
    clean[component_alias] = clean_numeric(clean[component_alias])


# -----------------------------
# Derived variables
# -----------------------------

clean["birth_rate_per_1000"] = (
    clean["births_2020_2021"] / clean["population_2021_estimate"] * 1000
)

# Main SoVI alias. Original BRATE90 is births per 1,000 population.
clean["birth_rate"] = clean["birth_rate_per_1000"]

clean["net_international_migration"] = (
    clean["immigrants_2020_2021"]
    - clean["emigrants_2020_2021"]
    + clean["returning_emigrants_2020_2021"]
    - clean["net_temporary_emigrants_2020_2021"]
    + clean["net_non_permanent_residents_2020_2021"]
)

clean["net_international_migration_per_1000"] = (
    clean["net_international_migration"] / clean["population_2021_estimate"] * 1000
)

if "population_total_2021" in clean.columns:
    clean["population_estimate_minus_census_population_2021"] = (
        clean["population_2021_estimate"] - clean["population_total_2021"]
    )
    clean["population_estimate_to_census_population_ratio_2021"] = (
        clean["population_2021_estimate"] / clean["population_total_2021"]
    )

clean["source_file"] = safe_relative(RAW_EXCEL)
clean["method_note"] = (
    "Derived from Statistics Canada Annual Demographic Estimates: Subprovincial Areas. "
    "birth_rate is 1000 * births_2020_2021 / population_2021_estimate. "
    "net_international_migration is immigrants - emigrants + returning emigrants "
    "- net temporary emigrants + net non-permanent residents for 2020-2021. "
    "Net non-permanent residents is treated as a signed stock-change component and added directly."
)


# -----------------------------
# Validation
# -----------------------------

if len(clean) != EXPECTED_QC_CD_COUNT:
    raise ValueError(f"Clean output has {len(clean)} rows; expected {EXPECTED_QC_CD_COUNT}.")

if clean["census_division_dguid"].duplicated().any():
    dupes = clean[clean["census_division_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicate census_division_dguid values in clean output:\n"
        + dupes[["census_division_code", "census_division_dguid", "census_division_name"]].to_string(index=False)
    )

required_numeric_cols = [
    "population_2021_estimate",
    "births_2020_2021",
    "immigrants_2020_2021",
    "emigrants_2020_2021",
    "returning_emigrants_2020_2021",
    "net_temporary_emigrants_2020_2021",
    "net_non_permanent_residents_2020_2021",
    "birth_rate",
    "birth_rate_per_1000",
    "net_international_migration",
    "net_international_migration_per_1000",
]

for col in required_numeric_cols:
    missing_count = int(clean[col].isna().sum())
    if missing_count != 0:
        raise ValueError(f"Unexpected missing values in {col}: {missing_count}")

if (clean["population_2021_estimate"] <= 0).any():
    raise ValueError("Non-positive population_2021_estimate values found.")

if (clean["births_2020_2021"] < 0).any():
    raise ValueError("Negative births_2020_2021 values found.")

birth_rate_formula_diff = (
    clean["birth_rate"]
    - (clean["births_2020_2021"] / clean["population_2021_estimate"] * 1000)
).abs().max(skipna=True)

birth_rate_alias_diff = (
    clean["birth_rate"] - clean["birth_rate_per_1000"]
).abs().max(skipna=True)

net_international_formula_check = (
    clean["immigrants_2020_2021"]
    - clean["emigrants_2020_2021"]
    + clean["returning_emigrants_2020_2021"]
    - clean["net_temporary_emigrants_2020_2021"]
    + clean["net_non_permanent_residents_2020_2021"]
)

net_international_formula_diff = (
    clean["net_international_migration"] - net_international_formula_check
).abs().max(skipna=True)

net_international_per_1000_formula_diff = (
    clean["net_international_migration_per_1000"]
    - clean["net_international_migration"] / clean["population_2021_estimate"] * 1000
).abs().max(skipna=True)

if birth_rate_formula_diff != 0:
    raise ValueError(f"birth_rate formula check failed: {birth_rate_formula_diff}")

if birth_rate_alias_diff != 0:
    raise ValueError(f"birth_rate alias check failed: {birth_rate_alias_diff}")

if net_international_formula_diff != 0:
    raise ValueError(f"net_international_migration formula check failed: {net_international_formula_diff}")

if net_international_per_1000_formula_diff != 0:
    raise ValueError(
        "net_international_migration_per_1000 formula check failed: "
        f"{net_international_per_1000_formula_diff}"
    )

base_names_with_mojibake = contains_mojibake(base["census_division_name"])
clean_names_with_mojibake = contains_mojibake(clean["census_division_name"])


# -----------------------------
# Metadata
# -----------------------------

metadata_rows = [
    {
        "variable": "birth_rate",
        "original_sovi_code": "BRATE90",
        "description": "Crude birth rate per 1,000 population",
        "source_sheet_or_component": "Births~Naissances and Population",
        "unit": "births_per_1000_population",
        "derivation": "1000 * births_2020_2021 / population_2021_estimate",
        "role": "recommended_sovi_birth_rate_variable",
        "notes": (
            "Uses 2020-2021 births and July 1, 2021 population estimate from Annual "
            "Demographic Estimates for census divisions."
        ),
    },
    {
        "variable": "birth_rate_per_1000",
        "original_sovi_code": "BRATE90",
        "description": "Explicit birth-rate audit variable",
        "source_sheet_or_component": "Births~Naissances and Population",
        "unit": "births_per_1000_population",
        "derivation": "1000 * births_2020_2021 / population_2021_estimate",
        "role": "birth_rate_audit_variable",
        "notes": "Same value as birth_rate.",
    },
    {
        "variable": "net_international_migration",
        "original_sovi_code": "MIGRA_97",
        "description": "Net international migration count",
        "source_sheet_or_component": (
            "Immigrants, Emigrants, Returning emigrants, Net temporary emigrants, "
            "Net non-permanent residents"
        ),
        "unit": "persons",
        "derivation": (
            "immigrants_2020_2021 - emigrants_2020_2021 + returning_emigrants_2020_2021 "
            "- net_temporary_emigrants_2020_2021 + net_non_permanent_residents_2020_2021"
        ),
        "role": "recommended_sovi_net_international_migration_variable",
        "notes": (
            "Uses 2020-2021 demographic components. Net non-permanent residents is treated "
            "as a signed stock-change component and added directly."
        ),
    },
    {
        "variable": "net_international_migration_per_1000",
        "original_sovi_code": "MIGRA_97_AUDIT",
        "description": "Net international migration per 1,000 population",
        "source_sheet_or_component": "Derived from net_international_migration and population_2021_estimate",
        "unit": "persons_per_1000_population",
        "derivation": "1000 * net_international_migration / population_2021_estimate",
        "role": "audit_or_sensitivity_variable",
        "notes": (
            "Retained for scale-adjusted interpretation. The default SoVI mapping should use "
            "net_international_migration unless the project explicitly chooses the per-1,000 version."
        ),
    },
]

for component_alias, config in TARGET_SHEETS.items():
    metadata_rows.append(
        {
            "variable": component_alias,
            "original_sovi_code": "",
            "description": config["role"],
            "source_sheet_or_component": config["sheet_name"],
            "unit": config["unit"],
            "derivation": "direct_sheet_value",
            "role": "component_audit_variable",
            "notes": "Retained for reproducibility of demographic-estimate derived variables.",
        }
    )

metadata = pd.DataFrame(metadata_rows)
metadata.to_csv(OUTPUT_METADATA, index=False, encoding="utf-8")


# -----------------------------
# Summary
# -----------------------------

summary_rows = [
    {"metric": "raw_excel", "value": safe_relative(RAW_EXCEL)},
    {"metric": "base_cd_frame", "value": safe_relative(BASE_CD_FRAME)},
    {"metric": "available_workbook_sheets", "value": ", ".join(available_sheets)},
    {"metric": "clean_rows", "value": len(clean)},
    {"metric": "unique_census_divisions", "value": clean["census_division_dguid"].nunique()},
    {"metric": "variables_cleaned", "value": "birth_rate, net_international_migration"},
    {"metric": "audit_variables", "value": "birth_rate_per_1000, net_international_migration_per_1000"},
    {"metric": "components_cleaned", "value": ", ".join(TARGET_SHEETS.keys())},
    {"metric": "all_required_numeric_columns_complete", "value": bool(clean[required_numeric_cols].notna().all().all())},
    {"metric": "birth_rate_formula_max_abs_difference", "value": birth_rate_formula_diff},
    {"metric": "birth_rate_alias_max_abs_difference", "value": birth_rate_alias_diff},
    {"metric": "net_international_migration_formula_max_abs_difference", "value": net_international_formula_diff},
    {"metric": "net_international_migration_per_1000_formula_max_abs_difference", "value": net_international_per_1000_formula_diff},
    {"metric": "base_names_with_mojibake", "value": base_names_with_mojibake},
    {"metric": "clean_names_with_mojibake", "value": clean_names_with_mojibake},
]

for _, row in parse_inventory.iterrows():
    summary_rows.append(
        {
            "metric": f"{row['component_alias']}_source_sheet",
            "value": row["sheet_name"],
        }
    )
    summary_rows.append(
        {
            "metric": f"{row['component_alias']}_target_column_label",
            "value": row["target_column_label"],
        }
    )
    summary_rows.append(
        {
            "metric": f"{row['component_alias']}_coverage_status",
            "value": row["coverage_status"],
        }
    )

for variable in required_numeric_cols:
    stats = numeric_summary(clean[variable])
    for key, value in stats.items():
        summary_rows.append(
            {
                "metric": f"{variable}_{key}",
                "value": value,
            }
        )

if "population_estimate_minus_census_population_2021" in clean.columns:
    for variable in [
        "population_estimate_minus_census_population_2021",
        "population_estimate_to_census_population_ratio_2021",
    ]:
        stats = numeric_summary(clean[variable])
        for key, value in stats.items():
            summary_rows.append(
                {
                    "metric": f"{variable}_{key}",
                    "value": value,
                }
            )

summary_rows.append(
    {
        "metric": "recommended_next_step",
        "value": (
            "If this summary shows 98 rows, complete variables, zero formula differences, "
            "and no mojibake, generate the README and add a SoVI YAML mapping for birth_rate and "
            "net_international_migration."
        ),
    }
)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Save clean output
# -----------------------------

ordered_cols = (
    identity_cols
    + [
        "birth_rate",
        "birth_rate_per_1000",
        "net_international_migration",
        "net_international_migration_per_1000",
        "population_2021_estimate",
        "births_2020_2021",
        "immigrants_2020_2021",
        "emigrants_2020_2021",
        "returning_emigrants_2020_2021",
        "net_temporary_emigrants_2020_2021",
        "net_non_permanent_residents_2020_2021",
    ]
)

optional_audit_cols = [
    "population_estimate_minus_census_population_2021",
    "population_estimate_to_census_population_ratio_2021",
]

ordered_cols = (
    [col for col in ordered_cols if col in clean.columns]
    + [col for col in optional_audit_cols if col in clean.columns]
    + ["source_file", "method_note"]
)

clean = clean[ordered_cols].copy()
clean.to_csv(OUTPUT_CLEAN, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CLEAN CENSUS DIVISION DEMOGRAPHIC ESTIMATES 2021")
print("=" * 72)

print("\nClean output:")
print("Rows:", len(clean))
print("Unique census divisions:", clean["census_division_dguid"].nunique())
print("Variables cleaned: birth_rate, net_international_migration")
print("All required numeric columns complete:", bool(clean[required_numeric_cols].notna().all().all()))

print("\nFormula checks:")
print("birth_rate formula max abs difference:", birth_rate_formula_diff)
print("birth_rate alias max abs difference:", birth_rate_alias_diff)
print("net_international_migration formula max abs difference:", net_international_formula_diff)
print("net_international_migration_per_1000 formula max abs difference:", net_international_per_1000_formula_diff)

print("\nMojibake check:")
print("Base names with mojibake:", base_names_with_mojibake)
print("Clean names with mojibake:", clean_names_with_mojibake)

print("\nMain variable summaries:")
for variable in [
    "birth_rate",
    "net_international_migration",
    "net_international_migration_per_1000",
]:
    print("\n", variable)
    print(clean[variable].describe())

print("\nComponent parse inventory:")
print(parse_inventory.to_string(index=False))

print("\nPreview:")
preview_cols = [
    "census_division_code",
    "census_division_name",
    "population_2021_estimate",
    "births_2020_2021",
    "birth_rate",
    "immigrants_2020_2021",
    "emigrants_2020_2021",
    "returning_emigrants_2020_2021",
    "net_temporary_emigrants_2020_2021",
    "net_non_permanent_residents_2020_2021",
    "net_international_migration",
    "net_international_migration_per_1000",
]
preview_cols = [col for col in preview_cols if col in clean.columns]
print(clean[preview_cols].head(20).to_string(index=False))

print("\nSaved:")
print(OUTPUT_CLEAN)
print(OUTPUT_COMPONENT_LONG)
print(OUTPUT_METADATA)
print(OUTPUT_SUMMARY)

print("\nDone.")