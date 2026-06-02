from pathlib import Path
import re
import pandas as pd


# ============================================================
# Strict Targeted Inspect Census Division Demographic Estimates 2021
# ============================================================
#
# Purpose:
#   Inspect the StatCan Annual Demographic Estimates Excel workbook using
#   its actual sheet structure, while avoiding false matches to title rows.
#
# Target SoVI variables:
#
#   BRATE90  -> birth_rate
#   MIGRA_97 -> net_international_migration
#
# Main correction from previous version:
#   The previous targeted script matched the sheet title row because it
#   contained "2021", then accidentally selected column 0. This stricter
#   version only accepts year/period labels from rows that look like real
#   tabular header rows, meaning rows with many year/period labels across
#   columns.
#
# Run from data/:
#
#   python census_division_demographic_estimates_2021/inspect_census_division_demographic_estimates_2021.py
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

OUTPUT_SHEET_PARSE_INVENTORY = OUTPUT_DIR / "demographic_estimates_targeted_sheet_parse_inventory_2021.csv"
OUTPUT_COMPONENT_COVERAGE = OUTPUT_DIR / "demographic_estimates_targeted_component_coverage_2021.csv"
OUTPUT_COMPONENT_WIDE_PREVIEW = OUTPUT_DIR / "demographic_estimates_targeted_component_wide_preview_2021.csv"
OUTPUT_COMPONENT_LONG = OUTPUT_DIR / "demographic_estimates_targeted_component_long_2021.csv"
OUTPUT_UNMATCHED_AUDIT = OUTPUT_DIR / "demographic_estimates_targeted_unmatched_audit_2021.csv"
OUTPUT_FORMULA_AUDIT = OUTPUT_DIR / "demographic_estimates_targeted_formula_audit_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "demographic_estimates_targeted_inspection_summary_2021.csv"


# -----------------------------
# Config
# -----------------------------

EXPECTED_QC_CD_COUNT = 98

TARGET_SHEETS = {
    "population_2021": {
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
        "role": "net_international_migration_component_positive_or_signed_stock_change",
        "unit": "persons",
    },
}


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

    # Avoid accepting long descriptive title cells.
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

    # Avoid accepting long descriptive title cells.
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

    # First choice: target label in a real header row.
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

    # Fallback: choose the rightmost numeric data column among Québec CD rows.
    # This is allowed only if no strict header match was found.
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
        # Prefer rightmost numeric column with some variation.
        numeric_candidates = sorted(
            numeric_candidates,
            key=lambda x: (x["col_idx"], x["unique_values"]),
            reverse=True,
        )

        selected = numeric_candidates[0]
        col_idx = selected["col_idx"]

        # Try to recover a nearby header label from candidate header rows.
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


def assess_value_plausibility(component_alias: str, values: pd.Series) -> tuple[bool, str]:
    numeric = clean_numeric(values)
    non_missing = int(numeric.notna().sum())
    unique_count = int(numeric.nunique(dropna=True))
    min_value = numeric.min(skipna=True)
    max_value = numeric.max(skipna=True)

    if non_missing != EXPECTED_QC_CD_COUNT:
        return False, "not_full_coverage"

    if unique_count <= 1:
        return False, "all_values_identical_or_single_unique_value"

    if component_alias == "population_2021":
        if min_value is None or pd.isna(min_value) or min_value < 1000:
            return False, "population_values_implausibly_small"

    if component_alias == "births_2020_2021":
        if max_value is None or pd.isna(max_value) or max_value <= 0:
            return False, "birth_values_non_positive_or_implausible"

    return True, "plausible"


def parse_component_sheet(
    raw_excel: Path,
    component_alias: str,
    config: dict,
    base: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
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

    parse_status = "parsed_candidate"
    if cd_code_col is None:
        parse_status = "cd_code_column_not_found"
    elif target_col is None:
        parse_status = "target_column_not_found"

    rows = []

    if target_col is not None and cd_code_col is not None:
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
        matched_base_rows = 0
        value_non_missing = 0
        value_missing = EXPECTED_QC_CD_COUNT
        duplicate_cd_codes = 0
        value_min = None
        value_max = None
        value_mean = None
        value_median = None
        value_unique_count = 0
        plausible = False
        plausibility_status = "no_rows_extracted"
    else:
        component_df["target_value"] = clean_numeric(component_df["target_value"])
        duplicate_cd_codes = int(component_df["census_division_code"].duplicated().sum())

        source_small = component_df[
            [
                "census_division_code",
                "source_geo_name",
                "target_value",
            ]
        ].drop_duplicates(subset=["census_division_code"], keep="first")

        joined = base[["census_division_code", "census_division_dguid", "census_division_name"]].merge(
            source_small,
            on="census_division_code",
            how="left",
            validate="one_to_one",
        )

        values = clean_numeric(joined["target_value"])
        matched_base_rows = int(values.notna().sum())
        value_non_missing = int(values.notna().sum())
        value_missing = int(values.isna().sum())
        value_min = values.min(skipna=True)
        value_max = values.max(skipna=True)
        value_mean = values.mean(skipna=True)
        value_median = values.median(skipna=True)
        value_unique_count = int(values.nunique(dropna=True))
        plausible, plausibility_status = assess_value_plausibility(component_alias, values)

    if (
        parse_status == "parsed_candidate"
        and matched_base_rows == EXPECTED_QC_CD_COUNT
        and duplicate_cd_codes == 0
        and plausible
    ):
        coverage_status = "ready_for_cleaner_candidate_full_coverage"
    elif parse_status == "parsed_candidate" and not plausible:
        coverage_status = f"parsed_but_implausible_values:{plausibility_status}"
    elif parse_status == "parsed_candidate":
        coverage_status = "parsed_but_needs_review"
    else:
        coverage_status = parse_status

    inventory = {
        "component_alias": component_alias,
        "sheet_name": sheet_name,
        "sheet_rows_after_drop_empty": len(raw),
        "sheet_cols_after_drop_empty": raw.shape[1],
        "cd_code_column_index_zero_indexed": cd_code_col if cd_code_col is not None else "",
        "cd_code_matches_in_sheet": cd_code_matches,
        "name_column_index_zero_indexed": name_col if name_col is not None else "",
        "target_header_row_zero_indexed": target_header_row if target_header_row is not None else "",
        "target_column_index_zero_indexed": target_col if target_col is not None else "",
        "target_column_label": target_col_label,
        "target_column_detection": target_col_detection,
        "detected_header_sample": detected_header_sample,
        "rows_extracted_for_base_qc_cds": len(component_df),
        "duplicate_cd_codes_extracted": duplicate_cd_codes,
        "matched_base_rows_non_missing": matched_base_rows,
        "value_non_missing": value_non_missing,
        "value_missing": value_missing,
        "value_unique_count": value_unique_count,
        "value_min": value_min,
        "value_max": value_max,
        "value_mean": value_mean,
        "value_median": value_median,
        "coverage_is_98_cds": matched_base_rows == EXPECTED_QC_CD_COUNT,
        "value_plausibility_status": plausibility_status,
        "parse_status": parse_status,
        "coverage_status": coverage_status,
    }

    if component_df.empty:
        unmatched = base[["census_division_code", "census_division_dguid", "census_division_name"]].copy()
        unmatched["component_alias"] = component_alias
        unmatched["missing_reason"] = "no_component_rows_extracted"
    else:
        source_small = component_df[
            [
                "census_division_code",
                "source_geo_name",
                "target_value",
            ]
        ].drop_duplicates(subset=["census_division_code"], keep="first")

        joined = base[["census_division_code", "census_division_dguid", "census_division_name"]].merge(
            source_small,
            on="census_division_code",
            how="left",
        )

        unmatched = joined[joined["target_value"].isna()].copy()
        unmatched["component_alias"] = component_alias
        unmatched["missing_reason"] = "missing_after_join"

    return component_df, inventory, unmatched


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


# -----------------------------
# Workbook sheet check
# -----------------------------

xl = pd.ExcelFile(RAW_EXCEL)
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

print("\nStrict targeted inspection of Census Division Demographic Estimates 2021")
print("Raw Excel:", safe_relative(RAW_EXCEL))
print("Base CD frame:", safe_relative(BASE_CD_FRAME))
print("Base rows:", len(base))
print("Available sheets:", ", ".join(available_sheets))


# -----------------------------
# Parse target sheets
# -----------------------------

component_frames = []
parse_inventory_rows = []
unmatched_frames = []

for component_alias, config in TARGET_SHEETS.items():
    print(f"\nParsing {component_alias} from sheet {config['sheet_name']}")

    component_df, inventory, unmatched = parse_component_sheet(
        raw_excel=RAW_EXCEL,
        component_alias=component_alias,
        config=config,
        base=base,
    )

    component_frames.append(component_df)
    parse_inventory_rows.append(inventory)
    unmatched_frames.append(unmatched)

    print(
        f"  status={inventory['coverage_status']}, "
        f"matches={inventory['matched_base_rows_non_missing']}/98, "
        f"unique={inventory['value_unique_count']}, "
        f"min={inventory['value_min']}, "
        f"max={inventory['value_max']}, "
        f"target_col={inventory['target_column_index_zero_indexed']}, "
        f"label={inventory['target_column_label']}"
    )


component_long = (
    pd.concat(component_frames, ignore_index=True)
    if component_frames
    else pd.DataFrame()
)

parse_inventory = pd.DataFrame(parse_inventory_rows)

unmatched_audit = (
    pd.concat(unmatched_frames, ignore_index=True)
    if unmatched_frames
    else pd.DataFrame()
)

if not unmatched_audit.empty:
    unmatched_audit = unmatched_audit[
        [
            "component_alias",
            "census_division_code",
            "census_division_dguid",
            "census_division_name",
            "missing_reason",
        ]
    ].copy()


# -----------------------------
# Build wide component preview
# -----------------------------

wide = base[
    [
        "census_division_code",
        "census_division_dguid",
        "census_division_name",
    ]
].copy()

if not component_long.empty:
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

    wide = wide.merge(
        component_wide,
        on="census_division_code",
        how="left",
        validate="one_to_one",
    )

if {"births_2020_2021", "population_2021"}.issubset(wide.columns):
    wide["birth_rate_per_1000_candidate"] = (
        clean_numeric(wide["births_2020_2021"])
        / clean_numeric(wide["population_2021"])
        * 1000
    )

net_components = [
    "immigrants_2020_2021",
    "emigrants_2020_2021",
    "returning_emigrants_2020_2021",
    "net_temporary_emigrants_2020_2021",
    "net_non_permanent_residents_2020_2021",
]

if set(net_components).issubset(wide.columns):
    wide["net_international_migration_candidate"] = (
        clean_numeric(wide["immigrants_2020_2021"])
        - clean_numeric(wide["emigrants_2020_2021"])
        + clean_numeric(wide["returning_emigrants_2020_2021"])
        - clean_numeric(wide["net_temporary_emigrants_2020_2021"])
        + clean_numeric(wide["net_non_permanent_residents_2020_2021"])
    )

    if "population_2021" in wide.columns:
        wide["net_international_migration_per_1000_candidate"] = (
            clean_numeric(wide["net_international_migration_candidate"])
            / clean_numeric(wide["population_2021"])
            * 1000
        )


# -----------------------------
# Component coverage
# -----------------------------

coverage_rows = []

for component_alias in TARGET_SHEETS:
    if component_alias in wide.columns:
        values = clean_numeric(wide[component_alias])
        coverage_rows.append(
            {
                "component_alias": component_alias,
                "non_missing": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "unique_values": int(values.nunique(dropna=True)),
                "min": values.min(skipna=True),
                "max": values.max(skipna=True),
                "mean": values.mean(skipna=True),
                "median": values.median(skipna=True),
                "coverage_is_98_cds": int(values.notna().sum()) == EXPECTED_QC_CD_COUNT,
                "values_are_not_constant": int(values.nunique(dropna=True)) > 1,
            }
        )
    else:
        coverage_rows.append(
            {
                "component_alias": component_alias,
                "non_missing": 0,
                "missing": EXPECTED_QC_CD_COUNT,
                "unique_values": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "coverage_is_98_cds": False,
                "values_are_not_constant": False,
            }
        )

component_coverage = pd.DataFrame(coverage_rows)


# -----------------------------
# Formula audit
# -----------------------------

birth_rate_components_ready = bool(
    component_coverage.loc[
        component_coverage["component_alias"].isin(["births_2020_2021", "population_2021"]),
        ["coverage_is_98_cds", "values_are_not_constant"],
    ].all().all()
)

net_international_components_ready = bool(
    component_coverage.loc[
        component_coverage["component_alias"].isin(net_components),
        ["coverage_is_98_cds", "values_are_not_constant"],
    ].all().all()
)

formula_rows = [
    {
        "target_variable": "birth_rate",
        "original_sovi_code": "BRATE90",
        "candidate_formula": "1000 * births_2020_2021 / population_2021",
        "required_components": "births_2020_2021, population_2021",
        "all_required_components_full_coverage_and_nonconstant": birth_rate_components_ready,
        "recommended_default_without_review": birth_rate_components_ready,
        "interpretation": (
            "Crude birth rate per 1,000 population using 2020/2021 births "
            "and July 1 2021 population estimate."
        ),
    },
    {
        "target_variable": "net_international_migration",
        "original_sovi_code": "MIGRA_97",
        "candidate_formula": (
            "immigrants_2020_2021 - emigrants_2020_2021 + returning_emigrants_2020_2021 "
            "- net_temporary_emigrants_2020_2021 + net_non_permanent_residents_2020_2021"
        ),
        "required_components": ", ".join(net_components),
        "all_required_components_full_coverage_and_nonconstant": net_international_components_ready,
        "recommended_default_without_review": False,
        "interpretation": (
            "Candidate net international migration construction from StatCan demographic components. "
            "Review sign conventions before final cleaner, especially net temporary emigrants and NPR stock change."
        ),
    },
]

formula_audit = pd.DataFrame(formula_rows)


# -----------------------------
# Save outputs
# -----------------------------

parse_inventory.to_csv(OUTPUT_SHEET_PARSE_INVENTORY, index=False, encoding="utf-8")
component_coverage.to_csv(OUTPUT_COMPONENT_COVERAGE, index=False, encoding="utf-8")
component_long.to_csv(OUTPUT_COMPONENT_LONG, index=False, encoding="utf-8")
wide.to_csv(OUTPUT_COMPONENT_WIDE_PREVIEW, index=False, encoding="utf-8")
unmatched_audit.to_csv(OUTPUT_UNMATCHED_AUDIT, index=False, encoding="utf-8")
formula_audit.to_csv(OUTPUT_FORMULA_AUDIT, index=False, encoding="utf-8")


# -----------------------------
# Summary
# -----------------------------

all_components_full_coverage = bool(component_coverage["coverage_is_98_cds"].all())
all_components_nonconstant = bool(component_coverage["values_are_not_constant"].all())

summary_rows = [
    {"metric": "raw_excel", "value": safe_relative(RAW_EXCEL)},
    {"metric": "base_cd_frame", "value": safe_relative(BASE_CD_FRAME)},
    {"metric": "base_rows", "value": len(base)},
    {"metric": "available_workbook_sheets", "value": ", ".join(available_sheets)},
    {"metric": "target_components_inspected", "value": ", ".join(TARGET_SHEETS.keys())},
    {"metric": "all_target_components_full_coverage", "value": all_components_full_coverage},
    {"metric": "all_target_components_nonconstant", "value": all_components_nonconstant},
    {"metric": "birth_rate_components_ready", "value": birth_rate_components_ready},
    {"metric": "net_international_migration_components_ready", "value": net_international_components_ready},
    {
        "metric": "components_not_full_coverage",
        "value": ", ".join(
            component_coverage.loc[
                ~component_coverage["coverage_is_98_cds"],
                "component_alias",
            ]
        ),
    },
    {
        "metric": "components_constant_or_implausible",
        "value": ", ".join(
            component_coverage.loc[
                ~component_coverage["values_are_not_constant"],
                "component_alias",
            ]
        ),
    },
    {
        "metric": "important_method_note",
        "value": (
            "This strict inspection ignores title rows and only accepts year/period labels from rows "
            "that contain many year-like columns. It also rejects all-constant extracted values, which "
            "catches accidental selection of province-code columns such as constant 24 for Québec."
        ),
    },
    {
        "metric": "recommended_next_step",
        "value": (
            "Review demographic_estimates_targeted_sheet_parse_inventory_2021.csv, "
            "demographic_estimates_targeted_component_coverage_2021.csv, and "
            "demographic_estimates_targeted_formula_audit_2021.csv. If all components have 98/98 coverage, "
            "nonconstant plausible values, and the selected columns are 2021 or 2020-2021, generate the cleaner."
        ),
    },
]

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("STRICT TARGETED CENSUS DIVISION DEMOGRAPHIC ESTIMATES INSPECTION 2021")
print("=" * 72)

print("\nSheet parse inventory:")
print(parse_inventory.to_string(index=False))

print("\nComponent coverage:")
print(component_coverage.to_string(index=False))

print("\nFormula audit:")
print(formula_audit.to_string(index=False))

print("\nWide preview:")
preview_cols = [
    "census_division_code",
    "census_division_name",
    "population_2021",
    "births_2020_2021",
    "birth_rate_per_1000_candidate",
    "immigrants_2020_2021",
    "emigrants_2020_2021",
    "returning_emigrants_2020_2021",
    "net_temporary_emigrants_2020_2021",
    "net_non_permanent_residents_2020_2021",
    "net_international_migration_candidate",
    "net_international_migration_per_1000_candidate",
]
preview_cols = [col for col in preview_cols if col in wide.columns]
print(wide[preview_cols].head(20).to_string(index=False))

print("\nSummary:")
print(summary.to_string(index=False))

print("\nSaved:")
print(OUTPUT_SHEET_PARSE_INVENTORY)
print(OUTPUT_COMPONENT_COVERAGE)
print(OUTPUT_COMPONENT_LONG)
print(OUTPUT_COMPONENT_WIDE_PREVIEW)
print(OUTPUT_UNMATCHED_AUDIT)
print(OUTPUT_FORMULA_AUDIT)
print(OUTPUT_SUMMARY)

print("\nDone.")