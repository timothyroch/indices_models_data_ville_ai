from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------
# Paths
# -----------------------------

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent

RAW_PROFILE_PATH = (
    DATA_DIR
    / "census_profile_2021"
    / "98-401-X2021007_English_CSV_data.csv"
)

OUTPUT_DIR = THIS_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "clean_census_tract_commuting_transport_2021.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_census_tract_commuting_transport_2021.parquet"


# -----------------------------
# Characteristic IDs
# -----------------------------
# Corrected from targeted inspection of IDs 2500-2650.
#
# The actual commuting / transport block is 2593-2623.
#
# Place of work status:
# 2593 = Total - Place of work status for the employed labour force aged 15 years and over - 25% sample data
# 2594 = Worked at home
# 2595 = Worked outside Canada
# 2596 = No fixed workplace address
# 2597 = Usual place of work
#
# Commuting destination:
# 2598 = Total - Commuting destination for the employed labour force aged 15 years and over with a usual place of work - 25% sample data
# 2599 = Commute within census subdivision (CSD) of residence
# 2600 = Commute to a different CSD within census division (CD) of residence
# 2601 = Commute to a different CSD and CD within province or territory of residence
# 2602 = Commute to a different province or territory
#
# Main mode of commuting:
# 2603 = Total - Main mode of commuting for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data
# 2604 = Car, truck or van
# 2605 = Car, truck or van - as a driver
# 2606 = Car, truck or van - as a passenger
# 2607 = Public transit
# 2608 = Walked
# 2609 = Bicycle
# 2610 = Other method
#
# Commuting duration:
# 2611 = Total - Commuting duration for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data
# 2612 = Less than 15 minutes
# 2613 = 15 to 29 minutes
# 2614 = 30 to 44 minutes
# 2615 = 45 to 59 minutes
# 2616 = 60 minutes and over
#
# Time leaving for work:
# 2617 = Total - Time leaving for work for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data
# 2618 = Between 5 a.m. and 5:59 a.m.
# 2619 = Between 6 a.m. and 6:59 a.m.
# 2620 = Between 7 a.m. and 7:59 a.m.
# 2621 = Between 8 a.m. and 8:59 a.m.
# 2622 = Between 9 a.m. and 11:59 a.m.
# 2623 = Between 12 p.m. and 4:59 a.m.
#
# Important methodological note:
# - This is NOT a direct "no vehicle available" variable.
# - It is a commuting / mobility-dependence feature family.
# - It can be useful for HGNN, SoVI-style sensitivity analysis, and Canadian
#   transportation-vulnerability proxies, but should not be described as exact
#   household vehicle access.

CHARACTERISTICS = {
    # Place of work status
    2593: "place_of_work_status_total_employed_labour_force",
    2594: "worked_at_home",
    2595: "worked_outside_canada",
    2596: "no_fixed_workplace_address",
    2597: "usual_place_of_work",

    # Commuting destination
    2598: "commuting_destination_total_usual_place_of_work",
    2599: "commute_within_csd_of_residence",
    2600: "commute_to_different_csd_within_cd",
    2601: "commute_to_different_csd_and_cd_within_province",
    2602: "commute_to_different_province_or_territory",

    # Main mode of commuting
    2603: "main_mode_commuting_total",
    2604: "commute_car_truck_van",
    2605: "commute_car_truck_van_driver",
    2606: "commute_car_truck_van_passenger",
    2607: "commute_public_transit",
    2608: "commute_walked",
    2609: "commute_bicycle",
    2610: "commute_other_method",

    # Commuting duration
    2611: "commuting_duration_total",
    2612: "commute_duration_less_than_15_min",
    2613: "commute_duration_15_to_29_min",
    2614: "commute_duration_30_to_44_min",
    2615: "commute_duration_45_to_59_min",
    2616: "commute_duration_60_min_and_over",

    # Time leaving for work
    2617: "time_leaving_for_work_total",
    2618: "leave_for_work_5_559_am",
    2619: "leave_for_work_6_659_am",
    2620: "leave_for_work_7_759_am",
    2621: "leave_for_work_8_859_am",
    2622: "leave_for_work_9_1159_am",
    2623: "leave_for_work_12pm_459am",
}

SOURCE_COMMUTING_TRANSPORT = (
    "Statistics Canada Census Profile, 2021 Census of Population; "
    "place of work, commuting destination, main commuting mode, commuting duration, "
    "and time leaving for work, 25% sample data"
)


# -----------------------------
# Helper functions
# -----------------------------

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Divide two numeric pandas Series safely.

    If the denominator is 0 or missing, the result is np.nan.
    This avoids inf values while keeping the output column numeric.
    """
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")

    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], np.nan)

    return result.astype(float)


def bounded_proportion(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Compute a proportion and clip it to [0, 1].

    This protects derived proportions from minor Census rounding artifacts.
    Raw counts are preserved unchanged.
    """
    result = safe_divide(numerator, denominator)
    result = result.clip(lower=0, upper=1)
    return result.astype(float)


# -----------------------------
# Load only needed columns
# -----------------------------

usecols = [
    "DGUID",
    "GEO_LEVEL",
    "GEO_NAME",
    "CHARACTERISTIC_ID",
    "CHARACTERISTIC_NAME",
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

print("Loaded Census Profile")
print("Rows:", len(df))


# -----------------------------
# Filter to census tract commuting / transport rows
# -----------------------------

df["CHARACTERISTIC_ID"] = pd.to_numeric(
    df["CHARACTERISTIC_ID"],
    errors="coerce",
).astype("Int64")

rows = df[
    (df["GEO_LEVEL"].astype(str).str.strip() == "Census tract")
    & (df["CHARACTERISTIC_ID"].isin(CHARACTERISTICS.keys()))
].copy()

print("\nCommuting / transport rows found:", len(rows))

if rows.empty:
    raise ValueError("No census tract commuting / transport rows found.")


# -----------------------------
# Inspect selected characteristics
# -----------------------------

selected_characteristics = (
    rows[
        [
            "CHARACTERISTIC_ID",
            "CHARACTERISTIC_NAME",
        ]
    ]
    .drop_duplicates()
    .sort_values("CHARACTERISTIC_ID")
)

print("\nSelected characteristics:")
print(selected_characteristics.to_string(index=False))


# -----------------------------
# Check that all expected characteristics were found
# -----------------------------

found_ids = set(rows["CHARACTERISTIC_ID"].dropna().astype(int).unique())
expected_ids = set(CHARACTERISTICS.keys())
missing_ids = sorted(expected_ids - found_ids)

if missing_ids:
    print("\nWarning: some expected characteristic IDs were not found:")
    for characteristic_id in missing_ids:
        print(f"  {characteristic_id}: {CHARACTERISTICS[characteristic_id]}")


# -----------------------------
# Prepare value column
# -----------------------------

rows["value"] = pd.to_numeric(
    rows["C1_COUNT_TOTAL"],
    errors="coerce",
)


# -----------------------------
# Report missing / suppressed values
# -----------------------------

missing = rows[rows["value"].isna()].copy()

print("\nMissing or non-numeric commuting / transport values:", len(missing))

if not missing.empty:
    print("\nMissing values by characteristic:")
    print(
        missing["CHARACTERISTIC_ID"]
        .map(CHARACTERISTICS)
        .value_counts(dropna=False)
    )

    print("\nSYMBOL counts for missing rows:")
    print(missing["SYMBOL"].value_counts(dropna=False))

    print("\nMissing preview:")
    print(
        missing[
            [
                "DGUID",
                "GEO_NAME",
                "CHARACTERISTIC_ID",
                "CHARACTERISTIC_NAME",
                "C1_COUNT_TOTAL",
                "C10_RATE_TOTAL",
                "SYMBOL",
            ]
        ]
        .head(50)
        .to_string(index=False)
    )


# -----------------------------
# Pivot to one row per census tract
# -----------------------------

rows["feature_name"] = rows["CHARACTERISTIC_ID"].map(CHARACTERISTICS)

wide = (
    rows
    .set_index(["DGUID", "GEO_NAME", "feature_name"])["value"]
    .unstack("feature_name")
    .reset_index()
)

wide.columns.name = None

clean = wide.rename(
    columns={
        "DGUID": "statcan_dguid",
        "GEO_NAME": "geo_name",
    }
).copy()


# -----------------------------
# Ensure all expected output columns exist
# -----------------------------

for column_name in CHARACTERISTICS.values():
    if column_name not in clean.columns:
        clean[column_name] = np.nan


# -----------------------------
# Compute place-of-work status proportions
# -----------------------------

clean["pct_worked_at_home"] = bounded_proportion(
    clean["worked_at_home"],
    clean["place_of_work_status_total_employed_labour_force"],
)

clean["pct_worked_outside_canada"] = bounded_proportion(
    clean["worked_outside_canada"],
    clean["place_of_work_status_total_employed_labour_force"],
)

clean["pct_no_fixed_workplace_address"] = bounded_proportion(
    clean["no_fixed_workplace_address"],
    clean["place_of_work_status_total_employed_labour_force"],
)

clean["pct_usual_place_of_work"] = bounded_proportion(
    clean["usual_place_of_work"],
    clean["place_of_work_status_total_employed_labour_force"],
)


# -----------------------------
# Compute commuting destination proportions
# -----------------------------

clean["pct_commute_within_csd_of_residence"] = bounded_proportion(
    clean["commute_within_csd_of_residence"],
    clean["commuting_destination_total_usual_place_of_work"],
)

clean["pct_commute_to_different_csd_within_cd"] = bounded_proportion(
    clean["commute_to_different_csd_within_cd"],
    clean["commuting_destination_total_usual_place_of_work"],
)

clean["pct_commute_to_different_csd_and_cd_within_province"] = bounded_proportion(
    clean["commute_to_different_csd_and_cd_within_province"],
    clean["commuting_destination_total_usual_place_of_work"],
)

clean["pct_commute_to_different_province_or_territory"] = bounded_proportion(
    clean["commute_to_different_province_or_territory"],
    clean["commuting_destination_total_usual_place_of_work"],
)


# -----------------------------
# Compute main commuting mode proportions
# -----------------------------

clean["pct_commute_car_truck_van"] = bounded_proportion(
    clean["commute_car_truck_van"],
    clean["main_mode_commuting_total"],
)

clean["pct_commute_car_truck_van_driver"] = bounded_proportion(
    clean["commute_car_truck_van_driver"],
    clean["main_mode_commuting_total"],
)

clean["pct_commute_car_truck_van_passenger"] = bounded_proportion(
    clean["commute_car_truck_van_passenger"],
    clean["main_mode_commuting_total"],
)

clean["pct_commute_public_transit"] = bounded_proportion(
    clean["commute_public_transit"],
    clean["main_mode_commuting_total"],
)

clean["pct_commute_walked"] = bounded_proportion(
    clean["commute_walked"],
    clean["main_mode_commuting_total"],
)

clean["pct_commute_bicycle"] = bounded_proportion(
    clean["commute_bicycle"],
    clean["main_mode_commuting_total"],
)

clean["pct_commute_other_method"] = bounded_proportion(
    clean["commute_other_method"],
    clean["main_mode_commuting_total"],
)

# Useful composite mobility features.
clean["commute_active_transport"] = (
    clean["commute_walked"]
    + clean["commute_bicycle"]
)

clean["pct_commute_active_transport"] = bounded_proportion(
    clean["commute_active_transport"],
    clean["main_mode_commuting_total"],
)

clean["commute_non_car_modes"] = (
    clean["commute_public_transit"]
    + clean["commute_walked"]
    + clean["commute_bicycle"]
    + clean["commute_other_method"]
)

clean["pct_commute_non_car_modes"] = bounded_proportion(
    clean["commute_non_car_modes"],
    clean["main_mode_commuting_total"],
)


# -----------------------------
# Compute commuting duration proportions
# -----------------------------

clean["pct_commute_duration_less_than_15_min"] = bounded_proportion(
    clean["commute_duration_less_than_15_min"],
    clean["commuting_duration_total"],
)

clean["pct_commute_duration_15_to_29_min"] = bounded_proportion(
    clean["commute_duration_15_to_29_min"],
    clean["commuting_duration_total"],
)

clean["pct_commute_duration_30_to_44_min"] = bounded_proportion(
    clean["commute_duration_30_to_44_min"],
    clean["commuting_duration_total"],
)

clean["pct_commute_duration_45_to_59_min"] = bounded_proportion(
    clean["commute_duration_45_to_59_min"],
    clean["commuting_duration_total"],
)

clean["pct_commute_duration_60_min_and_over"] = bounded_proportion(
    clean["commute_duration_60_min_and_over"],
    clean["commuting_duration_total"],
)

clean["commute_duration_45_min_and_over"] = (
    clean["commute_duration_45_to_59_min"]
    + clean["commute_duration_60_min_and_over"]
)

clean["pct_commute_duration_45_min_and_over"] = bounded_proportion(
    clean["commute_duration_45_min_and_over"],
    clean["commuting_duration_total"],
)

clean["commute_duration_30_min_and_over"] = (
    clean["commute_duration_30_to_44_min"]
    + clean["commute_duration_45_to_59_min"]
    + clean["commute_duration_60_min_and_over"]
)

clean["pct_commute_duration_30_min_and_over"] = bounded_proportion(
    clean["commute_duration_30_min_and_over"],
    clean["commuting_duration_total"],
)


# -----------------------------
# Compute time-leaving-for-work proportions
# -----------------------------

clean["pct_leave_for_work_5_559_am"] = bounded_proportion(
    clean["leave_for_work_5_559_am"],
    clean["time_leaving_for_work_total"],
)

clean["pct_leave_for_work_6_659_am"] = bounded_proportion(
    clean["leave_for_work_6_659_am"],
    clean["time_leaving_for_work_total"],
)

clean["pct_leave_for_work_7_759_am"] = bounded_proportion(
    clean["leave_for_work_7_759_am"],
    clean["time_leaving_for_work_total"],
)

clean["pct_leave_for_work_8_859_am"] = bounded_proportion(
    clean["leave_for_work_8_859_am"],
    clean["time_leaving_for_work_total"],
)

clean["pct_leave_for_work_9_1159_am"] = bounded_proportion(
    clean["leave_for_work_9_1159_am"],
    clean["time_leaving_for_work_total"],
)

clean["pct_leave_for_work_12pm_459am"] = bounded_proportion(
    clean["leave_for_work_12pm_459am"],
    clean["time_leaving_for_work_total"],
)

clean["leave_for_work_before_7am"] = (
    clean["leave_for_work_5_559_am"]
    + clean["leave_for_work_6_659_am"]
)

clean["pct_leave_for_work_before_7am"] = bounded_proportion(
    clean["leave_for_work_before_7am"],
    clean["time_leaving_for_work_total"],
)


# -----------------------------
# Add default mobility / transport proxy fields
# -----------------------------

clean["car_commuting_measure_default"] = clean["pct_commute_car_truck_van"].astype(float)

clean["car_commuting_measure_default_description"] = (
    "pct_commute_car_truck_van; commuting-mode proxy for car dependence, not household vehicle availability"
)

clean["public_transit_commuting_measure_default"] = clean[
    "pct_commute_public_transit"
].astype(float)

clean["public_transit_commuting_measure_default_description"] = (
    "pct_commute_public_transit; commuting-mode proxy for transit reliance"
)

clean["long_commute_measure_default"] = clean[
    "pct_commute_duration_45_min_and_over"
].astype(float)

clean["long_commute_measure_default_description"] = (
    "pct_commute_duration_45_min_and_over; proxy for long commuting burden"
)

clean["work_from_home_measure_default"] = clean["pct_worked_at_home"].astype(float)

clean["work_from_home_measure_default_description"] = (
    "pct_worked_at_home; share of employed labour force working at home"
)


# -----------------------------
# Add metadata
# -----------------------------

clean["unit_type"] = "census_tract"
clean["census_year"] = 2021
clean["source_commuting_transport"] = SOURCE_COMMUTING_TRANSPORT


# -----------------------------
# Reorder columns
# -----------------------------

ordered_columns = [
    "statcan_dguid",
    "geo_name",
    "unit_type",
    "census_year",

    # Place of work status
    "place_of_work_status_total_employed_labour_force",
    "worked_at_home",
    "worked_outside_canada",
    "no_fixed_workplace_address",
    "usual_place_of_work",
    "pct_worked_at_home",
    "pct_worked_outside_canada",
    "pct_no_fixed_workplace_address",
    "pct_usual_place_of_work",

    # Commuting destination
    "commuting_destination_total_usual_place_of_work",
    "commute_within_csd_of_residence",
    "commute_to_different_csd_within_cd",
    "commute_to_different_csd_and_cd_within_province",
    "commute_to_different_province_or_territory",
    "pct_commute_within_csd_of_residence",
    "pct_commute_to_different_csd_within_cd",
    "pct_commute_to_different_csd_and_cd_within_province",
    "pct_commute_to_different_province_or_territory",

    # Main commuting mode
    "main_mode_commuting_total",
    "commute_car_truck_van",
    "commute_car_truck_van_driver",
    "commute_car_truck_van_passenger",
    "commute_public_transit",
    "commute_walked",
    "commute_bicycle",
    "commute_other_method",
    "pct_commute_car_truck_van",
    "pct_commute_car_truck_van_driver",
    "pct_commute_car_truck_van_passenger",
    "pct_commute_public_transit",
    "pct_commute_walked",
    "pct_commute_bicycle",
    "pct_commute_other_method",
    "commute_active_transport",
    "pct_commute_active_transport",
    "commute_non_car_modes",
    "pct_commute_non_car_modes",

    # Commuting duration
    "commuting_duration_total",
    "commute_duration_less_than_15_min",
    "commute_duration_15_to_29_min",
    "commute_duration_30_to_44_min",
    "commute_duration_45_to_59_min",
    "commute_duration_60_min_and_over",
    "pct_commute_duration_less_than_15_min",
    "pct_commute_duration_15_to_29_min",
    "pct_commute_duration_30_to_44_min",
    "pct_commute_duration_45_to_59_min",
    "pct_commute_duration_60_min_and_over",
    "commute_duration_45_min_and_over",
    "pct_commute_duration_45_min_and_over",
    "commute_duration_30_min_and_over",
    "pct_commute_duration_30_min_and_over",

    # Time leaving for work
    "time_leaving_for_work_total",
    "leave_for_work_5_559_am",
    "leave_for_work_6_659_am",
    "leave_for_work_7_759_am",
    "leave_for_work_8_859_am",
    "leave_for_work_9_1159_am",
    "leave_for_work_12pm_459am",
    "pct_leave_for_work_5_559_am",
    "pct_leave_for_work_6_659_am",
    "pct_leave_for_work_7_759_am",
    "pct_leave_for_work_8_859_am",
    "pct_leave_for_work_9_1159_am",
    "pct_leave_for_work_12pm_459am",
    "leave_for_work_before_7am",
    "pct_leave_for_work_before_7am",

    # Defaults / named proxies
    "car_commuting_measure_default",
    "car_commuting_measure_default_description",
    "public_transit_commuting_measure_default",
    "public_transit_commuting_measure_default_description",
    "long_commute_measure_default",
    "long_commute_measure_default_description",
    "work_from_home_measure_default",
    "work_from_home_measure_default_description",

    "source_commuting_transport",
]

clean = clean[ordered_columns]


# -----------------------------
# Validation
# -----------------------------

if clean["statcan_dguid"].duplicated().any():
    duplicated = clean[clean["statcan_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated statcan_dguid values found:\n"
        + duplicated.to_string(index=False)
    )

computed_cols = [
    "pct_worked_at_home",
    "pct_worked_outside_canada",
    "pct_no_fixed_workplace_address",
    "pct_usual_place_of_work",

    "pct_commute_within_csd_of_residence",
    "pct_commute_to_different_csd_within_cd",
    "pct_commute_to_different_csd_and_cd_within_province",
    "pct_commute_to_different_province_or_territory",

    "pct_commute_car_truck_van",
    "pct_commute_car_truck_van_driver",
    "pct_commute_car_truck_van_passenger",
    "pct_commute_public_transit",
    "pct_commute_walked",
    "pct_commute_bicycle",
    "pct_commute_other_method",
    "pct_commute_active_transport",
    "pct_commute_non_car_modes",

    "pct_commute_duration_less_than_15_min",
    "pct_commute_duration_15_to_29_min",
    "pct_commute_duration_30_to_44_min",
    "pct_commute_duration_45_to_59_min",
    "pct_commute_duration_60_min_and_over",
    "pct_commute_duration_45_min_and_over",
    "pct_commute_duration_30_min_and_over",

    "pct_leave_for_work_5_559_am",
    "pct_leave_for_work_6_659_am",
    "pct_leave_for_work_7_759_am",
    "pct_leave_for_work_8_859_am",
    "pct_leave_for_work_9_1159_am",
    "pct_leave_for_work_12pm_459am",
    "pct_leave_for_work_before_7am",

    "car_commuting_measure_default",
    "public_transit_commuting_measure_default",
    "long_commute_measure_default",
    "work_from_home_measure_default",
]

for col in computed_cols:
    clean[col] = pd.to_numeric(clean[col], errors="coerce").astype(float)

    if np.isinf(clean[col]).any():
        raise ValueError(f"Infinite values found in computed column: {col}")

    min_value = clean[col].min(skipna=True)
    max_value = clean[col].max(skipna=True)

    if min_value < 0 or max_value > 1:
        raise ValueError(
            f"Out-of-bounds values remain in {col}: "
            f"min={min_value}, max={max_value}"
        )

print("\nClean commuting / transport table")
print("Rows:", len(clean))
print("Columns:", list(clean.columns))

print("\nMissing values by column:")
print(clean.isna().sum())

print("\nDefault car commuting measure summary:")
print(clean["car_commuting_measure_default"].describe())

print("\nDefault public transit commuting measure summary:")
print(clean["public_transit_commuting_measure_default"].describe())

print("\nDefault long commute measure summary:")
print(clean["long_commute_measure_default"].describe())

print("\nDefault work-from-home measure summary:")
print(clean["work_from_home_measure_default"].describe())

print("\nNo fixed workplace address summary:")
print(clean["pct_no_fixed_workplace_address"].describe())

print("\nNon-car commuting modes summary:")
print(clean["pct_commute_non_car_modes"].describe())

print("\nPreview:")
print(clean.head(10).to_string(index=False))


# -----------------------------
# Save outputs
# -----------------------------

clean.to_csv(OUTPUT_CSV, index=False)
clean.to_parquet(OUTPUT_PARQUET, index=False)

print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_PARQUET)