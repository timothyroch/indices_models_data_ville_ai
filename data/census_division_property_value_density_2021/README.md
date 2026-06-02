# Census Division Property Value Density 2021

This folder contains the Québec census-division transformation for the SoVI-like variable:

```text
property_value_density
```

It corresponds to the original SoVI variable:

```text
RPROPDEN92 -> property_value_density
```

## Source data

The source is the 2021 Census Profile for Canadian census divisions:

```text
census_profile_census_division_2021/raw/98-401-X2021004_English_CSV_data.csv
```

The transformation also uses the cleaned housing tenure/costs output:

```text
census_division_housing_tenure_costs_2021/output/clean_census_division_housing_tenure_costs_2021.csv
```

and joins to the cleaned Québec census-division base frame:

```text
census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.csv
```

The base frame supplies the denominator:

```text
land_area_km2
```

## Scripts

Inspection script:

```bash
python census_division_property_value_density_2021/inspect_census_division_property_value_density_2021.py
```

Cleaning script:

```bash
python census_division_property_value_density_2021/clean_census_division_property_value_density_2021.py
```

## Variable construction

The original SoVI variable `RPROPDEN92` refers to a broad property-value / farm-products-sold density concept. A direct Canadian total assessed-property-value variable was not available in the existing cleaned sources. This implementation therefore uses a documented weak fallback proxy based on owner-occupied residential dwelling value.

The main source components are:

```text
CHARACTERISTIC_ID = 1415
Owner

CHARACTERISTIC_ID = 1488
Median value of dwellings ($)
```

The cleaned variable is:

```text
property_value_density =
    median_owner_occupied_housing_value
    *
    owner_households_direct_count
    /
    land_area_km2
```

Its unit is:

```text
estimated dollars per square kilometre
```

The numerator is retained as:

```text
estimated_owner_occupied_residential_property_value =
    median_owner_occupied_housing_value
    *
    owner_households_direct_count
```

## Proxy quality

This variable is not a true total property-assessment density. It should be described as:

```text
weak_residential_owner_occupied_property_value_density_proxy
```

It approximates the intensity of owner-occupied residential dwelling value per unit land area. It does not include rental property, commercial property, industrial property, institutional property, farm property, land assessment, or other components of total assessed property value.

## Audit variables

The cleaner retains two alternative owner-count reconstructions as sensitivity checks:

```text
property_value_density_estimated_owner_from_pct_renter
property_value_density_estimated_owner_from_total_minus_renter
```

The relevant audit components are:

```text
total_tenure_households_count
owner_households_direct_count
renter_households_direct_count
owner_households_rate_pct
renter_households_rate_pct
pct_renter_occupied
median_value_of_dwellings_profile_value
```

The cleaner also retains consistency diagnostics:

```text
owner_plus_renter_minus_total_tenure_count
owner_direct_minus_owner_estimated_from_pct_renter
median_value_profile_minus_housing_clean
```

Small differences in the owner/renter/total-tenure consistency diagnostics are expected because the inputs come from Census Profile 25% sample data and rounded published counts/rates. The main formula uses the direct owner household count.

## Coverage

The cleaned output has full coverage:

```text
98 / 98 Québec census divisions
```

The latest successful run produced:

```text
Rows: 98
Unique census divisions: 98
Variables cleaned: property_value_density
Original SoVI code: RPROPDEN92
Proxy quality: weak_residential_owner_occupied_property_value_density_proxy

main_formula_diff: 1.862645149230957e-09
pct_formula_diff: 2.9802322387695312e-08
minus_renter_formula_diff: 3.725290298461914e-09
estimated_value_formula_diff: 0.0
median_profile_housing_max_abs_diff: 0.0

Base names with mojibake: 0
Clean names with mojibake: 0
Raw characteristic names with mojibake: 0
```

Summary statistics from the latest run:

```text
property_value_density:
    min    1,099.593994
    mean   15,710,349.106935
    median 1,412,629.058446
    max    393,159,031.250753

estimated_owner_occupied_residential_property_value:
    min    409,500,000
    mean   7,610,355,612.244898
    median 2,545,500,000
    max    195,908,000,000
```

## Outputs

Main clean output:

```text
census_division_property_value_density_2021/output/clean_census_division_property_value_density_2021.csv
```

Audit outputs:

```text
census_division_property_value_density_2021/output/clean_census_division_property_value_density_source_rows_2021.csv
census_division_property_value_density_2021/output/clean_census_division_property_value_density_variable_metadata_2021.csv
census_division_property_value_density_2021/output/clean_census_division_property_value_density_formula_audit_2021.csv
census_division_property_value_density_2021/output/clean_census_division_property_value_density_unmatched_audit_2021.csv
census_division_property_value_density_2021/output/clean_census_division_property_value_density_summary_2021.csv
```

## Interpretation

`property_value_density` measures estimated owner-occupied residential dwelling-value intensity per square kilometre. It is a full-coverage fallback proxy for `RPROPDEN92`, but it should not be interpreted as total assessed property-value density.