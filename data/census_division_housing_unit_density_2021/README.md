# Census Division Housing Unit Density 2021

This folder contains the Québec census-division transformation for the SoVI-like housing unit density variable.

The cleaned output supports:

```text
housing_unit_density
````

This corresponds to the original SoVI variable:

```text
HODENUT90 = housing units per square mile
```

## Source data

The source is the cleaned Québec census-division spatial/population frame:

```text
census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.csv
```

The transformation uses two source columns:

```text
total_private_dwellings_2021
land_area_km2
```

No new raw Census Profile extraction is performed in this block. The variable is derived from the already-cleaned base geography and population table.

## Method

The cleaner computes housing-unit density in two units:

```text
housing_unit_density_per_km2 =
    total_private_dwellings_2021 / land_area_km2
```

and:

```text
housing_unit_density_per_sq_mile =
    total_private_dwellings_2021 / (land_area_km2 * 0.3861021585424458)
```

The main SoVI-ready alias is:

```text
housing_unit_density
```

This alias uses the square-mile version because the original SoVI variable is defined as housing units per square mile.

## Script

Run from the `data/` folder:

```bash
python census_division_housing_unit_density_2021/clean_census_division_housing_unit_density_2021.py
```

## Outputs

Main clean output:

```text
census_division_housing_unit_density_2021/output/clean_census_division_housing_unit_density_2021.csv
```

Additional documentation outputs:

```text
census_division_housing_unit_density_2021/output/clean_census_division_housing_unit_density_variable_metadata_2021.csv
census_division_housing_unit_density_2021/output/clean_census_division_housing_unit_density_summary_2021.csv
```

## Validation

The latest successful run produced:

```text
Rows: 98
Unique census divisions: 98
All housing_unit_density values complete: True
housing_unit_density alias max abs difference: 0.0
Base names with mojibake: 0
Clean names with mojibake: 0
```

The output therefore provides complete coverage for all 98 Québec census divisions.

## Interpretation

`housing_unit_density` measures the density of private dwellings relative to land area. It is a built-environment density variable, not a direct measure of social vulnerability by itself.

For the SoVI-like table, it is retained because the original SoVI methodology includes housing units per square mile as a structural density indicator. In this Canadian adaptation, the numerator is total private dwellings in 2021 and the denominator is census-division land area converted to square miles.

````

After that, rerun:

```bash
python sovi_2021/inspect_census_division_sovi_input_sources_2021.py
````

Expected result:

```text
variables_ready_for_draft_table: 25 -> 26
variables_not_ready_or_unmapped: 17 -> 16
```
