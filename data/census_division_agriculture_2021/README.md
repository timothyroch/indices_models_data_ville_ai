# Census Division Agriculture 2021

This folder contains the Québec census-division transformation for the SoVI-like agricultural land variable:

```text
pct_land_farms
````

It corresponds to the original SoVI variable:

```text
PCTFARMS92 -> pct_land_farms
```

The related original SoVI variable `PCTRFRM90`, percent rural farm population, is not cleaned in this section because the available Census of Agriculture land-use table does not contain a rural farm population measure.

## Source data

The main source is Statistics Canada Census of Agriculture table 32-10-0249-01:

```text
Land use, Census of Agriculture, 2021
```

Raw files:

```text
census_division_agriculture_2021/raw/land_use_32100249_2021.csv
census_division_agriculture_2021/raw/land_use_32100249_2021_MetaData.csv
```

The transformation joins this source to the cleaned Québec census-division base frame:

```text
census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.csv
```

The base frame supplies the census-division identifiers and the denominator `land_area_km2`.

## Scripts

Inspection script:

```bash
python census_division_agriculture_2021/inspect_census_division_agriculture_2021.py
```

Cleaning script:

```bash
python census_division_agriculture_2021/clean_census_division_agriculture_2021.py
```

The inspection step was important because the first downloaded agriculture table, `32-10-0232-01`, classified farms by farm-area size classes and did not provide total farm area as an area value. The usable source for `pct_land_farms` is the land-use table, `32-10-0249-01`.

## Variable construction

`pct_land_farms` is computed as:

```text
pct_land_farms =
    100 * total_farm_area_km2 / land_area_km2
```

The numerator is taken from the Census of Agriculture land-use table using:

```text
Land use = Total farm area
Unit of measure = Hectares
```

Hectares are converted to square kilometres using:

```text
total_farm_area_km2 = total_farm_area_hectares * 0.01
```

The cleaner also preserves acres as an audit fallback, but the current successful run used hectares where numeric area values were available.

## Coverage and missingness

The cleaned output contains all 98 Québec census divisions.

`pct_land_farms` is numeric for 94 of 98 census divisions. Four census divisions have positive farm counts but unavailable or suppressed total-farm-area values in both hectares and acres:

```text
Communauté maritime des Îles-de-la-Madeleine
Montréal
Sept-Rivières--Caniapiscau
Minganie--Le Golfe-du-Saint-Laurent
```

These missing values are not interpreted as zero. The source table reports positive numbers of farms for these divisions, but the area values have status `F`. The cleaner therefore leaves `pct_land_farms` missing and records the missingness reason.

This gives the variable partial but documented coverage.

## Outputs

Main clean output:

```text
census_division_agriculture_2021/output/clean_census_division_agriculture_2021.csv
```

Audit outputs:

```text
census_division_agriculture_2021/output/clean_census_division_agriculture_total_farm_area_source_rows_2021.csv
census_division_agriculture_2021/output/clean_census_division_agriculture_missing_area_audit_2021.csv
census_division_agriculture_2021/output/clean_census_division_agriculture_variable_metadata_2021.csv
census_division_agriculture_2021/output/clean_census_division_agriculture_summary_2021.csv
```

## Latest validation

The latest successful run produced:

```text
Rows: 98
Unique census divisions: 98
pct_land_farms non-missing: 94
pct_land_farms missing: 4
pct_land_farms values over 100: 0
Base names with mojibake: 0
Clean names with mojibake: 0
```

Summary statistics for the 94 numeric values:

```text
pct_land_farms:
    min    0.0041
    mean   27.1584
    median 19.8790
    max    96.2205
```

## Interpretation

`pct_land_farms` measures the share of each census division's land area that is reported as total farm area in the 2021 Census of Agriculture.

This is a strong Canadian adaptation of the original SoVI land-in-farms variable, but it is not complete coverage. The four missing values are retained as missing because replacing suppressed or unavailable farm-area values with zero would be methodologically incorrect.

`pct_rural_farm` remains unresolved. A defensible version would require a separate farm-population or Agriculture–Population Linkage source, not a land-use table.