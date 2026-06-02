# Census Division Business Establishments 2021

This folder contains the Québec census-division transformations for two SoVI-like variables:

```text
manufacturing_density
commercial_density
```

They correspond to the original SoVI variables:

```text
MAESDEN92  -> manufacturing_density
COMDEVDN92 -> commercial_density
```

## Source data

The source is the Business Counts in Rural and Small Town Canada dashboard extract:

```text
census_division_business_establishments_2021/raw/canada_rural_business_counts_dashboard.csv
```

The transformation joins the business-count data to the cleaned Québec census-division base frame:

```text
census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.csv
```

The base frame supplies the census-division identifiers and the denominator:

```text
land_area_km2
```

A previous Canadian Business Counts CMA/CSD table was inspected but did not provide full Québec census-division coverage after CSD-to-CD aggregation. The dashboard extract was used because it provides complete Québec CD coverage when 2021 census subdivisions are aggregated to census divisions.

## Scripts

Older inspection script for the first Business Counts table:

```bash
python census_division_business_establishments_2021/inspect_census_division_business_establishments_2021.py
```

Dashboard-specific inspection script:

```bash
python census_division_business_establishments_2021/inspect_census_division_business_establishments_business_count_2021.py
```

Cleaning script:

```bash
python census_division_business_establishments_2021/clean_census_division_business_establishments_2021.py
```

The cleaner uses the dashboard extract, filters to Québec, selects total businesses with employees, aggregates 2021 census subdivision counts to census divisions, and divides by census-division land area.

## Temporal choice

The selected source period is:

```text
2022-01
```

The dashboard extract contains semester periods beginning in 2022. The `2022-01` period is used because it is the earliest available period and the closest available business-count period to the 2021 SoVI-like table.

The geography identifiers are 2021 census subdivision DGUIDs, so the spatial frame remains aligned with the 2021 census geography even though the business-count observation period is 2022-01.

## Variable construction

### Manufacturing density

The manufacturing variable uses:

```text
NAICS 31-33 Manufacturing
Employment size: Total, with employees
Time period: 2022-01
```

The cleaned variable is:

```text
manufacturing_density =
    manufacturing_business_count / land_area_km2
```

Its unit is:

```text
businesses with employees per square kilometre
```

### Commercial density

The original SoVI variable `COMDEVDN92` refers to commercial establishments per unit area, but it does not map to a single NAICS sector. This implementation uses a broad Canadian commercial-business proxy composed of:

```text
NAICS 41 Wholesale trade
NAICS 44-45 Retail trade
NAICS 72 Accommodation and food services
```

The cleaned variable is:

```text
commercial_density =
    (
        wholesale_trade_business_count
        + retail_trade_business_count
        + accommodation_food_services_business_count
    ) / land_area_km2
```

The cleaner also retains a narrower trade-only audit variable:

```text
commercial_trade_only_density_per_km2 =
    (
        wholesale_trade_business_count
        + retail_trade_business_count
    ) / land_area_km2
```

## Audit variables

The clean output retains the component counts and densities:

```text
manufacturing_business_count
wholesale_trade_business_count
retail_trade_business_count
accommodation_food_services_business_count
commercial_trade_only_business_count
commercial_trade_accommodation_food_business_count

manufacturing_density_per_km2
wholesale_trade_density_per_km2
retail_trade_density_per_km2
accommodation_food_services_density_per_km2
commercial_trade_only_density_per_km2
commercial_trade_accommodation_food_density_per_km2
```

These audit variables make the commercial-density definition explicit and allow narrower or broader commercial proxies to be compared later.

## Coverage

The cleaned output has full coverage:

```text
98 / 98 Québec census divisions
```

The latest successful run produced:

```text
Rows: 98
Unique census divisions: 98
Selected time period: 2022-01
Selected unique Québec CSD codes: 1051
Selected unique Québec CD codes: 98

Base names with mojibake: 0
Clean names with mojibake: 0
Source REF_AREA with mojibake: 0
Source INDUSTRY with mojibake: 0
```

Formula differences were only floating-point roundoff:

```text
manufacturing_formula_diff: 2.220446049250313e-16
commercial_density_formula_diff: 8.881784197001252e-16
commercial_alias_diff: 0.0
```

Summary statistics from the latest run:

```text
manufacturing_density:
    min    0.000025
    mean   0.246603
    median 0.046135
    max    6.281457

commercial_density:
    min    0.000293
    mean   1.203596
    median 0.129735
    max    33.771363
```

## Outputs

Main clean output:

```text
census_division_business_establishments_2021/output/clean_census_division_business_establishments_2021.csv
```

Audit outputs:

```text
census_division_business_establishments_2021/output/clean_census_division_business_establishments_source_rows_2021.csv
census_division_business_establishments_2021/output/clean_census_division_business_establishments_variable_metadata_2021.csv
census_division_business_establishments_2021/output/clean_census_division_business_establishments_component_audit_2021.csv
census_division_business_establishments_2021/output/clean_census_division_business_establishments_unmatched_audit_2021.csv
census_division_business_establishments_2021/output/clean_census_division_business_establishments_time_period_audit_2021.csv
census_division_business_establishments_2021/output/clean_census_division_business_establishments_summary_2021.csv
```

## Interpretation

`manufacturing_density` is a strong Canadian NAICS proxy for manufacturing-establishment density.

`commercial_density` is a constructed Canadian proxy for commercial-establishment density. It is broader than trade alone because it includes accommodation and food services, which are part of local commercial/service activity. The narrower trade-only version is retained as an audit variable.

Both variables measure business locations with employees per square kilometre, aggregated from census subdivisions to census divisions.