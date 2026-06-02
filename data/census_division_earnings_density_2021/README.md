# Census Division Earnings Density 2021

This folder contains the Québec census-division transformation for the SoVI-like variable:

```text
earnings_density
```

It corresponds to the original SoVI variable:

```text
EARNDEN90 -> earnings_density
```

## Source data

The source is the 2021 Census Profile for Canadian census divisions:

```text
census_profile_census_division_2021/raw/98-401-X2021004_English_CSV_data.csv
```

The transformation joins Census Profile rows to the cleaned Québec census-division base frame:

```text
census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.csv
```

The base frame supplies the census-division identifiers and the land-area denominator:

```text
land_area_km2
```

## Scripts

Inspection script:

```bash
python census_division_earnings_density_2021/inspect_census_division_earnings_density_2021.py
```

Cleaning script:

```bash
python census_division_earnings_density_2021/clean_census_division_earnings_density_2021.py
```

The inspection did not find a direct aggregate employment-income row in the Census Profile file. It therefore evaluated derived candidates using recipient counts and average income values. The cleaner uses forced characteristic IDs for the selected derived employment-income proxy.

## Variable construction

The original SoVI variable is earnings in `$1,000` per square mile. This implementation constructs a Canadian census-division proxy by estimating aggregate employment income from:

```text
number of employment income recipients
*
average employment income among recipients
```

The main 2020 source rows are:

```text
CHARACTERISTIC_ID = 133
Number of employment income recipients aged 15 years and over in private households in 2020 - 25% sample data

CHARACTERISTIC_ID = 134
Average employment income in 2020 among recipients ($)
```

The estimated aggregate numerator is:

```text
estimated_aggregate_employment_income_2020 =
    employment_income_recipients_2020_25pct_count
    *
    average_employment_income_2020_25pct_dollars
```

The main cleaned variable is:

```text
earnings_density =
    estimated_aggregate_employment_income_2020 / land_area_km2
```

Its unit is:

```text
dollars per square kilometre
```

The cleaner also retains the original-style SoVI scaling:

```text
earnings_density_thousands_per_square_mile =
    (estimated_aggregate_employment_income_2020 / 1000)
    / land_area_square_miles
```

Because the original SoVI variable used `$1,000 per square mile`, this audit variable is retained for interpretability. The main SoVI input uses `earnings_density`; the two versions are related by a constant scaling transformation.

## Audit variables

The cleaner retains the analogous 2019 pair as a sensitivity check:

```text
CHARACTERISTIC_ID = 224
Number of employment income recipients aged 15 years and over in private households in 2019 - 25% sample data

CHARACTERISTIC_ID = 225
Average employment income in 2019 among recipients ($)
```

Important audit variables include:

```text
estimated_aggregate_employment_income_2020
estimated_aggregate_employment_income_2020_per_capita
earnings_density_thousands_per_square_mile
earnings_density_thousands_per_km2
earnings_density_2019
earnings_density_2019_thousands_per_square_mile
earnings_density_2020_minus_2019
```

## Coverage

The cleaned output has full coverage:

```text
98 / 98 Québec census divisions
```

The latest successful run produced:

```text
Rows: 98
Unique census divisions: 98
Variables cleaned: earnings_density
aggregate 2020 formula max abs difference: 0
aggregate 2019 formula max abs difference: 0
earnings_density formula max abs difference: 1.862645149230957e-09
original-style formula max abs difference: 1.4551915228366852e-11
Base names with mojibake: 0
Clean names with mojibake: 0
Raw characteristic names with mojibake: 0
```

Summary statistics from the latest run:

```text
earnings_density:
    min    1,554.7729
    mean   4,570,097.0759
    median 473,954.1221
    max    113,027,064.0508

earnings_density_thousands_per_square_mile:
    min    4.0268
    mean   11,836.4971
    median 1,227.5355
    max    292,738.7520

estimated_aggregate_employment_income_2020:
    min    155,903,200
    mean   2,300,367,449.49
    median 849,341,500
    max    56,320,481,800
```

## Outputs

Main clean output:

```text
census_division_earnings_density_2021/output/clean_census_division_earnings_density_2021.csv
```

Audit outputs:

```text
census_division_earnings_density_2021/output/clean_census_division_earnings_density_source_rows_2021.csv
census_division_earnings_density_2021/output/clean_census_division_earnings_density_variable_metadata_2021.csv
census_division_earnings_density_2021/output/clean_census_division_earnings_density_summary_2021.csv
```