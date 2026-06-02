# Census Division Social Security Recipients 2021

This folder contains the Québec census-division transformation for the SoVI-like variable:

```text
social_security_recipients_per_capita
````

It corresponds to the original SoVI variable:

```text
SSBENPC90 -> social_security_recipients_per_capita
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

The base frame supplies the census-division identifiers and the denominator:

```text
population_total_2021
```

## Scripts

Inspection script:

```bash
python census_division_social_security_recipients_2021/inspect_census_division_social_security_recipients_2021.py
```

Cleaning script:

```bash
python census_division_social_security_recipients_2021/clean_census_division_social_security_recipients_2021.py
```

The cleaner uses strict full-file encoding detection before loading the raw Census Profile. This avoids the partial-read encoding problem where early rows may decode as UTF-8 while later StatCan rows require `cp1252`.

## Variable construction

The original SoVI variable measures per-capita Social Security recipients in the United States. The Canadian Census Profile does not provide a direct equivalent at census-division level. This implementation therefore uses a broad Canadian government-transfer-recipient proxy.

The main variable is built from:

```text
CHARACTERISTIC_ID = 213
CHARACTERISTIC_NAME = Number of government transfers recipients aged 15 years and over in private households in 2019 - 100% data
SOURCE COLUMN = C1_COUNT_TOTAL
```

The cleaned variable is:

```text
social_security_recipients_per_capita =
    government_transfer_recipients_2019_100pct_count / population_total_2021
```

A scaled audit version is also retained:

```text
social_security_recipients_per_1000 =
    1000 * social_security_recipients_per_capita
```

The 2019 row is used as the default because it is less affected by COVID-era emergency and recovery benefit uptake than the 2020 transfer-recipient rows. The 2020 rows are still retained as audit and sensitivity variables.

## Audit variables

The cleaner retains the following audit variables:

```text
government_transfer_recipients_2019_100pct_count
government_transfer_recipients_2020_100pct_count
government_transfer_recipients_2019_25pct_count
government_transfer_recipients_2020_25pct_count
employment_insurance_recipients_2020_100pct_count
covid_benefit_recipients_2020_100pct_count
government_transfer_recipients_2020_100pct_per_capita
government_transfer_recipients_2019_25pct_per_capita
government_transfer_recipients_2020_25pct_per_capita
employment_insurance_recipients_2020_100pct_per_capita
covid_benefit_recipients_2020_100pct_per_capita
government_transfer_recipients_2020_minus_2019_per_capita
```

These variables document the sensitivity of the chosen proxy to sample definition, year, and pandemic-era transfer programs.

## Coverage

The cleaned output has full coverage:

```text
98 / 98 Québec census divisions
```

The latest successful run produced:

```text
Rows: 98
Unique census divisions: 98
social_security_recipients_per_capita formula max abs difference: 0.0
Base names with mojibake: 0
Clean names with mojibake: 0
Raw characteristic names with mojibake: 0
```

Summary statistics from the latest run:

```text
social_security_recipients_per_capita:
    min    0.4836
    mean   0.6218
    median 0.6222
    max    0.7634

social_security_recipients_per_1000:
    min    483.55
    mean   621.77
    median 622.16
    max    763.40
```

## Outputs

Main clean output:

```text
census_division_social_security_recipients_2021/output/clean_census_division_social_security_recipients_2021.csv
```

Audit outputs:

```text
census_division_social_security_recipients_2021/output/clean_census_division_social_security_recipients_source_rows_2021.csv
census_division_social_security_recipients_2021/output/clean_census_division_social_security_recipients_variable_metadata_2021.csv
census_division_social_security_recipients_2021/output/clean_census_division_social_security_recipients_summary_2021.csv
```

## Interpretation

`social_security_recipients_per_capita` should be interpreted as a broad Canadian government-transfer-recipient proxy, not as a direct U.S. Social Security equivalent.

The variable captures the prevalence of people aged 15 years and over in private households who received government transfers in 2019, divided by total 2021 census-division population. This is a reasonable Canadian adaptation of the original SoVI income-transfer-dependence concept, but it is broader than old-age public pension receipt alone.
