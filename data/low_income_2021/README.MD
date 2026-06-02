# Low Income 2021

This folder contains the cleaned low-income feature table for Québec census tracts, derived from the 2021 Statistics Canada Census Profile.

The cleaned table provides low-income counts and low-income rates using two after-tax concepts:

```text
LIM-AT: Low-income measure, after tax
LICO-AT: Low-income cut-offs, after tax
````

The main SVI-relevant variable is likely:

```text
pct_low_income_lim_at
```

because SVI needs a poverty / low-income prevalence variable.

## Current status

This section is complete for census-tract-level low-income variables.

The cleaner produces one row per census tract and includes:

```text
LIM-AT denominator
LIM-AT low-income population count
computed LIM-AT low-income proportion
published LIM-AT low-income percentage

LICO-AT denominator
LICO-AT low-income population count
computed LICO-AT low-income proportion
published LICO-AT low-income percentage
```

The output files are:

```text
low_income_2021/output/clean_census_tract_low_income_2021.csv
low_income_2021/output/clean_census_tract_low_income_2021.parquet
```

## Source data

The source file is:

```text
census_profile_2021/98-401-X2021007_English_CSV_data.csv
```

The source is described in the cleaner as:

```text
Statistics Canada Census Profile, 2021 Census of Population; low-income status in 2020
```

The income reference year is:

```text
2020
```

The census year is:

```text
2021
```

## Geographic filter

The cleaner keeps only:

```text
GEO_LEVEL == "Census tract"
```

The output geography unit is:

```text
unit_type = census_tract
```

The primary join key is:

```text
statcan_dguid
```

## Selected Census Profile characteristics

The cleaner extracts six `CHARACTERISTIC_ID` values from the Census Profile: 

```text
335 → lim_at_denominator
340 → lim_at_low_income_population
345 → published_pct_low_income_lim_at
350 → lico_at_denominator
355 → lico_at_low_income_population
360 → published_pct_low_income_lico_at
```

These correspond to low-income status in 2020 using LIM-AT and LICO-AT concepts.

## Value logic

For count variables, the cleaner uses:

```text
C1_COUNT_TOTAL
```

For published percentage variables, the cleaner uses:

```text
C10_RATE_TOTAL
```

The published percentage columns are:

```text
published_pct_low_income_lim_at
published_pct_low_income_lico_at
```

These are percentages on a 0–100 scale.

The cleaner also computes its own proportions:

```text
pct_low_income_lim_at = lim_at_low_income_population / lim_at_denominator
pct_low_income_lico_at = lico_at_low_income_population / lico_at_denominator
```

These computed columns are proportions on a 0–1 scale. 

For example, the first CSV rows show values such as:

```text
pct_low_income_lim_at = 0.234756...
published_pct_low_income_lim_at = 23.5
pct_low_income_lico_at = 0.060975...
published_pct_low_income_lico_at = 6.0
```

So the computed fields and published fields represent the same idea but on different scales. 

## Output columns

The cleaned table contains:

```text
statcan_dguid
geo_name
unit_type
census_year
income_reference_year
lim_at_denominator
lim_at_low_income_population
pct_low_income_lim_at
published_pct_low_income_lim_at
lico_at_denominator
lico_at_low_income_population
pct_low_income_lico_at
published_pct_low_income_lico_at
source_low_income
```

## Main feature columns

Recommended SVI input feature:

```text
pct_low_income_lim_at
```

Alternative / supplementary low-income feature:

```text
pct_low_income_lico_at
```

Published percentage equivalents:

```text
published_pct_low_income_lim_at
published_pct_low_income_lico_at
```

Use the `pct_*` computed columns when building indices that expect proportions between 0 and 1.

Use the `published_pct_*` columns only when a 0–100 percentage scale is desired.

## Interpretation

### LIM-AT

```text
pct_low_income_lim_at
```

is the proportion of people in the census tract who are in low income based on the Low-income measure, after tax.

### LICO-AT

```text
pct_low_income_lico_at
```

is the proportion of people in the census tract who are in low income based on the Low-income cut-offs, after tax.

Both are after-tax low-income measures for the 2020 income reference year.

## Role in the SVI pipeline

The original SVI socioeconomic-status domain includes:

```text
Percent individuals below poverty
```

For a Canadian / Québec adaptation, this folder provides the closest Census Profile equivalent:

```text
pct_low_income_lim_at
```

Recommended mapping:

```text
SVI poverty / low-income variable
→ pct_low_income_lim_at
```

Possible alternative sensitivity variable:

```text
pct_low_income_lico_at
```

For SVI scoring, this variable has the direction:

```text
higher value = higher vulnerability
```

So no reversal is needed before percentile ranking.

## Validation performed by cleaner

The cleaner checks that:

```text
statcan_dguid is unique
```

It also reports:

```text
number of clean rows
missing values by column
summary statistics for pct_low_income_lim_at
summary statistics for pct_low_income_lico_at
preview of cleaned rows
```

## Notes on missing values

The cleaner reports missing or non-numeric low-income values after extracting the six selected characteristics. It also prints missing values by characteristic and `SYMBOL` counts when missing values are present. 

Downstream SVI construction should decide how to handle tracts with missing values, especially because SVI percentile ranks require comparable values across all included tracts.

## Recommended downstream use

For the SVI input table, join this file using:

```text
statcan_dguid
```

Recommended selected columns:

```text
statcan_dguid
pct_low_income_lim_at
lim_at_low_income_population
lim_at_denominator
source_low_income
```

Optional sensitivity columns:

```text
pct_low_income_lico_at
lico_at_low_income_population
lico_at_denominator
published_pct_low_income_lim_at
published_pct_low_income_lico_at
```

## Remaining work

This section is complete.

Future work will happen downstream in:

```text
svi_2021/
```

where `pct_low_income_lim_at` should be joined with the other SVI variables into a single census-tract-level SVI input table.
