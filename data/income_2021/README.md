# Income 2021

This folder contains the cleaned income feature table derived from the 2021 Statistics Canada Census Profile.

The table is designed as a reusable feature source for SVI, SoVI, BRIC-like indices, HGNN node features, and other model-ready datasets. It does **not** compute any index directly.

## Source data

Original source file:

```text
data/census_profile_2021/98-401-X2021007_English_CSV_data.csv
````

Source product:

```text
Statistics Canada
Census Profile, 2021 Census of Population
Census Metropolitan Areas, Tracted Census Agglomerations and Census Tracts
```

Geographic level used:

```text
GEO_LEVEL == "Census tract"
```

Encoding note:

```python
pd.read_csv(path, encoding="iso-8859-1", low_memory=False)
```

## Extracted characteristics

The script extracts selected 2020 income characteristics for census tracts.

### Individual income, 100% data

| Characteristic ID | Characteristic name                                                                                             | Output field                                                 |
| ----------------: | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
|             `111` | `Total - Income statistics in 2020 for the population aged 15 years and over in private households - 100% data` | `individual_income_stats_total_15plus_100pct_denominator`    |
|             `112` | `Number of total income recipients aged 15 years and over in private households in 2020 - 100% data`            | `individual_total_income_recipients_15plus_2020_100pct`      |
|             `113` | `Median total income in 2020 among recipients ($)`                                                              | `median_total_income_15plus_2020`                            |
|             `114` | `Number of after-tax income recipients aged 15 years and over in private households in 2020 - 100% data`        | `individual_after_tax_income_recipients_15plus_2020_100pct`  |
|             `115` | `Median after-tax income in 2020 among recipients ($)`                                                          | `median_after_tax_income_15plus_2020`                        |
|             `116` | `Number of market income recipients aged 15 years and over in private households in 2020 - 100% data`           | `individual_market_income_recipients_15plus_2020_100pct`     |
|             `117` | `Median market income in 2020 among recipients ($)`                                                             | `median_market_income_15plus_2020`                           |
|             `118` | `Number of employment income recipients aged 15 years and over in private households in 2020 - 100% data`       | `individual_employment_income_recipients_15plus_2020_100pct` |
|             `119` | `Median employment income in 2020 among recipients ($)`                                                         | `median_employment_income_15plus_2020`                       |

### Individual income, 25% sample data

| Characteristic ID | Characteristic name                                                                                                   | Output field                                                |
| ----------------: | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
|             `126` | `Total - Income statistics in 2020 for the population aged 15 years and over in private households - 25% sample data` | `individual_income_stats_total_15plus_25pct_denominator`    |
|             `127` | `Number of total income recipients aged 15 years and over in private households in 2020 - 25% sample data`            | `individual_total_income_recipients_15plus_2020_25pct`      |
|             `128` | `Average total income in 2020 among recipients ($)`                                                                   | `average_total_income_15plus_2020`                          |
|             `129` | `Number of after-tax income recipients aged 15 years and over in private households in 2020 - 25% sample data`        | `individual_after_tax_income_recipients_15plus_2020_25pct`  |
|             `130` | `Average after-tax income in 2020 among recipients ($)`                                                               | `average_after_tax_income_15plus_2020`                      |
|             `131` | `Number of market income recipients aged 15 years and over in private households in 2020 - 25% sample data`           | `individual_market_income_recipients_15plus_2020_25pct`     |
|             `132` | `Average market income in 2020 among recipients ($)`                                                                  | `average_market_income_15plus_2020`                         |
|             `133` | `Number of employment income recipients aged 15 years and over in private households in 2020 - 25% sample data`       | `individual_employment_income_recipients_15plus_2020_25pct` |
|             `134` | `Average employment income in 2020 among recipients ($)`                                                              | `average_employment_income_15plus_2020`                     |

### Household income

| Characteristic ID | Characteristic name                                                  | Output field                                           |
| ----------------: | -------------------------------------------------------------------- | ------------------------------------------------------ |
|             `242` | `Total - Income statistics for private households - 100% data`       | `household_income_stats_total_2020_100pct_denominator` |
|             `243` | `Median total income of household in 2020 ($)`                       | `median_household_total_income_2020`                   |
|             `244` | `Median after-tax income of household in 2020 ($)`                   | `median_household_after_tax_income_2020`               |
|             `251` | `Total - Income statistics for private households - 25% sample data` | `household_income_stats_total_2020_25pct_denominator`  |
|             `252` | `Average total income of household in 2020 ($)`                      | `average_household_total_income_2020`                  |
|             `253` | `Average after-tax income of household in 2020 ($)`                  | `average_household_after_tax_income_2020`              |

## Clean output

The script creates:

```text
output/clean_census_tract_income_2021.csv
output/clean_census_tract_income_2021.parquet
```

The clean table contains one row per census tract.

## Default income measure

The script defines:

```text
income_measure_default = median_after_tax_income_15plus_2020
```

This is the current default income proxy for later SVI-style construction.

The original U.S. SVI uses per-capita income. The Canadian Census Profile table used here does not directly provide a per-capita income field in the selected tract-level profile rows, so the default Canadian proxy is:

```text
median_after_tax_income_15plus_2020
```

For SVI-style scoring, this variable must be interpreted in the reverse vulnerability direction:

```text
lower income = higher vulnerability
higher income = lower vulnerability
```

## Missing values

The script preserves all census tracts.

In the first run, the output contained:

```text
6247 census tracts
147 missing values for the default income measure
```

Most missing values are Statistics Canada suppressed or unavailable values marked with:

```text
SYMBOL = x
```

There was also one `...` symbol in the selected income block.

Missing values are kept as missing. They are not dropped or imputed at this stage.

Index-specific scripts can later decide how to handle these cases.

## Role in the pipeline

This folder produces a clean feature table only.

The broader pipeline is:

```text
raw Census Profile
    ↓
clean income feature table
    ↓
master census-tract feature table
    ↓
SVI / SoVI / BRIC / HGNN-specific inputs
```

For the SVI pipeline, the likely selected field will be:

```text
income_measure_default
```

with reverse ranking during SVI normalization.