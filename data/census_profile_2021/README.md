# Census Profile 2021 — Census Tract Feature Source

Download: https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/details/download-telecharger/comp/GetFile.cfm?Lang=E&FILETYPE=CSV&GEONO=007

This folder contains the 2021 Statistics Canada Census Profile file for:

```text
Census Metropolitan Areas, Tracted Census Agglomerations and Census Tracts
````

This file is used as the main source for many census-derived features needed for SVI, SoVI, BRIC-like indices, and later model-ready feature tables.

The raw Census Profile file is in **long format**:

```text
one row = one characteristic for one geography
```

Important columns include:

| Column                | Meaning                                                |
| --------------------- | ------------------------------------------------------ |
| `DGUID`               | Statistics Canada geography identifier, used for joins |
| `GEO_LEVEL`           | Geographic level, e.g. `Census tract`                  |
| `GEO_NAME`            | Geography name/code                                    |
| `CHARACTERISTIC_ID`   | Numeric identifier for a Census Profile characteristic |
| `CHARACTERISTIC_NAME` | Human-readable characteristic name                     |
| `C1_COUNT_TOTAL`      | Main count/value for total population                  |
| `C10_RATE_TOTAL`      | Main rate/percentage value, when provided              |
| `SYMBOL`              | Suppression, missing-data, or special-value marker     |

## Encoding note

Statistics Canada CSV files may use ISO-8859-1 encoding. When loading raw Census Profile CSV files with pandas, use:

```python
pd.read_csv(path, encoding="iso-8859-1", low_memory=False)
```

## Confirmed characteristic IDs

### Population

The population variable was extracted from:

| Characteristic ID | Characteristic name | Output field       |
| ----------------: | ------------------- | ------------------ |
|               `1` | `Population, 2021`  | `population_total` |

The clean population table is saved as:

```text
output/clean_census_tract_population_2021.csv
output/clean_census_tract_population_2021.parquet
```

### Low-income candidates

The following low-income-related characteristics were found in the Census Profile file:

| Characteristic ID | Characteristic name                                                                                                                        | Proposed role                |
| ----------------: | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------- |
|             `335` | `Total - LIM low-income status in 2020 for the population in private households - 100% data`                                               | LIM-AT denominator           |
|             `340` | `In low income based on the Low-income measure, after tax (LIM-AT)`                                                                        | LIM-AT numerator             |
|             `345` | `Prevalence of low income based on the Low-income measure, after tax (LIM-AT) (%)`                                                         | Published LIM-AT percentage  |
|             `350` | `Total - LICO low-income status in 2020 for the population in private households to whom the low-income concept is applicable - 100% data` | LICO-AT denominator          |
|             `355` | `In low income based on the Low-income cut-offs, after tax (LICO-AT)`                                                                      | LICO-AT numerator            |
|             `360` | `Prevalence of low income based on the Low-income cut-offs, after tax (LICO-AT) (%)`                                                       | Published LICO-AT percentage |

For the first Canadian/Québec SVI-style feature layer, the preferred low-income measure is:

```text
LIM-AT
```

The planned clean feature fields are:

```text
statcan_dguid
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

The SVI-specific implementation can later select:

```text
pct_low_income_lim_at
```

as the default `pct_low_income` variable.

## Important note

This folder does not directly compute the SVI.

Its role is to extract and clean reusable Census Profile features. Index-specific scripts will later select the relevant columns and apply their own normalization, missing-data, and aggregation rules.

````

This keeps the logic clean:

```text
census_profile_2021/
    documents what Census Profile variables exist

spatial_frame_population_2021/
    documents the joined geographic base layer

future SVI folder/
    documents how we select and transform variables for SVI
````
