# Unemployment 2021

This folder contains the cleaned unemployment and labour-force feature table derived from the 2021 Statistics Canada Census Profile.

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

The script extracts the following Census Profile characteristics:

| Characteristic ID | Characteristic name                                                                  | Output field                             |
| ----------------: | ------------------------------------------------------------------------------------ | ---------------------------------------- |
|            `2223` | `Total - Population aged 15 years and over by labour force status - 25% sample data` | `labour_force_status_denominator_15plus` |
|            `2224` | `In the labour force`                                                                | `labour_force_population`                |
|            `2225` | `Employed`                                                                           | `employed_population`                    |
|            `2226` | `Unemployed`                                                                         | `unemployed_population`                  |
|            `2227` | `Not in the labour force`                                                            | `not_in_labour_force_population`         |
|            `2228` | `Participation rate`                                                                 | `published_participation_rate`           |
|            `2229` | `Employment rate`                                                                    | `published_employment_rate`              |
|            `2230` | `Unemployment rate`                                                                  | `published_unemployment_rate`            |

These variables come from the 25% sample data section of the Census Profile.

## Clean output

The script creates:

```text
output/clean_census_tract_unemployment_2021.csv
output/clean_census_tract_unemployment_2021.parquet
```

The clean table contains one row per census tract and the following columns:

| Column                                   | Description                                                              |
| ---------------------------------------- | ------------------------------------------------------------------------ |
| `statcan_dguid`                          | Statistics Canada geography identifier used for joins                    |
| `geo_name`                               | Census tract name/code from the Census Profile                           |
| `unit_type`                              | Always `census_tract`                                                    |
| `census_year`                            | Census year, currently `2021`                                            |
| `labour_force_status_denominator_15plus` | Population aged 15+ used as the labour-force-status denominator          |
| `labour_force_population`                | Number of people aged 15+ in the labour force                            |
| `employed_population`                    | Number of employed people aged 15+                                       |
| `unemployed_population`                  | Number of unemployed people aged 15+                                     |
| `not_in_labour_force_population`         | Number of people aged 15+ not in the labour force                        |
| `pct_unemployed`                         | Computed unemployment proportion: unemployed / labour force              |
| `published_unemployment_rate`            | Statistics Canada published unemployment rate, percentage from 0 to 100  |
| `pct_employed`                           | Computed employment proportion: employed / population aged 15+           |
| `published_employment_rate`              | Statistics Canada published employment rate, percentage from 0 to 100    |
| `pct_in_labour_force`                    | Computed labour-force participation proportion                           |
| `published_participation_rate`           | Statistics Canada published participation rate, percentage from 0 to 100 |
| `pct_not_in_labour_force`                | Computed share of people aged 15+ not in the labour force                |
| `source_unemployment`                    | Source description                                                       |

Computed `pct_*` columns are stored as proportions between `0` and `1`.

Published rate columns from Statistics Canada are stored as percentages between `0` and `100`.

## Relation to SVI

The original SVI uses:

```text
Percent civilian unemployed
```

For the Canadian Census Profile version, the closest clean variable is:

```text
pct_unemployed = unemployed_population / labour_force_population
```

The script also keeps:

```text
published_unemployment_rate
```

as a validation field. It should approximately equal:

```text
pct_unemployed * 100
```

## Missing values

The script preserves all census tracts.

In the first run, the output contained:

```text
6247 census tracts
89 tracts with suppressed/missing labour-force values marked SYMBOL = x
```

Missing values are kept as missing. They are not dropped or imputed at this stage.

Index-specific scripts can later decide how to handle these cases.

## Role in the pipeline

This folder produces a clean feature table only.

The broader pipeline is:

```text
raw Census Profile
    ↓
clean unemployment / labour-force feature table
    ↓
master census-tract feature table
    ↓
SVI / SoVI / BRIC / HGNN-specific inputs
```

For the SVI pipeline, the likely selected field will be:

```text
pct_unemployed
```
