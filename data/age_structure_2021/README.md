# Age Structure 2021

This folder contains the cleaned age-structure feature table derived from the 2021 Statistics Canada Census Profile.

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

| Characteristic ID | Characteristic name                                | Output field           |
| ----------------: | -------------------------------------------------- | ---------------------- |
|               `8` | `Total - Age groups of the population - 100% data` | `age_total_population` |
|               `9` | `0 to 14 years`                                    | `population_0_14`      |
|              `24` | `65 years and over`                                | `population_65_plus`   |

These are 100% Census data variables.

## Clean output

The script creates:

```text
output/clean_census_tract_age_structure_2021.csv
output/clean_census_tract_age_structure_2021.parquet
```

The clean table contains one row per census tract and the following columns:

| Column                 | Description                                              |
| ---------------------- | -------------------------------------------------------- |
| `statcan_dguid`        | Statistics Canada geography identifier used for joins    |
| `geo_name`             | Census tract name/code from the Census Profile           |
| `unit_type`            | Always `census_tract`                                    |
| `census_year`          | Census year, currently `2021`                            |
| `age_total_population` | Total population used as denominator for age groups      |
| `population_0_14`      | Number of people aged 0 to 14                            |
| `pct_age_0_14`         | Share of the age total population aged 0 to 14           |
| `population_65_plus`   | Number of people aged 65 years and over                  |
| `pct_age_65_plus`      | Share of the age total population aged 65 years and over |
| `source_age_structure` | Source description                                       |

Percent columns are stored as proportions between `0` and `1`, not percentages between `0` and `100`.

## Relation to SVI

The original SVI uses:

```text
Percent persons 17 or younger
Percent persons 65 years and over
```

In the Canadian Census Profile, the directly available child-age variable used here is:

```text
0 to 14 years
```

Therefore:

```text
pct_age_0_14
```

is currently treated as the available Canadian proxy for the SVI children/youth variable.

The elderly variable is directly available as:

```text
pct_age_65_plus
```

## Missing values

The script preserves all census tracts.

In the first run, the output contained:

```text
6247 census tracts
81 tracts with suppressed/missing age values marked SYMBOL = x
```

Missing values are kept as missing. They are not dropped or imputed at this stage.

Index-specific scripts can later decide how to handle these cases.

## Role in the pipeline

This folder produces a clean feature table only.

The broader pipeline is:

```text
raw Census Profile
    ↓
clean age-structure feature table
    ↓
master census-tract feature table
    ↓
SVI / SoVI / BRIC / HGNN-specific inputs
```

For the SVI pipeline, the likely selected fields will be:

```text
pct_age_0_14
pct_age_65_plus
```
