# Education 2021

This folder contains the cleaned education feature table derived from the 2021 Statistics Canada Census Profile.

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

The script extracts selected education characteristics from the 25% sample data section of the Census Profile.

### Population aged 15 years and over

| Characteristic ID | Characteristic name                                                                                                                | Output field                       |
| ----------------: | ---------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
|            `1998` | `Total - Highest certificate, diploma or degree for the population aged 15 years and over in private households - 25% sample data` | `education_total_15plus`           |
|            `1999` | `No certificate, diploma or degree`                                                                                                | `no_certificate_15plus`            |
|            `2000` | `High (secondary) school diploma or equivalency certificate`                                                                       | `high_school_certificate_15plus`   |
|            `2001` | `Postsecondary certificate, diploma or degree`                                                                                     | `postsecondary_certificate_15plus` |

### Population aged 25 to 64 years

| Characteristic ID | Characteristic name                                                                                                             | Output field                      |
| ----------------: | ------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
|            `2014` | `Total - Highest certificate, diploma or degree for the population aged 25 to 64 years in private households - 25% sample data` | `education_total_25_64`           |
|            `2015` | `No certificate, diploma or degree`                                                                                             | `no_certificate_25_64`            |
|            `2016` | `High (secondary) school diploma or equivalency certificate`                                                                    | `high_school_certificate_25_64`   |
|            `2017` | `Postsecondary certificate, diploma or degree`                                                                                  | `postsecondary_certificate_25_64` |

## Clean output

The script creates:

```text
output/clean_census_tract_education_2021.csv
output/clean_census_tract_education_2021.parquet
```

The clean table contains one row per census tract and the following main fields:

| Column                             | Description                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------ |
| `statcan_dguid`                    | Statistics Canada geography identifier used for joins                          |
| `geo_name`                         | Census tract name/code from the Census Profile                                 |
| `unit_type`                        | Always `census_tract`                                                          |
| `census_year`                      | Census year, currently `2021`                                                  |
| `education_total_15plus`           | Population aged 15+ in private households used as education denominator        |
| `no_certificate_15plus`            | Population aged 15+ with no certificate, diploma, or degree                    |
| `high_school_certificate_15plus`   | Population aged 15+ with high school diploma/equivalency as highest credential |
| `postsecondary_certificate_15plus` | Population aged 15+ with postsecondary certificate, diploma, or degree         |
| `pct_no_certificate_15plus`        | Share of population aged 15+ with no certificate, diploma, or degree           |
| `education_total_25_64`            | Population aged 25 to 64 in private households used as education denominator   |
| `no_certificate_25_64`             | Population aged 25 to 64 with no certificate, diploma, or degree               |
| `pct_no_certificate_25_64`         | Share of population aged 25 to 64 with no certificate, diploma, or degree      |
| `education_measure_default`        | Default education proxy for SVI-style construction                             |
| `source_education`                 | Source description                                                             |

Computed `pct_*` columns are stored as proportions between `0` and `1`.

## Default education measure

The script defines:

```text
education_measure_default = pct_no_certificate_15plus
```

This is the current default Canadian proxy for the SVI education variable.

The original U.S. SVI uses:

```text
percent persons with no high school diploma
```

The closest available Canadian Census Profile measure used here is:

```text
No certificate, diploma or degree
```

The script also keeps:

```text
pct_no_certificate_25_64
```

as an alternative/sensitivity variant.

## Missing values

The script preserves all census tracts.

In the first run, the output contained:

```text
6247 census tracts
89 tracts with suppressed/missing education values marked SYMBOL = x
```

Missing values are kept as missing. They are not dropped or imputed at this stage.

Index-specific scripts can later decide how to handle these cases.

## Role in the pipeline

This folder produces a clean feature table only.

The broader pipeline is:

```text
raw Census Profile
    ↓
clean education feature table
    ↓
master census-tract feature table
    ↓
SVI / SoVI / BRIC / HGNN-specific inputs
```

For the SVI pipeline, the likely selected field will be:

```text
education_measure_default
```

or, depending on methodological choice:

```text
pct_no_certificate_25_64
```