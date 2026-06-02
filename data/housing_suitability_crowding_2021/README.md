# Housing Suitability and Crowding 2021

This folder contains the cleaned housing suitability and crowding feature table derived from the 2021 Statistics Canada Census Profile.

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

The script extracts selected housing suitability and crowding characteristics from the 25% sample data section of the Census Profile.

### Persons per room

| Characteristic ID | Characteristic name                                                          | Output field                               |
| ----------------: | ---------------------------------------------------------------------------- | ------------------------------------------ |
|            `1434` | `Total - Private households by number of persons per room - 25% sample data` | `persons_per_room_total_households`        |
|            `1435` | `One person or fewer per room`                                               | `one_person_or_fewer_per_room_households`  |
|            `1436` | `More than one person per room`                                              | `more_than_one_person_per_room_households` |

### Housing suitability

| Characteristic ID | Characteristic name                                                   | Output field                           |
| ----------------: | --------------------------------------------------------------------- | -------------------------------------- |
|            `1437` | `Total - Private households by housing suitability - 25% sample data` | `housing_suitability_total_households` |
|            `1438` | `Suitable`                                                            | `suitable_housing_households`          |
|            `1439` | `Not suitable`                                                        | `not_suitable_housing_households`      |

## Clean output

The script creates:

```text
output/clean_census_tract_housing_suitability_crowding_2021.csv
output/clean_census_tract_housing_suitability_crowding_2021.parquet
```

The clean table contains one row per census tract.

In the validated run, the output contained:

```text
6247 census tracts
```

## Derived fields

The script computes the following reusable proportions:

| Column                              | Description                                                    |
| ----------------------------------- | -------------------------------------------------------------- |
| `pct_one_person_or_fewer_per_room`  | Share of private households with one person or fewer per room  |
| `pct_more_than_one_person_per_room` | Share of private households with more than one person per room |
| `pct_suitable_housing`              | Share of private households classified as suitable housing     |
| `pct_not_suitable_housing`          | Share of private households classified as not suitable housing |

Computed `pct_*` columns are stored as proportions between `0` and `1`.

The script uses safe bounded division. If a denominator is zero or missing, the derived value is kept as missing. Derived proportions are also clipped to the valid range:

```text
0 <= proportion <= 1
```

Raw counts are preserved unchanged.

## Default SVI-style crowding measure

The script defines:

```text
crowding_measure_default = pct_more_than_one_person_per_room
```

This is the current Canadian Census proxy for the SVI crowding variable.

The script also defines:

```text
housing_suitability_measure_default = pct_not_suitable_housing
```

This is a Canadian housing-stress measure based on whether the dwelling has enough bedrooms for the size and composition of the household.

## Validation notes

The validated run confirmed that the table was created successfully with:

```text
6247 census tracts
```

The default crowding measure had:

```text
count    6158
mean     0.027036
min      0.000000
max      0.540541
```

The housing suitability stress measure had:

```text
count    6158
mean     0.060335
min      0.000000
max      0.675676
```

## Missing values

The script preserves all census tracts.

In the validated run, each raw selected variable had:

```text
89 missing values
```

All missing values in the selected block were marked with:

```text
SYMBOL = x
```

This means 89 census tracts had the whole 25% sample housing suitability / crowding block suppressed or unavailable.

Missing values are kept as missing. They are not dropped or imputed at this stage.

## Role in the pipeline

This folder produces a clean feature table only.

The broader pipeline is:

```text
raw Census Profile
    ↓
clean housing suitability / crowding feature table
    ↓
master census-tract feature table
    ↓
SVI / SoVI / BRIC / HGNN-specific inputs
```

For the SVI pipeline, the likely selected field will be:

```text
crowding_measure_default
```

For SoVI and HGNN, useful additional fields include:

```text
pct_more_than_one_person_per_room
pct_not_suitable_housing
pct_suitable_housing
```

```

This matches the validated run: 6,247 census tracts preserved, 89 suppressed rows kept as missing, and valid numeric summaries for both the crowding and housing-suitability stress measures.