# Household and Family 2021

This folder contains the cleaned household and family composition feature table derived from the 2021 Statistics Canada Census Profile.

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

The script extracts selected household, family, living-arrangement, and one-parent economic-family characteristics from the Census Profile.

### Household size

| Characteristic ID | Characteristic name                                        | Output field                       |
| ----------------: | ---------------------------------------------------------- | ---------------------------------- |
|              `50` | `Total - Private households by household size - 100% data` | `private_households_total_by_size` |
|              `56` | `Number of persons in private households`                  | `persons_in_private_households`    |

### Census families

| Characteristic ID | Characteristic name                                                        | Output field                                        |
| ----------------: | -------------------------------------------------------------------------- | --------------------------------------------------- |
|              `71` | `Total - Census families in private households by family size - 100% data` | `census_families_total_by_family_size`              |
|              `76` | `Average size of census families`                                          | `average_size_census_families`                      |
|              `77` | `Average number of children in census families with children`              | `average_children_in_census_families_with_children` |
|              `78` | `Total number of census families in private households - 100% data`        | `census_families_total`                             |
|              `79` | `Total couple families`                                                    | `couple_families_total`                             |
|              `86` | `Total one-parent families`                                                | `one_parent_families_total`                         |

### Persons in families and living arrangements

| Characteristic ID | Characteristic name                                                        | Output field                       |
| ----------------: | -------------------------------------------------------------------------- | ---------------------------------- |
|              `89` | `Total - Persons in private households - 100% data`                        | `persons_private_households_total` |
|              `90` | `Total - Persons in census families`                                       | `persons_in_census_families`       |
|              `92` | `Parents in one-parent families`                                           | `parents_in_one_parent_families`   |
|              `95` | `In a one-parent family`                                                   | `persons_in_one_parent_family`     |
|              `96` | `Total - Persons not in census families in private households - 100% data` | `persons_not_in_census_families`   |
|              `97` | `Living alone`                                                             | `persons_living_alone`             |

### Household type

| Characteristic ID | Characteristic name                  | Output field                        |
| ----------------: | ------------------------------------ | ----------------------------------- |
|             `100` | `Total - Household type - 100% data` | `household_type_total`              |
|             `105` | `One-parent-family households`       | `one_parent_family_households`      |
|             `106` | `Multigenerational households`       | `multigenerational_households`      |
|             `107` | `Multiple-census-family households`  | `multiple_census_family_households` |
|             `110` | `One-person households`              | `one_person_households`             |

### One-parent economic-family income

| Characteristic ID | Characteristic name                                                                                  | Output field                                                   |
| ----------------: | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
|             `313` | `Total - Income statistics for one-parent economic families in private households - 100% data`       | `one_parent_economic_families_income_total_100pct_denominator` |
|             `314` | `Median total income of one-parent economic families in 2020 ($)`                                    | `median_total_income_one_parent_economic_families_2020`        |
|             `315` | `Median after-tax income of one-parent economic families in 2020 ($)`                                | `median_after_tax_income_one_parent_economic_families_2020`    |
|             `316` | `Average family size of one-parent economic families`                                                | `average_size_one_parent_economic_families`                    |
|             `329` | `Total - Income statistics for one-parent economic families in private households - 25% sample data` | `one_parent_economic_families_income_total_25pct_denominator`  |
|             `330` | `Average total income of one-parent economic families in 2020 ($)`                                   | `average_total_income_one_parent_economic_families_2020`       |
|             `331` | `Average after-tax income of one-parent economic families in 2020 ($)`                               | `average_after_tax_income_one_parent_economic_families_2020`   |

## Clean output

The script creates:

```text
output/clean_census_tract_household_family_2021.csv
output/clean_census_tract_household_family_2021.parquet
```

The clean table contains one row per census tract.

In the validated run, the output contained:

```text
6247 census tracts
```

## Derived fields

The script computes several reusable measures:

| Column                                          | Description                                                                                                        |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `average_private_household_size_computed`       | Persons in private households divided by private households. This is not a proportion and can be greater than `1`. |
| `pct_one_parent_families_among_census_families` | One-parent families divided by all census families                                                                 |
| `pct_couple_families_among_census_families`     | Couple families divided by all census families                                                                     |
| `pct_one_parent_family_households`              | One-parent-family households divided by all household types                                                        |
| `pct_multigenerational_households`              | Multigenerational households divided by all household types                                                        |
| `pct_multiple_census_family_households`         | Multiple-census-family households divided by all household types                                                   |
| `pct_one_person_households`                     | One-person households divided by all household types                                                               |
| `pct_persons_in_one_parent_family`              | Persons in one-parent families divided by persons in private households                                            |
| `pct_persons_not_in_census_families`            | Persons not in census families divided by persons in private households                                            |
| `pct_persons_in_census_families`                | Persons in census families divided by persons in private households                                                |
| `pct_persons_living_alone`                      | Persons living alone divided by persons in private households                                                      |

Computed `pct_*` columns are stored as proportions between `0` and `1`.

The script uses safe bounded division for proportions. If a denominator is zero or missing, the derived value is kept as missing rather than becoming infinite. Raw counts are preserved unchanged.

## Default measures

The script defines:

```text
single_parent_measure_default = pct_one_parent_family_households
```

This is the current default Canadian proxy for the SVI single-parent household variable.

The script also defines HGNN / SoVI enrichment fields:

```text
living_alone_measure_default = pct_persons_living_alone
one_person_household_measure_default = pct_one_person_households
multigenerational_household_measure_default = pct_multigenerational_households
multiple_census_family_household_measure_default = pct_multiple_census_family_households
```

These are not direct SVI variables, but they are useful household-structure features for SoVI-style analysis, HGNN node features, and broader vulnerability/resilience modeling.

## Validation notes

The validated run confirmed that the updated script completed successfully and saved both output files.

The table had:

```text
6247 census tracts
```

The main derived variables had the following summaries:

```text
single_parent_measure_default mean                         0.090366
living_alone_measure_default mean                          0.126158
one_person_household_measure_default mean                  0.277765
multigenerational_household_measure_default mean           0.035973
multiple_census_family_household_measure_default mean      0.007571
pct_one_parent_families_among_census_families mean         0.174132
average_private_household_size_computed mean               2.517702
```

The computed average private household size had:

```text
count    6161
mean     2.517702
min      1.000000
max      4.904899
```

This is valid because average household size is not a `0–1` proportion.

## Missing values

The script preserves all census tracts.

In the validated run, the main household/family count variables had:

```text
81 missing values
```

The new household-structure variables also had:

```text
persons_living_alone                         81 missing values
multigenerational_households                 81 missing values
multiple_census_family_households            81 missing values
one_person_households                        81 missing values
```

The one-parent economic-family income variables had more missing values because they are more often suppressed or unavailable.

Missing symbols included:

```text
SYMBOL = x
SYMBOL = ...
```

Missing values are kept as missing. They are not dropped or imputed at this stage.

## Role in the pipeline

This folder produces a clean feature table only.

The broader pipeline is:

```text
raw Census Profile
    ↓
clean household/family feature table
    ↓
master census-tract feature table
    ↓
SVI / SoVI / BRIC / HGNN-specific inputs
```

For the SVI pipeline, the likely selected field will be:

```text
single_parent_measure_default
```

For SoVI and HGNN, useful additional fields include:

```text
average_private_household_size_computed
pct_one_parent_families_among_census_families
pct_persons_not_in_census_families
living_alone_measure_default
one_person_household_measure_default
multigenerational_household_measure_default
multiple_census_family_household_measure_default
median_after_tax_income_one_parent_economic_families_2020
average_after_tax_income_one_parent_economic_families_2020
```

