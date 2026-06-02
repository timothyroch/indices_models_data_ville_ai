# Housing Tenure and Costs 2021

This folder contains the cleaned housing tenure, shelter-cost, dwelling-condition, core-housing-need, and owner/tenant housing-cost feature table derived from the 2021 Statistics Canada Census Profile.

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

The script extracts selected housing tenure, cost, and dwelling-condition characteristics from the 25% sample data section of the Census Profile.

### Tenure

| Characteristic ID | Characteristic name                                                      | Output field                                      |
| ----------------: | ------------------------------------------------------------------------ | ------------------------------------------------- |
|            `1414` | `Total - Private households by tenure - 25% sample data`                 | `tenure_total_households`                         |
|            `1415` | `Owner`                                                                  | `owner_households`                                |
|            `1416` | `Renter`                                                                 | `renter_households`                               |
|            `1417` | `Dwelling provided by the local government, First Nation or Indian band` | `government_or_band_provided_dwelling_households` |

### Condominium status

| Characteristic ID | Characteristic name                                                          | Output field                         |
| ----------------: | ---------------------------------------------------------------------------- | ------------------------------------ |
|            `1418` | `Total - Occupied private dwellings by condominium status - 25% sample data` | `condominium_status_total_dwellings` |
|            `1419` | `Condominium`                                                                | `condominium_dwellings`              |
|            `1420` | `Not condominium`                                                            | `not_condominium_dwellings`          |

### Dwelling condition

| Characteristic ID | Characteristic name                                                          | Output field                                     |
| ----------------: | ---------------------------------------------------------------------------- | ------------------------------------------------ |
|            `1449` | `Total - Occupied private dwellings by dwelling condition - 25% sample data` | `dwelling_condition_total`                       |
|            `1450` | `Only regular maintenance and minor repairs needed`                          | `regular_maintenance_or_minor_repairs_dwellings` |
|            `1451` | `Major repairs needed`                                                       | `major_repairs_needed_dwellings`                 |

### Shelter-cost burden

| Characteristic ID | Characteristic name                                                                                                                                                               | Output field                                        |
| ----------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
|            `1465` | `Total - Owner and tenant households with household total income greater than zero, in non-farm, non-reserve private dwellings by shelter-cost-to-income ratio - 25% sample data` | `shelter_cost_income_ratio_total_households`        |
|            `1466` | `Spending less than 30% of income on shelter costs`                                                                                                                               | `shelter_cost_less_than_30pct_households`           |
|            `1467` | `Spending 30% or more of income on shelter costs`                                                                                                                                 | `shelter_cost_30pct_or_more_households`             |
|            `1468` | `30% to less than 100%`                                                                                                                                                           | `shelter_cost_30pct_to_less_than_100pct_households` |

### Housing indicators

| Characteristic ID | Characteristic name                                                                                                | Output field                                       |
| ----------------: | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------- |
|            `1469` | `Total - Occupied private dwellings by housing indicators - 25% sample data`                                       | `housing_indicators_total_dwellings`               |
|            `1470` | `Total - Households 'spending 30% or more of income on shelter costs' or 'not suitable' or 'major repairs needed'` | `housing_indicator_problem_any`                    |
|            `1471` | `Spending 30% or more of income on shelter costs only`                                                             | `housing_indicator_cost_burden_only`               |
|            `1472` | `Not suitable only`                                                                                                | `housing_indicator_not_suitable_only`              |
|            `1473` | `Major repairs needed only`                                                                                        | `housing_indicator_major_repairs_only`             |
|            `1474` | `'Spending 30% or more of income on shelter costs' and 'not suitable'`                                             | `housing_indicator_cost_burden_and_not_suitable`   |
|            `1475` | `'Spending 30% or more of income on shelter costs' and 'major repairs needed'`                                     | `housing_indicator_cost_burden_and_major_repairs`  |
|            `1476` | `'Not suitable' and 'major repairs needed'`                                                                        | `housing_indicator_not_suitable_and_major_repairs` |
|            `1477` | `'Spending 30% or more of income on shelter costs' and 'not suitable' and 'major repairs needed'`                  | `housing_indicator_all_three`                      |
|            `1478` | `Acceptable housing`                                                                                               | `acceptable_housing`                               |

### Core housing need

| Characteristic ID | Characteristic name                                                                                                                                                                               | Output field                         |
| ----------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
|            `1479` | `Total - Owner and tenant households with household total income greater than zero and shelter-cost-to-income ratio less than 100%, in non-farm, non-reserve private dwellings - 25% sample data` | `core_housing_need_total_households` |
|            `1480` | `In core need`                                                                                                                                                                                    | `core_housing_need_households`       |
|            `1481` | `Not in core need`                                                                                                                                                                                | `not_core_housing_need_households`   |

### Owner housing costs and dwelling values

| Characteristic ID | Characteristic name                                                                     | Output field                                     |
| ----------------: | --------------------------------------------------------------------------------------- | ------------------------------------------------ |
|            `1482` | `Total - Owner households in non-farm, non-reserve private dwellings - 25% sample data` | `owner_households_nonfarm_nonreserve_total`      |
|            `1483` | `% of owner households with a mortgage`                                                 | `published_pct_owner_households_with_mortgage`   |
|            `1484` | `% of owner households spending 30% or more of its income on shelter costs`             | `published_pct_owner_shelter_cost_30pct_or_more` |
|            `1485` | `% in core housing need`                                                                | `published_pct_owner_core_housing_need`          |
|            `1486` | `Median monthly shelter costs for owned dwellings ($)`                                  | `median_monthly_shelter_costs_owned_dwellings`   |
|            `1487` | `Average monthly shelter costs for owned dwellings ($)`                                 | `average_monthly_shelter_costs_owned_dwellings`  |
|            `1488` | `Median value of dwellings ($)`                                                         | `median_value_owned_dwellings`                   |
|            `1489` | `Average value of dwellings ($)`                                                        | `average_value_owned_dwellings`                  |

### Tenant housing costs

| Characteristic ID | Characteristic name                                                                      | Output field                                      |
| ----------------: | ---------------------------------------------------------------------------------------- | ------------------------------------------------- |
|            `1490` | `Total - Tenant households in non-farm, non-reserve private dwellings - 25% sample data` | `tenant_households_nonfarm_nonreserve_total`      |
|            `1491` | `% of tenant households in subsidized housing`                                           | `published_pct_tenant_subsidized_housing`         |
|            `1492` | `% of tenant households spending 30% or more of its income on shelter costs`             | `published_pct_tenant_shelter_cost_30pct_or_more` |
|            `1493` | `% in core housing need`                                                                 | `published_pct_tenant_core_housing_need`          |
|            `1494` | `Median monthly shelter costs for rented dwellings ($)`                                  | `median_monthly_shelter_costs_rented_dwellings`   |
|            `1495` | `Average monthly shelter costs for rented dwellings ($)`                                 | `average_monthly_shelter_costs_rented_dwellings`  |

## Clean output

The script creates:

```text
output/clean_census_tract_housing_tenure_costs_2021.csv
output/clean_census_tract_housing_tenure_costs_2021.parquet
```

The clean table contains one row per census tract.

In the validated run, the output contained:

```text
6247 census tracts
```

## Derived fields

The script computes reusable proportions, including:

| Column                                     | Description                                                                                         |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| `pct_owner`                                | Share of private households that are owner households                                               |
| `pct_renter`                               | Share of private households that are renter households                                              |
| `pct_government_or_band_provided_dwelling` | Share of private households in dwellings provided by local government, First Nation, or Indian band |
| `pct_condominium`                          | Share of occupied private dwellings that are condominiums                                           |
| `pct_major_repairs_needed`                 | Share of occupied private dwellings needing major repairs                                           |
| `pct_shelter_cost_30pct_or_more`           | Share of owner/tenant households spending 30% or more of income on shelter                          |
| `pct_housing_indicator_problem_any`        | Share of dwellings with at least one housing indicator problem                                      |
| `pct_acceptable_housing`                   | Share of dwellings classified as acceptable housing                                                 |
| `pct_core_housing_need`                    | Share of eligible owner/tenant households in core housing need                                      |

Computed `pct_*` columns are stored as proportions between `0` and `1`.

The script uses safe bounded division. If a denominator is zero or missing, the derived value is kept as missing. Derived proportions are clipped to the valid range:

```text
0 <= proportion <= 1
```

Raw counts and dollar values are preserved unchanged.

## Published percentage fields

Some Census Profile rows are already published percentages. The script preserves those original `0–100` values and also creates `0–1` proportion versions.

Examples:

| Published percentage field                        | Proportion field                                  |
| ------------------------------------------------- | ------------------------------------------------- |
| `published_pct_owner_households_with_mortgage`    | `pct_owner_households_with_mortgage_published`    |
| `published_pct_owner_shelter_cost_30pct_or_more`  | `pct_owner_shelter_cost_30pct_or_more_published`  |
| `published_pct_owner_core_housing_need`           | `pct_owner_core_housing_need_published`           |
| `published_pct_tenant_subsidized_housing`         | `pct_tenant_subsidized_housing_published`         |
| `published_pct_tenant_shelter_cost_30pct_or_more` | `pct_tenant_shelter_cost_30pct_or_more_published` |
| `published_pct_tenant_core_housing_need`          | `pct_tenant_core_housing_need_published`          |

## Default measures

The script defines:

```text
renter_measure_default = pct_renter
```

This is a useful SoVI-style tenure feature.

The script defines:

```text
housing_cost_burden_measure_default = pct_shelter_cost_30pct_or_more
```

This captures the share of owner/tenant households spending 30% or more of income on shelter.

The script defines:

```text
major_repairs_measure_default = pct_major_repairs_needed
```

This captures the share of occupied private dwellings needing major repairs.

The script defines:

```text
core_housing_need_measure_default = pct_core_housing_need
```

This captures the share of eligible owner/tenant households in core housing need.

The script defines:

```text
acceptable_housing_measure_default = pct_acceptable_housing
```

This captures the share of occupied private dwellings classified as acceptable housing.

## Validation notes

The validated run confirmed that the corrected IDs fixed the earlier denominator issue.

The main default variables had realistic summaries:

```text
renter_measure_default mean                    0.339067
housing_cost_burden_measure_default mean       0.221821
major_repairs_measure_default mean             0.058380
core_housing_need_measure_default mean         0.106785
acceptable_housing_measure_default mean        0.692908
```

Owner/tenant published shelter-cost burden proportions also looked coherent:

```text
pct_tenant_shelter_cost_30pct_or_more_published mean    0.325842
pct_owner_shelter_cost_30pct_or_more_published mean     0.170744
```

Selected monetary fields also produced plausible summaries:

```text
median_monthly_shelter_costs_rented_dwellings mean    1316.332832
median_value_owned_dwellings mean                       660068.4
```

## Missing values

The script preserves all census tracts.

Missingness varies by variable because this block combines base household counts, housing-condition fields, housing-cost fields, published percentages, owner-specific fields, tenant-specific fields, and dollar-value variables.

In the validated run:

```text
base 25% sample housing fields generally had 89 missing values
shelter-cost and core-housing-need fields had 148 missing values
owner/tenant published percentages and dollar-value fields had higher variable-specific missingness
```

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
clean housing tenure/cost feature table
    ↓
master census-tract feature table
    ↓
SVI / SoVI / BRIC / HGNN-specific inputs
```

For SoVI and HGNN, useful fields include:

```text
renter_measure_default
housing_cost_burden_measure_default
major_repairs_measure_default
core_housing_need_measure_default
acceptable_housing_measure_default
pct_tenant_subsidized_housing_published
pct_owner_households_with_mortgage_published
median_monthly_shelter_costs_rented_dwellings
median_monthly_shelter_costs_owned_dwellings
median_value_owned_dwellings
```
