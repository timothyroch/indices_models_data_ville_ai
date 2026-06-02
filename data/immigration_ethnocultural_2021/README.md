# Immigration and Ethnocultural Features 2021

This folder contains the cleaned immigration, citizenship, generation-status, Indigenous identity, visible-minority, admission-category, and ethnocultural feature table derived from the 2021 Statistics Canada Census Profile.

The table is designed as a reusable feature source for SVI, SoVI, BRIC-like indices, HGNN node features, sensitivity analysis, and other model-ready datasets. It does **not** compute any index directly.

## Important interpretation note

These variables are **area-level demographic and contextual features**.

They should **not** be interpreted as individual vulnerability labels.

For example, a high value of `pct_visible_minority`, `pct_immigrant`, `pct_non_permanent_resident`, `pct_indigenous_identity`, `pct_not_canadian_citizens`, or `pct_first_generation` does not mean that individuals in those groups are inherently vulnerable.

These variables are used only as tract-level proxies for structural context, differential exposure, access barriers, institutional inequities, language/service-access patterns, immigration context, and possible social-resource constraints.

Any index-specific use of these variables must document the theoretical justification and avoid essentializing demographic groups.

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

The script extracts selected immigration, citizenship, Indigenous identity, generation-status, admission-category, visible-minority, and ethnocultural characteristics from the 25% sample data section of the Census Profile.

### Indigenous identity

| Characteristic ID | Characteristic name                                                                      | Output field                         |
| ----------------: | ---------------------------------------------------------------------------------------- | ------------------------------------ |
|            `1402` | `Total - Indigenous identity for the population in private households - 25% sample data` | `indigenous_identity_total`          |
|            `1403` | `Indigenous identity`                                                                    | `indigenous_identity_population`     |
|            `1410` | `Non-Indigenous identity`                                                                | `non_indigenous_identity_population` |

### Citizenship

| Characteristic ID | Characteristic name                                                              | Output field                       |
| ----------------: | -------------------------------------------------------------------------------- | ---------------------------------- |
|            `1522` | `Total - Citizenship for the population in private households - 25% sample data` | `citizenship_total`                |
|            `1523` | `Canadian citizens`                                                              | `canadian_citizens_population`     |
|            `1526` | `Not Canadian citizens`                                                          | `not_canadian_citizens_population` |

### Immigrant status and non-permanent residents

| Characteristic ID | Characteristic name                                                                                             | Output field                        |
| ----------------: | --------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
|            `1527` | `Total - Immigrant status and period of immigration for the population in private households - 25% sample data` | `immigrant_status_total`            |
|            `1528` | `Non-immigrants`                                                                                                | `non_immigrant_population`          |
|            `1529` | `Immigrants`                                                                                                    | `immigrant_population`              |
|            `1537` | `Non-permanent residents`                                                                                       | `non_permanent_resident_population` |

### Place of birth

| Characteristic ID | Characteristic name                                                                                  | Output field                                       |
| ----------------: | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
|            `1544` | `Total - Place of birth for the immigrant population in private households - 25% sample data`        | `place_of_birth_immigrant_population_total`        |
|            `1604` | `Total - Place of birth for the recent immigrant population in private households - 25% sample data` | `place_of_birth_recent_immigrant_population_total` |

### Generation status

| Characteristic ID | Characteristic name                                                                    | Output field                          |
| ----------------: | -------------------------------------------------------------------------------------- | ------------------------------------- |
|            `1665` | `Total - Generation status for the population in private households - 25% sample data` | `generation_status_total`             |
|            `1666` | `First generation`                                                                     | `first_generation_population`         |
|            `1667` | `Second generation`                                                                    | `second_generation_population`        |
|            `1668` | `Third generation or more`                                                             | `third_generation_or_more_population` |

### Admission category

| Characteristic ID | Characteristic name                                                                                                                                          | Output field                                             |
| ----------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------- |
|            `1669` | `Total - Admission category and applicant type for the immigrant population in private households who were admitted between 1980 and 2021 - 25% sample data` | `admission_category_total_immigrants_admitted_1980_2021` |
|            `1670` | `Economic immigrants`                                                                                                                                        | `economic_immigrants_population`                         |
|            `1673` | `Immigrants sponsored by family`                                                                                                                             | `family_sponsored_immigrants_population`                 |
|            `1674` | `Refugees`                                                                                                                                                   | `refugees_population`                                    |
|            `1675` | `Other immigrants`                                                                                                                                           | `other_immigrants_population`                            |

Important denominator note:

```text
1670, 1673, 1674, and 1675 are subcategories of 1669.
```

Therefore, the preferred admission-category proportions use:

```text
admission_category_total_immigrants_admitted_1980_2021
```

as the denominator, not the broader immigrant-status total.

### Visible minority and ethnic/cultural origin

| Characteristic ID | Characteristic name                                                                            | Output field                      |
| ----------------: | ---------------------------------------------------------------------------------------------- | --------------------------------- |
|            `1683` | `Total - Visible minority for the population in private households - 25% sample data`          | `visible_minority_total`          |
|            `1684` | `Total visible minority population`                                                            | `visible_minority_population`     |
|            `1695` | `Visible minority, n.i.e.`                                                                     | `visible_minority_nie_population` |
|            `1697` | `Not a visible minority`                                                                       | `not_visible_minority_population` |
|            `1698` | `Total - Ethnic or cultural origin for the population in private households - 25% sample data` | `ethnic_or_cultural_origin_total` |

## Clean output

The script creates:

```text
output/clean_census_tract_immigration_ethnocultural_2021.csv
output/clean_census_tract_immigration_ethnocultural_2021.parquet
```

The clean table contains one row per census tract.

In the validated run, the output contained:

```text
6247 census tracts
```

## Derived fields

The script computes reusable proportions.

### Indigenous identity

| Column                        | Description                                                                       |
| ----------------------------- | --------------------------------------------------------------------------------- |
| `pct_indigenous_identity`     | Indigenous identity population divided by the Indigenous identity denominator     |
| `pct_non_indigenous_identity` | Non-Indigenous identity population divided by the Indigenous identity denominator |

### Citizenship

| Column                      | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| `pct_canadian_citizens`     | Canadian citizens divided by the citizenship denominator     |
| `pct_not_canadian_citizens` | Not Canadian citizens divided by the citizenship denominator |

### Immigrant status

| Column                       | Description                                                                   |
| ---------------------------- | ----------------------------------------------------------------------------- |
| `pct_immigrant`              | Immigrant population divided by the immigrant-status denominator              |
| `pct_non_immigrant`          | Non-immigrant population divided by the immigrant-status denominator          |
| `pct_non_permanent_resident` | Non-permanent resident population divided by the immigrant-status denominator |

### Generation status

| Column                         | Description                                                                      |
| ------------------------------ | -------------------------------------------------------------------------------- |
| `pct_first_generation`         | First-generation population divided by the generation-status denominator         |
| `pct_second_generation`        | Second-generation population divided by the generation-status denominator        |
| `pct_third_generation_or_more` | Third-generation-or-more population divided by the generation-status denominator |

### Admission category

Preferred admission-category proportions:

| Column                                                        | Description                                                               |
| ------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `pct_economic_immigrants_of_admission_category_total`         | Economic immigrants divided by the admission-category denominator         |
| `pct_family_sponsored_immigrants_of_admission_category_total` | Family-sponsored immigrants divided by the admission-category denominator |
| `pct_refugees_of_admission_category_total`                    | Refugees divided by the admission-category denominator                    |
| `pct_other_immigrants_of_admission_category_total`            | Other immigrants divided by the admission-category denominator            |

The script also keeps legacy/contextual broad-denominator ratios:

| Column                                                      | Description                                                                     |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `pct_economic_immigrants_of_immigrant_status_total`         | Economic immigrants divided by the broader immigrant-status denominator         |
| `pct_family_sponsored_immigrants_of_immigrant_status_total` | Family-sponsored immigrants divided by the broader immigrant-status denominator |
| `pct_refugees_of_immigrant_status_total`                    | Refugees divided by the broader immigrant-status denominator                    |
| `pct_other_immigrants_of_immigrant_status_total`            | Other immigrants divided by the broader immigrant-status denominator            |

These broad-denominator ratios are retained for context only. The admission-category denominator is preferred for admission-category interpretation.

### Visible minority

| Column                     | Description                                                                         |
| -------------------------- | ----------------------------------------------------------------------------------- |
| `pct_visible_minority`     | Visible minority population divided by the visible-minority denominator             |
| `pct_not_visible_minority` | Not visible minority population divided by the visible-minority denominator         |
| `pct_visible_minority_nie` | Visible minority not included elsewhere divided by the visible-minority denominator |

Computed `pct_*` columns are stored as proportions between `0` and `1`.

The script uses safe bounded division. If a denominator is zero or missing, the derived value is kept as missing rather than becoming infinite. Raw counts are preserved unchanged.

## Default and named contextual measures

The script defines:

```text
ethnocultural_measure_default = pct_visible_minority
```

This is the current default area-level structural/demographic proxy for SVI/SoVI-style construction.

This default should be interpreted carefully. It should not be treated as a claim that visible-minority populations are intrinsically vulnerable. It is a tract-level contextual feature that may reflect broader structural conditions, historical inequities, language access patterns, housing-market exposure, institutional access, or other social determinants depending on the final index framing.

The script also defines:

```text
immigrant_measure_default = pct_immigrant
non_permanent_resident_measure_default = pct_non_permanent_resident
not_canadian_citizen_measure_default = pct_not_canadian_citizens
first_generation_measure_default = pct_first_generation
refugee_admission_category_measure_default = pct_refugees_of_admission_category_total
```

These are useful for SoVI variants, HGNN node features, sensitivity analysis, and alternative Canadian adaptations of social vulnerability.

## Validation notes

The validated run confirmed that the table was created successfully with:

```text
6247 census tracts
```

The selected characteristics included all intended enrichment and correction fields:

```text
1523 = Canadian citizens
1526 = Not Canadian citizens
1666 = First generation
1667 = Second generation
1668 = Third generation or more
1669 = Admission-category denominator
1674 = Refugees
```

The main summary values from the validated run were:

```text
ethnocultural_measure_default mean              0.310139
pct_visible_minority mean                       0.310139
pct_immigrant mean                              0.265339
pct_non_permanent_resident mean                 0.030615
pct_indigenous_identity mean                    0.039898
not_canadian_citizen_measure_default mean       0.101345
first_generation_measure_default mean           0.305934
refugee_admission_category_measure_default mean 0.154527
```

The admission-category denominator had:

```text
count    6158
mean     1046.541897
min      0.000000
max      16370.000000
```

Some tracts have `0` as the admission-category denominator. In those cases, admission-category proportions such as `refugee_admission_category_measure_default` are kept as missing rather than forced to zero.

## Missing values

The script preserves all census tracts.

In the validated run, each selected raw 25% sample variable had:

```text
89 missing values
```

All missing raw values in the selected block were marked with:

```text
SYMBOL = x
```

Because the updated cleaner extracts 26 raw variables, the total number of missing raw values reported was:

```text
2314
```

This is:

```text
26 selected variables × 89 suppressed census tracts
```

Missing values are kept as missing. They are not dropped or imputed at this stage.

## Role in the pipeline

This folder produces a clean feature table only.

The broader pipeline is:

```text
raw Census Profile
    ↓
clean immigration/ethnocultural feature table
    ↓
master census-tract feature table
    ↓
SVI / SoVI / BRIC / HGNN-specific inputs
```

For the SVI pipeline, the likely selected field will be:

```text
ethnocultural_measure_default
```

For SoVI and HGNN, useful additional fields include:

```text
pct_visible_minority
pct_immigrant
pct_non_permanent_resident
pct_indigenous_identity
pct_not_visible_minority
not_canadian_citizen_measure_default
first_generation_measure_default
refugee_admission_category_measure_default
pct_refugees_of_admission_category_total
pct_economic_immigrants_of_admission_category_total
pct_family_sponsored_immigrants_of_admission_category_total
```