# Language 2021

This folder contains the cleaned language feature table derived from the 2021 Statistics Canada Census Profile.

The table is designed as a reusable feature source for SVI, SoVI, BRIC-like indices, HGNN node features, language-access analysis, and other model-ready datasets. It does **not** compute any index directly.

## Source data

Original source file:

```text
data/census_profile_2021/98-401-X2021007_English_CSV_data.csv
```

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

The script extracts selected language characteristics from the Census Profile.

### Knowledge of official languages

| Characteristic ID | Characteristic name | Output field |
|---:|---|---|
| `383` | `Total - Knowledge of official languages for the total population excluding institutional residents - 100% data` | `official_language_knowledge_total` |
| `384` | `English only` | `knows_english_only` |
| `385` | `French only` | `knows_french_only` |
| `386` | `English and French` | `knows_english_and_french` |
| `387` | `Neither English nor French` | `knows_neither_english_nor_french` |

### First official language spoken

| Characteristic ID | Characteristic name | Output field |
|---:|---|---|
| `388` | `Total - First official language spoken for the total population excluding institutional residents - 100% data` | `first_official_language_total` |
| `389` | `English` | `first_official_language_english` |
| `390` | `French` | `first_official_language_french` |
| `391` | `English and French` | `first_official_language_english_and_french` |
| `392` | `Neither English nor French` | `first_official_language_neither_english_nor_french` |

### Mother tongue

| Characteristic ID | Characteristic name | Output field |
|---:|---|---|
| `393` | `Total - Mother tongue for the total population excluding institutional residents - 100% data` | `mother_tongue_total` |
| `398` | `Non-official languages` | `mother_tongue_non_official_single_response` |

Important note: `398` belongs to the single-response branch of the mother-tongue block. The output field is therefore named explicitly as `mother_tongue_non_official_single_response`.

### Language spoken most often at home

| Characteristic ID | Characteristic name | Output field |
|---:|---|---|
| `735` | `Total - Language spoken most often at home for the total population excluding institutional residents - 100% data` | `home_language_most_often_total` |
| `740` | `Non-official languages` | `home_language_most_often_non_official_single_response` |
| `1060` | `Multiple responses` | `home_language_most_often_multiple_responses` |
| `1062` | `English and non-official language(s)` | `home_language_most_often_english_and_non_official_multiple` |
| `1063` | `French and non-official language(s)` | `home_language_most_often_french_and_non_official_multiple` |
| `1064` | `English, French and non-official language(s)` | `home_language_most_often_english_french_and_non_official_multiple` |
| `1065` | `Multiple non-official languages` | `home_language_most_often_multiple_non_official_languages` |

### Other language(s) spoken regularly at home

| Characteristic ID | Characteristic name | Output field |
|---:|---|---|
| `1066` | `Total - Other language(s) spoken regularly at home for the total population excluding institutional residents - 100% data` | `other_home_language_regularly_total` |
| `1067` | `None` | `other_home_language_regularly_none` |
| `1068` | `English` | `other_home_language_regularly_english` |
| `1069` | `French` | `other_home_language_regularly_french` |
| `1070` | `Non-official language` | `other_home_language_regularly_non_official` |
| `1071` | `Indigenous` | `other_home_language_regularly_non_official_indigenous` |
| `1072` | `Non-Indigenous` | `other_home_language_regularly_non_official_non_indigenous` |

## Important correction from the audit

The earlier version of this cleaner used:

```text
1070 / 735
```

as `pct_home_language_non_official`.

The audit showed that this was semantically imprecise. Characteristic `1070` is not under the `735` language-spoken-most-often-at-home denominator. It belongs under:

```text
1066 = Total - Other language(s) spoken regularly at home
```

The corrected cleaner now separates these two concepts:

```text
pct_home_language_most_often_non_official_including_multiple
```

and

```text
pct_other_home_language_regularly_non_official
```

This makes the language feature table cleaner and easier to interpret.

## Clean output

The script creates:

```text
output/clean_census_tract_language_2021.csv
output/clean_census_tract_language_2021.parquet
```

The clean table contains one row per census tract.

In the validated run, the output contained:

```text
6247 census tracts
```

## Derived fields

The script computes reusable proportions.

### Official-language knowledge

| Column | Description |
|---|---|
| `pct_knows_english_only` | Share of population knowing English only |
| `pct_knows_french_only` | Share of population knowing French only |
| `pct_knows_english_and_french` | Share of population knowing both English and French |
| `pct_knows_neither_english_nor_french` | Share of population knowing neither official language |

### First official language spoken

| Column | Description |
|---|---|
| `pct_first_official_language_english` | Share whose first official language spoken is English |
| `pct_first_official_language_french` | Share whose first official language spoken is French |
| `pct_first_official_language_english_and_french` | Share whose first official language spoken is English and French |
| `pct_first_official_language_neither_english_nor_french` | Share whose first official language spoken is neither English nor French |

### Mother tongue

| Column | Description |
|---|---|
| `pct_mother_tongue_non_official_single_response` | Share with a non-official mother tongue in the single-response branch |

### Language spoken most often at home

| Column | Description |
|---|---|
| `pct_home_language_most_often_non_official_single_response` | Share whose language spoken most often at home is a non-official language in the single-response branch |
| `pct_home_language_most_often_non_official_including_multiple` | Share whose language spoken most often at home includes a non-official language, including selected multiple-response categories |
| `pct_home_language_most_often_multiple_responses` | Share with multiple responses for language spoken most often at home |

### Other language(s) spoken regularly at home

| Column | Description |
|---|---|
| `pct_other_home_language_regularly_none` | Share reporting no other language spoken regularly at home |
| `pct_other_home_language_regularly_english` | Share reporting English as another language spoken regularly at home |
| `pct_other_home_language_regularly_french` | Share reporting French as another language spoken regularly at home |
| `pct_other_home_language_regularly_non_official` | Share reporting a non-official language as another language spoken regularly at home |
| `pct_other_home_language_regularly_non_official_indigenous` | Share reporting an Indigenous non-official language as another language spoken regularly at home |
| `pct_other_home_language_regularly_non_official_non_indigenous` | Share reporting a non-Indigenous non-official language as another language spoken regularly at home |

Computed `pct_*` columns are stored as proportions between `0` and `1`.

The script uses safe bounded division. If a denominator is zero or missing, the derived value is kept as missing rather than becoming infinite. Raw counts are preserved unchanged.

## Default language-barrier measure

The script defines:

```text
language_barrier_measure_default = pct_knows_neither_english_nor_french
```

This is the current Canadian official-language-access proxy for an SVI-style language-barrier variable.

The original U.S. SVI language variable is based on limited English ability. This Canadian/Québec adaptation is not identical. It captures people who report knowing neither official language.

This is a strict and conservative proxy. In Québec, it should be interpreted carefully because English-only, French-only, bilingual, and neither-official-language access can have different implications depending on the municipal service context.

## Additional named language-context measures

The script also defines:

```text
home_language_most_often_non_official_measure_default
    = pct_home_language_most_often_non_official_including_multiple

other_home_language_regularly_non_official_measure_default
    = pct_other_home_language_regularly_non_official

mother_tongue_non_official_measure_default
    = pct_mother_tongue_non_official_single_response
```

These are not direct SVI language-barrier variables. They are broader language-diversity and communication-access features that may be useful for SoVI-style modeling, HGNN node features, sensitivity analysis, or alternative Canadian adaptations of social vulnerability.

## Validation notes

The validated run confirmed that the updated script completed successfully and saved both output files.

The table had:

```text
6247 census tracts
```

The selected characteristics included the corrected and enriched language fields:

```text
393  = Mother tongue total
398  = Non-official languages
740  = Non-official languages under language spoken most often at home
1066 = Other language(s) spoken regularly at home total
1070 = Non-official language under other language(s) spoken regularly at home
```

The main derived variables had the following summaries:

```text
language_barrier_measure_default mean                         0.021937
home_language_most_often_non_official_measure_default mean    0.184476
other_home_language_regularly_non_official_measure_default    0.069574
mother_tongue_non_official_measure_default mean               0.242746
```

The default language-barrier measure had:

```text
count    6163
mean     0.021937
min      0.000000
max      0.435484
```

The corrected home-language and mother-tongue measures had:

```text
home_language_most_often_non_official_measure_default
count    6163
mean     0.184476
max      0.919355

other_home_language_regularly_non_official_measure_default
count    6163
mean     0.069574
max      0.411765

mother_tongue_non_official_measure_default
count    6163
mean     0.242746
max      0.935484
```

## Missing values

The script preserves all census tracts.

In the validated run, each selected raw language variable had:

```text
81 missing values
```

The updated cleaner extracts 26 raw language variables, so the total raw missing-value count was:

```text
2106
```

This is:

```text
26 selected variables × 81 suppressed census tracts
```

All missing raw values in the selected block were marked with:

```text
SYMBOL = x
```

The computed proportion fields had:

```text
84 missing values
```

This happens because some tracts have suppressed or missing raw values, and a few derived proportions have missing or zero denominators.

Missing values are kept as missing. They are not dropped or imputed at this stage.

## Role in the pipeline

This folder produces a clean feature table only.

The broader pipeline is:

```text
raw Census Profile
    ↓
clean language feature table
    ↓
master census-tract feature table
    ↓
SVI / SoVI / BRIC / HGNN-specific inputs
```

For the SVI pipeline, the likely selected field will be:

```text
language_barrier_measure_default
```

For SoVI and HGNN, useful additional fields include:

```text
pct_knows_neither_english_nor_french
pct_first_official_language_neither_english_nor_french
pct_knows_english_and_french
pct_knows_french_only
pct_knows_english_only
home_language_most_often_non_official_measure_default
other_home_language_regularly_non_official_measure_default
mother_tongue_non_official_measure_default
pct_home_language_most_often_non_official_including_multiple
pct_other_home_language_regularly_non_official
pct_mother_tongue_non_official_single_response
```