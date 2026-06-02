# Housing Type 2021

This folder contains the cleaned housing-structure feature table derived from the 2021 Statistics Canada Census Profile.

The table is designed as a reusable feature source for SVI, SoVI, BRIC-like indices, HGNN node features, built-form analysis, and other model-ready datasets. It does **not** compute any index directly.

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

The script extracts the full structural dwelling type block from the 100% data section of the Census Profile.

| Characteristic ID | Characteristic name | Output field |
|---:|---|---|
| `41` | `Total - Occupied private dwellings by structural type of dwelling - 100% data` | `occupied_private_dwellings_total_by_structure` |
| `42` | `Single-detached house` | `single_detached_house_dwellings` |
| `43` | `Semi-detached house` | `semi_detached_house_dwellings` |
| `44` | `Row house` | `row_house_dwellings` |
| `45` | `Apartment or flat in a duplex` | `apartment_or_flat_in_duplex_dwellings` |
| `46` | `Apartment in a building that has fewer than five storeys` | `apartment_building_fewer_than_5_storeys_dwellings` |
| `47` | `Apartment in a building that has five or more storeys` | `apartment_building_5_or_more_storeys_dwellings` |
| `48` | `Other single-attached house` | `other_single_attached_house_dwellings` |
| `49` | `Movable dwelling` | `movable_dwelling_dwellings` |

## Important update from the audit

The previous version of this cleaner used IDs:

```text
41, 42, 43, 44, 45, 46, 47, 49
```

The audit showed that ID `48` is also part of the same structural dwelling block:

```text
48 = Other single-attached house
```

The updated cleaner now preserves ID `48` explicitly as:

```text
other_single_attached_house_dwellings
pct_other_single_attached_house
```

This does **not** change the default SVI-style multi-unit or movable-dwelling proxies. It simply makes the housing-type feature table more complete and more auditable.

## Clean output

The script creates:

```text
output/clean_census_tract_housing_type_2021.csv
output/clean_census_tract_housing_type_2021.parquet
```

The clean table contains one row per census tract.

In the validated run, the output contained:

```text
6247 census tracts
```

## Derived fields

The script computes the following reusable proportions:

| Column | Description |
|---|---|
| `pct_single_detached_house` | Share of occupied private dwellings that are single-detached houses |
| `pct_semi_detached_house` | Share of occupied private dwellings that are semi-detached houses |
| `pct_row_house` | Share of occupied private dwellings that are row houses |
| `pct_apartment_or_flat_in_duplex` | Share of occupied private dwellings that are apartments/flats in duplexes |
| `pct_apartment_building_fewer_than_5_storeys` | Share of occupied private dwellings in apartment buildings under five storeys |
| `pct_apartment_building_5_or_more_storeys` | Share of occupied private dwellings in apartment buildings of five or more storeys |
| `pct_other_single_attached_house` | Share of occupied private dwellings that are other single-attached houses |
| `pct_movable_dwelling` | Share of occupied private dwellings that are movable dwellings |
| `apartment_multiunit_dwellings` | Sum of apartment/duplex, apartment under five storeys, and apartment five or more storeys |
| `pct_apartment_multiunit` | Apartment/multi-unit proxy divided by total occupied private dwellings |
| `single_attached_non_apartment_dwellings` | Sum of semi-detached, row house, and other single-attached dwellings |
| `pct_single_attached_non_apartment` | Share of occupied private dwellings that are single-attached but not apartments |
| `non_single_detached_dwellings` | Total occupied private dwellings minus single-detached dwellings |
| `pct_non_single_detached` | Share of occupied private dwellings that are not single-detached |

Computed `pct_*` columns are stored as proportions between `0` and `1`.

## Rounding and bounded proportions

Statistics Canada count categories may be independently rounded. Because of this, component categories can sometimes sum to slightly more than the published total, or a derived residual such as `non_single_detached_dwellings` can become slightly negative.

The script preserves raw counts as provided, but clips derived proportions to the valid range:

```text
0 <= proportion <= 1
```

This prevents impossible derived values such as:

```text
pct_apartment_multiunit > 1
pct_non_single_detached < 0
```

Raw counts are preserved unchanged.

## Default SVI-style housing-type measures

The script defines:

```text
multiunit_measure_default = pct_apartment_multiunit
```

This is the current Canadian apartment/duplex-based proxy for the SVI multi-unit housing variable.

The multi-unit proxy includes:

```text
apartment_or_flat_in_duplex_dwellings
+ apartment_building_fewer_than_5_storeys_dwellings
+ apartment_building_5_or_more_storeys_dwellings
```

It does **not** include semi-detached houses, row houses, or other single-attached houses by default.

The script also defines:

```text
mobile_home_measure_default = pct_movable_dwelling
```

This is the current Canadian Census structural dwelling proxy for the SVI mobile-home / movable-dwelling variable.

## Validation notes

The validated run confirmed that the updated script completed successfully and saved both output files.

The table had:

```text
6247 census tracts
```

The selected characteristics included the newly added structural category:

```text
48 = Other single-attached house
```

The main derived variables had the following summaries:

```text
multiunit_measure_default mean                  0.370423
mobile_home_measure_default mean                0.008029
pct_other_single_attached_house mean            0.002059
pct_single_attached_non_apartment mean          0.133539
pct_non_single_detached mean                    0.512391
```

The validated run also confirmed that derived proportions stayed within the valid range:

```text
0 <= proportion <= 1
```

## Missing values

The script preserves all census tracts.

In the validated run, each raw housing-type variable had:

```text
81 missing values
```

The updated cleaner extracts 9 raw structural dwelling variables, so the total raw missing-value count was:

```text
729
```

This is:

```text
9 selected variables × 81 suppressed census tracts
```

All missing raw values in the selected block were marked with:

```text
SYMBOL = x
```

Missing values are kept as missing. They are not dropped or imputed at this stage.

## Role in the pipeline

This folder produces a clean feature table only.

The broader pipeline is:

```text
raw Census Profile
    ↓
clean housing-type feature table
    ↓
master census-tract feature table
    ↓
SVI / SoVI / BRIC / HGNN-specific inputs
```

For the SVI pipeline, the likely selected fields will be:

```text
multiunit_measure_default
mobile_home_measure_default
```

For SoVI and HGNN, useful additional fields include:

```text
pct_single_detached_house
pct_apartment_multiunit
pct_non_single_detached
pct_single_attached_non_apartment
pct_other_single_attached_house
pct_row_house
pct_movable_dwelling
```