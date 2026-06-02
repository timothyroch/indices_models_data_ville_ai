# Doctors per 100k Inhabitants

Download data from:

```text
https://www.cihi.ca/en/indicators/physicians-per-100000-population-by-specialty
```

This folder contains the cleaned source data, crosswalk, and census-division proxy table needed to approximate the SoVI variable:

```text
physicians per 100,000 population
```

for a Canadian census-division-level SoVI-style benchmark.

The original SoVI variable is county-level. In this Canadian adaptation, the closest target scale is the census division. However, the available physician-rate data is not census-division-native. It is available at the CIHI health-region level. Therefore, this folder constructs a regional proxy by assigning each census division the physician rate of its corresponding CIHI health region.

## Current status

This section is mostly complete.

It now contains:

```text
1. A cleaned CIHI health-region physician-rate table
2. A Québec census-division inventory
3. A census-division-to-health-region crosswalk
4. A final census-division physician-rate proxy table
```

The final proxy table has:

```text
98 Québec census divisions
97 non-missing physician-rate proxy values
1 unresolved census division
```

The unresolved census division is:

```text
2499 = Nord-du-Québec
```

This missing value is intentional and methodologically documented.

## Folder structure

```text
doctors_per_100khabs/
├── raw/
│   └── cihi_family_medicine_physicians_per_100k_quebec_2024.csv
│
├── lookup/
│   ├── quebec_census_division_to_health_region_crosswalk.csv
│   ├── quebec_census_division_to_health_region_crosswalk_filled.csv
│   └── quebec_census_division_to_health_region_crosswalk_unresolved.csv
│
├── output/
│   ├── clean_health_region_doctors_per_100k_2024.csv
│   ├── clean_health_region_doctors_per_100k_2024.parquet
│   ├── clean_census_division_doctors_per_100k_proxy_2024.csv
│   ├── clean_census_division_doctors_per_100k_proxy_2024.parquet
│   ├── clean_census_division_doctors_per_100k_proxy_summary_2024.csv
│   ├── clean_census_division_doctors_per_100k_proxy_audit_2024.csv
│   ├── clean_census_division_doctors_per_100k_proxy_unresolved_2024.csv
│   ├── quebec_census_divisions_2021_inventory.csv
│   └── candidate_census_division_sources.csv
│
├── clean_doctors_per_100khabs.py
├── clean_census_division_doctors_per_100k_proxy_2024.py
├── inspect_census_division_sources.py
├── inspect_census_division_boundary.py
├── create_health_region_crosswalk_template.py
├── fill_health_region_crosswalk.py
└── README.md
```

## Source data

The raw physician-rate data comes from CIHI:

```text
Physicians per 100,000 Population, by Specialty
```

The file used here contains Québec and Canada rows for:

```text
Family Medicine Physicians per 100,000 Population
```

The useful rows are:

```text
Province/territory == Quebec
Reporting level == Health region
Time frame == 2024
Indicator == Family Medicine Physicians per 100,000 Population
```

The province-level Québec row and the national Canada row are retained in the raw file but excluded from the cleaned health-region output.

## Clean health-region output

The first cleaner script is:

```text
clean_doctors_per_100khabs.py
```

It reads:

```text
raw/cihi_family_medicine_physicians_per_100k_quebec_2024.csv
```

and writes:

```text
output/clean_health_region_doctors_per_100k_2024.csv
output/clean_health_region_doctors_per_100k_2024.parquet
```

This output contains 18 Québec health regions.

Main useful column:

```text
physicians_per_100k_health_region
```

This is the health-region-native physician rate.

## Québec health-region values

The cleaned CIHI table contains the following 18 health-region values:

| Health region                               | Physicians per 100k |
| ------------------------------------------- | ------------------: |
| Abitibi-Témiscamingue Region (Que.)         |                 173 |
| Bas-Saint-Laurent Region (Que.)             |                 154 |
| Capitale-Nationale Region (Que.)            |                 155 |
| Chaudière-Appalaches Region (Que.)          |                 109 |
| Côte-Nord Region (Que.)                     |                 189 |
| Estrie Region (Que.)                        |                 134 |
| Gaspésie–Îles-de-la-Madeleine Region (Que.) |                 225 |
| Lanaudière Region (Que.)                    |                  98 |
| Laurentides Region (Que.)                   |                 113 |
| Laval Region (Que.)                         |                 108 |
| Mauricie et Centre-du-Québec Region (Que.)  |                 123 |
| Montérégie Region (Que.)                    |                 106 |
| Montréal Region (Que.)                      |                 140 |
| Nord-du-Québec Region (Que.)                |                 259 |
| Nunavik Region (Que.)                       |                 435 |
| Outaouais Region (Que.)                     |                 120 |
| Saguenay–Lac-Saint-Jean Region (Que.)       |                 150 |
| Terres-Cries-de-la-Baie-James Region (Que.) |                 351 |

Summary from the validated health-region run:

```text
count    18
mean     174.56
std       91.33
min       98
max      435
```

## Census-division inventory

The census-division inventory was generated from the 2021 Statistics Canada census division boundary shapefile:

```text
2021-census-division-boundary-file/lcd_000b21a_e/lcd_000b21a_e.shp
```

The inspection script is:

```text
inspect_census_division_boundary.py
```

It confirmed that the boundary file contains:

```text
CDUID
DGUID
CDNAME
CDTYPE
LANDAREA
PRUID
geometry
```

For Québec, the boundary file contains:

```text
98 census divisions
```

The non-spatial inventory was saved to:

```text
output/quebec_census_divisions_2021_inventory.csv
```

## Crosswalk template

The script:

```text
create_health_region_crosswalk_template.py
```

created the blank crosswalk template:

```text
lookup/quebec_census_division_to_health_region_crosswalk.csv
```

This file contains one row per Québec census division, with empty columns for:

```text
health_region_name
crosswalk_method
crosswalk_note
```

## Filled crosswalk

The script:

```text
fill_health_region_crosswalk.py
```

fills the crosswalk manually using Québec MRC / territoire équivalent regional membership.

It writes:

```text
lookup/quebec_census_division_to_health_region_crosswalk_filled.csv
```

and unresolved rows to:

```text
lookup/quebec_census_division_to_health_region_crosswalk_unresolved.csv
```

Validated run:

```text
Total census divisions: 98
Assigned census divisions: 97
Unresolved census divisions: 1
```

The assigned census divisions are distributed as follows:

```text
Abitibi-Témiscamingue Region (Que.)             5
Bas-Saint-Laurent Region (Que.)                 8
Capitale-Nationale Region (Que.)                7
Chaudière-Appalaches Region (Que.)             10
Côte-Nord Region (Que.)                         4
Estrie Region (Que.)                            9
Gaspésie–Îles-de-la-Madeleine Region (Que.)     6
Lanaudière Region (Que.)                        6
Laurentides Region (Que.)                       8
Laval Region (Que.)                             1
Mauricie et Centre-du-Québec Region (Que.)     10
Montréal Region (Que.)                          1
Montérégie Region (Que.)                       13
Outaouais Region (Que.)                         5
Saguenay–Lac-Saint-Jean Region (Que.)           4
```

## Final census-division proxy output

The final proxy cleaner is:

```text
clean_census_division_doctors_per_100k_proxy_2024.py
```

It reads:

```text
output/clean_health_region_doctors_per_100k_2024.csv
lookup/quebec_census_division_to_health_region_crosswalk_filled.csv
lookup/quebec_census_division_to_health_region_crosswalk_unresolved.csv
census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.csv
```

and writes:

```text
output/clean_census_division_doctors_per_100k_proxy_2024.csv
output/clean_census_division_doctors_per_100k_proxy_2024.parquet
output/clean_census_division_doctors_per_100k_proxy_summary_2024.csv
output/clean_census_division_doctors_per_100k_proxy_audit_2024.csv
output/clean_census_division_doctors_per_100k_proxy_unresolved_2024.csv
```

Main clean feature columns:

```text
physicians_per_100k
physicians_per_100k_health_region_proxy
physicians_per_100k_health_region
```

The three columns currently carry the same numeric proxy value. They are included to make the downstream SoVI mapping explicit while preserving the fact that the value is health-region-derived.

## Final proxy validation

The final census-division proxy table should contain:

```text
98 rows
98 unique census divisions
97 non-missing physician proxy values
1 missing physician proxy value
```

Expected missing row:

```text
2499 = Nord-du-Québec
```

This missing value is intentional. The cleaner does not fail when one value is missing because the unresolved northern case is methodologically expected.

## Unresolved case

The only unresolved census division is:

```text
2499 = Nord-du-Québec
```

It is unresolved because the census division is too coarse relative to the CIHI northern health-region split.

CIHI reports separate northern health-region values for:

```text
Nord-du-Québec Region (Que.)                  259
Nunavik Region (Que.)                         435
Terres-Cries-de-la-Baie-James Region (Que.)   351
```

At the census-division level, assigning all of `Nord-du-Québec` to only one of these three health regions would be misleading without an additional spatial or population-weighted allocation rule.

For now, this row is left unresolved intentionally.

The unresolved row is saved to:

```text
output/clean_census_division_doctors_per_100k_proxy_unresolved_2024.csv
```

## Methodological interpretation

This variable should be interpreted as:

```text
physicians_per_100k_health_region_proxy
```

not as a direct census-division measurement.

The physician rate is measured at the CIHI health-region level and then assigned to census divisions using a manually filled CD-to-health-region crosswalk.

This preserves meaningful regional variation but does not create true census-division-native physician rates.

## Role in the SoVI pipeline

The intended SoVI feature is:

```text
physicians_per_100k
```

This is a proxy for the original SoVI variable:

```text
PHYSICN90
Physicians per 100,000 population
```

The broader pipeline is now:

```text
CIHI raw physician-rate data
    ↓
clean health-region physician table
    ↓
census-division inventory
    ↓
census-division → health-region crosswalk
    ↓
census-division physician-rate proxy table
    ↓
SoVI census-division benchmark
```

## SoVI integration

The SoVI input-source mapping should use:

```text
physicians_per_100k
```

from:

```text
doctors_per_100khabs/output/clean_census_division_doctors_per_100k_proxy_2024.csv
```

The SoVI missing-data workflow should handle the remaining missing value for `Nord-du-Québec`.

## Scripts and run order

From the `data/` folder:

```bash
python doctors_per_100khabs/clean_doctors_per_100khabs.py
python doctors_per_100khabs/inspect_census_division_boundary.py
python doctors_per_100khabs/create_health_region_crosswalk_template.py
python doctors_per_100khabs/fill_health_region_crosswalk.py
python doctors_per_100khabs/clean_census_division_doctors_per_100k_proxy_2024.py
```

In practice, the first four scripts only need to be rerun if the raw CIHI file, census-division boundary source, or crosswalk logic changes.

## Remaining work

No remaining transformation work is required for this block unless we decide to resolve `Nord-du-Québec`.

Optional future improvement:

```text
Develop a documented spatial or population-weighted allocation rule for Nord-du-Québec.
```

Until then, the missing value should remain explicit rather than silently imputed.