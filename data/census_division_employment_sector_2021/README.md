# Census Division Employment Sector 2021

This folder contains the Québec census-division transformation for three SoVI-like employment variables:

```text
pct_extractive_employment
pct_transport_utility_employment
pct_service_employment
````

These correspond to the original SoVI variables:

```text
AGRIPC90  -> pct_extractive_employment
TRANPC90  -> pct_transport_utility_employment
SERVPC90  -> pct_service_employment
```

## Source data

The source is the 2021 Statistics Canada Census Profile at census-division geography:

```text
census_profile_census_division_2021/raw/98-401-X2021004_English_CSV_data.csv
```

The base geography frame is:

```text
census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.csv
```

The output keeps one row per Québec census division:

```text
98 Québec census divisions
```

## Scripts

Inspection script:

```bash
python census_division_employment_sector_2021/inspect_census_division_employment_sector_2021.py
```

Cleaning script:

```bash
python census_division_employment_sector_2021/clean_census_division_employment_sector_2021.py
```

The inspection step was necessary because the original SoVI employment variables can be interpreted through either industry categories or occupation categories. The cleaner therefore uses explicit characteristic IDs rather than relying on broad keyword matching.

## Methodological choices

### Extractive employment

`pct_extractive_employment` is derived as a primary/extractive industry proxy:

```text
pct_extractive_employment =
    Agriculture, forestry, fishing and hunting
  + Mining, quarrying, and oil and gas extraction
```

Source Census Profile rows:

```text
2262  11 Agriculture, forestry, fishing and hunting
2263  21 Mining, quarrying, and oil and gas extraction
```

This is treated as the Canadian census-division proxy for `AGRIPC90`.

### Transportation, utilities, and communications employment

`pct_transport_utility_employment` is derived as an industry proxy:

```text
pct_transport_utility_employment =
    Transportation and warehousing
  + Information and cultural industries
  + Utilities
```

Source Census Profile rows:

```text
2269  48-49 Transportation and warehousing
2270  51 Information and cultural industries
2264  22 Utilities
```

This is treated as the Canadian proxy for `TRANPC90`. The cleaner intentionally uses the `Utilities` industry row, not the row `Occupations in manufacturing and utilities`, because the latter is occupation-based and would mix concepts.

### Service employment

`pct_service_employment` uses the occupation row:

```text
2255  6 Sales and service occupations
```

This choice preserves the service-occupation interpretation of `SERVPC90`. The cleaner does not sum service industries such as retail, health care, education, finance, accommodation, or public administration, because that would change the variable from an occupation measure into an industry measure.

## Outputs

Main clean output:

```text
census_division_employment_sector_2021/output/clean_census_division_employment_sector_2021.csv
```

Additional audit outputs:

```text
census_division_employment_sector_2021/output/clean_census_division_employment_sector_component_long_2021.csv
census_division_employment_sector_2021/output/clean_census_division_employment_sector_variable_metadata_2021.csv
census_division_employment_sector_2021/output/clean_census_division_employment_sector_summary_2021.csv
```

The clean table includes the three main variables and component columns retained for audit.

## Validation

The latest successful run produced:

```text
Rows: 98
Unique census divisions: 98
All main variables complete: True
All components complete: True
pct_extractive_employment formula max abs difference: 0.0
pct_transport_utility_employment formula max abs difference: 0.0
pct_service_employment formula max abs difference: 0.0
Base names with mojibake: 0
Clean names with mojibake: 0
Profile names with mojibake: 0
```

Summary statistics from the latest run:

```text
pct_extractive_employment:
    mean 6.09
    min 0.4
    max 20.4

pct_transport_utility_employment:
    mean 6.45
    min 3.2
    max 10.6

pct_service_employment:
    mean 23.32
    min 16.6
    max 30.7
```

## Interpretation

These variables are Canadian Census Profile proxies for the original SoVI employment structure variables.

`pct_extractive_employment` and `pct_transport_utility_employment` are industry-based derived measures. `pct_service_employment` is occupation-based. This mixed treatment is intentional because it better preserves the conceptual meaning of the original SoVI variables than forcing all three into either industry categories or occupation categories.

````

After saving both files, rerun:

```bash
python sovi_2021/inspect_census_division_sovi_input_sources_2021.py
````

Expected update:

```text
variables_ready_for_draft_table: 26 -> 29
variables_not_ready_or_unmapped: 16 -> 13
```
