# Census Division Ethnocultural / Indigenous Identity 2021

This folder contains the 2021 Québec census-division transformation for SoVI-like ethnocultural and Indigenous identity variables.

The cleaned output supports four SoVI input variables:

```text
pct_black
pct_hispanic
pct_indigenous
pct_asian
````

These variables correspond to the following original SoVI-style variables:

```text
PCTBLACK90      -> pct_black
PCTHISPANIC90   -> pct_hispanic
PCTINDIAN90     -> pct_indigenous
PCTASIAN90      -> pct_asian
```

## Source data

The source file is the 2021 Census Profile at census-division geography:

```text
census_profile_census_division_2021/raw/98-401-X2021004_English_CSV_data.csv
```

The base geography frame is:

```text
census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.csv
```

The transformation keeps one row per Québec census division:

```text
98 Québec census divisions
```

## Scripts

### Inspection script

```text
census_division_ethnocultural_identity_2021/inspect_census_division_ethnocultural_identity_2021.py
```

Run from the `data/` folder:

```bash
python census_division_ethnocultural_identity_2021/inspect_census_division_ethnocultural_identity_2021.py
```

The inspection script verifies the Census Profile characteristic IDs used by the cleaner.

A first keyword-based inspection was too broad and produced false matches such as:

```text
Blackfoot
Chinese languages
Tagalog
```

The corrected inspection therefore uses forced characteristic IDs for the visible-minority and Indigenous identity rows.

### Cleaning script

```text
census_division_ethnocultural_identity_2021/clean_census_division_ethnocultural_identity_2021.py
```

Run from the `data/` folder:

```bash
python census_division_ethnocultural_identity_2021/clean_census_division_ethnocultural_identity_2021.py
```

The cleaner scans the Census Profile file, extracts the forced characteristic IDs, validates coverage and names, computes the derived Asian-group proxy, and writes the clean output files.

## Main clean output

The main clean output is:

```text
census_division_ethnocultural_identity_2021/output/clean_census_division_ethnocultural_identity_2021.csv
```

It contains:

```text
98 rows
98 unique census divisions
4 main SoVI-ready variables
7 Asian component audit variables
```

Main variables:

```text
pct_black
pct_hispanic
pct_indigenous
pct_asian
```

Asian component audit variables:

```text
pct_asian_component_south_asian
pct_asian_component_chinese
pct_asian_component_filipino
pct_asian_component_southeast_asian
pct_asian_component_west_asian
pct_asian_component_korean
pct_asian_component_japanese
```

## Additional output files

The cleaner also writes:

```text
census_division_ethnocultural_identity_2021/output/clean_census_division_ethnocultural_identity_component_long_2021.csv
census_division_ethnocultural_identity_2021/output/clean_census_division_ethnocultural_identity_variable_metadata_2021.csv
census_division_ethnocultural_identity_2021/output/clean_census_division_ethnocultural_identity_summary_2021.csv
```

### `clean_census_division_ethnocultural_identity_component_long_2021.csv`

Long-format audit table containing one row per census division and source characteristic/component.

Useful for checking:

```text
source characteristic ID
source characteristic name
rate value
count value
component alias
method note
```

### `clean_census_division_ethnocultural_identity_variable_metadata_2021.csv`

Metadata table documenting each cleaned variable, original SoVI code, source characteristic ID, unit, derivation, and methodological notes.

### `clean_census_division_ethnocultural_identity_summary_2021.csv`

Summary file documenting row counts, coverage, formula checks, mojibake checks, and descriptive statistics.

## Characteristic IDs used

The cleaner uses the following forced Census Profile characteristic IDs.

### `pct_black`

Original SoVI code:

```text
PCTBLACK90
```

Census Profile source:

```text
CHARACTERISTIC_ID = 1687
CHARACTERISTIC_NAME = Black
```

Method:

```text
direct_rate
```

Selected value column:

```text
C10_RATE_TOTAL
```

Methodological note:

```text
Uses the visible-minority / population-group row Black. This intentionally avoids Blackfoot language or ethnocultural-origin rows.
```

### `pct_hispanic`

Original SoVI code:

```text
PCTHISPANIC90
```

Census Profile source:

```text
CHARACTERISTIC_ID = 1690
CHARACTERISTIC_NAME = Latin American
```

Method:

```text
direct_rate
```

Selected value column:

```text
C10_RATE_TOTAL
```

Methodological note:

```text
The Canadian Census Profile does not use the U.S. Hispanic category directly. Latin American is used as the closest visible-minority / population-group proxy.
```

### `pct_indigenous`

Original SoVI code:

```text
PCTINDIAN90
```

Census Profile source:

```text
CHARACTERISTIC_ID = 1403
CHARACTERISTIC_NAME = Indigenous identity
```

Method:

```text
direct_rate
```

Selected value column:

```text
C10_RATE_TOTAL
```

Methodological note:

```text
Uses the broad Indigenous identity rate. This avoids total-denominator rows, Non-Indigenous identity rows, and narrower First Nations, Métis, or Inuit component rows.
```

### `pct_asian`

Original SoVI code:

```text
PCTASIAN90
```

Method:

```text
derived_component_sum
```

Formula:

```text
pct_asian =
    pct_asian_component_south_asian
  + pct_asian_component_chinese
  + pct_asian_component_filipino
  + pct_asian_component_southeast_asian
  + pct_asian_component_west_asian
  + pct_asian_component_korean
  + pct_asian_component_japanese
```

Selected value column for all components:

```text
C10_RATE_TOTAL
```

Component characteristic IDs:

```text
1685 South Asian
1686 Chinese
1688 Filipino
1691 Southeast Asian
1692 West Asian
1693 Korean
1694 Japanese
```

Methodological note:

```text
pct_asian is a derived Canadian visible-minority Asian-group proxy. It is not a single direct Census Profile row. It intentionally uses visible-minority / population-group components and avoids language rows such as Chinese languages and Tagalog.
```

## Validation results

The latest successful cleaning run produced:

```text
Rows: 98
Unique census divisions: 98
All main variables complete: True
All Asian components complete: True
pct_asian formula max abs difference: 0.0
```

Mojibake checks:

```text
Base names with mojibake: 0
Clean names with mojibake: 0
Profile names with mojibake: 0
```

This means the transformation successfully produced a complete 98-row census-division table with no missing values in the main variables or Asian components.

## Variable summaries from latest run

### `pct_black`

```text
count: 98
mean: 1.392857
min: 0.0
median: 0.55
max: 10.7
```

### `pct_hispanic`

```text
count: 98
mean: 0.689796
min: 0.0
median: 0.4
max: 4.2
```

### `pct_indigenous`

```text
count: 98
mean: 4.593878
min: 0.8
median: 2.0
max: 68.5
```

### `pct_asian`

```text
count: 98
mean: 0.958163
min: 0.0
median: 0.4
max: 13.8
```

## Methodological caveats

This block adapts original U.S. SoVI demographic variables to Canadian Census Profile categories.

Important caveats:

```text
pct_hispanic:
    Uses Latin American as a Canadian proxy for Hispanic.

pct_asian:
    Derived from multiple visible-minority population-group components.

pct_indigenous:
    Uses broad Indigenous identity, which is the closest Canadian analogue to the original SoVI Indigenous / Native American variable.

pct_black:
    Uses the visible-minority Black row and explicitly avoids unrelated Blackfoot rows.
```

These variables should be described as Canadian Census Profile proxies rather than exact reproductions of the original U.S. SoVI variables.

## Current status

This section is complete and ready to be mapped into the SoVI input-source inspection.

The next step is to update:

```text
sovi_2021/inspect_census_division_sovi_input_sources_2021.py
```

so that the following SoVI variables map to this clean output:

```text
pct_black
pct_indigenous
pct_asian
pct_hispanic
```

Expected effect after updating the SoVI inspection:

```text
variables_ready_for_draft_table: 20 -> 24
variables_not_ready_or_unmapped: 22 -> 18
```
