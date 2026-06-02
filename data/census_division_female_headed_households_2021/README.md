# Census Division Female-Headed Households 2021

This folder contains the Québec census-division transformation for the SoVI-like variable:

```text
pct_female_headed_households
````

It corresponds to the original SoVI variable:

```text
PCTF_HH90 -> pct_female_headed_households
```

## Source data

The source is the 2021 Census Profile for Canadian census divisions:

```text
census_profile_census_division_2021/raw/98-401-X2021004_English_CSV_data.csv
```

The transformation joins Census Profile rows to the cleaned Québec census-division base frame:

```text
census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.csv
```

The output contains one row per Québec census division:

```text
98 Québec census divisions
```

## Scripts

Inspection script:

```bash
python census_division_female_headed_households_2021/inspect_census_division_female_headed_households_2021.py
```

Cleaning script:

```bash
python census_division_female_headed_households_2021/clean_census_division_female_headed_households_2021.py
```

The inspection script did not automatically classify the final target because the Census Profile row is context-dependent. The relevant row is labelled `in which the parent is a woman+`, and its meaning depends on the surrounding one-parent-family block. The cleaner therefore uses forced Census Profile characteristic IDs.

## Variable construction

The main variable is taken from:

```text
CHARACTERISTIC_ID = 87
CHARACTERISTIC_NAME = in which the parent is a woman+
SOURCE COLUMN = C10_RATE_TOTAL
```

The cleaned variable is:

```text
pct_female_headed_households
```

This is a Canadian Census Profile adaptation of the original SoVI concept “female-headed households, no spouse present.” It measures female-parent one-parent census families as a percentage of the Census Profile family universe.

The cleaner also preserves context and audit variables:

```text
female_one_parent_families_count
total_one_parent_families_count
pct_one_parent_families
total_census_families_count
female_share_of_one_parent_families
```

The context rows are:

```text
CHARACTERISTIC_ID = 86
CHARACTERISTIC_NAME = Total one-parent families

CHARACTERISTIC_ID = 78
CHARACTERISTIC_NAME = Total number of census families in private households - 100% data
```

The audit variable `female_share_of_one_parent_families` is computed as:

```text
female_share_of_one_parent_families =
    100 * female_one_parent_families_count / total_one_parent_families_count
```

This audit variable is not the default SoVI input. It is retained to check the composition of one-parent families.

## Coverage

The cleaned output has full coverage:

```text
98 / 98 Québec census divisions
```

The latest successful run produced:

```text
Rows: 98
Unique census divisions: 98
pct_female_headed_households non-missing: 98
female_share_of_one_parent_families formula max abs difference: 0.0
pct_female_headed_households alias max abs difference: 0.0
Base names with mojibake: 0
Clean names with mojibake: 0
```

Summary statistics from the latest run:

```text
pct_female_headed_households:
    min    5.4
    mean   10.3551
    median 10.15
    max    21.5

female_share_of_one_parent_families:
    min    58.1940
    mean   69.9441
    median 70.1190
    max    80.6873
```

## Interpretation

`pct_female_headed_households` should be interpreted as a census-family proxy, not as a literal count of all private households headed by women.

The original SoVI variable refers to female-headed households with no spouse present. The Canadian Census Profile does not expose that exact U.S.-style variable directly at the same conceptual level in the inspected rows. The chosen adaptation uses female-parent one-parent census families because it captures the family-structure dimension most closely related to the original concept.

This variable is therefore a strong Canadian proxy, but final reporting should state that the universe is census families rather than all private households.
