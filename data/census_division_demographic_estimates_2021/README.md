# Census Division Demographic Estimates 2021

This folder contains the Québec census-division transformation for two SoVI-like demographic variables:

```text
birth_rate
net_international_migration
````

These correspond to the original SoVI variables:

```text
BRATE90   -> birth_rate
MIGRA_97  -> net_international_migration
```

## Source data

The source is the Statistics Canada Annual Demographic Estimates workbook for subprovincial areas:

```text
census_division_demographic_estimates_2021/raw/population_estimates_for_canada_subprovincial_areas.xlsx
```

The workbook contains population estimates and demographic components for Canadian census divisions. The transformation uses the Québec census divisions and joins them to the existing cleaned census-division base frame:

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
python census_division_demographic_estimates_2021/inspect_census_division_demographic_estimates_2021.py
```

Cleaning script:

```bash
python census_division_demographic_estimates_2021/clean_census_division_demographic_estimates_2021.py
```

The inspection script was necessary because the Excel workbook is organized by component sheets rather than as a single tidy table. The final inspection uses strict year-column detection to avoid accidentally selecting title or province-code columns.

## Variables and formulas

### Birth rate

`birth_rate` is computed as a crude birth rate per 1,000 population:

```text
birth_rate =
    1000 * births_2020_2021 / population_2021_estimate
```

Source sheets:

```text
Population
Births~Naissances
```

The population denominator is the July 1, 2021 population estimate. The births numerator is the 2020-2021 annual births component.

### Net international migration

`net_international_migration` is computed from 2020-2021 demographic components:

```text
net_international_migration =
    immigrants_2020_2021
  - emigrants_2020_2021
  + returning_emigrants_2020_2021
  - net_temporary_emigrants_2020_2021
  + net_non_permanent_residents_2020_2021
```

Source sheets:

```text
Immigrants
Emigrants~Émigrants
Ret.Emi~Émi de retour
Net temp emi~Solde émig temp
NPR(n)~RNP(s)
```

`net_non_permanent_residents_2020_2021` is treated as a signed stock-change component and is added directly.

Negative values are valid. They indicate that international-migration losses or negative signed stock changes exceeded gains from immigrants and returning emigrants during the period. Values are not truncated at zero.

The cleaner also computes:

```text
net_international_migration_per_1000
```

This rate is retained as an audit and sensitivity variable. The default SoVI mapping uses the count, because the original SoVI variable is a net international migration measure rather than an explicitly per-capita rate.

## Outputs

Main clean output:

```text
census_division_demographic_estimates_2021/output/clean_census_division_demographic_estimates_2021.csv
```

Additional audit outputs:

```text
census_division_demographic_estimates_2021/output/clean_census_division_demographic_estimates_component_long_2021.csv
census_division_demographic_estimates_2021/output/clean_census_division_demographic_estimates_variable_metadata_2021.csv
census_division_demographic_estimates_2021/output/clean_census_division_demographic_estimates_summary_2021.csv
```

## Validation

The latest successful run produced:

```text
Rows: 98
Unique census divisions: 98
All required numeric columns complete: True
birth_rate formula max abs difference: 0.0
birth_rate alias max abs difference: 0.0
net_international_migration formula max abs difference: 0
net_international_migration_per_1000 formula max abs difference: 0.0
Base names with mojibake: 0
Clean names with mojibake: 0
```

Summary statistics from the latest run:

```text
birth_rate:
    mean 9.15
    min 4.64
    max 19.99

net_international_migration:
    mean 153.01
    min -460
    max 2610

net_international_migration_per_1000:
    mean 1.82
    min -4.57
    max 11.04
```

## Interpretation

These variables adapt the original SoVI demographic variables to a 2021 Québec census-division setting.

`birth_rate` is a crude annual birth rate based on 2020-2021 births and the July 1, 2021 population estimate.

`net_international_migration` measures the net count of international-migration-related population change over 2020-2021. It can be positive or negative. Positive values indicate net gains from international migration components; negative values indicate net losses.

