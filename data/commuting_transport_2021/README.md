# Commuting and Transport 2021

This folder contains the cleaned commuting and transport feature table derived from the 2021 Statistics Canada Census Profile.

The table is designed as a reusable feature source for SoVI-style mobility variables, HGNN node features, transportation-dependence analysis, and other model-ready datasets. It does **not** compute any index directly.

## Important interpretation note

This table does **not** measure household vehicle availability directly.

It measures commuting behaviour and mobility dependence among the employed labour force. Therefore, variables such as `pct_commute_car_truck_van`, `pct_commute_public_transit`, or `pct_commute_non_car_modes` should not be interpreted as exact equivalents of an SVI “no vehicle available” variable.

These features are still useful as mobility-context variables, especially for HGNN models, SoVI-style sensitivity analysis, and Canadian transportation-vulnerability proxies.

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

The script extracts selected commuting and transport characteristics from the 25% sample data section of the Census Profile.

### Place of work status

| Characteristic ID | Characteristic name                                                                                   | Output field                                       |
| ----------------: | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
|            `2593` | `Total - Place of work status for the employed labour force aged 15 years and over - 25% sample data` | `place_of_work_status_total_employed_labour_force` |
|            `2594` | `Worked at home`                                                                                      | `worked_at_home`                                   |
|            `2595` | `Worked outside Canada`                                                                               | `worked_outside_canada`                            |
|            `2596` | `No fixed workplace address`                                                                          | `no_fixed_workplace_address`                       |
|            `2597` | `Usual place of work`                                                                                 | `usual_place_of_work`                              |

### Commuting destination

| Characteristic ID | Characteristic name                                                                                                               | Output field                                      |
| ----------------: | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
|            `2598` | `Total - Commuting destination for the employed labour force aged 15 years and over with a usual place of work - 25% sample data` | `commuting_destination_total_usual_place_of_work` |
|            `2599` | `Commute within census subdivision (CSD) of residence`                                                                            | `commute_within_csd_of_residence`                 |
|            `2600` | `Commute to a different census subdivision (CSD) within census division (CD) of residence`                                        | `commute_to_different_csd_within_cd`              |
|            `2601` | `Commute to a different census subdivision (CSD) and census division (CD) within province or territory of residence`              | `commute_to_different_csd_and_cd_within_province` |
|            `2602` | `Commute to a different province or territory`                                                                                    | `commute_to_different_province_or_territory`      |

### Main mode of commuting

| Characteristic ID | Characteristic name                                                                                                                                              | Output field                      |
| ----------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
|            `2603` | `Total - Main mode of commuting for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data` | `main_mode_commuting_total`       |
|            `2604` | `Car, truck or van`                                                                                                                                              | `commute_car_truck_van`           |
|            `2605` | `Car, truck or van - as a driver`                                                                                                                                | `commute_car_truck_van_driver`    |
|            `2606` | `Car, truck or van - as a passenger`                                                                                                                             | `commute_car_truck_van_passenger` |
|            `2607` | `Public transit`                                                                                                                                                 | `commute_public_transit`          |
|            `2608` | `Walked`                                                                                                                                                         | `commute_walked`                  |
|            `2609` | `Bicycle`                                                                                                                                                        | `commute_bicycle`                 |
|            `2610` | `Other method`                                                                                                                                                   | `commute_other_method`            |

### Commuting duration

| Characteristic ID | Characteristic name                                                                                                                                          | Output field                        |
| ----------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------- |
|            `2611` | `Total - Commuting duration for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data` | `commuting_duration_total`          |
|            `2612` | `Less than 15 minutes`                                                                                                                                       | `commute_duration_less_than_15_min` |
|            `2613` | `15 to 29 minutes`                                                                                                                                           | `commute_duration_15_to_29_min`     |
|            `2614` | `30 to 44 minutes`                                                                                                                                           | `commute_duration_30_to_44_min`     |
|            `2615` | `45 to 59 minutes`                                                                                                                                           | `commute_duration_45_to_59_min`     |
|            `2616` | `60 minutes and over`                                                                                                                                        | `commute_duration_60_min_and_over`  |

### Time leaving for work

| Characteristic ID | Characteristic name                                                                                                                                             | Output field                  |
| ----------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
|            `2617` | `Total - Time leaving for work for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data` | `time_leaving_for_work_total` |
|            `2618` | `Between 5 a.m. and 5:59 a.m.`                                                                                                                                  | `leave_for_work_5_559_am`     |
|            `2619` | `Between 6 a.m. and 6:59 a.m.`                                                                                                                                  | `leave_for_work_6_659_am`     |
|            `2620` | `Between 7 a.m. and 7:59 a.m.`                                                                                                                                  | `leave_for_work_7_759_am`     |
|            `2621` | `Between 8 a.m. and 8:59 a.m.`                                                                                                                                  | `leave_for_work_8_859_am`     |
|            `2622` | `Between 9 a.m. and 11:59 a.m.`                                                                                                                                 | `leave_for_work_9_1159_am`    |
|            `2623` | `Between 12 p.m. and 4:59 a.m.`                                                                                                                                 | `leave_for_work_12pm_459am`   |

## Clean output

The script creates:

```text
output/clean_census_tract_commuting_transport_2021.csv
output/clean_census_tract_commuting_transport_2021.parquet
```

The clean table contains one row per census tract.

In the validated run, the output contained:

```text
6247 census tracts
```

## Derived fields

The script computes reusable proportions, including:

| Column                                                | Description                                                                     |
| ----------------------------------------------------- | ------------------------------------------------------------------------------- |
| `pct_worked_at_home`                                  | Share of employed labour force that worked at home                              |
| `pct_no_fixed_workplace_address`                      | Share of employed labour force with no fixed workplace address                  |
| `pct_usual_place_of_work`                             | Share of employed labour force with a usual place of work                       |
| `pct_commute_within_csd_of_residence`                 | Share commuting within the CSD of residence                                     |
| `pct_commute_to_different_csd_within_cd`              | Share commuting to a different CSD within the same CD                           |
| `pct_commute_to_different_csd_and_cd_within_province` | Share commuting to a different CSD and CD within the same province or territory |
| `pct_commute_car_truck_van`                           | Share commuting by car, truck, or van                                           |
| `pct_commute_public_transit`                          | Share commuting by public transit                                               |
| `pct_commute_walked`                                  | Share commuting by walking                                                      |
| `pct_commute_bicycle`                                 | Share commuting by bicycle                                                      |
| `pct_commute_active_transport`                        | Share commuting by walking or bicycle                                           |
| `pct_commute_non_car_modes`                           | Share commuting by public transit, walking, bicycle, or other method            |
| `pct_commute_duration_45_min_and_over`                | Share with commute duration of 45 minutes or more                               |
| `pct_commute_duration_30_min_and_over`                | Share with commute duration of 30 minutes or more                               |
| `pct_leave_for_work_before_7am`                       | Share leaving for work before 7 a.m.                                            |

Computed `pct_*` columns are stored as proportions between `0` and `1`.

The script uses safe bounded division. If a denominator is zero or missing, the derived value is kept as missing. Derived proportions are clipped to the valid range:

```text
0 <= proportion <= 1
```

Raw counts are preserved unchanged.

## Default mobility measures

The script defines:

```text
car_commuting_measure_default = pct_commute_car_truck_van
```

This is a commuting-mode proxy for car dependence. It is **not** a direct measure of household vehicle availability.

The script defines:

```text
public_transit_commuting_measure_default = pct_commute_public_transit
```

This is a commuting-mode proxy for transit reliance.

The script defines:

```text
long_commute_measure_default = pct_commute_duration_45_min_and_over
```

This is a proxy for long commuting burden.

The script defines:

```text
work_from_home_measure_default = pct_worked_at_home
```

This captures the share of the employed labour force working from home.

## Validation notes

The validated run confirmed that the table was created successfully with:

```text
6247 census tracts
```

The main default variables had the following summaries:

```text
car_commuting_measure_default mean                  0.800342
public_transit_commuting_measure_default mean       0.105325
long_commute_measure_default mean                   0.142711
work_from_home_measure_default mean                 0.261977
```

Additional useful summaries:

```text
pct_no_fixed_workplace_address mean                 0.123455
pct_commute_non_car_modes mean                      0.197649
```

## Missing values

The script preserves all census tracts.

In the validated run, each selected raw commuting/transport variable had:

```text
89 missing values
```

All missing values in the selected block were marked with:

```text
SYMBOL = x
```

This means 89 census tracts had the whole 25% sample commuting/transport block suppressed or unavailable.

Missing values are kept as missing. They are not dropped or imputed at this stage.

## Role in the pipeline

This folder produces a clean feature table only.

The broader pipeline is:

```text
raw Census Profile
    ↓
clean commuting / transport feature table
    ↓
master census-tract feature table
    ↓
SoVI / BRIC / HGNN-specific inputs
```

For HGNN and SoVI-style mobility analysis, useful fields include:

```text
car_commuting_measure_default
public_transit_commuting_measure_default
long_commute_measure_default
work_from_home_measure_default
pct_no_fixed_workplace_address
pct_commute_non_car_modes
pct_commute_active_transport
pct_leave_for_work_before_7am
```

```

This matches the validated run: 6,247 census tracts preserved, 89 suppressed rows kept as missing, and valid mobility/commuting summaries generated.