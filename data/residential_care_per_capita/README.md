# Residential Care per Capita / Residential-Care Facility Counts

This folder contains the cleaned ODHF residential-care facility-count and residential-care-facilities-per-100k-population feature tables for a census-division-level SoVI-style benchmark.

The original SoVI variable is conceptually closer to:

```text
nursing-home residents per capita
```

In this Canadian / Québec adaptation, this folder produces two related features:

```text
residential_care_facility_count_odhf
residential_care_facilities_per_100k_population_odhf
```

The first is a raw ODHF nursing/residential-care facility-record count by census division. The second normalizes that count by 2021 census-division population.

## Current status

This section is complete for both:

```text
residential-care facility counts by census division
residential-care facilities per 100k population by census division
```

Validated residential-care count output:

```text
Total ODHF Quebec residential-care records: 603
Automatically assigned using CSDuid: 603
Manual repairs: 0
Final census-division rows: 98
Census divisions with at least one residential-care facility: 91
Census divisions with zero residential-care facilities: 7
```

Validated residential-care-per-100k output:

```text
Final rows: 98
CRS: EPSG:3347
Total residential_care_facility_count_odhf: 603
Total automatic assignments: 603
Total manual repairs: 0
Census divisions with at least one residential-care facility: 91
Census divisions with zero residential-care facilities: 7
```

Rate summary:

```text
mean residential-care facilities per 100k: 8.646607
median residential-care facilities per 100k: 7.986735
minimum: 0
maximum: 27.397260
```

## Main outputs

Residential-care count outputs:

```text
residential_care_per_capita/output/clean_census_division_residential_care_counts_odhf_2021.csv
residential_care_per_capita/output/clean_census_division_residential_care_counts_odhf_2021.parquet
```

Facility-level audit outputs:

```text
residential_care_per_capita/output/clean_odhf_quebec_residential_care_assigned_to_census_divisions.csv
residential_care_per_capita/output/clean_odhf_quebec_residential_care_assigned_to_census_divisions.parquet
```

Residential-care-per-100k outputs:

```text
residential_care_per_capita/output/clean_census_division_residential_care_per_100k_population_odhf_2021.csv
residential_care_per_capita/output/clean_census_division_residential_care_per_100k_population_odhf_2021.parquet
residential_care_per_capita/output/clean_census_division_residential_care_per_100k_population_odhf_2021.geojson
residential_care_per_capita/output/clean_census_division_residential_care_per_100k_population_odhf_2021.gpkg
```

The most important final feature column is:

```text
residential_care_facilities_per_100k_population_odhf
```

## Folder structure

```text
residential_care_per_capita/
├── output/
│   ├── odhf_columns_inventory.csv
│   ├── odhf_facility_type_counts.csv
│   ├── odhf_province_counts.csv
│   ├── odhf_quebec_residential_care_preview.csv
│   ├── odhf_quebec_residential_care_counts_by_csd.csv
│   ├── clean_odhf_quebec_residential_care_assigned_to_census_divisions.csv
│   ├── clean_odhf_quebec_residential_care_assigned_to_census_divisions.parquet
│   ├── clean_census_division_residential_care_counts_odhf_2021.csv
│   ├── clean_census_division_residential_care_counts_odhf_2021.parquet
│   ├── clean_census_division_residential_care_per_100k_population_odhf_2021.csv
│   ├── clean_census_division_residential_care_per_100k_population_odhf_2021.parquet
│   ├── clean_census_division_residential_care_per_100k_population_odhf_2021.geojson
│   └── clean_census_division_residential_care_per_100k_population_odhf_2021.gpkg
│
├── inspect_odhf_residential_care.py
├── clean_census_division_residential_care_counts_odhf_2021.py
├── clean_census_division_residential_care_per_100k_population_odhf_2021.py
└── README.md
```

## Shared source data

This folder reuses the ODHF raw file stored in:

```text
hospitals_per_capita/raw/odhf_v1.1.csv
```

The raw ODHF metadata file is also stored there:

```text
hospitals_per_capita/raw/ODHF_metadata_v1.1.pdf
```

This avoids duplicating the same ODHF source file across multiple SoVI variable folders.

## Source data

The raw residential-care source is:

```text
Statistics Canada
Open Database of Healthcare Facilities
ODHF version 1.1
Catalogue number: 13260001
```

The ODHF contains healthcare facilities across Canada and includes harmonized facility categories:

```text
Ambulatory health care services
Hospitals
Nursing and residential care facilities
```

For this feature family, we keep only records where:

```text
province == qc
odhf_facility_type == Nursing and residential care facilities
```

## Population denominator source

The per-100k denominator comes from the reusable census-division spatial frame:

```text
census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.geojson
```

This layer provides:

```text
population_total_2021
geometry
census_division_dguid
census_division_code
```

The denominator source is:

```text
Statistics Canada
Table 98-10-0007-01
Population and dwelling counts: Canada and census divisions
```

The rate formula is:

```text
residential_care_facilities_per_100k_population_odhf =
residential_care_facility_count_odhf / population_total_2021 * 100000
```

## Important metadata notes

ODHF is a harmonized open database built from multiple public and administrative sources.

Important limitations:

```text
ODHF does not guarantee exhaustive coverage.
Facility classification errors are possible.
Geolocation errors are possible.
Some duplicates may remain despite deduplication.
The unit is a healthcare facility record, not a resident, bed, or unique institution system.
```

Because of this, the final feature should be interpreted as:

```text
ODHF nursing/residential-care facility record count per 100k population
```

not as:

```text
CHSLD residents
LTC beds
nursing-home residents
```

## Encoding note

The ODHF CSV is not UTF-8. The inspection script found that it loads correctly with:

```text
cp1252
```

If the file is opened with the wrong encoding, accented characters may appear broken, for example:

```text
Qu�bec
```

This is an encoding-display issue, not necessarily a data-content issue.

The inspection script uses encoding fallback to handle this.

## Inspection workflow

### 1. ODHF residential-care inspection

Script:

```text
inspect_odhf_residential_care.py
```

Purpose:

```text
Inspect ODHF columns, facility types, province values, Quebec records, Quebec residential-care records, CSD identifiers, and missing coordinates.
```

Main findings:

```text
ODHF total records: 7033
Quebec healthcare facility records: 1606
Quebec residential-care records: 603
Quebec residential-care records missing CSDuid: 0
Quebec residential-care records missing coordinates: 0
```

Useful outputs:

```text
output/odhf_quebec_residential_care_preview.csv
output/odhf_quebec_residential_care_counts_by_csd.csv
```

No manual repair file was needed because all 603 Québec residential-care records had a usable `CSDuid`.

## Census-division assignment

The census-division assignment uses the same logic as the hospital feature:

```text
CSDuid example: 2466023
derived census division code: first 4 digits = 2466
```

This works because Québec CSD identifiers contain the census-division code as their first four digits.

The derived census-division code is validated against:

```text
doctors_per_100khabs/output/quebec_census_divisions_2021_inventory.csv
```

That inventory contains the 98 Québec census divisions from the 2021 Statistics Canada census division boundary file.

## Residential-care count cleaner

Script:

```text
clean_census_division_residential_care_counts_odhf_2021.py
```

Purpose:

```text
Assign ODHF Quebec nursing/residential-care facility records to 2021 Quebec census divisions and aggregate facility counts by census division.
```

Inputs:

```text
residential_care_per_capita/output/odhf_quebec_residential_care_preview.csv
doctors_per_100khabs/output/quebec_census_divisions_2021_inventory.csv
```

Outputs:

```text
output/clean_odhf_quebec_residential_care_assigned_to_census_divisions.csv
output/clean_odhf_quebec_residential_care_assigned_to_census_divisions.parquet
output/clean_census_division_residential_care_counts_odhf_2021.csv
output/clean_census_division_residential_care_counts_odhf_2021.parquet
```

Validated final run:

```text
Final assigned residential-care facility-level table: 603 rows
Automatic assignments: 603
Manual repairs: 0

Final census-division residential-care count table: 98 rows
Total residential_care_facility_count_odhf: 603
Census divisions with at least one residential-care facility: 91
Census divisions with zero residential-care facilities: 7
```

Top census divisions by ODHF residential-care facility count:

```text
Montréal: 163
Québec: 45
Laval: 28
Longueuil: 25
La Rivière-du-Nord: 17
Gatineau: 14
Le Saguenay-et-son-Fjord: 13
Francheville: 12
Nord-du-Québec: 10
Sherbrooke: 9
```

## Residential-care-per-100k cleaner

Script:

```text
clean_census_division_residential_care_per_100k_population_odhf_2021.py
```

Purpose:

```text
Join residential_care_facility_count_odhf to the reusable census-division population/geography layer and compute residential-care facilities per 100k population.
```

Inputs:

```text
output/clean_census_division_residential_care_counts_odhf_2021.csv
../census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.geojson
```

Outputs:

```text
output/clean_census_division_residential_care_per_100k_population_odhf_2021.csv
output/clean_census_division_residential_care_per_100k_population_odhf_2021.parquet
output/clean_census_division_residential_care_per_100k_population_odhf_2021.geojson
output/clean_census_division_residential_care_per_100k_population_odhf_2021.gpkg
```

Validated final run:

```text
Final residential-care-per-100k table: 98 rows
CRS: EPSG:3347

Total residential_care_facility_count_odhf: 603
Total automatic assignments: 603
Total manual repairs: 0
Census divisions with at least one residential-care facility: 91
Census divisions with zero residential-care facilities: 7
```

Rate summary:

```text
count: 98
mean: 8.646607
std: 5.448220
min: 0
25%: 5.214257
50%: 7.986735
75%: 11.515262
max: 27.397260
```

Top census divisions by residential-care facilities per 100k population:

```text
La Haute-Gaspésie: 27.397260
Nord-du-Québec: 21.862702
L'Érable: 21.245857
La Tuque: 19.949461
Charlevoix-Est: 19.469141
La Haute-Côte-Nord: 19.459039
Témiscamingue: 18.596578
Bonaventure: 17.087202
Communauté maritime des Îles-de-la-Madeleine: 15.805279
Mékinac: 15.671525
```

Top census divisions by raw residential-care facility count:

```text
Montréal: 163 facilities, 8.132657 per 100k
Québec: 45 facilities, 7.642962 per 100k
Laval: 28 facilities, 6.387357 per 100k
Longueuil: 25 facilities, 5.723640 per 100k
La Rivière-du-Nord: 17 facilities, 12.108780 per 100k
Gatineau: 14 facilities, 4.810319 per 100k
Le Saguenay-et-son-Fjord: 13 facilities, 7.745795 per 100k
Francheville: 12 facilities, 7.578485 per 100k
Nord-du-Québec: 10 facilities, 21.862702 per 100k
Sherbrooke: 9 facilities, 5.203816 per 100k
```

## Main output columns

The residential-care count table contains:

```text
census_division_code
census_division_dguid
census_division_name
census_division_type
census_division_land_area_km2
province_code
unit_type
geography_year
odhf_source_year
residential_care_facility_count_odhf
residential_care_facility_count_automatic_csd_uid
residential_care_facility_count_manual_repair
source_residential_care
source_catalogue
source_odhf_version
feature_description
```

The residential-care-per-100k table contains:

```text
census_division_code
census_division_dguid
census_division_name
census_division_type
unit_type
geography_year
odhf_source_year
residential_care_facility_count_odhf
residential_care_facility_count_automatic_csd_uid
residential_care_facility_count_manual_repair
has_residential_care_facility_odhf
population_total_2021
population_total_2016
population_change_pct_2016_2021
total_private_dwellings_2021
private_dwellings_occupied_by_usual_residents_2021
land_area_km2
population_density_per_km2_2021
residential_care_facilities_per_100k_population_odhf
residential_care_facilities_per_10k_population_odhf
residential_care_facility_count_per_km2_odhf
matched_population_frame
source_residential_care
source_population
feature_description
geometry
```

The main feature columns are:

```text
residential_care_facility_count_odhf
residential_care_facilities_per_100k_population_odhf
```

The audit columns are:

```text
residential_care_facility_count_automatic_csd_uid
residential_care_facility_count_manual_repair
matched_population_frame
```

For this variable, all assignments are automatic:

```text
residential_care_facility_count_automatic_csd_uid = residential_care_facility_count_odhf
residential_care_facility_count_manual_repair = 0
```

## Interpretation

This feature is a count/rate of ODHF records classified as:

```text
Nursing and residential care facilities
```

and assigned to 2021 Québec census divisions.

Recommended feature names:

```text
residential_care_facility_count_odhf
residential_care_facilities_per_100k_population_odhf
```

Avoid calling it:

```text
CHSLD residents per capita
```

because ODHF gives facility records, not resident counts or bed counts.

The per-100k rate should be interpreted as:

```text
ODHF nursing/residential-care facility records per 100,000 residents
```

not as:

```text
CHSLD residents per capita
LTC beds per capita
nursing-home residents per capita
```

## Role in the SoVI pipeline

This folder contributes to the census-division-level SoVI-style benchmark.

Current pipeline stage:

```text
ODHF nursing/residential-care records
    ↓
Quebec residential-care records
    ↓
CSDuid-derived census division assignment
    ↓
residential-care facility counts by census division
    ↓
join to census-division population denominator
    ↓
residential-care facilities per 100k population
```

The normalized feature supports the SoVI-style concept:

```text
nursing-home residents per capita
```

with the Canadian open-data proxy:

```text
ODHF nursing/residential-care facility records per 100,000 residents
```

## Methodological warning

This is a proxy for the original SoVI nursing-home / institutional-care dimension.

Original concept:

```text
nursing-home residents per capita
```

Current open-data proxy:

```text
ODHF nursing/residential-care facility records per 100,000 residents
```

This captures the spatial presence of residential-care infrastructure, but not the number of residents, beds, or care capacity.

## Remaining work

This section is complete.

No remaining work is required for:

```text
residential_care_facility_count_odhf
residential_care_facilities_per_100k_population_odhf
```

Potential future improvements:

```text
Compare ODHF residential-care facility records against a more specific CHSLD inventory if one becomes available.
Build a bed-count version if reliable open LTC/CHSLD bed data becomes available.
Test sensitivity of SoVI results to raw facility counts versus per-100k facility rates.
```