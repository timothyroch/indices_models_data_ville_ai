# Hospitals per Capita / Hospital Counts

This folder contains the cleaned ODHF hospital-count and hospitals-per-100k-population feature tables for a census-division-level SoVI-style benchmark.

The original SoVI variable is conceptually:

```text
community hospitals per capita
```

In this Canadian / Québec adaptation, this folder produces two related features:

```text
hospital_count_odhf
hospitals_per_100k_population_odhf
```

The first is a raw ODHF hospital facility-record count by census division. The second normalizes that count by 2021 census-division population.

## Current status

This section is complete for both:

```text
hospital counts by census division
hospitals per 100k population by census division
```

Validated hospital-count output:

```text
Total ODHF Quebec hospital records: 349
Automatically assigned using CSDuid: 327
Manually repaired: 22
Final census-division rows: 98
Census divisions with at least one hospital: 78
Census divisions with zero hospitals: 20
```

Validated hospitals-per-100k output:

```text
Final rows: 98
CRS: EPSG:3347
Total hospital_count_odhf: 349
Total automatic assignments: 327
Total manual repairs: 22
Census divisions with at least one hospital: 78
Census divisions with zero hospitals: 20
```

Rate summary:

```text
mean hospitals per 100k: 5.664043
median hospitals per 100k: 4.108602
minimum: 0
maximum: 28.494899
```

## Main outputs

Hospital count outputs:

```text
hospitals_per_capita/output/clean_census_division_hospital_counts_odhf_2021.csv
hospitals_per_capita/output/clean_census_division_hospital_counts_odhf_2021.parquet
```

Hospital-level audit outputs:

```text
hospitals_per_capita/output/clean_odhf_quebec_hospitals_assigned_to_census_divisions.csv
hospitals_per_capita/output/clean_odhf_quebec_hospitals_assigned_to_census_divisions.parquet
```

Hospitals-per-100k outputs:

```text
hospitals_per_capita/output/clean_census_division_hospitals_per_100k_population_odhf_2021.csv
hospitals_per_capita/output/clean_census_division_hospitals_per_100k_population_odhf_2021.parquet
hospitals_per_capita/output/clean_census_division_hospitals_per_100k_population_odhf_2021.geojson
hospitals_per_capita/output/clean_census_division_hospitals_per_100k_population_odhf_2021.gpkg
```

The most important final feature column is:

```text
hospitals_per_100k_population_odhf
```

## Folder structure

```text
hospitals_per_capita/
├── raw/
│   ├── odhf_v1.1.csv
│   └── ODHF_metadata_v1.1.pdf
│
├── lookup/
│   ├── odhf_quebec_hospitals_missing_csd_manual_repair.csv
│   ├── odhf_quebec_hospitals_missing_csd_manual_repair_filled.csv
│   └── quebec_census_division_reference_for_manual_repair.csv
│
├── output/
│   ├── odhf_columns_inventory.csv
│   ├── odhf_facility_type_counts.csv
│   ├── odhf_province_counts.csv
│   ├── odhf_quebec_hospitals_preview.csv
│   ├── odhf_quebec_hospital_counts_by_csd.csv
│   ├── odhf_quebec_hospitals_missing_csd.csv
│   ├── odhf_quebec_hospitals_missing_coordinates.csv
│   ├── odhf_quebec_hospitals_with_derived_cd.csv
│   ├── odhf_quebec_hospital_counts_by_derived_cd.csv
│   ├── odhf_quebec_hospitals_missing_csd_for_manual_review.csv
│   ├── clean_odhf_quebec_hospitals_assigned_to_census_divisions.csv
│   ├── clean_odhf_quebec_hospitals_assigned_to_census_divisions.parquet
│   ├── clean_census_division_hospital_counts_odhf_2021.csv
│   ├── clean_census_division_hospital_counts_odhf_2021.parquet
│   ├── clean_census_division_hospitals_per_100k_population_odhf_2021.csv
│   ├── clean_census_division_hospitals_per_100k_population_odhf_2021.parquet
│   ├── clean_census_division_hospitals_per_100k_population_odhf_2021.geojson
│   └── clean_census_division_hospitals_per_100k_population_odhf_2021.gpkg
│
├── inspect_odhf_hospitals.py
├── inspect_odhf_csd_to_cd_bridge.py
├── create_missing_csd_manual_repair_template.py
├── fill_missing_csd_manual_repair.py
├── clean_census_division_hospital_counts_odhf_2021.py
├── clean_census_division_hospitals_per_100k_population_odhf_2021.py
└── README.md
```

## Source data

The raw hospital source is:

```text
Statistics Canada
Open Database of Healthcare Facilities
ODHF version 1.1
Catalogue number: 13260001
```

Raw file:

```text
raw/odhf_v1.1.csv
```

Metadata file:

```text
raw/ODHF_metadata_v1.1.pdf
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
odhf_facility_type == Hospitals
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
hospitals_per_100k_population_odhf =
hospital_count_odhf / population_total_2021 * 100000
```

## Important metadata notes

ODHF is a harmonized open database built from multiple public and administrative sources.

Important limitations:

```text
ODHF does not guarantee exhaustive coverage.
Facility classification errors are possible.
Geolocation errors are possible.
Some duplicates may remain despite deduplication.
The unit is a healthcare facility record, not necessarily a unique hospital system.
```

Because of this, the final feature should be interpreted as:

```text
ODHF hospital facility record count per 100k population
```

not as a perfect official count of unique community hospitals per capita.

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

### 1. ODHF hospital inspection

Script:

```text
inspect_odhf_hospitals.py
```

Purpose:

```text
Inspect ODHF columns, facility types, province values, Quebec records, Quebec hospital records, CSD identifiers, and missing coordinates.
```

Main findings:

```text
ODHF total records: 7033
ODHF hospital records: 1140
Quebec healthcare facility records: 1606
Quebec hospital records: 349
Quebec hospital records missing CSDuid: 22
```

Useful outputs:

```text
output/odhf_quebec_hospitals_preview.csv
output/odhf_quebec_hospital_counts_by_csd.csv
output/odhf_quebec_hospitals_missing_csd.csv
```

### 2. CSD to census-division bridge inspection

Script:

```text
inspect_odhf_csd_to_cd_bridge.py
```

Purpose:

```text
Test whether ODHF hospital records can be assigned to census divisions using CSDuid.
```

Logic:

```text
CSDuid example: 2466023
derived census division code: first 4 digits = 2466
```

Main finding:

```text
327 of 349 Quebec hospital records had CSDuid.
All 327 derived census division codes were valid.
22 records required manual repair.
```

Useful outputs:

```text
output/odhf_quebec_hospitals_with_derived_cd.csv
output/odhf_quebec_hospital_counts_by_derived_cd.csv
output/odhf_quebec_hospitals_missing_csd_for_manual_review.csv
```

## Manual repair workflow

### 3. Create manual repair template

Script:

```text
create_missing_csd_manual_repair_template.py
```

Purpose:

```text
Create an auditable repair template for the 22 Quebec hospital records missing CSDuid.
```

Outputs:

```text
lookup/odhf_quebec_hospitals_missing_csd_manual_repair.csv
lookup/quebec_census_division_reference_for_manual_repair.csv
```

The repair template contains blank manual assignment columns:

```text
manual_cd_code
manual_census_division_name
manual_census_division_dguid
manual_repair_method
manual_repair_note
```

### 4. Fill manual repair

Script:

```text
fill_missing_csd_manual_repair.py
```

Purpose:

```text
Fill the census-division assignment for the 22 records missing CSDuid.
```

Output:

```text
lookup/odhf_quebec_hospitals_missing_csd_manual_repair_filled.csv
```

Validated repair summary:

```text
Rows repaired: 22
```

Examples:

```text
Alma → Lac-Saint-Jean-Est
Amos → Abitibi
Amqui → La Matapédia
Gatineau → Gatineau
Longueuil → Longueuil
Montréal → Montréal
Rimouski → Rimouski-Neigette
Rouyn-Noranda → Rouyn-Noranda
Saint-Jean-sur-Richelieu → Le Haut-Richelieu
```

Repair methods used:

```text
manual_csdname_to_cd_exact
manual_city_to_cd_exact
```

## Hospital-count cleaner

Script:

```text
clean_census_division_hospital_counts_odhf_2021.py
```

Purpose:

```text
Combine automatically assigned hospital records and manually repaired records, then aggregate hospital counts by 2021 Quebec census division.
```

Inputs:

```text
output/odhf_quebec_hospitals_with_derived_cd.csv
lookup/odhf_quebec_hospitals_missing_csd_manual_repair_filled.csv
doctors_per_100khabs/output/quebec_census_divisions_2021_inventory.csv
```

Outputs:

```text
output/clean_odhf_quebec_hospitals_assigned_to_census_divisions.csv
output/clean_odhf_quebec_hospitals_assigned_to_census_divisions.parquet
output/clean_census_division_hospital_counts_odhf_2021.csv
output/clean_census_division_hospital_counts_odhf_2021.parquet
```

Validated final run:

```text
Final assigned hospital-level table: 349 rows
Automatic assignments: 327
Manual repairs: 22

Final census-division hospital count table: 98 rows
Total hospital_count_odhf: 349
Census divisions with at least one hospital: 78
Census divisions with zero hospitals: 20
```

Top census divisions by ODHF hospital count:

```text
Montréal: 96
Québec: 24
Longueuil: 20
Gatineau: 14
Laval: 9
Nord-du-Québec: 9
Sherbrooke: 8
Le Haut-Richelieu: 7
Sept-Rivières--Caniapiscau: 7
Lévis: 6
```

## Hospitals-per-100k cleaner

Script:

```text
clean_census_division_hospitals_per_100k_population_odhf_2021.py
```

Purpose:

```text
Join hospital_count_odhf to the reusable census-division population/geography layer and compute hospitals per 100k population.
```

Inputs:

```text
output/clean_census_division_hospital_counts_odhf_2021.csv
../census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.geojson
```

Outputs:

```text
output/clean_census_division_hospitals_per_100k_population_odhf_2021.csv
output/clean_census_division_hospitals_per_100k_population_odhf_2021.parquet
output/clean_census_division_hospitals_per_100k_population_odhf_2021.geojson
output/clean_census_division_hospitals_per_100k_population_odhf_2021.gpkg
```

Validated final run:

```text
Final hospitals-per-100k table: 98 rows
CRS: EPSG:3347

Total hospital_count_odhf: 349
Total automatic assignments: 327
Total manual repairs: 22
Census divisions with at least one hospital: 78
Census divisions with zero hospitals: 20
```

Rate summary:

```text
count: 98
mean: 5.664043
std: 5.732373
min: 0
25%: 1.548193
50%: 4.108602
75%: 8.076477
max: 28.494899
```

Top census divisions by hospitals per 100k population:

```text
La Côte-de-Gaspé: 28.494899
Minganie--Le Golfe-du-Saint-Laurent: 20.306630
Nord-du-Québec: 19.676432
La Haute-Côte-Nord: 19.459039
Sept-Rivières--Caniapiscau: 18.305439
La Matapédia: 17.053206
Communauté maritime des Îles-de-la-Madeleine: 15.805279
Témiscouata: 15.390930
Charlevoix: 14.957744
Avignon: 14.908684
```

Top census divisions by raw hospital count:

```text
Montréal: 96 hospitals, 4.789786 per 100k
Québec: 24 hospitals, 4.076246 per 100k
Longueuil: 20 hospitals, 4.578912 per 100k
Gatineau: 14 hospitals, 4.810319 per 100k
Laval: 9 hospitals, 2.053079 per 100k
Nord-du-Québec: 9 hospitals, 19.676432 per 100k
Sherbrooke: 8 hospitals, 4.625614 per 100k
Le Haut-Richelieu: 7 hospitals, 5.771911 per 100k
Sept-Rivières--Caniapiscau: 7 hospitals, 18.305439 per 100k
Lévis: 6 hospitals, 4.008471 per 100k
```

## Main output columns

The hospital-count table contains:

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
hospital_count_odhf
hospital_count_automatic_csd_uid
hospital_count_manual_repair
source_hospitals
source_catalogue
source_odhf_version
feature_description
```

The hospitals-per-100k table contains:

```text
census_division_code
census_division_dguid
census_division_name
census_division_type
unit_type
geography_year
odhf_source_year
hospital_count_odhf
hospital_count_automatic_csd_uid
hospital_count_manual_repair
has_hospital_odhf
population_total_2021
population_total_2016
population_change_pct_2016_2021
total_private_dwellings_2021
private_dwellings_occupied_by_usual_residents_2021
land_area_km2
population_density_per_km2_2021
hospitals_per_100k_population_odhf
hospitals_per_10k_population_odhf
hospital_count_per_km2_odhf
matched_population_frame
source_hospitals
source_population
feature_description
geometry
```

The main feature columns are:

```text
hospital_count_odhf
hospitals_per_100k_population_odhf
```

The audit columns are:

```text
hospital_count_automatic_csd_uid
hospital_count_manual_repair
matched_population_frame
```

## Interpretation

This feature is a count/rate of ODHF records classified as hospitals and assigned to 2021 Québec census divisions.

Recommended feature names:

```text
hospital_count_odhf
hospitals_per_100k_population_odhf
```

Avoid calling it simply:

```text
community_hospitals_per_capita
```

without explanation, because ODHF counts facility records and may include hospital sites, pavilions, specialized hospitals, and other records classified as hospitals.

The per-100k rate should be interpreted as:

```text
ODHF hospital facility records per 100,000 residents
```

not as:

```text
official hospitals per capita
```

or:

```text
hospital beds per capita
```

## Role in the SoVI pipeline

This folder contributes to the census-division-level SoVI-style benchmark.

Current pipeline stage:

```text
ODHF hospital records
    ↓
Quebec hospital records
    ↓
CSDuid-derived census division assignment
    ↓
manual repair for missing CSDuid records
    ↓
hospital counts by census division
    ↓
join to census-division population denominator
    ↓
hospitals per 100k population
```

The normalized feature supports the SoVI-style concept:

```text
community hospitals per capita
```

with the Canadian open-data proxy:

```text
ODHF hospital facility records per 100,000 residents
```


Potential future improvements:

```text
Compare ODHF hospital facility counts against another source if a better official hospital inventory becomes available.
Build a bed-count version if reliable open hospital-bed data becomes available.
Test sensitivity of SoVI results to raw hospital counts versus per-100k hospital rates.
```