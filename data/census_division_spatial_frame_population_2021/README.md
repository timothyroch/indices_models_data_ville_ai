# Census Division Spatial Frame with Population 2021

This folder contains the reusable 2021 Québec census-division spatial frame with population and dwelling-count denominators.

It is a core support layer for census-division-level SoVI-style variables that require population normalization, spatial joins, or census-division geometry.

## Current status

This section is complete.

Validated final output:

```text
Quebec census divisions: 98
CRS: EPSG:3347
Quebec total population, 2021: 8,501,833
Quebec total population, 2016: 8,164,361
Population table matched to boundary file by DGUID: yes
Unmatched census divisions: 0
```

The main output files are:

```text
census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.csv
census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.parquet
census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.geojson
census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.gpkg
```

## Folder structure

```text
census_division_spatial_frame_population_2021/
├── raw/
│   ├── 98100007.csv
│   └── 98100007_MetaData.csv
│
├── output/
│   ├── census_profile_candidate_files_inventory.csv
│   ├── census_profile_geo_level_counts.csv
│   ├── population_dwelling_98100007_dguid_join_validation.csv
│   ├── population_dwelling_98100007_dguid_join_matched_quebec.csv
│   ├── population_dwelling_98100007_dguid_join_unmatched_quebec.csv
│   ├── clean_quebec_census_division_spatial_frame_with_population_2021.csv
│   ├── clean_quebec_census_division_spatial_frame_with_population_2021.parquet
│   ├── clean_quebec_census_division_spatial_frame_with_population_2021.geojson
│   └── clean_quebec_census_division_spatial_frame_with_population_2021.gpkg
│
├── inspect_census_division_population_sources.py
├── inspect_population_dwelling_counts_98100007.py
├── inspect_population_dwelling_counts_dguid_join.py
├── clean_census_division_spatial_frame_population_2021.py
└── README.md
```

## Source data

### Boundary source

The spatial boundary source is:

```text
2021-census-division-boundary-file/lcd_000b21a_e/lcd_000b21a_e.shp
```

This is the Statistics Canada 2021 census division boundary file.

The boundary file contains:

```text
CDUID
DGUID
CDNAME
CDTYPE
LANDAREA
PRUID
geometry
```

The Québec subset contains exactly:

```text
98 census divisions
```

### Population and dwelling source

The population and dwelling source is:

```text
census_division_spatial_frame_population_2021/raw/98100007.csv
```

Source product:

```text
Statistics Canada
Table 98-10-0007-01
Population and dwelling counts: Canada and census divisions
```

Metadata file:

```text
census_division_spatial_frame_population_2021/raw/98100007_MetaData.csv
```

The table contains:

```text
Population, 2021
Population, 2016
Population percentage change, 2016 to 2021
Total private dwellings, 2021
Total private dwellings, 2016
Total private dwellings percentage change, 2016 to 2021
Private dwellings occupied by usual residents, 2021
Private dwellings occupied by usual residents, 2016
Private dwellings occupied by usual residents percentage change, 2016 to 2021
Land area in square kilometres, 2021
Population density per square kilometre, 2021
National population rank, 2021
Province/territory population rank, 2021
```

## Why this folder exists

Several SoVI-style variables are available as raw counts at census-division level but require a population denominator.

Examples:

```text
hospital_count_odhf
residential_care_facility_count_odhf
commercial_establishments_count
manufacturing_establishments_count
housing_permits_count
birth_count
migration_count
```

This folder provides the reusable denominator:

```text
population_total_2021
```

so downstream variables can compute:

```text
value_per_100k_population = count / population_total_2021 * 100000
```

## Why Census Profile was not enough

An earlier inspection checked the existing local Census Profile file:

```text
census_profile_2021/98-401-X2021007_English_CSV_data.csv
```

That file did not contain census-division rows. It contained census tracts, census metropolitan areas, and census agglomerations, but no census divisions.

Therefore, a separate population and dwelling counts table was downloaded:

```text
98100007.csv
```

This table directly supports Canada, province/territory, and census-division geography.

## Inspection workflow

### 1. Census Profile population-source inspection

Script:

```text
inspect_census_division_population_sources.py
```

Purpose:

```text
Check whether existing Census Profile files already contain census-division population rows.
```

Main finding:

```text
The local Census Profile extract did not contain census-division rows.
A separate population/dwelling-count table was required.
```

### 2. Initial StatCan table inspection

Script:

```text
inspect_population_dwelling_counts_98100007.py
```

Purpose:

```text
Inspect the downloaded StatCan table 98-10-0007-01 and confirm available columns.
```

Important finding:

```text
The downloaded full-table CSV uses database-style column names rather than display-style column names.
```

For example:

```text
GEO
DGUID
Population and dwelling counts (13): Population, 2021 [1]
Population and dwelling counts (13): Land area in square kilometres, 2021 [10]
```

This meant that the first inspection script’s display-style assumptions had to be replaced with DGUID-based logic.

### 3. DGUID join inspection

Script:

```text
inspect_population_dwelling_counts_dguid_join.py
```

Purpose:

```text
Validate that the downloaded StatCan population/dwelling table can be joined to the Québec census-division boundary inventory using DGUID.
```

Validated result:

```text
Quebec CD inventory rows: 98
Population table rows: 307
Matched Quebec CDs: 98
Unmatched Quebec CDs: 0
Duplicate DGUIDs in population table: 0
Duplicate DGUIDs in Quebec CD inventory: 0
```

Numeric validation:

```text
population_total_2021: missing=0, non_missing=98
population_total_2016: missing=0, non_missing=98
population_change_pct_2016_2021: missing=0, non_missing=98
total_private_dwellings_2021: missing=0, non_missing=98
private_dwellings_occupied_by_usual_residents_2021: missing=0, non_missing=98
land_area_km2_population_table_2021: missing=0, non_missing=98
population_density_per_km2_2021: missing=0, non_missing=98
```

## Final cleaner

Script:

```text
clean_census_division_spatial_frame_population_2021.py
```

Purpose:

```text
Join the 2021 Québec census-division boundary file to the StatCan population/dwelling table using DGUID.
```

Inputs:

```text
2021-census-division-boundary-file/lcd_000b21a_e/lcd_000b21a_e.shp
census_division_spatial_frame_population_2021/raw/98100007.csv
```

Outputs:

```text
output/clean_quebec_census_division_spatial_frame_with_population_2021.csv
output/clean_quebec_census_division_spatial_frame_with_population_2021.parquet
output/clean_quebec_census_division_spatial_frame_with_population_2021.geojson
output/clean_quebec_census_division_spatial_frame_with_population_2021.gpkg
```

Validated final run:

```text
Final reusable Quebec census-division spatial frame
Rows: 98
CRS: EPSG:3347
Quebec total population across CDs: 8,501,833
Quebec total population 2016 across CDs: 8,164,361
```

## Main output columns

The final table contains:

```text
census_division_code
census_division_dguid
census_division_name
census_division_type
province_code
province_name
geography_level
census_year
population_geo_name
population_total_2021
population_total_2016
population_change_pct_2016_2021
total_private_dwellings_2021
total_private_dwellings_2016
total_private_dwellings_change_pct_2016_2021
private_dwellings_occupied_by_usual_residents_2021
private_dwellings_occupied_by_usual_residents_2016
private_dwellings_occupied_by_usual_residents_change_pct_2016_2021
land_area_km2
land_area_km2_boundary
land_area_km2_population_table_2021
land_area_difference_boundary_minus_population_table_km2
population_density_per_km2_2021
national_population_rank_2021
province_population_rank_2021
has_positive_population
matched_population_table
source_boundary
source_population
population_table_id
geometry
```

The most important denominator column is:

```text
population_total_2021
```

The most important spatial column is:

```text
geometry
```

The most important identifier columns are:

```text
census_division_code
census_division_dguid
```

## Main output formats

### CSV

```text
clean_quebec_census_division_spatial_frame_with_population_2021.csv
```

Use this for non-spatial joins and tabular processing.

The CSV does not store native geometry objects. Geometry is stored as WKT in:

```text
geometry_wkt
```

### Parquet

```text
clean_quebec_census_division_spatial_frame_with_population_2021.parquet
```

Use this for fast Python/GeoPandas workflows.

### GeoJSON

```text
clean_quebec_census_division_spatial_frame_with_population_2021.geojson
```

Use this for web maps, lightweight GIS inspection, and visual debugging.

### GeoPackage

```text
clean_quebec_census_division_spatial_frame_with_population_2021.gpkg
```

Use this for GIS workflows in QGIS or other desktop GIS tools.

Layer name:

```text
quebec_census_divisions_2021
```

## CRS

The final spatial files use:

```text
EPSG:3347
```

This is Statistics Canada Lambert / NAD83, a projected CRS appropriate for Canadian national-scale spatial data.

## Population summary

The final population summary was:

```text
count: 98
mean: 86,753.40
min: 6,817
median: 37,590.5
max: 2,004,265
```

Largest census divisions by 2021 population:

```text
Montréal: 2,004,265
Québec: 588,777
Laval: 438,366
Longueuil: 436,785
Gatineau: 291,041
Roussillon: 185,568
Sherbrooke: 172,950
Les Moulins: 171,127
Le Saguenay-et-son-Fjord: 167,833
Thérèse-De Blainville: 163,632
```

Smallest census divisions by 2021 population:

```text
L'Île-d'Orléans: 6,817
Les Basques: 8,873
Minganie--Le Golfe-du-Saint-Laurent: 9,849
La Haute-Côte-Nord: 10,278
La Haute-Gaspésie: 10,950
Communauté maritime des Îles-de-la-Madeleine: 12,654
Mékinac: 12,762
Charlevoix: 13,371
Avignon: 13,415
Les Sources: 14,623
```

## Land-area validation

The final cleaner compares land area from the boundary file against land area from the population table.

The largest differences were approximately:

```text
0.005 km²
```

These are negligible and likely due to rounding differences between the shapefile and the table.

Therefore:

```text
Boundary geometry and population table are aligned.
```

## Role in the SoVI pipeline

This folder is not a SoVI variable itself.

It is a support layer for census-division-level SoVI variables.

Current downstream uses:

```text
hospital_count_odhf
    ÷ population_total_2021
    × 100000
    =
hospitals_per_100k_population_odhf

residential_care_facility_count_odhf
    ÷ population_total_2021
    × 100000
    =
residential_care_facilities_per_100k_population_odhf
```

Future downstream uses may include:

```text
voter turnout spatial allocation
commercial establishments per capita or per square kilometre
manufacturing establishments per capita or per square kilometre
building permits per capita
birth rate
migration rate
population change
```

## Recommended join keys

For census-division-level tabular joins, prefer:

```text
census_division_dguid
```

or:

```text
census_division_code
```

Recommended order of reliability:

```text
1. census_division_dguid
2. census_division_code
3. census_division_name + census_division_type
```

Avoid joining by name only, because some place names can be ambiguous or change across data products.
