# Québec Census Tract Spatial Frame — 2021

Download: https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/index2021-eng.cfm?year=21
Select: Cartographic Boundary Files (CBF), Census tracts, Shapefile (.shp)

This folder contains the cleaned spatial boundary layer used as the geographic backbone for the SVI and future index/model pipelines.

## Purpose

The goal of this dataset is to define the spatial units on which social, economic, housing, environmental, and infrastructure variables will later be joined.

For the first version of the project, we use **2021 Statistics Canada census tracts in Québec** as the base unit of analysis.

This file does **not** compute the SVI yet. It only prepares the clean geographic table that later scripts can reuse.

## Source data

Original source:

```text
Statistics Canada
2021 Census Tract Cartographic Boundary File
File: lct_000b21a_e.shp
CRS: EPSG:3347
````

The original file is Canada-wide. The cleaning script filters it to Québec using:

```text
PRUID == "24"
```

## Clean output

The cleaning script produces a standardized Québec census tract layer with the following columns:

| Column          | Description                                           |
| --------------- | ----------------------------------------------------- |
| `unit_id`       | Census tract unique identifier from Statistics Canada |
| `statcan_dguid` | Official Statistics Canada geographic identifier      |
| `unit_name`     | Census tract name/number                              |
| `unit_type`     | Always `census_tract`                                 |
| `census_year`   | Census year, currently `2021`                         |
| `province_id`   | Province code, `24` for Québec                        |
| `province_name` | Province name, `Quebec`                               |
| `land_area_km2` | Land area of the census tract in square kilometers    |
| `source`        | Source description                                    |
| `geometry`      | Census tract polygon or multipolygon geometry         |

## Output files

The script saves the cleaned layer in multiple formats:

```text
output/clean_quebec_census_tracts_2021.gpkg
output/clean_quebec_census_tracts_2021.geojson
output/clean_quebec_census_tracts_2021.parquet
```

Recommended uses:

| Format     | Use                                        |
| ---------- | ------------------------------------------ |
| `.gpkg`    | GIS software and robust geospatial storage |
| `.geojson` | Web maps and quick inspection              |
| `.parquet` | Efficient Python/data-science workflows    |

## Role in the pipeline

This dataset is the **spatial backbone** of the project.

Later scripts will load this clean census tract table and join additional variables to it, such as:

```text
population
income
poverty / low income
unemployment
education
age
disability
language
housing
vehicle access
collective dwellings
hazard exposure
infrastructure features
```

The intended pipeline is:

```text
raw boundary file
    ↓
clean Québec census tract spatial frame
    ↓
join SVI variables
    ↓
compute SVI
    ↓
reuse for SoVI, other indices, HGNN, and municipal analysis
```

## Important note

This table only defines **where** each spatial unit is.

It does not yet contain the 15 SVI variables or the final vulnerability scores. Those will be added in later processing steps.
