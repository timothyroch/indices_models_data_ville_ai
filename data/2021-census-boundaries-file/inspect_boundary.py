import geopandas as gpd

path = "lct_000b21a_e.shp"

gdf = gpd.read_file(path)

print("\n--- BASIC INFO ---")
print("Number of rows:", len(gdf))
print("Number of columns:", len(gdf.columns))

print("\n--- COLUMNS ---")
print(list(gdf.columns))

print("\n--- CRS ---")
print(gdf.crs)

print("\n--- GEOMETRY TYPES ---")
print(gdf.geometry.geom_type.value_counts())

print("\n--- FIRST 5 ROWS, NO GEOMETRY ---")
print(gdf.drop(columns="geometry").head())

print("\n--- MISSING VALUES BY COLUMN ---")
print(gdf.isna().sum())

print("\n--- SAMPLE UNIQUE VALUES ---")
for col in gdf.columns:
    if col != "geometry":
        nunique = gdf[col].nunique()
        print(f"{col}: {nunique} unique values")
        if nunique <= 20:
            print(gdf[col].unique())