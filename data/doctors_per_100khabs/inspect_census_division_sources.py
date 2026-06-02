from pathlib import Path
import pandas as pd


# ============================================================
# Inspect Existing Files for Census Division Information
# ============================================================
#
# Purpose:
#   Check whether existing boundary/spatial-frame files already contain
#   census-division identifiers needed for the doctors_per_100khabs crosswalk.
#
# Run from data/:
#   python doctors_per_100khabs/inspect_census_division_sources.py
#
# ============================================================


DATA_DIR = Path(__file__).resolve().parent.parent

CANDIDATE_DIRS = [
    DATA_DIR / "2021-census-boundaries-file",
    DATA_DIR / "spatial_frame_population_2021",
]

OUTPUT_DIR = DATA_DIR / "doctors_per_100khabs" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "candidate_census_division_sources.csv"


KEYWORDS = [
    "CDUID",
    "CDNAME",
    "CDTYPE",
    "CSDUID",
    "CSDNAME",
    "CTUID",
    "DGUID",
    "PRUID",
    "PRNAME",
    "GEO_NAME",
    "GEO_LEVEL",
    "LANDAREA",
    "geometry",
]


def inspect_csv(path: Path):
    try:
        df = pd.read_csv(path, nrows=5, low_memory=False)
        return {
            "path": str(path.relative_to(DATA_DIR)),
            "file_type": "csv",
            "readable": True,
            "rows_previewed": len(df),
            "columns": list(df.columns),
            "matched_columns": [c for c in df.columns if c in KEYWORDS],
            "error": "",
        }
    except Exception as e:
        return {
            "path": str(path.relative_to(DATA_DIR)),
            "file_type": "csv",
            "readable": False,
            "rows_previewed": 0,
            "columns": [],
            "matched_columns": [],
            "error": str(e),
        }


def inspect_parquet(path: Path):
    try:
        df = pd.read_parquet(path)
        return {
            "path": str(path.relative_to(DATA_DIR)),
            "file_type": "parquet",
            "readable": True,
            "rows_previewed": min(len(df), 5),
            "columns": list(df.columns),
            "matched_columns": [c for c in df.columns if c in KEYWORDS],
            "error": "",
        }
    except Exception as e:
        return {
            "path": str(path.relative_to(DATA_DIR)),
            "file_type": "parquet",
            "readable": False,
            "rows_previewed": 0,
            "columns": [],
            "matched_columns": [],
            "error": str(e),
        }


def inspect_geospatial(path: Path):
    try:
        import geopandas as gpd

        gdf = gpd.read_file(path)
        return {
            "path": str(path.relative_to(DATA_DIR)),
            "file_type": path.suffix.lower(),
            "readable": True,
            "rows_previewed": min(len(gdf), 5),
            "columns": list(gdf.columns),
            "matched_columns": [c for c in gdf.columns if c in KEYWORDS],
            "error": "",
        }
    except Exception as e:
        return {
            "path": str(path.relative_to(DATA_DIR)),
            "file_type": path.suffix.lower(),
            "readable": False,
            "rows_previewed": 0,
            "columns": [],
            "matched_columns": [],
            "error": str(e),
        }


results = []

for folder in CANDIDATE_DIRS:
    print(f"\nInspecting folder: {folder}")

    if not folder.exists():
        print("  Folder does not exist.")
        continue

    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix == ".csv":
            result = inspect_csv(path)
        elif suffix == ".parquet":
            result = inspect_parquet(path)
        elif suffix in [".shp", ".geojson", ".gpkg"]:
            result = inspect_geospatial(path)
        else:
            continue

        results.append(result)

        print("\nFile:", result["path"])
        print("Readable:", result["readable"])
        print("Matched columns:", result["matched_columns"])
        print("All columns:", result["columns"])

        if result["error"]:
            print("Error:", result["error"])


if not results:
    raise ValueError("No candidate CSV, Parquet, SHP, GeoJSON, or GPKG files found.")

summary = pd.DataFrame(results)
summary["columns"] = summary["columns"].apply(lambda x: ", ".join(x))
summary["matched_columns"] = summary["matched_columns"].apply(lambda x: ", ".join(x))

summary.to_csv(OUTPUT_CSV, index=False)

print("\nSaved inspection summary:")
print(OUTPUT_CSV)

print("\nBest signs that a file can help us:")
print("- It contains CDUID and CDNAME directly.")
print("- Or it contains CSDUID/CSDNAME, which may allow CSD → CD mapping.")
print("- Or it contains CTUID plus higher-level geography fields.")