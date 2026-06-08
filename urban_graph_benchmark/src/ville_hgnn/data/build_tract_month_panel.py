"""
Build Dataset v0 for the Montréal 311 water/drainage benchmark.

This module builds the first benchmark dataset:

    Montréal census tract × month
    target = reported water/drainage 311 requests

It does NOT implement baselines, GraphSAGE, HGNN models,
explainability, OSM routing, road-network travel distances, or population-
weighted centroids. Those belong in later modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    _PANDAS_IMPORT_ERROR = exc
else:
    _PANDAS_IMPORT_ERROR = None

try:
    import geopandas as gpd
except ImportError as exc:  # pragma: no cover
    gpd = None  # type: ignore[assignment]
    _GEOPANDAS_IMPORT_ERROR = exc
else:
    _GEOPANDAS_IMPORT_ERROR = None

from ville_hgnn.utils.io import (
    config_hash,
    file_hash,
    load_config,
    to_jsonable,
    write_json,
    write_markdown,
)
from ville_hgnn.utils.paths import (
    find_repo_root,
    get_nested,
    is_unresolved_value,
    resolve_path,
)


DEFAULT_CONFIG_PATH = "urban_graph_benchmark/configs/mtl_311_water_v0.yaml"
TARGET_CRS_EPSG = 3347
WEB_CRS_EPSG = 4326
DEFAULT_HASH_MAX_BYTES = 50_000_000

MANDATORY_GRID_COLUMNS = {
    "unit_id",
    "period_month",
    "requests_total",
    "water_drainage_requests",
    "x_centroid",
    "y_centroid",
}

GRID_METADATA_COLUMNS = {
    "unit_id",
    "period_month",
    "x_centroid",
    "y_centroid",
    "lat_centroid",
    "lon_centroid",
}

OPTIONAL_SUM_COLUMNS = [
    "requests_total",
    "complaints_total",
    "citizen_requests_total",
    "comments_total",
    "urgent_total",
    "finished_total",
    "other_requests",
    "road_mobility_requests",
    "snow_winter_requests",
    "tree_canopy_requests",
    "waste_cleanliness_requests",
    "water_drainage_requests",
]

OMITTED_GRID_COLUMNS = {
    "unique_activity_count",
    "unique_responsible_units",
}

SHARE_NUMERATOR_MAP = {
    "share_complaints_total": "complaints_total",
    "share_urgent_total": "urgent_total",
    "share_water_drainage_requests": "water_drainage_requests",
    "share_road_mobility_requests": "road_mobility_requests",
    "share_tree_canopy_requests": "tree_canopy_requests",
    "share_snow_winter_requests": "snow_winter_requests",
    "share_waste_cleanliness_requests": "waste_cleanliness_requests",
}

DELAY_COLUMNS = {
    "avg_resolution_delay_hours": "weighted_mean",
    "median_resolution_delay_hours": "weighted_mean_of_grid_medians_not_true_median",
}

COMMON_TRACT_ID_CANDIDATES = [
    "unit_id",
    "zone_id",
    "census_tract_dguid",
    "DGUID",
    "dguid",
    "CTUID",
    "ctuid",
]

COMMON_SVI_ID_CANDIDATES = [
    "unit_id",
    "zone_id",
    "census_tract_dguid",
    "DGUID",
    "dguid",
    "CTUID",
    "ctuid",
]

SVI_RENAME_MAP = {
    "score_normalized_0_1": "svi_score_normalized_0_1",
    "score": "svi_score",
    "score_raw": "svi_score_raw",
    "score_z": "svi_score_z",
    "percentile": "svi_percentile",
    "rank": "svi_rank",
    "class": "svi_class",
    "quality_flag": "svi_quality_flag",
    "missing_count": "svi_missing_count",
}


class DatasetBuildError(RuntimeError):
    """Raised when Dataset v0 cannot be built safely."""


@dataclass(frozen=True)
class BuildOutputs:
    """Paths to Dataset v0 output artifacts."""

    tract_month_panel: Path
    tract_static_features: Path
    target_water_drainage: Path
    dataset_validation: Path
    dataset_report: Path
    spatial_join_audit: Path
    missingness_report: Path
    feature_dictionary: Path
    provenance: Path

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in self.__dict__.items()}


def require_runtime_dependencies() -> None:
    """Fail clearly if required geospatial/dataframe dependencies are missing."""

    if pd is None:
        raise DatasetBuildError(
            "pandas is required to build Dataset v0. Install pandas first."
        ) from _PANDAS_IMPORT_ERROR

    if gpd is None:
        raise DatasetBuildError(
            "geopandas is required to build Dataset v0. Install geopandas first."
        ) from _GEOPANDAS_IMPORT_ERROR


def normalized_column_lookup(columns: Iterable[str]) -> dict[str, str]:
    """Map lowercase column names to original names."""

    return {str(col).strip().lower(): str(col) for col in columns}


def choose_column(
    columns: Iterable[str],
    candidates: Iterable[str],
    label: str,
    required: bool = True,
) -> str | None:
    """Choose the first matching column from candidate names."""

    lookup = normalized_column_lookup(columns)

    for candidate in candidates:
        if is_unresolved_value(candidate):
            continue
        key = str(candidate).strip().lower()
        if key in lookup:
            return lookup[key]

    if required:
        raise DatasetBuildError(
            f"Could not find required {label} column. "
            f"Tried candidates: {list(candidates)}. "
            f"Available columns: {list(columns)}"
        )

    return None

def canonicalize_tract_unit_id(value: Any) -> str:
    """
    Normalize census tract unit IDs for stable joins.

    Examples:
        4620001.0  -> 4620001.00
        4620001.00 -> 4620001.00
        4620107.02 -> 4620107.02

    This is needed because pandas may read tract identifiers from CSV files
    as numeric values and collapse trailing zeroes.
    """

    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text == "":
        return ""

    try:
        return f"{float(text):.2f}"
    except Exception:
        return text


def resolve_required_config_path(
    config: Mapping[str, Any],
    keys: list[str],
    repo_root: Path,
    label: str,
) -> Path:
    """Resolve a required config path and ensure it exists."""

    value = get_nested(config, keys)

    if is_unresolved_value(value):
        raise DatasetBuildError(
            f"Missing required config path for {label}: {'.'.join(keys)} is {value!r}."
        )

    resolved = resolve_path(value, repo_root=repo_root, allow_unresolved=False)

    if resolved is None:
        raise DatasetBuildError(f"Could not resolve required path for {label}: {value!r}")

    if not resolved.exists():
        raise DatasetBuildError(f"Configured path for {label} does not exist: {resolved}")

    return resolved


def resolve_optional_config_path(
    config: Mapping[str, Any],
    keys: list[str],
    repo_root: Path,
) -> Path | None:
    """Resolve an optional config path if it is present and not unresolved."""

    value = get_nested(config, keys)

    if is_unresolved_value(value):
        return None

    resolved = resolve_path(value, repo_root=repo_root, allow_unresolved=True)

    if resolved is None or not resolved.exists():
        return None

    return resolved


def output_path(
    config: Mapping[str, Any],
    repo_root: Path,
    key: str,
    default_relative_path: str,
) -> Path:
    """Resolve an output path from config expected_output_files."""

    configured = get_nested(config, ["paths", "expected_output_files", key], default=None)
    value = configured if not is_unresolved_value(configured) else default_relative_path
    resolved = resolve_path(value, repo_root=repo_root, allow_unresolved=False)

    if resolved is None:
        raise DatasetBuildError(f"Could not resolve output path for {key}")

    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def get_build_output_paths(config: Mapping[str, Any], repo_root: Path) -> BuildOutputs:
    """Resolve all Dataset v0 output artifact paths."""

    base = "urban_graph_benchmark/outputs/mtl_311_water_v0/datasets"

    return BuildOutputs(
        tract_month_panel=output_path(
            config, repo_root, "tract_month_panel", f"{base}/tract_month_panel.parquet"
        ),
        tract_static_features=output_path(
            config, repo_root, "tract_static_features", f"{base}/tract_static_features.parquet"
        ),
        target_water_drainage=output_path(
            config, repo_root, "target_water_drainage", f"{base}/target_water_drainage.parquet"
        ),
        dataset_validation=output_path(
            config, repo_root, "dataset_validation", f"{base}/dataset_validation.json"
        ),
        dataset_report=output_path(
            config, repo_root, "dataset_report", f"{base}/dataset_report.md"
        ),
        spatial_join_audit=output_path(
            config, repo_root, "spatial_join_audit", f"{base}/spatial_join_audit.csv"
        ),
        missingness_report=output_path(
            config, repo_root, "missingness_report", f"{base}/missingness_report.csv"
        ),
        feature_dictionary=output_path(
            config, repo_root, "feature_dictionary", f"{base}/feature_dictionary.csv"
        ),
        provenance=output_path(
            config, repo_root, "provenance", f"{base}/provenance.json"
        ),
    )


def read_table(path: Path) -> pd.DataFrame:
    """Load a table from parquet, CSV, TSV, JSON, or Excel."""

    require_runtime_dependencies()

    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)

    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if suffix == ".json":
        return pd.read_json(path)

    raise DatasetBuildError(f"Unsupported table format for {path}")


def load_grid311(path: Path) -> pd.DataFrame:
    """Load and validate the grid25m-month 311 table."""

    grid = read_table(path)
    grid.columns = [str(col) for col in grid.columns]

    missing = sorted(MANDATORY_GRID_COLUMNS - set(grid.columns))
    if missing:
        raise DatasetBuildError(
            f"311 grid table is missing required columns: {missing}. "
            f"Available columns: {list(grid.columns)}"
        )

    grid = grid.copy()
    grid["period_month"] = normalize_period_month(grid["period_month"])

    for col in grid.columns:
        if col in GRID_METADATA_COLUMNS:
            continue
        if is_likely_numeric_grid_column(col):
            grid[col] = pd.to_numeric(grid[col], errors="coerce")

    for col in ["x_centroid", "y_centroid", "lat_centroid", "lon_centroid"]:
        if col in grid.columns:
            grid[col] = pd.to_numeric(grid[col], errors="coerce")

    if grid["unit_id"].isna().any():
        raise DatasetBuildError("311 grid table contains missing unit_id values.")

    if grid["period_month"].isna().any():
        raise DatasetBuildError("311 grid table contains period_month values that could not be parsed.")

    return grid


def normalize_period_month(series: pd.Series) -> pd.Series:
    """Normalize period-month values to YYYY-MM strings."""

    parsed = pd.to_datetime(series.astype(str), errors="coerce")
    return parsed.dt.to_period("M").astype(str)


def period_metadata(month_values: Iterable[str]) -> pd.DataFrame:
    """Create period metadata columns for observed YYYY-MM months."""

    periods = pd.PeriodIndex(sorted(set(month_values)), freq="M")
    df = pd.DataFrame({"period_month": periods.astype(str)})
    starts = periods.to_timestamp(how="start")
    ends = periods.to_timestamp(how="end").normalize()

    df["year"] = starts.year.astype(int)
    df["month"] = starts.month.astype(int)
    df["period_start"] = starts.date.astype(str)
    df["period_end"] = ends.date.astype(str)

    return df


def is_likely_numeric_grid_column(column: str) -> bool:
    """Return True for known numeric grid table features."""

    col = column.strip().lower()

    if col in {"period_month", "unit_id"}:
        return False

    if col.startswith("share_"):
        return True

    if col.endswith("_requests"):
        return True

    if col.endswith("_total"):
        return True

    if col.endswith("_hours"):
        return True

    if col in {
        "other_requests",
        "unique_activity_count",
        "unique_responsible_units",
        "x_centroid",
        "y_centroid",
        "lat_centroid",
        "lon_centroid",
    }:
        return True

    return False


def load_tract_geometry(path: Path) -> gpd.GeoDataFrame:
    """Load census tract geometry and reproject to EPSG:3347."""

    require_runtime_dependencies()

    suffix = path.suffix.lower()

    if suffix in {".geojson", ".gpkg", ".shp"}:
        gdf = gpd.read_file(path)
    elif suffix in {".parquet", ".pq"}:
        gdf = gpd.read_parquet(path)
    else:
        raise DatasetBuildError(
            f"Tract geometry must be a spatial file or GeoParquet, got: {path}"
        )

    if "geometry" not in gdf.columns:
        raise DatasetBuildError(f"Tract geometry file has no geometry column: {path}")

    if gdf.crs is None:
        raise DatasetBuildError(
            f"Tract geometry CRS is missing. Expected EPSG:{TARGET_CRS_EPSG}."
        )

    if gdf.crs.to_epsg() != TARGET_CRS_EPSG:
        gdf = gdf.to_crs(epsg=TARGET_CRS_EPSG)

    gdf = gdf.copy()
    gdf.columns = [str(col) for col in gdf.columns]

    return gdf


def choose_tract_id_column(config: Mapping[str, Any], tracts: gpd.GeoDataFrame) -> str:
    """Choose the census tract ID column."""

    configured = get_nested(config, ["inputs", "census_tract_geometry", "id_column"])
    candidates = []

    if not is_unresolved_value(configured):
        candidates.append(str(configured))

    candidates.extend(
        get_nested(
            config,
            ["inputs", "census_tract_geometry", "expected_id_column_candidates"],
            default=[],
        )
    )
    candidates.extend(COMMON_TRACT_ID_CANDIDATES)

    return choose_column(tracts.columns, candidates, label="tract ID", required=True)  # type: ignore[return-value]


def choose_svi_id_column(config: Mapping[str, Any], svi: pd.DataFrame) -> str:
    """Choose the SVI ID column."""

    configured = get_nested(config, ["inputs", "svi", "id_column"])
    candidates = []

    if not is_unresolved_value(configured):
        candidates.append(str(configured))

    candidates.extend(
        get_nested(config, ["inputs", "svi", "expected_id_column_candidates"], default=[])
    )
    candidates.extend(COMMON_SVI_ID_CANDIDATES)

    return choose_column(svi.columns, candidates, label="SVI ID", required=True)  # type: ignore[return-value]


def build_tract_static_features(
    tracts: gpd.GeoDataFrame,
    tract_id_col: str,
) -> pd.DataFrame:
    """Build tract-level static features from geometry."""

    gdf = tracts.copy()
    gdf["zone_id"] = gdf[tract_id_col].map(canonicalize_tract_unit_id)

    if gdf["zone_id"].duplicated().any():
        dupes = gdf.loc[gdf["zone_id"].duplicated(keep=False), "zone_id"].head(20).tolist()
        raise DatasetBuildError(f"Duplicate tract zone_id values in geometry: {dupes}")

    centroids_native = gdf.geometry.centroid
    gdf["tract_centroid_x"] = centroids_native.x
    gdf["tract_centroid_y"] = centroids_native.y

    centroids_web = gpd.GeoSeries(centroids_native, crs=gdf.crs).to_crs(epsg=WEB_CRS_EPSG)
    gdf["tract_centroid_lon"] = centroids_web.x
    gdf["tract_centroid_lat"] = centroids_web.y

    population_col = choose_column(
        gdf.columns,
        ["population_total_2021", "population", "population_total"],
        label="population",
        required=False,
    )
    land_col = choose_column(
        gdf.columns,
        ["land_area_km2", "area_km2", "land_area"],
        label="land area",
        required=False,
    )

    if population_col is None:
        raise DatasetBuildError("Tract geometry/static frame lacks population_total_2021.")
    if land_col is None:
        raise DatasetBuildError("Tract geometry/static frame lacks land_area_km2.")

    gdf["population_total_2021"] = pd.to_numeric(gdf[population_col], errors="coerce")
    gdf["land_area_km2"] = pd.to_numeric(gdf[land_col], errors="coerce")

    density_col = choose_column(
        gdf.columns,
        ["population_density", "population_density_per_km2"],
        label="population density",
        required=False,
    )

    if density_col is not None:
        gdf["population_density"] = pd.to_numeric(gdf[density_col], errors="coerce")
    else:
        gdf["population_density"] = gdf["population_total_2021"] / gdf["land_area_km2"].replace(0, pd.NA)

    gdf["population_weighted_centroid_x"] = pd.NA
    gdf["population_weighted_centroid_y"] = pd.NA
    gdf["population_weighted_centroid_source"] = "planned_v1_not_available"

    optional_identity_cols = [
        "census_tract_dguid",
        "census_tract_name",
        "municipality_name",
        "borough_name",
        "province_code",
        "province_name",
    ]

    output_cols = [
        "zone_id",
        tract_id_col,
        *[col for col in optional_identity_cols if col in gdf.columns and col != tract_id_col],
        "population_total_2021",
        "land_area_km2",
        "population_density",
        "tract_centroid_x",
        "tract_centroid_y",
        "tract_centroid_lon",
        "tract_centroid_lat",
        "population_weighted_centroid_x",
        "population_weighted_centroid_y",
        "population_weighted_centroid_source",
    ]

    static = pd.DataFrame(gdf[output_cols])
    if "census_tract_dguid" not in static.columns:
        static["census_tract_dguid"] = pd.NA

    if "census_tract_unit_id" not in static.columns:
        static["census_tract_unit_id"] = static["zone_id"]

    static = static.loc[:, ~static.columns.duplicated()]

    return static


def load_and_canonicalize_svi(path: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    """Load scored SVI output and canonicalize useful score columns."""

    svi = read_table(path)
    svi.columns = [str(col) for col in svi.columns]

    id_col = choose_svi_id_column(config, svi)

    out = pd.DataFrame()
    out["zone_id"] = svi[id_col].map(canonicalize_tract_unit_id)

    for col in svi.columns:
        if col == id_col:
            continue

        lower = col.strip().lower()

        if lower in {"geometry", "geometry_wkt", "geom", "wkt"}:
            continue

        canonical = SVI_RENAME_MAP.get(col, None)

        if canonical is None:
            if col.startswith("svi_"):
                canonical = col
            elif lower.startswith("theme_"):
                canonical = f"svi_{col}"
            elif lower.startswith("svi_theme_"):
                canonical = col
            elif "score" in lower or "percentile" in lower or lower in {"rank", "class", "quality_flag"}:
                canonical = f"svi_{col}"
            else:
                continue

        canonical = canonical.replace("svi_svi_", "svi_")
        out[canonical] = svi[col]

    if out["zone_id"].duplicated().any():
        dupes = out.loc[out["zone_id"].duplicated(keep=False), "zone_id"].head(20).tolist()
        raise DatasetBuildError(f"Duplicate zone_id values in SVI output: {dupes}")

    return out


def assign_grid_units_to_tracts(
    grid: pd.DataFrame,
    tracts: gpd.GeoDataFrame,
    tract_id_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Assign each distinct grid coordinate row to a census tract.

    Important:
      - The assignment is NOT one row per unit_id.
      - It is one row per distinct coordinate row, usually:
            unit_id + lon_centroid + lat_centroid

    This matters because the same grid25m unit_id can appear with multiple
    centroid rows across months. Some of those coordinate rows can fall into
    different census tracts near tract boundaries.

    Preferred v0 behavior:
      - use lon_centroid / lat_centroid as EPSG:4326
      - reproject points to the tract geometry CRS
      - spatially join to census tract polygons

    x_centroid / y_centroid are kept in the audit, but are not assumed to be
    EPSG:3347 because the observed bounds do not match the tract CRS.
    """

    if "unit_id" not in grid.columns:
        raise DatasetBuildError("Cannot assign grid cells; missing required column: unit_id")

    metadata_cols = [
        col for col in [
            "unit_id",
            "x_centroid",
            "y_centroid",
            "lat_centroid",
            "lon_centroid",
        ]
        if col in grid.columns
    ]

    grid_units = grid[metadata_cols].copy()

    for col in ["x_centroid", "y_centroid", "lat_centroid", "lon_centroid"]:
        if col in grid_units.columns:
            grid_units[col] = pd.to_numeric(grid_units[col], errors="coerce")

    has_lonlat = {"lon_centroid", "lat_centroid"}.issubset(grid_units.columns)
    has_xy = {"x_centroid", "y_centroid"}.issubset(grid_units.columns)

    if has_lonlat:
        coordinate_key_cols = ["unit_id", "lon_centroid", "lat_centroid"]
        coordinate_source = "lon_lat_centroid_epsg4326_to_tract_crs"
        source_crs = "EPSG:4326"
        target_crs = str(tracts.crs)

        valid_coords = (
            grid_units["lon_centroid"].notna()
            & grid_units["lat_centroid"].notna()
            & grid_units["lon_centroid"].between(-180, 180)
            & grid_units["lat_centroid"].between(-90, 90)
        )

    elif has_xy:
        coordinate_key_cols = ["unit_id", "x_centroid", "y_centroid"]
        coordinate_source = "xy_centroid_assumed_tract_crs_fallback"
        source_crs = str(tracts.crs)
        target_crs = str(tracts.crs)

        valid_coords = (
            grid_units["x_centroid"].notna()
            & grid_units["y_centroid"].notna()
        )

    else:
        raise DatasetBuildError(
            "Cannot assign grid cells. Need either lon_centroid/lat_centroid "
            "or x_centroid/y_centroid."
        )

    # Keep one row per distinct coordinate row, not one row per unit_id.
    coordinate_rows = (
        grid_units
        .drop_duplicates(subset=coordinate_key_cols)
        .sort_values(coordinate_key_cols)
        .reset_index(drop=True)
        .copy()
    )
    coordinate_rows["coordinate_row_id"] = range(len(coordinate_rows))

    if coordinate_source == "lon_lat_centroid_epsg4326_to_tract_crs":
        valid_coordinate_mask = (
            coordinate_rows["lon_centroid"].notna()
            & coordinate_rows["lat_centroid"].notna()
            & coordinate_rows["lon_centroid"].between(-180, 180)
            & coordinate_rows["lat_centroid"].between(-90, 90)
        )
    else:
        valid_coordinate_mask = (
            coordinate_rows["x_centroid"].notna()
            & coordinate_rows["y_centroid"].notna()
        )

    valid_coordinate_rows = coordinate_rows.loc[valid_coordinate_mask].copy()
    missing_coord_audit = coordinate_rows.loc[~valid_coordinate_mask].copy()

    if valid_coordinate_rows.empty:
        raise DatasetBuildError(
            "Cannot assign grid cells. No valid centroid coordinates were available."
        )

    if coordinate_source == "lon_lat_centroid_epsg4326_to_tract_crs":
        points = gpd.GeoDataFrame(
            valid_coordinate_rows.copy(),
            geometry=gpd.points_from_xy(
                valid_coordinate_rows["lon_centroid"],
                valid_coordinate_rows["lat_centroid"],
            ),
            crs="EPSG:4326",
        ).to_crs(tracts.crs)
    else:
        points = gpd.GeoDataFrame(
            valid_coordinate_rows.copy(),
            geometry=gpd.points_from_xy(
                valid_coordinate_rows["x_centroid"],
                valid_coordinate_rows["y_centroid"],
            ),
            crs=tracts.crs,
        )

    points = points.set_index("coordinate_row_id", drop=True)
    points.index.name = "coordinate_row_id"

    tract_lookup = tracts[[tract_id_col, "geometry"]].copy()
    tract_lookup = tract_lookup.rename(columns={tract_id_col: "assigned_zone_id"})
    tract_lookup["assigned_zone_id"] = tract_lookup["assigned_zone_id"].astype(str)

    joined = gpd.sjoin(points, tract_lookup, how="left", predicate="within")

    # Defensive: a point should normally join to at most one tract. If geometry
    # overlaps ever produce duplicates, keep the first and report the count.
    within_duplicate_count = int(joined.index.duplicated().sum())
    if within_duplicate_count:
        joined = joined[~joined.index.duplicated(keep="first")].copy()

    joined["assignment_method"] = joined["assigned_zone_id"].notna().map(
        {True: "centroid_within_polygon", False: "unassigned"}
    )

    unassigned_index = joined.index[joined["assigned_zone_id"].isna()].unique().tolist()
    fallback_duplicate_count = 0

    if unassigned_index:
        fallback_points = points.loc[unassigned_index]
        fallback = gpd.sjoin(fallback_points, tract_lookup, how="left", predicate="intersects")

        if not fallback.empty:
            fallback_duplicate_count = int(fallback.index.duplicated().sum())
            fallback_first = fallback[~fallback.index.duplicated(keep="first")]
            matched = fallback_first["assigned_zone_id"].notna()

            for idx, row in fallback_first.loc[matched].iterrows():
                joined.loc[idx, "assigned_zone_id"] = row["assigned_zone_id"]
                joined.loc[idx, "assignment_method"] = "centroid_intersects_polygon_boundary_fallback"

    if "coordinate_row_id" not in joined.columns:
        joined = joined.reset_index()

    audit = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
    audit["assigned"] = audit["assigned_zone_id"].notna()
    audit["coordinate_source"] = coordinate_source
    audit["coordinate_source_crs"] = source_crs
    audit["coordinate_target_crs"] = target_crs

    if not missing_coord_audit.empty:
        missing_coord_audit["assigned_zone_id"] = pd.NA
        missing_coord_audit["assignment_method"] = "missing_centroid_coordinates"
        missing_coord_audit["assigned"] = False
        missing_coord_audit["coordinate_source"] = coordinate_source
        missing_coord_audit["coordinate_source_crs"] = source_crs
        missing_coord_audit["coordinate_target_crs"] = target_crs

        keep_cols = [
            col for col in [
                "coordinate_row_id",
                "unit_id",
                "x_centroid",
                "y_centroid",
                "lat_centroid",
                "lon_centroid",
                "assigned_zone_id",
                "assignment_method",
                "assigned",
                "coordinate_source",
                "coordinate_source_crs",
                "coordinate_target_crs",
            ]
            if col in missing_coord_audit.columns
        ]

        audit = pd.concat(
            [audit, missing_coord_audit[keep_cols]],
            ignore_index=True,
        )

    # Coordinate-row-level summary.
    total_coordinate_rows = int(len(coordinate_rows))
    assigned_coordinate_rows = int(audit["assigned"].sum())
    unassigned_coordinate_rows = total_coordinate_rows - assigned_coordinate_rows

    # Unit-level summary.
    total_unique_grid_units = int(coordinate_rows["unit_id"].nunique())

    assigned_by_unit = (
        audit.groupby("unit_id")["assigned"]
        .any()
        .reset_index(name="unit_has_any_assignment")
    )
    assigned_unique_grid_units = int(assigned_by_unit["unit_has_any_assignment"].sum())
    unassigned_unique_grid_units = total_unique_grid_units - assigned_unique_grid_units

    coordinate_rows_per_unit = (
        coordinate_rows.groupby("unit_id")
        .size()
        .reset_index(name="n_coordinate_rows")
    )
    units_with_multiple_coords = (
        coordinate_rows_per_unit.loc[
            coordinate_rows_per_unit["n_coordinate_rows"] > 1,
            "unit_id",
        ]
        .astype(str)
        .tolist()
    )

    assigned_tracts_per_unit = (
        audit.loc[audit["assigned"]]
        .groupby("unit_id")["assigned_zone_id"]
        .nunique(dropna=True)
        .reset_index(name="n_assigned_tracts")
    )
    units_with_multiple_assigned_tracts = (
        assigned_tracts_per_unit.loc[
            assigned_tracts_per_unit["n_assigned_tracts"] > 1,
            "unit_id",
        ]
        .astype(str)
        .tolist()
    )

    spatial_join_method = (
        "grid_coordinate_row_centroid_in_polygon_using_lon_lat_epsg4326_reprojected_to_tract_crs"
        if coordinate_source == "lon_lat_centroid_epsg4326_to_tract_crs"
        else "grid_coordinate_row_centroid_in_polygon_using_xy_assumed_tract_crs_fallback"
    )

    summary = {
        "total_unique_grid_units": total_unique_grid_units,
        "assigned_unique_grid_units": assigned_unique_grid_units,
        "unassigned_unique_grid_units": unassigned_unique_grid_units,
        "assignment_success_rate_unique_grid_units": (
            assigned_unique_grid_units / total_unique_grid_units
            if total_unique_grid_units
            else None
        ),

        "total_unique_coordinate_rows": total_coordinate_rows,
        "assigned_unique_coordinate_rows": assigned_coordinate_rows,
        "unassigned_unique_coordinate_rows": unassigned_coordinate_rows,
        "assignment_success_rate_coordinate_rows": (
            assigned_coordinate_rows / total_coordinate_rows
            if total_coordinate_rows
            else None
        ),

        "coordinate_key_cols": coordinate_key_cols,
        "coordinate_source": coordinate_source,
        "coordinate_source_crs": source_crs,
        "coordinate_target_crs": target_crs,
        "spatial_join_method": spatial_join_method,

        "units_with_multiple_coordinate_rows": len(units_with_multiple_coords),
        "units_with_multiple_coordinate_examples": units_with_multiple_coords[:20],
        "units_with_multiple_assigned_tracts": len(units_with_multiple_assigned_tracts),
        "units_with_multiple_assigned_tract_examples": units_with_multiple_assigned_tracts[:20],

        "within_duplicate_assignments": within_duplicate_count,
        "boundary_fallback_duplicate_assignments": fallback_duplicate_count,
        "assignment_method_counts": audit["assignment_method"].value_counts(dropna=False).to_dict(),
    }

    return audit, summary


def weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    """Compute weighted mean with safe fallback."""

    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0)

    valid = numeric_values.notna() & numeric_weights.notna()
    if not valid.any():
        return None

    values_valid = numeric_values[valid]
    weights_valid = numeric_weights[valid]

    total_weight = float(weights_valid.sum())
    if total_weight > 0:
        return float((values_valid * weights_valid).sum() / total_weight)

    return float(values_valid.mean()) if len(values_valid) else None


def aggregate_grid_to_tract_month(
    grid: pd.DataFrame,
    assignment_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """
    Aggregate assigned grid25m-month features to tract × month.

    Important:
      - assignment is merged back at coordinate-row level, not only unit_id.
      - this avoids the "first coordinate wins" problem for grid unit_ids that
        have multiple observed centroids and may cross tract boundaries.
    """

    # Prefer lon/lat coordinate-row assignment, which is the normal Dataset v0 path.
    if {"unit_id", "lon_centroid", "lat_centroid"}.issubset(grid.columns) and {
        "unit_id",
        "lon_centroid",
        "lat_centroid",
    }.issubset(assignment_audit.columns):
        assignment_key_cols = ["unit_id", "lon_centroid", "lat_centroid"]

    # Fallback for projected x/y assignment.
    elif {"unit_id", "x_centroid", "y_centroid"}.issubset(grid.columns) and {
        "unit_id",
        "x_centroid",
        "y_centroid",
    }.issubset(assignment_audit.columns):
        assignment_key_cols = ["unit_id", "x_centroid", "y_centroid"]

    else:
        raise DatasetBuildError(
            "Cannot merge spatial assignments back to grid rows. Expected either "
            "unit_id + lon_centroid + lat_centroid or unit_id + x_centroid + y_centroid "
            "in both grid and assignment_audit."
        )

    grid_for_merge = grid.copy()

    # Ensure numeric coordinate columns have identical dtype/format before merge.
    for col in ["x_centroid", "y_centroid", "lat_centroid", "lon_centroid"]:
        if col in grid_for_merge.columns:
            grid_for_merge[col] = pd.to_numeric(grid_for_merge[col], errors="coerce")
        if col in assignment_audit.columns:
            assignment_audit[col] = pd.to_numeric(assignment_audit[col], errors="coerce")

    assignment_cols = [
        col for col in [
            *assignment_key_cols,
            "coordinate_row_id",
            "assigned_zone_id",
            "assigned",
            "assignment_method",
            "coordinate_source",
            "coordinate_source_crs",
            "coordinate_target_crs",
        ]
        if col in assignment_audit.columns
    ]

    assignment = assignment_audit[assignment_cols].copy()

    duplicate_assignment_keys = int(assignment.duplicated(assignment_key_cols).sum())
    if duplicate_assignment_keys:
        examples = (
            assignment.loc[assignment.duplicated(assignment_key_cols, keep=False), assignment_key_cols]
            .head(20)
            .to_dict(orient="records")
        )
        raise DatasetBuildError(
            "Spatial assignment audit contains duplicate coordinate-key rows. "
            f"Duplicate count: {duplicate_assignment_keys}. Examples: {examples}"
        )

    assigned_grid = grid_for_merge.merge(
        assignment,
        on=assignment_key_cols,
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    rows_without_assignment_record = int((assigned_grid["_merge"] == "left_only").sum())
    assigned_grid = assigned_grid.drop(columns=["_merge"])

    assigned_grid["assigned"] = assigned_grid["assigned"].fillna(False).astype(bool)

    unassigned_grid_month_rows = int((~assigned_grid["assigned"]).sum())
    assigned_grid = assigned_grid[assigned_grid["assigned"]].copy()
    assigned_grid["zone_id"] = assigned_grid["assigned_zone_id"].astype(str)

    if assigned_grid.empty:
        raise DatasetBuildError("No 311 grid-month rows could be assigned to census tracts.")

    sum_cols = [col for col in OPTIONAL_SUM_COLUMNS if col in assigned_grid.columns]

    for col in assigned_grid.columns:
        if col in sum_cols or col in GRID_METADATA_COLUMNS or col in OMITTED_GRID_COLUMNS:
            continue
        lower = col.lower()
        if lower.startswith("share_") or lower.endswith("_hours"):
            continue
        if lower.endswith("_requests") or lower.endswith("_total"):
            sum_cols.append(col)

    sum_cols = list(dict.fromkeys(sum_cols))

    for col in sum_cols:
        assigned_grid[col] = pd.to_numeric(assigned_grid[col], errors="coerce").fillna(0)

    group_cols = ["zone_id", "period_month"]

    aggregated = assigned_grid.groupby(group_cols, as_index=False)[sum_cols].sum(min_count=1)

    active_grid_counts = (
        assigned_grid.groupby(group_cols, as_index=False)["unit_id"]
        .nunique()
        .rename(columns={"unit_id": "active_grid_cell_count"})
    )
    aggregated = aggregated.merge(active_grid_counts, on=group_cols, how="left")

    if "coordinate_row_id" in assigned_grid.columns:
        active_coordinate_counts = (
            assigned_grid.groupby(group_cols, as_index=False)["coordinate_row_id"]
            .nunique()
            .rename(columns={"coordinate_row_id": "active_grid_coordinate_row_count"})
        )
        aggregated = aggregated.merge(active_coordinate_counts, on=group_cols, how="left")

    delay_outputs: list[str] = []
    if "requests_total" in assigned_grid.columns:
        for col, method in DELAY_COLUMNS.items():
            if col not in assigned_grid.columns:
                continue

            output_col = (
                col
                if method == "weighted_mean"
                else f"{col}_grid_weighted_mean_not_true_median"
            )

            delay_df = (
                assigned_grid.groupby(group_cols)
                .apply(lambda part, value_col=col: weighted_mean(part[value_col], part["requests_total"]))
                .reset_index(name=output_col)
            )
            aggregated = aggregated.merge(delay_df, on=group_cols, how="left")
            delay_outputs.append(output_col)

    recomputed_share_cols: list[str] = []
    if "requests_total" in aggregated.columns:
        denom = aggregated["requests_total"].replace(0, pd.NA)
        for share_col, numerator_col in SHARE_NUMERATOR_MAP.items():
            if numerator_col in aggregated.columns:
                aggregated[share_col] = (aggregated[numerator_col] / denom).fillna(0)
                recomputed_share_cols.append(share_col)

    omitted_present = sorted([col for col in OMITTED_GRID_COLUMNS if col in assigned_grid.columns])

    unstable_unit_count = None
    if "assigned_zone_id" in assignment_audit.columns:
        unstable_unit_count = int(
            assignment_audit.loc[assignment_audit["assigned"]]
            .groupby("unit_id")["assigned_zone_id"]
            .nunique(dropna=True)
            .gt(1)
            .sum()
        )

    aggregation_summary = {
        "assignment_merge_key_cols": assignment_key_cols,
        "rows_without_assignment_record_after_coordinate_merge": rows_without_assignment_record,
        "assigned_grid_month_rows_used": int(len(assigned_grid)),
        "unassigned_grid_month_rows_excluded": unassigned_grid_month_rows,
        "grid_units_with_multiple_assigned_tracts": unstable_unit_count,
        "sum_columns": sum_cols,
        "delay_columns_aggregated": delay_outputs,
        "share_columns_recomputed": recomputed_share_cols,
        "omitted_grid_columns": omitted_present,
        "omitted_grid_column_reason": (
            "Exact tract-month recomputation is impossible from grid-level aggregates; "
            "do not sum unique counts in v0."
        ),
    }

    feature_rows = build_aggregation_feature_dictionary(
        sum_cols=sum_cols,
        delay_outputs=delay_outputs,
        share_cols=recomputed_share_cols,
        omitted_cols=omitted_present,
    )

    return aggregated, aggregation_summary, feature_rows


def build_aggregation_feature_dictionary(
    sum_cols: list[str],
    delay_outputs: list[str],
    share_cols: list[str],
    omitted_cols: list[str],
) -> pd.DataFrame:
    """Create feature dictionary rows for aggregated 311 features."""

    rows: list[dict[str, Any]] = []

    for col in sum_cols:
        rows.append(
            {
                "column": col,
                "role": "aggregated_311_count_or_sum",
                "source": "grid25m_month_311",
                "aggregation": "sum_over_grid_cells_within_tract_month",
                "description": f"Tract-month sum of grid-level {col}.",
                "included_in_panel": True,
            }
        )

    for col in delay_outputs:
        rows.append(
            {
                "column": col,
                "role": "aggregated_311_delay",
                "source": "grid25m_month_311",
                "aggregation": "weighted_mean_by_requests_total_when_possible",
                "description": (
                    "Tract-month delay feature aggregated from grid-level delay values. "
                    "Median-derived output is not a true tract-month median."
                ),
                "included_in_panel": True,
            }
        )

    for col in share_cols:
        rows.append(
            {
                "column": col,
                "role": "recomputed_311_share",
                "source": "grid25m_month_311",
                "aggregation": "recomputed_from_tract_month_numerator_divided_by_requests_total",
                "description": f"Tract-month recomputed share {col}.",
                "included_in_panel": True,
            }
        )

    for col in omitted_cols:
        rows.append(
            {
                "column": col,
                "role": "omitted_grid_unique_count",
                "source": "grid25m_month_311",
                "aggregation": "omitted",
                "description": (
                    "Omitted from Dataset v0 because exact tract-month unique-count "
                    "recomputation is impossible from pre-aggregated grid features."
                ),
                "included_in_panel": False,
            }
        )

    rows.append(
        {
            "column": "active_grid_cell_count",
            "role": "aggregation_diagnostic",
            "source": "grid25m_month_311",
            "aggregation": "nunique_grid_units_with_records_in_tract_month",
            "description": "Number of distinct grid25m cells contributing rows to a tract-month.",
            "included_in_panel": True,
        }
    )

    return pd.DataFrame(rows)


def build_complete_panel(
    aggregated: pd.DataFrame,
    static_features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build complete tract × month panel and zero-fill target/count columns."""

    months = sorted(aggregated["period_month"].dropna().astype(str).unique().tolist())
    zones = sorted(static_features["zone_id"].dropna().astype(str).unique().tolist())

    if not months:
        raise DatasetBuildError("No period_month values found after aggregation.")
    if not zones:
        raise DatasetBuildError("No in-scope zones found for panel construction.")

    base = pd.MultiIndex.from_product([zones, months], names=["zone_id", "period_month"]).to_frame(index=False)
    panel = base.merge(aggregated, on=["zone_id", "period_month"], how="left")

    zero_fill_cols = [
        col for col in panel.columns
        if col.endswith("_requests") or col.endswith("_total") or col in {"active_grid_cell_count", "other_requests"}
    ]

    zero_filled_rows = int(panel["requests_total"].isna().sum()) if "requests_total" in panel.columns else 0

    for col in zero_fill_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0)

    share_cols = [col for col in panel.columns if col.startswith("share_")]
    for col in share_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0)

    periods = period_metadata(months)
    panel = panel.merge(periods, on="period_month", how="left", validate="many_to_one")
    panel = panel.merge(static_features, on="zone_id", how="left", validate="many_to_one")

    summary = {
        "n_zones": len(zones),
        "n_months": len(months),
        "expected_rows": len(zones) * len(months),
        "actual_rows": len(panel),
        "zero_filled_tract_month_rows": zero_filled_rows,
        "period_month_min": min(months),
        "period_month_max": max(months),
    }

    return panel, summary


def add_targets_and_reporting_controls(
    panel: pd.DataFrame,
    source_water_col: str = "water_drainage_requests",
    source_total_col: str = "requests_total",
) -> pd.DataFrame:
    """Create canonical target and reporting-control columns."""

    if source_water_col not in panel.columns:
        raise DatasetBuildError(f"Missing water/drainage source count column: {source_water_col}")

    if source_total_col not in panel.columns:
        raise DatasetBuildError(f"Missing total 311 source count column: {source_total_col}")

    out = panel.copy()
    out["water_drainage_count"] = pd.to_numeric(out[source_water_col], errors="coerce").fillna(0)
    out["water_drainage_binary"] = (out["water_drainage_count"] > 0).astype(int)
    out["total_311_count_all"] = pd.to_numeric(out[source_total_col], errors="coerce").fillna(0)
    out["total_311_count_non_water_drainage"] = out["total_311_count_all"] - out["water_drainage_count"]

    near_zero_negative = (
        (out["total_311_count_non_water_drainage"] < 0)
        & (out["total_311_count_non_water_drainage"] > -1e-9)
    )
    out.loc[near_zero_negative, "total_311_count_non_water_drainage"] = 0

    return out


def select_in_scope_static_features(
    static_features: pd.DataFrame,
    assignment_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select v0 in-scope tracts using assigned 311 grid cells as service-territory proxy."""

    assigned_zone_ids = sorted(
        assignment_audit.loc[assignment_audit["assigned"], "assigned_zone_id"]
        .dropna().astype(str).unique().tolist()
    )

    scoped = static_features[static_features["zone_id"].astype(str).isin(assigned_zone_ids)].copy()
    missing_static = sorted(set(assigned_zone_ids) - set(scoped["zone_id"].astype(str)))

    if missing_static:
        raise DatasetBuildError(
            "Some assigned 311 grid units map to tracts missing from static features: "
            f"{missing_static[:20]}"
        )

    summary = {
        "study_area_rule_v0": "tracts_with_at_least_one_assigned_311_grid25m_unit",
        "n_assigned_zone_ids": len(assigned_zone_ids),
        "n_static_tracts_in_scope": len(scoped),
        "note": (
            "This is a v0 empirical service-territory proxy. It does not yet use "
            "a formal Ville de Montréal/agglomeration service boundary."
        ),
    }

    return scoped, summary


def make_missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Create missingness report for a dataframe."""

    rows = []
    n = len(df)

    for col in df.columns:
        missing = int(df[col].isna().sum())
        non_missing = n - missing
        rows.append(
            {
                "column": col,
                "non_missing": non_missing,
                "missing": missing,
                "missing_pct": missing / n if n else None,
                "dtype": str(df[col].dtype),
            }
        )

    return pd.DataFrame(rows)


def add_static_feature_dictionary_rows(feature_dict: pd.DataFrame, static_features: pd.DataFrame) -> pd.DataFrame:
    """Append feature dictionary rows for static/SVI/target columns."""

    rows: list[dict[str, Any]] = []

    for col in static_features.columns:
        if col == "zone_id":
            role = "identifier"
        elif col.startswith("svi_"):
            role = "svi_scored_baseline_or_context"
        elif "centroid" in col:
            role = "tract_representative_point"
        elif col in {"population_total_2021", "land_area_km2", "population_density"}:
            role = "static_geographic_control"
        else:
            role = "static_tract_attribute"

        rows.append(
            {
                "column": col,
                "role": role,
                "source": "tract_geometry_or_svi",
                "aggregation": "one_to_one_tract_join",
                "description": f"Static tract feature {col}.",
                "included_in_panel": True,
            }
        )

    rows.extend(
        [
            {
                "column": "water_drainage_count",
                "role": "target_count",
                "source": "derived_from_aggregated_water_drainage_requests",
                "aggregation": "tract_month_sum",
                "description": "Official raw count target for Dataset v0.",
                "included_in_panel": True,
            },
            {
                "column": "water_drainage_binary",
                "role": "target_binary",
                "source": "water_drainage_count",
                "aggregation": "water_drainage_count > 0",
                "description": "Binary target indicating any reported water/drainage request.",
                "included_in_panel": True,
            },
            {
                "column": "total_311_count_all",
                "role": "reporting_control_retrospective_only",
                "source": "requests_total",
                "aggregation": "tract_month_sum",
                "description": "Total 311 requests, contains target and should not be used for forecasting.",
                "included_in_panel": True,
            },
            {
                "column": "total_311_count_non_water_drainage",
                "role": "preferred_reporting_control_retrospective_only",
                "source": "requests_total_minus_water_drainage_count",
                "aggregation": "tract_month_difference",
                "description": "Preferred same-month reporting-control proxy for retrospective models.",
                "included_in_panel": True,
            },
        ]
    )

    return pd.concat([feature_dict, pd.DataFrame(rows)], ignore_index=True)


def validate_dataset(
    panel: pd.DataFrame,
    static_features: pd.DataFrame,
    target_table: pd.DataFrame,
    assignment_summary: Mapping[str, Any],
    svi_join_summary: Mapping[str, Any],
    panel_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Run required Dataset v0 validation checks."""

    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, severity: str = "error", details: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "severity": severity, "details": to_jsonable(details)})

    duplicate_rows = int(panel.duplicated(["zone_id", "period_month"]).sum())
    add_check("one_row_per_zone_month", duplicate_rows == 0, details={"duplicate_rows": duplicate_rows})

    expected_rows = panel_summary.get("expected_rows")
    actual_rows = len(panel)
    add_check(
        "all_expected_tracts_represented_in_every_month",
        expected_rows == actual_rows,
        details={"expected_rows": expected_rows, "actual_rows": actual_rows},
    )

    target_missing = int(panel["water_drainage_count"].isna().sum())
    add_check("zero_filled_missing_target_rows", target_missing == 0, details={"missing_water_drainage_count": target_missing})

    negative_water = int((panel["water_drainage_count"] < 0).sum())
    add_check("water_drainage_count_nonnegative", negative_water == 0, details={"negative_rows": negative_water})

    negative_non_water = int((panel["total_311_count_non_water_drainage"] < 0).sum())
    add_check(
        "total_311_count_non_water_drainage_nonnegative",
        negative_non_water == 0,
        details={"negative_rows": negative_non_water},
    )

    assignment_rate = assignment_summary.get("assignment_success_rate_unique_grid_units")
    add_check("spatial_assignment_success_rate_reported", assignment_rate is not None, details=assignment_summary)

    unassigned = assignment_summary.get("unassigned_unique_grid_units", 0)
    add_check(
        "unassigned_grid_cells_reported",
        True,
        severity="warning" if unassigned else "info",
        details={"unassigned_unique_grid_units": unassigned},
    )

    svi_rate = svi_join_summary.get("svi_join_success_rate")
    add_check("svi_join_success_rate_reported", svi_rate is not None, details=svi_join_summary)

    missing_svi = svi_join_summary.get("missing_svi_rows", 0)
    add_check("svi_join_complete_for_in_scope_tracts", missing_svi == 0, severity="warning", details=svi_join_summary)

    sovi_cols = [col for col in panel.columns if "sovi" in col.lower()]
    add_check("no_sovi_columns_in_track_a", len(sovi_cols) == 0, details={"sovi_like_columns": sovi_cols})

    add_check("no_missing_zone_id", int(panel["zone_id"].isna().sum()) == 0, details={"missing_zone_id_rows": int(panel["zone_id"].isna().sum())})

    add_check(
        "target_table_row_count_matches_panel",
        len(target_table) == len(panel),
        details={"target_rows": len(target_table), "panel_rows": len(panel)},
    )

    hard_failures = [check for check in checks if not check["passed"] and check["severity"] == "error"]
    warnings = [check for check in checks if not check["passed"] and check["severity"] == "warning"]

    return {
        "overall_status": "fail" if hard_failures else ("warning" if warnings else "pass"),
        "checks": checks,
        "hard_failure_count": len(hard_failures),
        "warning_count": len(warnings),
        "summary": {
            "panel_rows": len(panel),
            "static_feature_rows": len(static_features),
            "target_rows": len(target_table),
            "zone_count": int(panel["zone_id"].nunique()),
            "month_count": int(panel["period_month"].nunique()),
        },
    }


def maybe_hash(path: Path, max_bytes: int = DEFAULT_HASH_MAX_BYTES) -> str | None:
    """Hash file only if small enough."""

    try:
        size = path.stat().st_size
    except OSError:
        return None

    if size > max_bytes or not path.is_file():
        return None

    return file_hash(path)


def build_provenance(
    config: Mapping[str, Any],
    config_path: Path,
    repo_root: Path,
    inputs: Mapping[str, Path | None],
    row_counts: Mapping[str, Any],
    assignment_summary: Mapping[str, Any],
    panel_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build provenance dictionary for Dataset v0."""

    input_records = {}
    for key, path in inputs.items():
        if path is None:
            input_records[key] = None
            continue

        input_records[key] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
            "sha256_if_small": maybe_hash(path),
        }

    return {
        "benchmark_id": config.get("benchmark_id"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "config_path": str(config_path),
        "config_hash": config_hash(config),
        "input_paths": input_records,
        "row_counts": to_jsonable(row_counts),
        "study_area_definition": {
            "v0_rule": "tracts_with_at_least_one_assigned_311_grid25m_unit",
            "formal_service_territory": "not_available_v0",
        },
        "crs_information": {
            "spatial_join_crs": f"EPSG:{TARGET_CRS_EPSG}",
            "web_centroid_crs": f"EPSG:{WEB_CRS_EPSG}",
        },
        "spatial_join_method": assignment_summary.get(
                "spatial_join_method",
                "grid_centroid_in_polygon"
            ),
        "target_category_selection_method": "precomputed_aggregated_columns",
        "assignment_summary": to_jsonable(assignment_summary),
        "panel_summary": to_jsonable(panel_summary),
    }


def render_dataset_report(
    validation: Mapping[str, Any],
    provenance: Mapping[str, Any],
    assignment_summary: Mapping[str, Any],
    aggregation_summary: Mapping[str, Any],
    panel_summary: Mapping[str, Any],
    svi_join_summary: Mapping[str, Any],
    outputs: BuildOutputs,
) -> str:
    """Render Dataset v0 report as Markdown."""

    lines: list[str] = []

    lines.append("# Dataset Report — Montréal 311 Water/Drainage v0\n")
    lines.append(f"Generated at: `{provenance.get('generated_at')}`\n")
    lines.append(f"Benchmark ID: `{provenance.get('benchmark_id')}`\n")
    lines.append(f"Validation status: `{validation.get('overall_status')}`\n")

    lines.append("## Dataset scope\n")
    lines.append(
        "Dataset v0 uses the derived grid25m-month 311 table and assigns grid-cell "
        "centroids to census tracts using lon/lat centroids in EPSG:4326 reprojected "
        "to the tract CRS before centroid-in-polygon. The in-scope "
        "tract set is the set of tracts receiving at least one assigned 311 grid cell. "
        "This is a v0 empirical 311 service-territory proxy, not a formal service-boundary definition.\n"
    )
    lines.append(
        "Population-weighted centroids and road-network accessibility features are not computed in v0. "
        "Null population-weighted centroid columns are reserved for v1.\n"
    )

    lines.append("## Panel summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key, value in panel_summary.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")

    lines.append("## Spatial assignment summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key, value in assignment_summary.items():
        if isinstance(value, (list, dict)):
            continue
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")

    lines.append("## Aggregation summary\n")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    for key in [
        "assigned_grid_month_rows_used",
        "unassigned_grid_month_rows_excluded",
        "sum_columns",
        "delay_columns_aggregated",
        "share_columns_recomputed",
        "omitted_grid_columns",
    ]:
        lines.append(f"| `{key}` | `{aggregation_summary.get(key)}` |")
    lines.append("")

    lines.append("## SVI join summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key, value in svi_join_summary.items():
        if isinstance(value, (list, dict)):
            continue
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")

    lines.append("## Validation checks\n")
    lines.append("| Check | Passed | Severity | Details |")
    lines.append("|---|:---:|---|---|")
    for check in validation.get("checks", []):
        details = json.dumps(check.get("details"), ensure_ascii=False)[:300]
        lines.append(
            f"| `{check.get('name')}` | `{check.get('passed')}` | "
            f"`{check.get('severity')}` | `{details}` |"
        )
    lines.append("")

    lines.append("## Output artifacts\n")
    lines.append("| Artifact | Path |")
    lines.append("|---|---|")
    for key, value in outputs.to_dict().items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")

    lines.append("## Notes and limitations\n")
    lines.append(
        "- The target is a reported municipal 311 disruption signal, not objective flood occurrence.\n"
        "- `total_311_count_all` contains the water/drainage target and is retrospective-only.\n"
        "- `total_311_count_non_water_drainage` is the preferred same-month reporting-control proxy for retrospective models.\n"
        "- Official magnitude classes are not generated here; they must be split-specific in the modeling pipeline.\n"
        "- Unique activity/responsible-unit counts are omitted from v0 because exact tract-month recomputation is impossible from grid-level aggregates.\n"
        "- Road-network travel distances, OSM routing, population-weighted centroids, and accessibility features are planned for later modules.\n"
    )

    return "\n".join(lines)


def write_dataframe_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write dataframe to parquet, creating parent dirs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def run_build_dataset(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build Dataset v0 and write all expected artifacts."""

    require_runtime_dependencies()

    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    resolved_config_path = resolve_path(config_path, repo_root=root, allow_unresolved=False)

    if resolved_config_path is None:
        raise DatasetBuildError(f"Could not resolve config path: {config_path}")

    config = load_config(resolved_config_path)
    outputs = get_build_output_paths(config, root)

    grid_path = resolve_required_config_path(config, ["inputs", "montreal_311", "raw_path"], root, "grid25m-month 311 table")
    tract_path = resolve_required_config_path(config, ["inputs", "census_tract_geometry", "path"], root, "census tract geometry")
    svi_path = resolve_required_config_path(config, ["inputs", "svi", "path"], root, "scored SVI output")
    upstream_raw_path = resolve_optional_config_path(config, ["inputs", "montreal_311", "upstream_raw_source", "path"], root)

    source_type = get_nested(config, ["inputs", "montreal_311", "source_type"])
    if source_type != "grid25m_month":
        raise DatasetBuildError(
            "Dataset v0 expects inputs.montreal_311.source_type = grid25m_month. "
            f"Current value: {source_type!r}"
        )

    spatial_method = get_nested(config, ["spatial_assignment", "method"])
    if spatial_method != "grid_centroid_in_polygon":
        raise DatasetBuildError(
            "Dataset v0 expects spatial_assignment.method = grid_centroid_in_polygon. "
            f"Current value: {spatial_method!r}"
        )

    category_method = get_nested(config, ["target", "category_selection", "method"])
    if category_method != "precomputed_aggregated_columns":
        raise DatasetBuildError(
            "Dataset v0 expects target.category_selection.method = precomputed_aggregated_columns. "
            f"Current value: {category_method!r}"
        )

    grid = load_grid311(grid_path)
    tracts = load_tract_geometry(tract_path)
    tract_id_col = choose_tract_id_column(config, tracts)
    static_all = build_tract_static_features(tracts, tract_id_col)

    assignment_audit, assignment_summary = assign_grid_units_to_tracts(grid=grid, tracts=tracts, tract_id_col=tract_id_col)
    static_scoped, scope_summary = select_in_scope_static_features(static_features=static_all, assignment_audit=assignment_audit)

    aggregated, aggregation_summary, feature_dict = aggregate_grid_to_tract_month(grid=grid, assignment_audit=assignment_audit)
    aggregated = aggregated[aggregated["zone_id"].isin(static_scoped["zone_id"])].copy()

    panel, panel_summary = build_complete_panel(aggregated=aggregated, static_features=static_scoped)

    source_water_col = get_nested(config, ["target", "columns", "count"], default="water_drainage_requests")
    if source_water_col == "water_drainage_count":
        source_water_col = "water_drainage_requests"

    panel = add_targets_and_reporting_controls(panel, source_water_col=source_water_col, source_total_col="requests_total")

    svi = load_and_canonicalize_svi(svi_path, config)
    before_svi = len(panel)
    panel = panel.merge(svi, on="zone_id", how="left", validate="many_to_one")
    after_svi = len(panel)
    static_features = static_scoped.merge(svi, on="zone_id", how="left", validate="one_to_one")

    svi_value_cols = [col for col in svi.columns if col != "zone_id"]
    if svi_value_cols:
        missing_svi_static = int(static_features[svi_value_cols].isna().all(axis=1).sum())
    else:
        missing_svi_static = len(static_features)

    svi_join_summary = {
        "panel_rows_before_svi_join": before_svi,
        "panel_rows_after_svi_join": after_svi,
        "static_tracts_in_scope": len(static_scoped),
        "svi_rows": len(svi),
        "matched_static_tracts": int(len(static_features) - missing_svi_static),
        "missing_svi_rows": missing_svi_static,
        "svi_join_success_rate": ((len(static_features) - missing_svi_static) / len(static_features) if len(static_features) else None),
        "svi_columns_joined": svi_value_cols,
    }

    target_cols = [
        "zone_id",
        "period_month",
        "year",
        "month",
        "period_start",
        "period_end",
        "water_drainage_count",
        "water_drainage_binary",
        "total_311_count_all",
        "total_311_count_non_water_drainage",
    ]
    target_table = panel[target_cols].copy()

    feature_dict = add_static_feature_dictionary_rows(feature_dict, static_features)
    missingness = make_missingness_report(panel)

    validation = validate_dataset(
        panel=panel,
        static_features=static_features,
        target_table=target_table,
        assignment_summary=assignment_summary,
        svi_join_summary=svi_join_summary,
        panel_summary=panel_summary,
    )

    row_counts = {
        "grid311_rows": len(grid),
        "tract_geometry_rows": len(tracts),
        "static_all_rows": len(static_all),
        "static_in_scope_rows": len(static_scoped),
        "aggregated_tract_month_rows_before_complete_panel": len(aggregated),
        "panel_rows": len(panel),
        "target_rows": len(target_table),
        "svi_rows": len(svi),
    }

    provenance = build_provenance(
        config=config,
        config_path=resolved_config_path,
        repo_root=root,
        inputs={
            "grid25m_month_311": grid_path,
            "upstream_raw_311": upstream_raw_path,
            "census_tract_geometry": tract_path,
            "svi_scored_output": svi_path,
        },
        row_counts=row_counts,
        assignment_summary={**assignment_summary, **scope_summary},
        panel_summary=panel_summary,
    )

    report = render_dataset_report(
        validation=validation,
        provenance=provenance,
        assignment_summary={**assignment_summary, **scope_summary},
        aggregation_summary=aggregation_summary,
        panel_summary=panel_summary,
        svi_join_summary=svi_join_summary,
        outputs=outputs,
    )

    write_dataframe_parquet(panel, outputs.tract_month_panel)
    write_dataframe_parquet(static_features, outputs.tract_static_features)
    write_dataframe_parquet(target_table, outputs.target_water_drainage)

    assignment_audit.to_csv(outputs.spatial_join_audit, index=False)
    missingness.to_csv(outputs.missingness_report, index=False)
    feature_dict.to_csv(outputs.feature_dictionary, index=False)

    write_json(outputs.dataset_validation, validation)
    write_json(outputs.provenance, provenance)
    write_markdown(outputs.dataset_report, report)

    result = {
        "status": validation.get("overall_status"),
        "outputs": outputs.to_dict(),
        "validation": validation,
        "panel_summary": panel_summary,
        "assignment_summary": assignment_summary,
        "aggregation_summary": aggregation_summary,
        "svi_join_summary": svi_join_summary,
        "row_counts": row_counts,
    }

    if validation.get("overall_status") == "fail":
        hard_failures = [
            check for check in validation.get("checks", [])
            if not check.get("passed") and check.get("severity") == "error"
        ]
        raise DatasetBuildError(
            "Dataset v0 build completed artifact writing but failed validation. "
            f"Hard failures: {hard_failures}"
        )

    return result


def build_brief(result: Mapping[str, Any]) -> str:
    """Return concise build summary."""

    panel = result.get("panel_summary", {})
    assignment = result.get("assignment_summary", {})
    svi = result.get("svi_join_summary", {})

    return (
        "Dataset v0 build completed.\n"
        f"Status: {result.get('status')}\n"
        f"Rows: {panel.get('actual_rows')}\n"
        f"Zones: {panel.get('n_zones')}\n"
        f"Months: {panel.get('n_months')}\n"
        f"Period: {panel.get('period_month_min')} to {panel.get('period_month_max')}\n"
        f"Grid assignment success: {assignment.get('assignment_success_rate_unique_grid_units')}\n"
        f"SVI join success: {svi.get('svi_join_success_rate')}\n"
    )


def main() -> None:
    """CLI entry point for direct module execution."""

    import argparse

    parser = argparse.ArgumentParser(description="Build Dataset v0 for the Montréal 311 water/drainage benchmark.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help=f"Config path. Default: {DEFAULT_CONFIG_PATH}")
    parser.add_argument("--repo-root", default=None, help="Repository root. Defaults to automatic detection.")
    args = parser.parse_args()

    result = run_build_dataset(config_path=args.config, repo_root=args.repo_root)
    print(build_brief(result).rstrip())
    print("\nWritten outputs:")
    for label, path in result.get("outputs", {}).items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "BuildOutputs",
    "DatasetBuildError",
    "DEFAULT_CONFIG_PATH",
    "TARGET_CRS_EPSG",
    "WEB_CRS_EPSG",
    "add_targets_and_reporting_controls",
    "assign_grid_units_to_tracts",
    "build_brief",
    "build_complete_panel",
    "load_grid311",
    "load_tract_geometry",
    "run_build_dataset",
]