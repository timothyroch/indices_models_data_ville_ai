"""
Input inventory for the Montréal 311 water/drainage benchmark.

This module performs a lightweight, non-destructive inventory of the inputs
needed for ``mtl_311_water_v0``. It does not build the benchmark dataset.

Main responsibilities:

- load the benchmark YAML config
- create configured output directories
- locate candidate input files from explicit paths, candidate paths, and
  recursive search blocks
- inspect candidate schemas without heavy full-file processing
- infer likely 311 source type: point records, grid25m-month, pre-aggregated,
  or unknown
- inspect tract geometry candidates
- inspect SVI output candidates
- test basic SVI ↔ tract-geometry join feasibility when IDs are available
- write ``input_inventory.json`` and ``input_inventory_report.md``

Dataset-building logic belongs in ``build_tract_month_panel.py``.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None  # type: ignore[assignment]

from ville_hgnn.utils.io import (
    config_hash,
    file_hash,
    load_config,
    to_jsonable,
    write_json,
    write_markdown,
)
from ville_hgnn.utils.paths import (
    collect_candidates_from_config_section,
    ensure_output_directories,
    find_repo_root,
    get_nested,
    is_unresolved_value,
    path_status,
    resolve_path,
)


DEFAULT_CONFIG_PATH = "urban_graph_benchmark/configs/mtl_311_water_v0.yaml"

DEFAULT_SAMPLE_ROWS = 10_000
DEFAULT_UNIQUE_VALUE_LIMIT = 200
DEFAULT_HASH_MAX_BYTES = 50_000_000
DEFAULT_PARQUET_FULL_READ_MAX_BYTES = 100_000_000

CSV_ENCODINGS = (
    "utf-8-sig",
    "utf-8",
    "cp1252",
    "latin1",
)

TABLE_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".txt",
    ".parquet",
    ".pq",
    ".xlsx",
    ".xls",
}

GEO_EXTENSIONS = {
    ".geojson",
    ".json",
    ".gpkg",
    ".shp",
    ".parquet",
}

WATER_KEYWORDS_FALLBACK = (
    "eau",
    "égout",
    "egout",
    "drainage",
    "inondation",
    "refoulement",
    "ruissellement",
    "puisard",
    "catch basin",
    "water",
    "sewer",
    "flood",
    "stormwater",
    "runoff",
)

GRID25M_VALUE_PATTERN = re.compile(
    r"^grid[_-]?25m[_-]?\d+[_-]\d+$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class CandidateInspection:
    """Lightweight candidate-file inspection result."""

    path: str
    exists: bool
    kind: str
    status: str
    error: str | None
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "kind": self.kind,
            "status": self.status,
            "error": self.error,
            "details": to_jsonable(self.details),
        }


def normalize_text(value: Any) -> str:
    """Normalize text for robust column/category matching."""

    if value is None:
        return ""

    text = str(value).strip().lower()
    text = text.replace("\u00a0", " ")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s+", " ", text)

    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))

    return ascii_text


def compact_list(values: Iterable[Any], max_items: int = 30) -> list[Any]:
    """Return a short JSON-friendly list."""

    out: list[Any] = []
    for value in values:
        if len(out) >= max_items:
            break
        out.append(to_jsonable(value))
    return out


def format_ratio(value: Any) -> str:
    """Format a ratio for Markdown tables without crashing on None."""

    if value is None:
        return "NA"

    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def lower_columns(columns: Iterable[Any]) -> dict[str, str]:
    """Map normalized column names to original column names."""

    mapping: dict[str, str] = {}
    for col in columns:
        original = str(col)
        mapping.setdefault(normalize_text(original), original)
    return mapping


def columns_matching(
    columns: Iterable[str],
    exact_candidates: Iterable[str] = (),
    contains_any: Iterable[str] = (),
) -> list[str]:
    """Find columns by normalized exact or substring matching."""

    exact = {normalize_text(value) for value in exact_candidates if not is_unresolved_value(value)}
    contains = [normalize_text(value) for value in contains_any if not is_unresolved_value(value)]

    matches: list[str] = []

    for col in columns:
        normalized = normalize_text(col)

        if exact and normalized in exact:
            matches.append(col)
            continue

        if contains and any(token in normalized for token in contains):
            matches.append(col)

    return matches


def extension_kind(path: Path) -> str:
    """Classify a file path by extension."""

    suffix = path.suffix.lower()

    if suffix in {".geojson", ".gpkg", ".shp"}:
        return "spatial"

    if suffix in TABLE_EXTENSIONS:
        return "table"

    if suffix == ".json":
        return "json_or_geojson"

    return "unknown"


def safe_file_size(path: Path) -> int | None:
    """Return file size when available."""

    try:
        return path.stat().st_size
    except OSError:
        return None


def maybe_hash_small_file(path: Path, max_bytes: int = DEFAULT_HASH_MAX_BYTES) -> str | None:
    """Hash a file only if it is not too large."""

    size = safe_file_size(path)
    if size is None or size > max_bytes or not path.is_file():
        return None
    return file_hash(path)


def detect_csv_dialect(path: Path, encoding: str) -> dict[str, Any]:
    """Best-effort delimiter detection for a CSV-like file."""

    try:
        with path.open("r", encoding=encoding, newline="") as handle:
            sample = handle.read(4096)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    try:
        dialect = csv.Sniffer().sniff(sample)
        return {
            "status": "detected",
            "delimiter": dialect.delimiter,
            "quotechar": dialect.quotechar,
        }
    except Exception:
        return {
            "status": "fallback",
            "delimiter": ",",
            "quotechar": '"',
        }


def read_parquet_sample(path: Path, sample_rows: int = DEFAULT_SAMPLE_ROWS) -> tuple[Any | None, dict[str, Any]]:
    """
    Read a bounded sample from a Parquet file.

    Prefer PyArrow row-batch sampling to avoid loading a large Parquet file
    entirely. If PyArrow is unavailable, only fall back to pandas full-read
    for smaller files.
    """

    if pd is None:
        return None, {
            "status": "error",
            "format": "parquet",
            "error": "pandas is not installed",
        }

    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except Exception as pyarrow_exc:
        size = safe_file_size(path)
        if size is not None and size > DEFAULT_PARQUET_FULL_READ_MAX_BYTES:
            return None, {
                "status": "error",
                "format": "parquet",
                "error": (
                    "pyarrow is unavailable and file is too large for safe full-read "
                    f"({size} bytes > {DEFAULT_PARQUET_FULL_READ_MAX_BYTES} bytes): {pyarrow_exc}"
                ),
            }

        try:
            df = pd.read_parquet(path)
            if len(df) > sample_rows:
                df = df.head(sample_rows)
            return df, {
                "status": "read",
                "format": "parquet",
                "rows_sampled": len(df),
                "sampling_method": "pandas_full_read_small_file",
                "file_size_bytes": size,
            }
        except Exception as exc:
            return None, {
                "status": "error",
                "format": "parquet",
                "error": str(exc),
            }

    try:
        parquet_file = pq.ParquetFile(path)
        batches = []
        rows_collected = 0
        batch_size = min(max(sample_rows, 1), 10_000)

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batches.append(batch)
            rows_collected += batch.num_rows
            if rows_collected >= sample_rows:
                break

        if not batches:
            df = pd.DataFrame()
        else:
            import pyarrow as pa  # type: ignore[import-not-found]

            table = pa.Table.from_batches(batches)
            df = table.to_pandas()
            if len(df) > sample_rows:
                df = df.head(sample_rows)

        return df, {
            "status": "read",
            "format": "parquet",
            "rows_sampled": len(df),
            "sampling_method": "pyarrow_iter_batches",
            "num_row_groups": parquet_file.num_row_groups,
            "schema_columns": parquet_file.schema.names,
        }

    except Exception as exc:
        return None, {
            "status": "error",
            "format": "parquet",
            "error": str(exc),
        }


def read_table_sample(
    path: Path,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
) -> tuple[Any | None, dict[str, Any]]:
    """
    Read a small sample from a tabular file.

    Returns ``(dataframe_or_none, metadata)``. The dataframe is a pandas object
    when pandas is available and the file can be read.
    """

    if pd is None:
        return None, {
            "status": "error",
            "error": "pandas is not installed",
        }

    suffix = path.suffix.lower()

    if suffix in {".csv", ".txt", ".tsv"}:
        delimiter_hint = "\t" if suffix == ".tsv" else None

        last_error = None

        for encoding in CSV_ENCODINGS:
            dialect = detect_csv_dialect(path, encoding=encoding)
            sep = delimiter_hint or dialect.get("delimiter") or ","

            try:
                df = pd.read_csv(
                    path,
                    nrows=sample_rows,
                    dtype=str,
                    encoding=encoding,
                    sep=sep,
                    low_memory=False,
                    on_bad_lines="skip",
                )
                return df, {
                    "status": "read",
                    "format": suffix.lstrip("."),
                    "encoding": encoding,
                    "delimiter": sep,
                    "rows_sampled": len(df),
                    "dialect_status": dialect.get("status"),
                }
            except UnicodeDecodeError:
                continue
            except Exception as exc:
                last_error = str(exc)

        return None, {
            "status": "error",
            "format": suffix.lstrip("."),
            "error": last_error or "could not read CSV sample",
        }

    if suffix in {".parquet", ".pq"}:
        return read_parquet_sample(path, sample_rows=sample_rows)

    if suffix in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(path, nrows=sample_rows, dtype=str)
            return df, {
                "status": "read",
                "format": suffix.lstrip("."),
                "rows_sampled": len(df),
            }
        except Exception as exc:
            return None, {
                "status": "error",
                "format": suffix.lstrip("."),
                "error": str(exc),
            }

    if suffix == ".json":
        try:
            df = pd.read_json(path)
            if len(df) > sample_rows:
                df = df.head(sample_rows)
            return df, {
                "status": "read",
                "format": "json",
                "rows_sampled": len(df),
            }
        except Exception as exc:
            return None, {
                "status": "error",
                "format": "json",
                "error": str(exc),
            }

    return None, {
        "status": "unsupported",
        "format": suffix.lstrip("."),
    }


def read_spatial_sample(path: Path, sample_rows: int = DEFAULT_SAMPLE_ROWS) -> tuple[Any | None, dict[str, Any]]:
    """Read a spatial candidate with GeoPandas when available."""

    if gpd is None:
        return None, {
            "status": "error",
            "error": "geopandas is not installed",
        }

    try:
        gdf = gpd.read_file(path)
        if len(gdf) > sample_rows:
            sample = gdf.head(sample_rows).copy()
        else:
            sample = gdf
        return sample, {
            "status": "read",
            "format": path.suffix.lower().lstrip("."),
            "rows_sampled": len(sample),
            "total_rows_loaded": len(gdf),
            "crs": str(gdf.crs) if gdf.crs is not None else None,
            "has_geometry": "geometry" in gdf.columns,
            "geometry_types_sample": (
                sorted(gdf.geometry.geom_type.dropna().astype(str).unique().tolist())
                if "geometry" in gdf.columns
                else []
            ),
            "bounds": (
                {
                    "minx": float(gdf.total_bounds[0]),
                    "miny": float(gdf.total_bounds[1]),
                    "maxx": float(gdf.total_bounds[2]),
                    "maxy": float(gdf.total_bounds[3]),
                }
                if "geometry" in gdf.columns and not gdf.empty
                else None
            ),
        }
    except Exception as exc:
        return None, {
            "status": "error",
            "format": path.suffix.lower().lstrip("."),
            "error": str(exc),
        }


def infer_id_columns(columns: Iterable[str], expected_candidates: Iterable[str] = ()) -> list[str]:
    """Infer likely spatial ID columns."""

    return columns_matching(
        columns,
        exact_candidates=expected_candidates,
        contains_any=[
            "zone_id",
            "unit_id",
            "dguid",
            "census_tract_dguid",
            "ctuid",
            "geo_uid",
            "geocode",
            "geo code",
            "id",
        ],
    )


def infer_date_columns(columns: Iterable[str]) -> list[str]:
    """Infer likely date/month/time columns."""

    return columns_matching(
        columns,
        contains_any=[
            "date",
            "datetime",
            "time",
            "created",
            "creation",
            "opened",
            "closed",
            "month",
            "mois",
            "periode",
            "period",
            "period_month",
            "year",
            "annee",
            "année",
        ],
    )


def infer_category_columns(columns: Iterable[str]) -> list[str]:
    """Infer likely official category/type columns."""

    return columns_matching(
        columns,
        contains_any=[
            "category",
            "categorie",
            "catégorie",
            "type",
            "nature",
            "service",
            "activity",
            "activite",
            "activité",
            "request",
            "requete",
            "requête",
            "objet",
            "sujet",
            "code",
        ],
    )


def infer_description_columns(columns: Iterable[str]) -> list[str]:
    """Infer likely free-text description columns."""

    return columns_matching(
        columns,
        contains_any=[
            "description",
            "detail",
            "détail",
            "details",
            "comment",
            "commentaire",
            "note",
            "texte",
            "text",
            "summary",
            "resume",
            "résumé",
        ],
    )


def infer_point_geometry_columns(columns: Iterable[str]) -> dict[str, list[str]]:
    """Infer likely point-coordinate columns."""

    latitude = columns_matching(
        columns,
        exact_candidates=["lat", "latitude"],
        contains_any=["latitude", "lat_dd", "lat_centroid"],
    )
    longitude = columns_matching(
        columns,
        exact_candidates=["lon", "lng", "long", "longitude"],
        contains_any=["longitude", "lng", "lon_dd", "lon_centroid"],
    )
    x_cols = columns_matching(
        columns,
        exact_candidates=["x", "x_coord", "coord_x"],
        contains_any=["coord_x", "xcoord", "x_coordinate", "x_centroid"],
    )
    y_cols = columns_matching(
        columns,
        exact_candidates=["y", "y_coord", "coord_y"],
        contains_any=["coord_y", "ycoord", "y_coordinate", "y_centroid"],
    )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "x": x_cols,
        "y": y_cols,
    }


def infer_grid_columns(columns: Iterable[str]) -> list[str]:
    """Infer likely grid/cell columns."""

    return columns_matching(
        columns,
        exact_candidates=[
            "unit_id",
            "grid_cell_id",
            "cell_id",
            "tile_id",
        ],
        contains_any=[
            "grid",
            "grille",
            "cell",
            "cellule",
            "maille",
            "tuile",
            "tile",
            "25m",
            "25 m",
        ],
    )


def infer_count_columns(columns: Iterable[str]) -> list[str]:
    """Infer likely count/value columns."""

    return columns_matching(
        columns,
        exact_candidates=["count", "n", "value", "obs_value"],
        contains_any=[
            "count",
            "nombre",
            "nb_",
            "nbr",
            "n_",
            "value",
            "valeur",
            "obs_value",
            "requests",
            "requetes",
            "requêtes",
            "complaints",
            "urgent",
            "finished",
            "share_",
        ],
    )


def infer_spatial_unit_columns(columns: Iterable[str]) -> list[str]:
    """Infer likely pre-aggregated spatial-unit columns."""

    return columns_matching(
        columns,
        contains_any=[
            "tract",
            "census tract",
            "secteur de recensement",
            "zone",
            "unit_id",
            "borough",
            "arrondissement",
            "municipality",
            "municipalite",
            "municipalité",
            "district",
            "geo",
            "dguid",
        ],
    )


def sample_unique_values(
    df: Any,
    columns: Iterable[str],
    value_limit: int = DEFAULT_UNIQUE_VALUE_LIMIT,
) -> dict[str, dict[str, Any]]:
    """Collect limited unique/top-value summaries for selected columns."""

    if pd is None or df is None:
        return {}

    summaries: dict[str, dict[str, Any]] = {}

    for col in columns:
        if col not in df.columns:
            continue

        series = df[col].dropna().astype(str)
        value_counts = series.value_counts().head(value_limit)

        summaries[col] = {
            "non_missing_sample": int(series.shape[0]),
            "unique_values_sample": int(series.nunique(dropna=True)),
            "top_values": [
                {"value": str(index), "count": int(count)}
                for index, count in value_counts.head(30).items()
            ],
        }

    return summaries


def infer_grid25m_value_hints(
    df: Any,
    columns: Iterable[str],
    max_columns: int = 12,
    max_values_per_column: int = 1000,
) -> dict[str, Any]:
    """
    Detect grid25m-style identifiers from sampled values.

    This is important because aggregated grid data may contain centroid
    coordinates. If centroid columns are treated as point-record coordinates,
    grid-month data can be misclassified as point-level 311 records.
    """

    if pd is None or df is None:
        return {}

    candidate_columns = list(dict.fromkeys(columns))[:max_columns]
    hints: dict[str, Any] = {}

    for col in candidate_columns:
        if col not in df.columns:
            continue

        values = (
            df[col]
            .dropna()
            .astype(str)
            .map(str.strip)
            .head(max_values_per_column)
        )

        if values.empty:
            continue

        match_mask = values.map(lambda value: bool(GRID25M_VALUE_PATTERN.match(value)))
        match_count = int(match_mask.sum())

        if match_count > 0:
            hints[col] = {
                "sample_checked": int(values.shape[0]),
                "grid25m_like_count": match_count,
                "grid25m_like_ratio": match_count / int(values.shape[0]),
                "examples": values[match_mask].drop_duplicates().head(10).tolist(),
            }

    return hints


def keyword_hit_summary(
    df: Any,
    columns: Iterable[str],
    keywords: Iterable[str],
    max_examples: int = 20,
) -> dict[str, Any]:
    """Summarize water/drainage keyword hits in selected columns."""

    if pd is None or df is None:
        return {}

    normalized_keywords = [
        normalize_text(keyword)
        for keyword in keywords
        if keyword is not None and normalize_text(keyword) != ""
    ]

    if not normalized_keywords:
        return {}

    summary: dict[str, Any] = {}

    for col in columns:
        if col not in df.columns:
            continue

        selected = df[col]

        # If duplicate column names exist, pandas may return a DataFrame.
        # For inventory purposes, inspect the first matching column defensively.
        if hasattr(selected, "iloc") and not hasattr(selected, "map"):
            selected = selected.iloc[:, 0]

        raw_series = selected.dropna().astype(str)
        normalized_series = raw_series.map(normalize_text)

        hit_mask = normalized_series.map(
            lambda text: any(keyword in text for keyword in normalized_keywords)
        )

        # Make sure the mask is actually boolean before counting.
        hit_mask = hit_mask.fillna(False).map(bool)
        hit_count = sum(1 for value in hit_mask.tolist() if value)

        hit_values = raw_series[hit_mask]
        top_hits = hit_values.value_counts().head(max_examples)

        summary[col] = {
            "rows_non_missing_sample": int(raw_series.shape[0]),
            "rows_with_keyword_hit_sample": int(hit_count),
            "top_keyword_hit_values": [
                {"value": str(index), "count": int(count)}
                for index, count in top_hits.items()
            ],
        }

    return summary


def has_column_like(columns: Iterable[str], tokens: Iterable[str]) -> bool:
    """Return True if any normalized column contains any normalized token."""

    normalized_columns = [normalize_text(col) for col in columns]
    normalized_tokens = [normalize_text(token) for token in tokens]

    return any(
        token in col
        for col in normalized_columns
        for token in normalized_tokens
    )


def infer_311_source_type(schema: Mapping[str, Any]) -> str:
    """
    Infer whether 311 data appears point-level, grid-level, pre-aggregated, or unknown.

    Grid/pre-aggregated evidence is intentionally checked before point evidence.
    A grid25m-month table may contain centroid coordinates, which should not make
    it a point-record file.
    """

    columns = schema.get("columns", []) or []
    point = schema.get("point_geometry_columns", {})
    has_latlon = bool(point.get("latitude")) and bool(point.get("longitude"))
    has_xy = bool(point.get("x")) and bool(point.get("y"))

    has_grid_columns = bool(schema.get("grid_columns"))
    has_grid_values = bool(schema.get("grid25m_value_hints"))
    has_temporal = bool(schema.get("date_columns"))
    has_spatial_unit = bool(schema.get("spatial_unit_columns"))
    has_count = bool(schema.get("count_columns"))

    has_period_month = has_column_like(columns, ["period_month", "periode_mois"])
    has_request_aggregate_columns = has_column_like(
        columns,
        [
            "requests_total",
            "water_drainage_requests",
            "road_mobility_requests",
            "snow_winter_requests",
            "tree_canopy_requests",
            "waste_cleanliness_requests",
            "avg_resolution_delay",
            "median_resolution_delay",
        ],
    )
    has_centroid_columns = has_column_like(
        columns,
        [
            "x_centroid",
            "y_centroid",
            "lat_centroid",
            "lon_centroid",
        ],
    )

    if has_grid_values:
        return "grid25m_month"

    if has_grid_columns and (has_period_month or has_temporal or has_count):
        return "grid25m_month"

    if has_period_month and has_request_aggregate_columns and has_centroid_columns:
        return "grid25m_month"

    if has_temporal and has_spatial_unit and has_count:
        return "pre_aggregated"

    if has_latlon or has_xy:
        return "point_records"

    return "unknown"


def inspect_table_candidate(
    path: Path,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
) -> CandidateInspection:
    """Inspect a generic tabular candidate."""

    if not path.exists():
        return CandidateInspection(
            path=str(path),
            exists=False,
            kind=extension_kind(path),
            status="missing",
            error=None,
            details={},
        )

    df, read_meta = read_table_sample(path, sample_rows=sample_rows)
    details: dict[str, Any] = {
        "file_size_bytes": safe_file_size(path),
        "small_file_sha256": maybe_hash_small_file(path),
        "read": read_meta,
    }

    if df is None:
        return CandidateInspection(
            path=str(path),
            exists=True,
            kind=extension_kind(path),
            status="read_failed",
            error=read_meta.get("error"),
            details=details,
        )

    columns = [str(col) for col in df.columns]
    details.update(
        {
            "columns": columns,
            "n_columns": len(columns),
            "n_rows_sampled": int(len(df)),
            "id_columns": infer_id_columns(columns),
            "date_columns": infer_date_columns(columns),
            "category_columns": infer_category_columns(columns),
            "description_columns": infer_description_columns(columns),
            "point_geometry_columns": infer_point_geometry_columns(columns),
            "grid_columns": infer_grid_columns(columns),
            "count_columns": infer_count_columns(columns),
            "spatial_unit_columns": infer_spatial_unit_columns(columns),
        }
    )

    return CandidateInspection(
        path=str(path),
        exists=True,
        kind=extension_kind(path),
        status="inspected",
        error=None,
        details=details,
    )


def inspect_311_candidate(
    path: Path,
    keywords: Iterable[str],
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
) -> CandidateInspection:
    """Inspect a Montréal 311 candidate file."""

    base = inspect_table_candidate(path, sample_rows=sample_rows)

    if base.status != "inspected":
        return base

    df, _ = read_table_sample(path, sample_rows=sample_rows)
    details = dict(base.details)

    columns = details.get("columns", [])
    category_columns = details.get("category_columns", [])
    description_columns = details.get("description_columns", [])
    id_columns = details.get("id_columns", [])
    grid_columns = details.get("grid_columns", [])
    spatial_unit_columns = details.get("spatial_unit_columns", [])

    text_columns = list(dict.fromkeys([*category_columns, *description_columns]))

    grid_hint_columns = list(dict.fromkeys([*grid_columns, *id_columns, *spatial_unit_columns]))
    details["grid25m_value_hints"] = infer_grid25m_value_hints(df, grid_hint_columns)
    details["inferred_source_type"] = infer_311_source_type(details)
    details["category_value_summary"] = sample_unique_values(df, category_columns)
    details["water_keyword_hits"] = keyword_hit_summary(df, text_columns, keywords)

    inferred = details["inferred_source_type"]
    if inferred in {"point_records", "grid25m_month"}:
        confidence = "medium"
    elif inferred == "pre_aggregated":
        confidence = "low"
    else:
        confidence = "unknown"

    if details.get("grid25m_value_hints") and inferred == "grid25m_month":
        confidence = "high"

    details["inferred_source_type_confidence"] = confidence

    return CandidateInspection(
        path=base.path,
        exists=base.exists,
        kind=base.kind,
        status=base.status,
        error=base.error,
        details=details,
    )


def inspect_spatial_candidate(
    path: Path,
    expected_id_candidates: Iterable[str] = (),
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
) -> CandidateInspection:
    """Inspect a spatial candidate file."""

    if not path.exists():
        return CandidateInspection(
            path=str(path),
            exists=False,
            kind=extension_kind(path),
            status="missing",
            error=None,
            details={},
        )

    suffix = path.suffix.lower()

    if suffix in {".geojson", ".gpkg", ".shp"}:
        gdf, read_meta = read_spatial_sample(path, sample_rows=sample_rows)
        details: dict[str, Any] = {
            "file_size_bytes": safe_file_size(path),
            "small_file_sha256": maybe_hash_small_file(path),
            "read": read_meta,
        }

        if gdf is None:
            return CandidateInspection(
                path=str(path),
                exists=True,
                kind="spatial",
                status="read_failed",
                error=read_meta.get("error"),
                details=details,
            )

        columns = [str(col) for col in gdf.columns]
        id_columns = infer_id_columns(columns, expected_candidates=expected_id_candidates)

        details.update(
            {
                "columns": columns,
                "n_columns": len(columns),
                "n_rows_sampled": int(len(gdf)),
                "id_columns": id_columns,
                "required_column_presence": {
                    "census_tract_dguid": "census_tract_dguid" in columns,
                    "land_area_km2": "land_area_km2" in columns,
                    "population_total_2021": "population_total_2021" in columns,
                },
                "id_samples": extract_id_samples(gdf, id_columns),
            }
        )

        return CandidateInspection(
            path=str(path),
            exists=True,
            kind="spatial",
            status="inspected",
            error=None,
            details=details,
        )

    table = inspect_table_candidate(path, sample_rows=sample_rows)
    details = dict(table.details)
    if table.status == "inspected":
        columns = details.get("columns", [])
        id_columns = infer_id_columns(columns, expected_candidates=expected_id_candidates)
        details["id_columns"] = id_columns
        df, _ = read_table_sample(path, sample_rows=sample_rows)
        details["id_samples"] = extract_id_samples(df, id_columns)

    return CandidateInspection(
        path=table.path,
        exists=table.exists,
        kind=table.kind,
        status=table.status,
        error=table.error,
        details=details,
    )


def inspect_svi_candidate(
    path: Path,
    expected_id_candidates: Iterable[str] = (),
    expected_score_columns: Iterable[str] = (),
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
) -> CandidateInspection:
    """Inspect an SVI candidate output/input file."""

    base = inspect_table_candidate(path, sample_rows=sample_rows)

    if base.status != "inspected":
        return base

    df, _ = read_table_sample(path, sample_rows=sample_rows)
    details = dict(base.details)

    columns = details.get("columns", [])
    id_columns = infer_id_columns(columns, expected_candidates=expected_id_candidates)
    score_columns = columns_matching(
        columns,
        exact_candidates=expected_score_columns,
        contains_any=[
            "svi_score",
            "score",
            "score_normalized",
            "percentile",
            "rank",
            "class",
            "theme",
        ],
    )

    details["id_columns"] = id_columns
    details["score_or_output_columns"] = score_columns
    details["id_samples"] = extract_id_samples(df, id_columns)

    return CandidateInspection(
        path=base.path,
        exists=base.exists,
        kind=base.kind,
        status=base.status,
        error=base.error,
        details=details,
    )


def extract_id_samples(df: Any, id_columns: Iterable[str], max_values: int = 5000) -> dict[str, list[str]]:
    """Extract limited ID samples from candidate ID columns."""

    if pd is None or df is None:
        return {}

    samples: dict[str, list[str]] = {}

    for col in id_columns:
        if col not in df.columns:
            continue

        values = (
            df[col]
            .dropna()
            .astype(str)
            .map(str.strip)
        )
        unique = values[values != ""].drop_duplicates().head(max_values).tolist()
        samples[col] = unique

    return samples


def best_candidate(inspections: list[CandidateInspection], score_key: str | None = None) -> CandidateInspection | None:
    """Pick a simple best available candidate for reporting purposes."""

    usable = [item for item in inspections if item.exists and item.status == "inspected"]
    if not usable:
        return None

    if score_key is None:
        return usable[0]

    def score(item: CandidateInspection) -> int:
        value = item.details.get(score_key)
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            return len(value)
        if value:
            return 1
        return 0

    return sorted(usable, key=score, reverse=True)[0]


def compare_id_overlap(
    left: CandidateInspection | None,
    right: CandidateInspection | None,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    """Compare sampled ID values between two inspected candidates."""

    if left is None or right is None:
        return {
            "status": "not_available",
            "reason": "one_or_both_candidates_missing",
        }

    left_samples = left.details.get("id_samples", {})
    right_samples = right.details.get("id_samples", {})

    if not left_samples or not right_samples:
        return {
            "status": "not_available",
            "reason": "id_samples_missing",
            "left_candidate": left.path,
            "right_candidate": right.path,
        }

    comparisons: list[dict[str, Any]] = []

    for left_col, left_values in left_samples.items():
        left_set = set(map(str, left_values))
        for right_col, right_values in right_samples.items():
            right_set = set(map(str, right_values))
            overlap = left_set & right_set
            union = left_set | right_set

            comparisons.append(
                {
                    "left_label": left_label,
                    "left_column": left_col,
                    "right_label": right_label,
                    "right_column": right_col,
                    "left_sample_unique": len(left_set),
                    "right_sample_unique": len(right_set),
                    "overlap_count": len(overlap),
                    "union_count": len(union),
                    "overlap_ratio_left": len(overlap) / len(left_set) if left_set else None,
                    "overlap_ratio_right": len(overlap) / len(right_set) if right_set else None,
                    "overlap_examples": compact_list(sorted(overlap), max_items=20),
                }
            )

    if not comparisons:
        return {
            "status": "not_available",
            "reason": "no_id_column_pairs",
            "left_candidate": left.path,
            "right_candidate": right.path,
        }

    best = sorted(
        comparisons,
        key=lambda row: (
            row["overlap_count"],
            row["overlap_ratio_left"] or 0,
            row["overlap_ratio_right"] or 0,
        ),
        reverse=True,
    )[0]

    return {
        "status": "computed",
        "left_candidate": left.path,
        "right_candidate": right.path,
        "best_overlap": best,
        "all_comparisons": comparisons,
    }


def keyword_list_from_config(config: Mapping[str, Any]) -> list[str]:
    """Read target keyword families from config, with fallback values."""

    strict = get_nested(
        config,
        ["target", "category_selection", "keyword_families", "strict_candidates"],
        default=[],
    )
    ambiguous = get_nested(
        config,
        ["target", "category_selection", "keyword_families", "ambiguous_needs_review"],
        default=[],
    )

    keywords = [*strict, *ambiguous]
    if not keywords:
        keywords = list(WATER_KEYWORDS_FALLBACK)

    return [str(keyword) for keyword in keywords if not is_unresolved_value(keyword)]


def inventory_config_paths(config: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    """Return config path statuses useful for debugging."""

    path_fields = {
        "contract_path": config.get("contract_path"),
        "dataset_dir": get_nested(config, ["paths", "outputs", "dataset_dir"]),
        "report_dir": get_nested(config, ["paths", "outputs", "report_dir"]),
        "input_inventory_json": get_nested(config, ["inventory", "output_json"]),
        "input_inventory_report": get_nested(config, ["inventory", "output_report"]),
    }

    return {
        key: path_status(value, repo_root=repo_root).to_dict()
        for key, value in path_fields.items()
    }


def collect_input_candidates(config: Mapping[str, Any], repo_root: Path) -> dict[str, list[Path]]:
    """Collect candidate paths for all current inventory inputs."""

    return {
        "montreal_311": collect_candidates_from_config_section(
            config,
            ["inputs", "montreal_311"],
            repo_root=repo_root,
            existing_only=False,
            require_file=False,
            max_recursive_results=50,
        ),
        "census_tract_geometry": collect_candidates_from_config_section(
            config,
            ["inputs", "census_tract_geometry"],
            repo_root=repo_root,
            existing_only=False,
            require_file=False,
            max_recursive_results=50,
        ),
        "svi": collect_candidates_from_config_section(
            config,
            ["inputs", "svi"],
            repo_root=repo_root,
            existing_only=False,
            require_file=False,
            max_recursive_results=50,
        ),
    }


def candidate_path_statuses(candidates: Mapping[str, list[Path]], repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Create path-status rows for candidate collections."""

    return {
        name: [path_status(path, repo_root=repo_root).to_dict() for path in paths]
        for name, paths in candidates.items()
    }


def inspect_all_candidates(
    config: Mapping[str, Any],
    candidates: Mapping[str, list[Path]],
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
) -> dict[str, list[dict[str, Any]]]:
    """Inspect all configured input candidates."""

    keywords = keyword_list_from_config(config)

    tract_expected_ids = get_nested(
        config,
        ["inputs", "census_tract_geometry", "expected_id_column_candidates"],
        default=[],
    )
    svi_expected_ids = get_nested(
        config,
        ["inputs", "svi", "expected_id_column_candidates"],
        default=[],
    )
    svi_expected_scores = get_nested(
        config,
        ["inputs", "svi", "expected_score_columns"],
        default=[],
    )

    montreal_311 = [
        inspect_311_candidate(path, keywords=keywords, sample_rows=sample_rows).to_dict()
        for path in candidates.get("montreal_311", [])
    ]

    census_tract_geometry = [
        inspect_spatial_candidate(
            path,
            expected_id_candidates=tract_expected_ids,
            sample_rows=sample_rows,
        ).to_dict()
        for path in candidates.get("census_tract_geometry", [])
    ]

    svi = [
        inspect_svi_candidate(
            path,
            expected_id_candidates=svi_expected_ids,
            expected_score_columns=svi_expected_scores,
            sample_rows=sample_rows,
        ).to_dict()
        for path in candidates.get("svi", [])
    ]

    return {
        "montreal_311": montreal_311,
        "census_tract_geometry": census_tract_geometry,
        "svi": svi,
    }


def dicts_to_inspections(rows: list[dict[str, Any]]) -> list[CandidateInspection]:
    """Convert serialized candidate dictionaries back into CandidateInspection objects."""

    return [
        CandidateInspection(
            path=str(row.get("path", "")),
            exists=bool(row.get("exists", False)),
            kind=str(row.get("kind", "")),
            status=str(row.get("status", "")),
            error=row.get("error"),
            details=dict(row.get("details", {}) or {}),
        )
        for row in rows
    ]


def summarize_decisions(
    config: Mapping[str, Any],
    inspections: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Summarize unresolved decisions and inventory-derived hints."""

    mtl_inspections = dicts_to_inspections(list(inspections.get("montreal_311", [])))
    tract_inspections = dicts_to_inspections(list(inspections.get("census_tract_geometry", [])))
    svi_inspections = dicts_to_inspections(list(inspections.get("svi", [])))

    best_311 = best_candidate(mtl_inspections)
    best_tract = best_candidate(tract_inspections, score_key="id_columns")
    best_svi = best_candidate(svi_inspections, score_key="score_or_output_columns")

    inferred_311_types = [
        item.details.get("inferred_source_type")
        for item in mtl_inspections
        if item.exists and item.status == "inspected"
    ]
    inferred_311_type_counts = dict(Counter(str(value) for value in inferred_311_types if value))

    join = compare_id_overlap(
        best_tract,
        best_svi,
        left_label="census_tract_geometry",
        right_label="svi",
    )

    return {
        "benchmark_id": config.get("benchmark_id"),
        "status": config.get("status"),
        "best_candidates": {
            "montreal_311": best_311.path if best_311 else None,
            "census_tract_geometry": best_tract.path if best_tract else None,
            "svi": best_svi.path if best_svi else None,
        },
        "candidate_counts": {
            "montreal_311": len(mtl_inspections),
            "census_tract_geometry": len(tract_inspections),
            "svi": len(svi_inspections),
        },
        "existing_candidate_counts": {
            "montreal_311": sum(item.exists for item in mtl_inspections),
            "census_tract_geometry": sum(item.exists for item in tract_inspections),
            "svi": sum(item.exists for item in svi_inspections),
        },
        "inferred_311_source_type_counts": inferred_311_type_counts,
        "tract_svi_join_feasibility": join,
        "open_decisions_from_config": {
            "study_area": get_nested(config, ["unit", "study_area", "selected"]),
            "311_source_type": get_nested(config, ["inputs", "montreal_311", "source_type"]),
            "spatial_assignment_method": get_nested(config, ["spatial_assignment", "method"]),
            "target_category_selection_method": get_nested(config, ["target", "category_selection", "method"]),
            "magnitude_threshold_strategy": get_nested(config, ["target", "magnitude_target", "threshold_strategy"]),
        },
    }


def render_candidate_table(rows: list[dict[str, Any]], max_rows: int = 20) -> str:
    """Render candidate inspections as a simple Markdown table."""

    if not rows:
        return "_No candidates configured or found._\n"

    lines = [
        "| # | Exists | Status | Kind | Path | Notes |",
        "|---:|:---:|---|---|---|---|",
    ]

    for idx, row in enumerate(rows[:max_rows], start=1):
        details = row.get("details", {}) or {}
        notes_parts: list[str] = []

        if "inferred_source_type" in details:
            notes_parts.append(f"source_type={details.get('inferred_source_type')}")

        if details.get("inferred_source_type_confidence"):
            notes_parts.append(f"confidence={details.get('inferred_source_type_confidence')}")

        if details.get("grid25m_value_hints"):
            notes_parts.append("grid25m_values=yes")

        if details.get("id_columns"):
            notes_parts.append(f"id_cols={len(details.get('id_columns', []))}")

        if details.get("score_or_output_columns"):
            notes_parts.append(f"score_cols={len(details.get('score_or_output_columns', []))}")

        read = details.get("read", {})
        if read.get("crs"):
            notes_parts.append(f"crs={read.get('crs')}")

        notes = "<br>".join(notes_parts)
        path = str(row.get("path", "")).replace("|", "\\|")

        lines.append(
            f"| {idx} | {row.get('exists')} | {row.get('status')} | "
            f"{row.get('kind')} | `{path}` | {notes} |"
        )

    if len(rows) > max_rows:
        lines.append(f"\n_Only first {max_rows} of {len(rows)} candidates shown._")

    return "\n".join(lines) + "\n"


def render_inventory_report(inventory: Mapping[str, Any]) -> str:
    """Render the inventory dictionary as Markdown."""

    generated_at = inventory.get("generated_at")
    config = inventory.get("config", {})
    decisions = inventory.get("decision_summary", {})
    inspections = inventory.get("inspections", {})

    best = decisions.get("best_candidates", {})
    candidate_counts = decisions.get("candidate_counts", {})
    existing_counts = decisions.get("existing_candidate_counts", {})
    join = decisions.get("tract_svi_join_feasibility", {})

    lines: list[str] = []

    lines.append("# Input Inventory Report — Montréal 311 Water/Drainage v0\n")
    lines.append(f"Generated at: `{generated_at}`\n")
    lines.append(f"Benchmark ID: `{config.get('benchmark_id')}`\n")
    lines.append(f"Config path: `{config.get('config_path')}`\n")
    lines.append(f"Config hash: `{config.get('config_hash')}`\n")
    lines.append(f"Repository root: `{config.get('repo_root')}`\n")

    lines.append("## Summary\n")
    lines.append("| Input | Candidates | Existing | Best candidate |")
    lines.append("|---|---:|---:|---|")
    for name in ["montreal_311", "census_tract_geometry", "svi"]:
        lines.append(
            f"| `{name}` | {candidate_counts.get(name, 0)} | "
            f"{existing_counts.get(name, 0)} | `{best.get(name)}` |"
        )
    lines.append("")

    lines.append("## Open decisions\n")
    open_decisions = decisions.get("open_decisions_from_config", {})
    lines.append("| Decision | Current value |")
    lines.append("|---|---|")
    for key, value in open_decisions.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")

    lines.append("## 311 source-type hints\n")
    type_counts = decisions.get("inferred_311_source_type_counts", {})
    if type_counts:
        lines.append("| Inferred source type | Candidate count |")
        lines.append("|---|---:|")
        for source_type, count in type_counts.items():
            lines.append(f"| `{source_type}` | {count} |")
        lines.append("")
    else:
        lines.append("_No inspected 311 candidate source type could be inferred._\n")

    lines.append("## Tract geometry ↔ SVI join feasibility\n")
    lines.append(f"Status: `{join.get('status')}`\n")
    if join.get("best_overlap"):
        best_overlap = join["best_overlap"]
        lines.append("| Left column | Right column | Overlap | Left ratio | Right ratio |")
        lines.append("|---|---|---:|---:|---:|")
        lines.append(
            f"| `{best_overlap.get('left_column')}` | `{best_overlap.get('right_column')}` | "
            f"{best_overlap.get('overlap_count')} | "
            f"{format_ratio(best_overlap.get('overlap_ratio_left'))} | "
            f"{format_ratio(best_overlap.get('overlap_ratio_right'))} |"
        )
        lines.append("")
    elif join.get("reason"):
        lines.append(f"Reason: `{join.get('reason')}`\n")

    lines.append("## Montréal 311 candidates\n")
    lines.append(render_candidate_table(list(inspections.get("montreal_311", []))))

    lines.append("## Census tract geometry candidates\n")
    lines.append(render_candidate_table(list(inspections.get("census_tract_geometry", []))))

    lines.append("## SVI candidates\n")
    lines.append(render_candidate_table(list(inspections.get("svi", []))))

    lines.append("## Next steps\n")
    lines.append(
        "1. Confirm the canonical 311 file path and whether it is point-level, "
        "grid25m-month, or already aggregated.\n"
        "2. Confirm the exact Montréal study area based on valid 311 coverage.\n"
        "3. Confirm the tract geometry file and SVI output file.\n"
        "4. Review candidate 311 categories before building the target.\n"
        "5. Revise the config values currently marked `DECISION_NEEDED`.\n"
    )

    return "\n".join(lines)


def run_inventory(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    repo_root: str | Path | None = None,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
) -> dict[str, Any]:
    """
    Run the input inventory and write configured report artifacts.

    Returns the full inventory dictionary.
    """

    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    resolved_config_path = resolve_path(config_path, repo_root=root, allow_unresolved=False)

    if resolved_config_path is None:
        raise ValueError(f"Could not resolve config path: {config_path}")

    config = load_config(resolved_config_path)

    created_dirs = ensure_output_directories(config, repo_root=root)
    candidates = collect_input_candidates(config, repo_root=root)
    statuses = candidate_path_statuses(candidates, repo_root=root)
    inspections = inspect_all_candidates(config, candidates, sample_rows=sample_rows)
    decisions = summarize_decisions(config, inspections)

    inventory: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "benchmark_id": config.get("benchmark_id"),
            "status": config.get("status"),
            "version": config.get("version"),
            "config_path": str(resolved_config_path),
            "config_hash": config_hash(config),
            "repo_root": str(root),
        },
        "created_output_directories": [str(path) for path in created_dirs],
        "config_path_statuses": inventory_config_paths(config, repo_root=root),
        "candidate_path_statuses": statuses,
        "inspections": inspections,
        "decision_summary": decisions,
        "inventory_limits": {
            "sample_rows": sample_rows,
            "unique_value_limit": DEFAULT_UNIQUE_VALUE_LIMIT,
            "hash_max_bytes": DEFAULT_HASH_MAX_BYTES,
            "parquet_full_read_max_bytes": DEFAULT_PARQUET_FULL_READ_MAX_BYTES,
        },
    }

    output_json = get_nested(
        config,
        ["inventory", "output_json"],
        default="urban_graph_benchmark/outputs/mtl_311_water_v0/reports/input_inventory.json",
    )
    output_report = get_nested(
        config,
        ["inventory", "output_report"],
        default="urban_graph_benchmark/outputs/mtl_311_water_v0/reports/input_inventory_report.md",
    )

    output_json_path = resolve_path(output_json, repo_root=root, allow_unresolved=False)
    output_report_path = resolve_path(output_report, repo_root=root, allow_unresolved=False)

    if output_json_path is None or output_report_path is None:
        raise ValueError("Inventory output paths could not be resolved.")

    write_json(output_json_path, inventory, sort_keys=False)
    write_markdown(output_report_path, render_inventory_report(inventory))

    inventory["written_outputs"] = {
        "json": str(output_json_path),
        "report": str(output_report_path),
    }

    write_json(output_json_path, inventory, sort_keys=False)

    return inventory


def inventory_brief(inventory: Mapping[str, Any]) -> str:
    """Return a concise human-readable inventory summary."""

    decisions = inventory.get("decision_summary", {})
    best = decisions.get("best_candidates", {})
    counts = decisions.get("existing_candidate_counts", {})

    return (
        "Input inventory completed.\n"
        f"Existing 311 candidates: {counts.get('montreal_311', 0)}\n"
        f"Existing tract geometry candidates: {counts.get('census_tract_geometry', 0)}\n"
        f"Existing SVI candidates: {counts.get('svi', 0)}\n"
        f"Best 311 candidate: {best.get('montreal_311')}\n"
        f"Best tract geometry candidate: {best.get('census_tract_geometry')}\n"
        f"Best SVI candidate: {best.get('svi')}\n"
    )


def main() -> None:
    """CLI entry point for direct module execution."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Montréal 311 water/drainage input inventory."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Config path. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root. Defaults to automatic detection.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=DEFAULT_SAMPLE_ROWS,
        help=f"Rows to sample from tabular files. Default: {DEFAULT_SAMPLE_ROWS}",
    )

    args = parser.parse_args()

    inventory = run_inventory(
        config_path=args.config,
        repo_root=args.repo_root,
        sample_rows=args.sample_rows,
    )

    print(inventory_brief(inventory))
    print("Written outputs:")
    for label, path in inventory.get("written_outputs", {}).items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "CandidateInspection",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_HASH_MAX_BYTES",
    "DEFAULT_PARQUET_FULL_READ_MAX_BYTES",
    "DEFAULT_SAMPLE_ROWS",
    "compare_id_overlap",
    "collect_input_candidates",
    "format_ratio",
    "infer_311_source_type",
    "inspect_311_candidate",
    "inspect_spatial_candidate",
    "inspect_svi_candidate",
    "inventory_brief",
    "render_inventory_report",
    "run_inventory",
]