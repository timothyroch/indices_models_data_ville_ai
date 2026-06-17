#!/usr/bin/env python3
"""
Build Québec census-division graph node and edge files for the civil-security / SoVI benchmark.

Purpose:
    Create the static CD graph assets that later graph baselines can consume.

Outputs:
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/datasets/cd_graph_nodes.parquet
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/datasets/cd_graph_edges_adjacency.parquet
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/datasets/cd_graph_edges_knn.parquet
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/datasets/cd_graph_edges_random_placebo.parquet

Graph types:
    1. Real CD adjacency graph:
       Edges connect CDs sharing a non-trivial boundary segment.

    2. kNN centroid graph:
       Directed k-nearest-neighbor edges based on CD centroids in a projected CRS.

    3. Random/placebo graph:
       Degree-preserving randomization of the adjacency graph when possible.
       Falls back to a same-edge-count random graph if rewiring cannot produce
       enough valid swaps.

Design choices:
    - Node IDs are exactly the normalized CD identifiers used in the panel.
    - Edge files include both string CD IDs and integer node indices.
    - Adjacency/random graphs are stored as bidirectional directed edges because
      most GNN libraries expect message-passing edges in both directions.
    - kNN is stored as directed source -> nearest-neighbor edges.
    - CSV copies and metadata are written for inspection/reproducibility.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from ville_hgnn.baselines.qc_cd_sovi_common import (
        CD_ID_COL,
        CD_NAME_COL,
        DATASETS_DIR,
        DEFAULT_PANEL_PATH,
        ensure_dir,
        write_metadata_json,
    )
except Exception:  # pragma: no cover - fallback for early bootstrapping only
    CD_ID_COL = "cd_id_norm"
    CD_NAME_COL = "cd_name"
    DATASETS_DIR = Path("urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/datasets")
    DEFAULT_PANEL_PATH = DATASETS_DIR / "cd_month_panel.parquet"

    def ensure_dir(path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        return out

    def write_metadata_json(metadata: dict[str, Any], path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        return out


DEFAULT_BOUNDARIES_PATH = Path(
    "data/2021-census-division-boundary-file/lcd_000b21a_e/lcd_000b21a_e.shp"
)

DEFAULT_NODES_PATH = DATASETS_DIR / "cd_graph_nodes.parquet"
DEFAULT_ADJACENCY_EDGES_PATH = DATASETS_DIR / "cd_graph_edges_adjacency.parquet"
DEFAULT_KNN_EDGES_PATH = DATASETS_DIR / "cd_graph_edges_knn.parquet"
DEFAULT_RANDOM_EDGES_PATH = DATASETS_DIR / "cd_graph_edges_random_placebo.parquet"

TIME_OR_TARGET_PREFIXES = (
    "target_",
    "event_count_current_month",
    "event_count_all_",
    "event_count_precise",
    "event_count_very",
)

TIME_OR_TARGET_CONTAINS = (
    "_lag_",
    "_rolling_",
    "future",
)

PANEL_NON_STATIC_COLUMNS = {
    "month",
    "year",
    "period_month",
    "split",
    "target_next_1_month",
    "target_next_3_months",
    "target_next_6_months",
    "target_next_1m_all",
    "target_next_3m_all",
    "target_next_6m_all",
    "target_next_1_month_complete",
    "target_next_3_months_complete",
    "target_next_6_months_complete",
    "lag_1",
    "rolling_3",
    "rolling_6",
    "rolling_12",
}


def lazy_import_geopandas() -> Any:
    """Import GeoPandas lazily and provide a clear error if unavailable."""
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Building CD adjacency requires geopandas. Install geopandas in the "
            "active environment, then rerun this script."
        ) from exc
    return gpd


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a pandas table by suffix."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input table does not exist: {p}")

    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)

    raise ValueError(f"Unsupported table suffix: {p}")


def write_table_with_csv_copy(df: pd.DataFrame, path: str | Path) -> dict[str, str]:
    """Write parquet/csv table and an inspection CSV copy."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    suffix = p.suffix.lower()

    if suffix == ".parquet":
        df.to_parquet(p, index=False)
        outputs["parquet"] = str(p)
        csv_path = p.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        outputs["csv"] = str(csv_path)
        return outputs

    if suffix == ".csv":
        df.to_csv(p, index=False)
        outputs["csv"] = str(p)
        return outputs

    raise ValueError(f"Unsupported output table suffix: {p}")


def normalize_cd_id(value: Any) -> str | None:
    """Normalize CD identifiers to compact 4-digit strings such as '2401'."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    text = re.sub(r"\.0$", "", text)
    digits = re.findall(r"\d+", text)
    if digits:
        joined = "".join(digits)
        if len(joined) >= 4:
            return joined[-4:]
        return joined.zfill(4)

    return text


def find_first_existing(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    """Find the first candidate column in a list, case-insensitive."""
    lower_to_original = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def load_panel(panel_path: Path, cd_id_col: str) -> pd.DataFrame:
    """Load the predictive panel and normalize CD IDs."""
    panel = read_table(panel_path)

    if cd_id_col not in panel.columns:
        raise KeyError(
            f"Panel is missing CD ID column '{cd_id_col}'. "
            f"Available columns: {list(panel.columns)}"
        )

    panel = panel.copy()
    panel[CD_ID_COL] = panel[cd_id_col].map(normalize_cd_id)

    if panel[CD_ID_COL].isna().any():
        missing = int(panel[CD_ID_COL].isna().sum())
        raise ValueError(f"Panel contains {missing} rows with missing normalized CD IDs.")

    return panel


def is_static_panel_column(panel: pd.DataFrame, col: str, cd_id_col: str) -> bool:
    """Whether a panel column is static by CD and safe for the graph node table."""
    lower = col.lower()

    if col == cd_id_col or col == CD_ID_COL:
        return False
    if col in PANEL_NON_STATIC_COLUMNS:
        return False
    if lower.startswith(TIME_OR_TARGET_PREFIXES):
        return False
    if any(token in lower for token in TIME_OR_TARGET_CONTAINS):
        return False
    if lower.startswith("pred_") or lower.startswith("prediction"):
        return False

    # Keep CD name via explicit handling.
    if col == CD_NAME_COL:
        return False

    nunique = panel.groupby(CD_ID_COL)[col].nunique(dropna=True)
    return bool((nunique <= 1).all())


def extract_static_panel_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Extract CD-level static features from the panel.

    This preserves SoVI score/components and any other static numeric/string node
    features that are constant across months for each CD.
    """
    static_cols = [CD_ID_COL]

    if CD_NAME_COL in panel.columns:
        static_cols.append(CD_NAME_COL)

    for col in panel.columns:
        if col in static_cols:
            continue
        if is_static_panel_column(panel, col, CD_ID_COL):
            static_cols.append(col)

    static = (
        panel.sort_values([CD_ID_COL])
        .groupby(CD_ID_COL, as_index=False)[static_cols]
        .first()
    )

    # groupby with as_index=False duplicates CD_ID_COL if present in aggregation list
    if isinstance(static.columns, pd.MultiIndex):
        static.columns = ["_".join(map(str, c)).strip("_") for c in static.columns]

    static = static.loc[:, ~static.columns.duplicated()].copy()

    if CD_NAME_COL not in static.columns and CD_NAME_COL in panel.columns:
        names = (
            panel[[CD_ID_COL, CD_NAME_COL]]
            .dropna(subset=[CD_ID_COL])
            .drop_duplicates(subset=[CD_ID_COL])
        )
        static = static.merge(names, on=CD_ID_COL, how="left")

    return static


def load_boundaries(
    boundaries_path: Path,
    *,
    boundary_id_col: str,
    boundary_name_col: str,
    assume_boundary_crs: str,
) -> Any:
    """Load CD boundary geometries."""
    gpd = lazy_import_geopandas()
    gdf = gpd.read_file(boundaries_path)

    if boundary_id_col not in gdf.columns:
        raise KeyError(
            f"Boundary file is missing ID column '{boundary_id_col}'. "
            f"Available columns: {list(gdf.columns)}"
        )

    if boundary_name_col not in gdf.columns:
        raise KeyError(
            f"Boundary file is missing name column '{boundary_name_col}'. "
            f"Available columns: {list(gdf.columns)}"
        )

    gdf = gdf.copy()
    gdf[CD_ID_COL] = gdf[boundary_id_col].map(normalize_cd_id)
    gdf[CD_NAME_COL] = gdf[boundary_name_col].astype("string")

    if gdf.crs is None:
        gdf = gdf.set_crs(assume_boundary_crs)

    # Repair invalid geometries conservatively.
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    gdf = gdf.dropna(subset=[CD_ID_COL]).drop_duplicates(subset=[CD_ID_COL], keep="first")
    return gdf


def align_boundaries_to_panel(
    boundaries: Any,
    panel_static: pd.DataFrame,
    *,
    allow_missing_boundaries: bool,
) -> Any:
    """Filter boundaries to the CD IDs present in the panel and check alignment."""
    panel_ids = set(panel_static[CD_ID_COL].astype(str))
    boundary_ids = set(boundaries[CD_ID_COL].astype(str))

    missing = sorted(panel_ids - boundary_ids)
    extra = sorted(boundary_ids - panel_ids)

    if missing and not allow_missing_boundaries:
        raise ValueError(
            "Some panel CDs are missing from the boundary file. "
            f"Missing count: {len(missing)}. Examples: {missing[:20]}"
        )

    aligned = boundaries[boundaries[CD_ID_COL].astype(str).isin(panel_ids)].copy()
    aligned = aligned.sort_values(CD_ID_COL).reset_index(drop=True)

    if aligned.empty:
        raise ValueError("No boundary geometries overlap the panel CD IDs.")

    aligned.attrs["missing_panel_ids"] = missing
    aligned.attrs["extra_boundary_ids"] = extra
    return aligned


def build_nodes(
    boundaries: Any,
    panel_static: pd.DataFrame,
    *,
    metric_crs: str,
    lonlat_crs: str,
) -> pd.DataFrame:
    """Build CD node table with geometry-derived fields and static panel features."""
    gdf_metric = boundaries.to_crs(metric_crs)
    gdf_lonlat = boundaries.to_crs(lonlat_crs)

    metric_centroids = gdf_metric.geometry.centroid
    lonlat_centroids = gdf_lonlat.geometry.centroid

    geom_features = pd.DataFrame(
        {
            CD_ID_COL: boundaries[CD_ID_COL].astype(str).to_numpy(),
            "boundary_cd_name": boundaries[CD_NAME_COL].astype("string").to_numpy(),
            "centroid_x_metric": metric_centroids.x.to_numpy(dtype=float),
            "centroid_y_metric": metric_centroids.y.to_numpy(dtype=float),
            "centroid_lon": lonlat_centroids.x.to_numpy(dtype=float),
            "centroid_lat": lonlat_centroids.y.to_numpy(dtype=float),
            "area_km2": (gdf_metric.geometry.area.to_numpy(dtype=float) / 1_000_000.0),
            "perimeter_km": (gdf_metric.geometry.length.to_numpy(dtype=float) / 1_000.0),
        }
    )

    nodes = geom_features.merge(panel_static, on=CD_ID_COL, how="left", validate="one_to_one")

    if CD_NAME_COL not in nodes.columns:
        nodes[CD_NAME_COL] = nodes["boundary_cd_name"].astype("string")
    else:
        nodes[CD_NAME_COL] = nodes[CD_NAME_COL].astype("string").fillna(nodes["boundary_cd_name"])

    nodes = nodes.sort_values(CD_ID_COL).reset_index(drop=True)
    nodes.insert(0, "node_index", np.arange(len(nodes), dtype=int))

    # Put core columns first.
    front = [
        "node_index",
        CD_ID_COL,
        CD_NAME_COL,
        "boundary_cd_name",
        "centroid_lon",
        "centroid_lat",
        "centroid_x_metric",
        "centroid_y_metric",
        "area_km2",
        "perimeter_km",
    ]
    front = [c for c in front if c in nodes.columns]
    rest = [c for c in nodes.columns if c not in front]
    return nodes[front + rest]


def node_lookup(nodes: pd.DataFrame) -> dict[str, int]:
    """Map CD ID to integer node index."""
    return dict(zip(nodes[CD_ID_COL].astype(str), nodes["node_index"].astype(int)))


def pair_id(a: str, b: str) -> str:
    """Stable undirected pair identifier."""
    x, y = sorted([str(a), str(b)])
    return f"{x}__{y}"


def directed_edges_from_undirected_pairs(
    pairs: Sequence[tuple[str, str]],
    nodes: pd.DataFrame,
    *,
    graph_name: str,
    edge_type: str,
    edge_weight: float = 1.0,
    pair_attributes: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Convert undirected CD pairs into bidirectional directed edge rows."""
    idx = node_lookup(nodes)
    rows: list[dict[str, Any]] = []

    for a, b in pairs:
        a = str(a)
        b = str(b)
        key = tuple(sorted([a, b]))
        attrs = dict(pair_attributes.get(key, {})) if pair_attributes else {}

        for src, dst, direction in [(a, b, "forward"), (b, a, "reverse")]:
            rows.append(
                {
                    "graph_name": graph_name,
                    "edge_type": edge_type,
                    "is_directed_storage": True,
                    "is_symmetric_graph": True,
                    "source_cd_id_norm": src,
                    "target_cd_id_norm": dst,
                    "src_cd_id_norm": src,
                    "dst_cd_id_norm": dst,
                    "source_node_index": idx[src],
                    "target_node_index": idx[dst],
                    "src_node_index": idx[src],
                    "dst_node_index": idx[dst],
                    "undirected_pair_id": pair_id(src, dst),
                    "direction": direction,
                    "edge_weight": float(edge_weight),
                    "random_seed": random_seed,
                    **attrs,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(
        ["source_cd_id_norm", "target_cd_id_norm", "edge_type"]
    ).reset_index(drop=True)


def build_adjacency_pairs(
    boundaries: Any,
    nodes: pd.DataFrame,
    *,
    metric_crs: str,
    min_shared_boundary_km: float,
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], dict[str, Any]]]:
    """
    Build real CD adjacency pairs.

    Adjacency is defined as sharing a boundary segment whose length exceeds
    min_shared_boundary_km. Point-only contacts are excluded.
    """
    metric = boundaries.to_crs(metric_crs).copy()
    metric = metric.sort_values(CD_ID_COL).reset_index(drop=True)

    id_to_node = node_lookup(nodes)

    pairs: list[tuple[str, str]] = []
    attrs: dict[tuple[str, str], dict[str, Any]] = {}

    geoms = list(metric.geometry)
    ids = metric[CD_ID_COL].astype(str).tolist()

    for i in range(len(metric)):
        geom_i = geoms[i]
        id_i = ids[i]
        if id_i not in id_to_node:
            continue

        for j in range(i + 1, len(metric)):
            geom_j = geoms[j]
            id_j = ids[j]
            if id_j not in id_to_node:
                continue

            if not geom_i.bounds or not geom_j.bounds:
                continue

            # Fast-ish bounding-box precheck.
            minx1, miny1, maxx1, maxy1 = geom_i.bounds
            minx2, miny2, maxx2, maxy2 = geom_j.bounds
            if maxx1 < minx2 or maxx2 < minx1 or maxy1 < miny2 or maxy2 < miny1:
                continue

            shared_boundary = geom_i.boundary.intersection(geom_j.boundary)
            shared_km = float(shared_boundary.length) / 1_000.0

            if shared_km >= min_shared_boundary_km:
                key = tuple(sorted([id_i, id_j]))
                pairs.append(key)
                attrs[key] = {
                    "shared_boundary_km": shared_km,
                    "distance_km": 0.0,
                    "adjacency_rule": "shared_boundary_length",
                    "min_shared_boundary_km": float(min_shared_boundary_km),
                }

    pairs = sorted(set(pairs))
    return pairs, attrs


def pairwise_centroid_distance_km(nodes: pd.DataFrame) -> np.ndarray:
    """Compute Euclidean centroid distances in the metric CRS, returned in km."""
    coords = nodes[["centroid_x_metric", "centroid_y_metric"]].to_numpy(dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_m = np.sqrt(np.sum(diff * diff, axis=2))
    return dist_m / 1_000.0


def build_knn_edges(
    nodes: pd.DataFrame,
    *,
    k: int,
) -> pd.DataFrame:
    """Build directed centroid kNN graph."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    n = len(nodes)
    if n <= 1:
        raise ValueError("At least two nodes are required for kNN edges.")

    k_eff = min(int(k), n - 1)
    distances = pairwise_centroid_distance_km(nodes)

    rows: list[dict[str, Any]] = []
    ids = nodes[CD_ID_COL].astype(str).tolist()
    indices = nodes["node_index"].astype(int).tolist()

    # First pass: directed nearest neighbors.
    directed_pairs: set[tuple[str, str]] = set()
    for i in range(n):
        order = np.argsort(distances[i])
        neighbors = [j for j in order if j != i][:k_eff]

        for rank, j in enumerate(neighbors, start=1):
            src = ids[i]
            dst = ids[j]
            directed_pairs.add((src, dst))
            rows.append(
                {
                    "graph_name": f"cd_centroid_knn_k{k_eff}",
                    "edge_type": "cd_centroid_knn",
                    "is_directed_storage": True,
                    "is_symmetric_graph": False,
                    "source_cd_id_norm": src,
                    "target_cd_id_norm": dst,
                    "src_cd_id_norm": src,
                    "dst_cd_id_norm": dst,
                    "source_node_index": indices[i],
                    "target_node_index": indices[j],
                    "src_node_index": indices[i],
                    "dst_node_index": indices[j],
                    "undirected_pair_id": pair_id(src, dst),
                    "direction": "directed_knn",
                    "knn_k": k_eff,
                    "knn_rank": rank,
                    "distance_km": float(distances[i, j]),
                    "edge_weight": 1.0 / (1.0 + float(distances[i, j])),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["is_mutual_knn"] = [
        (str(dst), str(src)) in directed_pairs
        for src, dst in zip(out["source_cd_id_norm"], out["target_cd_id_norm"])
    ]

    return out.sort_values(["source_cd_id_norm", "knn_rank", "target_cd_id_norm"]).reset_index(drop=True)


def undirected_pairs_from_directed_edges(edges: pd.DataFrame) -> list[tuple[str, str]]:
    """Extract unique undirected pairs from edge rows."""
    if edges.empty:
        return []
    pairs = {
        tuple(sorted([str(a), str(b)]))
        for a, b in zip(edges["source_cd_id_norm"], edges["target_cd_id_norm"])
        if str(a) != str(b)
    }
    return sorted(pairs)


def degree_preserving_rewire(
    pairs: Sequence[tuple[str, str]],
    nodes: Sequence[str],
    *,
    seed: int,
    swaps_per_edge: int,
    max_attempts_multiplier: int = 50,
) -> list[tuple[str, str]]:
    """
    Degree-preserving randomization by double-edge swaps.

    This preserves the undirected degree sequence while avoiding self-loops and
    duplicate edges. It is deterministic under seed.
    """
    rng = random.Random(seed)

    edge_set = {tuple(sorted([str(a), str(b)])) for a, b in pairs if str(a) != str(b)}
    edges = list(edge_set)

    if len(edges) < 2:
        return sorted(edge_set)

    target_swaps = max(1, int(len(edges) * swaps_per_edge))
    max_attempts = max(target_swaps * max_attempts_multiplier, 1_000)

    swaps = 0
    attempts = 0

    while swaps < target_swaps and attempts < max_attempts:
        attempts += 1

        e1, e2 = rng.sample(edges, 2)
        a, b = e1
        c, d = e2

        if len({a, b, c, d}) < 4:
            continue

        if rng.random() < 0.5:
            new1 = tuple(sorted([a, d]))
            new2 = tuple(sorted([c, b]))
        else:
            new1 = tuple(sorted([a, c]))
            new2 = tuple(sorted([b, d]))

        if new1[0] == new1[1] or new2[0] == new2[1]:
            continue
        if new1 == new2:
            continue
        if new1 in edge_set or new2 in edge_set:
            continue

        edge_set.remove(e1)
        edge_set.remove(e2)
        edge_set.add(new1)
        edge_set.add(new2)

        edges = list(edge_set)
        swaps += 1

    return sorted(edge_set)


def random_same_edge_count_pairs(
    nodes: Sequence[str],
    *,
    n_edges: int,
    seed: int,
) -> list[tuple[str, str]]:
    """Build a random undirected graph with a fixed number of edges."""
    rng = random.Random(seed)
    node_list = [str(x) for x in nodes]
    possible = [
        tuple(sorted([node_list[i], node_list[j]]))
        for i in range(len(node_list))
        for j in range(i + 1, len(node_list))
    ]

    if n_edges > len(possible):
        raise ValueError(f"Requested {n_edges} random edges but only {len(possible)} possible pairs exist.")

    return sorted(rng.sample(possible, n_edges))


def build_random_placebo_edges(
    adjacency_pairs: Sequence[tuple[str, str]],
    nodes: pd.DataFrame,
    *,
    seed: int,
    swaps_per_edge: int,
) -> pd.DataFrame:
    """
    Build random/placebo graph.

    Preferred mode: degree-preserving rewired adjacency.
    Fallback: same-edge-count random graph.
    """
    node_ids = nodes[CD_ID_COL].astype(str).tolist()
    n_edges = len(adjacency_pairs)

    if n_edges <= 0:
        raise ValueError("Cannot build random placebo graph: adjacency graph has zero edges.")

    rewired = degree_preserving_rewire(
        adjacency_pairs,
        node_ids,
        seed=seed,
        swaps_per_edge=swaps_per_edge,
    )

    # If rewiring failed to change anything, use a same-edge-count random graph.
    original_set = {tuple(sorted(p)) for p in adjacency_pairs}
    rewired_set = {tuple(sorted(p)) for p in rewired}

    if rewired_set == original_set:
        random_pairs = random_same_edge_count_pairs(node_ids, n_edges=n_edges, seed=seed)
        mode = "same_edge_count_random"
    else:
        random_pairs = sorted(rewired_set)
        mode = "degree_preserving_double_edge_swap"

    attrs = {
        tuple(sorted([a, b])): {
            "randomization_mode": mode,
            "distance_km": np.nan,
            "shared_boundary_km": np.nan,
            "random_seed": int(seed),
        }
        for a, b in random_pairs
    }

    return directed_edges_from_undirected_pairs(
        random_pairs,
        nodes,
        graph_name="cd_random_placebo",
        edge_type="cd_random_placebo",
        edge_weight=1.0,
        pair_attributes=attrs,
        random_seed=seed,
    )


def connected_components(nodes: Sequence[str], pairs: Sequence[tuple[str, str]]) -> list[list[str]]:
    """Compute connected components for an undirected graph."""
    adjacency: dict[str, set[str]] = {str(n): set() for n in nodes}
    for a, b in pairs:
        a = str(a)
        b = str(b)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    seen: set[str] = set()
    components: list[list[str]] = []

    for node in adjacency:
        if node in seen:
            continue

        comp: list[str] = []
        queue: deque[str] = deque([node])
        seen.add(node)

        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in adjacency.get(cur, []):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)

        components.append(sorted(comp))

    components.sort(key=len, reverse=True)
    return components


def graph_audit_row(
    *,
    graph_name: str,
    nodes: pd.DataFrame,
    directed_edges: pd.DataFrame,
) -> dict[str, Any]:
    """Build graph-level audit metrics."""
    node_ids = nodes[CD_ID_COL].astype(str).tolist()
    pairs = undirected_pairs_from_directed_edges(directed_edges)

    degree_counter = Counter()
    for a, b in pairs:
        degree_counter[a] += 1
        degree_counter[b] += 1

    degrees = np.array([degree_counter.get(n, 0) for n in node_ids], dtype=float)
    comps = connected_components(node_ids, pairs)

    return {
        "graph_name": graph_name,
        "n_nodes": int(len(node_ids)),
        "n_directed_edges": int(len(directed_edges)),
        "n_unique_undirected_pairs": int(len(pairs)),
        "mean_degree_undirected": float(degrees.mean()) if len(degrees) else np.nan,
        "min_degree_undirected": int(degrees.min()) if len(degrees) else 0,
        "max_degree_undirected": int(degrees.max()) if len(degrees) else 0,
        "n_isolates": int((degrees == 0).sum()) if len(degrees) else 0,
        "n_connected_components": int(len(comps)),
        "largest_component_size": int(len(comps[0])) if comps else 0,
        "is_connected": bool(len(comps) == 1),
    }


def add_common_edge_indices(edges: pd.DataFrame) -> pd.DataFrame:
    """Add stable edge IDs after sorting."""
    if edges.empty:
        return edges

    out = edges.copy()
    out = out.sort_values(
        ["graph_name", "source_cd_id_norm", "target_cd_id_norm", "edge_type"]
    ).reset_index(drop=True)
    out.insert(0, "edge_index", np.arange(len(out), dtype=int))
    return out


def build_graph_assets(
    *,
    panel_path: Path,
    boundaries_path: Path,
    output_nodes_path: Path,
    output_adjacency_edges_path: Path,
    output_knn_edges_path: Path,
    output_random_edges_path: Path,
    boundary_id_col: str,
    boundary_name_col: str,
    metric_crs: str,
    lonlat_crs: str,
    assume_boundary_crs: str,
    min_shared_boundary_km: float,
    knn_k: int,
    random_seed: int,
    random_swaps_per_edge: int,
    allow_missing_boundaries: bool,
) -> dict[str, Any]:
    """Build graph nodes and edge files."""
    ensure_dir(output_nodes_path.parent)

    panel = load_panel(panel_path, CD_ID_COL)
    panel_static = extract_static_panel_features(panel)

    boundaries_raw = load_boundaries(
        boundaries_path,
        boundary_id_col=boundary_id_col,
        boundary_name_col=boundary_name_col,
        assume_boundary_crs=assume_boundary_crs,
    )
    boundaries = align_boundaries_to_panel(
        boundaries_raw,
        panel_static,
        allow_missing_boundaries=allow_missing_boundaries,
    )

    nodes = build_nodes(
        boundaries,
        panel_static,
        metric_crs=metric_crs,
        lonlat_crs=lonlat_crs,
    )

    # Ensure all edges use the exact node subset/order.
    node_ids = set(nodes[CD_ID_COL].astype(str))
    boundaries = boundaries[boundaries[CD_ID_COL].astype(str).isin(node_ids)].copy()

    adjacency_pairs, adjacency_pair_attrs = build_adjacency_pairs(
        boundaries,
        nodes,
        metric_crs=metric_crs,
        min_shared_boundary_km=min_shared_boundary_km,
    )

    adjacency_edges = directed_edges_from_undirected_pairs(
        adjacency_pairs,
        nodes,
        graph_name="cd_adjacency",
        edge_type="cd_adjacency_shared_boundary",
        edge_weight=1.0,
        pair_attributes=adjacency_pair_attrs,
    )
    adjacency_edges = add_common_edge_indices(adjacency_edges)

    knn_edges = build_knn_edges(nodes, k=knn_k)
    knn_edges = add_common_edge_indices(knn_edges)

    random_edges = build_random_placebo_edges(
        adjacency_pairs,
        nodes,
        seed=random_seed,
        swaps_per_edge=random_swaps_per_edge,
    )
    random_edges = add_common_edge_indices(random_edges)

    node_outputs = write_table_with_csv_copy(nodes, output_nodes_path)
    adjacency_outputs = write_table_with_csv_copy(adjacency_edges, output_adjacency_edges_path)
    knn_outputs = write_table_with_csv_copy(knn_edges, output_knn_edges_path)
    random_outputs = write_table_with_csv_copy(random_edges, output_random_edges_path)

    audit_rows = [
        graph_audit_row(graph_name="cd_adjacency", nodes=nodes, directed_edges=adjacency_edges),
        graph_audit_row(graph_name=f"cd_centroid_knn_k{min(knn_k, max(len(nodes) - 1, 1))}", nodes=nodes, directed_edges=knn_edges),
        graph_audit_row(graph_name="cd_random_placebo", nodes=nodes, directed_edges=random_edges),
    ]
    audit = pd.DataFrame(audit_rows)
    audit_path = output_nodes_path.parent / "cd_graph_audit.csv"
    audit.to_csv(audit_path, index=False)

    edge_schema_rows = []
    for graph_name, edge_df in [
        ("cd_adjacency", adjacency_edges),
        ("cd_centroid_knn", knn_edges),
        ("cd_random_placebo", random_edges),
    ]:
        for col in edge_df.columns:
            edge_schema_rows.append(
                {
                    "graph_name": graph_name,
                    "column": col,
                    "dtype": str(edge_df[col].dtype),
                    "nonmissing": int(edge_df[col].notna().sum()),
                    "missing": int(edge_df[col].isna().sum()),
                    "unique": int(edge_df[col].nunique(dropna=True)),
                }
            )

    edge_schema = pd.DataFrame(edge_schema_rows)
    edge_schema_path = output_nodes_path.parent / "cd_graph_edge_schema.csv"
    edge_schema.to_csv(edge_schema_path, index=False)

    metadata = {
        "script": "urban_graph_benchmark/scripts/15_build_qc_cd_civil_security_graph.py",
        "purpose": "Build static Québec CD graph node and edge files for graph baselines.",
        "inputs": {
            "panel_path": str(panel_path),
            "boundaries_path": str(boundaries_path),
            "boundary_id_col": boundary_id_col,
            "boundary_name_col": boundary_name_col,
        },
        "crs": {
            "boundary_input_crs": str(boundaries_raw.crs),
            "metric_crs": metric_crs,
            "lonlat_crs": lonlat_crs,
            "assume_boundary_crs_if_missing": assume_boundary_crs,
        },
        "parameters": {
            "min_shared_boundary_km": float(min_shared_boundary_km),
            "knn_k_requested": int(knn_k),
            "knn_k_effective": int(min(knn_k, max(len(nodes) - 1, 1))),
            "random_seed": int(random_seed),
            "random_swaps_per_edge": int(random_swaps_per_edge),
            "allow_missing_boundaries": bool(allow_missing_boundaries),
        },
        "alignment": {
            "panel_cd_count": int(panel[CD_ID_COL].nunique()),
            "boundary_cd_count_total": int(boundaries_raw[CD_ID_COL].nunique()),
            "node_cd_count": int(nodes[CD_ID_COL].nunique()),
            "missing_panel_ids_from_boundaries": boundaries.attrs.get("missing_panel_ids", []),
            "extra_boundary_ids_not_in_panel_count": int(len(boundaries.attrs.get("extra_boundary_ids", []))),
        },
        "outputs": {
            **{f"nodes_{k}": v for k, v in node_outputs.items()},
            **{f"adjacency_edges_{k}": v for k, v in adjacency_outputs.items()},
            **{f"knn_edges_{k}": v for k, v in knn_outputs.items()},
            **{f"random_placebo_edges_{k}": v for k, v in random_outputs.items()},
            "graph_audit_csv": str(audit_path),
            "edge_schema_csv": str(edge_schema_path),
        },
        "graph_audit": audit.to_dict(orient="records"),
        "edge_storage_convention": {
            "adjacency": "undirected graph stored as bidirectional directed edges",
            "knn": "directed source-to-neighbor edges with k outgoing edges per node",
            "random_placebo": "undirected placebo graph stored as bidirectional directed edges",
            "self_loops": "not included",
        },
    }

    metadata_path = write_metadata_json(
        metadata,
        output_nodes_path.parent / "cd_graph_metadata.json",
    )
    metadata["outputs"]["metadata_json"] = str(metadata_path)

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Québec CD graph node and edge files for graph baselines."
    )

    parser.add_argument(
        "--panel-path",
        type=Path,
        default=DEFAULT_PANEL_PATH,
        help="Predictive CD-month panel used to define node IDs and static node features.",
    )
    parser.add_argument(
        "--boundaries-path",
        type=Path,
        default=DEFAULT_BOUNDARIES_PATH,
        help="Census division boundary file.",
    )
    parser.add_argument(
        "--output-nodes-path",
        type=Path,
        default=DEFAULT_NODES_PATH,
        help="Output node parquet path.",
    )
    parser.add_argument(
        "--output-adjacency-edges-path",
        type=Path,
        default=DEFAULT_ADJACENCY_EDGES_PATH,
        help="Output real adjacency edge parquet path.",
    )
    parser.add_argument(
        "--output-knn-edges-path",
        type=Path,
        default=DEFAULT_KNN_EDGES_PATH,
        help="Output kNN edge parquet path.",
    )
    parser.add_argument(
        "--output-random-edges-path",
        type=Path,
        default=DEFAULT_RANDOM_EDGES_PATH,
        help="Output random/placebo edge parquet path.",
    )
    parser.add_argument(
        "--boundary-id-col",
        default="CDUID",
        help="CD ID column in the boundary file.",
    )
    parser.add_argument(
        "--boundary-name-col",
        default="CDNAME",
        help="CD name column in the boundary file.",
    )
    parser.add_argument(
        "--metric-crs",
        default="EPSG:3347",
        help="Projected CRS used for distances/areas/boundary lengths.",
    )
    parser.add_argument(
        "--lonlat-crs",
        default="EPSG:4326",
        help="Longitude/latitude CRS used for centroid lon/lat output.",
    )
    parser.add_argument(
        "--assume-boundary-crs",
        default="EPSG:3347",
        help="CRS to assign if the boundary file has no CRS metadata.",
    )
    parser.add_argument(
        "--min-shared-boundary-km",
        type=float,
        default=0.001,
        help="Minimum shared boundary length for real adjacency edges, in km.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=5,
        help="Number of outgoing nearest-neighbor edges per node.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for placebo graph.",
    )
    parser.add_argument(
        "--random-swaps-per-edge",
        type=int,
        default=20,
        help="Approximate number of double-edge-swap attempts per adjacency edge.",
    )
    parser.add_argument(
        "--allow-missing-boundaries",
        action="store_true",
        help="Allow panel CDs that are missing from boundaries to be dropped from graph assets.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata = build_graph_assets(
        panel_path=args.panel_path,
        boundaries_path=args.boundaries_path,
        output_nodes_path=args.output_nodes_path,
        output_adjacency_edges_path=args.output_adjacency_edges_path,
        output_knn_edges_path=args.output_knn_edges_path,
        output_random_edges_path=args.output_random_edges_path,
        boundary_id_col=args.boundary_id_col,
        boundary_name_col=args.boundary_name_col,
        metric_crs=args.metric_crs,
        lonlat_crs=args.lonlat_crs,
        assume_boundary_crs=args.assume_boundary_crs,
        min_shared_boundary_km=args.min_shared_boundary_km,
        knn_k=args.knn_k,
        random_seed=args.random_seed,
        random_swaps_per_edge=args.random_swaps_per_edge,
        allow_missing_boundaries=args.allow_missing_boundaries,
    )

    print("Québec CD civil-security graph assets completed.")
    print(f"Panel path: {args.panel_path}")
    print(f"Boundary path: {args.boundaries_path}")
    print("Outputs:")
    for key, value in metadata["outputs"].items():
        print(f"  {key}: {value}")

    print("Graph audit:")
    for row in metadata["graph_audit"]:
        print(
            "  "
            f"{row['graph_name']}: "
            f"nodes={row['n_nodes']}, "
            f"directed_edges={row['n_directed_edges']}, "
            f"undirected_pairs={row['n_unique_undirected_pairs']}, "
            f"components={row['n_connected_components']}, "
            f"isolates={row['n_isolates']}"
        )


if __name__ == "__main__":
    main()
