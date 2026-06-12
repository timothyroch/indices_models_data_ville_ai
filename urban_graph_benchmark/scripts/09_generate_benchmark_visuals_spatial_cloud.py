#!/usr/bin/env python3
"""
Generate visuals for the index -> ML -> graph benchmark layer.

This script is intentionally post-hoc: it reads already-produced benchmark outputs
and graph artifacts, then writes figures and Cytoscape-compatible graph exports.
It does not train, tune, or reselect models.

Default inputs expected from the current benchmark pipeline:

  urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/08_index_ml_graph_benchmark/
  urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/
  urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_spatial_core_ndcg_monitor/
  urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/

Outputs include:

  figures/
    01_benchmark_metric_panels.png
    02_index_vs_learned_ranking_gap.png
    03_family_margin_panels.png
    04_g1_family_comparison.png
    05_g1_validation_sweep_heatmap.png
    06_benchmark_pipeline_schema.png
    07_tract_month_graph_sample.png
    08_one_month_spatial_graph_dense.png
    09_full_tract_month_graph_spatial_cloud.png
    10_full_artifact_graph_spatial_cloud_with_placebo.png

  cytoscape/
    tract_month_graph_sample_nodes.csv
    tract_month_graph_sample_edges.csv
    tract_month_graph_sample.cyjs
    tract_month_graph_sample.graphml  (when networkx is installed)
    one_month_spatial_graph_dense_nodes.csv
    one_month_spatial_graph_dense_edges.csv
    one_month_spatial_graph_dense.cyjs
    full_tract_month_graph_nodes.csv
    full_tract_month_graph_edges.csv
    full_tract_month_graph.cyjs
    full_artifact_graph_with_placebo_nodes.csv
    full_artifact_graph_with_placebo_edges.csv
    full_artifact_graph_with_placebo.cyjs

  visual_manifest.csv
  visual_summary.md

Recommended run from repo root:

  PYTHONPATH=urban_graph_benchmark/src python urban_graph_benchmark/scripts/09_generate_benchmark_visuals.py
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.collections import LineCollection

try:  # optional; only used for GraphML export and fallback layout.
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore[assignment]


STAGE_SLUG = "09_visuals_index_ml_graph"
DEFAULT_BASE_OUTPUT = Path("urban_graph_benchmark/outputs/mtl_311_water_v0")
DEFAULT_COMPARISON_DIR = DEFAULT_BASE_OUTPUT / "comparisons/08_index_ml_graph_benchmark"
DEFAULT_G15_DIR = DEFAULT_BASE_OUTPUT / "baselines/G1_5_validation_sweep_spatial_ndcg"
DEFAULT_G1_DIR = DEFAULT_BASE_OUTPUT / "baselines/G1_spatiotemporal_gnn_spatial_core_ndcg_monitor"
DEFAULT_GRAPH_DIR = DEFAULT_BASE_OUTPUT / "graphs/G1_tract_month_graph"
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_OUTPUT / "comparisons/09_visuals_index_ml_graph"

TARGET_COLUMN = "water_drainage_count"
ZONE_COL = "zone_id"
PERIOD_COL = "period_month"
NODE_ID_COL = "node_id"

METRIC_SPECS = [
    ("mae", "MAE", False),
    ("spearman", "Spearman", True),
    ("ndcg_at_100", "NDCG@100", True),
    ("top_10pct_overlap_rate", "Top-10% overlap", True),
]

GROUP_ORDER = [
    "Composite index",
    "Calibrated index",
    "Naive temporal baseline",
    "Tabular ML",
    "Neural control",
    "Graph/neural",
    "Placebo control",
]

GROUP_COLORS = {
    "Composite index": "#9e9e9e",
    "Calibrated index": "#757575",
    "Naive temporal baseline": "#8d6e63",
    "Tabular ML": "#2f6f9f",
    "Neural control": "#6a51a3",
    "Graph/neural": "#1b9e77",
    "Placebo control": "#d95f02",
}

EDGE_COLORS = {
    "temporal_self_lag_1": "#756bb1",
    "temporal_self_lag_12": "#9e9ac8",
    "spatial_knn_same_month": "#31a354",
    "spatial_knn_same_month_random_placebo": "#e6550d",
}


class VisualError(RuntimeError):
    """Raised when visual generation fails."""


@dataclass(frozen=True)
class VisualPaths:
    comparison_dir: Path
    g15_dir: Path
    g1_dir: Path
    graph_dir: Path
    output_dir: Path

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def cytoscape_dir(self) -> Path:
        return self.output_dir / "cytoscape"


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_if_exists(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise VisualError(f"Missing required CSV: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def read_parquet_if_exists(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise VisualError(f"Missing required parquet: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def savefig(path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def shorten_label(label: str, max_len: int = 34) -> str:
    label = str(label)
    if len(label) <= max_len:
        return label
    return label[: max_len - 1] + "…"


def group_color(group: str) -> str:
    return GROUP_COLORS.get(str(group), "#4d4d4d")


def metric_sort_value(series: pd.Series, higher_is_better: bool) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals if higher_is_better else -vals


def annotate_bar_values(ax: plt.Axes, bars: Iterable[Any], values: Sequence[float], fmt: str = "{:.3f}") -> None:
    for bar, val in zip(bars, values):
        if not math.isfinite(float(val)):
            continue
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            " " + fmt.format(float(val)),
            va="center",
            ha="left",
            fontsize=8,
        )


# ---------------------------------------------------------------------------
# Plot 01: benchmark metric panels
# ---------------------------------------------------------------------------


def plot_benchmark_metric_panels(compact: pd.DataFrame, output_path: Path) -> None:
    """Create 2x2 metric panel from compact comparison table."""

    if compact.empty:
        raise VisualError("compact comparison table is empty.")

    rows = compact.copy()
    rows["label_short"] = rows["label"].map(lambda x: shorten_label(str(x), 38))
    rows["comparison_group"] = rows["comparison_group"].fillna("Other")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for ax, (metric, label, higher) in zip(axes, METRIC_SPECS):
        sub = rows[pd.to_numeric(rows[metric], errors="coerce").notna()].copy()
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{label}: unavailable")
            continue
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub.sort_values(metric, ascending=not higher).head(12)
        sub = sub.iloc[::-1]  # best at top after horizontal bar inversion.
        colors = [group_color(g) for g in sub["comparison_group"]]
        bars = ax.barh(sub["label_short"], sub[metric], color=colors, alpha=0.88)
        annotate_bar_values(ax, bars, sub[metric].tolist())
        ax.set_title(f"{label} ({'higher is better' if higher else 'lower is better'})")
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", labelsize=8)
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax * 1.18 if xmax > 0 else xmax)

    handles = []
    seen = set()
    for group in GROUP_ORDER:
        if group in set(rows["comparison_group"]):
            handles.append(plt.Line2D([0], [0], color=group_color(group), lw=8, label=group))
            seen.add(group)
    for group in sorted(set(rows["comparison_group"]) - seen):
        handles.append(plt.Line2D([0], [0], color=group_color(group), lw=8, label=group))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Index vs Tabular ML vs Graph/Neural Benchmark", fontsize=16, y=0.995)
    fig.subplots_adjust(bottom=0.14)
    savefig(output_path)


# ---------------------------------------------------------------------------
# Plot 02: index-vs-learned ranking gap
# ---------------------------------------------------------------------------


def best_by_group(compact: pd.DataFrame, metric: str, groups: Sequence[str], higher: bool) -> pd.DataFrame:
    rows = compact[compact["comparison_group"].isin(groups)].copy()
    rows[metric] = pd.to_numeric(rows[metric], errors="coerce")
    rows = rows[rows[metric].notna()]
    if rows.empty:
        return rows
    out = []
    for group, sub in rows.groupby("comparison_group"):
        sub = sub.sort_values(metric, ascending=not higher)
        out.append(sub.iloc[0])
    return pd.DataFrame(out)


def plot_index_vs_learned_gap(compact: pd.DataFrame, output_path: Path) -> None:
    """Show gap between best index rows and learned benchmark rows on ranking metrics."""

    comparison_groups = ["Composite index", "Calibrated index", "Tabular ML", "Neural control", "Graph/neural", "Placebo control"]
    records = []
    for metric, label, higher in [m for m in METRIC_SPECS if m[0] != "mae"]:
        best = best_by_group(compact, metric, comparison_groups, higher=True)
        for _, row in best.iterrows():
            records.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "comparison_group": row["comparison_group"],
                    "label": row["label"],
                    "value": row[metric],
                }
            )
    df = pd.DataFrame(records)
    if df.empty:
        raise VisualError("No ranking metrics available for index-vs-learned gap plot.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, metric_label in zip(axes, ["Spearman", "NDCG@100", "Top-10% overlap"]):
        sub = df[df["metric_label"] == metric_label].copy()
        sub["comparison_group"] = pd.Categorical(sub["comparison_group"], categories=comparison_groups, ordered=True)
        sub = sub.sort_values("comparison_group")
        bars = ax.bar(
            sub["comparison_group"].astype(str),
            sub["value"],
            color=[group_color(g) for g in sub["comparison_group"].astype(str)],
            alpha=0.9,
        )
        for bar, val in zip(bars, sub["value"]):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8, rotation=0)
        ax.set_title(metric_label)
        ax.set_ylim(0, max(0.05, float(sub["value"].max()) * 1.18))
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Best available ranking performance by benchmark family", fontsize=15)
    savefig(output_path)


# ---------------------------------------------------------------------------
# Plot 03: family margins
# ---------------------------------------------------------------------------


def plot_family_margin_panels(margins: pd.DataFrame, output_path: Path) -> None:
    """Plot family margins from family_margin_table.csv."""

    if margins.empty:
        raise VisualError("family margin table is empty.")
    rows = margins.copy()
    rows["positive_margin_means_left_better"] = pd.to_numeric(rows["positive_margin_means_left_better"], errors="coerce")
    rows = rows[rows["positive_margin_means_left_better"].notna()]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()
    for ax, (metric, label, _higher) in zip(axes, METRIC_SPECS):
        sub = rows[rows["metric"] == metric].copy()
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{label}: unavailable")
            continue
        sub["comparison_short"] = sub["comparison"].map(lambda x: shorten_label(str(x), 42))
        sub = sub.sort_values("positive_margin_means_left_better")
        vals = sub["positive_margin_means_left_better"].to_numpy(dtype=float)
        colors = ["#1b9e77" if v >= 0 else "#d95f02" for v in vals]
        ax.barh(sub["comparison_short"], vals, color=colors, alpha=0.9)
        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_title(label)
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Benchmark family margins (positive = left side better)", fontsize=15)
    savefig(output_path)


# ---------------------------------------------------------------------------
# Plot 04: G1 family comparison
# ---------------------------------------------------------------------------


def plot_g1_family_comparison(final_comparison: pd.DataFrame, output_path: Path) -> None:
    """Plot G1.5 family representatives."""

    if final_comparison.empty:
        raise VisualError("G1.5 final_comparison.csv is unavailable or empty.")

    rows = final_comparison.copy()
    role_col = "comparison_role" if "comparison_role" in rows.columns else "role"
    family_col = "family" if "family" in rows.columns else "winner_family"
    rows["label"] = rows.get(role_col, rows.get(family_col, pd.Series(range(len(rows))))).astype(str)
    rows = rows[rows["label"].str.contains("G1|A3", case=False, regex=True, na=False)].copy()
    if rows.empty:
        rows = final_comparison.copy()
        rows["label"] = rows.get(family_col, pd.Series(range(len(rows)))).astype(str)

    rename = {
        "test_mae": "mae",
        "test_spearman": "spearman",
        "test_ndcg_at_100": "ndcg_at_100",
        "test_top_10pct_overlap_rate": "top_10pct_overlap_rate",
    }
    for src, dst in rename.items():
        if dst not in rows.columns and src in rows.columns:
            rows[dst] = rows[src]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, (metric, label, higher) in zip(axes, METRIC_SPECS):
        if metric not in rows.columns:
            ax.axis("off")
            ax.set_title(f"{label}: unavailable")
            continue
        sub = rows[pd.to_numeric(rows[metric], errors="coerce").notna()].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub.sort_values(metric, ascending=not higher)
        sub = sub.iloc[::-1]
        colors = [group_color("Placebo control") if "placebo" in str(x).lower() else group_color("Neural control") if "no_edges" in str(x).lower() else group_color("Tabular ML") if "A3" in str(x) else group_color("Graph/neural") for x in sub["label"]]
        bars = ax.barh(sub["label"].map(lambda x: shorten_label(str(x), 38)), sub[metric], color=colors, alpha=0.9)
        annotate_bar_values(ax, bars, sub[metric].tolist())
        ax.set_title(label)
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", labelsize=8)
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax * 1.16 if xmax > 0 else xmax)
    fig.suptitle("Selected G1.5 family representatives", fontsize=15)
    savefig(output_path)


# ---------------------------------------------------------------------------
# Plot 05: validation sweep heatmap
# ---------------------------------------------------------------------------


def parse_arch_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure architecture columns exist if possible."""

    out = df.copy()
    for col in ["hidden_dim", "num_layers", "dropout"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def plot_validation_sweep_heatmap(selection_audit: pd.DataFrame, output_path: Path) -> None:
    """Plot validation NDCG@100 heatmap by hidden_dim/layers for each family."""

    if selection_audit.empty:
        raise VisualError("sweep_model_selection_audit.csv is unavailable or empty.")
    rows = parse_arch_columns(selection_audit)

    value_col = "validation_ndcg_at_100"
    if value_col not in rows.columns:
        raise VisualError(f"{value_col} not present in sweep audit.")
    rows[value_col] = pd.to_numeric(rows[value_col], errors="coerce")
    rows = rows[rows[value_col].notna()].copy()
    if rows.empty:
        raise VisualError("No finite validation NDCG@100 values in sweep audit.")

    if "family" not in rows.columns:
        if "edge_regime" in rows.columns:
            rows["family"] = rows["edge_regime"]
        else:
            rows["family"] = "model"

    families = [f for f in ["no_edges", "temporal_only", "spatial_temporal", "random_spatial_placebo"] if f in set(rows["family"].astype(str))]
    if not families:
        families = list(rows["family"].astype(str).drop_duplicates().head(4))

    fig, axes = plt.subplots(1, len(families), figsize=(4.5 * len(families), 4.2), squeeze=False)
    axes = axes.flatten()
    for ax, family in zip(axes, families):
        sub = rows[rows["family"].astype(str) == family].copy()
        if sub.empty or "hidden_dim" not in sub.columns or "num_layers" not in sub.columns:
            ax.axis("off")
            ax.set_title(family)
            continue
        pivot = sub.pivot_table(index="num_layers", columns="hidden_dim", values=value_col, aggfunc="max")
        pivot = pivot.sort_index().sort_index(axis=1)
        im = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) if float(c).is_integer() else str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(int(c)) if float(c).is_integer() else str(c) for c in pivot.index])
        ax.set_xlabel("hidden_dim")
        ax.set_ylabel("num_layers")
        ax.set_title(family)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.to_numpy()[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("G1.5 validation NDCG@100 sweep: best value by family / depth / width", fontsize=14)
    savefig(output_path)


# ---------------------------------------------------------------------------
# Plot 06: conceptual pipeline schema
# ---------------------------------------------------------------------------


def draw_box(ax: plt.Axes, xy: tuple[float, float], w: float, h: float, text: str, color: str) -> None:
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.2,
        edgecolor="#333333",
        facecolor=color,
        alpha=0.88,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=10, wrap=True)


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="#333333")
    ax.add_patch(arrow)


def plot_pipeline_schema(output_path: Path) -> None:
    """Draw a conceptual benchmark pipeline diagram."""

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        ((0.4, 2.7), 2.0, 1.0, "Static composite\nindices\nSVI / SoVI", "#d9d9d9"),
        ((0.4, 1.1), 2.0, 1.0, "Observed 311\ntract-month\ntarget", "#f0f0f0"),
        ((3.0, 2.7), 2.3, 1.0, "Calibrated index\nA2", "#bdbdbd"),
        ((5.9, 2.7), 2.3, 1.0, "Feature-parity\ntabular ML\nA3", "#9ecae1"),
        ((8.8, 3.25), 2.3, 1.0, "No-edge neural\ncontrol", "#bcbddc"),
        ((8.8, 1.9), 2.3, 1.0, "Typed tract-month\ngraph G1/G1.5", "#a1d99b"),
        ((11.7, 2.55), 1.9, 1.0, "High-risk\nranking metrics\nNDCG / top-k", "#fee6ce"),
    ]
    for xy, w, h, text, color in boxes:
        draw_box(ax, xy, w, h, text, color)

    draw_arrow(ax, (2.4, 3.2), (3.0, 3.2))
    draw_arrow(ax, (5.3, 3.2), (5.9, 3.2))
    draw_arrow(ax, (8.2, 3.2), (8.8, 3.75))
    draw_arrow(ax, (8.2, 3.05), (8.8, 2.4))
    draw_arrow(ax, (11.1, 3.75), (11.7, 3.1))
    draw_arrow(ax, (11.1, 2.4), (11.7, 3.0))
    draw_arrow(ax, (1.4, 2.1), (1.4, 2.7))
    draw_arrow(ax, (1.4, 1.1), (6.95, 2.7))

    ax.text(7.0, 0.55, "Benchmark principle: compare static indices, supervised ML, graph/neural models, and controls on held-out ranking metrics.", ha="center", fontsize=11)
    ax.set_title("Index → ML → Graph/Neural Benchmark Layer", fontsize=16)
    savefig(output_path)


# ---------------------------------------------------------------------------
# Graph sample and Cytoscape exports
# ---------------------------------------------------------------------------


def choose_graph_month(node_table: pd.DataFrame, requested: str) -> str:
    """Choose a representative month for graph visualization."""

    if requested and requested != "auto":
        return str(requested)

    rows = node_table.copy()
    target_col = TARGET_COLUMN if TARGET_COLUMN in rows.columns else None
    split_candidates = ["split_spatial_block", "spatial_block_split", "split_temporal", "temporal_split"]
    split_col = next((c for c in split_candidates if c in rows.columns), None)
    if split_col is not None:
        test_rows = rows[rows[split_col].astype(str).str.lower().eq("test")].copy()
        if not test_rows.empty:
            rows = test_rows

    if target_col is not None:
        month_scores = rows.groupby(PERIOD_COL)[target_col].sum().sort_values(ascending=False)
        if not month_scores.empty:
            return str(month_scores.index[0])
    return str(rows[PERIOD_COL].max())


def build_graph_sample(
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    *,
    month: str,
    anchors: int,
    max_nodes: int,
    include_temporal: bool,
    include_placebo: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small tract-month graph sample for visualization/export."""

    if NODE_ID_COL not in node_table.columns:
        raise VisualError(f"node_table is missing {NODE_ID_COL!r}.")
    if PERIOD_COL not in node_table.columns:
        raise VisualError(f"node_table is missing {PERIOD_COL!r}.")

    month_nodes = node_table[node_table[PERIOD_COL].astype(str) == str(month)].copy()
    if month_nodes.empty:
        raise VisualError(f"No nodes found for period_month={month!r}.")

    score_col = TARGET_COLUMN if TARGET_COLUMN in month_nodes.columns else "requests_total" if "requests_total" in month_nodes.columns else NODE_ID_COL
    month_nodes["_sample_score"] = pd.to_numeric(month_nodes[score_col], errors="coerce").fillna(0.0)
    anchor_nodes = month_nodes.sort_values("_sample_score", ascending=False).head(max(1, anchors))
    selected_ids: set[int] = set(anchor_nodes[NODE_ID_COL].astype(int).tolist())

    # Include spatial neighbors from the same month around anchors.
    spatial_types = {"spatial_knn_same_month"}
    if include_placebo:
        spatial_types.add("spatial_knn_same_month_random_placebo")

    same_month_edges = edge_table[
        edge_table.get("target_period_month", edge_table.get("source_period_month", "")).astype(str).eq(str(month))
        & edge_table["edge_type"].astype(str).isin(spatial_types)
    ].copy()
    neighborhood = same_month_edges[
        same_month_edges["source_node_id"].astype(int).isin(selected_ids)
        | same_month_edges["target_node_id"].astype(int).isin(selected_ids)
    ].copy()

    if not neighborhood.empty:
        edge_node_ids = set(neighborhood["source_node_id"].astype(int).tolist()) | set(neighborhood["target_node_id"].astype(int).tolist())
        selected_ids |= set(list(edge_node_ids)[: max(0, max_nodes - len(selected_ids))])

    # Trim to max_nodes using target score, preserving anchors first.
    selected_nodes = node_table[node_table[NODE_ID_COL].astype(int).isin(selected_ids)].copy()
    if len(selected_nodes) > max_nodes:
        selected_nodes["_is_anchor"] = selected_nodes[NODE_ID_COL].astype(int).isin(set(anchor_nodes[NODE_ID_COL].astype(int)))
        selected_nodes["_score"] = pd.to_numeric(selected_nodes.get(score_col, 0.0), errors="coerce").fillna(0.0)
        selected_nodes = selected_nodes.sort_values(["_is_anchor", "_score"], ascending=[False, False]).head(max_nodes)
        selected_ids = set(selected_nodes[NODE_ID_COL].astype(int).tolist())

    edge_mask = edge_table["source_node_id"].astype(int).isin(selected_ids) & edge_table["target_node_id"].astype(int).isin(selected_ids)
    if not include_temporal:
        edge_mask &= ~edge_table["edge_type"].astype(str).str.startswith("temporal_")
    if not include_placebo:
        edge_mask &= ~edge_table["edge_type"].astype(str).str.contains("placebo", case=False, regex=True)
    sample_edges = edge_table[edge_mask].copy()

    # Limit edge count for a legible figure while preserving real spatial and temporal edges.
    if len(sample_edges) > max_nodes * 4:
        sample_edges["_priority"] = sample_edges["edge_type"].map(
            lambda x: 0 if str(x) == "spatial_knn_same_month" else 1 if str(x).startswith("temporal") else 2
        )
        sample_edges = sample_edges.sort_values(["_priority", "source_node_id", "target_node_id"]).head(max_nodes * 4)

    selected_ids = set(sample_edges["source_node_id"].astype(int).tolist()) | set(sample_edges["target_node_id"].astype(int).tolist()) | selected_ids
    sample_nodes = node_table[node_table[NODE_ID_COL].astype(int).isin(selected_ids)].copy()
    sample_nodes["is_anchor"] = sample_nodes[NODE_ID_COL].astype(int).isin(set(anchor_nodes[NODE_ID_COL].astype(int)))
    sample_nodes["sample_month"] = month
    sample_nodes["sample_score_column"] = score_col
    sample_nodes["sample_score"] = pd.to_numeric(sample_nodes.get(score_col, 0.0), errors="coerce").fillna(0.0)

    return sample_nodes.reset_index(drop=True), sample_edges.reset_index(drop=True)


def normalize_positions(nodes: pd.DataFrame) -> pd.DataFrame:
    """Create normalized visual_x/visual_y columns."""

    out = nodes.copy()
    if "graph_x" in out.columns and "graph_y" in out.columns:
        x = pd.to_numeric(out["graph_x"], errors="coerce")
        y = pd.to_numeric(out["graph_y"], errors="coerce")
        if x.notna().sum() >= 2 and y.notna().sum() >= 2:
            x_range = float(x.max() - x.min()) or 1.0
            y_range = float(y.max() - y.min()) or 1.0
            out["visual_x"] = (x - x.min()) / x_range
            out["visual_y"] = (y - y.min()) / y_range
            return out

    # fallback: circular layout
    n = len(out)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    out["visual_x"] = 0.5 + 0.45 * np.cos(theta)
    out["visual_y"] = 0.5 + 0.45 * np.sin(theta)
    return out


def plot_graph_sample(nodes: pd.DataFrame, edges: pd.DataFrame, output_path: Path) -> None:
    """Plot a Cytoscape-like sample of tract-month graph."""

    if nodes.empty:
        raise VisualError("graph sample nodes are empty.")
    nodes = normalize_positions(nodes)
    pos = {
        int(row[NODE_ID_COL]): (float(row["visual_x"]), float(row["visual_y"]))
        for _, row in nodes.iterrows()
    }

    fig, ax = plt.subplots(figsize=(10.5, 9))
    ax.set_aspect("equal")
    ax.axis("off")

    # Edges first.
    for _, edge in edges.iterrows():
        src = int(edge["source_node_id"])
        dst = int(edge["target_node_id"])
        if src not in pos or dst not in pos:
            continue
        etype = str(edge.get("edge_type", "edge"))
        color = EDGE_COLORS.get(etype, "#737373")
        alpha = 0.32 if "spatial" in etype else 0.22
        lw = 0.8 if "spatial" in etype else 0.65
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=lw, zorder=1)

    scores = pd.to_numeric(nodes.get("sample_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sizes = 40 + 260 * (np.log1p(scores) / max(float(np.log1p(scores).max()), 1.0))
    is_anchor = nodes.get("is_anchor", False).astype(bool).to_numpy()
    colors = np.where(is_anchor, "#e31a1c", "#3182bd")

    ax.scatter(nodes["visual_x"], nodes["visual_y"], s=sizes, c=colors, edgecolors="white", linewidths=0.8, alpha=0.92, zorder=3)

    # Label only anchors to keep figure readable.
    for _, row in nodes[nodes.get("is_anchor", False).astype(bool)].iterrows():
        label = f"{row.get(ZONE_COL, row[NODE_ID_COL])}\n{int(row.get('sample_score', 0))}"
        ax.text(float(row["visual_x"]), float(row["visual_y"]) + 0.025, label, ha="center", va="bottom", fontsize=7, zorder=4)

    edge_handles = []
    for etype, color in EDGE_COLORS.items():
        if not edges.empty and etype in set(edges["edge_type"].astype(str)):
            edge_handles.append(plt.Line2D([0], [0], color=color, lw=2, label=etype))
    node_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e31a1c", markersize=8, label="anchor/top burden"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#3182bd", markersize=8, label="neighbor"),
    ]
    ax.legend(handles=edge_handles + node_handles, loc="lower left", fontsize=8, frameon=True)

    month = str(nodes["sample_month"].iloc[0]) if "sample_month" in nodes.columns else "sample"
    ax.set_title(f"Tract-month graph sample ({month}) — Cytoscape-style view", fontsize=14)
    savefig(output_path)


def export_cytoscape(nodes: pd.DataFrame, edges: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    """Write Cytoscape-compatible CSV, Cytoscape.js JSON, and optional GraphML."""

    ensure_dir(output_dir)
    nodes = normalize_positions(nodes)
    node_path = output_dir / "tract_month_graph_sample_nodes.csv"
    edge_path = output_dir / "tract_month_graph_sample_edges.csv"
    cyjs_path = output_dir / "tract_month_graph_sample.cyjs"
    graphml_path = output_dir / "tract_month_graph_sample.graphml"

    node_export_cols = [
        c for c in [
            NODE_ID_COL,
            "node_key",
            ZONE_COL,
            PERIOD_COL,
            "period_index",
            TARGET_COLUMN,
            "requests_total",
            "svi_percentile",
            "svi_class",
            "sample_score",
            "is_anchor",
            "visual_x",
            "visual_y",
            "graph_x",
            "graph_y",
        ] if c in nodes.columns
    ]
    nodes[node_export_cols].to_csv(node_path, index=False)

    edge_export = edges.copy()
    if "edge_id" not in edge_export.columns:
        edge_export.insert(0, "edge_id", [f"e{i}" for i in range(len(edge_export))])
    edge_export_cols = [
        c for c in [
            "edge_id",
            "source_node_id",
            "target_node_id",
            "edge_type",
            "edge_weight",
            "distance_m",
            "source_zone_id",
            "target_zone_id",
            "source_period_month",
            "target_period_month",
            "is_temporal",
            "is_spatial",
            "is_placebo",
        ] if c in edge_export.columns
    ]
    edge_export[edge_export_cols].to_csv(edge_path, index=False)

    elements: list[dict[str, Any]] = []
    for _, row in nodes.iterrows():
        node_id = str(int(row[NODE_ID_COL]))
        data = {k: (None if pd.isna(v) else v) for k, v in row[node_export_cols].to_dict().items()}
        data["id"] = node_id
        data["label"] = str(row.get(ZONE_COL, node_id))
        elements.append(
            {
                "data": data,
                "position": {
                    "x": float(row["visual_x"]) * 1000.0,
                    "y": (1.0 - float(row["visual_y"])) * 1000.0,
                },
            }
        )
    for i, row in edge_export.iterrows():
        data = {k: (None if pd.isna(v) else v) for k, v in row[edge_export_cols].to_dict().items()}
        data["id"] = str(data.get("edge_id", f"e{i}"))
        data["source"] = str(int(row["source_node_id"]))
        data["target"] = str(int(row["target_node_id"]))
        elements.append({"data": data})
    cyjs = {
        "format_version": "1.0",
        "generated_by": "09_generate_benchmark_visuals.py",
        "data": {"name": "tract_month_graph_sample"},
        "elements": elements,
    }
    write_json(cyjs_path, cyjs)

    outputs = {"cytoscape_nodes": str(node_path), "cytoscape_edges": str(edge_path), "cytoscape_cyjs": str(cyjs_path)}

    if nx is not None:
        G = nx.DiGraph()
        for _, row in nodes.iterrows():
            attrs = {k: ("" if pd.isna(v) else v) for k, v in row[node_export_cols].to_dict().items()}
            G.add_node(str(int(row[NODE_ID_COL])), **attrs)
        for i, row in edge_export.iterrows():
            attrs = {k: ("" if pd.isna(v) else v) for k, v in row[edge_export_cols].to_dict().items() if k not in {"source_node_id", "target_node_id"}}
            G.add_edge(str(int(row["source_node_id"])), str(int(row["target_node_id"])), **attrs)
        nx.write_graphml(G, graphml_path)
        outputs["cytoscape_graphml"] = str(graphml_path)

    return outputs




# ---------------------------------------------------------------------------
# Dense/full graph representations
# ---------------------------------------------------------------------------

REAL_GRAPH_EDGE_TYPES = {
    "temporal_self_lag_1",
    "temporal_self_lag_12",
    "spatial_knn_same_month",
}
SPATIAL_EDGE_TYPES = {"spatial_knn_same_month"}
PLACEBO_EDGE_TYPES = {"spatial_knn_same_month_random_placebo"}


def _json_safe(value: Any) -> Any:
    """Small JSON-safe conversion for Cytoscape exports."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return val if math.isfinite(val) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def hex_to_rgba(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    """Convert #RRGGBB to matplotlib RGBA tuple."""
    value = hex_color.lstrip("#")
    if len(value) != 6:
        return (0.2, 0.2, 0.2, alpha)
    r = int(value[0:2], 16) / 255.0
    g = int(value[2:4], 16) / 255.0
    b = int(value[4:6], 16) / 255.0
    return (r, g, b, alpha)


def edge_color_for_type(edge_type: str, alpha: float) -> tuple[float, float, float, float]:
    """Return RGBA edge color for edge type."""
    return hex_to_rgba(EDGE_COLORS.get(str(edge_type), "#737373"), alpha)


def add_normalized_spatial_positions(nodes: pd.DataFrame) -> pd.DataFrame:
    """Add spatial_x/spatial_y normalized to [0, 1] using graph coordinates when available."""
    out = nodes.copy()
    candidates = [("graph_x", "graph_y"), ("centroid_x", "centroid_y"), ("x", "y")]
    for x_col, y_col in candidates:
        if x_col in out.columns and y_col in out.columns:
            x = pd.to_numeric(out[x_col], errors="coerce")
            y = pd.to_numeric(out[y_col], errors="coerce")
            if x.notna().sum() >= 2 and y.notna().sum() >= 2:
                x_range = float(x.max() - x.min()) or 1.0
                y_range = float(y.max() - y.min()) or 1.0
                out["spatial_x"] = (x - x.min()) / x_range
                out["spatial_y"] = (y - y.min()) / y_range
                return out

    # Stable fallback: lay out unique zones on a deterministic circle, then repeat by month.
    zone_col = ZONE_COL if ZONE_COL in out.columns else NODE_ID_COL
    zones = sorted(out[zone_col].astype(str).unique())
    theta = np.linspace(0, 2 * np.pi, len(zones), endpoint=False)
    zone_pos = {
        zone: (0.5 + 0.45 * math.cos(t), 0.5 + 0.45 * math.sin(t))
        for zone, t in zip(zones, theta)
    }
    out["spatial_x"] = out[zone_col].astype(str).map(lambda z: zone_pos[z][0]).astype(float)
    out["spatial_y"] = out[zone_col].astype(str).map(lambda z: zone_pos[z][1]).astype(float)
    return out


def add_layered_tract_month_positions(nodes: pd.DataFrame, *, month_jitter: float = 0.28) -> pd.DataFrame:
    """Add full_x/full_y coordinates for an interpretable full tract-month graph view.

    x is the normalized tract spatial coordinate. y is month index with a small
    spatial-y jitter inside each month band. This makes the full 28k-node graph
    visible as a temporal stack rather than a single force-layout hairball.
    """
    out = add_normalized_spatial_positions(nodes)
    if "period_index" in out.columns:
        pidx = pd.to_numeric(out["period_index"], errors="coerce")
    else:
        months = {m: i for i, m in enumerate(sorted(out[PERIOD_COL].astype(str).unique()))}
        pidx = out[PERIOD_COL].astype(str).map(months).astype(float)
    pidx = pidx.fillna(0.0)
    out["full_x"] = pd.to_numeric(out["spatial_x"], errors="coerce").fillna(0.5)
    out["full_y"] = pidx + month_jitter * (pd.to_numeric(out["spatial_y"], errors="coerce").fillna(0.5) - 0.5)
    return out


def edges_between_nodes(edges: pd.DataFrame, node_ids: set[int], edge_types: set[str] | None = None) -> pd.DataFrame:
    """Return edges whose endpoints are both in node_ids and optionally match edge_types."""
    if edges.empty:
        return edges.copy()
    mask = edges["source_node_id"].astype(int).isin(node_ids) & edges["target_node_id"].astype(int).isin(node_ids)
    if edge_types is not None:
        mask &= edges["edge_type"].astype(str).isin(edge_types)
    return edges.loc[mask].copy()


def build_one_month_spatial_graph(
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    *,
    month: str,
    include_placebo: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build all-tract one-month spatial graph: roughly 540 nodes and kNN edges."""
    month_nodes = node_table[node_table[PERIOD_COL].astype(str) == str(month)].copy()
    if month_nodes.empty:
        raise VisualError(f"No nodes found for one-month graph at period_month={month!r}.")
    node_ids = set(month_nodes[NODE_ID_COL].astype(int).tolist())
    edge_types = set(SPATIAL_EDGE_TYPES)
    if include_placebo:
        edge_types |= PLACEBO_EDGE_TYPES
    month_edges = edges_between_nodes(edge_table, node_ids, edge_types=edge_types)
    month_nodes = add_normalized_spatial_positions(month_nodes)
    score_col = TARGET_COLUMN if TARGET_COLUMN in month_nodes.columns else "requests_total" if "requests_total" in month_nodes.columns else NODE_ID_COL
    month_nodes["plot_score"] = pd.to_numeric(month_nodes.get(score_col, 0.0), errors="coerce").fillna(0.0)
    month_nodes["plot_month"] = str(month)
    return month_nodes.reset_index(drop=True), month_edges.reset_index(drop=True)


def _edge_segments_from_positions(edges: pd.DataFrame, nodes: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Join edge endpoints to coordinates."""
    pos = nodes[[NODE_ID_COL, x_col, y_col]].copy()
    pos[NODE_ID_COL] = pos[NODE_ID_COL].astype(int)
    left = pos.rename(columns={NODE_ID_COL: "source_node_id", x_col: "x1", y_col: "y1"})
    right = pos.rename(columns={NODE_ID_COL: "target_node_id", x_col: "x2", y_col: "y2"})
    joined = edges.copy()
    joined["source_node_id"] = joined["source_node_id"].astype(int)
    joined["target_node_id"] = joined["target_node_id"].astype(int)
    joined = joined.merge(left, on="source_node_id", how="inner").merge(right, on="target_node_id", how="inner")
    return joined


def plot_one_month_spatial_graph_dense(nodes: pd.DataFrame, edges: pd.DataFrame, output_path: Path) -> None:
    """Plot full one-month spatial graph, dense but still geographically interpretable."""
    if nodes.empty:
        raise VisualError("one-month dense nodes are empty.")
    joined = _edge_segments_from_positions(edges, nodes, "spatial_x", "spatial_y") if not edges.empty else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")
    ax.axis("off")

    if not joined.empty:
        for edge_type, sub in joined.groupby("edge_type", sort=True):
            segs = sub[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float).reshape(-1, 2, 2)
            lc = LineCollection(
                segs,
                colors=[edge_color_for_type(str(edge_type), 0.115)],
                linewidths=0.28,
                zorder=1,
                rasterized=True,
            )
            ax.add_collection(lc)

    scores = pd.to_numeric(nodes.get("plot_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    score_scale = np.log1p(scores) / max(float(np.log1p(scores).max()), 1.0)
    node_sizes = 7.0 + 75.0 * score_scale
    scatter = ax.scatter(
        nodes["spatial_x"],
        nodes["spatial_y"],
        s=node_sizes,
        c=np.log1p(scores),
        cmap="viridis",
        edgecolors="none",
        alpha=0.9,
        zorder=3,
        rasterized=True,
    )
    top = nodes.assign(_score=scores).sort_values("_score", ascending=False).head(10)
    ax.scatter(top["spatial_x"], top["spatial_y"], s=110, facecolors="none", edgecolors="#e31a1c", linewidths=1.0, zorder=4)

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.030, pad=0.015)
    cbar.set_label("log1p(water/drainage count)", fontsize=9)
    month = str(nodes["plot_month"].iloc[0]) if "plot_month" in nodes.columns else "one month"
    ax.set_title(
        f"Dense one-month spatial graph ({month}) — {len(nodes):,} tract nodes, {len(edges):,} spatial edges",
        fontsize=15,
    )
    ax.text(
        0.01,
        0.01,
        "All tract nodes for one month; edges are k-nearest spatial links. Red rings mark highest-burden tract-months.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )
    savefig(output_path, dpi=300)


def build_full_edge_table(edge_table: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    """Return full graph edge table for real graph or full artifact graph."""
    mode = str(mode)
    if mode == "real":
        return edge_table[edge_table["edge_type"].astype(str).isin(REAL_GRAPH_EDGE_TYPES)].copy()
    if mode == "artifact":
        return edge_table.copy()
    raise VisualError(f"Unknown full edge mode {mode!r}; expected 'real' or 'artifact'.")


def plot_full_tract_month_hairball(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    max_edges_to_draw: int = 600_000,
) -> None:
    """Plot full tract-month graph using layered month geometry and very transparent edges."""
    if nodes.empty:
        raise VisualError("full graph nodes are empty.")
    nodes = add_layered_tract_month_positions(nodes)
    draw_edges = edges.copy()
    if len(draw_edges) > max_edges_to_draw:
        # Deterministic stratified thinning by edge type to protect rendering time.
        parts = []
        per_type = max(1, int(max_edges_to_draw / max(1, draw_edges["edge_type"].nunique())))
        for _, sub in draw_edges.groupby("edge_type", sort=True):
            if len(sub) > per_type:
                parts.append(sub.sample(n=per_type, random_state=20240610))
            else:
                parts.append(sub)
        draw_edges = pd.concat(parts, ignore_index=True) if parts else draw_edges.head(max_edges_to_draw)

    joined = _edge_segments_from_positions(draw_edges, nodes, "full_x", "full_y") if not draw_edges.empty else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_facecolor("#fbfbfb")

    if not joined.empty:
        for edge_type, sub in joined.groupby("edge_type", sort=True):
            segs = sub[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float).reshape(-1, 2, 2)
            if str(edge_type).startswith("temporal"):
                alpha = 0.020
                lw = 0.055
            elif "placebo" in str(edge_type):
                alpha = 0.009
                lw = 0.045
            else:
                alpha = 0.012
                lw = 0.045
            lc = LineCollection(
                segs,
                colors=[edge_color_for_type(str(edge_type), alpha)],
                linewidths=lw,
                zorder=1,
                rasterized=True,
            )
            ax.add_collection(lc)

    scores = pd.to_numeric(nodes.get(TARGET_COLUMN, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    score_scale = np.log1p(scores) / max(float(np.log1p(scores).max()), 1.0)
    ax.scatter(
        nodes["full_x"],
        nodes["full_y"],
        s=0.55 + 4.0 * score_scale,
        c="#252525",
        alpha=0.22,
        linewidths=0,
        zorder=2,
        rasterized=True,
    )
    top = nodes.assign(_score=scores).sort_values("_score", ascending=False).head(80)
    ax.scatter(
        top["full_x"],
        top["full_y"],
        s=8.0 + 35.0 * (np.log1p(top["_score"]) / max(float(np.log1p(top["_score"]).max()), 1.0)),
        c="#e31a1c",
        alpha=0.72,
        linewidths=0,
        zorder=3,
        rasterized=True,
    )

    # Month ticks: show first, last and every 6th/12th month.
    month_map = nodes[[PERIOD_COL, "full_y"]].copy()
    month_map["period_str"] = month_map[PERIOD_COL].astype(str)
    month_tick = month_map.groupby("period_str")["full_y"].median().reset_index().sort_values("period_str")
    if len(month_tick) > 0:
        step = 6 if len(month_tick) <= 60 else max(1, len(month_tick) // 10)
        tick_rows = month_tick.iloc[::step].copy()
        if month_tick.iloc[-1]["period_str"] not in set(tick_rows["period_str"]):
            tick_rows = pd.concat([tick_rows, month_tick.tail(1)], ignore_index=True)
        ax.set_yticks(tick_rows["full_y"])
        ax.set_yticklabels(tick_rows["period_str"], fontsize=8)
    ax.set_xlabel("normalized tract spatial x", fontsize=10)
    ax.set_ylabel("month layer", fontsize=10)
    ax.grid(axis="y", alpha=0.12, linewidth=0.5)
    ax.set_xlim(-0.03, 1.03)
    y_min, y_max = float(nodes["full_y"].min()), float(nodes["full_y"].max())
    ax.set_ylim(y_min - 0.8, y_max + 0.8)
    ax.set_title(f"{title} — {len(nodes):,} nodes, {len(edges):,} edges", fontsize=16)

    legend_handles = []
    for etype in sorted(set(draw_edges["edge_type"].astype(str))):
        if etype in EDGE_COLORS:
            legend_handles.append(plt.Line2D([0], [0], color=EDGE_COLORS[etype], lw=2, label=etype))
    legend_handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e31a1c", markersize=6, label="highest burden nodes"))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", fontsize=8, frameon=True)

    ax.text(
        0.995,
        0.012,
        "Layered view: x = tract spatial coordinate; y = month. Edges are highly transparent to reveal full graph density.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )
    savefig(output_path, dpi=320)



def add_full_graph_spatial_cloud_positions(
    nodes: pd.DataFrame,
    *,
    temporal_radius: float = 0.115,
    jitter_scale: float = 0.010,
    perspective_strength: float = 0.055,
) -> pd.DataFrame:
    """Add full-graph visual_x/visual_y coordinates for a no-axis spatial-cloud view.

    This projection is designed for presentation, not metric geometry. It keeps the
    tract geography-like layout visible while spreading repeated tract-month nodes
    into a dense 2.5D cloud. This avoids the y-axis month snowball view while still
    showing all tract-month nodes.

    Design:
    - Base position: normalized tract spatial coordinate.
    - Temporal separation: deterministic low-amplitude orbit by month.
    - Jitter: deterministic node-level noise to prevent overplotting.
    - Perspective: small diagonal shear to make the cloud look less flat.
    """
    out = add_normalized_spatial_positions(nodes)
    x0 = pd.to_numeric(out["spatial_x"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
    y0 = pd.to_numeric(out["spatial_y"], errors="coerce").fillna(0.5).to_numpy(dtype=float)

    if PERIOD_COL in out.columns:
        months = {m: i for i, m in enumerate(sorted(out[PERIOD_COL].astype(str).unique()))}
        month_idx = out[PERIOD_COL].astype(str).map(months).fillna(0).to_numpy(dtype=float)
        month_count = max(1, len(months))
    elif "period_index" in out.columns:
        month_idx = pd.to_numeric(out["period_index"], errors="coerce").fillna(0).to_numpy(dtype=float)
        month_count = int(np.nanmax(month_idx) + 1) if len(month_idx) else 1
    else:
        month_idx = np.zeros(len(out), dtype=float)
        month_count = 1

    # Golden-angle orbit avoids visible radial spokes when there are many months.
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = month_idx * golden_angle
    month_norm = month_idx / max(float(month_count - 1), 1.0)

    # Month-specific orbit is modulated by spatial position so repeated tract-month
    # copies do not form perfectly circular local halos.
    local_radius = temporal_radius * (0.45 + 0.55 * (0.35 * x0 + 0.65 * y0))
    x = x0 + local_radius * np.cos(theta + 1.3 * y0)
    y = y0 + local_radius * np.sin(theta + 1.1 * x0)

    # Stable deterministic jitter based on node_id, not global RNG state.
    if NODE_ID_COL in out.columns:
        node_id = pd.to_numeric(out[NODE_ID_COL], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    else:
        node_id = np.arange(len(out), dtype=np.int64)
    h1 = ((node_id * 1103515245 + 12345) & 0x7FFFFFFF) / float(0x7FFFFFFF)
    h2 = ((node_id * 1664525 + 1013904223) & 0x7FFFFFFF) / float(0x7FFFFFFF)
    x += jitter_scale * (h1 - 0.5)
    y += jitter_scale * (h2 - 0.5)

    # Small pseudo-perspective/shear. This gives the figure a 2.5D cloud feeling
    # without using a real 3D plot or axes.
    depth = 0.55 * month_norm + 0.45 * y0
    x = x + perspective_strength * (depth - 0.5)
    y = y - perspective_strength * (depth - 0.5)

    # Normalize to plotting square while preserving aspect.
    x_range = float(np.nanmax(x) - np.nanmin(x)) or 1.0
    y_range = float(np.nanmax(y) - np.nanmin(y)) or 1.0
    out["visual_x"] = (x - np.nanmin(x)) / x_range
    out["visual_y"] = (y - np.nanmin(y)) / y_range
    out["visual_depth"] = depth
    out["visual_month_norm"] = month_norm
    return out


def plot_full_tract_month_spatial_cloud(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    max_edges_to_draw: int = 600_000,
) -> None:
    """Plot the full graph as a no-axis 2.5D spatial cloud.

    This is the presentation-oriented full graph view: all tract-month nodes are
    drawn in a geography-like layout with temporal copies spread into a dense cloud.
    Temporal/spatial edges are retained but drawn very lightly.
    """
    if nodes.empty:
        raise VisualError("full graph nodes are empty.")
    nodes = add_full_graph_spatial_cloud_positions(nodes)
    draw_edges = edges.copy()
    if len(draw_edges) > max_edges_to_draw:
        parts = []
        per_type = max(1, int(max_edges_to_draw / max(1, draw_edges["edge_type"].nunique())))
        for _, sub in draw_edges.groupby("edge_type", sort=True):
            if len(sub) > per_type:
                parts.append(sub.sample(n=per_type, random_state=20240610))
            else:
                parts.append(sub)
        draw_edges = pd.concat(parts, ignore_index=True) if parts else draw_edges.head(max_edges_to_draw)

    joined = _edge_segments_from_positions(draw_edges, nodes, "visual_x", "visual_y") if not draw_edges.empty else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    # Edge underlay: deliberately subtle so the node cloud remains legible.
    if not joined.empty:
        for edge_type, sub in joined.groupby("edge_type", sort=True):
            segs = sub[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float).reshape(-1, 2, 2)
            et = str(edge_type)
            if et.startswith("temporal"):
                alpha = 0.018
                lw = 0.050
            elif "placebo" in et:
                alpha = 0.010
                lw = 0.040
            else:
                alpha = 0.018
                lw = 0.045
            lc = LineCollection(
                segs,
                colors=[edge_color_for_type(et, alpha)],
                linewidths=lw,
                zorder=1,
                rasterized=True,
            )
            ax.add_collection(lc)

    scores = pd.to_numeric(nodes.get(TARGET_COLUMN, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    log_scores = np.log1p(scores)
    denom = max(float(np.nanmax(log_scores)), 1.0)
    score_scale = log_scores / denom
    depth = pd.to_numeric(nodes.get("visual_depth", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    depth_scale = (depth - np.nanmin(depth)) / (float(np.nanmax(depth) - np.nanmin(depth)) or 1.0)

    # Sort by depth so near/high-depth nodes appear on top, creating a 2.5D feel.
    order = np.argsort(depth_scale)
    x = nodes["visual_x"].to_numpy(dtype=float)[order]
    y = nodes["visual_y"].to_numpy(dtype=float)[order]
    color_values = log_scores[order]
    sizes = (0.55 + 8.0 * score_scale)[order]
    alphas = 0.18 + 0.38 * depth_scale[order]

    # Matplotlib scatter cannot vary alpha per point through scalar alpha, so use
    # RGBA colors from a colormap.
    cmap = plt.get_cmap("turbo")
    color_norm = color_values / denom
    rgba = cmap(np.clip(color_norm, 0.0, 1.0))
    rgba[:, 3] = np.clip(alphas, 0.16, 0.62)
    ax.scatter(
        x,
        y,
        s=sizes,
        c=rgba,
        linewidths=0,
        zorder=2,
        rasterized=True,
    )

    # Red ring / red dot high-burden overlay: not too many labels, enough visual focus.
    top = nodes.assign(_score=scores).sort_values("_score", ascending=False).head(120)
    top_scale = np.log1p(top["_score"].to_numpy(dtype=float)) / max(float(np.log1p(top["_score"]).max()), 1.0)
    ax.scatter(
        top["visual_x"],
        top["visual_y"],
        s=10.0 + 65.0 * top_scale,
        facecolors="none",
        edgecolors="#e31a1c",
        linewidths=0.55,
        alpha=0.68,
        zorder=4,
        rasterized=True,
    )
    ax.scatter(
        top.head(35)["visual_x"],
        top.head(35)["visual_y"],
        s=5.0 + 35.0 * top_scale[:35],
        c="#e31a1c",
        linewidths=0,
        alpha=0.62,
        zorder=5,
        rasterized=True,
    )

    # Legend-style mini key, intentionally compact and not axis-like.
    handles = []
    for etype in sorted(set(draw_edges["edge_type"].astype(str))):
        if etype in EDGE_COLORS:
            handles.append(plt.Line2D([0], [0], color=EDGE_COLORS[etype], lw=2, alpha=0.75, label=etype))
    handles.append(plt.Line2D([0], [0], marker="o", color="w", markeredgecolor="#e31a1c", markerfacecolor="none", markersize=7, label="highest burden nodes"))
    if handles:
        ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True, framealpha=0.88)

    ax.set_title(f"{title} — {len(nodes):,} tract-month nodes, {len(edges):,} edges", fontsize=17, pad=10)
    ax.text(
        0.995,
        0.012,
        "No-axis spatial-cloud projection: repeated tract-month nodes are spread by month; color encodes log1p burden.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
    )
    savefig(output_path, dpi=340)

def export_named_cytoscape(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    output_dir: Path,
    *,
    prefix: str,
    x_col: str,
    y_col: str,
    write_cyjs: bool = True,
    cyjs_edge_limit: int = 600_000,
) -> dict[str, str]:
    """Export named nodes/edges CSV and optional Cytoscape.js JSON without requiring networkx."""
    ensure_dir(output_dir)
    nodes = nodes.copy()
    edges = edges.copy()

    # Defensive cleanup: downstream Cytoscape export selects x_col/y_col plus
    # generic position columns such as spatial_x/spatial_y or full_x/full_y.
    # When x_col == "spatial_x" (or y_col == "spatial_y"), a naive column list
    # can contain duplicate labels. In pandas, row["spatial_x"] then returns a
    # Series rather than a scalar, which breaks float(row[x_col]).
    nodes = nodes.loc[:, ~nodes.columns.duplicated()].copy()
    edges = edges.loc[:, ~edges.columns.duplicated()].copy()

    if x_col not in nodes.columns or y_col not in nodes.columns:
        raise VisualError(f"Cannot export {prefix}: nodes lack {x_col!r}/{y_col!r}.")

    nodes_path = output_dir / f"{prefix}_nodes.csv"
    edges_path = output_dir / f"{prefix}_edges.csv"
    cyjs_path = output_dir / f"{prefix}.cyjs"

    node_cols = list(dict.fromkeys(
        c for c in [
            NODE_ID_COL,
            "node_key",
            ZONE_COL,
            PERIOD_COL,
            "period_index",
            TARGET_COLUMN,
            "requests_total",
            "svi_percentile",
            "svi_class",
            x_col,
            y_col,
            "spatial_x",
            "spatial_y",
            "full_x",
            "full_y",
            "graph_x",
            "graph_y",
            "plot_score",
        ] if c in nodes.columns
    ))
    nodes[node_cols].to_csv(nodes_path, index=False)

    if "edge_id" not in edges.columns:
        edges.insert(0, "edge_id", [f"e{i}" for i in range(len(edges))])
    edge_cols = list(dict.fromkeys(
        c for c in [
            "edge_id",
            "source_node_id",
            "target_node_id",
            "edge_type",
            "edge_weight",
            "distance_m",
            "source_zone_id",
            "target_zone_id",
            "source_period_month",
            "target_period_month",
            "is_temporal",
            "is_spatial",
            "is_placebo",
        ] if c in edges.columns
    ))
    edges[edge_cols].to_csv(edges_path, index=False)

    outputs = {
        f"{prefix}_nodes": str(nodes_path),
        f"{prefix}_edges": str(edges_path),
    }

    if write_cyjs:
        if len(edges) > cyjs_edge_limit:
            # Avoid silently producing a Cytoscape.js file that most laptops cannot load.
            # The full CSV edge list is still exported and is the safer Cytoscape import path.
            outputs[f"{prefix}_cyjs_skipped_reason"] = f"edge_count {len(edges):,} exceeds cyjs_edge_limit {cyjs_edge_limit:,}; use CSV import or raise --full-cyjs-edge-limit"
        else:
            node_export = nodes[node_cols].copy()
            edge_export = edges[edge_cols].copy()
            with cyjs_path.open("w", encoding="utf-8") as f:
                f.write('{"format_version":"1.0","generated_by":"09_generate_benchmark_visuals.py","data":{"name":')
                f.write(json.dumps(prefix))
                f.write('},"elements":[')
                first = True
                for _, row in node_export.iterrows():
                    node_id = str(int(row[NODE_ID_COL]))
                    data = {k: _json_safe(v) for k, v in row.to_dict().items()}
                    data["id"] = node_id
                    data["label"] = str(row.get(ZONE_COL, node_id))
                    elem = {
                        "data": data,
                        "position": {
                            "x": float(row[x_col]) * 1000.0,
                            "y": (1.0 - float(row[y_col])) * 1000.0 if y_col in {"spatial_y", "visual_y"} else float(row[y_col]) * 18.0,
                        },
                    }
                    if not first:
                        f.write(",")
                    f.write(json.dumps(elem, default=str, separators=(",", ":")))
                    first = False
                for i, row in edge_export.iterrows():
                    data = {k: _json_safe(v) for k, v in row.to_dict().items()}
                    data["id"] = str(data.get("edge_id", f"e{i}"))
                    data["source"] = str(int(row["source_node_id"]))
                    data["target"] = str(int(row["target_node_id"]))
                    elem = {"data": data}
                    if not first:
                        f.write(",")
                    f.write(json.dumps(elem, default=str, separators=(",", ":")))
                    first = False
                f.write("]}")
            outputs[f"{prefix}_cyjs"] = str(cyjs_path)
    return outputs


# ---------------------------------------------------------------------------
# Manifest/report
# ---------------------------------------------------------------------------


def write_manifest(output_dir: Path, outputs: Mapping[str, str], paths: VisualPaths) -> None:
    rows = [{"artifact": k, "path": v} for k, v in outputs.items()]
    pd.DataFrame(rows).to_csv(output_dir / "visual_manifest.csv", index=False)

    lines = [
        "# Benchmark Visuals",
        "",
        f"Generated at: `{now_utc()}`",
        "",
        "## Purpose",
        "",
        "This folder contains post-hoc visuals for the index → tabular ML → graph/neural benchmark layer. The script reads existing benchmark outputs and graph artifacts; it does not retrain or reselect models.",
        "",
        "## Inputs",
        "",
        f"- Comparison directory: `{paths.comparison_dir}`",
        f"- G1.5 sweep directory: `{paths.g15_dir}`",
        f"- G1 pilot directory: `{paths.g1_dir}`",
        f"- Graph artifact directory: `{paths.graph_dir}`",
        "",
        "## Visual artifacts",
        "",
        "| Artifact | Path |",
        "|---|---|",
    ]
    for key, path in outputs.items():
        lines.append(f"| `{key}` | `{path}` |")
    lines.extend(
        [
            "",
            "## Interpretation note",
            "",
            "Use these visuals to show that learned models are far stronger than static composite indices. Keep the no-edge and random-placebo controls visible: they are central to the scientific interpretation and prevent overclaiming the current spatial topology.",
        ]
    )
    (output_dir / "visual_summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def generate_visuals(
    paths: VisualPaths,
    *,
    graph_month: str = "auto",
    graph_sample_anchors: int = 8,
    graph_sample_max_nodes: int = 80,
    include_temporal_edges: bool = True,
    include_placebo_edges: bool = False,
    dense_graph_month: str = "auto",
    full_graph_edge_mode: str = "both",
    max_full_edges_to_draw: int = 600_000,
    write_full_cyjs: bool = True,
    full_cyjs_edge_limit: int = 600_000,
) -> dict[str, Any]:
    ensure_dir(paths.output_dir)
    ensure_dir(paths.figures_dir)
    ensure_dir(paths.cytoscape_dir)

    compact = read_csv_if_exists(paths.comparison_dir / "benchmark_comparison_compact.csv")
    margins = read_csv_if_exists(paths.comparison_dir / "family_margin_table.csv")
    final_comparison = read_csv_if_exists(paths.g15_dir / "final_comparison.csv", required=False)
    sweep_audit = read_csv_if_exists(paths.g15_dir / "sweep_model_selection_audit.csv", required=False)
    if sweep_audit.empty:
        # Older wrapper name or fallback to combined model-selection audit.
        sweep_audit = read_csv_if_exists(paths.g15_dir / "model_selection_audit.csv", required=False)
    if sweep_audit.empty:
        sweep_audit = read_csv_if_exists(paths.g15_dir / "selection_by_family.csv", required=False)

    outputs: dict[str, str] = {}

    fig01 = paths.figures_dir / "01_benchmark_metric_panels.png"
    plot_benchmark_metric_panels(compact, fig01)
    outputs["fig_01_benchmark_metric_panels"] = str(fig01)

    fig02 = paths.figures_dir / "02_index_vs_learned_ranking_gap.png"
    plot_index_vs_learned_gap(compact, fig02)
    outputs["fig_02_index_vs_learned_ranking_gap"] = str(fig02)

    fig03 = paths.figures_dir / "03_family_margin_panels.png"
    plot_family_margin_panels(margins, fig03)
    outputs["fig_03_family_margin_panels"] = str(fig03)

    if not final_comparison.empty:
        fig04 = paths.figures_dir / "04_g1_family_comparison.png"
        plot_g1_family_comparison(final_comparison, fig04)
        outputs["fig_04_g1_family_comparison"] = str(fig04)

    if not sweep_audit.empty:
        fig05 = paths.figures_dir / "05_g1_validation_sweep_heatmap.png"
        plot_validation_sweep_heatmap(sweep_audit, fig05)
        outputs["fig_05_g1_validation_sweep_heatmap"] = str(fig05)

    fig06 = paths.figures_dir / "06_benchmark_pipeline_schema.png"
    plot_pipeline_schema(fig06)
    outputs["fig_06_benchmark_pipeline_schema"] = str(fig06)

    node_table = read_parquet_if_exists(paths.graph_dir / "node_table.parquet")
    edge_table = read_parquet_if_exists(paths.graph_dir / "edge_table.parquet")
    chosen_month = choose_graph_month(node_table, graph_month)
    sample_nodes, sample_edges = build_graph_sample(
        node_table,
        edge_table,
        month=chosen_month,
        anchors=graph_sample_anchors,
        max_nodes=graph_sample_max_nodes,
        include_temporal=include_temporal_edges,
        include_placebo=include_placebo_edges,
    )
    fig07 = paths.figures_dir / "07_tract_month_graph_sample.png"
    plot_graph_sample(sample_nodes, sample_edges, fig07)
    outputs["fig_07_tract_month_graph_sample"] = str(fig07)
    outputs.update(export_cytoscape(sample_nodes, sample_edges, paths.cytoscape_dir))

    # Dense one-month graph: all tracts for one month, all same-month spatial kNN edges.
    dense_month = choose_graph_month(node_table, dense_graph_month)
    dense_nodes, dense_edges = build_one_month_spatial_graph(
        node_table,
        edge_table,
        month=dense_month,
        include_placebo=include_placebo_edges,
    )
    fig08 = paths.figures_dir / "08_one_month_spatial_graph_dense.png"
    plot_one_month_spatial_graph_dense(dense_nodes, dense_edges, fig08)
    outputs["fig_08_one_month_spatial_graph_dense"] = str(fig08)
    outputs.update(
        export_named_cytoscape(
            dense_nodes,
            dense_edges,
            paths.cytoscape_dir,
            prefix="one_month_spatial_graph_dense",
            x_col="spatial_x",
            y_col="spatial_y",
            write_cyjs=True,
            cyjs_edge_limit=full_cyjs_edge_limit,
        )
    )

    # Full tract-month graph visualizations. The real graph excludes random-placebo edges;
    # the artifact graph includes every edge in edge_table, including placebo controls.
    # The new full-graph view uses a no-axis 2.5D spatial-cloud projection rather than
    # the earlier month-layer snowball plot.
    full_nodes = add_full_graph_spatial_cloud_positions(node_table)
    full_modes: list[str]
    if full_graph_edge_mode == "both":
        full_modes = ["real", "artifact"]
    else:
        full_modes = [full_graph_edge_mode]

    full_graph_stats: dict[str, Any] = {}
    for mode in full_modes:
        full_edges = build_full_edge_table(edge_table, mode=mode)
        if mode == "real":
            fig_name = "09_full_tract_month_graph_spatial_cloud.png"
            title = "Full real tract-month graph"
            export_prefix = "full_tract_month_graph_spatial_cloud"
        else:
            fig_name = "10_full_artifact_graph_spatial_cloud_with_placebo.png"
            title = "Full artifact graph including placebo controls"
            export_prefix = "full_artifact_graph_spatial_cloud_with_placebo"
        fig_path = paths.figures_dir / fig_name
        plot_full_tract_month_spatial_cloud(
            full_nodes,
            full_edges,
            fig_path,
            title=title,
            max_edges_to_draw=max_full_edges_to_draw,
        )
        outputs[f"fig_{fig_name.removesuffix('.png')}"] = str(fig_path)
        outputs.update(
            export_named_cytoscape(
                full_nodes,
                full_edges,
                paths.cytoscape_dir,
                prefix=export_prefix,
                x_col="visual_x",
                y_col="visual_y",
                write_cyjs=write_full_cyjs,
                cyjs_edge_limit=full_cyjs_edge_limit,
            )
        )
        full_graph_stats[mode] = {"nodes": int(len(full_nodes)), "edges": int(len(full_edges))}

    metadata = {
        "stage_slug": STAGE_SLUG,
        "generated_at": now_utc(),
        "comparison_dir": str(paths.comparison_dir),
        "g15_dir": str(paths.g15_dir),
        "g1_dir": str(paths.g1_dir),
        "graph_dir": str(paths.graph_dir),
        "output_dir": str(paths.output_dir),
        "graph_sample_month": chosen_month,
        "graph_sample_nodes": int(len(sample_nodes)),
        "graph_sample_edges": int(len(sample_edges)),
        "dense_graph_month": dense_month,
        "dense_graph_nodes": int(len(dense_nodes)),
        "dense_graph_edges": int(len(dense_edges)),
        "full_graph_edge_mode": full_graph_edge_mode,
        "full_graph_stats": full_graph_stats,
        "max_full_edges_to_draw": int(max_full_edges_to_draw),
        "write_full_cyjs": bool(write_full_cyjs),
        "full_cyjs_edge_limit": int(full_cyjs_edge_limit),
        "networkx_available": bool(nx is not None),
        "networkx_required": False,
        "outputs": outputs,
    }
    write_json(paths.output_dir / "visual_metadata.json", metadata)
    outputs["metadata"] = str(paths.output_dir / "visual_metadata.json")
    write_manifest(paths.output_dir, outputs, paths)

    return {
        "status": "completed",
        "output_dir": str(paths.output_dir),
        "graph_sample_month": chosen_month,
        "graph_sample_nodes": int(len(sample_nodes)),
        "graph_sample_edges": int(len(sample_edges)),
        "dense_graph_month": dense_month,
        "dense_graph_nodes": int(len(dense_nodes)),
        "dense_graph_edges": int(len(dense_edges)),
        "full_graph_stats": full_graph_stats,
        "outputs": outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visuals for the index-ML-graph benchmark layer.")
    parser.add_argument("--comparison-dir", default=str(DEFAULT_COMPARISON_DIR))
    parser.add_argument("--g15-dir", default=str(DEFAULT_G15_DIR))
    parser.add_argument("--g1-dir", default=str(DEFAULT_G1_DIR))
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--graph-month", default="auto", help="Month for graph sample, or 'auto'.")
    parser.add_argument("--graph-sample-anchors", type=int, default=8)
    parser.add_argument("--graph-sample-max-nodes", type=int, default=80)
    parser.add_argument("--dense-graph-month", default="auto", help="Month for dense one-month spatial graph, or 'auto'.")
    parser.add_argument("--full-graph-edge-mode", default="both", choices=["real", "artifact", "both"], help="Which full tract-month graph exports to create. 'real' excludes placebo; 'artifact' includes all edges.")
    parser.add_argument("--max-full-edges-to-draw", type=int, default=600000, help="Maximum number of edges drawn in full graph PNGs before deterministic thinning.")
    parser.add_argument("--no-full-cyjs", action="store_true", help="Skip large full-graph Cytoscape.js JSON exports; CSV exports are still written.")
    parser.add_argument("--full-cyjs-edge-limit", type=int, default=600000, help="Maximum full-graph edge count allowed for Cytoscape.js export.")
    parser.add_argument("--no-temporal-edges", action="store_true", help="Exclude temporal edges from graph sample figure/export.")
    parser.add_argument("--include-placebo-edges", action="store_true", help="Include random-placebo spatial edges in graph sample and dense one-month figure/export.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = VisualPaths(
        comparison_dir=Path(args.comparison_dir),
        g15_dir=Path(args.g15_dir),
        g1_dir=Path(args.g1_dir),
        graph_dir=Path(args.graph_dir),
        output_dir=Path(args.output_dir),
    )
    if paths.output_dir.exists() and args.overwrite:
        import shutil
        shutil.rmtree(paths.output_dir)

    result = generate_visuals(
        paths,
        graph_month=args.graph_month,
        graph_sample_anchors=args.graph_sample_anchors,
        graph_sample_max_nodes=args.graph_sample_max_nodes,
        include_temporal_edges=not args.no_temporal_edges,
        include_placebo_edges=args.include_placebo_edges,
        dense_graph_month=args.dense_graph_month,
        full_graph_edge_mode=args.full_graph_edge_mode,
        max_full_edges_to_draw=args.max_full_edges_to_draw,
        write_full_cyjs=not args.no_full_cyjs,
        full_cyjs_edge_limit=args.full_cyjs_edge_limit,
    )
    print("Benchmark visuals completed.")
    print(f"Status: {result['status']}")
    print(f"Output dir: {result['output_dir']}")
    print(f"Graph sample month: {result['graph_sample_month']}")
    print(f"Graph sample: {result['graph_sample_nodes']} nodes, {result['graph_sample_edges']} edges")
    print(f"Dense one-month graph: {result['dense_graph_month']} with {result['dense_graph_nodes']} nodes, {result['dense_graph_edges']} edges")
    print(f"Full graph stats: {result['full_graph_stats']}")
    print("Key outputs:")
    for key, path in result["outputs"].items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
