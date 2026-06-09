#!/usr/bin/env python3
"""
Compare A0 naive baselines and A1 SVI direct-ranking results.

This script is a lightweight reporting layer. It does not train a model, rebuild
the dataset, rebuild splits, or introduce new baseline methodology. It only
reads A0/A1 metrics and writes comparison tables plus a Markdown report.

Default inputs:
  urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_naive_temporal/metrics.csv
  urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_direct_ranking/metrics.csv

Default outputs:
  urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_A1_comparison/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG_PATH = "urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml"
DEFAULT_BASELINE_ROOT = "urban_graph_benchmark/outputs/mtl_311_water_v0/baselines"
DEFAULT_A0_METRICS = f"{DEFAULT_BASELINE_ROOT}/A0_naive_temporal/metrics.csv"
DEFAULT_A1_METRICS = f"{DEFAULT_BASELINE_ROOT}/A1_svi_direct_ranking/metrics.csv"
DEFAULT_OUTPUT_DIR = f"{DEFAULT_BASELINE_ROOT}/A0_A1_comparison"

TEST_SPLIT_NAME = "temporal_test"

KEY_RANKING_METRICS = [
    "spearman_corr",
    "ndcg_at_100",
    "top_10pct_overlap_rate",
]
KEY_COUNT_METRICS = [
    "mae",
    "rmse",
    "mean_poisson_deviance",
]

HEADLINE_MODELS = [
    "A0_3_tract_train_mean",
    "A0_4_tract_month_of_year_train_mean",
    "A0_5_previous_month_persistence",
    "A0_8_non_water_311_reporting_exposure_retrospective",
    "A1_svi_direct_ranking__svi_percentile",
    "A1_svi_direct_ranking__svi_score_raw",
    "A1_svi_direct_ranking__svi_class__ordinal_class_diagnostic",
]


class ComparisonError(RuntimeError):
    """Raised when A0/A1 comparison cannot be produced."""


def _bootstrap_package_path() -> None:
    """Add urban_graph_benchmark/src to sys.path when running from source."""

    script_path = Path(__file__).resolve()
    for parent in [script_path.parent, *script_path.parents]:
        for candidate in [
            parent / "urban_graph_benchmark" / "src",
            parent / "src",
        ]:
            if (candidate / "ville_hgnn").exists():
                candidate_str = str(candidate)
                if candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare A0 naive baselines and A1 SVI direct-ranking results."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Config path used for provenance only. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root. Defaults to automatic detection.",
    )
    parser.add_argument(
        "--a0-metrics",
        default=DEFAULT_A0_METRICS,
        help=f"Path to A0 metrics.csv. Default: {DEFAULT_A0_METRICS}",
    )
    parser.add_argument(
        "--a1-metrics",
        default=DEFAULT_A1_METRICS,
        help=f"Path to A1 metrics.csv. Default: {DEFAULT_A1_METRICS}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ComparisonError("pandas is required for A0/A1 comparison.") from exc
    return pd


def find_repo_root() -> Path:
    """Find repository root by walking up from cwd and script path."""

    starts = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    for start in starts:
        for parent in [start, *start.parents]:
            if (parent / "urban_graph_benchmark").exists():
                return parent
    return Path.cwd().resolve()


def resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_metrics_files(a0_path: Path, a1_path: Path) -> None:
    missing = [str(path) for path in [a0_path, a1_path] if not path.exists()]
    if missing:
        raise ComparisonError(
            "Missing required metrics file(s). Run scripts/04_run_a0_a1_baselines.py first. "
            f"Missing: {missing}"
        )


def split_name_to_partition(split_name: Any) -> str:
    text = str(split_name)
    for suffix in ["_validation", "_test", "_train"]:
        if text.endswith(suffix):
            return suffix[1:]
    return "unknown"


def split_name_to_scheme(split_name: Any) -> str:
    text = str(split_name)
    for suffix in ["_validation", "_test", "_train"]:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def classify_model_stage(model_name: str) -> str:
    name = str(model_name)
    if name.startswith("A0_"):
        return "A0_naive_temporal"
    if name.startswith("A1_"):
        return "A1_svi_direct_ranking"
    return "unknown"


def classify_model_role(model_name: str) -> str:
    name = str(model_name)

    if name.startswith("A0_"):
        if name == "A0_3_tract_train_mean":
            return "strong_naive_history"
        if "tract_month" in name:
            return "seasonal_tract_history"
        if "previous_month" in name or "previous_year" in name:
            return "persistence_history"
        if "non_water_311" in name:
            return "retrospective_reporting_exposure"
        if "population" in name:
            return "population_exposure"
        return "naive_baseline"

    if name.startswith("A1_"):
        if "svi_percentile" in name or "svi_score_raw" in name:
            return "primary_continuous_svi"
        if "svi_class" in name:
            return "diagnostic_svi_class"
        if "svi_rank" in name:
            return "diagnostic_svi_rank"
        return "svi_diagnostic"

    return "unknown"


def normalize_metric_name(metric_name: str) -> tuple[str, str]:
    """
    Map source metric names into comparable scopes.

    A0 count__mae                                      -> count_prediction / mae
    A0 ranking__spearman_corr                         -> tract_month_ranking / spearman_corr
    A1 tract_month_ranking__spearman_corr             -> tract_month_ranking / spearman_corr
    A1 tract_level_ranking__target_label__spearman    -> tract_level_ranking / spearman_corr
    """

    name = str(metric_name)

    if name.startswith("count__"):
        return "count_prediction", name.removeprefix("count__")

    if name.startswith("ranking__"):
        return "tract_month_ranking", name.removeprefix("ranking__")

    if name.startswith("tract_month_ranking__"):
        return "tract_month_ranking", name.removeprefix("tract_month_ranking__")

    if name.startswith("tract_level_ranking__"):
        parts = name.split("__")
        if len(parts) >= 3:
            return "tract_level_ranking", "__".join(parts[2:])
        return "tract_level_ranking", name.removeprefix("tract_level_ranking__")

    if name.startswith("binary__"):
        return "binary_diagnostic", name.removeprefix("binary__")

    return "other", name


def load_and_normalize_metrics(a0_path: Path, a1_path: Path):
    pd = import_pandas()

    a0 = pd.read_csv(a0_path)
    a1 = pd.read_csv(a1_path)
    a0["source_stage"] = "A0"
    a1["source_stage"] = "A1"

    metrics = pd.concat([a0, a1], ignore_index=True)

    required = [
        "split_name",
        "model_name",
        "metric_name",
        "metric_value",
        "higher_is_better",
        "n_rows",
    ]
    missing = [col for col in required if col not in metrics.columns]
    if missing:
        raise ComparisonError(f"Metrics table missing required columns: {missing}")

    normalized = metrics["metric_name"].apply(normalize_metric_name)
    metrics["comparison_scope"] = [x[0] for x in normalized]
    metrics["comparison_metric"] = [x[1] for x in normalized]
    metrics["split_scheme"] = metrics["split_name"].apply(split_name_to_scheme)
    metrics["split_partition"] = metrics["split_name"].apply(split_name_to_partition)
    metrics["model_stage_detected"] = metrics["model_name"].apply(classify_model_stage)
    metrics["model_role"] = metrics["model_name"].apply(classify_model_role)
    metrics["metric_value"] = pd.to_numeric(metrics["metric_value"], errors="coerce")

    return metrics


def build_tract_month_ranking_wide(metrics):
    wanted = metrics[
        (metrics["comparison_scope"] == "tract_month_ranking")
        & (metrics["comparison_metric"].isin(KEY_RANKING_METRICS))
        & (metrics["split_partition"].isin(["validation", "test"]))
    ].copy()

    if wanted.empty:
        return wanted

    wide = wanted.pivot_table(
        index=["split_name", "comparison_metric"],
        columns="model_name",
        values="metric_value",
        aggfunc="first",
    ).reset_index()

    fixed = ["split_name", "comparison_metric"]
    headline_present = [col for col in HEADLINE_MODELS if col in wide.columns]
    extras = [col for col in wide.columns if col not in fixed and col not in headline_present]
    return wide[fixed + headline_present + extras]


def build_test_headline_table(metrics):
    wanted = metrics[
        (metrics["split_name"] == TEST_SPLIT_NAME)
        & (
            (
                (metrics["comparison_scope"] == "tract_month_ranking")
                & (metrics["comparison_metric"].isin(KEY_RANKING_METRICS))
            )
            | (
                (metrics["comparison_scope"] == "count_prediction")
                & (metrics["comparison_metric"].isin(KEY_COUNT_METRICS))
            )
        )
        & (metrics["model_name"].isin(HEADLINE_MODELS))
    ].copy()

    if wanted.empty:
        return wanted

    wanted["display_metric"] = wanted["comparison_scope"] + "__" + wanted["comparison_metric"]

    keep = [
        "source_stage",
        "model_name",
        "model_role",
        "target_name",
        "target_type",
        "display_metric",
        "metric_value",
        "higher_is_better",
        "n_rows",
    ]
    keep = [col for col in keep if col in wanted.columns]
    return wanted[keep].sort_values(["display_metric", "source_stage", "model_name"]).reset_index(drop=True)


def metric_value_for(metrics, model_name: str, split_name: str, scope: str, metric: str) -> float | None:
    subset = metrics[
        (metrics["model_name"] == model_name)
        & (metrics["split_name"] == split_name)
        & (metrics["comparison_scope"] == scope)
        & (metrics["comparison_metric"] == metric)
    ]

    if subset.empty:
        return None

    value = subset["metric_value"].iloc[0]
    if value is None:
        return None

    try:
        value = float(value)
    except Exception:
        return None

    if math.isnan(value) or math.isinf(value):
        return None

    return value


def best_by_metric(metrics, *, split_name: str, scope: str, metric: str):
    subset = metrics[
        (metrics["split_name"] == split_name)
        & (metrics["comparison_scope"] == scope)
        & (metrics["comparison_metric"] == metric)
    ].copy()

    if subset.empty:
        return None

    subset = subset.dropna(subset=["metric_value"])
    if subset.empty:
        return None

    higher = bool(subset["higher_is_better"].dropna().iloc[0])
    subset = subset.sort_values("metric_value", ascending=not higher)
    return subset.iloc[0].to_dict()


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        x = float(value)
    except Exception:
        return str(value)
    if math.isnan(x) or math.isinf(x):
        return "NA"
    return f"{x:.{digits}f}"


def dataframe_to_markdown(df, max_rows: int = 80) -> str:
    if df is None or df.empty:
        return "_No rows._"
    display = df.head(max_rows).copy()
    try:
        return display.to_markdown(index=False)
    except Exception:
        return "```text\n" + display.to_string(index=False) + "\n```"


def make_interpretation_table(metrics):
    pd = import_pandas()
    rows = []

    pairs = [
        ("A0 strongest history baseline", "A0_3_tract_train_mean"),
        ("A1 primary SVI percentile", "A1_svi_direct_ranking__svi_percentile"),
        ("A1 primary SVI raw score", "A1_svi_direct_ranking__svi_score_raw"),
        ("A1 diagnostic SVI class", "A1_svi_direct_ranking__svi_class__ordinal_class_diagnostic"),
    ]

    for label, model in pairs:
        rows.append(
            {
                "label": label,
                "model_name": model,
                "spearman_corr": metric_value_for(
                    metrics, model, TEST_SPLIT_NAME, "tract_month_ranking", "spearman_corr"
                ),
                "ndcg_at_100": metric_value_for(
                    metrics, model, TEST_SPLIT_NAME, "tract_month_ranking", "ndcg_at_100"
                ),
                "top_10pct_overlap_rate": metric_value_for(
                    metrics, model, TEST_SPLIT_NAME, "tract_month_ranking", "top_10pct_overlap_rate"
                ),
            }
        )

    return pd.DataFrame(rows)


def render_report(*, metrics, headline, wide, metadata: Mapping[str, Any]) -> str:
    interpretation = make_interpretation_table(metrics)

    best_spearman = best_by_metric(
        metrics,
        split_name=TEST_SPLIT_NAME,
        scope="tract_month_ranking",
        metric="spearman_corr",
    )
    best_ndcg = best_by_metric(
        metrics,
        split_name=TEST_SPLIT_NAME,
        scope="tract_month_ranking",
        metric="ndcg_at_100",
    )
    best_top10 = best_by_metric(
        metrics,
        split_name=TEST_SPLIT_NAME,
        scope="tract_month_ranking",
        metric="top_10pct_overlap_rate",
    )
    best_mae = best_by_metric(
        metrics,
        split_name=TEST_SPLIT_NAME,
        scope="count_prediction",
        metric="mae",
    )

    a0_s = metric_value_for(metrics, "A0_3_tract_train_mean", TEST_SPLIT_NAME, "tract_month_ranking", "spearman_corr")
    a1_s = metric_value_for(metrics, "A1_svi_direct_ranking__svi_percentile", TEST_SPLIT_NAME, "tract_month_ranking", "spearman_corr")
    a0_top = metric_value_for(metrics, "A0_3_tract_train_mean", TEST_SPLIT_NAME, "tract_month_ranking", "top_10pct_overlap_rate")
    a1_top = metric_value_for(metrics, "A1_svi_direct_ranking__svi_percentile", TEST_SPLIT_NAME, "tract_month_ranking", "top_10pct_overlap_rate")

    lines: list[str] = []
    lines.append("# A0/A1 Baseline Comparison — Montréal 311 Water/Drainage v0\n")
    lines.append(f"Generated at: `{metadata.get('generated_at')}`\n")
    lines.append(f"Config path: `{metadata.get('config_path')}`\n")

    lines.append("## Purpose\n")
    lines.append(
        "This report compares the first two baseline layers: A0 naive temporal/exposure "
        "baselines and A1 static SVI direct-ranking baselines. It introduces no new "
        "modeling methodology; it only summarizes metrics already produced by A0 and A1.\n"
    )

    lines.append("## Headline conclusion\n")
    lines.append(
        "The strongest A0 history baseline is much stronger than raw SVI for tract-month "
        "ranking of future water/drainage 311 burden. SVI has a positive but weak "
        "standalone ranking signal. This supports using A1 as a vulnerability-prior "
        "diagnostic, not as the main predictive benchmark.\n"
    )

    lines.append("## Test split: core ranking comparison\n")
    lines.append(dataframe_to_markdown(interpretation))
    lines.append("")

    if a0_s is not None and a1_s is not None:
        ratio = a0_s / a1_s if a1_s != 0 else None
        lines.append(
            f"On temporal test, `A0_3_tract_train_mean` has Spearman `{fmt(a0_s)}`; "
            f"`A1_svi_percentile` has Spearman `{fmt(a1_s)}`. Ratio: `{fmt(ratio)}`.\n"
        )

    if a0_top is not None and a1_top is not None:
        ratio = a0_top / a1_top if a1_top != 0 else None
        lines.append(
            f"For top-10% overlap, `A0_3_tract_train_mean` has `{fmt(a0_top)}`; "
            f"`A1_svi_percentile` has `{fmt(a1_top)}`. Ratio: `{fmt(ratio)}`.\n"
        )

    lines.append("## Best rows by temporal-test metric\n")
    lines.append("| Metric | Best model | Value | Scope | Higher is better |")
    lines.append("|---|---|---:|---|:---:|")
    for label, row in [
        ("Tract-month Spearman", best_spearman),
        ("Tract-month NDCG@100", best_ndcg),
        ("Tract-month top-10% overlap", best_top10),
        ("Count MAE", best_mae),
    ]:
        if row is None:
            continue
        lines.append(
            f"| {label} | `{row.get('model_name')}` | {fmt(row.get('metric_value'))} | "
            f"`{row.get('comparison_scope')}` | `{row.get('higher_is_better')}` |"
        )
    lines.append("")

    lines.append("## Headline metric rows\n")
    lines.append(dataframe_to_markdown(headline, max_rows=120))
    lines.append("")

    lines.append("## Wide tract-month ranking table\n")
    lines.append(dataframe_to_markdown(wide, max_rows=20))
    lines.append("")

    lines.append("## Interpretation guardrails\n")
    lines.append(
        "- A0 and A1 are not the same type of baseline: A0 includes historical target information, while A1 is a static vulnerability score.\n"
        "- A1 should not be expected to beat historical tract burden on direct monthly 311 prediction. Its fair role is vulnerability-prior ranking.\n"
        "- `svi_class` is diagnostic because it is an ordinal class label; primary A1 interpretation should use `svi_percentile` or `svi_score_raw`.\n"
        "- This comparison supports A2: calibrated SVI/regression-style baselines are the fairer analogue to literature that validates SVI with controls.\n"
    )

    lines.append("## Output artifacts\n")
    lines.append("| Artifact | Path |")
    lines.append("|---|---|")
    for key, value in metadata.get("outputs", {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")

    return "\n".join(lines)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_compare(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    repo_root: str | Path | None = None,
    a0_metrics: str | Path = DEFAULT_A0_METRICS,
    a1_metrics: str | Path = DEFAULT_A1_METRICS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()

    config = resolve_path(config_path, root)
    a0_path = resolve_path(a0_metrics, root)
    a1_path = resolve_path(a1_metrics, root)
    out_dir = resolve_path(output_dir, root)

    require_metrics_files(a0_path, a1_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_and_normalize_metrics(a0_path, a1_path)
    wide = build_tract_month_ranking_wide(metrics)
    headline = build_test_headline_table(metrics)

    metrics_long_path = out_dir / "a0_a1_metrics_long.csv"
    wide_path = out_dir / "a0_a1_tract_month_ranking_wide.csv"
    headline_path = out_dir / "a0_a1_test_headline_table.csv"
    report_path = out_dir / "a0_a1_comparison_report.md"
    metadata_path = out_dir / "comparison_metadata.json"

    outputs = {
        "metrics_long": str(metrics_long_path),
        "tract_month_ranking_wide": str(wide_path),
        "test_headline_table": str(headline_path),
        "comparison_report": str(report_path),
        "comparison_metadata": str(metadata_path),
    }

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config),
        "config_sha256": file_sha256(config),
        "repo_root": str(root),
        "a0_metrics_path": str(a0_path),
        "a0_metrics_sha256": file_sha256(a0_path),
        "a1_metrics_path": str(a1_path),
        "a1_metrics_sha256": file_sha256(a1_path),
        "n_metric_rows_total": int(len(metrics)),
        "n_a0_metric_rows": int((metrics["source_stage"] == "A0").sum()),
        "n_a1_metric_rows": int((metrics["source_stage"] == "A1").sum()),
        "outputs": outputs,
    }

    report = render_report(metrics=metrics, headline=headline, wide=wide, metadata=metadata)

    metrics.to_csv(metrics_long_path, index=False)
    wide.to_csv(wide_path, index=False)
    headline.to_csv(headline_path, index=False)
    write_text(report_path, report)
    write_json(metadata_path, metadata)

    return {
        "status": "completed",
        "output_dir": str(out_dir),
        "n_metric_rows_total": int(len(metrics)),
        "n_a0_metric_rows": int((metrics["source_stage"] == "A0").sum()),
        "n_a1_metric_rows": int((metrics["source_stage"] == "A1").sum()),
        "outputs": outputs,
    }


def compare_brief(result: Mapping[str, Any]) -> str:
    outputs = result.get("outputs", {})
    return (
        "A0/A1 comparison completed.\n"
        f"Status: {result.get('status')}\n"
        f"Metric rows total: {result.get('n_metric_rows_total')}\n"
        f"A0 metric rows: {result.get('n_a0_metric_rows')}\n"
        f"A1 metric rows: {result.get('n_a1_metric_rows')}\n"
        f"Report: {outputs.get('comparison_report')}\n"
        f"Headline table: {outputs.get('test_headline_table')}\n"
    )


def main() -> None:
    _bootstrap_package_path()
    args = parse_args()

    result = run_compare(
        config_path=args.config,
        repo_root=args.repo_root,
        a0_metrics=args.a0_metrics,
        a1_metrics=args.a1_metrics,
        output_dir=args.output_dir,
    )

    print(compare_brief(result).rstrip())
    print("\nWritten outputs:")
    for label, path in result.get("outputs", {}).items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()