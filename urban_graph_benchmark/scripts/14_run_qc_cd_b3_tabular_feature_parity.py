#!/usr/bin/env python3
"""
Thin runner for B3 tabular feature-parity ML in the Québec CD civil-security / SoVI benchmark.

This script intentionally contains no benchmark logic. The reusable implementation lives in:

    ville_hgnn.baselines.b3_cd_tabular_feature_parity

Purpose:
    Load cd_month_panel.parquet, run B3 non-graph ML with feature parity, and write
    predictions/metrics under:

    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B3_tabular_feature_parity/

Run from repository root:

    PYTHONPATH=urban_graph_benchmark/src python \
      urban_graph_benchmark/scripts/14_run_qc_cd_b3_tabular_feature_parity.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ville_hgnn.baselines.b3_cd_tabular_feature_parity import (
    Config,
    DEFAULT_MODELS,
    DEFAULT_OUTPUT_DIR,
    RIDGE_ALPHAS,
    run_b3_cd_tabular_feature_parity,
)
from ville_hgnn.baselines.qc_cd_sovi_common import (
    CD_ID_COL,
    CD_NAME_COL,
    DEFAULT_PANEL_PATH,
    SPLIT_COL,
)


def parse_float_list(values: list[str] | None, default: list[float]) -> list[float]:
    """Parse CLI float lists while keeping clean defaults."""
    if not values:
        return [float(v) for v in default]
    return [float(v) for v in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run B3 tabular feature-parity ML for the Québec CD civil-security / "
            "SoVI predictive panel. This is a thin runner; all benchmark logic "
            "is implemented in ville_hgnn.baselines.b3_cd_tabular_feature_parity."
        )
    )

    parser.add_argument(
        "--panel-path",
        type=Path,
        default=DEFAULT_PANEL_PATH,
        help=(
            "Predictive CD × month panel. Default: "
            "urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/"
            "datasets/cd_month_panel.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "B3 output directory. Default: "
            "urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/"
            "baselines/B3_tabular_feature_parity/"
        ),
    )
    parser.add_argument(
        "--target-col",
        default="target_next_3_months",
        help="Target column to forecast. Default: target_next_3_months.",
    )
    parser.add_argument(
        "--cd-id-col",
        default=CD_ID_COL,
        help=f"CD ID column in the panel. Default: {CD_ID_COL}.",
    )
    parser.add_argument(
        "--cd-name-col",
        default=CD_NAME_COL,
        help=f"CD name column in the panel. Default: {CD_NAME_COL}.",
    )
    parser.add_argument(
        "--period-month-col",
        default="period_month",
        help="Panel period-month column. Default: period_month.",
    )
    parser.add_argument(
        "--split-col",
        default=SPLIT_COL,
        help=f"Train/validation/test split column. Default: {SPLIT_COL}.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=list(DEFAULT_MODELS),
        help=(
            "B3 model families to run. Default: ridge random_forest "
            "hist_gradient_boosting."
        ),
    )
    parser.add_argument(
        "--ridge-alphas",
        nargs="+",
        default=None,
        help="Ridge alpha values. Default: 0.1 1.0 10.0 100.0.",
    )
    parser.add_argument(
        "--selection-metric",
        default="mae",
        choices=["mae", "rmse", "mean_poisson_deviance", "spearman", "ndcg_at_25"],
        help="Validation metric used to select one candidate per model. Default: mae.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for random forest. Default: -1.",
    )
    parser.add_argument(
        "--keep-missing-targets",
        action="store_true",
        help=(
            "Keep rows with missing targets in candidate/prediction outputs. "
            "Metrics still ignore missing targets."
        ),
    )
    parser.add_argument(
        "--no-clip-predictions",
        action="store_true",
        help="Do not clip count predictions at zero.",
    )

    feature_group = parser.add_argument_group("feature ablations")
    feature_group.add_argument(
        "--no-sovi-features",
        action="store_true",
        help="Exclude SoVI score/static features.",
    )
    feature_group.add_argument(
        "--no-history-features",
        action="store_true",
        help="Exclude generic history features.",
    )
    feature_group.add_argument(
        "--no-hazard-history-features",
        action="store_true",
        help="Exclude hazard-specific history features.",
    )
    feature_group.add_argument(
        "--no-current-month-counts",
        action="store_true",
        help="Exclude current-month event count features.",
    )
    feature_group.add_argument(
        "--no-seasonality",
        action="store_true",
        help="Exclude month sin/cos seasonality features.",
    )
    feature_group.add_argument(
        "--no-year-trend",
        action="store_true",
        help="Exclude origin-year trend feature.",
    )
    feature_group.add_argument(
        "--no-other-numeric-features",
        action="store_true",
        help="Exclude otherwise eligible numeric features.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        panel_path=args.panel_path,
        output_dir=args.output_dir,
        target_col=args.target_col,
        cd_id_col=args.cd_id_col,
        cd_name_col=args.cd_name_col,
        period_month_col=args.period_month_col,
        split_col=args.split_col,
        models=list(args.models),
        ridge_alphas=parse_float_list(args.ridge_alphas, RIDGE_ALPHAS),
        selection_metric=args.selection_metric,
        random_seed=args.random_seed,
        n_jobs=args.n_jobs,
        drop_missing_target=not args.keep_missing_targets,
        clip_predictions_at_zero=not args.no_clip_predictions,
        include_sovi_features=not args.no_sovi_features,
        include_history_features=not args.no_history_features,
        include_hazard_history_features=not args.no_hazard_history_features,
        include_current_month_counts=not args.no_current_month_counts,
        include_seasonality=not args.no_seasonality,
        include_year_trend=not args.no_year_trend,
        include_all_other_numeric_features=not args.no_other_numeric_features,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    result = run_b3_cd_tabular_feature_parity(config)

    print("B3 tabular feature-parity runner completed.")
    print(f"Panel path: {config.panel_path}")
    print(f"Output directory: {config.output_dir}")
    print(f"Target column: {config.target_col}")
    print(f"Feature count: {result.get('feature_count') if isinstance(result, dict) else 'unknown'}")
    print("Models:")
    for model_name in config.models:
        print(f"  - {model_name}")

    selected = result.get("selected_model") if isinstance(result, dict) else None
    if selected:
        print("Selected model by validation metric:")
        print(f"  model_name: {selected.get('model_name')}")
        print(f"  candidate_name: {selected.get('candidate_name')}")
        print(f"  split: {selected.get('selection_split')}")
        print(f"  metric: {selected.get('selection_metric')}")
        print(f"  MAE: {selected.get('mae')}")
        print(f"  RMSE: {selected.get('rmse')}")
        print(f"  Spearman: {selected.get('spearman')}")

    outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
    if outputs:
        print("Outputs:")
        for key, value in outputs.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
