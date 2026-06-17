#!/usr/bin/env python3
"""
Thin runner for B0 history-only baselines in the Québec CD civil-security / SoVI benchmark.

This script intentionally contains no benchmark logic. The reusable implementation lives in:

    ville_hgnn.baselines.b0_cd_history_baseline

Purpose:
    Load cd_month_panel.parquet, run B0 history-only predictors, and write
    predictions/metrics under:

    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B0_history_only/

Run from repository root:

    PYTHONPATH=urban_graph_benchmark/src python \
      urban_graph_benchmark/scripts/12_run_qc_cd_b0_history_baselines.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ville_hgnn.baselines.b0_cd_history_baseline import (
    Config,
    DEFAULT_MODELS,
    DEFAULT_OUTPUT_DIR,
    run_b0_cd_history_baseline,
)
from ville_hgnn.baselines.qc_cd_sovi_common import (
    CD_ID_COL,
    CD_NAME_COL,
    DEFAULT_PANEL_PATH,
    SPLIT_COL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run B0 history-only baselines for the Québec CD civil-security / "
            "SoVI predictive panel. This is a thin runner; all benchmark logic "
            "is implemented in ville_hgnn.baselines.b0_cd_history_baseline."
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
            "B0 output directory. Default: "
            "urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/"
            "baselines/B0_history_only/"
        ),
    )
    parser.add_argument(
        "--target-col",
        default="target_next_3_months",
        help="Target column to forecast. Default: target_next_3_months.",
    )
    parser.add_argument(
        "--current-count-col",
        default="event_count_current_month_all",
        help=(
            "Current-month count column used by history-only baselines. "
            "Default: event_count_current_month_all."
        ),
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
            "History-only models to run. Default: previous_month rolling_3_months "
            "rolling_6_months rolling_12_months seasonal_historical_mean."
        ),
    )
    parser.add_argument(
        "--keep-missing-targets",
        action="store_true",
        help=(
            "Keep rows with missing targets in prediction outputs. "
            "Metrics still ignore missing targets."
        ),
    )
    parser.add_argument(
        "--no-clip-predictions",
        action="store_true",
        help="Do not clip count predictions at zero.",
    )
    parser.add_argument(
        "--no-global-fallback",
        action="store_true",
        help=(
            "Do not fill cold-start seasonal-history predictions with the "
            "global train mean."
        ),
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        panel_path=args.panel_path,
        output_dir=args.output_dir,
        target_col=args.target_col,
        current_count_col=args.current_count_col,
        cd_id_col=args.cd_id_col,
        cd_name_col=args.cd_name_col,
        period_month_col=args.period_month_col,
        split_col=args.split_col,
        models=list(args.models),
        drop_missing_target=not args.keep_missing_targets,
        clip_predictions_at_zero=not args.no_clip_predictions,
        fallback_to_train_global_mean=not args.no_global_fallback,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    result = run_b0_cd_history_baseline(config)

    print("B0 history-only baseline runner completed.")
    print(f"Panel path: {config.panel_path}")
    print(f"Output directory: {config.output_dir}")
    print(f"Target column: {config.target_col}")
    print("Models:")
    for model_name in config.models:
        print(f"  - {model_name}")

    selected = result.get("selected_model") if isinstance(result, dict) else None
    if selected:
        print("Selected model by validation MAE:")
        print(f"  model_name: {selected.get('model_name')}")
        print(f"  split: {selected.get('selection_split')}")
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
