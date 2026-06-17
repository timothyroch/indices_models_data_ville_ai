#!/usr/bin/env python3
"""
Thin runner for B2 calibrated SoVI predictors in the Québec CD civil-security / SoVI benchmark.

This script intentionally contains no benchmark logic. The reusable implementation lives in:

    ville_hgnn.baselines.b2_calibrated_sovi

Purpose:
    Load cd_month_panel.parquet, run B2 calibrated SoVI predictors, and write
    predictions/metrics under:

    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B2_calibrated_sovi/

Run from repository root:

    PYTHONPATH=urban_graph_benchmark/src python \
      urban_graph_benchmark/scripts/13_run_qc_cd_b2_calibrated_sovi.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ville_hgnn.baselines.b2_calibrated_sovi import (
    Config,
    DEFAULT_MODELS,
    DEFAULT_OUTPUT_DIR,
    POISSON_ALPHAS,
    RIDGE_ALPHAS,
    run_b2_calibrated_sovi,
)
from ville_hgnn.baselines.qc_cd_sovi_common import (
    CD_ID_COL,
    CD_NAME_COL,
    DEFAULT_PANEL_PATH,
    SOVI_SCORE_COL,
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
            "Run B2 calibrated SoVI predictors for the Québec CD civil-security / "
            "SoVI predictive panel. This is a thin runner; all benchmark logic "
            "is implemented in ville_hgnn.baselines.b2_calibrated_sovi."
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
            "B2 output directory. Default: "
            "urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/"
            "baselines/B2_calibrated_sovi/"
        ),
    )
    parser.add_argument(
        "--target-col",
        default="target_next_3_months",
        help="Target column to forecast. Default: target_next_3_months.",
    )
    parser.add_argument(
        "--sovi-score-col",
        default=SOVI_SCORE_COL,
        help=f"SoVI score column used as predictor. Default: {SOVI_SCORE_COL}.",
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
        "--month-col",
        default="month",
        help="Calendar month column used for seasonal features. Default: month.",
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
            "B2 calibrated SoVI models to run. Default: linear_sovi ridge_sovi "
            "poisson_sovi linear_sovi_seasonal ridge_sovi_seasonal "
            "poisson_sovi_seasonal."
        ),
    )
    parser.add_argument(
        "--ridge-alphas",
        nargs="+",
        default=None,
        help="Ridge alpha values. Default: 0.1 1.0 10.0.",
    )
    parser.add_argument(
        "--poisson-alphas",
        nargs="+",
        default=None,
        help="PoissonRegressor alpha values. Default: 0.0 0.1 1.0.",
    )
    parser.add_argument(
        "--selection-metric",
        default="mae",
        choices=["mae", "rmse", "mean_poisson_deviance", "spearman", "ndcg_at_25"],
        help="Validation metric used to select one candidate per model. Default: mae.",
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
        help="Do not clip linear/ridge predictions at zero.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    parser.add_argument(
        "--poisson-max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for PoissonRegressor. Default: 1000.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        panel_path=args.panel_path,
        output_dir=args.output_dir,
        target_col=args.target_col,
        sovi_score_col=args.sovi_score_col,
        cd_id_col=args.cd_id_col,
        cd_name_col=args.cd_name_col,
        period_month_col=args.period_month_col,
        month_col=args.month_col,
        split_col=args.split_col,
        models=list(args.models),
        ridge_alphas=parse_float_list(args.ridge_alphas, RIDGE_ALPHAS),
        poisson_alphas=parse_float_list(args.poisson_alphas, POISSON_ALPHAS),
        selection_metric=args.selection_metric,
        drop_missing_target=not args.keep_missing_targets,
        clip_predictions_at_zero=not args.no_clip_predictions,
        random_seed=args.random_seed,
        poisson_max_iter=args.poisson_max_iter,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    result = run_b2_calibrated_sovi(config)

    print("B2 calibrated SoVI runner completed.")
    print(f"Panel path: {config.panel_path}")
    print(f"Output directory: {config.output_dir}")
    print(f"Target column: {config.target_col}")
    print(f"SoVI score column: {config.sovi_score_col}")
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
