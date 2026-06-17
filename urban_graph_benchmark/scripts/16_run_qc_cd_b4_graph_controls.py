#!/usr/bin/env python3
"""
Thin runner for B4 graph-control baselines in the Québec CD civil-security / SoVI benchmark.

This script intentionally contains no benchmark logic. The reusable implementation lives in:

    ville_hgnn.baselines.b4_cd_graph_controls

Purpose:
    Load the CD-month panel, graph nodes, and graph edge files; run B4 neural
    controls; write predictions/metrics under:

    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B4_no_edge_neural/
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B4_random_edge_graph/
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B4_knn_graph/
    urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B4_real_cd_graph/

Run from repository root:

    PYTHONPATH=urban_graph_benchmark/src python \
      urban_graph_benchmark/scripts/16_run_qc_cd_b4_graph_controls.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ville_hgnn.baselines.b4_cd_graph_controls import (
    Config,
    DEFAULT_ADJACENCY_EDGES_PATH,
    DEFAULT_KNN_EDGES_PATH,
    DEFAULT_MODELS,
    DEFAULT_NODES_PATH,
    DEFAULT_RANDOM_EDGES_PATH,
    run_b4_cd_graph_controls,
)
from ville_hgnn.baselines.qc_cd_sovi_common import (
    BASELINES_DIR,
    CD_ID_COL,
    CD_NAME_COL,
    DEFAULT_PANEL_PATH,
    SPLIT_COL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run B4 no-edge, random-edge, kNN, and real-adjacency graph-control "
            "baselines for the Québec CD civil-security / SoVI benchmark. This is "
            "a thin runner; all model logic lives in "
            "ville_hgnn.baselines.b4_cd_graph_controls."
        )
    )

    input_group = parser.add_argument_group("inputs")
    input_group.add_argument(
        "--panel-path",
        type=Path,
        default=DEFAULT_PANEL_PATH,
        help=(
            "Predictive CD × month panel. Default: "
            "urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/"
            "datasets/cd_month_panel.parquet"
        ),
    )
    input_group.add_argument(
        "--nodes-path",
        type=Path,
        default=DEFAULT_NODES_PATH,
        help=(
            "CD graph node table. Default: "
            "urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/"
            "datasets/cd_graph_nodes.parquet"
        ),
    )
    input_group.add_argument(
        "--adjacency-edges-path",
        type=Path,
        default=DEFAULT_ADJACENCY_EDGES_PATH,
        help=(
            "Real CD adjacency edge table. Default: "
            "urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/"
            "datasets/cd_graph_edges_adjacency.parquet"
        ),
    )
    input_group.add_argument(
        "--knn-edges-path",
        type=Path,
        default=DEFAULT_KNN_EDGES_PATH,
        help=(
            "Centroid kNN edge table. Default: "
            "urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/"
            "datasets/cd_graph_edges_knn.parquet"
        ),
    )
    input_group.add_argument(
        "--random-edges-path",
        type=Path,
        default=DEFAULT_RANDOM_EDGES_PATH,
        help=(
            "Random/placebo edge table. Default: "
            "urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/"
            "datasets/cd_graph_edges_random_placebo.parquet"
        ),
    )

    output_group = parser.add_argument_group("outputs")
    output_group.add_argument(
        "--base-output-dir",
        type=Path,
        default=BASELINES_DIR,
        help=(
            "Base baseline output directory. Default: "
            "urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/"
        ),
    )

    data_group = parser.add_argument_group("panel columns")
    data_group.add_argument(
        "--target-col",
        default="target_next_3_months",
        help="Target column to forecast. Default: target_next_3_months.",
    )
    data_group.add_argument(
        "--cd-id-col",
        default=CD_ID_COL,
        help=f"CD ID column in the panel. Default: {CD_ID_COL}.",
    )
    data_group.add_argument(
        "--cd-name-col",
        default=CD_NAME_COL,
        help=f"CD name column in the panel. Default: {CD_NAME_COL}.",
    )
    data_group.add_argument(
        "--period-month-col",
        default="period_month",
        help="Panel period-month column. Default: period_month.",
    )
    data_group.add_argument(
        "--split-col",
        default=SPLIT_COL,
        help=f"Train/validation/test split column. Default: {SPLIT_COL}.",
    )

    model_group = parser.add_argument_group("models")
    model_group.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=list(DEFAULT_MODELS),
        help=(
            "B4 models to run. Default: B4_no_edge_neural B4_random_edge_graph "
            "B4_knn_graph B4_real_cd_graph."
        ),
    )

    architecture_group = parser.add_argument_group("neural architecture")
    architecture_group.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for MLP/GraphSAGE models. Default: 64.",
    )
    architecture_group.add_argument(
        "--num-layers",
        type=int,
        default=2,
        choices=[1, 2],
        help="Number of neural/message-passing layers. Default: 2.",
    )
    architecture_group.add_argument(
        "--dropout",
        type=float,
        default=0.10,
        help="Dropout rate. Default: 0.10.",
    )
    architecture_group.add_argument(
        "--no-layer-norm",
        action="store_true",
        help="Disable layer normalization.",
    )
    architecture_group.add_argument(
        "--output-activation",
        default="softplus",
        choices=["softplus", "relu", "identity"],
        help="Output activation for count forecasts. Default: softplus.",
    )

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--max-epochs",
        type=int,
        default=500,
        help="Maximum training epochs. Default: 500.",
    )
    optimization_group.add_argument(
        "--patience",
        type=int,
        default=60,
        help="Early stopping patience on validation MAE. Default: 60.",
    )
    optimization_group.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate. Default: 1e-3.",
    )
    optimization_group.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay. Default: 1e-4.",
    )
    optimization_group.add_argument(
        "--loss",
        default="mse",
        choices=["mse", "huber", "poisson_nll"],
        help="Training loss. Default: mse.",
    )
    optimization_group.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="Huber delta if --loss huber. Default: 1.0.",
    )

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    runtime_group.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, etc. Default: auto.",
    )
    runtime_group.add_argument(
        "--keep-missing-targets",
        action="store_true",
        help=(
            "Keep rows with missing targets in prediction outputs. "
            "Metrics still ignore missing targets."
        ),
    )
    runtime_group.add_argument(
        "--no-clip-predictions",
        action="store_true",
        help="Do not clip predictions at zero.",
    )

    edge_group = parser.add_argument_group("edge handling")
    edge_group.add_argument(
        "--add-self-loops",
        action="store_true",
        help="Add row-level self-loops to graph models.",
    )
    edge_group.add_argument(
        "--no-normalize-edge-weights",
        action="store_true",
        help=(
            "Disable edge-weight normalization flag in config. The current B4 "
            "implementation uses mean aggregation and stores this for audit/control."
        ),
    )

    feature_group = parser.add_argument_group("feature parity / ablation controls")
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
        nodes_path=args.nodes_path,
        adjacency_edges_path=args.adjacency_edges_path,
        knn_edges_path=args.knn_edges_path,
        random_edges_path=args.random_edges_path,
        base_output_dir=args.base_output_dir,
        target_col=args.target_col,
        cd_id_col=args.cd_id_col,
        cd_name_col=args.cd_name_col,
        period_month_col=args.period_month_col,
        split_col=args.split_col,
        models=list(args.models),
        include_sovi_features=not args.no_sovi_features,
        include_history_features=not args.no_history_features,
        include_hazard_history_features=not args.no_hazard_history_features,
        include_current_month_counts=not args.no_current_month_counts,
        include_seasonality=not args.no_seasonality,
        include_year_trend=not args.no_year_trend,
        include_all_other_numeric_features=not args.no_other_numeric_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_layer_norm=not args.no_layer_norm,
        output_activation=args.output_activation,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss=args.loss,
        huber_delta=args.huber_delta,
        random_seed=args.random_seed,
        device=args.device,
        drop_missing_target=not args.keep_missing_targets,
        clip_predictions_at_zero=not args.no_clip_predictions,
        add_self_loops_to_graph_models=args.add_self_loops,
        normalize_edge_weights=not args.no_normalize_edge_weights,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    result = run_b4_cd_graph_controls(config)

    print("B4 graph-control runner completed.")
    print(f"Panel path: {config.panel_path}")
    print(f"Nodes path: {config.nodes_path}")
    print(f"Adjacency edges path: {config.adjacency_edges_path}")
    print(f"kNN edges path: {config.knn_edges_path}")
    print(f"Random edges path: {config.random_edges_path}")
    print(f"Base output directory: {config.base_output_dir}")
    print(f"Target column: {config.target_col}")
    print(f"Feature count: {result.get('feature_count') if isinstance(result, dict) else 'unknown'}")
    print("Models:")
    for model_name in config.models:
        print(f"  - {model_name}")

    summaries = result.get("model_summaries", {}) if isinstance(result, dict) else {}
    if summaries:
        print("Model summaries:")
        for model_name, summary in summaries.items():
            print(
                "  "
                f"{model_name}: "
                f"graph={summary.get('graph_kind')}, "
                f"best_epoch={summary.get('best_epoch')}, "
                f"best_val_mae={summary.get('best_val_mae')}, "
                f"row_edges={summary.get('row_edge_count')}"
            )

    outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
    if outputs:
        print("Outputs:")
        for key, value in outputs.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
