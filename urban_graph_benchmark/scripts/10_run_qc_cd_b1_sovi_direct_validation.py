#!/usr/bin/env python3
"""
Thin runner for B1 direct SoVI external validation against Québec civil-security targets.

This file intentionally contains no benchmark logic. The reusable implementation lives in:

    ville_hgnn.baselines.b1_sovi_direct_validation

Run from repository root:

    PYTHONPATH=urban_graph_benchmark/src python \
      urban_graph_benchmark/scripts/10_run_qc_cd_b1_sovi_direct_validation.py \
      --sovi-score-col score_normalized_0_1 \
      --cd-boundaries data/2021-census-division-boundary-file/lcd_000b21a_e/lcd_000b21a_e.shp \
      --cd-boundary-id-col CDUID \
      --cd-boundary-name-col CDNAME
"""

from __future__ import annotations

import argparse
import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from ville_hgnn.baselines.b1_sovi_direct_validation import (
    Config,
    run_b1_sovi_direct_validation,
)


DEFAULT_TARGETS_PATH = Path(
    "data/external/quebec_civil_security_events/processed/"
    "cd_civil_security_sovi_validation_targets_cumulative.parquet"
)

DEFAULT_OUTPUT_DIR = Path(
    "urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/"
    "baselines/B1_sovi_direct_validation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run B1 direct SoVI external validation against Québec civil-security "
            "event-burden targets. This is a thin runner; all benchmark logic is "
            "implemented in ville_hgnn.baselines.b1_sovi_direct_validation."
        )
    )

    parser.add_argument(
        "--targets-path",
        type=Path,
        default=DEFAULT_TARGETS_PATH,
        help="Aligned CD-level SoVI/civil-security target table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for B1 validation metrics/reports.",
    )
    parser.add_argument(
        "--sovi-score-col",
        default=None,
        help="SoVI score column to use as the direct ranking signal.",
    )
    parser.add_argument(
        "--cd-id-col",
        default=None,
        help="CD identifier column in the aligned target table.",
    )
    parser.add_argument(
        "--cd-name-col",
        default=None,
        help="CD display-name column in the aligned target table.",
    )
    parser.add_argument(
        "--reverse-sovi-score",
        action="store_true",
        help="Reverse SoVI score orientation if lower score means higher vulnerability.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for random/null controls, permutations, and bootstrap sampling.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of permutation-null runs.",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=2000,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--cd-boundaries",
        type=Path,
        default=None,
        help="Optional CD boundary file used for CD-name repair/audits.",
    )
    parser.add_argument(
        "--cd-boundary-id-col",
        default=None,
        help="CD ID column in the boundary file, for example CDUID.",
    )
    parser.add_argument(
        "--cd-boundary-name-col",
        default=None,
        help="CD name column in the boundary file, for example CDNAME.",
    )

    return parser.parse_args()


def config_field_names() -> set[str]:
    """Return field/parameter names accepted by Config."""
    if is_dataclass(Config):
        return {field.name for field in fields(Config)}

    signature = inspect.signature(Config)
    return {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }


def build_config(args: argparse.Namespace) -> Config:
    """
    Instantiate the package Config without duplicating benchmark logic.

    This passes only fields accepted by the module's Config class, which makes the
    thin runner resilient to small Config changes while keeping the CLI stable.
    """
    cli_values: dict[str, Any] = {
        "targets_path": args.targets_path,
        "output_dir": args.output_dir,
        "sovi_score_col": args.sovi_score_col,
        "cd_id_col": args.cd_id_col,
        "cd_name_col": args.cd_name_col,
        "reverse_sovi_score": args.reverse_sovi_score,
        "random_seed": args.random_seed,
        "n_permutations": args.n_permutations,
        "n_bootstraps": args.n_bootstraps,
        "cd_boundaries": args.cd_boundaries,
        "cd_boundary_id_col": args.cd_boundary_id_col,
        "cd_boundary_name_col": args.cd_boundary_name_col,
    }

    accepted = config_field_names()
    kwargs = {key: value for key, value in cli_values.items() if key in accepted}

    # Common aliases in case the refactored module used slightly different names.
    aliases = {
        "targets_path": ["input_path", "target_path", "targets_input_path"],
        "output_dir": ["out_dir", "benchmark_output_dir"],
        "sovi_score_col": ["score_col", "index_score_col"],
        "cd_id_col": ["id_col"],
        "cd_name_col": ["name_col"],
        "random_seed": ["seed"],
        "n_permutations": ["num_permutations", "permutations"],
        "n_bootstraps": ["num_bootstraps", "bootstraps"],
        "cd_boundaries": ["cd_boundaries_path", "cd_boundary_path", "boundary_path"],
        "cd_boundary_id_col": ["boundary_id_col"],
        "cd_boundary_name_col": ["boundary_name_col"],
    }

    for cli_key, possible_config_keys in aliases.items():
        for config_key in possible_config_keys:
            if config_key in accepted and config_key not in kwargs:
                kwargs[config_key] = cli_values[cli_key]

    return Config(**kwargs)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    result = run_b1_sovi_direct_validation(config)

    print("B1 direct SoVI validation runner completed.")
    print(f"Targets path: {args.targets_path}")
    print(f"Output directory: {args.output_dir}")
    if isinstance(result, dict):
        status = result.get("status")
        if status is not None:
            print(f"Status: {status}")


if __name__ == "__main__":
    main()
