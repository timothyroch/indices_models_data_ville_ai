#!/usr/bin/env python3
"""
Thin CLI wrapper for running A0 and A1 baselines for Montréal 311 Water/Drainage v0.

This script intentionally contains no baseline methodology. It only:

- parses CLI arguments
- ensures the local ``urban_graph_benchmark/src`` package path is importable
- calls ``ville_hgnn.baselines.a0_naive_temporal.run_a0_naive_temporal()``
- calls ``ville_hgnn.baselines.a1_svi_direct_ranking.run_a1_svi_direct_ranking()``
- prints concise summaries and written output paths

Run from the repository root:

    python urban_graph_benchmark/scripts/04_run_a0_a1_baselines.py

Optional:

    python urban_graph_benchmark/scripts/04_run_a0_a1_baselines.py \
      --config urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml \
      --split-scheme temporal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_CONFIG_PATH = "urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml"
DEFAULT_SPLIT_SCHEME = "temporal"


def _bootstrap_package_path() -> None:
    """
    Add ``urban_graph_benchmark/src`` to ``sys.path`` when running from source.

    This keeps the script usable before the package is installed in editable
    mode. It does not perform benchmark, baseline, or modeling logic.
    """

    script_path = Path(__file__).resolve()

    candidate_src_paths: list[Path] = []

    for parent in [script_path.parent, *script_path.parents]:
        candidate_src_paths.append(parent / "urban_graph_benchmark" / "src")
        candidate_src_paths.append(parent / "src")

    for src_path in candidate_src_paths:
        package_dir = src_path / "ville_hgnn"
        if package_dir.exists() and package_dir.is_dir():
            src_str = str(src_path)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            return

    # Let imports fail naturally with a clear ModuleNotFoundError if the package
    # path cannot be found.
    return


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Run A0 and A1 baselines for the Montréal 311 water/drainage benchmark."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to benchmark YAML/JSON config. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root. Defaults to automatic detection in baseline modules.",
    )
    parser.add_argument(
        "--split-scheme",
        default=DEFAULT_SPLIT_SCHEME,
        choices=["temporal", "random_debug", "spatial_block"],
        help=f"Split scheme to evaluate. Default: {DEFAULT_SPLIT_SCHEME}",
    )
    parser.add_argument(
        "--skip-a0",
        action="store_true",
        help="Skip A0 naive temporal baselines.",
    )
    parser.add_argument(
        "--skip-a1",
        action="store_true",
        help="Skip A1 SVI direct-ranking baseline.",
    )

    return parser.parse_args()


def print_outputs(label: str, outputs: dict[str, str] | None) -> None:
    """Print output paths in a stable format."""

    print(f"\n{label} written outputs:")

    if not outputs:
        print("  none reported")
        return

    for output_label, path in outputs.items():
        print(f"  {output_label}: {path}")


def main() -> None:
    """Run A0/A1 baselines and print summaries."""

    _bootstrap_package_path()

    from ville_hgnn.baselines.a0_naive_temporal import a0_brief, run_a0_naive_temporal
    from ville_hgnn.baselines.a1_svi_direct_ranking import a1_brief, run_a1_svi_direct_ranking

    args = parse_args()

    if args.skip_a0 and args.skip_a1:
        raise SystemExit("Both --skip-a0 and --skip-a1 were provided; nothing to run.")

    results: dict[str, dict[str, object]] = {}

    if not args.skip_a0:
        print("Running A0 naive temporal/exposure baselines...")
        a0_result = run_a0_naive_temporal(
            config_path=args.config,
            repo_root=args.repo_root,
            split_scheme=args.split_scheme,
        )
        results["A0"] = a0_result
        print()
        print(a0_brief(a0_result).rstrip())
        print_outputs("A0", a0_result.get("outputs", {}))

    if not args.skip_a1:
        if results:
            print("\n" + "=" * 80 + "\n")

        print("Running A1 SVI direct-ranking baseline...")
        a1_result = run_a1_svi_direct_ranking(
            config_path=args.config,
            repo_root=args.repo_root,
            split_scheme=args.split_scheme,
        )
        results["A1"] = a1_result
        print()
        print(a1_brief(a1_result).rstrip())
        print_outputs("A1", a1_result.get("outputs", {}))

    print("\nA0/A1 baseline wrapper completed.")


if __name__ == "__main__":
    main()