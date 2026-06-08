#!/usr/bin/env python3
"""
Thin CLI wrapper for building split artifacts for Montréal 311 Water/Drainage v0.

This script intentionally contains no split-building, baseline, or modeling logic.
It only:

- parses CLI arguments
- ensures the local ``urban_graph_benchmark/src`` package path is importable
- calls ``ville_hgnn.data.build_splits.run_build_splits()``
- prints a concise split-build summary and written output paths

Run from the repository root:

    python urban_graph_benchmark/scripts/03_build_splits_v0.py

Optional:

    python urban_graph_benchmark/scripts/03_build_splits_v0.py \
      --config urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_CONFIG_PATH = "urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml"


def _bootstrap_package_path() -> None:
    """
    Add ``urban_graph_benchmark/src`` to ``sys.path`` when running from source.

    This keeps the script usable before the package is installed in editable
    mode. It does not perform benchmark, split-building, baseline, or modeling
    logic.
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

    # Let the import fail naturally with a clear ModuleNotFoundError if the
    # package path cannot be found.
    return


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Build split artifacts for the Montréal 311 water/drainage benchmark."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to benchmark YAML/JSON config. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root. Defaults to automatic detection in build_splits.py.",
    )

    return parser.parse_args()


def main() -> None:
    """Run split builder and print a concise summary."""

    _bootstrap_package_path()

    from ville_hgnn.data.build_splits import run_build_splits, split_brief

    args = parse_args()

    result = run_build_splits(
        config_path=args.config,
        repo_root=args.repo_root,
    )

    print(split_brief(result).rstrip())

    written_outputs = result.get("outputs", {})
    if written_outputs:
        print("\nWritten outputs:")
        for label, path in written_outputs.items():
            print(f"  {label}: {path}")
    else:
        print("\nWritten outputs: none reported")


if __name__ == "__main__":
    main()