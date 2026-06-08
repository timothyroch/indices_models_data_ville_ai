#!/usr/bin/env python3
"""
Thin CLI wrapper for building Montréal 311 Water/Drainage Dataset v0.

This script intentionally contains no dataset-building logic. It only:

- parses CLI arguments
- ensures the local ``urban_graph_benchmark/src`` package path is importable
- calls ``ville_hgnn.data.build_tract_month_panel.run_build_dataset()``
- prints a concise build summary and written output paths

Run from the repository root:

    python urban_graph_benchmark/scripts/02_build_dataset_v0.py

Optional:

    python urban_graph_benchmark/scripts/02_build_dataset_v0.py \
      --config urban_graph_benchmark/configs/mtl_311_water_v0.yaml
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
    mode. It does not perform benchmark or dataset-building logic.
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
        description="Build Dataset v0 for the Montréal 311 water/drainage benchmark."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to benchmark YAML/JSON config. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root. Defaults to automatic detection in build_tract_month_panel.py.",
    )

    return parser.parse_args()


def main() -> None:
    """Run Dataset v0 builder and print a concise summary."""

    _bootstrap_package_path()

    from ville_hgnn.data.build_tract_month_panel import build_brief, run_build_dataset

    args = parse_args()

    result = run_build_dataset(
        config_path=args.config,
        repo_root=args.repo_root,
    )

    print(build_brief(result).rstrip())

    written_outputs = result.get("outputs", {})
    if written_outputs:
        print("\nWritten outputs:")
        for label, path in written_outputs.items():
            print(f"  {label}: {path}")
    else:
        print("\nWritten outputs: none reported")


if __name__ == "__main__":
    main()