#!/usr/bin/env python3
"""
Thin CLI wrapper for the Montréal 311 water/drainage input inventory.

This script intentionally contains no inventory logic. It only:

- parses CLI arguments
- ensures the local ``urban_graph_benchmark/src`` package path is importable
- calls ``ville_hgnn.data.inventory.run_inventory()``
- prints a concise summary and written output paths

Run from the repository root:

    python urban_graph_benchmark/scripts/01_inventory_inputs.py

Optional:

    python urban_graph_benchmark/scripts/01_inventory_inputs.py \
      --config urban_graph_benchmark/configs/mtl_311_water_v0.yaml \
      --sample-rows 10000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_CONFIG_PATH = "urban_graph_benchmark/configs/mtl_311_water_v0.yaml"
DEFAULT_SAMPLE_ROWS = 10_000


def _bootstrap_package_path() -> None:
    """
    Add ``urban_graph_benchmark/src`` to ``sys.path`` when running from source.

    This keeps the script usable before the package is installed in editable
    mode. It does not perform benchmark logic.
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
        description="Run input inventory for the Montréal 311 water/drainage benchmark."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to benchmark YAML/JSON config. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root. Defaults to automatic detection in inventory.py.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=DEFAULT_SAMPLE_ROWS,
        help=f"Number of rows to sample from tabular candidate files. Default: {DEFAULT_SAMPLE_ROWS}",
    )

    return parser.parse_args()


def main() -> None:
    """Run inventory and print a concise summary."""

    _bootstrap_package_path()

    from ville_hgnn.data.inventory import inventory_brief, run_inventory

    args = parse_args()

    inventory = run_inventory(
        config_path=args.config,
        repo_root=args.repo_root,
        sample_rows=args.sample_rows,
    )

    print(inventory_brief(inventory).rstrip())

    written_outputs = inventory.get("written_outputs", {})
    if written_outputs:
        print("\nWritten outputs:")
        for label, path in written_outputs.items():
            print(f"  {label}: {path}")
    else:
        print("\nWritten outputs: none reported")


if __name__ == "__main__":
    main()