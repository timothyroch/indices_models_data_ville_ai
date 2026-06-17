#!/usr/bin/env python3
"""
Run the full Québec CD civil-security / SoVI benchmark suite.

Purpose:
    Execute the complete benchmark pipeline in the correct research order:

        10_run_qc_cd_b1_sovi_direct_validation.py
        11_build_qc_cd_civil_security_panel.py
        12_run_qc_cd_b0_history_baselines.py
        13_run_qc_cd_b2_calibrated_sovi.py
        14_run_qc_cd_b3_tabular_feature_parity.py
        15_build_qc_cd_civil_security_graph.py
        16_run_qc_cd_b4_graph_controls.py
        17_compare_qc_cd_civil_security_sovi_benchmark.py

Run from repository root:
    PYTHONPATH=urban_graph_benchmark/src python \
      urban_graph_benchmark/scripts/run_qc_cd_civil_security_sovi_benchmark_suite.py

Design:
    - This is an orchestration script only. It does not duplicate benchmark logic.
    - Each stage writes its own canonical outputs.
    - The suite writes a timestamped run manifest, per-step stdout/stderr logs,
      status tables, and a compact Markdown run report.
    - By default it stops on the first failed step, so the benchmark cannot
      silently continue with stale or missing outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


OUTPUT_ROOT = Path("urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0")
SUITE_RUNS_DIR = OUTPUT_ROOT / "suite_runs"
DEFAULT_PYTHONPATH = "urban_graph_benchmark/src"


@dataclass(frozen=True)
class Step:
    """One executable benchmark stage."""

    step_id: str
    script: Path
    description: str
    expected_outputs: tuple[Path, ...] = ()
    default_args: tuple[str, ...] = ()


STEPS: tuple[Step, ...] = (
    Step(
        step_id="10_b1_sovi_direct_validation",
        script=Path("urban_graph_benchmark/scripts/10_run_qc_cd_b1_sovi_direct_validation.py"),
        description="Run B1 direct SoVI validation against cumulative civil-security targets.",
        expected_outputs=(
            OUTPUT_ROOT / "baselines/B1_sovi_direct_validation/metrics.csv",
            OUTPUT_ROOT / "baselines/B1_sovi_direct_validation/metadata.json",
        ),
    ),
    Step(
        step_id="11_build_panel",
        script=Path("urban_graph_benchmark/scripts/11_build_qc_cd_civil_security_panel.py"),
        description="Build predictive CD × month panel.",
        expected_outputs=(
            OUTPUT_ROOT / "datasets/cd_month_panel.parquet",
            OUTPUT_ROOT / "datasets/cd_month_panel_metadata.json",
        ),
    ),
    Step(
        step_id="12_b0_history",
        script=Path("urban_graph_benchmark/scripts/12_run_qc_cd_b0_history_baselines.py"),
        description="Run B0 history-only baselines.",
        expected_outputs=(
            OUTPUT_ROOT / "baselines/B0_history_only/predictions.parquet",
            OUTPUT_ROOT / "baselines/B0_history_only/metrics.csv",
            OUTPUT_ROOT / "baselines/B0_history_only/metadata.json",
        ),
    ),
    Step(
        step_id="13_b2_calibrated_sovi",
        script=Path("urban_graph_benchmark/scripts/13_run_qc_cd_b2_calibrated_sovi.py"),
        description="Run B2 calibrated SoVI predictors.",
        expected_outputs=(
            OUTPUT_ROOT / "baselines/B2_calibrated_sovi/predictions.parquet",
            OUTPUT_ROOT / "baselines/B2_calibrated_sovi/metrics.csv",
            OUTPUT_ROOT / "baselines/B2_calibrated_sovi/metadata.json",
        ),
    ),
    Step(
        step_id="14_b3_tabular_feature_parity",
        script=Path("urban_graph_benchmark/scripts/14_run_qc_cd_b3_tabular_feature_parity.py"),
        description="Run B3 non-graph feature-parity ML.",
        expected_outputs=(
            OUTPUT_ROOT / "baselines/B3_tabular_feature_parity/predictions.parquet",
            OUTPUT_ROOT / "baselines/B3_tabular_feature_parity/metrics.csv",
            OUTPUT_ROOT / "baselines/B3_tabular_feature_parity/metadata.json",
        ),
    ),
    Step(
        step_id="15_build_graph_assets",
        script=Path("urban_graph_benchmark/scripts/15_build_qc_cd_civil_security_graph.py"),
        description="Build CD graph nodes and adjacency/kNN/random edge files.",
        expected_outputs=(
            OUTPUT_ROOT / "datasets/cd_graph_nodes.parquet",
            OUTPUT_ROOT / "datasets/cd_graph_edges_adjacency.parquet",
            OUTPUT_ROOT / "datasets/cd_graph_edges_knn.parquet",
            OUTPUT_ROOT / "datasets/cd_graph_edges_random_placebo.parquet",
            OUTPUT_ROOT / "datasets/cd_graph_metadata.json",
        ),
    ),
    Step(
        step_id="16_b4_graph_controls",
        script=Path("urban_graph_benchmark/scripts/16_run_qc_cd_b4_graph_controls.py"),
        description="Run B4 no-edge, random-edge, kNN, and real-adjacency graph controls.",
        expected_outputs=(
            OUTPUT_ROOT / "baselines/B4_no_edge_neural/metrics.csv",
            OUTPUT_ROOT / "baselines/B4_random_edge_graph/metrics.csv",
            OUTPUT_ROOT / "baselines/B4_knn_graph/metrics.csv",
            OUTPUT_ROOT / "baselines/B4_real_cd_graph/metrics.csv",
            OUTPUT_ROOT / "baselines/B4_graph_control_comparison/b4_graph_control_comparison.csv",
        ),
    ),
    Step(
        step_id="17_compare",
        script=Path("urban_graph_benchmark/scripts/17_compare_qc_cd_civil_security_sovi_benchmark.py"),
        description="Merge all B1/B0/B2/B3/B4 metrics into final comparison tables and report.",
        expected_outputs=(
            OUTPUT_ROOT / "comparisons/benchmark_comparison.csv",
            OUTPUT_ROOT / "comparisons/benchmark_comparison_compact.csv",
            OUTPUT_ROOT / "comparisons/metrics_long.csv",
            OUTPUT_ROOT / "comparisons/metric_winners.csv",
            OUTPUT_ROOT / "reports/qc_cd_civil_security_sovi_benchmark_report.md",
        ),
    ),
)


@dataclass
class StepResult:
    """Execution result for one step."""

    step_id: str
    script: str
    description: str
    status: str
    command: list[str]
    returncode: int | None
    started_at_utc: str
    ended_at_utc: str
    duration_seconds: float
    stdout_log: str | None
    stderr_log: str | None
    missing_expected_outputs: list[str]
    extra_args: list[str] = field(default_factory=list)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def repo_root_from_cwd() -> Path:
    """
    Return current working directory as repository root.

    This runner is designed to be executed from the repository root. It checks
    for the expected urban_graph_benchmark directory to catch accidental runs
    from the wrong location.
    """
    root = Path.cwd()
    expected = root / "urban_graph_benchmark" / "scripts"
    if not expected.exists():
        raise FileNotFoundError(
            "This suite runner should be run from the repository root. "
            f"Could not find expected directory: {expected}"
        )
    return root


def step_index() -> dict[str, Step]:
    return {step.step_id: step for step in STEPS}


def step_ids() -> list[str]:
    return [step.step_id for step in STEPS]


def parse_step_args(values: Sequence[str] | None) -> dict[str, list[str]]:
    """
    Parse repeated --step-args entries.

    Format:
        --step-args "16_b4_graph_controls::--max-epochs 100 --patience 20"
        --step-args "14_b3_tabular_feature_parity::--models ridge random_forest"

    Also accepts numeric script prefixes:
        --step-args "16::--max-epochs 100"
    """
    mapping: dict[str, list[str]] = {}
    if not values:
        return mapping

    valid = step_index()

    for raw in values:
        if "::" not in raw:
            raise ValueError(
                f"Invalid --step-args value: {raw!r}. "
                "Expected format STEP_ID::ARG_STRING."
            )

        key, arg_string = raw.split("::", 1)
        key = key.strip()
        arg_string = arg_string.strip()

        matched = None
        if key in valid:
            matched = key
        else:
            matches = [sid for sid in valid if sid.startswith(key + "_") or sid.startswith(key)]
            if len(matches) == 1:
                matched = matches[0]

        if matched is None:
            raise ValueError(
                f"Unknown or ambiguous step key in --step-args: {key!r}. "
                f"Valid step IDs: {step_ids()}"
            )

        mapping.setdefault(matched, []).extend(shlex.split(arg_string))

    return mapping


def selected_steps(
    *,
    only: Sequence[str] | None,
    skip: Sequence[str] | None,
    start_at: str | None,
    stop_after: str | None,
) -> list[Step]:
    """Resolve step-selection CLI options."""
    steps = list(STEPS)
    valid_ids = step_ids()

    def resolve_key(key: str) -> str:
        if key in valid_ids:
            return key
        matches = [sid for sid in valid_ids if sid.startswith(key + "_") or sid.startswith(key)]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"Unknown or ambiguous step key: {key!r}. Valid: {valid_ids}")

    if only:
        only_ids = [resolve_key(x) for x in only]
        only_set = set(only_ids)
        steps = [s for s in steps if s.step_id in only_set]

    if start_at:
        start_id = resolve_key(start_at)
        idx = valid_ids.index(start_id)
        steps = [s for s in steps if valid_ids.index(s.step_id) >= idx]

    if stop_after:
        stop_id = resolve_key(stop_after)
        idx = valid_ids.index(stop_id)
        steps = [s for s in steps if valid_ids.index(s.step_id) <= idx]

    if skip:
        skip_ids = {resolve_key(x) for x in skip}
        steps = [s for s in steps if s.step_id not in skip_ids]

    if not steps:
        raise ValueError("Step selection is empty after applying --only/--skip/--start-at/--stop-after.")

    return steps


def make_run_dir(base_dir: Path, run_name: str | None) -> Path:
    if run_name:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in run_name)
    else:
        safe = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base_dir / safe
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def build_env(pythonpath: str, inherit_env: bool = True) -> dict[str, str]:
    env = dict(os.environ) if inherit_env else {}
    existing = env.get("PYTHONPATH", "")
    paths = [pythonpath]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def check_scripts_exist(steps: Sequence[Step]) -> list[str]:
    missing = []
    for step in steps:
        if not step.script.exists():
            missing.append(str(step.script))
    return missing


def outputs_missing(step: Step) -> list[str]:
    return [str(path) for path in step.expected_outputs if not path.exists()]


def outputs_ready(step: Step) -> bool:
    return len(outputs_missing(step)) == 0


def command_for_step(
    *,
    python_executable: str,
    step: Step,
    extra_args: Sequence[str],
) -> list[str]:
    return [
        python_executable,
        str(step.script),
        *step.default_args,
        *extra_args,
    ]


def run_step(
    *,
    step: Step,
    command: list[str],
    env: dict[str, str],
    run_dir: Path,
    dry_run: bool,
    resume: bool,
    extra_args: list[str],
) -> StepResult:
    started = utc_now_iso()
    start_time = time.time()

    log_stem = f"{step.step_id}"
    stdout_log = run_dir / "logs" / f"{log_stem}.stdout.log"
    stderr_log = run_dir / "logs" / f"{log_stem}.stderr.log"

    if resume and outputs_ready(step):
        ended = utc_now_iso()
        stdout_log.write_text(
            "Skipped by --resume because all expected outputs already existed.\n"
            f"Expected outputs:\n" + "\n".join(str(p) for p in step.expected_outputs) + "\n",
            encoding="utf-8",
        )
        stderr_log.write_text("", encoding="utf-8")
        return StepResult(
            step_id=step.step_id,
            script=str(step.script),
            description=step.description,
            status="skipped_ready",
            command=command,
            returncode=0,
            started_at_utc=started,
            ended_at_utc=ended,
            duration_seconds=time.time() - start_time,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            missing_expected_outputs=[],
            extra_args=list(extra_args),
        )

    if dry_run:
        ended = utc_now_iso()
        stdout_log.write_text(
            "Dry run only. Command was not executed.\n\n"
            + " ".join(shlex.quote(x) for x in command)
            + "\n",
            encoding="utf-8",
        )
        stderr_log.write_text("", encoding="utf-8")
        return StepResult(
            step_id=step.step_id,
            script=str(step.script),
            description=step.description,
            status="dry_run",
            command=command,
            returncode=None,
            started_at_utc=started,
            ended_at_utc=ended,
            duration_seconds=time.time() - start_time,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            missing_expected_outputs=outputs_missing(step),
            extra_args=list(extra_args),
        )

    with stdout_log.open("w", encoding="utf-8") as out_f, stderr_log.open("w", encoding="utf-8") as err_f:
        out_f.write(f"$ {' '.join(shlex.quote(x) for x in command)}\n\n")
        out_f.flush()

        proc = subprocess.run(
            command,
            stdout=out_f,
            stderr=err_f,
            text=True,
            env=env,
            cwd=Path.cwd(),
        )

    ended = utc_now_iso()
    missing = outputs_missing(step)

    if proc.returncode != 0:
        status = "failed_returncode"
    elif missing:
        status = "failed_missing_outputs"
    else:
        status = "success"

    return StepResult(
        step_id=step.step_id,
        script=str(step.script),
        description=step.description,
        status=status,
        command=command,
        returncode=int(proc.returncode),
        started_at_utc=started,
        ended_at_utc=ended,
        duration_seconds=time.time() - start_time,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        missing_expected_outputs=missing,
        extra_args=list(extra_args),
    )


def write_status_csv(results: Sequence[StepResult], path: Path) -> None:
    rows = []
    for result in results:
        row = asdict(result)
        row["command"] = " ".join(shlex.quote(x) for x in result.command)
        row["extra_args"] = " ".join(shlex.quote(x) for x in result.extra_args)
        row["missing_expected_outputs"] = ";".join(result.missing_expected_outputs)
        rows.append(row)

    # Avoid importing pandas just for this small table.
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    *,
    run_dir: Path,
    results: Sequence[StepResult],
    args: argparse.Namespace,
    selected: Sequence[Step],
    started_at_utc: str,
    ended_at_utc: str,
) -> Path:
    manifest = {
        "suite": "qc_cd_civil_security_sovi_benchmark_suite",
        "purpose": "Run full B1/B0/B2/B3/B4 Québec CD civil-security / SoVI benchmark in order.",
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_seconds": sum(r.duration_seconds for r in results),
        "repo_root": str(Path.cwd()),
        "python_executable": args.python,
        "pythonpath": args.pythonpath,
        "continue_on_error": bool(args.continue_on_error),
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
        "selected_steps": [s.step_id for s in selected],
        "all_steps_order": [s.step_id for s in STEPS],
        "results": [asdict(r) for r in results],
        "outputs": {
            "status_csv": str(run_dir / "suite_status.csv"),
            "report_md": str(run_dir / "suite_report.md"),
            "logs_dir": str(run_dir / "logs"),
        },
    }

    path = run_dir / "suite_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def status_symbol(status: str) -> str:
    if status == "success":
        return "success"
    if status == "skipped_ready":
        return "skipped_ready"
    if status == "dry_run":
        return "dry_run"
    return "failed"


def write_markdown_report(
    *,
    run_dir: Path,
    results: Sequence[StepResult],
    selected: Sequence[Step],
    started_at_utc: str,
    ended_at_utc: str,
    manifest_path: Path,
) -> Path:
    success_count = sum(1 for r in results if r.status == "success")
    skipped_count = sum(1 for r in results if r.status == "skipped_ready")
    failed_count = sum(1 for r in results if r.status.startswith("failed"))

    lines = [
        "# Québec CD civil-security / SoVI benchmark suite run",
        "",
        f"Started: **{started_at_utc}**",
        f"Ended: **{ended_at_utc}**",
        "",
        "## Summary",
        "",
        f"- Steps selected: **{len(selected)}**",
        f"- Success: **{success_count}**",
        f"- Skipped by resume: **{skipped_count}**",
        f"- Failed: **{failed_count}**",
        f"- Manifest: `{manifest_path}`",
        "",
        "## Step status",
        "",
        "| Status | Step | Duration seconds | Return code | Missing expected outputs | Logs |",
        "|---|---|---:|---:|---|---|",
    ]

    for result in results:
        missing = "<br>".join(f"`{x}`" for x in result.missing_expected_outputs) if result.missing_expected_outputs else "—"
        logs = f"[stdout]({Path(result.stdout_log).name if result.stdout_log else ''}) / [stderr]({Path(result.stderr_log).name if result.stderr_log else ''})"
        lines.append(
            "| "
            f"{status_symbol(result.status)} {result.status} | "
            f"`{result.step_id}` | "
            f"{result.duration_seconds:.2f} | "
            f"{result.returncode if result.returncode is not None else '—'} | "
            f"{missing} | "
            f"`logs/{Path(result.stdout_log).name if result.stdout_log else ''}` / "
            f"`logs/{Path(result.stderr_log).name if result.stderr_log else ''}` |"
        )

    lines.extend(
        [
            "",
            "## Commands",
            "",
        ]
    )

    for result in results:
        lines.extend(
            [
                f"### {result.step_id}",
                "",
                "```bash",
                " ".join(shlex.quote(x) for x in result.command),
                "```",
                "",
            ]
        )

    report_path = run_dir / "suite_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def print_live_step_header(step: Step, command: list[str], index: int, total: int) -> None:
    print("")
    print("=" * 88)
    print(f"[{index}/{total}] {step.step_id}")
    print(step.description)
    print("$ " + " ".join(shlex.quote(x) for x in command))
    print("=" * 88)


def print_result(result: StepResult) -> None:
    print(
        f"{status_symbol(result.status)} {result.step_id}: "
        f"{result.status} in {result.duration_seconds:.2f}s"
    )
    if result.stdout_log:
        print(f"  stdout: {result.stdout_log}")
    if result.stderr_log:
        print(f"  stderr: {result.stderr_log}")
    if result.missing_expected_outputs:
        print("  missing expected outputs:")
        for path in result.missing_expected_outputs:
            print(f"    - {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Québec CD civil-security / SoVI benchmark suite in order."
    )

    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run each stage. Default: current interpreter.",
    )
    parser.add_argument(
        "--pythonpath",
        default=DEFAULT_PYTHONPATH,
        help=f"PYTHONPATH prepended for child processes. Default: {DEFAULT_PYTHONPATH}.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Explicit suite run directory. Default: "
            "urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/"
            "suite_runs/<timestamp>/"
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name under suite_runs/. Ignored if --run-dir is given.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later steps after a failure. Default: stop on first failure.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip steps whose expected outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write manifest/logs but do not execute steps.",
    )

    selection = parser.add_argument_group("step selection")
    selection.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only selected step IDs or numeric prefixes, preserving suite order.",
    )
    selection.add_argument(
        "--skip",
        nargs="+",
        default=None,
        help="Skip selected step IDs or numeric prefixes.",
    )
    selection.add_argument(
        "--start-at",
        default=None,
        help="Start at this step ID or numeric prefix.",
    )
    selection.add_argument(
        "--stop-after",
        default=None,
        help="Stop after this step ID or numeric prefix.",
    )

    parser.add_argument(
        "--step-args",
        action="append",
        default=None,
        help=(
            "Extra arguments for a specific step. Format: "
            "'STEP_ID::--arg value --flag'. May be repeated. "
            "Example: --step-args '16::--max-epochs 100 --patience 20'"
        ),
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip preflight checks for script existence.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root_from_cwd()

    selected = selected_steps(
        only=args.only,
        skip=args.skip,
        start_at=args.start_at,
        stop_after=args.stop_after,
    )

    if not args.no_preflight:
        missing_scripts = check_scripts_exist(selected)
        if missing_scripts:
            print("Missing required suite scripts:")
            for script in missing_scripts:
                print(f"  - {script}")
            raise SystemExit(2)

    step_extra_args = parse_step_args(args.step_args)

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(SUITE_RUNS_DIR, args.run_name)

    env = build_env(args.pythonpath)

    started_at = utc_now_iso()
    results: list[StepResult] = []

    print("Québec CD civil-security / SoVI benchmark suite")
    print(f"Run directory: {run_dir}")
    print(f"Selected steps: {', '.join(s.step_id for s in selected)}")
    print(f"Stop on first failure: {not args.continue_on_error}")
    print(f"Resume: {args.resume}")
    print(f"Dry run: {args.dry_run}")

    for i, step in enumerate(selected, start=1):
        extra_args = step_extra_args.get(step.step_id, [])
        command = command_for_step(
            python_executable=args.python,
            step=step,
            extra_args=extra_args,
        )

        print_live_step_header(step, command, i, len(selected))

        result = run_step(
            step=step,
            command=command,
            env=env,
            run_dir=run_dir,
            dry_run=args.dry_run,
            resume=args.resume,
            extra_args=extra_args,
        )
        results.append(result)
        print_result(result)

        if result.status.startswith("failed") and not args.continue_on_error:
            print("")
            print("Stopping on first failed step. Use --continue-on-error to keep going.")
            break

    ended_at = utc_now_iso()

    status_path = run_dir / "suite_status.csv"
    write_status_csv(results, status_path)

    manifest_path = write_manifest(
        run_dir=run_dir,
        results=results,
        args=args,
        selected=selected,
        started_at_utc=started_at,
        ended_at_utc=ended_at,
    )

    report_path = write_markdown_report(
        run_dir=run_dir,
        results=results,
        selected=selected,
        started_at_utc=started_at,
        ended_at_utc=ended_at,
        manifest_path=manifest_path,
    )

    failures = [r for r in results if r.status.startswith("failed")]

    print("")
    print("=" * 88)
    print("Suite finished.")
    print(f"Status CSV: {status_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Markdown report: {report_path}")
    print("=" * 88)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
