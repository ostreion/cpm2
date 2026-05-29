#!/usr/bin/env python
"""Canonical headless entry point for CPM2.

Generates a per-run RUN_ID, then drives the Snakemake DAG. Replaces the
papermill-driven scripts/run_benchmark.sh as the recommended way to run a
benchmark; the shell script remains for backward compatibility.

Usage:
    scripts/run.py <config_name> [--cores N] [--dry-run] [extra snakemake args]

Examples:
    scripts/run.py mdm2_p53_v1 --cores 8
    scripts/run.py mdm2_p53_v1 --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path


def _git_sha8(pipeline_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=str(pipeline_root), capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "nogit"
    except (FileNotFoundError, OSError):
        return "nogit"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_name", help="config stem under configs/, e.g. mdm2_p53_v1")
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="snakemake -n")
    parser.add_argument("--run-id", default=None, help="override generated run_id")
    # parse_known_args (not REMAINDER): our own flags get parsed regardless of
    # position, and any flag we don't recognise is forwarded to snakemake.
    args, snakemake_extra = parser.parse_known_args()

    pipeline_root = Path(__file__).resolve().parent.parent
    config_yaml = pipeline_root / "configs" / f"{args.config_name}.yaml"
    if not config_yaml.exists():
        print(f"ERROR: config not found: {config_yaml}", file=sys.stderr)
        return 1

    if args.run_id:
        run_id = args.run_id
    else:
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_id = f"{args.config_name}_{ts}_{_git_sha8(pipeline_root)}"

    run_data_root = pipeline_root / "data" / "runs" / run_id

    # No --use-conda: that would make snakemake recreate envs from
    # envs/*.yml under .snakemake/conda/<hash>/ (slow, duplicates the
    # named envs we already have). Instead, the per-rule helpers shell
    # out via `conda run -n <env>` themselves through the runners.
    # The Snakefile's `conda:` directives stay as documentation.
    # Use the sibling `snakemake` of whichever python is running this
    # script, so PATH-less invocations (e.g. from cron) still find it.
    snakemake_bin = Path(sys.executable).parent / "snakemake"
    cmd = [
        str(snakemake_bin),
        "--cores", str(args.cores),
        "--resources", "gpu=1",
        "--keep-going",
        "--config", f"config_name={args.config_name}", f"run_id={run_id}",
    ]
    if args.dry_run:
        cmd.append("-n")
    if snakemake_extra:
        # Strip leading '--' separator if user added one explicitly.
        extra = snakemake_extra
        if extra and extra[0] == "--":
            extra = extra[1:]
        cmd.extend(extra)

    print("=== CPM2 run ===")
    print(f"config:        {args.config_name}")
    print(f"run_id:        {run_id}")
    print(f"run_data_root: {run_data_root}")
    print(f"command:       {' '.join(cmd)}")
    print("================")

    # Pass the cpm2 python path through env so the Snakefile can use it
    # explicitly in shell: directives. Doing it via PATH-prepend would
    # break `conda run -n <env> python ...` invocations downstream
    # (conda picks up the first `python` on PATH, ignoring `-n`).
    env = dict(os.environ)
    env["CPM2_PYTHON"] = sys.executable
    env.setdefault("CPM2_HEADLESS", "1")
    return subprocess.run(cmd, cwd=str(pipeline_root), env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
