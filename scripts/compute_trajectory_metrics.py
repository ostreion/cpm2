#!/usr/bin/env python
"""Compute trajectory quality metrics for all finished MD jobs in an
overnight benchmark batch.

Usage:
    /path/to/your-mdanalysis-env/bin/python \
        scripts/compute_trajectory_metrics.py \
        [benchmarks/md_20260526_overnight]

Reads ``results_running.csv`` to enumerate jobs, skips any whose
trajectory or MMGBSA output is missing, and writes:

* ``<batch>/trajectory_metrics.csv``  -- one row per finished job
* ``<batch>/trajectory_diagnostics/<job>.png``  -- 3x3 dashboard per job
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trajectory_metrics import compute_job_metrics  # noqa: E402


def _job_dirname(name: str, target: str) -> str:
    """Match logic used by submit script: design jobs are
    ``<target>__<name>``; ref jobs are bare ``<name>``."""
    if name.startswith("ref_"):
        return name
    return f"{target}__{name}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "batch",
        nargs="?",
        default="benchmarks/md_20260526_overnight",
        help="Path to overnight batch dir.",
    )
    p.add_argument("--only", nargs="*", default=None, help="Limit to specific job dir names.")
    args = p.parse_args()

    batch = (REPO_ROOT / args.batch).resolve() if not Path(args.batch).is_absolute() else Path(args.batch)
    jobs_dir = batch / "jobs"
    diag_dir = batch / "trajectory_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    out_csv = batch / "trajectory_metrics.csv"

    enum_csv = batch / "results_running.csv"
    if enum_csv.exists():
        idx = pd.read_csv(enum_csv)
        job_names = [_job_dirname(r["name"], r["target"]) for _, r in idx.iterrows()]
    else:
        job_names = [p.name for p in jobs_dir.iterdir() if p.is_dir()]

    if args.only:
        job_names = [n for n in job_names if n in args.only]

    rows = []
    for name in job_names:
        job_root = jobs_dir / name
        if not job_root.exists():
            print(f"[skip] missing dir: {name}")
            continue
        png_out = diag_dir / f"{name}.png"
        print(f"[run ] {name}")
        try:
            row = compute_job_metrics(job_root, png_out=png_out)
        except Exception as exc:
            traceback.print_exc()
            print(f"[FAIL] {name}: {exc}")
            continue
        if row is None:
            print(f"[skip] not finished: {name}")
            continue
        rows.append(row)
        print(f"[ ok ] {name}  dG_final={row.get('dg_running_final'):.2f}  comDrift={row.get('com_dist_drift'):.2f}A")

    if not rows:
        print("no rows produced")
        return 1
    df = pd.DataFrame(rows)
    # put name first
    cols = ["job_name", "cyclic", "n_frames"] + [c for c in df.columns if c not in ("job_name", "cyclic", "n_frames")]
    df = df[cols]
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
