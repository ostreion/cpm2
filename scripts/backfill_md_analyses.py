"""Backfill md_analyses outputs for finished MD benchmark jobs.

Usage
-----
  python scripts/backfill_md_analyses.py [BUNDLE_OR_JOB_PATH ...] [--stride N] [--overwrite]

Each positional argument can be:
  - a bundle directory (contains a ``jobs/`` subdirectory), in which case
    every job under ``jobs/`` that has ``free_run/run.nc``,
    ``system_wb.prmtop``, and ``ranges.env`` is processed;
  - a job directory directly (must have those three files).

Per-job failures are caught and reported as warnings; processing continues.
Jobs that already have all three outputs (and whose inputs have not changed)
are skipped unless ``--overwrite`` is set.

Output per job line:
  <job_name> -> n_residues=NN n_frames=NN took=X.Xs
  <job_name> -> SKIPPED (outputs up to date)
  <job_name> -> FAILED: <error message>
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

# Ensure the project src/ is importable when run directly from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from md_analyses import compute_and_write  # noqa: E402


def _is_finished_job(job_dir: Path) -> bool:
    """Return True if the job has the three required input files."""
    return (
        (job_dir / "free_run" / "run.nc").exists()
        and (job_dir / "system_wb.prmtop").exists()
        and (job_dir / "ranges.env").exists()
    )


def _collect_jobs(path: Path) -> list[Path]:
    """Expand a bundle or job path into a list of job directories."""
    jobs_dir = path / "jobs"
    if jobs_dir.is_dir():
        return sorted(
            j for j in jobs_dir.iterdir() if j.is_dir() and _is_finished_job(j)
        )
    if _is_finished_job(path):
        return [path]
    warnings.warn(
        f"Skipping {path}: not a bundle (no jobs/ subdir) and not a finished job "
        "(missing run.nc, system_wb.prmtop, or ranges.env)."
    )
    return []


def _already_done(job_dir: Path) -> bool:
    """Return True if all three outputs exist and inputs haven't changed."""
    out_dir = job_dir / "analysis"
    out_pr = out_dir / "per_residue.parquet"
    out_tr = out_dir / "traces.parquet"
    out_meta = out_dir / "meta.json"
    if not (out_pr.exists() and out_tr.exists() and out_meta.exists()):
        return False
    meta_mtime = out_meta.stat().st_mtime
    nc_mtime = (job_dir / "free_run" / "run.nc").stat().st_mtime
    prm_mtime = (job_dir / "system_wb.prmtop").stat().st_mtime
    return meta_mtime >= nc_mtime and meta_mtime >= prm_mtime


def process_job(job_dir: Path, stride: int, overwrite: bool) -> None:
    name = job_dir.name

    if not overwrite and _already_done(job_dir):
        print(f"{name} -> SKIPPED (outputs up to date)")
        return

    t0 = time.monotonic()
    try:
        meta = compute_and_write(job_dir, stride=stride, overwrite=overwrite)
    except Exception as exc:
        print(f"{name} -> FAILED: {exc}", file=sys.stderr)
        return

    elapsed = time.monotonic() - t0
    n_frames = meta.get("n_frames", "?")

    # Count residues from the written parquet
    try:
        import pandas as pd
        pr = pd.read_parquet(job_dir / "analysis" / "per_residue.parquet")
        n_residues = len(pr)
    except Exception:
        n_residues = "?"

    print(f"{name} -> n_residues={n_residues} n_frames={n_frames} took={elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill md_analyses parquet outputs for finished MD benchmark jobs."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="BUNDLE_OR_JOB_PATH",
        help="Bundle directory (contains jobs/) or individual job directory.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        metavar="N",
        help="Load every N-th frame (default: 1 = all frames).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing outputs.",
    )
    args = parser.parse_args()

    all_jobs: list[Path] = []
    for raw in args.paths:
        p = Path(raw).resolve()
        if not p.exists():
            warnings.warn(f"Path does not exist, skipping: {p}")
            continue
        all_jobs.extend(_collect_jobs(p))

    if not all_jobs:
        print("No finished jobs found. Nothing to do.")
        return

    print(f"Processing {len(all_jobs)} job(s) with stride={args.stride} overwrite={args.overwrite}\n")
    for job in all_jobs:
        process_job(job, stride=args.stride, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
