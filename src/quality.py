"""cpm2.quality -- read-only loaders for the two "above the per-run" lenses
on pipeline quality:

* MLflow filesystem store (``mlruns/<experiment>/<run>/``) -> one row per
  pipeline invocation, with per-rule timing + per-match ipTM/pLDDT.
* MD/MMGBSA overnight benchmark bundles (``benchmarks/md_*_overnight/``) ->
  per-job status, AmberTools MMGBSA dG, shortlist join, monitor log tail,
  trajectory + verification image discovery.

Pure pandas/pathlib/regex; no mlflow, no MDAnalysis, no torch. The
notebooks/CPM2_eval.py UI layer is built strictly on top of this.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
MLRUNS_DIR = _REPO_ROOT / "mlruns"
BENCHMARKS_DIR = _REPO_ROOT / "benchmarks"

# AmberTools' MMPBSA.py writes per-job results to
#   <job>/mmgbsa/AL_out/AL_output.dat
# with a "DELTA TOTAL  -50.43  5.36  0.75" line at the end of the GB block.
_DELTA_TOTAL_RE = re.compile(
    r"^\s*DELTA TOTAL\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)",
    re.MULTILINE,
)

# Production MD pipeline stages, in order. The presence of each subdir's
# canonical .rst file decides the furthest stage a job has reached.
MD_STAGE_FILES = [
    ("em",       "em/em_2.rst"),
    ("heat",     "heat/heat.rst"),
    ("nvt",      "nvt/nvt_2.rst"),
    ("npt",      "npt/npt_3.rst"),
    ("free_run", "free_run/run.rst"),
    ("mmgbsa",   "mmgbsa/AL_out/AL_output.dat"),
]


# --------------------------------------------------------------------------
# MLflow filesystem loader
# --------------------------------------------------------------------------
def _read_metric(path: Path) -> float | None:
    """MLflow metric file: ``<ms> <value> <step>`` per line; return last."""
    try:
        last = path.read_text().strip().splitlines()[-1]
        return float(last.split()[1])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def _read_tag_or_param(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None


def _scan_mlflow_run(run_dir: Path) -> dict | None:
    """Flatten one ``mlruns/<exp>/<run>/`` directory into a single row.

    Returns None if the run is empty / corrupt.
    """
    meta = run_dir / "meta.yaml"
    if not meta.is_file():
        return None
    # Hand-parsed (keep yaml dependency optional) -- only a handful of fields.
    info: dict[str, str] = {}
    for line in meta.read_text().splitlines():
        if ":" in line and not line.startswith(" "):
            k, _, v = line.partition(":")
            info[k.strip()] = v.strip().strip("'\"")

    out: dict = {
        "mlflow_run_id":   run_dir.name,
        "run_name":        info.get("run_name") or run_dir.name,
        "status":          {"1": "RUNNING", "2": "SCHEDULED", "3": "FINISHED",
                            "4": "FAILED", "5": "KILLED"}.get(info.get("status", ""),
                                                              info.get("status", "")),
        "start_time":      pd.to_datetime(int(info.get("start_time") or 0),
                                          unit="ms", errors="coerce"),
        "end_time":        pd.to_datetime(int(info.get("end_time") or 0),
                                          unit="ms", errors="coerce"),
        "experiment_id":   info.get("experiment_id"),
        "user_id":         info.get("user_id"),
        "artifact_uri":    info.get("artifact_uri"),
    }
    out["wall_seconds"] = (out["end_time"] - out["start_time"]).total_seconds() \
        if out["end_time"] is not pd.NaT and out["start_time"] is not pd.NaT else None

    tags_dir, params_dir, metrics_dir = (run_dir / "tags",
                                         run_dir / "params",
                                         run_dir / "metrics")
    if tags_dir.is_dir():
        for f in tags_dir.iterdir():
            out[f"tag.{f.name}"] = _read_tag_or_param(f)
    if params_dir.is_dir():
        for f in params_dir.iterdir():
            out[f"param.{f.name}"] = _read_tag_or_param(f)
    if metrics_dir.is_dir():
        for f in metrics_dir.iterdir():
            out[f"metric.{f.name}"] = _read_metric(f)
    return out


def list_mlflow_runs(mlruns_dir: Path = MLRUNS_DIR) -> pd.DataFrame:
    """Tabulate every MLflow run found under ``mlruns/``.

    One row per run. Columns are flattened with ``tag.``, ``param.``, and
    ``metric.`` prefixes; the rest is core run metadata.
    """
    if not mlruns_dir.is_dir():
        return pd.DataFrame()
    rows: list[dict] = []
    for exp in sorted(p for p in mlruns_dir.iterdir() if p.is_dir()):
        if exp.name in ("models", ".trash"):
            continue
        for run in sorted(p for p in exp.iterdir() if p.is_dir()):
            row = _scan_mlflow_run(run)
            if row is not None:
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "start_time" in df.columns:
        df = df.sort_values("start_time", ascending=False, ignore_index=True)
    return df


def mlflow_summary(df: pd.DataFrame) -> pd.DataFrame:
    """A clutter-free projection of list_mlflow_runs() for the overview table.

    Picks the columns a scientist actually wants to scan: name, status, wall
    time, designs, top ipTM/pLDDT, and the most expensive per-rule timing.
    """
    if df.empty:
        return df

    rule_cols = [c for c in df.columns if c.endswith("_seconds_total")
                 or (c.endswith("_seconds") and not c.endswith("_seconds_mean"))]
    rule_cols = [c for c in rule_cols if c.startswith("metric.")]

    keep = [
        "run_name", "status", "start_time", "wall_seconds",
        "tag.config.name", "tag.git.sha", "tag.git.dirty",
        "metric.num_designs_PH", "metric.num_matches_PH",
        "metric.top_iptm_PH", "metric.top_plddt_PH",
    ] + rule_cols
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    # Headline timings, prettified.
    rename = {
        "tag.config.name": "config",
        "tag.git.sha": "git_sha",
        "tag.git.dirty": "git_dirty",
        "metric.num_designs_PH": "n_designs",
        "metric.num_matches_PH": "n_matches",
        "metric.top_iptm_PH": "top_iptm",
        "metric.top_plddt_PH": "top_plddt",
    }
    out = out.rename(columns=rename)
    if "git_sha" in out.columns:
        out["git_sha"] = out["git_sha"].astype(str).str[:8]
    if "wall_seconds" in out.columns:
        out["wall_hours"] = (out["wall_seconds"] / 3600.0).round(2)
    return out


# --------------------------------------------------------------------------
# MD-overnight bundles
# --------------------------------------------------------------------------
def list_md_overnight(bench_root: Path = BENCHMARKS_DIR) -> list[Path]:
    """Find every ``md_*_overnight/`` bundle directory under benchmarks/."""
    if not bench_root.is_dir():
        return []
    return sorted(p for p in bench_root.glob("md_*_overnight*")
                  if p.is_dir() and (p / "jobs").is_dir())


def _parse_delta_total(al_output: Path) -> tuple[float, float, float] | None:
    """Pull (mean, std, sem) from AL_output.dat's 'DELTA TOTAL' line.

    Robust to MMPBSA.py rerunning: takes the *last* match.
    """
    if not al_output.is_file():
        return None
    text = al_output.read_text()
    matches = list(_DELTA_TOTAL_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def _md_job_stage(job_dir: Path) -> tuple[str, str | None]:
    """Return (furthest_stage, status) for one MD job dir.

    Status is 'done' if mmgbsa AL_output.dat exists and has a DELTA TOTAL line,
    'running' if any intermediate stage exists but mmgbsa isn't done,
    'queued' if only setup files exist, 'failed' if a *.failed marker is set.
    """
    if (job_dir / "FAILED").exists():
        return ("setup", "failed")
    reached = "setup"
    for stage, fname in MD_STAGE_FILES:
        if (job_dir / fname).is_file():
            reached = stage
    if reached == "mmgbsa" and _parse_delta_total(job_dir / "mmgbsa" / "AL_out" / "AL_output.dat"):
        return (reached, "done")
    if reached == "setup":
        return (reached, "queued")
    return (reached, "running")


def _parse_target(job_name: str) -> str:
    """Split job-dir name into a target tag.

    Conventions seen on disk:
      * ``<target>__<match...>_design<k>``  (the 5-26 bundle)
      * ``match44_mdm2`` / ``niklas_run_038``  (the 5-20 bundle)
      * ``ref_<pdb>_<target>_<note>``  (reference jobs)
    """
    if "__" in job_name:
        return job_name.split("__", 1)[0]
    if job_name.startswith("ref_"):
        toks = job_name.split("_")
        return toks[2] if len(toks) > 2 else "ref"
    if job_name.startswith("niklas_run"):
        return "niklas_ref"
    # match44_mdm2 -> mdm2
    toks = job_name.split("_", 1)
    return toks[1] if len(toks) > 1 else job_name


def scan_md_overnight_jobs(bundle: Path) -> pd.DataFrame:
    """One row per job under ``<bundle>/jobs/``, with status + dG."""
    jobs_dir = bundle / "jobs"
    if not jobs_dir.is_dir():
        return pd.DataFrame()
    rows = []
    for j in sorted(p for p in jobs_dir.iterdir() if p.is_dir()):
        stage, status = _md_job_stage(j)
        dg = _parse_delta_total(j / "mmgbsa" / "AL_out" / "AL_output.dat")
        rows.append({
            "job": j.name,
            "target": _parse_target(j.name),
            "is_reference": j.name.startswith("ref_") or j.name.startswith("niklas_"),
            "stage": stage,
            "status": status,
            "dG_kcal_mol": dg[0] if dg else None,
            "SD":          dg[1] if dg else None,
            "SE":          dg[2] if dg else None,
            "trajectory_mp4": str(j / "trajectory.mp4")
                              if (j / "trajectory.mp4").is_file() else None,
            "free_run_nc":   str(j / "free_run" / "run.nc")
                              if (j / "free_run" / "run.nc").is_file() else None,
            "prmtop":        str(j / "system_wb.prmtop")
                              if (j / "system_wb.prmtop").is_file() else None,
            "job_dir": str(j),
        })
    return pd.DataFrame(rows)


@dataclass
class MDBundle:
    """A loaded ``md_*_overnight/`` bundle."""

    path: Path
    jobs: pd.DataFrame
    shortlist: pd.DataFrame = field(default_factory=pd.DataFrame)
    results_running: pd.DataFrame = field(default_factory=pd.DataFrame)
    monitor_log: str = ""
    verification_images: list[Path] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def is_live(self) -> bool:
        """Heuristic: any job not yet 'done' or 'failed'."""
        if self.jobs.empty:
            return False
        return self.jobs["status"].isin(["running", "queued"]).any()

    def joined(self) -> pd.DataFrame:
        """Left-join shortlist <- jobs by name match.

        Shortlists carry the design provenance (run_id, ipTM, topology,
        optimized_sequence); jobs carry the MD outcome (stage, dG, SE).
        The notebook's MD tab pivots on this table.
        """
        if self.jobs.empty:
            return self.shortlist.copy() if not self.shortlist.empty else pd.DataFrame()
        if self.shortlist.empty:
            return self.jobs.copy()
        # job names look like '<target>__<shortlist.name>_design<k>'; the
        # shortlist's 'name' column is '<short_name>_design<k>'. Join by suffix.
        sl = self.shortlist.copy()
        jb = self.jobs.copy()
        sl["_join_key"] = sl["name"].astype(str)
        jb["_join_key"] = jb["job"].astype(str).str.split("__").str[-1]
        out = sl.merge(jb, on="_join_key", how="outer", suffixes=("", "_md"))
        return out.drop(columns="_join_key")


def load_md_overnight(bundle: Path | str) -> MDBundle:
    """Load one overnight bundle into an MDBundle.

    Cheap (filesystem only). Safe to call on a live, growing bundle.
    """
    bundle = Path(bundle)
    if not bundle.is_dir():
        raise FileNotFoundError(f"no MD overnight bundle at {bundle}")

    jobs = scan_md_overnight_jobs(bundle)

    sl_path = bundle / "shortlist_master.csv"
    shortlist = pd.read_csv(sl_path) if sl_path.is_file() else pd.DataFrame()

    rr_path = bundle / "results_running.csv"
    results_running = pd.read_csv(rr_path) if rr_path.is_file() else pd.DataFrame()

    mlog = bundle / "monitor.log"
    monitor_log = mlog.read_text() if mlog.is_file() else ""

    images = sorted(bundle.glob("verification_*.png"))
    return MDBundle(
        path=bundle,
        jobs=jobs,
        shortlist=shortlist,
        results_running=results_running,
        monitor_log=monitor_log,
        verification_images=images,
    )


# --------------------------------------------------------------------------
# MM-GBSA per-residue decomposition
# --------------------------------------------------------------------------
_DECOMP_REL = Path("mmgbsa") / "AL_out" / "AL_decomp.csv"


def _parse_ranges_env(ranges_path: Path) -> dict[str, tuple[int, int]]:
    """Parse `ranges.env` written by the MD setup. Lines look like:
        A_RES=1-85
        L_RES=86-101
    Returns {'A': (1, 85), 'L': (86, 101)} (the keys are the prefix letter).
    """
    out: dict[str, tuple[int, int]] = {}
    if not ranges_path.is_file():
        return out
    for line in ranges_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key.endswith("_RES"):
            continue
        prefix = key[:-4]  # 'A_RES' -> 'A'
        if "-" not in val:
            continue
        try:
            lo, hi = (int(x) for x in val.split("-", 1))
        except ValueError:
            continue
        out[prefix] = (lo, hi)
    return out


def parse_mmgbsa_decomp(
    decomp_csv: Path,
    target_chain: str = "A",
    ligand_chain: str = "B",
    ranges_env: Path | None = None,
) -> dict[tuple[str, int], float]:
    """Parse MMPBSA.py per-residue decomposition CSV.

    AmberTools MMPBSA.py `AL_decomp.csv` format (idecomp=1, verified on disk):
      - First ~8 lines are a header / banner ending in two CSV header rows
        (`Residue,Location,Internal,,,van der Waals,,,Electrostatic,,,Polar
        Solvation,,,Non-Polar Solv.,,,TOTAL,,`).
      - Each data row has 20 comma-separated fields: `RESNAME N` (0),
        `R RESNAME N` (1), then six energy terms (Internal, vdW, Electrostatic,
        Polar Solv., Non-Polar Solv., TOTAL) with avg/std/SEM each. TOTAL_avg
        sits at index 17.
    The CSV omits chain IDs. We infer chain by residue index using `ranges.env`
    from the MD setup (`A_RES=lo-hi`, `L_RES=lo-hi`); fall back to the
    `target_chain` / `ligand_chain` arguments and a simple split heuristic if
    `ranges.env` is missing.

    Returns {(chain_id, residue_number): TOTAL_kcal_mol}.
    """
    decomp_csv = Path(decomp_csv)
    if not decomp_csv.is_file():
        return {}

    ranges = _parse_ranges_env(ranges_env) if ranges_env else {}
    # MD prefixes A_RES / L_RES -> target / ligand by convention.
    target_range = ranges.get("A")
    ligand_range = ranges.get("L")

    def _chain_and_resnum(resnum: int) -> tuple[str, int]:
        # MMPBSA.py numbers residues sequentially through the Amber prmtop
        # (target 1..N_t, then ligand N_t+1..N_t+N_l). Design PDBs renumber
        # each chain from 1, so we have to subtract the range offset on the
        # ligand side to match the PDB. Same logic on the target side keeps
        # the mapping correct even if the target range doesn't start at 1.
        if target_range and target_range[0] <= resnum <= target_range[1]:
            return target_chain, resnum - target_range[0] + 1
        if ligand_range and ligand_range[0] <= resnum <= ligand_range[1]:
            return ligand_chain, resnum - ligand_range[0] + 1
        # No ranges info: assume target precedes ligand. Caller passes the
        # split via target_range/ligand_range if known; otherwise everything
        # is dumped under target_chain at its raw number.
        return target_chain, resnum

    out: dict[tuple[str, int], float] = {}
    # Skip the banner; rows that parse cleanly are data rows.
    for raw in decomp_csv.read_text().splitlines():
        s = raw.strip()
        if not s or "," not in s:
            continue
        toks = raw.split(",")
        # Data rows start with `RESNAME N` and `R RESNAME N` as the first two
        # fields; TOTAL_avg is field index 17 (0-based) of 20.
        head = toks[0].strip().split()
        if len(head) < 2:
            continue
        resname = head[0]
        try:
            resnum = int(head[-1])
        except ValueError:
            continue
        if not resname.isalpha() or len(resname) > 4:
            continue
        if len(toks) < 20:
            continue
        try:
            total = float(toks[17])
        except (ValueError, IndexError):
            continue
        chain, pdb_resnum = _chain_and_resnum(resnum)
        out[(chain, pdb_resnum)] = total
    return out


def _md_job_dir_for_design(mdb: "MDBundle", design_name: str) -> Path | None:
    """Find the job dir whose name matches `<target>__<design_name>`.

    Falls back to suffix match if the bundle uses a different naming scheme.
    """
    if mdb.jobs.empty or "job" not in mdb.jobs.columns:
        return None
    jobs = mdb.jobs
    # Prefer exact suffix match.
    cand = jobs[jobs["job"].astype(str).str.endswith(f"__{design_name}")]
    if cand.empty:
        cand = jobs[jobs["job"].astype(str).str.contains(design_name, regex=False)]
    if cand.empty:
        return None
    return Path(str(cand["job_dir"].iloc[0]))


def mmgbsa_decomp_for_designs(
    mdb: "MDBundle",
    design_names: list[str],
    target_chain: str = "A",
    ligand_chain: str = "B",
) -> dict[str, dict[tuple[str, int], float]]:
    """Map each selected design name to its parsed decomp dict (if MD job exists).

    Returns {design_name: {(chain, resnum): dG_kcal_mol}}. Designs without a
    completed MD job (or without an `AL_decomp.csv` on disk) are omitted.
    """
    out: dict[str, dict[tuple[str, int], float]] = {}
    if mdb is None:
        return out
    for name in design_names:
        jd = _md_job_dir_for_design(mdb, name)
        if jd is None:
            continue
        decomp = jd / _DECOMP_REL
        if not decomp.is_file():
            continue
        ranges_env = jd / "ranges.env"
        parsed = parse_mmgbsa_decomp(
            decomp, target_chain=target_chain, ligand_chain=ligand_chain,
            ranges_env=ranges_env if ranges_env.is_file() else None,
        )
        if parsed:
            out[name] = parsed
    return out


# --------------------------------------------------------------------------
# Convenience: link MLflow runs back to triage RunData
# --------------------------------------------------------------------------
def mlflow_to_triage_run_id(row: pd.Series) -> str | None:
    """Get the ``data/runs/<run_id>/`` that an MLflow row corresponds to.

    Snakemake's mlflow_start logs ``run_name = <run_id>``, so this is just
    the run_name field. Returns None for the legacy notebook path.
    """
    name = row.get("run_name")
    if not name or "_v" not in str(name):
        return None
    return str(name)
