"""cpm2.trajectory_metrics -- per-job MD trajectory quality diagnostics.

Reads a finished MD/MMGBSA benchmark job (``benchmarks/md_*/jobs/<name>/``)
and computes ten quality metrics on the production trajectory + MMGBSA
output, returning a single flat dict suitable for a CSV row, and writing
a small multi-panel PNG dashboard for visual audit.

Designed to be invoked from ``scripts/compute_trajectory_metrics.py``.
Requires pytraj (AmberTools). The cpm2 conda env has a broken pytraj
install; use the ``biophys`` micromamba env at
``/path/to/your-mdanalysis-env/bin/python``.

Public entry point: :func:`compute_job_metrics`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytraj as pt

# ---------------------------------------------------------------------------
# Input discovery + ranges
# ---------------------------------------------------------------------------


@dataclass
class JobPaths:
    name: str
    root: Path
    prmtop: Path
    traj: Path
    mmgbsa_dir: Path
    al_prmtop: Path
    al_traj: Path
    decomp_csv: Path
    al_output_dat: Path
    complex_mdout: Path
    receptor_mdout: Path
    ligand_mdout: Path
    a_res: tuple[int, int]
    l_res: tuple[int, int]
    cyclic: bool


# Refs whose peptide is linear (per task spec).
LINEAR_REF_NAMES = {
    "ref_1ycr_mdm2_p53",
    "ref_1g5j_bclxl_bad",
    "ref_5m36_14_3_3_cdc25c_phos",
    "ref_2x2d_cypA_capsid",
}


def _parse_ranges_env(path: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    a_range = lig_range = None
    for line in path.read_text().splitlines():
        m = re.match(r"\s*([AL])_RES=(\d+)(?:-(\d+))?", line)
        if m:
            tag = m.group(1)
            lo = int(m.group(2))
            hi = int(m.group(3)) if m.group(3) else lo
            if tag == "A":
                a_range = (lo, hi)
            else:
                lig_range = (lo, hi)
    if a_range is None or lig_range is None:
        raise ValueError(f"could not parse ranges.env at {path}")
    return a_range, lig_range


def discover_job(job_root: Path) -> Optional[JobPaths]:
    """Return a :class:`JobPaths` if the job has finished, else None."""
    name = job_root.name
    traj = job_root / "free_run" / "run.nc"
    al_dat = job_root / "mmgbsa" / "AL_out" / "AL_output.dat"
    if not (traj.exists() and al_dat.exists()):
        return None
    a_res, l_res = _parse_ranges_env(job_root / "ranges.env")
    return JobPaths(
        name=name,
        root=job_root,
        prmtop=job_root / "system_wb.prmtop",
        traj=traj,
        mmgbsa_dir=job_root / "mmgbsa",
        al_prmtop=job_root / "mmgbsa" / "AL.prmtop",
        al_traj=job_root / "mmgbsa" / "AL.nc",
        decomp_csv=job_root / "mmgbsa" / "AL_out" / "AL_decomp.csv",
        al_output_dat=al_dat,
        complex_mdout=job_root / "mmgbsa" / "AL_out" / "_MMPBSA_complex_gb.mdout.0",
        receptor_mdout=job_root / "mmgbsa" / "AL_out" / "_MMPBSA_receptor_gb.mdout.0",
        ligand_mdout=job_root / "mmgbsa" / "AL_out" / "_MMPBSA_ligand_gb.mdout.0",
        a_res=a_res,
        l_res=l_res,
        cyclic=name not in LINEAR_REF_NAMES,
    )


# ---------------------------------------------------------------------------
# Per-frame MMGBSA energy parser
# ---------------------------------------------------------------------------

# After every "FINAL RESULTS" header sander writes:
#   BOND    = ...  ANGLE  = ...  DIHED   = ...
#   VDWAALS = ...  EEL    = ...  EGB     = ...
#   ESURF   = ...
# We sum these to a per-frame total internal+solv energy.
_FINAL_RE = re.compile(r"FINAL RESULTS")
_BOND_RE = re.compile(r"BOND\s*=\s*(-?\d+\.\d+)\s+ANGLE\s*=\s*(-?\d+\.\d+)\s+DIHED\s*=\s*(-?\d+\.\d+)")
_VDW_RE = re.compile(r"VDWAALS\s*=\s*(-?\d+\.\d+)\s+EEL\s*=\s*(-?\d+\.\d+)\s+EGB\s*=\s*(-?\d+\.\d+)")
_ESURF_RE = re.compile(r"ESURF\s*=\s*(-?\d+\.\d+)")


def parse_perframe_total(mdout: Path) -> np.ndarray:
    """Parse per-frame total (gas + GB + ESURF) energies from a MMPBSA sander
    mdout. The first FINAL RESULTS block is the minimization end-state; we
    take the *first* energy block after each FINAL RESULTS header.
    """
    txt = mdout.read_text()
    # Split on FINAL RESULTS markers; index 0 is preamble, 1+ are per-frame.
    blocks = _FINAL_RE.split(txt)[1:]
    totals = []
    for blk in blocks:
        m1 = _BOND_RE.search(blk)
        m2 = _VDW_RE.search(blk)
        m3 = _ESURF_RE.search(blk)
        if not (m1 and m2 and m3):
            continue
        bond, ang, dih = map(float, m1.groups())
        vdw, eel, egb = map(float, m2.groups())
        esurf = float(m3.group(1))
        totals.append(bond + ang + dih + vdw + eel + egb + esurf)
    return np.array(totals, dtype=float)


def compute_running_dg(job: JobPaths) -> tuple[np.ndarray, np.ndarray]:
    """Return (per-frame delta_G_total, cumulative running mean)."""
    c = parse_perframe_total(job.complex_mdout)
    r = parse_perframe_total(job.receptor_mdout)
    lig = parse_perframe_total(job.ligand_mdout)
    n = min(len(c), len(r), len(lig))
    if n == 0:
        return np.array([]), np.array([])
    delta = c[:n] - r[:n] - lig[:n]
    cumavg = np.cumsum(delta) / np.arange(1, n + 1)
    return delta, cumavg


# ---------------------------------------------------------------------------
# Decomposition top-N
# ---------------------------------------------------------------------------


def parse_decomp_top5(job: JobPaths) -> tuple[str, str]:
    """Read AL_decomp.csv and return (top5_receptor, top5_ligand) as
    comma-joined "Loc:Res123=-7.2" strings ranked by abs(TOTAL avg).
    """
    # File has two header rows; the second is the unit row. TOTAL avg is col 15.
    df = pd.read_csv(job.decomp_csv, skiprows=8, header=None)
    # Columns: 0=Residue, 1=Location, then triples for Internal, vdW, EEL,
    # PolarSolv, NonPolar, TOTAL (each: Avg, SD, SE). TOTAL avg is column 16.
    if df.shape[1] < 17:
        return "", ""
    df = df.rename(columns={0: "residue", 1: "loc", 16: "total"})
    df["total"] = pd.to_numeric(df["total"], errors="coerce")
    df = df.dropna(subset=["total"])
    df["side"] = df["loc"].astype(str).str.strip().str[0]  # 'R' or 'L'
    out = []
    for side in ("R", "L"):
        sub = df[df["side"] == side].copy()
        sub["abs"] = sub["total"].abs()
        top = sub.sort_values("abs", ascending=False).head(5)
        parts = [
            f"{side}:{str(r['residue']).strip().replace(' ', '')}={r['total']:.2f}"
            for _, r in top.iterrows()
        ]
        out.append(",".join(parts))
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Trajectory metrics (pytraj)
# ---------------------------------------------------------------------------


def _ca_mask(lo: int, hi: int) -> str:
    return f":{lo}-{hi}@CA"


def _heavy_mask(lo: int, hi: int) -> str:
    return f":{lo}-{hi}&!@H="


def _select_indices(top, mask: str) -> np.ndarray:
    return np.array(top.select(mask), dtype=int)


def _per_frame_contacts_and_hbonds(traj, job: JobPaths):
    """Compute per-frame interface heavy-atom contact counts (< 4.0 A) and
    interface H-bonds (donor-H...acceptor: d_DA < 3.5 A, angle > 120 deg)
    between the receptor (residues a_lo..a_hi) and the peptide.

    Implementation: pull atom subsets once via topology, then iterate frames
    in-place reading coordinates. Uses scipy.spatial.cKDTree when available
    for the contact search; otherwise falls back to brute-force numpy.
    """
    top = traj.top
    a_lo, a_hi = job.a_res
    l_lo, l_hi = job.l_res

    # Heavy atom indices (no H) for receptor / peptide.
    rec_heavy_idx = _select_indices(top, f":{a_lo}-{a_hi}&!@H=")
    pep_heavy_idx = _select_indices(top, f":{l_lo}-{l_hi}&!@H=")

    # For H-bonds we need donor-heavy(N/O/S)-with-H + acceptor-heavy(N/O/S)
    # on each side. Build donor (heavy, attached-H) pairs and acceptor lists.
    def _donor_acceptor(idx):
        donors = []  # (heavy_idx, h_idx)
        acceptors = []
        for ai in idx:
            atom = top.atom(int(ai))
            elem = atom.name[0]
            if elem not in ("N", "O", "S"):
                continue
            acceptors.append(int(ai))
            # find bonded H atoms via topology.bonds_indices
        # find bonded H atoms by iterating bonds list
        idx_set = set(int(x) for x in idx)
        for b in top.bond_indices:
            i, j = int(b[0]), int(b[1])
            if i in idx_set and top.atom(j).name.startswith("H"):
                donors.append((i, j))
            elif j in idx_set and top.atom(i).name.startswith("H"):
                donors.append((j, i))
        return donors, np.array(acceptors, dtype=int)

    rec_donors, rec_acceptors = _donor_acceptor(rec_heavy_idx)
    pep_donors, pep_acceptors = _donor_acceptor(pep_heavy_idx)

    try:
        from scipy.spatial import cKDTree  # type: ignore

        have_kdtree = True
    except Exception:
        have_kdtree = False

    n_frames = traj.n_frames
    contacts = np.zeros(n_frames, dtype=float)
    hbonds = np.zeros(n_frames, dtype=float)

    cutoff_c = 4.0
    cutoff_h = 3.5
    cos_angle_cut = np.cos(np.deg2rad(120.0))  # for angle > 120, cos < this

    for fi, frame in enumerate(traj):
        coords = np.asarray(frame.xyz)  # (n_atoms, 3)
        rec_xyz = coords[rec_heavy_idx]
        pep_xyz = coords[pep_heavy_idx]

        # contacts
        if have_kdtree:
            tree = cKDTree(rec_xyz)
            pairs = tree.query_ball_point(pep_xyz, r=cutoff_c)
            n_c = sum(len(p) for p in pairs)
        else:
            d2 = np.sum((rec_xyz[None, :, :] - pep_xyz[:, None, :]) ** 2, axis=-1)
            n_c = int(np.sum(d2 < cutoff_c * cutoff_c))
        contacts[fi] = n_c

        # h-bonds: receptor-donor -> peptide-acceptor, and peptide-donor -> receptor-acceptor
        n_h = 0
        for donors, acceptors in (
            (rec_donors, pep_acceptors),
            (pep_donors, rec_acceptors),
        ):
            if len(donors) == 0 or len(acceptors) == 0:
                continue
            a_xyz = coords[acceptors]
            for hi_idx, h_idx in donors:
                d_xyz = coords[hi_idx]
                # distance D-A
                diff = a_xyz - d_xyz
                d2 = np.einsum("ij,ij->i", diff, diff)
                mask = d2 < cutoff_h * cutoff_h
                if not np.any(mask):
                    continue
                # angle D-H...A: vector H->D, H->A; angle at H. We use:
                # cos(theta) = ((D-H) . (A-H)) / (|D-H||A-H|).
                h_xyz = coords[h_idx]
                v1 = d_xyz - h_xyz  # H->D
                v2 = a_xyz[mask] - h_xyz  # H->A
                num = v2 @ v1
                denom = np.linalg.norm(v1) * np.linalg.norm(v2, axis=1) + 1e-12
                cosang = num / denom
                # angle > 120 deg => cos < cos(120) = -0.5
                n_h += int(np.sum(cosang < cos_angle_cut))
        hbonds[fi] = n_h

    return contacts, hbonds


def compute_traj_metrics(job: JobPaths) -> dict:
    """Compute all pytraj-based per-job metrics. Returns a flat dict of
    scalars + numpy arrays (arrays keyed with leading underscore for plots).
    """
    traj = pt.iterload(str(job.traj), str(job.prmtop))
    n_frames = traj.n_frames
    # Time axis: MD wrote at 20 ps interval -> use frame index but label ns
    # assuming 10 ns / n_frames. Fall back to frame index if metadata absent.
    time_ns = np.linspace(0.0, 10.0, n_frames)

    a_lo, a_hi = job.a_res
    l_lo, l_hi = job.l_res
    recv_ca = _ca_mask(a_lo, a_hi)
    pep_ca = _ca_mask(l_lo, l_hi)

    out: dict = {}

    # ---- 1. peptide CA drift RMSD (fit on receptor CA, ref = frame 0)
    pep_rmsd = pt.rmsd(traj, mask=pep_ca, ref=0, ref_mask=recv_ca, mask_ref=recv_ca, nofit=True) \
        if False else None
    # Simpler & correct approach: pre-fit on receptor, then compute RMSD on
    # peptide without further fitting.
    traj_aligned = traj[:]  # in-memory copy
    pt.align(traj_aligned, mask=recv_ca, ref=0)
    pep_rmsd = pt.rmsd_nofit(traj_aligned, mask=pep_ca, ref=0)
    out["pep_rmsd_mean"] = float(np.mean(pep_rmsd))
    out["pep_rmsd_max"] = float(np.max(pep_rmsd))
    half = n_frames // 2
    if n_frames - half >= 5:
        slope, _ = np.polyfit(time_ns[half:], pep_rmsd[half:], 1)
        out["pep_rmsd_slope_2nd_half"] = float(slope)
    else:
        out["pep_rmsd_slope_2nd_half"] = float("nan")
    out["_pep_rmsd"] = pep_rmsd
    out["_time_ns"] = time_ns

    # ---- 2. peptide RMSF (per-residue CA)
    # pt.rmsf needs already-aligned trajectory; aligned above.
    pep_rmsf = pt.rmsf(traj_aligned, mask=pep_ca, options="byres")
    # pt.rmsf returns 2D array [res_idx, rmsf]
    rmsf_vals = np.array(pep_rmsf)[:, 1]
    out["pep_rmsf_mean"] = float(np.mean(rmsf_vals))
    out["_pep_rmsf"] = rmsf_vals
    out["_pep_resids"] = np.arange(l_lo, l_hi + 1)

    # ---- 3. receptor CA RMSD (self-aligned)
    traj_self = traj[:]
    pt.align(traj_self, mask=recv_ca, ref=0)
    rec_rmsd = pt.rmsd_nofit(traj_self, mask=recv_ca, ref=0)
    out["rec_rmsd_mean"] = float(np.mean(rec_rmsd))
    out["rec_rmsd_max"] = float(np.max(rec_rmsd))
    out["_rec_rmsd"] = rec_rmsd

    # ---- 4. CoM distance peptide-CA centroid vs receptor-CA centroid
    com_dist = pt.distance(traj, f"{recv_ca} {pep_ca}")
    out["com_dist_mean"] = float(np.mean(com_dist))
    out["com_dist_max"] = float(np.max(com_dist))
    out["com_dist_drift"] = float(np.max(np.abs(com_dist - com_dist[0])))
    out["_com_dist"] = com_dist

    # ---- 5/6. Heavy-atom contacts + interface H-bonds via per-frame numpy
    contacts, hbonds = _per_frame_contacts_and_hbonds(traj, job)
    out["contacts_mean"] = float(np.nanmean(contacts))
    out["contacts_min"] = float(np.nanmin(contacts))
    out["_contacts"] = contacts
    out["hbond_mean"] = float(np.nanmean(hbonds))
    out["_hbond"] = hbonds

    # ---- 9. Buried surface area: SASA(complex_only) on stripped copies.
    # Using pt.molsurf (more reliable than LCPO 'surf' for absolute SASA),
    # we strip non-solute atoms from in-memory trajectory copies so that
    # rec-only and lig-only SASA correctly omit the partner.
    try:
        rec_traj = traj[:].copy()
        rec_traj.strip(f"!:{a_lo}-{a_hi}")
        lig_traj = traj[:].copy()
        lig_traj.strip(f"!:{l_lo}-{l_hi}")
        com_traj = traj[:].copy()
        com_traj.strip(f"!:{a_lo}-{l_hi}")
        sasa_r = np.array(pt.molsurf(rec_traj))
        sasa_l = np.array(pt.molsurf(lig_traj))
        sasa_c = np.array(pt.molsurf(com_traj))
        bsa = sasa_r + sasa_l - sasa_c
    except Exception:
        bsa = np.full(n_frames, np.nan)
    out["bsa_mean"] = float(np.nanmean(bsa))
    out["_bsa"] = bsa

    # ---- 8. cyclic omega torsion stability
    if job.cyclic:
        # omega = Ca(i)-C(i)-N(i+1)-Ca(i+1). For a cyclic peptide we include
        # the head->tail bond linking residue l_hi -> l_lo.
        omega_stds = []
        omega_labels = []
        pep_indices = list(range(l_lo, l_hi + 1))
        # all i -> i+1 plus tail->head
        pairs = [(pep_indices[k], pep_indices[k + 1]) for k in range(len(pep_indices) - 1)]
        pairs.append((pep_indices[-1], pep_indices[0]))
        for i, j in pairs:
            spec = f":{i}@CA :{i}@C :{j}@N :{j}@CA"
            try:
                tor = pt.dihedral(traj, spec)
                tor = np.array(tor)
                # circular stddev in degrees
                rad = np.deg2rad(tor)
                R = np.sqrt(np.mean(np.cos(rad)) ** 2 + np.mean(np.sin(rad)) ** 2)
                circ_std_deg = np.rad2deg(np.sqrt(-2.0 * np.log(max(R, 1e-12))))
                omega_stds.append(float(circ_std_deg))
                omega_labels.append(f"{i}-{j}")
            except Exception:
                continue
        if omega_stds:
            out["omega_std_mean"] = float(np.mean(omega_stds))
            out["omega_std_max"] = float(np.max(omega_stds))
            out["_omega_std"] = np.array(omega_stds)
            out["_omega_labels"] = omega_labels
        else:
            out["omega_std_mean"] = float("nan")
            out["omega_std_max"] = float("nan")
            out["_omega_std"] = np.array([])
            out["_omega_labels"] = []
    else:
        out["omega_std_mean"] = float("nan")
        out["omega_std_max"] = float("nan")
        out["_omega_std"] = np.array([])
        out["_omega_labels"] = []

    return out


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def compute_job_metrics(job_root: Path, png_out: Optional[Path] = None) -> Optional[dict]:
    job = discover_job(job_root)
    if job is None:
        return None
    metrics = compute_traj_metrics(job)

    # ---- 7. running mean ΔG_GB convergence
    perframe, cumavg = compute_running_dg(job)
    if cumavg.size:
        n = cumavg.size
        half_idx = n // 2
        metrics["dg_running_half"] = float(cumavg[half_idx]) if half_idx > 0 else float("nan")
        metrics["dg_running_final"] = float(cumavg[-1])
        metrics["dg_convergence_delta"] = abs(metrics["dg_running_final"] - metrics["dg_running_half"])
    else:
        metrics["dg_running_half"] = float("nan")
        metrics["dg_running_final"] = float("nan")
        metrics["dg_convergence_delta"] = float("nan")
    metrics["_dg_perframe"] = perframe
    metrics["_dg_cumavg"] = cumavg

    # ---- 10. top-5 MMGBSA decomp
    try:
        rec_top, lig_top = parse_decomp_top5(job)
    except Exception as e:
        rec_top, lig_top = "", f"ERR: {e}"
    metrics["decomp_top5_receptor"] = rec_top
    metrics["decomp_top5_ligand"] = lig_top

    metrics["job_name"] = job.name
    metrics["cyclic"] = job.cyclic
    metrics["n_frames"] = job.a_res[1]  # placeholder; overwritten below
    metrics["n_frames"] = int(metrics["_pep_rmsd"].size)

    if png_out is not None:
        png_out.parent.mkdir(parents=True, exist_ok=True)
        render_dashboard(metrics, job, png_out)

    # strip array fields from the public dict for CSV
    public = {k: v for k, v in metrics.items() if not k.startswith("_")}
    return public


# ---------------------------------------------------------------------------
# Dashboard plot
# ---------------------------------------------------------------------------


def render_dashboard(m: dict, job: JobPaths, out_png: Path) -> None:
    time_ns = m["_time_ns"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.suptitle(f"{job.name}  (cyclic={job.cyclic})", fontsize=12)

    ax = axes[0, 0]
    ax.plot(time_ns, m["_pep_rmsd"], lw=1)
    ax.set_title(f"Peptide CA RMSD (mean {m['pep_rmsd_mean']:.2f} A, slope {m['pep_rmsd_slope_2nd_half']:.3f} A/ns)")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("RMSD (A)")

    ax = axes[0, 1]
    ax.bar(m["_pep_resids"], m["_pep_rmsf"])
    ax.set_title(f"Peptide RMSF (mean {m['pep_rmsf_mean']:.2f} A)")
    ax.set_xlabel("residue")
    ax.set_ylabel("RMSF (A)")

    ax = axes[0, 2]
    ax.plot(time_ns, m["_rec_rmsd"], lw=1, color="C3")
    ax.set_title(f"Receptor CA RMSD (mean {m['rec_rmsd_mean']:.2f} A)")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("RMSD (A)")

    ax = axes[1, 0]
    ax.plot(time_ns, m["_com_dist"], lw=1, color="C2")
    ax.set_title(f"CoM distance (drift {m['com_dist_drift']:.2f} A)")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("distance (A)")

    ax = axes[1, 1]
    ax.plot(time_ns, m["_contacts"], lw=1, color="C4")
    ax.set_title(f"Heavy contacts <4A (mean {m['contacts_mean']:.0f}, min {m['contacts_min']:.0f})")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("count")

    ax = axes[1, 2]
    ax.plot(time_ns, m["_hbond"], lw=1, color="C5")
    ax.set_title(f"Interface H-bonds (mean {m['hbond_mean']:.1f})")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("count")

    ax = axes[2, 0]
    if m["_dg_cumavg"].size:
        x = np.arange(1, m["_dg_cumavg"].size + 1)
        ax.plot(x, m["_dg_cumavg"], lw=1.5, color="C0")
        ax.axhline(m["dg_running_final"], ls="--", color="grey", lw=0.7)
        ax.set_title(f"Running <dG_GB> (final {m['dg_running_final']:.1f}, dconv {m['dg_convergence_delta']:.1f})")
        ax.set_xlabel("MMGBSA frame index")
        ax.set_ylabel("dG (kcal/mol)")
    else:
        ax.set_title("Running dG: no data")

    ax = axes[2, 1]
    bsa = m["_bsa"]
    if np.isfinite(bsa).any():
        ax.plot(time_ns, bsa, lw=1, color="C6")
        ax.set_title(f"BSA (mean {m['bsa_mean']:.0f} A^2)")
    else:
        ax.set_title("BSA: unavailable")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("BSA (A^2)")

    ax = axes[2, 2]
    if m["_omega_std"].size:
        ax.bar(range(len(m["_omega_std"])), m["_omega_std"])
        ax.set_xticks(range(len(m["_omega_labels"])))
        ax.set_xticklabels(m["_omega_labels"], rotation=70, fontsize=7)
        ax.set_title(f"Omega circ-std (max {m['omega_std_max']:.1f} deg)")
        ax.set_ylabel("circ. std (deg)")
    else:
        ax.set_title("Omega: linear peptide / N/A")
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
