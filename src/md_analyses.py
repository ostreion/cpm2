"""cpm2.md_analyses -- per-job MD trajectory signal extraction (mdtraj-based).

Computes three signal sets per finished MD benchmark job:

  - per-residue signals (RMSF, contact frequency) -- ``compute_residue_signals``
  - per-frame timeline traces (peptide RMSD, CoM distance) -- ``compute_traces``
  - orchestrated write to parquet + meta.json -- ``compute_and_write``

Requires mdtraj (already in the ``cpm2`` conda env). All mdtraj imports are
lazy (inside function bodies) to keep module-level import cost near zero.

Chain assignment follows the ``amber_to_bytes`` convention in
``molstar_marimo.amber``: after stripping solvent/ions the largest connected
molecule is chain A (target), the second-largest is chain B (peptide). In
practice the ranges.env A_RES / L_RES assignment encodes this explicitly, so
we use those ranges directly rather than running connectivity analysis.

Residue indexing contract
-------------------------
- ``ranges.env`` gives 1-indexed prmtop residue numbers.
- After stripping solvent/ions the stripped topology has exactly N_protein
  residues; mdtraj's 0-indexed ``residue.index`` maps as
  ``prmtop_1indexed = residue.index + 1``.
- The ``resnum`` column in the per-residue parquet is the 1-indexed prmtop
  number so the caller can cross-reference back to the design PDB.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers: ranges.env parsing
# ---------------------------------------------------------------------------

_STRIP_SELECTION = (
    "not (resname 'HOH' or resname 'WAT' or resname 'Na+' or resname 'Cl-')"
)


def _parse_ranges_env(path: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    """Parse A_RES and L_RES from a shell-style ranges.env file."""
    a_range = l_range = None
    for line in path.read_text().splitlines():
        m = re.match(r"\s*([AL])_RES=(\d+)(?:-(\d+))?", line)
        if m:
            tag = m.group(1)
            lo = int(m.group(2))
            hi = int(m.group(3)) if m.group(3) else lo
            if tag == "A":
                a_range = (lo, hi)
            else:
                l_range = (lo, hi)
    if a_range is None or l_range is None:
        raise ValueError(f"could not parse A_RES/L_RES from {path}")
    return a_range, l_range


def _load_stripped(
    nc_path: str,
    prmtop_path: str,
    stride: int = 1,
):
    """Load and strip an Amber .nc+.prmtop pair via mdtraj.

    Returns the stripped mdtraj.Trajectory.
    """
    import mdtraj  # noqa: PLC0415

    traj = mdtraj.load(nc_path, top=prmtop_path, stride=stride)
    indices = traj.topology.select(_STRIP_SELECTION)
    if len(indices) == 0:
        raise RuntimeError(
            f"Stripping solvent/ions left zero atoms in {nc_path}. "
            "Check that the trajectory is a standard Amber solvated system."
        )
    return traj.atom_slice(indices)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_residue_signals(
    nc_path: Union[Path, str],
    prmtop_path: Union[Path, str],
    a_range: tuple[int, int],
    l_range: tuple[int, int],
    stride: int = 1,
    contact_cutoff_A: float = 4.0,
) -> pd.DataFrame:
    """Per-residue signals across the trajectory.

    Returns a DataFrame with columns:
      chain       -- "A" (target) or "B" (peptide)
      resnum      -- prmtop residue index (1-indexed)
      resname     -- 3-letter residue name
      rmsf_A      -- atomic RMSF (Angstrom): CA RMSF if CA exists, else
                     mean over all heavy atoms in the residue
      contact_freq -- fraction of frames where any heavy atom of this residue
                      is within ``contact_cutoff_A`` of any heavy atom of the
                      opposite chain (0.0 = never, 1.0 = always)
    """
    import mdtraj  # noqa: PLC0415

    traj = _load_stripped(str(nc_path), str(prmtop_path), stride=stride)
    top = traj.topology
    n_frames = traj.n_frames

    a_lo, a_hi = a_range
    l_lo, l_hi = l_range

    # --- Superpose on target CA (chain A) before computing RMSF ---
    # mdtraj residues are 0-indexed; prmtop residues in ranges.env are 1-indexed.
    a_ca_indices = top.select(
        f"(residue >= {a_lo - 1} and residue <= {a_hi - 1}) and name CA"
    )
    if len(a_ca_indices) == 0:
        raise RuntimeError(
            f"No CA atoms found in receptor range {a_lo}-{a_hi}. "
            "Verify A_RES values against the stripped topology."
        )
    traj.superpose(traj, 0, atom_indices=a_ca_indices)

    # --- RMSF: per-residue ---
    # mdtraj.rmsf computes RMSF per atom; we average over heavy atoms in the
    # residue (or use just CA when available) for the scalar summary.
    # Use all frames as the reference average (average=True is default).
    # We compute RMSF for every protein atom in the two ranges then bucket by
    # residue.
    all_range_res = list(range(a_lo - 1, a_hi)) + list(range(l_lo - 1, l_hi))
    # mdtraj residue indices are 0-indexed == prmtop-1

    heavy_sel = " or ".join(
        f"(residue == {r} and element != H)" for r in all_range_res
    )
    heavy_indices = top.select(heavy_sel)
    rmsf_vals = mdtraj.rmsf(traj, None, atom_indices=heavy_indices)  # per-atom RMSF

    # Build lookup: atom global index -> rmsf
    rmsf_by_atom = dict(zip(heavy_indices, rmsf_vals))

    # --- Contact frequency: precompute heavy-atom indices per chain ---
    a_heavy = top.select(
        f"(residue >= {a_lo - 1} and residue <= {a_hi - 1}) and element != H"
    )
    l_heavy = top.select(
        f"(residue >= {l_lo - 1} and residue <= {l_hi - 1}) and element != H"
    )

    # Contacts: for each frame find pairs within cutoff_A using mdtraj.compute_neighbors
    # Compute contact presence per residue across frames.
    # Strategy: use mdtraj.compute_contacts with custom atom-pair lists is slow;
    # instead compute the N x M distance matrix per-frame via broadcasting.
    # With ~100-300 atoms per chain and 200 frames this is fast enough.
    cutoff_nm = contact_cutoff_A / 10.0  # mdtraj uses nm

    # Per-residue contact frequency arrays (0-indexed residue within range)
    a_n_res = a_hi - a_lo + 1
    l_n_res = l_hi - l_lo + 1
    a_contact_count = np.zeros(a_n_res, dtype=np.int32)
    l_contact_count = np.zeros(l_n_res, dtype=np.int32)

    # Precompute residue membership for heavy atoms
    a_atom_res = np.array(
        [top.atom(int(i)).residue.index - (a_lo - 1) for i in a_heavy], dtype=np.int32
    )
    l_atom_res = np.array(
        [top.atom(int(i)).residue.index - (l_lo - 1) for i in l_heavy], dtype=np.int32
    )

    coords = traj.xyz  # (n_frames, n_atoms, 3) in nm
    a_coords = coords[:, a_heavy, :]  # (F, nA, 3)
    l_coords = coords[:, l_heavy, :]  # (F, nL, 3)

    # Vectorised per-frame distance matrix: (F, nA, nL)
    # Split into chunks to avoid huge memory if trajectories are long
    chunk_size = 50
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        ac = a_coords[start:end]  # (C, nA, 3)
        lc = l_coords[start:end]  # (C, nL, 3)
        # pairwise distances: (C, nA, nL)
        diff = ac[:, :, np.newaxis, :] - lc[:, np.newaxis, :, :]
        dist2 = np.einsum("cfij,cfij->cfi", diff, diff)  # wrong dims -- fix below
        # diff shape: (C, nA, nL, 3); einsum over last axis
        dist2 = np.sum(diff ** 2, axis=-1)  # (C, nA, nL)
        in_contact = dist2 < cutoff_nm ** 2  # (C, nA, nL)

        # Any contact per receptor atom (across all peptide atoms)
        a_any = in_contact.any(axis=2)  # (C, nA)
        l_any = in_contact.any(axis=1)  # (C, nL)

        # Accumulate per-residue: did *any* atom of this residue make contact?
        for ri in range(a_n_res):
            mask = a_atom_res == ri
            if mask.any():
                a_contact_count[ri] += int(a_any[:, mask].any(axis=1).sum())
        for ri in range(l_n_res):
            mask = l_atom_res == ri
            if mask.any():
                l_contact_count[ri] += int(l_any[:, mask].any(axis=1).sum())

    a_contact_freq = a_contact_count / n_frames
    l_contact_freq = l_contact_count / n_frames

    # --- Assemble per-residue DataFrame ---
    rows = []
    for ri in range(a_n_res):
        res_0idx = a_lo - 1 + ri
        res = list(top.residues)[res_0idx]
        # CA RMSF if available, else mean over heavy atoms
        ca_sel = top.select(f"residue == {res_0idx} and name CA")
        if len(ca_sel) > 0 and ca_sel[0] in rmsf_by_atom:
            rmsf = float(rmsf_by_atom[ca_sel[0]]) * 10.0  # nm -> Angstrom
        else:
            heavy_in_res = top.select(
                f"residue == {res_0idx} and element != H"
            )
            vals = [rmsf_by_atom[i] for i in heavy_in_res if i in rmsf_by_atom]
            rmsf = float(np.mean(vals)) * 10.0 if vals else np.nan
        rows.append(
            dict(
                chain="A",
                resnum=res_0idx + 1,
                resname=res.name,
                rmsf_A=rmsf,
                contact_freq=float(a_contact_freq[ri]),
            )
        )

    for ri in range(l_n_res):
        res_0idx = l_lo - 1 + ri
        res = list(top.residues)[res_0idx]
        ca_sel = top.select(f"residue == {res_0idx} and name CA")
        if len(ca_sel) > 0 and ca_sel[0] in rmsf_by_atom:
            rmsf = float(rmsf_by_atom[ca_sel[0]]) * 10.0
        else:
            heavy_in_res = top.select(
                f"residue == {res_0idx} and element != H"
            )
            vals = [rmsf_by_atom[i] for i in heavy_in_res if i in rmsf_by_atom]
            rmsf = float(np.mean(vals)) * 10.0 if vals else np.nan
        rows.append(
            dict(
                chain="B",
                resnum=res_0idx + 1,
                resname=res.name,
                rmsf_A=rmsf,
                contact_freq=float(l_contact_freq[ri]),
            )
        )

    return pd.DataFrame(rows)


def compute_traces(
    nc_path: Union[Path, str],
    prmtop_path: Union[Path, str],
    a_range: tuple[int, int],
    l_range: tuple[int, int],
    stride: int = 1,
    total_ns: float = 10.0,
) -> pd.DataFrame:
    """Per-frame timeline traces.

    Returns DataFrame with columns:
      frame              -- 0-indexed
      time_ns            -- linspace(0, total_ns, n_frames)
      peptide_rmsd_A     -- peptide CA RMSD vs frame 0, after receptor CA
                            superposition (isolates peptide drift); Angstrom
      com_distance_A     -- center-of-mass distance peptide<->target using
                            all heavy atoms; Angstrom
    """

    traj = _load_stripped(str(nc_path), str(prmtop_path), stride=stride)
    top = traj.topology
    n_frames = traj.n_frames

    a_lo, a_hi = a_range
    l_lo, l_hi = l_range

    # Superpose on receptor CA to remove global tumbling
    a_ca = top.select(
        f"(residue >= {a_lo - 1} and residue <= {a_hi - 1}) and name CA"
    )
    if len(a_ca) == 0:
        raise RuntimeError(f"No CA atoms in receptor range {a_lo}-{a_hi}")
    traj.superpose(traj, 0, atom_indices=a_ca)

    # Peptide CA RMSD vs frame 0 (receptor already superposed)
    l_ca = top.select(
        f"(residue >= {l_lo - 1} and residue <= {l_hi - 1}) and name CA"
    )
    if len(l_ca) == 0:
        # Fallback to all heavy atoms if no CA (shouldn't happen for peptide)
        l_ca = top.select(
            f"(residue >= {l_lo - 1} and residue <= {l_hi - 1}) and element != H"
        )
    ref_coords = traj.xyz[0, l_ca, :]  # (nCA, 3) nm
    pep_rmsd_nm = np.sqrt(
        np.mean(np.sum((traj.xyz[:, l_ca, :] - ref_coords[np.newaxis]) ** 2, axis=-1), axis=-1)
    )
    pep_rmsd_A = pep_rmsd_nm * 10.0

    # CoM distance: mean position of all heavy atoms per chain
    a_heavy = top.select(
        f"(residue >= {a_lo - 1} and residue <= {a_hi - 1}) and element != H"
    )
    l_heavy = top.select(
        f"(residue >= {l_lo - 1} and residue <= {l_hi - 1}) and element != H"
    )
    a_com = traj.xyz[:, a_heavy, :].mean(axis=1)  # (F, 3)
    l_com = traj.xyz[:, l_heavy, :].mean(axis=1)  # (F, 3)
    com_dist_nm = np.linalg.norm(l_com - a_com, axis=1)
    com_dist_A = com_dist_nm * 10.0

    time_ns = np.linspace(0.0, total_ns, n_frames)

    return pd.DataFrame(
        {
            "frame": np.arange(n_frames),
            "time_ns": time_ns,
            "peptide_rmsd_A": pep_rmsd_A,
            "com_distance_A": com_dist_A,
        }
    )


def compute_and_write(
    job_dir: Union[Path, str],
    stride: int = 1,
    overwrite: bool = False,
) -> dict:
    """Orchestrator: read ranges.env, compute both signal sets, write outputs.

    Writes to ``<job_dir>/analysis/``:
      - ``per_residue.parquet``
      - ``traces.parquet``
      - ``meta.json``

    Returns the meta dict. Skips computation if all three outputs already
    exist and ``overwrite=False``, using mtime of run.nc / prmtop to
    invalidate (if either input is newer than meta.json, recomputes).
    """
    job_dir = Path(job_dir)
    nc_path = job_dir / "free_run" / "run.nc"
    prmtop_path = job_dir / "system_wb.prmtop"
    ranges_path = job_dir / "ranges.env"

    for p in (nc_path, prmtop_path, ranges_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    out_dir = job_dir / "analysis"
    out_pr = out_dir / "per_residue.parquet"
    out_tr = out_dir / "traces.parquet"
    out_meta = out_dir / "meta.json"

    # Mtime-based cache invalidation
    if not overwrite and out_pr.exists() and out_tr.exists() and out_meta.exists():
        meta_mtime = out_meta.stat().st_mtime
        nc_mtime = nc_path.stat().st_mtime
        prm_mtime = prmtop_path.stat().st_mtime
        if meta_mtime >= nc_mtime and meta_mtime >= prm_mtime:
            with open(out_meta) as f:
                return json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)

    a_range, l_range = _parse_ranges_env(ranges_path)

    CONTACT_CUTOFF_A = 4.0

    pr_df = compute_residue_signals(
        nc_path, prmtop_path, a_range, l_range,
        stride=stride, contact_cutoff_A=CONTACT_CUTOFF_A,
    )
    tr_df = compute_traces(
        nc_path, prmtop_path, a_range, l_range, stride=stride,
    )

    pr_df.to_parquet(out_pr, index=False)
    tr_df.to_parquet(out_tr, index=False)

    meta = {
        "target_chain": "A",
        "peptide_chain": "B",
        "n_frames": int(len(tr_df)),
        "stride": stride,
        "contact_cutoff_A": CONTACT_CUTOFF_A,
        "version": 1,
        "run_nc_mtime": nc_path.stat().st_mtime,
        "prmtop_mtime": prmtop_path.stat().st_mtime,
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    return meta
