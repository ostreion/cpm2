"""cpm2.analysis -- post-hoc triage analysis for headless CPM2 / CPNext runs.

Headless Slurm runs no longer go through the pipeline notebook, so this module
is the read-only layer for *engaging* with a finished run: it loads
``data/runs/<run_id>/`` (or an archived copy) into a tidy per-design DataFrame
and provides triage plots to decide what advances to MM-GBSA.

It has no torch / pipeline imports, so it runs in a light env (pandas,
numpy, matplotlib, pyyaml).

Enriched columns added on top of ``output/summary.csv``:

  topology               cyclization class parsed from the Stage-2 Boltz YAML
                         (head-to-tail / monocyclic / bicyclic / linear)
  cpepmatch_dist_rmsd    Stage-1 cPEPmatch distance-RMSD of the backbone match
  cpepmatch_fit_rmsd     Stage-1 cPEPmatch fit-RMSD after superposition
  peptide_drift_ca_rmsd  CA-RMSD of the final PH-refined peptide backbone vs the
                         original cPEPmatch scaffold (Kabsch on the peptide only)
  target_drift_ca_rmsd   renamed from metadata.target_ca_rmsd; CA-RMSD of the
                         target chain vs the input template.

  NOTE on target_drift_ca_rmsd: when the run used proteinhunter.template_force,
  Boltz actively pulls the target toward the template, so this value is bounded
  by the forcing potential. It is then a one-sided diagnostic -- a low value is
  expected and uninformative; only a HIGH value (forcing failed) is meaningful.
  Use RunData.template_force to check.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

_REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = _REPO_ROOT / "data" / "runs"
ARCHIVES_DIR = _REPO_ROOT / "archives"
CPEPMATCH_DB_CSV = _REPO_ROOT / "lib" / "cPEPmatch" / "database" / "cyclo_pep.csv"

# Default MMGBSA benchmark location (the overnight 2026-05-26 run). Helpers
# accept overrides so this is a default, not a hard-coded path.
MMGBSA_BENCH_DIR = _REPO_ROOT / "benchmarks" / "md_20260526_overnight"
MMGBSA_RESULTS_CSV = MMGBSA_BENCH_DIR / "results_running.csv"
MMGBSA_SHORTLIST_CSV = MMGBSA_BENCH_DIR / "shortlist_master.csv"

# Order/colours for the cyclization classes. Cyclic classes first, linear last.
TOPOLOGY_ORDER = [
    "head-to-tail",
    "monocyclic (1SS)",
    "bicyclic (htt+1SS)",
    "bicyclic (2SS)",
    "linear",
    "unknown",
]
_TOPOLOGY_COLORS = dict(zip(TOPOLOGY_ORDER, plt.get_cmap("tab10").colors))

CYCLIC_TOPOLOGIES = {t for t in TOPOLOGY_ORDER if t not in ("linear", "unknown")}


# --------------------------------------------------------------------------
# Run discovery
# --------------------------------------------------------------------------
def list_runs() -> pd.DataFrame:
    """Tabulate every run found under data/runs/ and archives/."""
    rows = []
    for root, kind in ((RUNS_DIR, "live"), (ARCHIVES_DIR, "archived")):
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir() or not (d / "output" / "summary.csv").is_file():
                continue
            rows.append({"run_id": d.name, "location": kind,
                         "modified": pd.Timestamp(d.stat().st_mtime, unit="s")})
    if not rows:
        # No runs on disk (e.g. a fresh checkout): return a typed-but-empty
        # frame so downstream .sort_values / column access never KeyErrors.
        return pd.DataFrame(columns=["run_id", "location", "modified"])
    return pd.DataFrame(rows).sort_values("modified", ascending=False, ignore_index=True)


def find_run(run_id: str) -> Path:
    """Resolve a run id (exact or substring) to its directory.

    Searches data/runs/ first, then archives/. If several match, the most
    recently modified one wins.
    """
    exact = RUNS_DIR / run_id
    if exact.is_dir():
        return exact
    cands: list[Path] = []
    for root in (RUNS_DIR, ARCHIVES_DIR):
        if root.is_dir():
            cands += [d for d in root.glob(f"*{run_id}*") if d.is_dir()]
    cands = [d for d in cands if (d / "output" / "summary.csv").is_file()]
    if not cands:
        raise FileNotFoundError(
            f"no finished run matching '{run_id}' under {RUNS_DIR} or {ARCHIVES_DIR}"
        )
    cands.sort(key=lambda d: d.stat().st_mtime)
    return cands[-1]


# --------------------------------------------------------------------------
# Low-level parsers (pure, dependency-light)
# --------------------------------------------------------------------------
def _pdb_ca_by_chain(path: Path) -> dict[str, np.ndarray]:
    """Map chain id -> (N, 3) array of CA coordinates from a PDB file."""
    chains: dict[str, list[tuple[float, float, float]]] = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[16] not in (" ", "A"):  # skip alternate locations
                continue
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            chains.setdefault(line[21], []).append(xyz)
    return {c: np.asarray(v, dtype=float) for c, v in chains.items()}


def _kabsch_fit(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kabsch superposition of P onto Q. Returns (R, t, P_centroid).

    Apply to a moving coordinate set X (any atoms) as:
        X_aligned = (X - P_centroid) @ R.T + Q_centroid
    Caller must keep Q_centroid alongside if needed.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    pc = P.mean(axis=0)
    qc = Q.mean(axis=0)
    Pc = P - pc
    Qc = Q - qc
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    # translation so that R @ (X - pc) + qc maps mobile onto ref
    t = qc - R @ pc  # so X_aligned = X @ R.T + t  (since R orthonormal)
    return R, t, pc


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """CA-RMSD between two N x 3 coordinate sets after optimal superposition.

    Same algorithm as src/runners/proteinhunter_refine.py, copied here so the
    triage module stays free of torch / pipeline imports.
    """
    R, t, _ = _kabsch_fit(P, Q)
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    diff = P @ R.T + t - Q
    return float(np.sqrt((diff ** 2).sum() / len(P)))


def _topology(yaml_path: Path) -> str:
    """Cyclization class from a Stage-2 Boltz YAML.

    ``cyclic: true`` on the peptide block is a head-to-tail amide cycle; each
    ``- bond:`` constraint is one disulfide. This reflects the design spec that
    Boltz built and (post commit 2e4a901) that ProteinHunter refined.
    """
    if not yaml_path.is_file():
        return "unknown"
    txt = yaml_path.read_text()
    htt = "cyclic: true" in txt
    nss = txt.count("- bond:")
    if htt and nss:
        return f"bicyclic (htt+{nss}SS)"
    if htt:
        return "head-to-tail"
    if nss >= 2:
        return "bicyclic (2SS)"
    if nss == 1:
        return "monocyclic (1SS)"
    return "linear"


def _parse_match_list(path: Path) -> dict[int, dict]:
    """Parse cPEPmatch's match_list.txt -> {match_num: {dist, fit, source}}."""
    out: dict[int, dict] = {}
    if not path.is_file():
        return out
    started = False
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("Match") and "PDB" in line:
            started = True
            continue
        if not started or not s:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            num = int(parts[0])
            out[num] = {
                "source_pdb": parts[1],
                "cpepmatch_dist_rmsd": float(parts[2]),
                "cpepmatch_fit_rmsd": float(parts[-1]),  # Fit-RMSD is always last
            }
        except (ValueError, IndexError):
            continue
    return out


def _load_db_topology() -> dict[str, str]:
    """Map source PDB id -> cPEPmatch DB declared cyclization `type` string.

    cPEPmatch's cyclo_pep.csv carries an authoritative `type` column per DB
    entry; this is the ground truth the geometric detector should agree with.
    """
    out: dict[str, str] = {}
    if not CPEPMATCH_DB_CSV.is_file():
        return out
    import csv
    for row in csv.DictReader(open(CPEPMATCH_DB_CSV)):
        pdb = (row.get("PDB") or "").strip().lower()
        if pdb:
            out[pdb] = (row.get("type") or "").strip()
    return out


def _normalize_db_type(raw: str) -> tuple[str, str]:
    """Map a DB `type` string to (topology class, representability).

    representability is 'representable' (head-to-tail / disulfide chemistry the
    pipeline can encode), 'unrepresentable' (hydrocarbon staples and generic
    side-chain crosslinks the YAML schema has no constraint for), or 'unknown'.
    """
    s = (raw or "").lower().strip()
    if not s or s == "none":
        return "unknown", "unknown"
    if "staple" in s:
        return "staple", "unrepresentable"
    if "side-chain to side chain" in s or "side-chain to backbone" in s:
        return "sidechain-crosslink", "unrepresentable"
    htt = "head to tail" in s
    m = re.search(r"(\d+)x disulfide", s)
    nss = int(m.group(1)) if m else 0
    if htt and nss:
        return f"bicyclic (htt+{nss}SS)", "representable"
    if htt:
        return "head-to-tail", "representable"
    if nss >= 2:
        return f"bicyclic ({nss}SS)", "representable"
    if nss == 1:
        return "monocyclic (1SS)", "representable"
    return "unknown", "unknown"


def _topology_status(built: str, db_class: str, representable: str) -> str:
    """Compare as-built topology against the DB-declared one.

      ok              built == declared
      bug             declared cyclic in representable chemistry, built wrong
      unrepresentable declared a staple / side-chain crosslink -> built linear
                      is an expected limitation, not a bug
      unverified      DB has no usable type for this source
    """
    if db_class == "unknown" or representable == "unknown":
        return "unverified"
    if representable == "unrepresentable":
        return "unrepresentable"
    return "ok" if built == db_class else "bug"


def _resolve(run_dir: Path, recorded: str) -> Path:
    """Resolve a path recorded in summary.csv against the current run_dir.

    Recorded paths are absolute and go stale when the vault moves; rebuild them
    from the run_dir plus the intermediate/ or output/ suffix.
    """
    p = Path(recorded)
    if p.is_file():
        return p
    parts = p.parts
    for marker in ("intermediate", "output"):
        if marker in parts:
            cand = run_dir / Path(*parts[parts.index(marker):])
            if cand.is_file():
                return cand
    return p


def superpose_pdbs(paths: list[Path], chain: str | None = None) -> list[str]:
    """Kabsch-superpose all PDBs onto the first using CA atoms.

    If `chain` is None, picks the longest-CA chain shared between each pair
    (the largest common subset by length). Returns a list of PDB strings
    (aligned, same order as input). The first PDB is passed through unmodified.

    Common-chain heuristic: use the longest chain that has the same CA count
    in the reference. Falls back to no-op (passthrough) on mismatch.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        return []
    ref_text = Path(paths[0]).read_text()
    out: list[str] = [ref_text]
    ref_chains = _pdb_ca_by_chain(paths[0])
    if not ref_chains:
        # Degenerate: just return raw text for all.
        for p in paths[1:]:
            out.append(Path(p).read_text())
        return out

    def _pick_chain(ch: dict[str, np.ndarray], target_len: int | None,
                    explicit: str | None) -> tuple[str, np.ndarray] | None:
        if explicit and explicit in ch:
            return explicit, ch[explicit]
        if target_len is not None:
            matches = [(k, v) for k, v in ch.items() if len(v) == target_len]
            if matches:
                # Prefer the longest match.
                k, v = max(matches, key=lambda kv: len(kv[1]))
                return k, v
        if ch:
            k = max(ch.keys(), key=lambda k: len(ch[k]))
            return k, ch[k]
        return None

    ref_pick = _pick_chain(ref_chains, None, chain)
    if ref_pick is None:
        for p in paths[1:]:
            out.append(Path(p).read_text())
        return out
    _, ref_ca = ref_pick

    for p in paths[1:]:
        mov_chains = _pdb_ca_by_chain(p)
        mov_pick = _pick_chain(mov_chains, len(ref_ca), chain)
        text = Path(p).read_text()
        if mov_pick is None or len(mov_pick[1]) != len(ref_ca) or len(ref_ca) < 3:
            out.append(text)
            continue
        _, mov_ca = mov_pick
        R, t, _ = _kabsch_fit(mov_ca, ref_ca)
        out.append(_apply_transform_pdb(text, R, t))
    return out


_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


def _pdb_ca_records_by_chain(path: Path) -> dict[str, list[tuple[int, str, str, np.ndarray]]]:
    """Per-chain ordered CA records: (resnum, icode, one_letter, xyz).

    Skips alternate locations and unknown residues (mapped to 'X'). Records are
    returned in file order, which for canonical PDBs equals N-to-C order along
    each chain.
    """
    chains: dict[str, list[tuple[int, str, str, np.ndarray]]] = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[16] not in (" ", "A"):
                continue
            try:
                resnum = int(line[22:26])
                xyz = np.array((float(line[30:38]), float(line[38:46]),
                                float(line[46:54])), dtype=float)
            except ValueError:
                continue
            resname = line[17:20].strip().upper()
            one = _AA3_TO_1.get(resname, "X")
            icode = line[26]
            chains.setdefault(line[21], []).append((resnum, icode, one, xyz))
    return chains


def superpose_pdbs_by_sequence(
    paths: list[Path],
    ref_chain: str | None = None,
    min_matched: int = 20,
) -> list[str]:
    """Sequence-aware Kabsch superposition onto the first PDB.

    For each mover, every chain's CA sequence is globally aligned to the
    reference target chain (longest CA chain of paths[0], or `ref_chain` if
    given). The chain with the highest alignment score is used as the
    correspondence, and Kabsch is fit on the matched (non-gap on both sides)
    CA pairs. The resulting rigid transform is then applied to every atom in
    the mover PDB, so non-target chains (e.g. the bound peptide) move with
    the target as a rigid body and the binding-site geometry is preserved.

    Falls back to ``superpose_pdbs`` for any mover where biopython is not
    available or the aligned-pair count is below ``min_matched``.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        return []
    ref_text = paths[0].read_text()
    out: list[str] = [ref_text]
    ref_chains = _pdb_ca_records_by_chain(paths[0])
    if not ref_chains:
        for p in paths[1:]:
            out.append(p.read_text())
        return out
    if ref_chain and ref_chain in ref_chains:
        ref_recs = ref_chains[ref_chain]
    else:
        ref_recs = max(ref_chains.values(), key=len)
    ref_seq = "".join(r[2] for r in ref_recs)
    ref_xyz = np.stack([r[3] for r in ref_recs])

    try:
        from Bio.Align import PairwiseAligner, substitution_matrices
        aligner = PairwiseAligner()
        aligner.mode = "global"
        try:
            aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
            aligner.open_gap_score = -10
            aligner.extend_gap_score = -1
        except Exception:
            aligner.match_score = 2
            aligner.mismatch_score = -1
            aligner.open_gap_score = -5
            aligner.extend_gap_score = -1
        have_aligner = True
    except Exception:
        have_aligner = False

    for p in paths[1:]:
        if not have_aligner:
            out.append(_fallback_superpose_one(paths[0], p))
            continue
        mov_chains = _pdb_ca_records_by_chain(p)
        if not mov_chains:
            out.append(p.read_text())
            continue
        best = None  # (score, matched_pairs, chain_id)
        for cid, recs in mov_chains.items():
            mov_seq = "".join(r[2] for r in recs)
            if len(mov_seq) < 3:
                continue
            try:
                aln = aligner.align(ref_seq, mov_seq)[0]
            except Exception:
                continue
            # PairwiseAligner.Alignment.aligned: pair of arrays of (start,end)
            # blocks on (target, query) that match without gaps. Iterate
            # blocks and emit per-residue index pairs.
            pairs: list[tuple[int, int]] = []
            try:
                ref_blocks, mov_blocks = aln.aligned
                for (rs, re_), (ms, me_) in zip(ref_blocks, mov_blocks):
                    n = min(re_ - rs, me_ - ms)
                    for k in range(n):
                        pairs.append((rs + k, ms + k))
            except Exception:
                continue
            if best is None or aln.score > best[0]:
                best = (float(aln.score), pairs, cid)
        if best is None or len(best[1]) < min_matched:
            out.append(_fallback_superpose_one(paths[0], p))
            continue
        _, pairs, cid = best
        mov_xyz = np.stack([mov_chains[cid][m][3] for _, m in pairs])
        ref_pts = np.stack([ref_xyz[r] for r, _ in pairs])
        R, t, _ = _kabsch_fit(mov_xyz, ref_pts)
        out.append(_apply_transform_pdb(p.read_text(), R, t))
    return out


def _fallback_superpose_one(ref_path: Path, mover_path: Path) -> str:
    """Single-mover wrapper around the legacy length-matching superposition."""
    aligned = superpose_pdbs([ref_path, mover_path])
    return aligned[1] if len(aligned) > 1 else mover_path.read_text()


def _apply_transform_pdb(pdb_text: str, R: np.ndarray, t: np.ndarray) -> str:
    """Apply X' = X @ R.T + t to every ATOM/HETATM record in a PDB string."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    lines: list[str] = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                lines.append(line)
                continue
            v = np.array([x, y, z]) @ R.T + t
            lines.append(f"{line[:30]}{v[0]:8.3f}{v[1]:8.3f}{v[2]:8.3f}{line[54:]}")
        else:
            lines.append(line)
    return "\n".join(lines) + ("\n" if pdb_text.endswith("\n") else "")


def _peptide_drift(scaffold_pdb: Path, final_pdb: Path) -> float:
    """CA-RMSD of the design peptide backbone vs its cPEPmatch scaffold.

    Kabsch-superposed on the peptide alone, so it measures pure conformational
    change of the binder, independent of where it sits on the target.
    """
    if not scaffold_pdb.is_file() or not final_pdb.is_file():
        return float("nan")
    try:
        scaf = _pdb_ca_by_chain(scaffold_pdb)
        fin = _pdb_ca_by_chain(final_pdb)
    except OSError:
        return float("nan")
    if not scaf or not fin:
        return float("nan")
    # The cPEPmatch match file is the peptide only -> take its longest chain.
    scaf_ca = max(scaf.values(), key=len)
    # The design has target + peptide -> the peptide is the chain whose CA
    # count equals the scaffold's (fall back to the shortest chain).
    same = [v for v in fin.values() if len(v) == len(scaf_ca)]
    fin_ca = same[0] if same else min(fin.values(), key=len)
    if len(scaf_ca) != len(fin_ca) or len(scaf_ca) < 3:
        return float("nan")
    return _kabsch_rmsd(scaf_ca, fin_ca)


def _load_config(run_dir: Path, manifest: dict) -> dict:
    """Load the run's config YAML (for template_force etc.)."""
    cfg_path = None
    cp = (manifest or {}).get("config_yaml_path")
    if cp and Path(cp).is_file():
        cfg_path = Path(cp)
    if cfg_path is None:  # derive <name> from run_id = <name>_<ts>_<sha>
        name = run_dir.name.rsplit("_", 2)[0]
        cand = _REPO_ROOT / "configs" / f"{name}.yaml"
        if cand.is_file():
            cfg_path = cand
    if cfg_path and yaml is not None:
        return yaml.safe_load(cfg_path.read_text()) or {}
    return {}


# --------------------------------------------------------------------------
# Run container
# --------------------------------------------------------------------------
@dataclass
class RunData:
    """A loaded run: the per-design DataFrame plus config / manifest context."""

    run_id: str
    run_dir: Path
    designs: pd.DataFrame
    config: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)

    @property
    def template_force(self) -> bool:
        return bool(self.config.get("proteinhunter", {}).get("template_force", False))

    @property
    def template_force_threshold(self):
        return self.config.get("proteinhunter", {}).get("template_force_threshold")

    @property
    def n_designs(self) -> int:
        return len(self.designs)

    @property
    def n_matches(self) -> int:
        return int(self.designs["match"].nunique())

    @property
    def cyclic(self) -> pd.DataFrame:
        """Designs whose topology is genuinely cyclic (drops linear/unknown)."""
        return self.designs[self.designs["topology"].isin(CYCLIC_TOPOLOGIES)]

    def overview(self) -> str:
        """One-paragraph text summary of the run."""
        d = self.designs
        topo = d["topology"].value_counts()
        lines = [
            f"Run            : {self.run_id}",
            f"Designs        : {self.n_designs} across {self.n_matches} matches",
            f"Cyclic designs : {len(self.cyclic)} / {self.n_designs}",
            f"ipTM           : {d['iptm'].min():.3f} - {d['iptm'].max():.3f} "
            f"(median {d['iptm'].median():.3f})",
            f"template_force : {self.template_force}"
            + (f" (threshold {self.template_force_threshold} A -- "
               "target_drift is forcing-bounded)" if self.template_force else ""),
            "Topology       : "
            + ", ".join(f"{k} x{v}" for k, v in topo.items()),
        ]
        status = d["topology_status"].value_counts().to_dict()
        n_bug = status.get("bug", 0)
        line = "Topology audit : " + ", ".join(f"{k} x{v}" for k, v in status.items())
        if n_bug:
            line += (f"  <-- {n_bug} designs built with the WRONG topology "
                     "vs the cPEPmatch DB")
        lines.append(line)
        return "\n".join(lines)

    def topology_audit(self) -> pd.DataFrame:
        """Per-match table of DB-declared vs as-built topology and verdict."""
        cols = ["match", "source_pdb", "topology_db_raw",
                "topology_db", "topology", "topology_status"]
        return (self.designs[cols].drop_duplicates("match")
                .sort_values("topology_status").reset_index(drop=True))


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------
def load_run(run_id: str) -> RunData:
    """Load a finished run into a RunData with the enriched per-design table."""
    run_dir = find_run(run_id)
    summary = run_dir / "output" / "summary.csv"
    if not summary.is_file():
        raise FileNotFoundError(f"no output/summary.csv in {run_dir}")
    raw = pd.read_csv(summary)

    manifest = {}
    mf = run_dir / "manifest.json"
    if mf.is_file():
        manifest = json.loads(mf.read_text())
    config = _load_config(run_dir, manifest)

    yaml_dir = run_dir / "intermediate" / "2_boltz" / "yaml_input"
    cpep_dir = run_dir / "intermediate" / "1_cpepmatch"
    match_list = _parse_match_list(cpep_dir / "match_list.txt")
    db_topo = _load_db_topology()

    rows = []
    for rec in raw.to_dict("records"):
        name = str(rec["name"])
        toks = name.split("_")
        match_key = "_".join(toks[:2])  # e.g. match47_6u7w
        try:
            match_num = int(toks[0].replace("match", ""))
        except ValueError:
            match_num = None

        try:
            meta = ast.literal_eval(rec.get("metadata") or "{}")
        except (ValueError, SyntaxError):
            meta = {}

        ml = match_list.get(match_num, {})
        scaffold = cpep_dir / f"{match_key}.pdb"
        final = _resolve(run_dir, str(rec.get("output_pdb", "")))

        # As-built topology (Stage-2 YAML) vs cPEPmatch DB declared topology.
        built_topo = _topology(yaml_dir / f"{match_key}.yaml")
        source = toks[1].split("-")[0].lower() if len(toks) > 1 else ""
        db_raw = db_topo.get(source, "")
        db_class, representable = _normalize_db_type(db_raw)

        rows.append({
            "name": name,
            "match": match_key,
            "match_num": match_num,
            "source_pdb": ml.get("source_pdb") or source,
            "topology": built_topo,
            "topology_db": db_class,
            "topology_db_raw": db_raw,
            "topology_status": _topology_status(built_topo, db_class, representable),
            "iptm": rec.get("iptm"),
            "plddt": rec.get("plddt"),
            "iplddt": rec.get("iplddt"),
            "cycle": rec.get("cycle"),
            "design_num": rec.get("design_num"),
            "target_drift_ca_rmsd": meta.get("target_ca_rmsd", float("nan")),
            "cpepmatch_dist_rmsd": ml.get("cpepmatch_dist_rmsd", float("nan")),
            "cpepmatch_fit_rmsd": ml.get("cpepmatch_fit_rmsd", float("nan")),
            "peptide_drift_ca_rmsd": _peptide_drift(scaffold, final),
            "n_res": len(str(rec.get("optimized_sequence", ""))),
            "input_sequence": rec.get("input_sequence"),
            "optimized_sequence": rec.get("optimized_sequence"),
            "_per_cycle": meta.get("target_ca_rmsd_per_cycle", {}),
            "_output_pdb": str(final),
        })

    designs = pd.DataFrame(rows).sort_values("iptm", ascending=False, ignore_index=True)
    return RunData(run_id=run_dir.name, run_dir=run_dir, designs=designs,
                   config=config, manifest=manifest)


# --------------------------------------------------------------------------
# Triage tables
# --------------------------------------------------------------------------
def rank(rd: RunData, by: str = "iptm", n: int = 20,
         cyclic_only: bool = False, ascending: bool = False) -> pd.DataFrame:
    """Ranked triage table, default top-20 cyclic-or-not by ipTM."""
    df = rd.cyclic if cyclic_only else rd.designs
    cols = ["name", "topology", "iptm", "plddt", "iplddt",
            "cpepmatch_fit_rmsd", "peptide_drift_ca_rmsd",
            "target_drift_ca_rmsd", "optimized_sequence"]
    return df.sort_values(by, ascending=ascending).head(n)[cols].reset_index(drop=True)


def shortlist(rd: RunData, n: int = 3, per_topology: bool = True,
              min_iptm: float = 0.0) -> pd.DataFrame:
    """Pick designs for MM-GBSA.

    per_topology=True  -> the best cyclic design from each topology family,
                          capped at n (good topology coverage).
    per_topology=False -> the top n cyclic designs by ipTM.
    """
    df = rd.cyclic
    df = df[df["iptm"] >= min_iptm]
    if per_topology:
        picks = (df.sort_values("iptm", ascending=False)
                   .groupby("topology", as_index=False).head(1)
                   .sort_values("iptm", ascending=False)
                   .head(n))
    else:
        picks = df.sort_values("iptm", ascending=False).head(n)
    cols = ["name", "topology", "iptm", "plddt",
            "cpepmatch_fit_rmsd", "peptide_drift_ca_rmsd",
            "optimized_sequence", "_output_pdb"]
    return picks[cols].reset_index(drop=True)


# --------------------------------------------------------------------------
# Plots  (each returns a matplotlib Figure)
# --------------------------------------------------------------------------
def _topo_color(topo: str):
    return _TOPOLOGY_COLORS.get(topo, (0.5, 0.5, 0.5))


def plot_topology(rd: RunData):
    """Bar chart of design counts per cyclization class."""
    counts = rd.designs["topology"].value_counts()
    counts = counts.reindex([t for t in TOPOLOGY_ORDER if t in counts.index])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values,
           color=[_topo_color(t) for t in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(int(v)), ha="center", fontsize=9)
    ax.set_ylabel("designs")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"{rd.run_id}\ncyclization topology  "
                 f"({len(rd.cyclic)}/{rd.n_designs} cyclic)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def plot_distributions(rd: RunData):
    """Histograms of the per-design scores and RMSD metrics.

    Top row = Boltz/PH confidence scores (higher is better, range 0-1).
    Bottom row = RMSDs in Angstrom (lower is better; target drift is
    forcing-bounded and shown in red when template_force is on).
    """
    specs = [
        ("iptm",                  "ipTM",                          False),
        ("plddt",                 "pLDDT",                         False),
        ("iplddt",                "ipLDDT (interface pLDDT)",      False),
        ("cpepmatch_fit_rmsd",    "cPEPmatch fit-RMSD (A)",        False),
        ("peptide_drift_ca_rmsd", "peptide drift CA-RMSD (A)",     False),
        ("target_drift_ca_rmsd",  "target drift CA-RMSD (A)",      rd.template_force),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (col, label, forced) in zip(axes.flat, specs):
        vals = rd.designs[col].dropna()
        if len(vals):
            ax.hist(vals, bins=20, color="#4C72B0", edgecolor="white")
            ax.axvline(vals.median(), color="#C44E52", ls="--", lw=1.2,
                       label=f"median {vals.median():.2f}")
            ax.legend(fontsize=8)
        title = label + ("  [forcing-bounded]" if forced else "")
        ax.set_title(title, fontsize=10,
                     color="#C44E52" if forced else "black")
        ax.set_xlabel(label)
        ax.set_ylabel("number of designs")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(f"{rd.run_id}  --  per-design score & RMSD distributions"
                 f"  (N={rd.n_designs})", fontsize=11)
    fig.tight_layout()
    return fig


def plot_topology_fidelity(rd: RunData):
    """Stacked bar: DB-declared topology, split by how it was actually built.

    Green = built correctly, red = built with the wrong topology (bug), grey =
    a staple / side-chain crosslink the pipeline cannot represent (expected).
    """
    status_color = {"ok": "#55A868", "bug": "#C44E52",
                    "unrepresentable": "#999999", "unverified": "#CCCCCC"}
    d = rd.designs
    declared = [t for t in
                sorted(d["topology_db"].unique(),
                       key=lambda x: -(d["topology_db"] == x).sum())]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(declared))
    for status in ("ok", "bug", "unrepresentable", "unverified"):
        vals = np.array([
            len(d[(d["topology_db"] == t) & (d["topology_status"] == status)])
            for t in declared])
        if vals.sum() == 0:
            continue
        ax.bar(declared, vals, bottom=bottom, label=status,
               color=status_color[status])
        bottom += vals
    ax.set_ylabel("designs")
    ax.set_xlabel("cPEPmatch DB declared topology")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    n_bug = int((d["topology_status"] == "bug").sum())
    ax.set_title(f"{rd.run_id}  --  topology fidelity (DB declared vs as-built)"
                 + (f"\n{n_bug} designs built with the WRONG topology"
                    if n_bug else ""))
    ax.legend(title="as-built verdict", fontsize=8)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def plot_triage_interactive(rd: RunData, x: str = "peptide_drift_ca_rmsd",
                            y: str = "iptm"):
    """Interactive triage scatter (plotly): click a legend entry to toggle a
    topology, double-click to isolate it; hover shows the design identity.

    Returns a plotly Figure. The MM-GBSA shortlist is the high-ipTM, low-drift
    corner; toggling topologies makes each class distribution easy to read.
    """
    import plotly.graph_objects as go

    df = rd.designs
    fig = go.Figure()
    for topo in [t for t in TOPOLOGY_ORDER if t in set(df["topology"])]:
        sub = df[df["topology"] == topo]
        c = _topo_color(topo)
        rgb = f"rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})"
        sizes = 8 + 42 * (sub["plddt"].fillna(0.7) - 0.7).clip(lower=0)
        fig.add_trace(go.Scatter(
            x=sub[x], y=sub[y], mode="markers", name=topo,
            marker=dict(size=sizes, color=rgb, opacity=0.8,
                        line=dict(width=0.5, color="white")),
            customdata=sub[["name", "plddt", "cpepmatch_fit_rmsd",
                            "topology_status", "optimized_sequence"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"{y}=%{{y:.3f}}<br>{x}=%{{x:.2f}}<br>"
                "pLDDT=%{customdata[1]:.3f}  fit-RMSD=%{customdata[2]:.2f}<br>"
                f"topology={topo} (%{{customdata[3]}})<br>"
                "seq=%{customdata[4]}<extra></extra>"),
        ))
    forced = x == "target_drift_ca_rmsd" and rd.template_force
    fig.update_layout(
        title=f"{rd.run_id} -- triage plane (click legend to toggle topology)",
        xaxis_title=x + ("  [forcing-bounded]" if forced else ""),
        yaxis_title=y, width=880, height=580,
        legend_title="topology  (click to filter)", template="plotly_white")
    return fig


def plot_triage(rd: RunData, x: str = "peptide_drift_ca_rmsd", y: str = "iptm"):
    """Triage scatter: y vs x, coloured by topology, sized by pLDDT.

    Default plane is ipTM vs peptide drift -- the MM-GBSA shortlist is the
    high-ipTM, low-drift corner (top-left).
    """
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for topo, sub in rd.designs.groupby("topology"):
        ax.scatter(sub[x], sub[y],
                   s=30 + 220 * (sub["plddt"].fillna(0.7) - 0.7).clip(lower=0),
                   color=_topo_color(topo), alpha=0.8, edgecolor="white",
                   linewidth=0.5, label=topo)
    ax.set_xlabel(x + ("  [forcing-bounded]"
                       if x == "target_drift_ca_rmsd" and rd.template_force else ""))
    ax.set_ylabel(y)
    ax.set_title(f"{rd.run_id}  --  triage plane (marker size = pLDDT)")
    ax.legend(fontsize=8, title="topology")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_convergence(rd: RunData, names: list[str] | None = None, n: int = 8):
    """Per-cycle target CA-RMSD traces for the top-n designs (PH refinement).

    Note: only target RMSD is logged per cycle; with template_force on this
    mostly tracks how fast forcing pulls the target into place.
    """
    df = rd.designs if names is None else rd.designs[rd.designs["name"].isin(names)]
    df = df.head(n) if names is None else df
    fig, ax = plt.subplots(figsize=(8, 5))
    for rec in df.to_dict("records"):
        pc = rec.get("_per_cycle") or {}
        if not pc:
            continue
        cycles = sorted(int(k) for k in pc)
        ax.plot(cycles, [pc[str(c)] for c in cycles], marker="o", ms=3,
                color=_topo_color(rec["topology"]), alpha=0.8,
                label=rec["name"])
    ax.set_xlabel("refinement cycle")
    ax.set_ylabel("target CA-RMSD vs template (A)")
    ax.set_title(f"{rd.run_id}  --  PH refinement convergence")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# MMGBSA validation: Boltz ipTM vs MMGBSA dG correlation
# --------------------------------------------------------------------------
def load_mmgbsa_join(
    results_csv: Path | str = MMGBSA_RESULTS_CSV,
    shortlist_csv: Path | str = MMGBSA_SHORTLIST_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the MMGBSA results CSV plus the design shortlist and inner-join
    them on ``name``.

    Returns ``(joined, refs)`` where:

      joined : per-design DataFrame with both Boltz scores (iptm, plddt,
               topology, RMSDs, sequence, run_id) and MMGBSA columns
               (dG_GB_kcal_mol, SE, SD, jobid). One row per design that
               appears in both files.
      refs   : the reference-binder rows from the results CSV (rows whose
               ``name`` starts with ``ref_``), useful for drawing a per-target
               reference dG line on plots.

    Both files are read fresh on every call (the results CSV is a *running*
    file -- more rows append as more jobs finish).
    """
    results = pd.read_csv(results_csv)
    shortlist = pd.read_csv(shortlist_csv)
    # Split refs out before the inner-join (refs aren't in the design shortlist).
    is_ref = results["name"].astype(str).str.startswith("ref_")
    refs = results[is_ref].copy().reset_index(drop=True)
    designs_res = results[~is_ref].copy()
    joined = shortlist.merge(designs_res, on="name", how="inner",
                             suffixes=("", "_mmgbsa"))
    # If shortlist + results both have a `target` column, prefer the shortlist
    # value (canonical) and drop the merge artefact.
    if "target_mmgbsa" in joined.columns:
        joined = joined.drop(columns=["target_mmgbsa"])
    return joined.reset_index(drop=True), refs


def _spearman_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Return (spearman_rho, pearson_r, n_used) over the finite, paired values.

    Uses scipy.stats if available; falls back to a pure-numpy implementation
    (ties broken by average rank, same as scipy's default).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(len(x))
    if n < 2:
        return float("nan"), float("nan"), n
    try:
        from scipy.stats import pearsonr, spearmanr
        rho = float(spearmanr(x, y).statistic)
        r = float(pearsonr(x, y).statistic)
        return rho, r, n
    except ImportError:
        # Numpy fallback. rankdata with average ties.
        def _rank(a: np.ndarray) -> np.ndarray:
            order = np.argsort(a, kind="mergesort")
            ranks = np.empty_like(a, dtype=float)
            ranks[order] = np.arange(1, len(a) + 1, dtype=float)
            # Average ties.
            _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
            sums = np.zeros_like(counts, dtype=float)
            np.add.at(sums, inv, ranks)
            avg = sums / counts
            return avg[inv]
        rx, ry = _rank(x), _rank(y)
        def _pearson(u, v):
            um, vm = u - u.mean(), v - v.mean()
            denom = float(np.sqrt((um ** 2).sum() * (vm ** 2).sum()))
            return float((um * vm).sum() / denom) if denom else float("nan")
        return _pearson(rx, ry), _pearson(x, y), n


def mmgbsa_correlations(joined: pd.DataFrame) -> pd.DataFrame:
    """Per-target Spearman + Pearson between iptm and dG_GB_kcal_mol.

    The expected sign is **negative**: higher ipTM should track more negative
    (better) dG. A near-zero or positive value flags disagreement between the
    cheap Boltz filter and the expensive MMGBSA truth.
    """
    rows = []
    for target, sub in joined.groupby("target", sort=True):
        rho, r, n = _spearman_pearson(sub["iptm"].values,
                                      sub["dG_GB_kcal_mol"].values)
        rows.append({"target": target, "n": n,
                     "spearman_rho": rho, "pearson_r": r,
                     "iptm_min": float(sub["iptm"].min()),
                     "iptm_max": float(sub["iptm"].max()),
                     "dG_min": float(sub["dG_GB_kcal_mol"].min()),
                     "dG_max": float(sub["dG_GB_kcal_mol"].max())})
    rho_all, r_all, n_all = _spearman_pearson(
        joined["iptm"].values, joined["dG_GB_kcal_mol"].values)
    rows.append({"target": "ALL", "n": n_all,
                 "spearman_rho": rho_all, "pearson_r": r_all,
                 "iptm_min": float(joined["iptm"].min()),
                 "iptm_max": float(joined["iptm"].max()),
                 "dG_min": float(joined["dG_GB_kcal_mol"].min()),
                 "dG_max": float(joined["dG_GB_kcal_mol"].max())})
    return pd.DataFrame(rows)


# Map MMGBSA reference-row `name` to the target key used in shortlist_master.
# Today only mdm2 has a ref in results_running.csv; the others are still queued.
# Extend this map when the wet refs land.
_MMGBSA_REF_TO_TARGET = {
    "ref_1ycr_mdm2_p53": "mdm2_p53",
    "ref_bclxl": "bclxl",
    "ref_14_3_3": "14_3_3",
    "ref_cypa": "cypA",
    "ref_cyp": "cypA",
}


def mmgbsa_refs_by_target(refs: pd.DataFrame) -> dict[str, float]:
    """Map target -> reference natural-partner dG_GB (kcal/mol), if known.

    Uses an explicit name -> target map so unexpected ref rows don't get
    silently assigned. Returns {} when no refs are in the CSV yet.
    """
    out: dict[str, float] = {}
    if refs is None or refs.empty:
        return out
    for rec in refs.to_dict("records"):
        tgt = _MMGBSA_REF_TO_TARGET.get(str(rec.get("name", "")).lower()) \
              or _MMGBSA_REF_TO_TARGET.get(str(rec.get("name", "")))
        if tgt is None:
            continue
        out[tgt] = float(rec["dG_GB_kcal_mol"])
    return out


def mmgbsa_ranking_flips(joined: pd.DataFrame) -> pd.DataFrame:
    """Per-target: best-by-ipTM vs best-by-MMGBSA. Flag if they disagree."""
    rows = []
    for target, sub in joined.groupby("target", sort=True):
        if sub.empty:
            continue
        best_iptm = sub.loc[sub["iptm"].idxmax()]
        # MMGBSA: most-negative dG wins.
        best_dg = sub.loc[sub["dG_GB_kcal_mol"].idxmin()]
        flipped = bool(best_iptm["name"] != best_dg["name"])
        rows.append({
            "target": target,
            "n_designs": int(len(sub)),
            "best_by_iptm": best_iptm["name"],
            "iptm_at_best_iptm": float(best_iptm["iptm"]),
            "dG_at_best_iptm": float(best_iptm["dG_GB_kcal_mol"]),
            "best_by_mmgbsa": best_dg["name"],
            "iptm_at_best_mmgbsa": float(best_dg["iptm"]),
            "dG_at_best_mmgbsa": float(best_dg["dG_GB_kcal_mol"]),
            "ranking_flipped": flipped,
        })
    return pd.DataFrame(rows)


def plot_mmgbsa_vs_iptm(joined: pd.DataFrame, refs: pd.DataFrame | None = None):
    """Faceted scatter of MMGBSA dG vs Boltz ipTM, one panel per target.

    One point per design, colored by topology, labeled with the match name.
    A dashed horizontal line marks the natural-partner reference dG when one
    is available for that target.
    """
    import altair as alt

    df = joined.copy()
    df["label"] = df["name"].astype(str).str.replace("_model_", "/m", regex=False) \
                                        .str.replace("_design", "/d", regex=False)
    ref_map = mmgbsa_refs_by_target(refs) if refs is not None else {}

    # Attach the reference dG as a per-row column so facet can use a single
    # data source (altair's facet-of-layer requires that). Designs without a
    # ref get NaN, which the rule layer naturally skips.
    df["ref_dG"] = df["target"].map(ref_map) if ref_map else float("nan")

    encx = alt.X("iptm:Q", title="Boltz ipTM",
                 scale=alt.Scale(zero=False, nice=True))
    ency = alt.Y("dG_GB_kcal_mol:Q",
                 title="MMGBSA dG_GB (kcal/mol)  --  more negative = better",
                 scale=alt.Scale(zero=False, nice=True))
    color = alt.Color("topology:N", title="topology")
    tooltip = ["name", "target", "topology", "iptm", "plddt",
               "dG_GB_kcal_mol", "SE", "SD",
               "cpepmatch_fit_rmsd", "peptide_drift_ca_rmsd",
               "optimized_sequence"]

    points = alt.Chart(df).mark_circle(
        size=140, opacity=0.85, stroke="white", strokeWidth=0.8
    ).encode(x=encx, y=ency, color=color, tooltip=tooltip)
    text = alt.Chart(df).mark_text(
        align="left", baseline="middle", dx=7, fontSize=9
    ).encode(x=encx, y=ency, text=alt.Text("label:N"),
             color=alt.value("#333"))
    layers = [points, text]

    if ref_map:
        # One horizontal rule per facet, drawn from the same df so facet works.
        # Aggregate to one row per target so the rule encodes a single y.
        rule = (alt.Chart(df)
                .mark_rule(strokeDash=[5, 4], color="#C44E52", size=1.5)
                .encode(y=alt.Y("mean(ref_dG):Q"),
                        tooltip=[alt.Tooltip("mean(ref_dG):Q",
                                             title="natural-partner ref dG")])
                .transform_filter("isValid(datum.ref_dG)"))
        layers.append(rule)

    return (alt.layer(*layers, data=df)
            .facet(facet=alt.Facet("target:N", title=None,
                                   header=alt.Header(labelFontSize=12,
                                                     labelFontWeight="bold")),
                   columns=2)
            .resolve_scale(x="independent", y="independent")
            .properties(title="MMGBSA validation: dG_GB vs Boltz ipTM "
                              "(per target; dashed red = natural-partner ref)"))


def convergence_altair(rd: RunData, n: int = 8):
    """Altair version of plot_convergence -- responsive, interactive.

    Use this from marimo notebooks where the matplotlib figure overflows
    fixed-width hstack columns. Returns an altair Chart (not a figure).
    """
    import altair as alt

    rows = []
    for rec in rd.designs.head(n).to_dict("records"):
        pc = rec.get("_per_cycle") or {}
        for k, v in pc.items():
            try:
                rows.append({"name": rec["name"], "topology": rec["topology"],
                             "cycle": int(k), "target_ca_rmsd": float(v)})
            except (TypeError, ValueError):
                continue
    if not rows:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})) \
            .mark_text(text="No per-cycle RMSD logged for the top-n designs.") \
            .properties(width="container", height=320)
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=35, opacity=0.85))
        .encode(
            x=alt.X("cycle:Q", title="PH refinement cycle",
                    axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("target_ca_rmsd:Q",
                    title="target CA-RMSD vs template (A)",
                    scale=alt.Scale(zero=False)),
            color=alt.Color("topology:N", title="topology"),
            detail="name:N",
            tooltip=["name", "topology", "cycle", "target_ca_rmsd"],
        )
        .properties(width="container", height=360,
                    title=f"{rd.run_id}  --  PH refinement convergence")
        .interactive()
    )
