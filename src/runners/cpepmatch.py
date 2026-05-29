"""
cPEPmatch runner - finds cyclic peptide binders by backbone matching.

Executes cPEPmatch in the 'cpepmatch' conda environment.

cPEPmatch finds cyclic peptides that mimic proteins and target their binding partners.
It matches CA distance motifs between a protein interface and a database of cyclic peptides.
"""

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CPEPmatchResult:
    """Result from a cPEPmatch run."""
    name: str
    pdb_path: Path
    dist_rmsd: float       # Distance RMSD from backbone matching
    fit_rmsd: float        # Fit RMSD after superimposition
    num_residues: int
    cp_residues: str       # Matched CP residues
    protein_residues: str  # Matched protein residues
    source_pdb: str        # Source PDB from database
    mutated: bool          # Whether sidechains were mutated

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pdb_path": str(self.pdb_path),
            "dist_rmsd": self.dist_rmsd,
            "fit_rmsd": self.fit_rmsd,
            "num_residues": self.num_residues,
            "cp_residues": self.cp_residues,
            "protein_residues": self.protein_residues,
            "source_pdb": self.source_pdb,
            "mutated": self.mutated,
        }


# Residue names that are NOT part of the peptide chain and must be excluded
# from a peptide-length count: bulk solvent and common crystallographic ions.
# Everything else (standard residues AND non-standard amino acids / N-/C-caps
# such as ACE, NH2, ABA, PTR, ALY, HCS, DBU, ... ) counts as a peptide residue.
_NON_PEPTIDE_RESNAMES = frozenset({
    "HOH", "WAT", "DOD",                       # water
    "NA", "K", "CL", "MG", "CA", "ZN", "MN",   # common ions
    "FE", "CU", "NI", "CO", "CD", "BR", "IOD",
    "SO4", "PO4", "GOL", "EDO", "PEG", "ACT",  # common buffer/cryo additives
})


def _is_peptide_residue(residue) -> bool:
    """True if a Biopython residue should count toward peptide length.

    cPEPmatch's database PDBs encode N-/C-terminal caps (ACE, NH2) and
    non-standard amino acids (ABA, PTR, ALY, HCS, staple residues, D-amino
    acids, ...) as HETATM records. Biopython tags those with a hetero flag
    (``residue.id[0]`` starts with ``"H_"``), so a naive
    ``residue.id[0] == " "`` test silently undercounts every macrocycle that
    contains a cap or a non-standard residue. We instead count every residue
    that is not bulk solvent / a crystallographic ion / a buffer additive.
    """
    if residue.id[0] == "W":  # Biopython water flag
        return False
    resname = residue.get_resname().strip().upper()
    return resname not in _NON_PEPTIDE_RESNAMES


def _count_residues_in_pdb(pdb_path: Path, chain: str | None = None) -> int:
    """Count number of peptide residues in a PDB chain.

    Counts standard residues *and* non-standard amino acids / terminal caps
    (which the cPEPmatch database stores as HETATM). Only bulk solvent and
    ions are excluded. See :func:`_is_peptide_residue`.

    Args:
        pdb_path: Path to PDB file
        chain: Chain ID to count. If None, sums all chains (cPEPmatch match
            PDBs may split a single macrocycle across chains A and B).
    """
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("cp", str(pdb_path))
        model = struct[0]

        if chain is None:
            # Sum peptide residues across every chain. cPEPmatch occasionally
            # emits a single macrocycle on two chains (A/B); the old
            # "first chain only" behaviour undercounted those.
            return sum(
                1
                for ch in model
                for r in ch
                if _is_peptide_residue(r)
            )
        else:
            chain_obj = model[chain]
            return sum(1 for r in chain_obj if _is_peptide_residue(r))
    except Exception:
        return 0


def _source_pdb_residue_count(
    database_location: Optional[Path], source_pdb: str
) -> Optional[int]:
    """Count peptide residues in the source database PDB for ``source_pdb``.

    cPEPmatch builds each match PDB as a residue-faithful copy of the source
    macrocycle ``<database>/<pdb>.pdb`` (it only superimposes coordinates and
    mutates side chains; it never adds or removes residues). So the source
    PDB's residue count is the correct ground truth for detecting a truncated
    output.

    NOTE: this is deliberately *not* ``cyclo_pep.csv``'s ``aminoacid_length``
    column. That column is a curated amino-acid count that (a) excludes ACE /
    NH2 terminal caps and (b) is occasionally wrong for exotic cross-linker
    residues (e.g. 6j67, 6xib), so it under-reports the true molecular residue
    count and would raise false truncation alarms.

    Returns the count, or None if the source PDB cannot be read.
    """
    if database_location is None:
        return None
    src = Path(database_location) / f"{source_pdb}.pdb"
    if not src.exists():
        src = Path(database_location) / f"{source_pdb.lower()}.pdb"
    if not src.exists():
        return None
    count = _count_residues_in_pdb(src)
    return count if count > 0 else None


def parse_match_list(
    match_list_path: Path,
    working_dir: Path,
    database_location: Optional[Path] = None,
) -> list[CPEPmatchResult]:
    """
    Parse the match_list.txt output from cPEPmatch.

    Actual format:
     Match  PDB   Dist-RMSD      cPep Residues              PP-Interface Residues      Fit-RMSD
         1  1a1p    0.9319       7    8    9   10   11     102  103  104  105  106      0.9178

    Residue columns vary based on motif_size. Fit-RMSD is always last.

    If ``database_location`` is given, the residue count of each emitted match
    PDB is cross-checked against the residue count of its source database PDB;
    a mismatch is logged as a warning so a genuinely truncated peptide can
    never proceed silently to Boltz.
    """
    results = []

    if not match_list_path.exists():
        return results

    # Cache of source-PDB residue counts, keyed by source pdb id.
    _src_count_cache: dict[str, Optional[int]] = {}

    content = match_list_path.read_text()
    lines = content.strip().split('\n')

    # Skip header lines (look for "Match" header)
    data_started = False
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith('Match') and 'PDB' in line:
            data_started = True
            continue
        if not data_started or not line_stripped:
            continue

        # Parse space-separated values
        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            match_num = int(parts[0])
            source_pdb = parts[1]
            dist_rmsd = float(parts[2])
            # Fit-RMSD is always the last column
            fit_rmsd = float(parts[-1])

            # Middle columns are residues - split evenly between CP and protein
            residue_parts = parts[3:-1]
            mid = len(residue_parts) // 2
            cp_residue_list = residue_parts[:mid]
            protein_residue_list = residue_parts[mid:]

            cp_residues = ",".join(cp_residue_list)
            protein_residues = ",".join(protein_residue_list)
            num_residues = len(cp_residue_list)

            # Find the actual PDB file (could be match1_1a1p.pdb or match1_1a1p-NotMutated.pdb)
            pattern = f"match{match_num}_{source_pdb}*.pdb"
            matching_files = list(working_dir.glob(pattern))

            if not matching_files:
                # Try alternate pattern
                pattern = f"match{match_num}_*.pdb"
                matching_files = [f for f in working_dir.glob(pattern) if source_pdb in f.name]

            if not matching_files:
                continue

            pdb_path = matching_files[0]
            mutated = "NotMutated" not in pdb_path.name

            # Get actual CP residue count from PDB file (sum all chains,
            # include non-standard residues / caps).
            num_residues = _count_residues_in_pdb(pdb_path)

            # Sanity check against the source database PDB. The match PDB is a
            # residue-faithful copy of the source macrocycle, so the counts
            # must agree exactly. A mismatch means a genuinely truncated /
            # malformed output, which must not proceed silently to Boltz.
            if source_pdb not in _src_count_cache:
                _src_count_cache[source_pdb] = _source_pdb_residue_count(
                    database_location, source_pdb
                )
            src_count = _src_count_cache[source_pdb]
            if src_count is not None and num_residues != src_count:
                logger.warning(
                    "cPEPmatch match%s_%s: emitted PDB has %d peptide "
                    "residues but the source database structure %s.pdb has "
                    "%d. The peptide is TRUNCATED or malformed; inspect %s "
                    "before trusting it downstream.",
                    match_num, source_pdb, num_residues, source_pdb,
                    src_count, pdb_path.name,
                )

            results.append(CPEPmatchResult(
                name=f"match_{match_num:03d}_{source_pdb}",
                pdb_path=pdb_path,
                dist_rmsd=dist_rmsd,
                fit_rmsd=fit_rmsd,
                num_residues=num_residues,
                cp_residues=cp_residues,
                protein_residues=protein_residues,
                source_pdb=source_pdb,
                mutated=mutated,
            ))
        except (ValueError, IndexError):
            continue

    return results


def run(
    pdb_file: Path,
    protein_chain: str,
    target_chain: str,
    output_dir: Path,
    motif_size: int = 5,
    consecutive: bool = True,
    interface_cutoff: int = 6,
    frmsd_threshold: float = 0.5,
    protein_specific_residues: Optional[str] = None,
    cyclization_type: Optional[str] = None,
    exclude_non_standard: bool = False,
    conda_env: Optional[str] = "cpepmatch",
    lib_path: Optional[Path] = None,
    clear_output: bool = True,
) -> list[CPEPmatchResult]:
    """
    Run cPEPmatch to find cyclic peptide matches.

    Args:
        pdb_file: Path to PDB file containing both protein and target
        protein_chain: Chain ID of protein to mimic (the one we want CP to replace)
        target_chain: Chain ID of target (the one CP should bind to)
        output_dir: Directory for output files
        motif_size: Number of CA carbons to match (4-7)
        consecutive: Whether to use consecutive motifs
        interface_cutoff: Distance in Angstroms for interface residues
        frmsd_threshold: Fit-RMSD threshold in Angstroms
        protein_specific_residues: Optional specific residues to match (e.g., "12,15,18")
        cyclization_type: Optional filter for cyclization type (e.g., "head to tail")
        exclude_non_standard: Whether to exclude non-standard amino acids
        conda_env: Name of conda environment with cpepmatch installed. Pass
            ``None`` to skip the inner ``conda run -n <env>`` prefix and invoke
            in the current environment (matches boltz_runner / proteinhunter;
            used by the single-env Colab path).
        lib_path: Path to cPEPmatch library (defaults to lib/cPEPmatch)
        clear_output: Clear output directory before running (default: True)

    Returns:
        List of CPEPmatchResult objects
    """
    pdb_file = Path(pdb_file)
    output_dir = Path(output_dir)

    # Clear existing output if requested
    if clear_output and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Default lib path
    if lib_path is None:
        lib_path = Path(__file__).parent.parent.parent / "lib" / "cPEPmatch"

    database_location = lib_path / "database"
    cpepmatch_script = lib_path / "cpepmatch.py"

    # Copy input PDB to working directory (cPEPmatch expects it there)
    working_pdb = output_dir / pdb_file.name
    shutil.copy(pdb_file, working_pdb)

    # Build command
    bare_cmd = [
        "python", str(cpepmatch_script),
        "-n", pdb_file.stem,
        "-p", protein_chain,
        "-t", target_chain,
        "-wl", str(output_dir) + "/",
        "-dl", str(database_location) + "/",
        "-ms", str(motif_size),
        "-cs", str(consecutive),
        "-ic", str(interface_cutoff),
        "-ft", str(frmsd_threshold),
    ]
    if conda_env:
        cmd = ["conda", "run", "-n", conda_env, "--no-capture-output", *bare_cmd]
    else:
        cmd = bare_cmd

    if protein_specific_residues:
        cmd.extend(["-psr", protein_specific_residues])
    if cyclization_type:
        cmd.extend(["-ct", cyclization_type])
    if exclude_non_standard:
        cmd.extend(["-ens", "True"])

    # Run cPEPmatch
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(output_dir),
    )

    if result.returncode != 0:
        raise RuntimeError(f"cPEPmatch failed:\n{result.stderr}\n{result.stdout}")

    # Parse results (cross-check residue counts against cyclo_pep.csv)
    match_list_path = output_dir / "match_list.txt"
    results = parse_match_list(
        match_list_path, output_dir, database_location=database_location
    )

    return results


def run_from_separate_pdbs(
    protein_pdb: Path,
    protein_chain: str,
    target_pdb: Path,
    target_chain: str,
    output_dir: Path,
    **kwargs,
) -> list[CPEPmatchResult]:
    """
    Run cPEPmatch when protein and target are in separate PDB files.

    This combines them first, then runs cPEPmatch.

    Args:
        protein_pdb: PDB file with protein to mimic
        protein_chain: Chain ID in protein PDB
        target_pdb: PDB file with target
        target_chain: Chain ID in target PDB
        output_dir: Output directory
        **kwargs: Additional arguments passed to run()

    Returns:
        List of CPEPmatchResult objects
    """
    from ..utils.pdb_utils import unify_pdbs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine PDBs
    combined_pdb = output_dir / "combined_input.pdb"
    unify_pdbs(
        cp_pdb=protein_pdb,  # "protein to mimic" goes first
        cp_chain=protein_chain,
        binder_pdb=target_pdb,
        binder_chain=target_chain,
        output_pdb=combined_pdb,
        cp_output_chain="A",
        binder_output_chain="B",
    )

    return run(
        pdb_file=combined_pdb,
        protein_chain="A",
        target_chain="B",
        output_dir=output_dir,
        **kwargs,
    )
