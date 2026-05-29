"""
Filters for cPEPmatch results.

Apply various criteria to select promising CP candidates.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..runners.cpepmatch import CPEPmatchResult


def filter_by_length(
    results: list["CPEPmatchResult"],
    min_residues: int = 6,
    max_residues: int = 20,
) -> list["CPEPmatchResult"]:
    """
    Filter matches by peptide length.

    Args:
        results: List of cPEPmatch results
        min_residues: Minimum peptide length
        max_residues: Maximum peptide length

    Returns:
        Filtered list of results
    """
    return [
        r for r in results
        if min_residues <= r.num_residues <= max_residues
    ]


def filter_unique_sources(
    results: list["CPEPmatchResult"],
) -> list["CPEPmatchResult"]:
    """
    Keep only one match per source CP (best RMSD).

    Args:
        results: List of cPEPmatch results

    Returns:
        Deduplicated list (best match per source)
    """
    best_by_source: dict[str, "CPEPmatchResult"] = {}

    for r in results:
        if r.source_pdb not in best_by_source or r.fit_rmsd < best_by_source[r.source_pdb].fit_rmsd:
            best_by_source[r.source_pdb] = r

    return list(best_by_source.values())


def filter_by_fit_rmsd(
    results: list["CPEPmatchResult"],
    max_fit_rmsd: float,
) -> list["CPEPmatchResult"]:
    """
    Filter matches by Fit-RMSD (superimposition RMSD).

    Args:
        results: List of cPEPmatch results
        max_fit_rmsd: Maximum allowed Fit-RMSD in Angstroms

    Returns:
        Filtered list of results
    """
    return [r for r in results if r.fit_rmsd <= max_fit_rmsd]


def filter_mutated_only(
    results: list["CPEPmatchResult"],
) -> list["CPEPmatchResult"]:
    """
    Keep only matches where sidechains were mutated.

    Filters out results from PDB files with "NotMutated" in the filename.

    Args:
        results: List of cPEPmatch results

    Returns:
        Filtered list with only mutated matches
    """
    return [r for r in results if r.mutated]


def apply_all_filters(
    results: list["CPEPmatchResult"],
    min_residues: int = 6,
    max_residues: int = 20,
    unique_sources: bool = False,
    mutated_only: bool = True,
    max_fit_rmsd: float | None = None,
) -> list["CPEPmatchResult"]:
    """
    Apply all standard filters in sequence.

    Note: Dist-RMSD (CA distance matrix) is filtered by cPEPmatch during execution
    via frmsd_threshold. Fit-RMSD (superimposition RMSD) is filtered here post-run
    via max_fit_rmsd.

    Args:
        results: List of cPEPmatch results
        min_residues: Minimum length
        max_residues: Maximum length
        unique_sources: Whether to deduplicate by source
        mutated_only: Whether to keep only mutated matches (exclude NotMutated)
        max_fit_rmsd: Maximum Fit-RMSD in Angstroms (None = no filter)

    Returns:
        Filtered list of results
    """
    filtered = results

    if mutated_only:
        filtered = filter_mutated_only(filtered)

    if max_fit_rmsd is not None:
        filtered = filter_by_fit_rmsd(filtered, max_fit_rmsd)

    filtered = filter_by_length(filtered, min_residues, max_residues)

    if unique_sources:
        filtered = filter_unique_sources(filtered)

    # Sort by fit_rmsd (best first)
    filtered.sort(key=lambda r: r.fit_rmsd)

    return filtered
