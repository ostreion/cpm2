"""
Filters for Boltz2 results.

Validate structure predictions and select promising candidates.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..runners.boltz_runner import BoltzResult


def filter_by_iptm(
    results: list["BoltzResult"],
    threshold: float = 0.7,
) -> list["BoltzResult"]:
    """
    Filter by interface pTM score.

    Higher ipTM indicates more confident interface prediction.

    Args:
        results: List of Boltz results
        threshold: Minimum ipTM to keep

    Returns:
        Filtered list of results
    """
    return [r for r in results if r.iptm >= threshold]


def filter_by_plddt(
    results: list["BoltzResult"],
    threshold: float = 0.70,
) -> list["BoltzResult"]:
    """
    Filter by mean pLDDT score (0-1 scale, matching Boltz2 native output).

    Higher pLDDT indicates more confident structure prediction.

    Args:
        results: List of Boltz results
        threshold: Minimum mean pLDDT to keep (0-1 scale)

    Returns:
        Filtered list of results
    """
    return [r for r in results if r.plddt >= threshold]


def filter_by_rmsd(
    results: list["BoltzResult"],
    threshold: float = 3.0,
) -> list["BoltzResult"]:
    """
    Filter by RMSD to input structure.

    Low RMSD means Boltz predicts similar structure to input,
    validating that the CP actually folds into the intended pose.

    Args:
        results: List of Boltz results
        threshold: Maximum RMSD to keep (Angstroms)

    Returns:
        Filtered list of results
    """
    return [
        r for r in results
        if r.rmsd_to_input is not None and r.rmsd_to_input <= threshold
    ]


def filter_by_confidence(
    results: list["BoltzResult"],
    threshold: float = 0.5,
) -> list["BoltzResult"]:
    """
    Filter by overall confidence score.

    Higher confidence_score indicates more reliable prediction.

    Args:
        results: List of Boltz results
        threshold: Minimum confidence score to keep

    Returns:
        Filtered list of results
    """
    return [r for r in results if r.confidence_score >= threshold]


def filter_validated(
    results: list["BoltzResult"],
) -> list["BoltzResult"]:
    """
    Keep only structures that passed Boltz validation.

    A structure is "validated" if it folds into roughly the same
    pose as the input (RMSD < 3.0 A).

    Args:
        results: List of Boltz results

    Returns:
        Validated results only
    """
    return [r for r in results if r.passed_validation]


def apply_all_filters(
    results: list["BoltzResult"],
    iptm_threshold: float = 0.7,
    plddt_threshold: float = 0.70,
    rmsd_threshold: float = 3.0,
    confidence_threshold: float = 0.0,
) -> list["BoltzResult"]:
    """
    Apply all standard filters in sequence.

    Args:
        results: List of Boltz results
        iptm_threshold: Minimum ipTM (0-1 scale)
        plddt_threshold: Minimum pLDDT (0-1 scale, matching Boltz2 native output)
        rmsd_threshold: Maximum RMSD to input (Angstroms)
        confidence_threshold: Minimum confidence score (0 = no filter)

    Returns:
        Filtered list of results
    """
    filtered = results

    if confidence_threshold > 0:
        filtered = filter_by_confidence(filtered, confidence_threshold)
    filtered = filter_by_iptm(filtered, iptm_threshold)
    filtered = filter_by_plddt(filtered, plddt_threshold)
    filtered = filter_by_rmsd(filtered, rmsd_threshold)

    # Sort by ipTM (best first)
    filtered.sort(key=lambda r: r.iptm, reverse=True)

    return filtered
