"""Regression tests for cPEPmatch peptide residue counting.

Background (the "fragment" bug):
The cPEPmatch database stores N-/C-terminal caps (ACE, NH2) and non-standard
amino acids (ABA, PTR, ALY, HCS, staple residues, D-amino acids, ...) as
HETATM records. Biopython tags those residues with a hetero flag, so the old
``_count_residues_in_pdb`` (which counted only ``residue.id[0] == " "``)
silently undercounted every macrocycle containing a cap or non-standard
residue. ``cpepmatch_filters.filter_by_length`` then dropped valid peptides
(e.g. an 8-residue capped macrocycle counted as 4 -> below min_residues=6)
before they ever reached Boltz.

Fixture: match7_1vwb.pdb is a real cPEPmatch output - an 8-residue disulfide
macrocycle whose chain is ACE-CYS-HIS-PRO-GLN-PHE-CYS-NH2. Only the 4 plain
standard residues are written as ATOM records; ACE, NH2 and both CYS are
HETATM. The correct peptide length is 8.
"""

from pathlib import Path

from cpm2.runners.cpepmatch import (
    _count_residues_in_pdb,
    _is_peptide_residue,
    _source_pdb_residue_count,
)

FIXTURES = Path(__file__).parent / "fixtures" / "pdbs"


def test_count_includes_caps_and_nonstandard_residues():
    """A capped macrocycle must count caps + non-standard residues.

    match7_1vwb.pdb is ACE-CYS-HIS-PRO-GLN-PHE-CYS-NH2 = 8 residues. The
    pre-fix counter returned 4 (standard ATOM residues only), which is below
    min_residues=6 and caused the peptide to be wrongly dropped.
    """
    n = _count_residues_in_pdb(FIXTURES / "match7_1vwb.pdb")
    assert n == 8, f"expected 8 peptide residues, got {n}"


def test_count_not_undercounted_below_min_residues():
    """The fixture must survive a default length filter (min_residues=6)."""
    n = _count_residues_in_pdb(FIXTURES / "match7_1vwb.pdb")
    assert n >= 6, "capped macrocycle wrongly counts below min_residues"


def test_standard_only_peptide_unchanged():
    """A peptide with only standard residues still counts correctly."""
    # match6_3ava.pdb is a plain 8-residue head-to-tail cyclic peptide.
    n = _count_residues_in_pdb(FIXTURES / "match6_3ava.pdb")
    assert n == 8


def test_is_peptide_residue_classification():
    """Caps / non-standard amino acids count; water and ions do not."""

    class _FakeResidue:
        def __init__(self, het_flag, resname):
            self.id = (het_flag, 1, " ")
            self._resname = resname

        def get_resname(self):
            return self._resname

    # Terminal caps and non-standard amino acids: count.
    assert _is_peptide_residue(_FakeResidue("H_ACE", "ACE")) is True
    assert _is_peptide_residue(_FakeResidue("H_NH2", "NH2")) is True
    assert _is_peptide_residue(_FakeResidue("H_PTR", "PTR")) is True
    assert _is_peptide_residue(_FakeResidue(" ", "HIS")) is True
    # Solvent and ions: do not count.
    assert _is_peptide_residue(_FakeResidue("W", "HOH")) is False
    assert _is_peptide_residue(_FakeResidue("H_NA", "NA")) is False
    assert _is_peptide_residue(_FakeResidue("H_SO4", "SO4")) is False


def test_source_pdb_residue_count_missing_db_is_none():
    """No database location -> cross-check is a graceful no-op."""
    assert _source_pdb_residue_count(None, "1vwb") is None
    assert _source_pdb_residue_count(Path("/nonexistent"), "1vwb") is None
