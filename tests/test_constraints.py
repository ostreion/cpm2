"""Smoke tests for cpm2.utils.constraints.detect_constraints_from_pdb.

Three representative match PDBs from a historical cPEPmatch run are frozen
under tests/fixtures/pdbs/ as a regression baseline. Expected values were
captured when the function lived in CPM2.ipynb Cell 13 and are byte-identical
to what the extracted module produces (verified via boltz_yaml_baseline diff).
"""

from pathlib import Path

import pytest
from cpm2.utils.constraints import detect_constraints_from_pdb

FIXTURES = Path(__file__).parent / "fixtures" / "pdbs"


def test_head_to_tail_cyclic():
    """match6_3ava.pdb is a head-to-tail cyclic peptide with no other bonds."""
    r = detect_constraints_from_pdb(FIXTURES / "match6_3ava.pdb", "P", 1.5, 2.5)
    assert r["head_to_tail"] is True
    assert r["head_to_tail_distance"] == pytest.approx(1.3438, abs=1e-3)
    assert r["disulfides"] == []
    assert r["lactams"] == []
    assert r["thioethers"] == []


def test_disulfide_bridge():
    """match2_2ck0.pdb has a single disulfide CYS1-CYS10 and is not head-to-tail."""
    r = detect_constraints_from_pdb(FIXTURES / "match2_2ck0.pdb", "P", 1.5, 2.5)
    assert r["head_to_tail"] is False
    assert r["head_to_tail_distance"] == pytest.approx(9.1435, abs=1e-3)
    assert r["disulfides"] == [(1, 10)]
    assert r["lactams"] == []
    assert r["thioethers"] == []


def test_no_constraints():
    """match25_6dl1.pdb has no cyclization bonds (linear, within cyc_dist>1.5A)."""
    r = detect_constraints_from_pdb(FIXTURES / "match25_6dl1.pdb", "P", 1.5, 2.5)
    assert r["head_to_tail"] is False
    assert r["head_to_tail_distance"] == pytest.approx(3.0251, abs=1e-3)
    assert r["disulfides"] == []
    assert r["lactams"] == []
    assert r["thioethers"] == []


def test_missing_chain_returns_empty():
    """A chain that does not exist returns empty defaults without raising."""
    r = detect_constraints_from_pdb(FIXTURES / "match6_3ava.pdb", "Z", 1.5, 2.5)
    assert r == {
        "head_to_tail": False,
        "head_to_tail_distance": None,
        "disulfides": [],
        "lactams": [],
        "thioethers": [],
    }


def test_missing_file_returns_empty():
    """A missing file returns empty defaults without raising."""
    r = detect_constraints_from_pdb(FIXTURES / "does_not_exist.pdb", "P", 1.5, 2.5)
    assert r["head_to_tail"] is False
    assert r["disulfides"] == []
