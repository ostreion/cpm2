"""Validation for the DB-aware cyclisation detector.

The cPEPmatch ``type`` column is the authoritative source for a macrocycle's
topology. ``detect_constraints_db_aware`` must reproduce that topology even
when the match PDB leaves the head-to-tail amide bond geometrically open.

These tests run the detector over the 37 match PDBs that the MDM2/p53
benchmark run actually built (``1_cpepmatch_renamed/`` -- chain "P", the
production input), and assert the resulting head_to_tail flag and disulfide
count agree with the DB ``type`` for every representable match.

The pure-geometry detector mismatches 13 of those 37; the DB-aware detector
must mismatch 0.
"""

from pathlib import Path

import pytest
from cpm2.utils.constraints import (
    detect_constraints_db_aware,
    detect_constraints_from_pdb,
    load_cpepmatch_db,
    parse_cpepmatch_type,
)

REPO = Path(__file__).resolve().parents[1]
DB_CSV = REPO / "lib" / "cPEPmatch" / "database" / "cyclo_pep.csv"
MATCH_DIR = (
    REPO / "data" / "runs" / "mdm2_p53_v1_20260521_051455_26bc2b5c"
    / "intermediate" / "1_cpepmatch_renamed"
)

# Skip the whole module if the benchmark run artifacts are not present
# (they are large run outputs, not guaranteed to exist in every checkout).
pytestmark = pytest.mark.skipif(
    not MATCH_DIR.exists() or not DB_CSV.exists(),
    reason="MDM2/p53 benchmark run artifacts or cPEPmatch DB not present",
)


def _source_pdb_id(match_pdb: Path) -> str:
    stem = match_pdb.stem
    if stem.endswith("-NotMutated"):
        stem = stem[: -len("-NotMutated")]
    return stem.rsplit("_", 1)[-1]


def _match_pdbs() -> list[Path]:
    return sorted(MATCH_DIR.glob("match*.pdb"))


def test_parse_cpepmatch_type_clauses():
    assert parse_cpepmatch_type("head to tail")["head_to_tail"] is True
    assert parse_cpepmatch_type("1x disulfide bridge")["disulfide_count"] == 1
    r = parse_cpepmatch_type("3x disulfide bridge, head to tail")
    assert r["disulfide_count"] == 3 and r["head_to_tail"] is True
    assert r["representable"] is True
    staple = parse_cpepmatch_type("0QZ STAPLE, head to tail")
    assert staple["has_staple"] is True and staple["representable"] is False
    assert staple["head_to_tail"] is True
    sc = parse_cpepmatch_type("side-chain to side chain")
    assert sc["has_sidechain_crosslink"] is True and sc["representable"] is False
    empty = parse_cpepmatch_type("")
    assert empty == {
        "head_to_tail": False, "disulfide_count": 0, "has_staple": False,
        "has_sidechain_crosslink": False, "representable": True, "raw_type": "",
    }


def test_load_cpepmatch_db_missing_returns_empty():
    assert load_cpepmatch_db(REPO / "does" / "not" / "exist.csv") == {}


def test_db_aware_agrees_with_db_type_on_all_built_matches():
    """All representable matches must agree with the DB-declared topology."""
    db = load_cpepmatch_db(DB_CSV)
    assert db, "cPEPmatch DB failed to load"

    matches = _match_pdbs()
    assert len(matches) == 37, f"expected 37 built matches, found {len(matches)}"

    geom_mismatches: list[str] = []
    db_mismatches: list[str] = []

    for pdb in matches:
        src = _source_pdb_id(pdb)
        expected = parse_cpepmatch_type(db.get(src.lower(), ""))

        # Pure geometry (the old behaviour) -- expected to mismatch ~13.
        geom = detect_constraints_from_pdb(pdb, "P", 1.5, 2.5)
        if expected["representable"]:
            if geom["head_to_tail"] != expected["head_to_tail"]:
                geom_mismatches.append(f"{pdb.name} h2t")
            elif len(geom["disulfides"]) != expected["disulfide_count"]:
                geom_mismatches.append(f"{pdb.name} ss")

        # DB-aware (the fix) -- must mismatch 0 for representable matches.
        res = detect_constraints_db_aware(pdb, "P", src, db, 1.5, 2.5)
        if expected["representable"]:
            if res["head_to_tail"] != expected["head_to_tail"]:
                db_mismatches.append(
                    f"{pdb.name}: h2t {res['head_to_tail']} != "
                    f"{expected['head_to_tail']}"
                )
            if len(res["disulfides"]) != expected["disulfide_count"]:
                db_mismatches.append(
                    f"{pdb.name}: disulfides {len(res['disulfides'])} != "
                    f"{expected['disulfide_count']}"
                )

    print(
        f"\npure-geometry mismatches: {len(geom_mismatches)} / {len(matches)}"
        f"\nDB-aware mismatches:      {len(db_mismatches)} / {len(matches)}"
    )
    assert not db_mismatches, "DB-aware detector disagrees with DB type:\n" + (
        "\n".join(db_mismatches)
    )


def test_db_aware_provenance_keys_present():
    """Provenance keys are always present so callers can rely on them."""
    db = load_cpepmatch_db(DB_CSV)
    pdb = _match_pdbs()[0]
    res = detect_constraints_db_aware(pdb, "P", _source_pdb_id(pdb), db, 1.5, 2.5)
    for key in (
        "head_to_tail", "disulfides", "lactams", "thioethers",
        "head_to_tail_source", "db_type", "unrepresentable",
        "unrepresentable_reason", "disulfide_count_mismatch",
    ):
        assert key in res, f"missing provenance key {key}"
    assert res["head_to_tail_source"] in ("db", "geometry_fallback")


def test_db_aware_geometry_fallback_when_not_in_db():
    """An unknown source PDB falls back to geometry with the loosened cutoff."""
    pdb = _match_pdbs()[0]
    res = detect_constraints_db_aware(pdb, "P", "zzzz", {}, 1.5, 2.5)
    assert res["head_to_tail_source"] == "geometry_fallback"
    assert res["db_type"] is None
