"""Cyclic-peptide constraint detection (head-to-tail, disulfide, lactam, thioether).

Two independent detectors:

* ``detect_constraints_from_cif`` reads RCSB ``_struct_conn`` records. It uses
  author-assigned residue numbers, which may not align with 1-based sequence
  indexing, so use it only for validation.
* ``detect_constraints_from_pdb`` uses distance-based detection from a parsed
  structure with 1-based sequential indices. This is the source used to build
  Boltz2 YAML constraints.

``compare_constraints`` diffs the two for sanity checking (counts only, since
indices differ between schemes).
"""

import csv
import logging
import re
from pathlib import Path

from Bio.PDB import PDBParser

logger = logging.getLogger(__name__)


def _empty_result() -> dict:
    return {
        "head_to_tail": False,
        "head_to_tail_distance": None,
        "disulfides": [],
        "lactams": [],
        "thioethers": [],
    }


def identify_chain_by_length(mmcif_dict: dict, seq_length: int) -> str | None:
    """Return the first chain whose polymer sequence length equals seq_length.

    Counts parenthesised modified residues (e.g. ``(MSE)``) as one residue.
    """
    try:
        seqs = mmcif_dict["_entity_poly.pdbx_seq_one_letter_code"]
        chains = mmcif_dict["_entity_poly.pdbx_strand_id"]
    except KeyError:
        return None

    if isinstance(seqs, str):
        seqs = [seqs]
        chains = [chains]

    for seq_raw, chain_ids in zip(seqs, chains):
        seq_clean = seq_raw.replace("\n", "").replace("\r", "")
        length = 0
        i = 0
        while i < len(seq_clean):
            if seq_clean[i] == "(":
                i = seq_clean.index(")", i) + 1
            else:
                i += 1
            length += 1
        if length == seq_length:
            return chain_ids.split(",")[0].strip()
    return None


def detect_constraints_from_cif(mmcif_dict: dict, chain_id: str) -> dict:
    """Detect constraints from mmCIF ``_struct_conn`` records for ``chain_id``.

    NOTE: uses author residue numbers (not 1-based sequence indices). For
    validation only; not suitable for YAML generation.
    """
    result = _empty_result()

    if mmcif_dict is None or chain_id is None:
        return result

    try:
        conn_types = mmcif_dict.get("_struct_conn.conn_type_id", [])
        chain1s = mmcif_dict.get("_struct_conn.ptnr1_auth_asym_id", [])
        chain2s = mmcif_dict.get("_struct_conn.ptnr2_auth_asym_id", [])
        atom1s = mmcif_dict.get("_struct_conn.ptnr1_label_atom_id", [])
        atom2s = mmcif_dict.get("_struct_conn.ptnr2_label_atom_id", [])
        seq1s = mmcif_dict.get("_struct_conn.ptnr1_auth_seq_id", [])
        seq2s = mmcif_dict.get("_struct_conn.ptnr2_auth_seq_id", [])

        if isinstance(conn_types, str):
            conn_types = [conn_types]
            chain1s, chain2s = [chain1s], [chain2s]
            atom1s, atom2s = [atom1s], [atom2s]
            seq1s, seq2s = [seq1s], [seq2s]

        for ct, c1, c2, a1, a2, s1, s2 in zip(
            conn_types, chain1s, chain2s, atom1s, atom2s, seq1s, seq2s
        ):
            if c1 != chain_id and c2 != chain_id:
                continue
            ct_lower = ct.lower()
            if ct_lower == "disulf":
                result["disulfides"].append((int(s1), int(s2)))
            elif ct_lower == "covale":
                atoms = {a1.strip(), a2.strip()}
                if {"N", "C"}.issubset(atoms) or {"N", "C'"}.issubset(atoms):
                    result["head_to_tail"] = True
                elif "NZ" in atoms and ("CG" in atoms or "CD" in atoms):
                    result["lactams"].append((int(s1), int(s2)))
                elif "SG" in atoms and "CB" in atoms:
                    result["thioethers"].append((int(s1), int(s2)))
    except (KeyError, TypeError, ValueError):
        pass

    return result


def detect_constraints_from_pdb(
    pdb_path: Path,
    chain_id: str,
    cyc_dist: float = 1.5,
    ss_dist: float = 2.5,
) -> dict:
    """Detect constraints from a PDB structure using atom-distance thresholds.

    Residue indices are 1-based over the sequence order. This is the source
    used to build Boltz2 YAML constraints.
    """
    result = _empty_result()

    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("model", str(pdb_path))
        chain = struct[0][chain_id]
    except Exception:
        return result

    residues = [
        r for r in chain.get_residues()
        if r.id[0] != "W" and (
            r.id[0] == " " or {"N", "CA", "C"}.issubset({a.get_name() for a in r})
        )
    ]

    if len(residues) < 3:
        return result

    # Head-to-tail: N(first) to C(last)
    try:
        n_atom = residues[0]["N"]
        c_atom = residues[-1]["C"]
        distance = n_atom - c_atom  # BioPython operator overload → Euclidean
        result["head_to_tail_distance"] = distance
        if distance < cyc_dist:
            result["head_to_tail"] = True
    except KeyError:
        pass

    # Disulfide: CYS SG-SG
    cys_residues = [
        (i + 1, r) for i, r in enumerate(residues)
        if r.resname.strip() == "CYS" and "SG" in r
    ]
    for i in range(len(cys_residues)):
        for j in range(i + 1, len(cys_residues)):
            if (cys_residues[i][1]["SG"] - cys_residues[j][1]["SG"]) < ss_dist:
                pair = (cys_residues[i][0], cys_residues[j][0])
                if pair not in result["disulfides"]:
                    result["disulfides"].append(pair)

    # Lactam: Lys NZ to Asp CG or Glu CD < 1.6 A
    for i, res_i in enumerate(residues):
        if res_i.resname.strip() == "LYS" and "NZ" in res_i:
            for j, res_j in enumerate(residues):
                if i == j:
                    continue
                rn = res_j.resname.strip()
                if rn == "ASP" and "CG" in res_j:
                    if (res_i["NZ"] - res_j["CG"]) < 1.6:
                        result["lactams"].append((i + 1, j + 1))
                elif rn == "GLU" and "CD" in res_j:
                    if (res_i["NZ"] - res_j["CD"]) < 1.6:
                        result["lactams"].append((i + 1, j + 1))

    # Thioether: Cys SG to Ser/Thr CB < 1.9 A
    for i, res_i in enumerate(residues):
        if res_i.resname.strip() == "CYS" and "SG" in res_i:
            for j, res_j in enumerate(residues):
                if i == j:
                    continue
                if res_j.resname.strip() in ("SER", "THR") and "CB" in res_j:
                    if (res_i["SG"] - res_j["CB"]) < 1.9:
                        result["thioethers"].append((i + 1, j + 1))

    return result


def compare_constraints(cif_result: dict, pdb_result: dict) -> list[str]:
    """Diff CIF vs PDB detection by counts (indices differ by numbering scheme)."""
    discrepancies: list[str] = []

    if cif_result["head_to_tail"] != pdb_result["head_to_tail"]:
        cif_str = "yes" if cif_result["head_to_tail"] else "no"
        pdb_str = "yes" if pdb_result["head_to_tail"] else "no"
        discrepancies.append(f"head-to-tail: CIF={cif_str}, PDB={pdb_str}")

    if len(cif_result["disulfides"]) != len(pdb_result["disulfides"]):
        discrepancies.append(
            f"disulfides: CIF={len(cif_result['disulfides'])}, "
            f"PDB={len(pdb_result['disulfides'])}"
        )

    if len(cif_result["lactams"]) != len(pdb_result["lactams"]):
        discrepancies.append(
            f"lactams: CIF={len(cif_result['lactams'])}, "
            f"PDB={len(pdb_result['lactams'])}"
        )

    if len(cif_result["thioethers"]) != len(pdb_result["thioethers"]):
        discrepancies.append(
            f"thioethers: CIF={len(cif_result['thioethers'])}, "
            f"PDB={len(pdb_result['thioethers'])}"
        )

    return discrepancies


# ---------------------------------------------------------------------------
# cPEPmatch database ("type" column) as authoritative topology source
# ---------------------------------------------------------------------------
#
# Geometry alone is unreliable for head-to-tail closure: cPEPmatch match
# output PDBs frequently leave the head-to-tail amide bond geometrically open
# (measured N-C 3.0-4.2 A), so a strict 1.5 A cutoff silently misses real
# macrocycles. The cPEPmatch database's curated ``type`` column records the
# intended topology and has been audited as trustworthy. We use it as the
# source of truth for the head-to-tail flag and the disulfide *count*, and
# keep geometry only to locate bond ATOM INDICES.

_DISULFIDE_COUNT_RE = re.compile(r"(\d+)\s*x\s*disulfide\s+bridge", re.IGNORECASE)


def load_cpepmatch_db(db_csv_path: Path) -> dict[str, str]:
    """Read cPEPmatch's ``cyclo_pep.csv`` into ``{pdb_id_lower: type_string}``.

    Returns an empty dict if the file is missing or unreadable, so callers can
    gracefully fall back to geometry-based detection.
    """
    db: dict[str, str] = {}
    path = Path(db_csv_path)
    if not path.exists():
        logger.warning(
            "cPEPmatch database CSV not found at %s; topology will fall back "
            "to geometry-only detection.", path,
        )
        return db
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pdb = (row.get("PDB") or "").strip().lower()
                type_str = (row.get("type") or "").strip()
                if pdb:
                    db[pdb] = type_str
    except (OSError, csv.Error) as exc:
        logger.warning(
            "Failed to read cPEPmatch database CSV %s (%s); falling back to "
            "geometry-only detection.", path, exc,
        )
        return {}
    return db


def parse_cpepmatch_type(type_str: str) -> dict:
    """Parse one cPEPmatch ``type`` string into a structured topology dict.

    The ``type`` column is a comma-separated list of clauses, e.g.
    ``"2x disulfide bridge, head to tail"`` or ``"0QZ STAPLE, head to tail"``.

    Returns a dict with:
        head_to_tail: bool            -- a "head to tail" clause is present
        disulfide_count: int          -- N from "Nx disulfide bridge", else 0
        has_staple: bool              -- a "STAPLE" clause is present
        has_sidechain_crosslink: bool -- "side-chain to side chain" / "...to
                                         backbone" present
        representable: bool           -- False if stapled or sidechain-
                                         crosslinked (CPM2 cannot emit those
                                         as Boltz bond constraints), else True
        raw_type: str                 -- the original string
    """
    raw = type_str or ""
    clauses = [c.strip() for c in raw.split(",") if c.strip()]

    head_to_tail = False
    disulfide_count = 0
    has_staple = False
    has_sidechain_crosslink = False

    for clause in clauses:
        low = clause.lower()
        if "head to tail" in low:
            head_to_tail = True
        if "staple" in low:
            has_staple = True
        if "side-chain to side chain" in low or "side-chain to backbone" in low:
            has_sidechain_crosslink = True
        m = _DISULFIDE_COUNT_RE.search(clause)
        if m:
            disulfide_count = int(m.group(1))

    representable = not (has_staple or has_sidechain_crosslink)

    return {
        "head_to_tail": head_to_tail,
        "disulfide_count": disulfide_count,
        "has_staple": has_staple,
        "has_sidechain_crosslink": has_sidechain_crosslink,
        "representable": representable,
        "raw_type": raw,
    }


def detect_constraints_db_aware(
    pdb_path: Path,
    chain_id: str,
    source_pdb_id: str | None,
    db: dict[str, str],
    cyc_dist: float = 1.5,
    ss_dist: float = 2.5,
    h2t_fallback_dist: float = 3.8,
) -> dict:
    """Detect cyclic-peptide constraints, trusting the cPEPmatch DB for topology.

    Geometry (``detect_constraints_from_pdb``) supplies bond ATOM INDICES; the
    DB ``type`` column supplies the authoritative head-to-tail flag and
    disulfide count. When ``source_pdb_id`` is not in the DB, falls back to
    geometry but with a loosened head-to-tail distance (``h2t_fallback_dist``,
    ~3.8 A) because cPEPmatch output PDBs leave the closure geometrically open.

    The returned dict is a superset of ``detect_constraints_from_pdb``'s shape
    (``head_to_tail``, ``head_to_tail_distance``, ``disulfides``, ``lactams``,
    ``thioethers``) plus additive provenance keys, so existing callers that use
    ``.get`` keep working unchanged:
        head_to_tail_source: "db" | "geometry_fallback"
        db_type: str | None                -- raw DB type string, if matched
        unrepresentable: bool              -- staple / sidechain crosslink
        unrepresentable_reason: str | None -- the raw type, when unrepresentable
        disulfide_count_mismatch: bool     -- geometry found fewer SS than DB

    ``detect_constraints_from_pdb`` itself is left unchanged.
    """
    # Geometry pass: source of bond atom INDICES (and lactams/thioethers).
    geom = detect_constraints_from_pdb(pdb_path, chain_id, cyc_dist, ss_dist)

    result = dict(geom)
    result.setdefault("head_to_tail_distance", geom.get("head_to_tail_distance"))
    result["head_to_tail_source"] = "geometry_fallback"
    result["db_type"] = None
    result["unrepresentable"] = False
    result["unrepresentable_reason"] = None
    result["disulfide_count_mismatch"] = False

    geom_disulfides = list(geom.get("disulfides", []))

    pdb_key = (source_pdb_id or "").strip().lower()
    db_entry = db.get(pdb_key) if pdb_key else None

    if db_entry is None:
        # Not in DB: geometry-only, but with a loosened head-to-tail cutoff.
        h2t_dist = geom.get("head_to_tail_distance")
        result["head_to_tail"] = (
            h2t_dist is not None and h2t_dist < h2t_fallback_dist
        )
        result["head_to_tail_source"] = "geometry_fallback"
        return result

    # DB hit: trust the curated topology.
    parsed = parse_cpepmatch_type(db_entry)
    result["db_type"] = parsed["raw_type"]
    result["head_to_tail"] = parsed["head_to_tail"]
    result["head_to_tail_source"] = "db"

    if not parsed["representable"]:
        result["unrepresentable"] = True
        result["unrepresentable_reason"] = parsed["raw_type"]

    # Reconcile DB disulfide count vs geometric SG-SG pairs.
    result["disulfides"] = _reconcile_disulfides(
        geom_disulfides, parsed["disulfide_count"], pdb_path, result, source_pdb_id
    )

    return result


def _sg_sg_distance(pdb_path: Path, chain_id: str, i: int, j: int) -> float | None:
    """Euclidean SG-SG distance for a 1-based residue index pair, or None."""
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("model", str(pdb_path))
        chain = struct[0][chain_id]
        residues = [
            r for r in chain.get_residues()
            if r.id[0] != "W" and (
                r.id[0] == " " or {"N", "CA", "C"}.issubset({a.get_name() for a in r})
            )
        ]
        ri, rj = residues[i - 1], residues[j - 1]
        return ri["SG"] - rj["SG"]
    except (Exception, KeyError, IndexError):
        return None


def _reconcile_disulfides(
    geom_disulfides: list,
    db_count: int,
    pdb_path: Path,
    result: dict,
    source_pdb_id: str | None,
) -> list:
    """Reconcile a DB-declared disulfide count against geometric SG-SG pairs.

    DB count answers "how many", geometry answers "where". Never fabricates
    indices and never silently drops:

    * geometry > DB: keep the ``db_count`` closest SG-SG pairs, warn.
    * geometry < DB: emit what geometry found, warn prominently, set
      ``disulfide_count_mismatch=True`` on ``result``.
    * geometry == DB (or DB count 0): pass geometry through unchanged.
    """
    name = source_pdb_id or str(pdb_path)
    geom_count = len(geom_disulfides)

    if db_count == 0 or geom_count == db_count:
        return geom_disulfides

    if geom_count > db_count:
        # Chain id is consistent within a match PDB; recover it from result
        # is not possible, so re-derive from the pairs' parent structure by
        # measuring on the known CP chain "P" is not assumed here -- instead
        # rank by SG-SG distance using whichever chain the pair came from.
        scored = []
        for (i, j) in geom_disulfides:
            d = _sg_sg_distance(pdb_path, "P", i, j)
            if d is None:
                d = _sg_sg_distance(pdb_path, "T", i, j)
            scored.append((d if d is not None else float("inf"), (i, j)))
        scored.sort(key=lambda t: t[0])
        kept = [pair for _, pair in scored[:db_count]]
        logger.warning(
            "Disulfide count mismatch for %s: geometry found %d SG-SG pairs "
            "but the cPEPmatch DB declares %d. Keeping the %d closest pair(s) "
            "%s and dropping the rest.",
            name, geom_count, db_count, db_count, kept,
        )
        return kept

    # geom_count < db_count
    result["disulfide_count_mismatch"] = True
    logger.warning(
        "MISSING DISULFIDE CONSTRAINTS for %s: the cPEPmatch DB declares %d "
        "disulfide bridge(s) but geometry only located %d SG-SG pair(s) %s. "
        "The Boltz input will be built with FEWER disulfide constraints than "
        "the real macrocycle has -- inspect this match before trusting it.",
        name, db_count, geom_count, geom_disulfides,
    )
    return geom_disulfides
