#!/usr/bin/env python
"""Per-match Boltz YAML builder (called by Snakemake rule build_boltz_yaml).

CLI: build_boltz_yaml_match.py <pipeline_root> <config_name> <run_root> <match_pdb> <out_yaml>
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def _source_pdb_id_from_match(match_pdb: Path) -> str:
    """Derive the cPEPmatch source PDB id from a match PDB filename.

    Match PDBs are named ``matchN_<pdb>.pdb`` or ``matchN_<pdb>-NotMutated.pdb``.
    Strip the extension and a trailing ``-NotMutated``, then take the substring
    after the last underscore.
    """
    stem = match_pdb.stem  # drops .pdb
    if stem.endswith("-NotMutated"):
        stem = stem[: -len("-NotMutated")]
    return stem.rsplit("_", 1)[-1]


def main() -> int:
    pipeline_root, config_name, run_root, match_pdb, out_yaml = (Path(p) for p in sys.argv[1:6])
    sys.path.insert(0, str(pipeline_root / "src"))
    from cpm2.boltz import build_boltz_yaml, write_boltz_yaml
    from cpm2.config_loader import load_config, resolve_msa_cache_path
    from cpm2.utils.constraints import (
        detect_constraints_db_aware,
        detect_constraints_from_pdb,
        load_cpepmatch_db,
    )
    from cpm2.utils.pdb_utils import get_modifications, get_sequence

    config = load_config(pipeline_root, str(config_name), run_root=run_root)
    processed = run_root / "intermediate" / "0_import" / "processed.pdb"
    target_seq = get_sequence(processed, "T")
    if not config["proteinhunter"].get("target_msa_path"):
        msa = resolve_msa_cache_path(pipeline_root, target_seq)
        msa.parent.mkdir(parents=True, exist_ok=True)
        config["proteinhunter"]["target_msa_path"] = str(msa)

    cyc_dist = config["boltz"].get("cyclization_distance", 1.5)
    ss_dist = config["boltz"].get("disulfide_distance", 2.5)
    # Target chain "T" is the real input protein, not a cPEPmatch DB entry, so
    # it stays on pure geometric detection.
    target_constraints = detect_constraints_from_pdb(processed, "T", cyc_dist, ss_dist)

    cp_seq = get_sequence(match_pdb, "P")
    mods = get_modifications(match_pdb, "P")

    # CP chain "P": trust the cPEPmatch DB ``type`` column for topology
    # (head-to-tail flag, disulfide count); geometry only locates bond indices.
    source_pdb_id = _source_pdb_id_from_match(match_pdb)
    db = load_cpepmatch_db(
        pipeline_root / "lib" / "cPEPmatch" / "database" / "cyclo_pep.csv"
    )
    cp_constraints = detect_constraints_db_aware(
        match_pdb, "P", source_pdb_id, db, cyc_dist, ss_dist,
    )
    if cp_constraints.get("unrepresentable"):
        logging.warning(
            "UNREPRESENTABLE TOPOLOGY: match %s (source PDB %s) has cPEPmatch "
            "type %r (staple / side-chain crosslink), which CPM2 cannot emit "
            "as Boltz bond constraints. It will be built as a plain peptide "
            "WITHOUT those crosslinks -- treat downstream results with caution.",
            match_pdb.name, source_pdb_id,
            cp_constraints.get("unrepresentable_reason"),
        )

    data = build_boltz_yaml(
        target_seq, cp_seq, mods, cp_constraints, target_constraints,
        target_msa_path=config["proteinhunter"].get("target_msa_path"),
    )
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    write_boltz_yaml(out_yaml, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
