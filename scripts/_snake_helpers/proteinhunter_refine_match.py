#!/usr/bin/env python
"""Per-match ProteinHunter refinement (called by Snakemake rule proteinhunter_refine).

CLI: proteinhunter_refine_match.py <pipeline_root> <config_name> <run_root> <match_name>
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml as _yaml


def main() -> int:
    pipeline_root = Path(sys.argv[1])
    config_name = sys.argv[2]
    run_root = Path(sys.argv[3])
    match_name = sys.argv[4]
    sys.path.insert(0, str(pipeline_root / "src"))

    from cpm2.config_loader import load_config
    from cpm2.runners import boltz as boltz_runner_mod
    from cpm2.runners import proteinhunter
    from cpm2.runners.proteinhunter import ProteinHunterConfig

    config = load_config(pipeline_root, config_name, run_root=run_root)
    intermediate = run_root / "intermediate"
    yaml_dir = intermediate / "2_boltz" / "yaml_input"
    renamed = intermediate / "1_cpepmatch_renamed"
    input_pdb = renamed / f"{match_name}.pdb"

    # Load the single Boltz result for this match.
    results = boltz_runner_mod.load_results(
        output_dir=intermediate / "2_boltz" / "predictions",
        input_pdbs={match_name: input_pdb} if input_pdb.exists() else None,
        cp_chain="P",
        output_format=config["boltz"].get("output_format", "mmcif"),
        yaml_dir=yaml_dir,
    )
    match_result = next((r for r in results if r.name == match_name), None)
    if match_result is None:
        raise RuntimeError(f"No Boltz result found for match {match_name}")

    # Read the full topology block (head-to-tail flag + bond constraints) from
    # the Stage 2 YAML so PH's internal Boltz call gets the same constraint
    # info Stage 2 built. Previously only the cyclic bool was forwarded,
    # silently linearising every disulfide / lactam / thioether scaffold
    # ("cyclization-dropped" issue, 2026-05-13 journal).
    is_cyclic = False
    cp_bond_constraints: list[dict] = []
    yp = yaml_dir / f"{match_name}.yaml"
    if yp.exists():
        yaml_doc = _yaml.safe_load(yp.read_text()) or {}
        for entry in yaml_doc.get("sequences", []):
            if "protein" in entry and entry["protein"].get("id") == "P":
                is_cyclic = entry["protein"].get("cyclic", False)
                break
        # Boltz schema format: {"bond": {"atom1": [chain, resnum, atom], "atom2": [...]}}.
        # Keep only bonds whose endpoints live on the P (peptide) chain — the
        # target chain T is held to its experimental structure by templates,
        # we don't want to re-impose its native crosslinks.
        for c in yaml_doc.get("constraints", []) or []:
            b = c.get("bond")
            if not b:
                continue
            a1, a2 = b.get("atom1") or [], b.get("atom2") or []
            if len(a1) == 3 and len(a2) == 3 and a1[0] == "P" and a2[0] == "P":
                cp_bond_constraints.append(c)

    # Cys positions implicated in disulfide pairs — pin these in LigandMPNN
    # via --fixed_residues so sequence redesign can't mutate the SS-forming
    # cysteines away.
    cp_fixed_cys: list[int] = []
    for c in cp_bond_constraints:
        a1, a2 = c["bond"]["atom1"], c["bond"]["atom2"]
        if a1[2] == "SG" and a2[2] == "SG":
            cp_fixed_cys.extend([int(a1[1]), int(a2[1])])
    cp_fixed_cys = sorted(set(cp_fixed_cys))

    ph_config = ProteinHunterConfig.from_dict(config["proteinhunter"])
    ph_out = intermediate / "3_proteinhunter" / match_name
    ph_out.mkdir(parents=True, exist_ok=True)
    # Use the runner's inner `conda run -n proteinhunter` activation. (We
    # don't invoke snakemake with --use-conda because reusing the named
    # env is preferable to letting snakemake recreate it.)
    proteinhunter.run_refine(
        input_structure=match_result.output_structure,
        cp_chain="P",
        target_chain="T",
        output_dir=ph_out,
        config=ph_config,
        is_cyclic=is_cyclic,
        cp_bond_constraints=cp_bond_constraints,
        cp_fixed_cys_positions=cp_fixed_cys,
        conda_env="proteinhunter",
    )
    # run_refine writes refine_results.json into ph_out; if PH found nothing
    # passing thresholds the file may be absent. Ensure the file exists so
    # the Snakemake output contract is satisfied (downstream collect handles
    # empty lists).
    rj = ph_out / "refine_results.json"
    if not rj.exists():
        rj.write_text("[]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
