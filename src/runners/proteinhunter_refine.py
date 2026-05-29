#!/usr/bin/env python
"""
ProteinHunter refinement script - runs in proteinhunter conda environment.

=============================================================================
SOURCE: PIPELINE/lib/Protein-Hunter/refiner_boltz.ipynb (cells 1-4)
MODIFIED:
  - Converted from Jupyter notebook to standalone script
  - Reads config from JSON file instead of inline parameters
  - Writes results to JSON file for subprocess communication
  - Added iPLDDT tracking to output metrics
  - Uses chain IDs from config (cp_chain, target_chain) instead of hardcoded A/B
GLUE:
  - main() function to handle CLI invocation
  - JSON config parsing and result serialization
=============================================================================

Usage: python proteinhunter_refine.py config.json
"""

import contextlib
import copy
import io
import json
import os
import sys

# =============================================================================
# SOURCE: PIPELINE/lib/Protein-Hunter/refiner_boltz.ipynb (cell 1)
# MODIFIED: Imports moved here, warnings suppression added
# =============================================================================
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.*",
    category=UserWarning,
)

import numpy as np
import torch


def setup_paths(lib_path: str):
    """Add ProteinHunter library to path."""
    lib_path = Path(lib_path)
    if str(lib_path) not in sys.path:
        sys.path.insert(0, str(lib_path))


def import_proteinhunter_modules():
    """Import ProteinHunter modules after path is set up."""
    # =============================================================================
    # SOURCE: PIPELINE/lib/Protein-Hunter/refiner_boltz.ipynb (cell 1)
    # =============================================================================
    from boltz_ph.constants import CHAIN_TO_NUMBER
    from boltz_ph.model_utils import (
        clean_memory,
        design_sequence,
        extract_sequence_from_structure,
        get_boltz_model,
        load_canonicals,
        run_prediction,
        save_pdb,
        shallow_copy_tensor_dict,
    )
    from LigandMPNN.wrapper import LigandMPNNWrapper
    return {
        "LigandMPNNWrapper": LigandMPNNWrapper,
        "CHAIN_TO_NUMBER": CHAIN_TO_NUMBER,
        "extract_sequence_from_structure": extract_sequence_from_structure,
        "clean_memory": clean_memory,
        "design_sequence": design_sequence,
        "get_boltz_model": get_boltz_model,
        "load_canonicals": load_canonicals,
        "run_prediction": run_prediction,
        "save_pdb": save_pdb,
        "shallow_copy_tensor_dict": shallow_copy_tensor_dict,
    }


# =============================================================================
# SOURCE: PIPELINE/lib/Protein-Hunter/refiner_boltz.ipynb (cell 4)
# Helper functions for metric extraction
# =============================================================================
def _extract_ca_coords(path: str, chain_id: str):
    """Return (N, 3) ndarray of CA coordinates for the given chain.

    Falls back to the chain with the most CA atoms if the named chain is absent.
    Handles both PDB and mmCIF via gemmi.
    """
    import gemmi
    st = gemmi.read_structure(path)
    if len(st) == 0:
        return np.empty((0, 3))
    model = st[0]

    def chain_cas(chain):
        return [atom.pos for res in chain for atom in res if atom.name == "CA"]

    positions = []
    for ch in model:
        if ch.name == chain_id:
            positions = chain_cas(ch)
            break
    if not positions:
        # fallback: pick the chain with the most CAs
        best = max(model, key=lambda c: len(chain_cas(c)), default=None)
        if best is not None:
            positions = chain_cas(best)
    return np.array([[p.x, p.y, p.z] for p in positions])


def _kabsch_rmsd(P, Q):
    """RMSD between two N×3 coordinate sets after optimal superposition."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    diff = Pc @ R.T - Qc
    return float(np.sqrt((diff ** 2).sum() / len(P)))


def compute_target_ca_rmsd(refined_path: str, reference_path: str, target_chain: str) -> float:
    """CA-RMSD of the target chain between refined output and reference structure.

    Diagnostic for template compliance: a successful template (soft or forced) should
    keep the main protein body close to the input, so this value stays low.
    """
    ref = _extract_ca_coords(reference_path, target_chain)
    mob = _extract_ca_coords(refined_path, target_chain)
    if len(ref) == 0 or len(mob) == 0:
        return float("nan")
    if len(ref) != len(mob):
        n = min(len(ref), len(mob))
        print(f"  WARN: target length mismatch (ref={len(ref)}, refined={len(mob)}); truncating to {n}")
        ref, mob = ref[:n], mob[:n]
    return _kabsch_rmsd(ref, mob)


def compute_iptm(pair_chains, ref_chain_idx):
    """Compute interface pTM from pair chain scores."""
    if len(pair_chains) > 1:
        vals = [
            (
                pair_chains[ref_chain_idx][i].detach().cpu().numpy()
                + pair_chains[i][ref_chain_idx].detach().cpu().numpy()
            ) / 2.0
            for i in range(len(pair_chains)) if i != ref_chain_idx
        ]
        return float(np.mean(vals) if vals else 0.0)
    return 0.0


def get_float(item, key, default_val):
    """Safely extract float from output dict."""
    return float(item.get(key, torch.tensor([default_val])).detach().cpu().numpy()[0])


def run_refinement(config: dict) -> list[dict]:
    """
    Run ProteinHunter refinement loop.

    =============================================================================
    SOURCE: PIPELINE/lib/Protein-Hunter/refiner_boltz.ipynb (cells 3-4)
    MODIFIED:
      - Accepts config dict instead of hardcoded parameters
      - Uses configurable chain IDs (cp_chain, target_chain)
      - Returns list of result dicts instead of printing
      - Tracks iPLDDT in addition to iPTM and pLDDT
    =============================================================================
    """
    # Setup paths and import modules
    setup_paths(config["lib_path"])
    modules = import_proteinhunter_modules()

    # Unpack modules
    LigandMPNNWrapper = modules["LigandMPNNWrapper"]
    CHAIN_TO_NUMBER = modules["CHAIN_TO_NUMBER"]
    extract_sequence_from_structure = modules["extract_sequence_from_structure"]
    clean_memory = modules["clean_memory"]
    design_sequence = modules["design_sequence"]
    get_boltz_model = modules["get_boltz_model"]
    load_canonicals = modules["load_canonicals"]
    run_prediction = modules["run_prediction"]
    save_pdb = modules["save_pdb"]
    shallow_copy_tensor_dict = modules["shallow_copy_tensor_dict"]

    # Extract config values
    input_structure = Path(config["input_structure"])
    output_dir = Path(config["output_dir"])
    cp_chain = config["cp_chain"]
    target_chain = config["target_chain"]
    is_cyclic = config.get("is_cyclic", False)

    num_designs = config.get("num_designs", 3)
    num_cycles = config.get("num_cycles", 5)
    gpu_id = config.get("gpu_id", 0)
    iptm_threshold = config.get("iptm_threshold", 0.7)

    # Seed RNGs for reproducible diffusion + sequence sampling. Logged in the
    # run manifest under proteinhunter_seed.
    seed = int(config.get("seed", 42))
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    diffuse_steps = config.get("diffuse_steps", 200)
    recycling_steps = config.get("recycling_steps", 3)
    diffusion_samples = int(config.get("diffusion_samples", 1))
    target_msa_path = config.get("target_msa_path")
    temperature = config.get("temperature", 0.1)
    omit_AA = config.get("omit_AA", "C")
    alanine_bias = config.get("alanine_bias", False)
    alanine_bias_start = config.get("alanine_bias_start", -0.2)
    alanine_bias_end = config.get("alanine_bias_end", 0.0)
    template_path = config.get("template_path")
    template_force = bool(config.get("template_force", False))
    template_force_threshold = float(config.get("template_force_threshold", 1.0))

    # Paths with defaults
    boltz_model_path = config.get("boltz_model_path")
    if boltz_model_path is None:
        boltz_model_path = os.path.expanduser("~/.boltz/boltz2_conf.ckpt")

    ccd_path = config.get("ccd_path")
    if ccd_path is None:
        ccd_path = Path(os.path.expanduser("~/.boltz/mols"))
    else:
        ccd_path = Path(ccd_path)

    lib_path = Path(config["lib_path"])

    # =============================================================================
    # SOURCE: PIPELINE/lib/Protein-Hunter/refiner_boltz.ipynb (cell 3)
    # Model initialization
    # =============================================================================
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Initializing ProteinHunter on {device}...")

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": diffuse_steps,
        "diffusion_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
        "max_parallel_samples": max(1, diffusion_samples),
    }

    ccd_lib = load_canonicals(str(ccd_path))
    boltz_model = get_boltz_model(
        checkpoint=str(boltz_model_path),
        predict_args=predict_args,
        device=device,
        model_version="boltz2",
        no_potentials=True,
        grad_enabled=False,
    )
    designer = LigandMPNNWrapper(str(lib_path / "LigandMPNN" / "run.py"))

    # Get sequences from input structure
    binder_seq = extract_sequence_from_structure(str(input_structure), cp_chain)
    target_seq = extract_sequence_from_structure(str(input_structure), target_chain)
    binder_length = len(binder_seq)

    print(f"Input: {input_structure.name}")
    print(f"  CP chain {cp_chain}: {binder_length} residues")
    print(f"  Target chain {target_chain}: {len(target_seq)} residues")
    print(f"  Cyclic: {is_cyclic}")

    # Build base data dict.
    # PH's vendored Boltz wrapper bypasses Boltz's own MSA pre-processing and
    # calls np.load(msa_id) directly (boltz_ph/model_utils.py:319), so the
    # path we hand it must be a pre-processed .npz (built via Boltz's
    # parse_csv|parse_a3m + MSA.dump). If the config points at a .csv/.a3m,
    # use the sibling .npz; otherwise fall back to single-sequence.
    target_msa_value = "empty"
    if target_msa_path:
        p = Path(target_msa_path)
        npz = p if p.suffix == ".npz" else p.with_suffix(".npz")
        if npz.exists():
            target_msa_value = str(npz)
            print(f"  Using target MSA (npz): {target_msa_value}")
        else:
            print(f"  WARN: target_msa_path={target_msa_path} has no sibling {npz.name}; "
                  "falling back to single-sequence (run scripts/build_msa_npz.py)")
    else:
        print("  Target MSA: empty (single-sequence; consider running scripts/prefetch_target_msa.py)")
    data = {
        "sequences": [
            {"protein": {"id": [target_chain], "sequence": target_seq, "msa": target_msa_value}},
            {"protein": {"id": [cp_chain], "sequence": binder_seq, "msa": "empty", "cyclic": is_cyclic}},
        ]
    }

    # Forward Stage 2's bond constraints (SS, lactam, thioether) to PH's
    # internal Boltz call. PH's vendored Boltz wrapper goes through
    # `parse_boltz_schema(data, ...)`, the same parser Stage 2's YAML targets,
    # so the bond list passes through unchanged.
    cp_bond_constraints = config.get("cp_bond_constraints") or []
    if cp_bond_constraints:
        data["constraints"] = cp_bond_constraints
        print(f"  Forwarding {len(cp_bond_constraints)} CP bond constraint(s) "
              f"to Boltz (SS / lactam / thioether)")

    # Handle template for target chain (provides structural constraints during refinement)
    # Uses "cif" key to avoid parse_pdb ignore_ligands bug in Protein-Hunter.
    # When template_force=True, Boltz adds a CB-distance guidance potential during sampling
    # (TemplateReferencePotential). Otherwise the template acts as a soft prior only.
    if template_path and Path(template_path).exists():
        template_block = {
            "cif": str(template_path),
            "chain_id": target_chain,
        }
        if template_force:
            template_block["force"] = True
            template_block["threshold"] = template_force_threshold
            print(f"  Using FORCED template: {template_path} "
                  f"(CB threshold={template_force_threshold}Å)")
        else:
            print(f"  Using template (soft prior): {template_path}")
        data["templates"] = [template_block]

    # =============================================================================
    # SOURCE: PIPELINE/lib/Protein-Hunter/refiner_boltz.ipynb (cell 4)
    # Design and optimization loop
    # MODIFIED: Returns results list, tracks iPLDDT, per-cycle target CA-RMSD vs template
    # =============================================================================
    all_results = []

    # Parse template CAs once and reuse across every cycle/design for RMSD diagnostic.
    template_ref_coords = None
    if template_path and Path(template_path).exists():
        try:
            template_ref_coords = _extract_ca_coords(str(template_path), target_chain)
            if len(template_ref_coords) == 0:
                print(f"  WARN: no CA atoms found in template for chain {target_chain}; RMSD disabled")
                template_ref_coords = None
        except Exception as e:
            print(f"  WARN: template parse failed ({e}); RMSD disabled")
            template_ref_coords = None

    def rmsd_to_template(pdb_path) -> float:
        if template_ref_coords is None:
            return float("nan")
        try:
            mob = _extract_ca_coords(str(pdb_path), target_chain)
            if len(mob) == 0:
                return float("nan")
            n = min(len(template_ref_coords), len(mob))
            return _kabsch_rmsd(template_ref_coords[:n], mob[:n])
        except Exception as e:
            print(f"  WARN: RMSD failed for {Path(pdb_path).name}: {e}")
            return float("nan")

    for design_id in range(num_designs):
        print(f"\n{'='*54}")
        print(f"Design {design_id + 1}/{num_designs}")
        print(f"{'='*54}")

        design_dir = output_dir / f"design_{design_id}"
        design_dir.mkdir(parents=True, exist_ok=True)

        data_cp = copy.deepcopy(data)
        best_iptm, best_seq, best_pdb = float("-inf"), None, None
        best_cycle, best_plddt, best_iplddt = -1, 0.0, 0.0
        per_cycle_rmsd = {}  # cycle_idx -> target CA-RMSD vs template (Å)
        # Per-cycle convergence trace (ph_convergence experiment, 2026-05-29).
        # Additive to the existing per-cycle RMSD logging: lets us tell a basin
        # (per-cycle ipTM plateaus + proposed sequence Hamming distance -> 0)
        # from a greedy ipTM hill-climb that keeps rising to the last cycle.
        # cycle_idx -> dict(proposed_seq, accepted_seq, iptm, plddt, iplddt,
        #                   alanine_pct, accepted). Cycle 0 carries the input
        # fold (no LigandMPNN proposal) with proposed_seq == input sequence.
        per_cycle_trace = {}

        # Cycle 0: initial prediction
        with contextlib.redirect_stdout(io.StringIO()):
            output, structure_pred = run_prediction(
                data_cp, cp_chain,
                boltz_model=boltz_model, ccd_lib=ccd_lib,
                ccd_path=ccd_path, device=device,
            )
        pdb_file = design_dir / "cycle_0.pdb"
        save_pdb(structure_pred, output["coords"],
                 output["plddt"].detach().cpu().numpy()[0], str(pdb_file))
        per_cycle_rmsd[0] = rmsd_to_template(pdb_file)

        binder_idx = CHAIN_TO_NUMBER.get(cp_chain, 0)
        cycle_0_iptm = compute_iptm(output["pair_chains_iptm"], binder_idx)
        cycle_0_plddt = get_float(output, "complex_plddt", 0.0)
        cycle_0_iplddt = get_float(output, "complex_iplddt", 0.0)
        rmsd0 = per_cycle_rmsd[0]
        rmsd0_str = f"{rmsd0:.2f}Å" if rmsd0 == rmsd0 else "n/a"
        print(f"  Cycle 0: ipTM={cycle_0_iptm:.3f}, pLDDT={cycle_0_plddt:.2f}, "
              f"iPLDDT={cycle_0_iplddt:.2f}, target_RMSD={rmsd0_str}")

        # Cycle 0 has no LigandMPNN proposal; it is the fold of the input
        # sequence. accepted=False because best is only set inside the loop
        # (cycles 1..N), keeping the "best-so-far" semantics consistent.
        per_cycle_trace[0] = {
            "proposed_seq": binder_seq,
            "accepted_seq": None,
            "iptm": cycle_0_iptm,
            "plddt": cycle_0_plddt,
            "iplddt": cycle_0_iplddt,
            "alanine_pct": (binder_seq.count("A") / binder_length) if binder_length else 0.0,
            "accepted": False,
        }

        # Cycles 1-N: optimization
        for cycle in range(num_cycles):
            cycle_norm = cycle / max(num_cycles - 1, 1)
            alpha = alanine_bias_start - cycle_norm * (alanine_bias_start - alanine_bias_end)

            design_kwargs = {
                "pdb_file": str(pdb_file),
                "temperature": temperature,
                "chains_to_design": cp_chain,
                "omit_AA": f"{omit_AA},P" if cycle == 0 else omit_AA,
            }
            if alanine_bias:
                design_kwargs["bias_AA"] = f"A:{alpha}"

            # Pin SS-forming Cys positions so LigandMPNN can't mutate them away.
            cp_fixed_cys = config.get("cp_fixed_cys_positions") or []
            if cp_fixed_cys:
                # LigandMPNN --fixed_residues expects "C10 C14"-style tokens
                # using the *PDB chain* + residue number, where the chain is
                # the post-PH-renaming CP chain id (here `cp_chain`, "P").
                fixed_tokens = " ".join(f"{cp_chain}{i}" for i in cp_fixed_cys)
                # Bypass design_sequence's hardcoded extra_args and call the
                # designer directly so we can carry --fixed_residues through.
                seq, _logits = designer.run(
                    model_type="soluble_mpnn",
                    pdb_path=str(pdb_file),
                    seed=111,
                    chains_to_design=cp_chain,
                    bias_AA=design_kwargs.get("bias_AA", ""),
                    omit_AA=design_kwargs["omit_AA"],
                    extra_args={
                        "--temperature": temperature,
                        "--batch_size": 1,
                        "--fixed_residues": fixed_tokens,
                    },
                )
                seq_str = seq[0]
            else:
                seq_str, _ = design_sequence(designer, "soluble_mpnn", **design_kwargs)
            seq = seq_str.split(":")[binder_idx]
            alanine_pct = seq.count("A") / binder_length if binder_length else 0

            # Update sequence in data dict
            for seq_entry in data_cp["sequences"]:
                if "protein" in seq_entry and cp_chain in seq_entry["protein"]["id"]:
                    seq_entry["protein"]["sequence"] = seq
                    break

            with contextlib.redirect_stdout(io.StringIO()):
                output, structure_pred = run_prediction(
                    data_cp, cp_chain, seq=seq,
                    boltz_model=boltz_model, ccd_lib=ccd_lib,
                    ccd_path=ccd_path, device=device,
                )

            current_iptm = compute_iptm(output["pair_chains_iptm"], binder_idx)
            current_plddt = get_float(output, "complex_plddt", 0.0)
            current_iplddt = get_float(output, "complex_iplddt", 0.0)

            # Track best (with alanine filter). `accepted` is the same
            # acceptance test used to advance the best (sequence, structure)
            # pair; we log it per cycle for the convergence experiment.
            accepted = alanine_pct <= 0.20 and current_iptm > best_iptm
            if accepted:
                best_iptm = current_iptm
                best_plddt = current_plddt
                best_iplddt = current_iplddt
                best_seq = seq
                best_cycle = cycle + 1
                best_pdb = design_dir / "best.pdb"
                save_pdb(structure_pred, output["coords"],
                         output["plddt"].detach().cpu().numpy()[0], str(best_pdb))

            # Per-cycle convergence trace (additive; see init above). Records
            # the LigandMPNN-proposed sequence, the symmetrized pair-chain ipTM
            # used for acceptance, alanine %, the accept/reject flag, and the
            # best-so-far accepted sequence after this cycle's decision.
            per_cycle_trace[cycle + 1] = {
                "proposed_seq": seq,
                "accepted_seq": best_seq,
                "iptm": current_iptm,
                "plddt": current_plddt,
                "iplddt": current_iplddt,
                "alanine_pct": alanine_pct,
                "accepted": bool(accepted),
            }

            pdb_file = design_dir / f"cycle_{cycle + 1}.pdb"
            save_pdb(structure_pred, output["coords"],
                     output["plddt"].detach().cpu().numpy()[0], str(pdb_file))
            per_cycle_rmsd[cycle + 1] = rmsd_to_template(pdb_file)
            rmsd_c = per_cycle_rmsd[cycle + 1]
            rmsd_c_str = f"{rmsd_c:.2f}Å" if rmsd_c == rmsd_c else "n/a"

            print(f"  Cycle {cycle + 1}: ipTM={current_iptm:.3f}, pLDDT={current_plddt:.2f}, "
                  f"iPLDDT={current_iplddt:.2f}, Ala={alanine_pct*100:.0f}%, target_RMSD={rmsd_c_str}")

            clean_memory()

        # JSON keys must be strings; preserve cycle order for easy plotting.
        per_cycle_rmsd_serialized = {
            str(k): per_cycle_rmsd[k] for k in sorted(per_cycle_rmsd)
        }
        per_cycle_trace_serialized = {
            str(k): per_cycle_trace[k] for k in sorted(per_cycle_trace)
        }

        # Always persist the full per-cycle convergence trace to disk,
        # independent of the iptm_threshold gate below. A design that never
        # crosses threshold is exactly the wandering / no-basin case we must
        # not silently drop. The parse script reads these per-design files.
        trace_file = design_dir / "per_cycle_trace.json"
        trace_file.write_text(json.dumps({
            "name": f"{input_structure.stem}_design{design_id}",
            "design_num": design_id,
            "input_sequence": binder_seq,
            "best_iptm": (best_iptm if best_iptm != float("-inf") else float("nan")),
            "best_cycle": best_cycle,
            "best_seq": best_seq,
            "passed_threshold": bool(best_iptm >= iptm_threshold and best_pdb is not None),
            "target_ca_rmsd_per_cycle": per_cycle_rmsd_serialized,
            "per_cycle_trace": per_cycle_trace_serialized,
        }, indent=2))

        # Record result if passes threshold
        if best_iptm >= iptm_threshold and best_pdb is not None:
            best_rmsd = per_cycle_rmsd.get(best_cycle, float("nan"))

            result = {
                "name": f"{input_structure.stem}_design{design_id}",
                "input_structure": str(input_structure),
                "output_pdb": str(best_pdb),
                "input_sequence": binder_seq,
                "optimized_sequence": best_seq,
                "iptm": best_iptm,
                "plddt": best_plddt,
                "iplddt": best_iplddt,
                "cycle": best_cycle,
                "design_num": design_id,
                "metadata": {
                    "target_ca_rmsd": best_rmsd,
                    "target_ca_rmsd_per_cycle": per_cycle_rmsd_serialized,
                    # Additive: per-cycle ipTM + sequence convergence trace.
                    "per_cycle_trace": per_cycle_trace_serialized,
                },
            }
            all_results.append(result)
            rmsd_str = f"{best_rmsd:.2f}Å" if best_rmsd == best_rmsd else "n/a"
            print(f"  PASS: ipTM={best_iptm:.3f} at cycle {best_cycle}, target_CA_RMSD={rmsd_str}")
        else:
            print(f"  FAIL: Best ipTM={best_iptm:.3f} below threshold {iptm_threshold}")

    return all_results


# =============================================================================
# GLUE: main() function to handle CLI invocation
# =============================================================================
def main():
    """CLI entry point for ProteinHunter refinement."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} config.json", file=sys.stderr)
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Load config
    config = json.loads(config_path.read_text())

    # Ensure output directory exists
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run refinement
    try:
        results = run_refinement(config)
    except Exception as e:
        print(f"Error during refinement: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Write results to JSON
    results_file = output_dir / "refine_results.json"
    results_file.write_text(json.dumps(results, indent=2))

    print(f"\nCompleted: {len(results)} designs passed threshold")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
