"""
ProteinHunter runner - sequence optimization for cyclic peptide binders.

Executes ProteinHunter refinement in the 'proteinhunter' conda environment via subprocess.
"""

import json
import subprocess
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional


@dataclass
class ProteinHunterConfig:
    """Configuration for ProteinHunter refinement."""

    # Design parameters
    num_designs: int = 3
    num_cycles: int = 5

    # Compute
    gpu_id: int = 0
    seed: int = 42  # seeds torch + numpy + random in proteinhunter_refine.main

    # Thresholds
    iptm_threshold: float = 0.7
    high_iptm_threshold: float = 0.8
    high_plddt_threshold: float = 0.8

    # Boltz model parameters
    diffuse_steps: int = 200
    recycling_steps: int = 3
    diffusion_samples: int = 1  # Per-cycle diffusion samples; >1 reduces ipTM ranking noise
    boltz_model_path: Optional[Path] = None  # None = ~/.boltz/boltz2_conf.ckpt
    ccd_path: Optional[Path] = None  # None = ~/.boltz/mols
    target_msa_path: Optional[str] = None  # Path to .a3m for target chain; None = "empty" (single-seq, not recommended)

    # Sequence design parameters
    temperature: float = 0.1
    omit_AA: str = "C"
    alanine_bias: bool = False
    alanine_bias_start: float = -0.2
    alanine_bias_end: float = 0.0

    # Template parameters
    template_path: Optional[str] = None  # Path to target template PDB/CIF
    template_force: bool = False  # If True, Boltz applies CB-distance guidance to target chain
    template_force_threshold: float = 1.0  # Å; max CB drift allowed when template_force=True

    def to_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization."""
        return {
            "num_designs": self.num_designs,
            "num_cycles": self.num_cycles,
            "gpu_id": self.gpu_id,
            "seed": self.seed,
            "iptm_threshold": self.iptm_threshold,
            "high_iptm_threshold": self.high_iptm_threshold,
            "high_plddt_threshold": self.high_plddt_threshold,
            "diffuse_steps": self.diffuse_steps,
            "recycling_steps": self.recycling_steps,
            "diffusion_samples": self.diffusion_samples,
            "boltz_model_path": str(self.boltz_model_path) if self.boltz_model_path else None,
            "ccd_path": str(self.ccd_path) if self.ccd_path else None,
            "target_msa_path": self.target_msa_path,
            "temperature": self.temperature,
            "omit_AA": self.omit_AA,
            "alanine_bias": self.alanine_bias,
            "alanine_bias_start": self.alanine_bias_start,
            "alanine_bias_end": self.alanine_bias_end,
            "template_path": self.template_path,
            "template_force": self.template_force,
            "template_force_threshold": self.template_force_threshold,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProteinHunterConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {}
        for k, v in d.items():
            if k in valid_fields:
                # Convert path strings to Path objects
                if k in ("boltz_model_path", "ccd_path") and v is not None:
                    filtered[k] = Path(v).expanduser()
                else:
                    filtered[k] = v
        return cls(**filtered)


@dataclass
class ProteinHunterResult:
    """Result from ProteinHunter optimization."""

    name: str
    input_structure: Path
    output_pdb: Path
    input_sequence: str
    optimized_sequence: str
    iptm: float
    plddt: float
    iplddt: float
    cycle: int  # Which optimization cycle produced the best result
    design_num: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "input_structure": str(self.input_structure),
            "output_pdb": str(self.output_pdb),
            "input_sequence": self.input_sequence,
            "optimized_sequence": self.optimized_sequence,
            "iptm": self.iptm,
            "plddt": self.plddt,
            "iplddt": self.iplddt,
            "cycle": self.cycle,
            "design_num": self.design_num,
            **self.metadata,
        }


def run_refine(
    input_structure: Path,
    cp_chain: str,
    target_chain: str,
    output_dir: Path,
    config: Optional[ProteinHunterConfig] = None,
    is_cyclic: bool = False,
    cp_bond_constraints: Optional[list[dict]] = None,
    cp_fixed_cys_positions: Optional[list[int]] = None,
    conda_env: Optional[str] = "proteinhunter",
) -> list[ProteinHunterResult]:
    """
    Run ProteinHunter to refine/optimize a cyclic peptide binder.

    Args:
        input_structure: Input CIF/PDB with CP bound to target (from Boltz2)
        cp_chain: Chain ID of the cyclic peptide (e.g., "P")
        target_chain: Chain ID of the target protein (e.g., "T")
        output_dir: Directory for output files
        config: ProteinHunter configuration
        is_cyclic: Whether the peptide is cyclic (head-to-tail)
        conda_env: Conda environment name with ProteinHunter installed.
            Pass ``None`` to skip the inner ``conda run -n <env>`` prefix
            when the caller has already activated the env (e.g.
            Snakemake's ``conda:`` directive). Defaults to
            ``"proteinhunter"`` so notebook callers stay unchanged.

    Returns:
        List of ProteinHunterResult objects (designs passing iptm_threshold)
    """
    if config is None:
        config = ProteinHunterConfig()

    input_structure = Path(input_structure)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = input_structure.stem

    # Get the path to the refinement script
    script_path = Path(__file__).parent / "proteinhunter_refine.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"ProteinHunter refinement script not found at {script_path}"
        )

    # Get lib path for ProteinHunter
    lib_path = Path(__file__).parent.parent.parent / "lib" / "Protein-Hunter"

    # Write config to JSON file for subprocess communication
    config_file = output_dir / "refine_config.json"
    config_data = {
        "input_structure": str(input_structure),
        "cp_chain": cp_chain,
        "target_chain": target_chain,
        "output_dir": str(output_dir),
        "is_cyclic": is_cyclic,
        # Bond constraints on the CP chain (head-to-tail handled separately via
        # is_cyclic; this list carries SS/lactam/thioether bonds Stage 2 detected).
        "cp_bond_constraints": cp_bond_constraints or [],
        # Cys positions to freeze during LigandMPNN sequence redesign so the
        # SS pairs survive every cycle.
        "cp_fixed_cys_positions": cp_fixed_cys_positions or [],
        "lib_path": str(lib_path),
        **config.to_dict(),
    }
    config_file.write_text(json.dumps(config_data, indent=2))

    # Run the refinement script (optionally inside the proteinhunter env).
    bare_cmd = ["python", str(script_path), str(config_file)]
    if conda_env is not None:
        cmd = ["conda", "run", "-n", conda_env, "--no-capture-output", *bare_cmd]
    else:
        cmd = bare_cmd

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ProteinHunter refinement failed for {name}:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # Parse results from output JSON
    results_file = output_dir / "refine_results.json"
    if not results_file.exists():
        return []

    raw_results = json.loads(results_file.read_text())

    return [
        ProteinHunterResult(
            name=r["name"],
            input_structure=input_structure,
            output_pdb=Path(r["output_pdb"]),
            input_sequence=r["input_sequence"],
            optimized_sequence=r["optimized_sequence"],
            iptm=r["iptm"],
            plddt=r["plddt"],
            iplddt=r.get("iplddt", 0.0),
            cycle=r["cycle"],
            design_num=r["design_num"],
            metadata=r.get("metadata", {}),
        )
        for r in raw_results
    ]


def load_results(
    output_dir: Path,
    input_structure: Optional[Path] = None,
) -> list[ProteinHunterResult]:
    """
    Load ProteinHunterResults from previously computed refinements.

    Use this to resume the pipeline without re-running ProteinHunter.

    Args:
        output_dir: Directory containing refinement outputs
        input_structure: Optional input structure path (for metadata)

    Returns:
        List of ProteinHunterResult objects
    """
    output_dir = Path(output_dir)

    results_file = output_dir / "refine_results.json"
    if not results_file.exists():
        return []

    raw_results = json.loads(results_file.read_text())

    return [
        ProteinHunterResult(
            name=r["name"],
            input_structure=Path(r.get("input_structure", input_structure or "")),
            output_pdb=Path(r["output_pdb"]),
            input_sequence=r["input_sequence"],
            optimized_sequence=r["optimized_sequence"],
            iptm=r["iptm"],
            plddt=r["plddt"],
            iplddt=r.get("iplddt", 0.0),
            cycle=r["cycle"],
            design_num=r["design_num"],
            metadata=r.get("metadata", {}),
        )
        for r in raw_results
    ]
