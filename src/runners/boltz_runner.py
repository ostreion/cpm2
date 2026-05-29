"""
Boltz2 runner - structure prediction and validation.

Executes `boltz predict` on YAML input files in the 'boltz' conda environment.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

from Bio.PDB import MMCIFParser, PDBParser, Superimposer

# --- Data Classes ---


@dataclass
class BoltzPredictConfig:
    """Configuration for boltz predict CLI options."""

    # Compute
    devices: int = 1
    accelerator: str = "gpu"  # gpu, cpu, tpu
    num_workers: int = 2
    preprocessing_threads: Optional[int] = None  # None = cpu_count()

    # Model / Sampling
    cache: Optional[Path] = None  # None = ~/.boltz
    checkpoint: Optional[Path] = None
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1
    max_parallel_samples: int = 5
    step_scale: float = 1.638
    method: Optional[str] = None
    seed: Optional[int] = None  # RNG seed for diffusion; None = unset (Boltz default)

    # Output
    output_format: str = "mmcif"  # mmcif, pdb
    override: bool = False
    write_full_pae: bool = False
    write_full_pde: bool = False

    # MSA
    use_msa_server: bool = True
    msa_server_url: str = "https://api.colabfold.com"
    msa_pairing_strategy: str = "greedy"  # greedy, complete
    max_msa_seqs: int = 8192
    subsample_msa: bool = False
    num_subsampled_msa: int = 1024

    # Inference
    use_potentials: bool = False
    no_kernels: bool = False

    # Affinity
    affinity_mw_correction: bool = False
    sampling_steps_affinity: int = 200
    diffusion_samples_affinity: int = 5
    affinity_checkpoint: Optional[Path] = None

    def to_cli_args(self) -> list[str]:
        """Convert config to CLI argument list."""
        args = []

        # Integer/float/string options
        value_options = {
            "devices": "--devices",
            "accelerator": "--accelerator",
            "num_workers": "--num_workers",
            "preprocessing_threads": "--preprocessing-threads",
            "recycling_steps": "--recycling_steps",
            "sampling_steps": "--sampling_steps",
            "diffusion_samples": "--diffusion_samples",
            "max_parallel_samples": "--max_parallel_samples",
            "step_scale": "--step_scale",
            "seed": "--seed",
            "output_format": "--output_format",
            "method": "--method",
            "cache": "--cache",
            "checkpoint": "--checkpoint",
            "msa_server_url": "--msa_server_url",
            "msa_pairing_strategy": "--msa_pairing_strategy",
            "max_msa_seqs": "--max_msa_seqs",
            "num_subsampled_msa": "--num_subsampled_msa",
            "sampling_steps_affinity": "--sampling_steps_affinity",
            "diffusion_samples_affinity": "--diffusion_samples_affinity",
            "affinity_checkpoint": "--affinity_checkpoint",
        }

        for attr, flag in value_options.items():
            val = getattr(self, attr)
            if val is not None:
                args.extend([flag, str(val)])

        # Boolean flags (only added when True)
        flag_options = {
            "override": "--override",
            "write_full_pae": "--write_full_pae",
            "write_full_pde": "--write_full_pde",
            "use_msa_server": "--use_msa_server",
            "subsample_msa": "--subsample_msa",
            "use_potentials": "--use_potentials",
            "no_kernels": "--no_kernels",
            "affinity_mw_correction": "--affinity_mw_correction",
        }

        for attr, flag in flag_options.items():
            if getattr(self, attr):
                args.append(flag)

        return args

    @classmethod
    def from_dict(cls, d: dict) -> "BoltzPredictConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class BoltzResult:
    """Result from a Boltz2 prediction."""

    name: str
    input_yaml: Path
    output_structure: Path  # CIF or PDB depending on output_format
    output_format: str  # "mmcif" or "pdb"

    # Core confidence metrics
    confidence_score: float
    iptm: float
    ptm: float
    plddt: float  # complex_plddt

    # Extended confidence metrics
    ligand_iptm: float = 0.0
    protein_iptm: float = 0.0
    complex_iplddt: float = 0.0
    complex_pde: float = 0.0
    complex_ipde: float = 0.0
    chains_ptm: dict = field(default_factory=dict)
    pair_chains_iptm: dict = field(default_factory=dict)

    # Structural comparison (requires input_pdb for RMSD)
    input_pdb: Optional[Path] = None
    rmsd_to_input: Optional[float] = None

    # Extra metadata
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "input_yaml": str(self.input_yaml),
            "output_structure": str(self.output_structure),
            "output_format": self.output_format,
            "confidence_score": self.confidence_score,
            "iptm": self.iptm,
            "ptm": self.ptm,
            "plddt": self.plddt,
            "ligand_iptm": self.ligand_iptm,
            "protein_iptm": self.protein_iptm,
            "complex_iplddt": self.complex_iplddt,
            "complex_pde": self.complex_pde,
            "complex_ipde": self.complex_ipde,
            "input_pdb": str(self.input_pdb) if self.input_pdb else None,
            "rmsd_to_input": self.rmsd_to_input,
        }

    @property
    def passed_validation(self) -> bool:
        """Check if structure folds similarly to input."""
        return self.rmsd_to_input is not None and self.rmsd_to_input < 3.0


# --- Output Parsing ---


def parse_confidence_json(confidence_path: Path) -> dict:
    """
    Parse Boltz2 confidence output JSON.

    Returns:
        Dictionary with all confidence metrics
    """
    data = json.loads(confidence_path.read_text())
    return {
        "confidence_score": data.get("confidence_score", 0.0),
        "ptm": data.get("ptm", 0.0),
        "iptm": data.get("iptm", 0.0),
        "ligand_iptm": data.get("ligand_iptm", 0.0),
        "protein_iptm": data.get("protein_iptm", 0.0),
        "plddt": data.get("complex_plddt", 0.0),
        "complex_iplddt": data.get("complex_iplddt", 0.0),
        "complex_pde": data.get("complex_pde", 0.0),
        "complex_ipde": data.get("complex_ipde", 0.0),
        "chains_ptm": data.get("chains_ptm", {}),
        "pair_chains_iptm": data.get("pair_chains_iptm", {}),
    }


def compute_cp_rmsd(
    input_pdb: Path,
    predicted_structure: Path,
    cp_chain: str = "P",
    output_format: str = "mmcif",
) -> Optional[float]:
    """
    Compute CA RMSD between the CP chain in input PDB and predicted structure.

    Args:
        input_pdb: Original input PDB
        predicted_structure: Boltz2 output (CIF or PDB)
        cp_chain: Chain ID of the cyclic peptide
        output_format: "mmcif" or "pdb"

    Returns:
        RMSD in Angstroms, or None if computation fails
    """
    try:
        # Parse input PDB
        pdb_parser = PDBParser(QUIET=True)
        input_struct = pdb_parser.get_structure("input", str(input_pdb))

        # Parse predicted structure
        if output_format == "mmcif":
            pred_parser = MMCIFParser(QUIET=True)
        else:
            pred_parser = PDBParser(QUIET=True)
        pred_struct = pred_parser.get_structure("pred", str(predicted_structure))

        # Get CA atoms from CP chain in input
        input_model = input_struct[0]
        if cp_chain not in input_model:
            return None
        input_cas = [
            atom for atom in input_model[cp_chain].get_atoms()
            if atom.get_name() == "CA"
        ]

        # Get CA atoms from predicted structure
        # Try the same chain ID first, then try all chains
        pred_model = pred_struct[0]
        pred_cas = None

        if cp_chain in pred_model:
            pred_cas = [
                atom for atom in pred_model[cp_chain].get_atoms()
                if atom.get_name() == "CA"
            ]

        # If chain not found by ID, try matching by sequence length
        if not pred_cas:
            for chain in pred_model.get_chains():
                cas = [
                    atom for atom in chain.get_atoms()
                    if atom.get_name() == "CA"
                ]
                if len(cas) == len(input_cas):
                    pred_cas = cas
                    break

        if not pred_cas or len(input_cas) != len(pred_cas):
            return None

        # Superimpose and compute RMSD
        sup = Superimposer()
        sup.set_atoms(input_cas, pred_cas)

        return sup.rms

    except Exception as e:
        warnings.warn(f"RMSD computation failed: {e}")
        return None


# --- Prediction Functions ---


def run_predict(
    input_yaml: Path,
    output_dir: Path,
    conda_env: Optional[str] = "boltz",
    predict_config: Optional[BoltzPredictConfig] = None,
    input_pdb: Optional[Path] = None,
    cp_chain: str = "P",
) -> BoltzResult:
    """
    Run Boltz2 structure prediction on a single YAML input.

    Args:
        input_yaml: Boltz2 YAML input file
        output_dir: Directory for output files
        conda_env: Name of conda environment with boltz installed. Pass
            ``None`` to skip the inner ``conda run -n <env>`` prefix when
            the caller has already activated the env (e.g. Snakemake's
            ``conda:`` directive). Defaults to ``"boltz"`` so notebook
            callers stay unchanged.
        predict_config: CLI options for boltz predict
        input_pdb: Optional input PDB for RMSD computation
        cp_chain: Chain ID of cyclic peptide (for RMSD computation)

    Returns:
        BoltzResult with prediction scores
    """
    if predict_config is None:
        predict_config = BoltzPredictConfig()

    input_yaml = Path(input_yaml)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = input_yaml.stem

    # Build CLI command
    cli_args = predict_config.to_cli_args()

    bare_cmd = [
        "boltz", "predict", str(input_yaml),
        "--out_dir", str(output_dir),
        *cli_args,
    ]
    if conda_env is not None:
        cmd = ["conda", "run", "-n", conda_env, "--no-capture-output", *bare_cmd]
    else:
        cmd = bare_cmd

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Boltz2 failed for {name}:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # Parse outputs
    output_ext = "cif" if predict_config.output_format == "mmcif" else "pdb"

    # Boltz2 creates output in boltz_results_{input_name}/ subdirectory
    boltz_results_dir = output_dir / f"boltz_results_{name}"
    if not boltz_results_dir.exists():
        # Fallback to direct output_dir if boltz didn't create subdirectory
        boltz_results_dir = output_dir
    pred_dir = boltz_results_dir / "predictions" / name

    # Find confidence JSON (may have _model_0 suffix)
    confidence_files = list(pred_dir.glob(f"confidence_{name}_model_*.json"))
    if not confidence_files:
        raise RuntimeError(f"No confidence JSON found in {pred_dir}")
    confidence_path = confidence_files[0]  # Best model (model_0)

    # Find structure output
    structure_files = list(pred_dir.glob(f"{name}_model_*.{output_ext}"))
    if not structure_files:
        raise RuntimeError(f"No structure output found in {pred_dir}")
    output_structure = structure_files[0]

    # Parse confidence scores
    scores = parse_confidence_json(confidence_path)

    # Compute RMSD if input PDB provided
    rmsd = None
    if input_pdb is not None:
        rmsd = compute_cp_rmsd(
            input_pdb, output_structure, cp_chain, predict_config.output_format
        )

    return BoltzResult(
        name=name,
        input_yaml=input_yaml,
        output_structure=output_structure,
        output_format=predict_config.output_format,
        confidence_score=scores["confidence_score"],
        iptm=scores["iptm"],
        ptm=scores["ptm"],
        plddt=scores["plddt"],
        ligand_iptm=scores["ligand_iptm"],
        protein_iptm=scores["protein_iptm"],
        complex_iplddt=scores["complex_iplddt"],
        complex_pde=scores["complex_pde"],
        complex_ipde=scores["complex_ipde"],
        chains_ptm=scores["chains_ptm"],
        pair_chains_iptm=scores["pair_chains_iptm"],
        input_pdb=input_pdb,
        rmsd_to_input=rmsd,
    )


def _run_boltz_subprocess(
    input_path: Path,
    output_dir: Path,
    conda_env: Optional[str],
    cli_args: list[str],
    timeout: Optional[int] = None,
) -> None:
    """
    Run a single `boltz predict` subprocess with real-time output streaming.

    Streams stdout/stderr to the current process so progress is visible
    in Jupyter notebooks.  Raises RuntimeError on non-zero exit.

    Args:
        input_path: YAML directory (or single file) to predict.
        output_dir: --out_dir for boltz.
        conda_env: Conda environment name, or ``None`` to invoke ``boltz``
            directly (when the caller has already activated the env).
        cli_args: Extra CLI flags from BoltzPredictConfig.
        timeout: Optional wall-clock timeout in seconds.
    """
    bare_cmd = [
        "boltz", "predict", str(input_path),
        "--out_dir", str(output_dir),
        *cli_args,
    ]
    if conda_env is not None:
        cmd = ["conda", "run", "-n", conda_env, "--no-capture-output", *bare_cmd]
    else:
        cmd = bare_cmd

    print(f"[boltz] Running: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output_lines: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            output_lines.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise RuntimeError(
            f"Boltz2 prediction timed out after {timeout}s.\n"
            f"Last output:\n{''.join(output_lines[-20:])}"
        )

    if proc.returncode != 0:
        tail = "".join(output_lines[-50:])
        raise RuntimeError(
            f"Boltz2 prediction failed (exit code {proc.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"Output (last 50 lines):\n{tail}"
        )


def run_batch(
    yaml_dir: Path,
    output_dir: Path,
    conda_env: Optional[str] = "boltz",
    predict_config: Optional[BoltzPredictConfig] = None,
    input_pdbs: Optional[dict[str, Path]] = None,
    cp_chain: str = "P",
    batch_size: Optional[int] = 10,
    timeout: Optional[int] = None,
) -> list[BoltzResult]:
    """
    Run Boltz2 on a directory of YAML files.

    Args:
        yaml_dir: Directory containing .yaml input files
        output_dir: Base directory for outputs
        conda_env: Conda environment name. Pass ``None`` to skip the inner
            ``conda run -n <env>`` prefix when the caller has already
            activated the env (e.g. Snakemake's ``conda:`` directive).
            Defaults to ``"boltz"`` so notebook callers stay unchanged.
        predict_config: CLI options for boltz predict
        input_pdbs: Optional mapping of yaml stem -> input PDB path for RMSD
        cp_chain: Chain ID of cyclic peptide (for RMSD computation)
        batch_size: Split YAMLs into sub-batches of this size and run
            ``boltz predict`` once per batch. Helps avoid GPU OOM on large
            input sets and gives finer retry granularity. Defaults to 10.
            Pass ``None`` to disable sub-batching and run a single
            ``boltz predict`` over all YAMLs (useful when the caller has
            already partitioned the work, e.g. Snakemake per-chunk rules).
        timeout: Optional wall-clock timeout in seconds per boltz invocation.

    Returns:
        List of BoltzResult objects
    """
    if predict_config is None:
        predict_config = BoltzPredictConfig()

    yaml_dir = Path(yaml_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect YAML files
    yaml_files = sorted(yaml_dir.glob("*.yaml"))
    if not yaml_files:
        warnings.warn(f"No .yaml files found in {yaml_dir}")
        return []

    cli_args = predict_config.to_cli_args()

    # --- Run boltz predict (optionally in sub-batches) ---
    if batch_size is not None and batch_size < len(yaml_files):
        batches = [
            yaml_files[i : i + batch_size]
            for i in range(0, len(yaml_files), batch_size)
        ]
        print(
            f"[boltz] Splitting {len(yaml_files)} inputs into "
            f"{len(batches)} batches of up to {batch_size}",
            flush=True,
        )
        for batch_idx, batch in enumerate(batches, 1):
            # Create a temporary directory with symlinks to this batch's YAMLs
            tmp_dir = Path(tempfile.mkdtemp(
                prefix=f"boltz_batch{batch_idx}_", dir=yaml_dir.parent
            ))
            try:
                for yf in batch:
                    (tmp_dir / yf.name).symlink_to(yf.resolve())
                print(
                    f"[boltz] Batch {batch_idx}/{len(batches)}: "
                    f"{len(batch)} files ({batch[0].stem} .. {batch[-1].stem})",
                    flush=True,
                )
                _run_boltz_subprocess(
                    tmp_dir, output_dir, conda_env, cli_args, timeout
                )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        _run_boltz_subprocess(
            yaml_dir, output_dir, conda_env, cli_args, timeout
        )

    # --- Parse outputs ---
    output_ext = "cif" if predict_config.output_format == "mmcif" else "pdb"
    results = []

    # Boltz2 creates output in boltz_results_{input_dir_name}/ subdirectory
    # When using batches, each batch creates its own subdirectory, but
    # predictions accumulate under output_dir.  We search all
    # boltz_results_* subdirs as well as the output_dir itself.
    candidate_dirs = sorted(output_dir.glob("boltz_results_*"))
    if not candidate_dirs:
        candidate_dirs = [output_dir]

    found_names: set[str] = set()
    for yaml_file in yaml_files:
        name = yaml_file.stem
        if name in found_names:
            continue

        # Search across candidate result directories
        pred_dir: Optional[Path] = None
        for cdir in candidate_dirs:
            candidate = cdir / "predictions" / name
            if candidate.exists():
                pred_dir = candidate
                break

        if pred_dir is None:
            warnings.warn(f"No prediction directory for {name}, skipping")
            continue

        # Find confidence JSON
        confidence_files = list(pred_dir.glob(f"confidence_{name}_model_*.json"))
        if not confidence_files:
            warnings.warn(f"No confidence JSON for {name}, skipping")
            continue
        confidence_path = confidence_files[0]

        # Find structure output
        structure_files = list(pred_dir.glob(f"{name}_model_*.{output_ext}"))
        if not structure_files:
            warnings.warn(f"No structure output for {name}, skipping")
            continue
        output_structure = structure_files[0]

        # Parse confidence scores
        scores = parse_confidence_json(confidence_path)

        # Compute RMSD if input PDB mapping provided
        rmsd = None
        input_pdb = None
        if input_pdbs and name in input_pdbs:
            input_pdb = input_pdbs[name]
            rmsd = compute_cp_rmsd(
                input_pdb, output_structure, cp_chain, predict_config.output_format
            )

        results.append(BoltzResult(
            name=name,
            input_yaml=yaml_file,
            output_structure=output_structure,
            output_format=predict_config.output_format,
            confidence_score=scores["confidence_score"],
            iptm=scores["iptm"],
            ptm=scores["ptm"],
            plddt=scores["plddt"],
            ligand_iptm=scores["ligand_iptm"],
            protein_iptm=scores["protein_iptm"],
            complex_iplddt=scores["complex_iplddt"],
            complex_pde=scores["complex_pde"],
            complex_ipde=scores["complex_ipde"],
            chains_ptm=scores["chains_ptm"],
            pair_chains_iptm=scores["pair_chains_iptm"],
            input_pdb=input_pdb,
            rmsd_to_input=rmsd,
        ))
        found_names.add(name)

    print(
        f"[boltz] Parsed {len(results)}/{len(yaml_files)} predictions",
        flush=True,
    )
    return results


def load_results(
    output_dir: Path | str,
    input_pdbs: Optional[dict[str, Path]] = None,
    cp_chain: str = "P",
    output_format: str = "mmcif",
    yaml_dir: Optional[Path | str] = None,
) -> list[BoltzResult]:
    """
    Load BoltzResults from previously computed predictions.

    Use this to resume the pipeline without re-running Boltz2.

    Args:
        output_dir: Base directory containing boltz outputs
        input_pdbs: Optional mapping of match name -> input PDB path for RMSD
        cp_chain: Chain ID of cyclic peptide (for RMSD computation)
        output_format: "mmcif" or "pdb" (must match what was used during prediction)
        yaml_dir: Optional path to YAML input directory (to find boltz_results_* subdir name)

    Returns:
        List of BoltzResult objects
    """
    output_dir = Path(output_dir)
    output_ext = "cif" if output_format == "mmcif" else "pdb"

    if yaml_dir is not None:
        yaml_dir = Path(yaml_dir)

    # Collect all boltz_results_* directories that contain predictions
    boltz_results_dirs = []

    candidates = sorted(output_dir.glob("boltz_results_*"))
    boltz_results_dirs = [c for c in candidates if (c / "predictions").exists()]

    if not boltz_results_dirs:
        if (output_dir / "predictions").exists():
            boltz_results_dirs = [output_dir]
        else:
            raise FileNotFoundError(f"No boltz_results_* directory found in {output_dir}")

    results = []

    for boltz_results_dir in boltz_results_dirs:
        predictions_dir = boltz_results_dir / "predictions"

        # Iterate over prediction directories (each match has its own directory)
        for pred_dir in sorted(predictions_dir.iterdir()):
            if not pred_dir.is_dir():
                continue

            name = pred_dir.name

            # Find confidence JSON
            confidence_files = list(pred_dir.glob(f"confidence_{name}_model_*.json"))
            if not confidence_files:
                warnings.warn(f"No confidence JSON for {name}, skipping")
                continue
            confidence_path = confidence_files[0]

            # Find structure output
            structure_files = list(pred_dir.glob(f"{name}_model_*.{output_ext}"))
            if not structure_files:
                warnings.warn(f"No structure output for {name}, skipping")
                continue
            output_structure = structure_files[0]

            # Parse confidence scores
            scores = parse_confidence_json(confidence_path)

            # Compute RMSD if input PDB mapping provided
            rmsd = None
            input_pdb = None
            if input_pdbs and name in input_pdbs:
                input_pdb = input_pdbs[name]
                rmsd = compute_cp_rmsd(input_pdb, output_structure, cp_chain, output_format)

            # Find input YAML if it exists (for completeness)
            input_yaml = None
            if yaml_dir is not None:
                yaml_path = yaml_dir / f"{name}.yaml"
                if yaml_path.exists():
                    input_yaml = yaml_path

            results.append(BoltzResult(
                name=name,
                input_yaml=input_yaml,
                output_structure=output_structure,
                output_format=output_format,
                confidence_score=scores["confidence_score"],
                iptm=scores["iptm"],
                ptm=scores["ptm"],
                plddt=scores["plddt"],
                ligand_iptm=scores["ligand_iptm"],
                protein_iptm=scores["protein_iptm"],
                complex_iplddt=scores["complex_iplddt"],
                complex_pde=scores["complex_pde"],
                complex_ipde=scores["complex_ipde"],
                chains_ptm=scores["chains_ptm"],
                pair_chains_iptm=scores["pair_chains_iptm"],
                input_pdb=input_pdb,
                rmsd_to_input=rmsd,
            ))

    print(f"[boltz] Loaded {len(results)} results from {len(boltz_results_dirs)} batch director{'y' if len(boltz_results_dirs) == 1 else 'ies'}", flush=True)
    return results
