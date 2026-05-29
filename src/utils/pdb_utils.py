"""
PDB manipulation utilities.
"""

from pathlib import Path

from Bio.PDB import PDBIO, Chain, Model, PDBParser, Structure


def unify_pdbs(
    cp_pdb: Path,
    cp_chain: str,
    binder_pdb: Path,
    binder_chain: str,
    output_pdb: Path,
    cp_output_chain: str = "P",
    binder_output_chain: str = "B",
) -> Path:
    """
    Combine a cyclic peptide and binder protein into a single PDB.

    Args:
        cp_pdb: Path to cyclic peptide PDB
        cp_chain: Chain ID of CP in source file
        binder_pdb: Path to binder protein PDB
        binder_chain: Chain ID of binder in source file
        output_pdb: Path for combined output
        cp_output_chain: Chain ID for CP in output (default: P)
        binder_output_chain: Chain ID for binder in output (default: B)

    Returns:
        Path to output file
    """
    parser = PDBParser(QUIET=True)

    cp_struct = parser.get_structure("cp", str(cp_pdb))
    binder_struct = parser.get_structure("binder", str(binder_pdb))

    cp_chain_obj = cp_struct[0][cp_chain]
    binder_chain_obj = binder_struct[0][binder_chain]

    new_struct = Structure.Structure("combined")
    new_model = Model.Model(0)
    new_struct.add(new_model)

    new_cp_chain = Chain.Chain(cp_output_chain)
    for residue in cp_chain_obj:
        new_cp_chain.add(residue.copy())
    new_model.add(new_cp_chain)

    new_binder_chain = Chain.Chain(binder_output_chain)
    for residue in binder_chain_obj:
        new_binder_chain.add(residue.copy())
    new_model.add(new_binder_chain)

    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    io = PDBIO()
    io.set_structure(new_struct)
    io.save(str(output_pdb))

    return output_pdb


def extract_chain(
    input_pdb: Path,
    chain_id: str,
    output_pdb: Path,
) -> Path:
    """
    Extract a single chain from a PDB file.

    Args:
        input_pdb: Source PDB file
        chain_id: Chain to extract
        output_pdb: Output path

    Returns:
        Path to output file
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("input", str(input_pdb))

    chain = struct[0][chain_id]

    new_struct = Structure.Structure("extracted")
    new_model = Model.Model(0)
    new_struct.add(new_model)

    new_chain = Chain.Chain(chain_id)
    for residue in chain:
        new_chain.add(residue.copy())
    new_model.add(new_chain)

    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    io = PDBIO()
    io.set_structure(new_struct)
    io.save(str(output_pdb))

    return output_pdb


# Modified amino acids → canonical parent residue. Used in extract_chain_to_cif
# to keep Boltz's mmcif parser (which only special-cases MSE→MET) from tripping
# on PTM residue names it doesn't know. We rename the residue but keep the
# backbone atoms (N, CA, C, O, CB where present), so the template still pins
# Cα positions correctly. Side-chain atoms beyond CB that are specific to the
# modification (acetyl on ALY, hydroxyl on CSO, phosphate on SEP, etc.) would
# break atom-name validation downstream, so we drop them.
_PTM_TO_PARENT = {
    "CSO": "CYS",   # S-hydroxycysteine
    "ALY": "LYS",   # N6-acetyl-lysine
    "SEP": "SER",   # phosphoserine
    "TPO": "THR",   # phosphothreonine
    "PTR": "TYR",   # phosphotyrosine
    "MLY": "LYS",   # mono-methyl-lysine
    "M3L": "LYS",   # tri-methyl-lysine
    "HYP": "PRO",   # hydroxyproline
    "PCA": "GLN",   # pyroglutamate
    # MSE is intentionally absent — Boltz handles MSE→MET natively.
}

# Atoms kept after a PTM-to-parent rename. Anything outside this set is dropped
# from the residue (e.g. the acetyl methyl of ALY, hydroxyl of CSO).
_BACKBONE_PLUS_CB = {"N", "CA", "C", "O", "CB", "OXT", "H"}

_CANONICAL_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE",  # Boltz remaps MSE to MET internally; pass through unchanged.
}


def extract_chain_to_cif(
    input_pdb: Path,
    chain_id: str,
    output_cif: Path,
) -> Path:
    """
    Extract a single chain from a PDB file and save as mmCIF.

    Uses gemmi for proper CIF output with entity setup.

    Sanitisation applied to keep Boltz's mmcif parser happy
    (parse_polymer only special-cases MSE→MET, anything else trips
    `gemmi.align_sequence_to_polymer`):

      1. Drop residues attached to the chain but not part of the polymer:
         small molecules (e.g. BEZ, GOL, drug-like 3-letter codes), waters,
         ions. These end up tagged as het_flag='H' / non-standard residue
         and gemmi was placing them into the polymer entity sequence.
      2. Rename known modified amino acids to their canonical parents
         (CSO→CYS, ALY→LYS, SEP→SER, …) and trim the residue down to
         backbone + CB so the structural pin survives but downstream parsers
         see a standard residue. MSE is left alone (Boltz handles it).

    Args:
        input_pdb: Source PDB file
        chain_id: Chain to extract
        output_cif: Output CIF path

    Returns:
        Path to output file
    """
    import gemmi

    structure = gemmi.read_structure(str(input_pdb))

    # Remove all chains except the target
    model = structure[0]
    chains_to_remove = [c.name for c in model if c.name != chain_id]
    for name in chains_to_remove:
        model.remove_chain(name)

    # Sanitise residues in the surviving chain. gemmi doesn't expose a
    # convenient delete-by-key on Chain, so rebuild the chain from a filtered
    # residue list (keep canonical, remap known PTMs to parent, drop the rest).
    src_chain = model[chain_id]
    chain_name = src_chain.name
    keep_residues = []
    ptm_renames = []
    drops = []
    for residue in src_chain:
        rn = residue.name.upper()
        if rn in _CANONICAL_AA:
            keep_residues.append(residue)
            continue
        if rn in _PTM_TO_PARENT:
            parent = _PTM_TO_PARENT[rn]
            new_res = gemmi.Residue()
            new_res.name = parent
            new_res.seqid = residue.seqid
            new_res.label_seq = residue.label_seq
            new_res.het_flag = "A"  # promote to polymer atom record
            for a in residue:
                if a.name in _BACKBONE_PLUS_CB:
                    new_res.add_atom(a)
            ptm_renames.append((rn, parent, residue.seqid.num))
            keep_residues.append(new_res)
            continue
        drops.append((residue.name, residue.seqid.num))

    new_chain = gemmi.Chain(chain_name)
    for r in keep_residues:
        new_chain.add_residue(r)
    model.remove_chain(chain_name)
    model.add_chain(new_chain)

    if ptm_renames:
        print("[extract_chain_to_cif] PTM remap: "
              + ", ".join(f"{old}{num}->{new}" for old, new, num in ptm_renames))
    if drops:
        print("[extract_chain_to_cif] stripped non-polymer residues: "
              + ", ".join(f"{n}{num}" for n, num in drops))

    # Setup entities for proper CIF output
    structure.setup_entities()

    output_cif = Path(output_cif)
    output_cif.parent.mkdir(parents=True, exist_ok=True)

    # Write as mmCIF
    structure.make_mmcif_document().write_file(str(output_cif))

    return output_cif


def rename_chain(
    input_pdb: Path,
    old_chain: str,
    new_chain: str,
    output_pdb: Path,
) -> Path:
    """
    Rename a chain in a PDB file.

    Args:
        input_pdb: Source PDB file
        old_chain: Current chain ID
        new_chain: New chain ID
        output_pdb: Output path

    Returns:
        Path to output file
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("input", str(input_pdb))

    chain = struct[0][old_chain]

    new_struct = Structure.Structure("renamed")
    new_model = Model.Model(0)
    new_struct.add(new_model)

    new_chain_obj = Chain.Chain(new_chain)
    for residue in chain:
        new_chain_obj.add(residue.copy())
    new_model.add(new_chain_obj)

    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    io = PDBIO()
    io.set_structure(new_struct)
    io.save(str(output_pdb))

    return output_pdb


# MODELLER can rename CCD codes; map back to correct amino acid codes.
# DC: MODELLER truncates DCY (D-cysteine) → DC, which collides with
#     deoxycytidine (DNA). Map back to DCY.
_MODELLER_CCD_FIXES = {
    "DC": "DCY",
}


def _is_polymer_residue(residue) -> bool:
    """Check if a residue is a polymer residue (has backbone N, CA, C atoms)."""
    atom_names = {a.get_name() for a in residue}
    return {"N", "CA", "C"}.issubset(atom_names)


def get_sequence(pdb_path: Path, chain_id: str) -> str:
    """
    Extract amino acid sequence from a PDB chain.

    Includes HETATM polymer residues (non-standard amino acids with backbone
    atoms N, CA, C). Standard residues get their 1-letter code, non-standard
    polymer residues get "X". Water and non-polymer HETATMs are skipped.

    Args:
        pdb_path: Path to PDB file
        chain_id: Chain ID to extract

    Returns:
        One-letter amino acid sequence
    """
    from Bio.PDB.Polypeptide import protein_letters_3to1

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("input", str(pdb_path))
    chain = struct[0][chain_id]

    sequence = []
    for residue in chain:
        het_flag = residue.id[0]
        if het_flag == "W":
            continue
        resname = residue.resname.strip()
        # Include if: has protein backbone, OR is a known MODELLER remap
        is_polymer = _is_polymer_residue(residue) or resname in _MODELLER_CCD_FIXES
        if het_flag != " " and not is_polymer:
            continue
        if het_flag == " " and not is_polymer:
            # ATOM record without backbone (e.g. DC with nucleotide coords)
            # — only include if it's a known remap
            if resname not in _MODELLER_CCD_FIXES:
                continue
        resname = _MODELLER_CCD_FIXES.get(resname, resname)
        if resname in protein_letters_3to1:
            sequence.append(protein_letters_3to1[resname])
        else:
            sequence.append("X")

    return "".join(sequence)


def get_modifications(pdb_path: Path, chain_id: str) -> list[dict]:
    """
    Identify non-standard amino acid positions and their CCD codes.

    Iterates polymer residues (those with backbone N, CA, C) and records
    any whose resname is not a standard amino acid.

    Args:
        pdb_path: Path to PDB file
        chain_id: Chain ID to inspect

    Returns:
        List of dicts with 'position' (1-indexed) and 'ccd' keys,
        e.g. [{"position": 2, "ccd": "HCS"}, {"position": 5, "ccd": "HCS"}]
    """
    from Bio.PDB.Polypeptide import protein_letters_3to1

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("input", str(pdb_path))
    chain = struct[0][chain_id]

    modifications = []
    pos = 0
    for residue in chain:
        het_flag = residue.id[0]
        if het_flag == "W":
            continue
        resname = residue.resname.strip()
        is_polymer = _is_polymer_residue(residue) or resname in _MODELLER_CCD_FIXES
        if het_flag != " " and not is_polymer:
            continue
        if het_flag == " " and not is_polymer:
            if resname not in _MODELLER_CCD_FIXES:
                continue
        pos += 1
        ccd = _MODELLER_CCD_FIXES.get(resname, resname)
        if ccd not in protein_letters_3to1:
            modifications.append({"position": pos, "ccd": ccd})

    return modifications


def get_chain_ids(pdb_path: Path) -> list[str]:
    """
    Get all chain IDs in a PDB file.

    Args:
        pdb_path: Path to PDB file

    Returns:
        List of chain IDs
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("input", str(pdb_path))
    return [chain.id for chain in struct[0]]


def import_complex(
    complex_pdb: Path,
    ligand_chain: str,
    target_chain: str,
    output_pdb: Path,
    ligand_output_chain: str = "L",
    target_output_chain: str = "T",
) -> Path:
    """
    Import a complex PDB and standardize chain naming for the pipeline.

    Extracts two chains from the input complex and renames them:
    - Ligand chain (interface that CP mimics) -> L
    - Target chain (protein that CP binds to) -> T

    Args:
        complex_pdb: Input PDB containing both ligand and target
        ligand_chain: Chain ID of ligand in input (interface CP mimics)
        target_chain: Chain ID of target in input (what CP binds to)
        output_pdb: Path for processed output
        ligand_output_chain: Output chain ID for ligand (default: "L")
        target_output_chain: Output chain ID for target (default: "T")

    Returns:
        Path to processed PDB file
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("complex", str(complex_pdb))

    # Validate chains exist
    available = [c.id for c in struct[0]]
    if ligand_chain not in available:
        raise ValueError(f"Ligand chain '{ligand_chain}' not found. Available: {available}")
    if target_chain not in available:
        raise ValueError(f"Target chain '{target_chain}' not found. Available: {available}")

    ligand_chain_obj = struct[0][ligand_chain]
    target_chain_obj = struct[0][target_chain]

    # Build new structure with standardized chain IDs
    new_struct = Structure.Structure("processed")
    new_model = Model.Model(0)
    new_struct.add(new_model)

    # Add ligand chain as L
    new_ligand = Chain.Chain(ligand_output_chain)
    for residue in ligand_chain_obj:
        new_ligand.add(residue.copy())
    new_model.add(new_ligand)

    # Add target chain as T
    new_target = Chain.Chain(target_output_chain)
    for residue in target_chain_obj:
        new_target.add(residue.copy())
    new_model.add(new_target)

    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    io = PDBIO()
    io.set_structure(new_struct)
    io.save(str(output_pdb))

    return output_pdb
