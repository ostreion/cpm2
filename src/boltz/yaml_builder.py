"""Boltz2 YAML input construction.

Boltz2 expects bond-atom triples rendered as inline flow sequences
(``[P, 2, SG]``). ``FlowList`` subclasses ``list`` and is registered with a
PyYAML representer that forces flow_style on those triples; the rest of the
document stays in block style.
"""

from pathlib import Path

import yaml


class FlowList(list):
    """List subclass that serialises to a YAML flow sequence."""
    pass


def _flow_list_representer(dumper, data):
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True
    )


yaml.add_representer(FlowList, _flow_list_representer)


def build_boltz_yaml(
    target_seq: str,
    cp_seq: str,
    modifications: list[dict],
    cp_constraints: dict,
    target_constraints: dict | None = None,
    target_msa_path: str | None = None,
) -> dict:
    """Build a Boltz2 prediction YAML as a plain dict.

    cp_constraints / target_constraints shape:
        {"head_to_tail": bool,
         "disulfides": [(i,j), ...],
         "lactams":    [(i,j), ...],
         "thioethers": [(i,j), ...]}
    Indices are 1-based over the sequence.

    If `target_msa_path` is given, that path (.a3m or .csv) is emitted as the
    target chain's MSA so Boltz reuses it instead of fetching the same MSA
    once per CP YAML. CP chain stays single-sequence (msa: empty) by design.
    """
    cp_entry = {
        "id": "P",
        "sequence": cp_seq,
        "cyclic": cp_constraints.get("head_to_tail", False),
        "msa": "empty",
    }
    if modifications:
        cp_entry["modifications"] = modifications

    target_entry = {"id": "T", "sequence": target_seq}
    if target_msa_path:
        target_entry["msa"] = str(target_msa_path)

    data = {
        "sequences": [
            {"protein": target_entry},
            {"protein": cp_entry},
        ],
    }

    constraints: list[dict] = []

    # Cyclic peptide (chain P) bonds
    for i, j in cp_constraints.get("disulfides", []):
        constraints.append({"bond": {
            "atom1": FlowList(["P", i, "SG"]),
            "atom2": FlowList(["P", j, "SG"]),
        }})
    for i, j in cp_constraints.get("lactams", []):
        constraints.append({"bond": {
            "atom1": FlowList(["P", i, "NZ"]),
            "atom2": FlowList(["P", j, "CG"]),
        }})
    for i, j in cp_constraints.get("thioethers", []):
        constraints.append({"bond": {
            "atom1": FlowList(["P", i, "SG"]),
            "atom2": FlowList(["P", j, "CB"]),
        }})

    # Target (chain T) bonds
    if target_constraints:
        for i, j in target_constraints.get("disulfides", []):
            constraints.append({"bond": {
                "atom1": FlowList(["T", i, "SG"]),
                "atom2": FlowList(["T", j, "SG"]),
            }})
        for i, j in target_constraints.get("lactams", []):
            constraints.append({"bond": {
                "atom1": FlowList(["T", i, "NZ"]),
                "atom2": FlowList(["T", j, "CG"]),
            }})
        for i, j in target_constraints.get("thioethers", []):
            constraints.append({"bond": {
                "atom1": FlowList(["T", i, "SG"]),
                "atom2": FlowList(["T", j, "CB"]),
            }})

    if constraints:
        data["constraints"] = constraints

    return data


def write_boltz_yaml(path: Path, data: dict) -> None:
    """Dump a Boltz2 YAML dict with the notebook's historical style."""
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
