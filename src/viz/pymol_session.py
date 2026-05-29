"""Notebook widget for building PyMOL sessions from a folder of PDB/CIF files.

Call ``build_pymol_session_widget(default_folder=...)`` and ``display()`` the
result. The widget remembers its last values across notebook restarts via a
tempfile-backed JSON state file.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import ipywidgets as widgets
from IPython.display import Javascript, display

_STATE_FILE = Path(tempfile.gettempdir()) / "pymol_widget_config.json"
_RAINBOW_COLORS = [
    "red", "orange", "yellow", "green", "cyan",
    "blue", "purple", "magenta", "pink", "salmon",
    "lime", "teal", "marine", "slate", "violet",
]


def _load_state() -> dict:
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _build_pymol_cmd(env_name: str, runner: str, *pymol_args: str) -> list[str]:
    if runner == "system" or not env_name:
        return ["pymol", *pymol_args]
    if runner == "micromamba":
        return ["micromamba", "run", "-n", env_name, "pymol", *pymol_args]
    return ["conda", "run", "-n", env_name, "pymol", *pymol_args]


def build_pymol_session_widget(default_folder: str | None = None) -> widgets.VBox:
    """Return a VBox widget that assembles a PyMOL .pse from a folder of structures."""
    saved = _load_state()
    initial_folder = saved.get("folder_path") or default_folder or str(Path.cwd())

    last_pse_path = {"value": None}

    folder_path = widgets.Text(
        value=initial_folder,
        description="Folder:",
        placeholder="Path to folder with PDB/CIF files",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="99%"),
    )
    reference_pdb = widgets.Text(
        value=saved.get("reference_pdb", ""),
        description="Reference:",
        placeholder="Path to reference PDB (optional)",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="99%"),
    )
    env_manager = widgets.ToggleButtons(
        options=["conda", "micromamba", "system"],
        value=saved.get("env_manager", "micromamba"),
        description="Runner:",
        style={"description_width": "80px", "button_width": "100px"},
    )
    conda_env = widgets.Text(
        value=saved.get("conda_env", "pym"),
        description="Env:",
        placeholder="Environment name",
        style={"description_width": "40px"},
        layout=widgets.Layout(width="200px"),
    )
    transparency = widgets.FloatSlider(
        value=saved.get("transparency", 0.7),
        min=0.0, max=1.0, step=0.05,
        description="Transparency:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="50%"),
    )
    color_by_file = widgets.Checkbox(
        value=saved.get("color_by_file", True),
        description="Color each file differently (rainbow)",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="auto"),
    )
    align_all = widgets.Checkbox(
        value=saved.get("align_all", False),
        description="Align all structures (to reference if provided)",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="auto"),
    )
    reference_opaque = widgets.Checkbox(
        value=saved.get("reference_opaque", True),
        description="Reference opaque (no transparency)",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="auto"),
    )
    create_button = widgets.Button(
        description="Create .pse", button_style="primary", icon="save",
        layout=widgets.Layout(width="150px"),
    )
    open_button = widgets.Button(
        description="Open in PyMOL", button_style="success", icon="eye",
        layout=widgets.Layout(width="150px"),
    )
    status_output = widgets.Output(layout=widgets.Layout(width="99%"))

    def _save_state(_=None):
        _STATE_FILE.write_text(json.dumps({
            "folder_path": folder_path.value,
            "reference_pdb": reference_pdb.value,
            "env_manager": env_manager.value,
            "conda_env": conda_env.value,
            "transparency": transparency.value,
            "color_by_file": color_by_file.value,
            "align_all": align_all.value,
            "reference_opaque": reference_opaque.value,
        }))

    for w in (folder_path, reference_pdb, env_manager, conda_env,
              transparency, color_by_file, align_all, reference_opaque):
        w.observe(_save_state, names="value")

    def _create_pse(_button):
        with status_output:
            status_output.clear_output()

            folder = Path(folder_path.value)
            if not folder.exists():
                print(f"Error: Folder does not exist: {folder}")
                return

            pdb_files = list(folder.glob("*.pdb")) + list(folder.glob("*.cif"))
            if not pdb_files:
                print(f"Error: No PDB or CIF files found in {folder}")
                return
            print(f"Found {len(pdb_files)} structure files")

            ref_path_str = reference_pdb.value.strip()
            ref_obj_name = None
            ref_path = None
            if ref_path_str:
                ref_path = Path(ref_path_str)
                if not ref_path.exists():
                    print(f"Error: Reference PDB does not exist: {ref_path}")
                    return
                ref_obj_name = "reference"
                print(f"Using reference: {ref_path.name}")

            script_lines = ["from pymol import cmd", ""]
            if ref_obj_name:
                script_lines.append(f'cmd.load(r"{ref_path}", "{ref_obj_name}")')
                script_lines.append(f'cmd.color("white", "{ref_obj_name}")')
                script_lines.append("")

            obj_names: list[str] = []
            for i, pdb_file in enumerate(sorted(pdb_files)):
                obj_name = pdb_file.stem.replace("-", "_").replace(" ", "_")
                obj_names.append(obj_name)
                script_lines.append(f'cmd.load(r"{pdb_file}", "{obj_name}")')
                if color_by_file.value:
                    color = _RAINBOW_COLORS[i % len(_RAINBOW_COLORS)]
                    script_lines.append(f'cmd.color("{color}", "{obj_name}")')
            script_lines.append("")

            if align_all.value and obj_names:
                if ref_obj_name:
                    for obj_name in obj_names:
                        script_lines.append(f'cmd.align("{obj_name}", "{ref_obj_name}")')
                else:
                    first_obj = obj_names[0]
                    for obj_name in obj_names[1:]:
                        script_lines.append(f'cmd.align("{obj_name}", "{first_obj}")')
                script_lines.append("")

            trans_val = transparency.value
            if ref_obj_name and reference_opaque.value:
                for obj_name in obj_names:
                    script_lines.append(f'cmd.set("cartoon_transparency", {trans_val}, "{obj_name}")')
                    script_lines.append(f'cmd.set("transparency", {trans_val}, "{obj_name}")')
                    script_lines.append(f'cmd.set("stick_transparency", {trans_val}, "{obj_name}")')
            else:
                script_lines.append(f'cmd.set("cartoon_transparency", {trans_val}, "all")')
                script_lines.append(f'cmd.set("transparency", {trans_val}, "all")')
                script_lines.append(f'cmd.set("stick_transparency", {trans_val}, "all")')

            script_lines.append('cmd.show("cartoon", "all")')
            script_lines.append('cmd.zoom("all")')

            session_path = folder / "session.pse"
            script_lines.append(f'cmd.save(r"{session_path}")')
            script_lines.append("cmd.quit()")

            script_path = Path(tempfile.gettempdir()) / "pymol_session_script.py"
            script_path.write_text("\n".join(script_lines))
            print(f"Script: {script_path}")

            cmd = _build_pymol_cmd(conda_env.value.strip(), env_manager.value,
                                   "-cq", "-r", str(script_path))
            print(f"Running: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.stdout:
                    print(f"stdout: {result.stdout}")
                if result.stderr:
                    print(f"stderr: {result.stderr}")
                if session_path.exists():
                    last_pse_path["value"] = str(session_path)
                    print(f"Created: {session_path}")
                    display(Javascript(f'navigator.clipboard.writeText("{session_path}")'))
                    print("(Path copied to clipboard)")
                else:
                    print(f"Error: .pse file was not created at {session_path}")
            except subprocess.TimeoutExpired:
                print("Error: PyMOL timed out (120s)")
            except FileNotFoundError as e:
                print(f"Error: {e}")

    def _open_pymol(_button):
        with status_output:
            status_output.clear_output()
            if not last_pse_path["value"]:
                print("Error: No .pse file created yet. Click 'Create .pse' first.")
                return
            pse = Path(last_pse_path["value"])
            if not pse.exists():
                print(f"Error: File not found: {pse}")
                return
            cmd = _build_pymol_cmd(conda_env.value.strip(), env_manager.value, str(pse))
            print(f"Opening {pse.name} in PyMOL...")
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("PyMOL launched!")
            except FileNotFoundError as e:
                print(f"Error: {e}")

    create_button.on_click(_create_pse)
    open_button.on_click(_open_pymol)

    return widgets.VBox([
        widgets.HTML("<h4>PyMOL Session Creator</h4>"),
        folder_path,
        reference_pdb,
        widgets.HBox([env_manager, conda_env],
                     layout=widgets.Layout(align_items="center", gap="20px")),
        transparency,
        widgets.VBox([color_by_file, align_all, reference_opaque]),
        widgets.HBox([create_button, open_button],
                     layout=widgets.Layout(gap="10px")),
        status_output,
    ], layout=widgets.Layout(width="99%", padding="10px"))
