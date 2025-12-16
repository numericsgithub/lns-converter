"""
Utility script to start Xilinx Vivado in batch mode from Python.

Example:
    python call_vivado.py --vivado "D:/Xilinx/2025.1.1/Vivado/bin/vivado.bat" \
        --tcl scripts/build_project.tcl --args top_module build_dir
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List


DEFAULT_VIVADO_PATH = Path(r"D:\Xilinx\2025.1.1\Vivado\bin\vivado.bat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Vivado in batch mode using a provided TCL script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vivado",
        dest="vivado_executable",
        type=Path,
        default=None,
        help="Path to the Vivado executable (vivado or vivado.bat). "
        "If omitted, the script attempts to find it in PATH.",
    )
    parser.add_argument(
        "--tcl",
        dest="tcl_script",
        type=Path,
        required=True,
        help="Path to the main TCL script Vivado should execute.",
    )
    parser.add_argument(
        "--args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Optional arguments forwarded to the TCL script via -tclargs.",
    )
    return parser.parse_args()


def resolve_vivado_executable(explicit_path: Path | None) -> Path:
    if explicit_path:
        resolved = explicit_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Vivado executable not found: {resolved}")
        return resolved

    vivado = shutil.which("vivado")
    if vivado:
        return Path(vivado).resolve()

    if DEFAULT_VIVADO_PATH.exists():
        return DEFAULT_VIVADO_PATH.resolve()

    raise FileNotFoundError(
        "Vivado executable not supplied, not found in PATH, and default path "
        f"{DEFAULT_VIVADO_PATH} does not exist. Use --vivado to provide the path explicitly."
    )


def build_command(vivado: Path, tcl_script: Path, script_args: List[str]) -> List[str]:
    full_path_main_tcl = tcl_script.expanduser().resolve()
    if not full_path_main_tcl.exists():
        raise FileNotFoundError(f"TCL script not found: {full_path_main_tcl}")

    return [
        str(vivado),
        "-mode",
        "batch",
        "-nolog",
        "-nojournal",
        "-source",
        str(full_path_main_tcl),
        "-tclargs",
        *script_args,
    ]


def main() -> None:
    args = parse_args()
    vivado_executable = resolve_vivado_executable(args.vivado_executable)
    command = build_command(vivado_executable, args.tcl_script, args.args)

    print("Launching Vivado batch process:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))

    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)


if __name__ == "__main__":
    main()

