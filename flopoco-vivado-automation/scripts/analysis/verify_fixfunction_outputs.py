"""
Verify FixFunctionByTable testbench outputs against the reference function
log2(x + 0.0001) with fixed-point scaling (lsbIn = lsbOut = -3).

Usage:
    python verify_fixfunction_outputs.py simulation/FixFunctionByTable_outputs.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

LSB_IN = -3
LSB_OUT = -6
INPUT_SCALE = 2 ** abs(LSB_IN)
OUTPUT_SCALE = 2 ** abs(LSB_OUT)


def evaluate_function(x: float, func: str = "log(x+0.0001)/log(2)") -> float:
    """Evaluate a function string with x as the variable."""
    return eval(func, {"log": math.log, "sqrt": math.sqrt, "exp": math.exp}, {"x": x})


def compute_expected(x_raw: int, lsb_in: int, lsb_out: int, func: str) -> int:
    input_scale = 2 ** abs(lsb_in)
    output_scale = 2 ** abs(lsb_out)
    real_in = x_raw / input_scale
    real_out = evaluate_function(real_in, func)
    return int(round(real_out * output_scale))


def parse_line(line: str) -> tuple[int, int]:
    stripped = line.strip()
    if not stripped:
        raise ValueError("Empty line encountered.")
    parts = stripped.split()
    if len(parts) != 2:
        raise ValueError(f"Line '{line}' does not have exactly two columns.")
    return int(parts[0]), int(parts[1])


def verify_file(path: Path, lsb_in: int, lsb_out: int, func: str) -> None:
    mismatches = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            x_val, y_measured = parse_line(line)
            expected = compute_expected(x_val, lsb_in, lsb_out, func)
            if expected != y_measured:
                mismatches.append((lineno, x_val, y_measured, expected))

    print(f"{'Input':>5} {'Measured':>10} {'Expected':>10} {'Status':>8}")
    print("-" * 37)
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            x_val, y_measured = parse_line(stripped)
            expected = compute_expected(x_val, lsb_in, lsb_out, func)
            status = "OK" if expected == y_measured else "MISMATCH"
            print(f"{x_val:5d} {y_measured:10d} {expected:10d} {status:>8}")

    if not mismatches:
        print(f"\nAll outputs match reference model for {path}.")
    else:
        print(f"\n{len(mismatches)} mismatches detected in {path}:")
        for lineno, x_val, measured, expected in mismatches:
            print(
                f"  line {lineno}: input={x_val} measured={measured} expected={expected}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify FixFunction outputs CSV.")
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=Path("simulation/FixFunctionByTable_outputs.csv"),
        help="Path to the CSV file with '<input> <output>' pairs.",
    )
    parser.add_argument(
        "--lsb-in",
        type=int,
        default=LSB_IN,
        help=f"LSB position for input (default: {LSB_IN})",
    )
    parser.add_argument(
        "--lsb-out",
        type=int,
        default=LSB_OUT,
        help=f"LSB position for output (default: {LSB_OUT})",
    )
    parser.add_argument(
        "--function",
        default="log(x+0.0001)/log(2)",
        help="Mathematical function to evaluate (default: log(x+0.0001)/log(2))",
    )
    args = parser.parse_args()
    verify_file(args.csv_path, args.lsb_in, args.lsb_out, args.function)


if __name__ == "__main__":
    main()

