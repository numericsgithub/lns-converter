import csv
import math

# -----------------------------
# Convert integer to fixed-point
# -----------------------------
def int_to_fixed(value, lsb):
    return value * (2 ** lsb)

# Evaluate a given function
def evaluate_function(x, func = "log(x+0.0001)/log(2)"):
    # You can safely use Python's eval with a restricted namespace
    return eval(func, {"log": math.log, "sqrt": math.sqrt, "exp": math.exp}, {"x": x})

# -----------------------------
# Compute error
# -----------------------------
def compute_error(csv_file, lsbIn, msbOut, lsbOut, func):
    results = []

    # Max numeric range based on MSB/LSB
    total_bits = msbOut - lsbOut + 1

    # Read CSV
    with open(csv_file) as f:
        reader = csv.reader(f)
        for row in reader:
            row = row[0].split(" ")
            if len(row) != 2:
                print(f"Skipping row: {row} because it has {len(row)} columns")
                continue
            input_int = int(row[0])
            output_int = int(row[1])

            # Convert integers to fixed-point real numbers
            x_real = int_to_fixed(input_int, lsbIn)
            y_real = int_to_fixed(output_int, lsbOut)

            # Expected mathematical result
            y_expected = evaluate_function(x_real, func)

            # Errors
            abs_error = y_real - y_expected

            results.append({
                "input": input_int,
                "x_real": x_real,
                "output": output_int,
                "y_real": y_real,
                "expected": y_expected,
                "abs_error": abs_error,
            })

    return results


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute errors between simulation outputs and expected values.")
    parser.add_argument("--csv", default="simulation/FixFunctionByTable_outputs.csv",
                        help="Path to CSV file with input/output pairs")
    parser.add_argument("--lsb-in", type=int, required=True,
                        help="LSB position for input (e.g., -6)")
    parser.add_argument("--msb-out", type=int, required=True,
                        help="MSB position for output (e.g., 4)")
    parser.add_argument("--lsb-out", type=int, required=True,
                        help="LSB position for output (e.g., -6)")
    parser.add_argument("--function", default="log(x+0.0001)/log(2)",
                        help="Mathematical function to evaluate (default: log(x+0.0001)/log(2))")

    args = parser.parse_args()

    # Determine decimal precision based on lsbOut
    # If lsbOut < -10 (more negative), use 12 decimal places; otherwise use 8
    decimal_precision = 12 if args.lsb_out < -10 else 8

    errors = compute_error(args.csv, args.lsb_in, args.msb_out, args.lsb_out, args.function)

    print("INPUT | FIXED INPUT | OUTPUT | FIXED OUTPUT | EXPECTED | ABS ERROR")
    print("----------------------------------------------------------------------")
    for e in errors:
        print(f"{e['input']:5d} | {e['x_real']:11.{decimal_precision}f} | {e['output']:6d} | {e['y_real']:12.{decimal_precision}f} | "
              f"{e['expected']:10.{decimal_precision}f} | {abs(e['abs_error']):10.{decimal_precision}f}")
    total_error = 0
    for e in errors:
        total_error += abs(e['abs_error'])  # Use absolute value for average error calculation
    print(f"Average error: {total_error / len(errors):.{decimal_precision}f}")
    print(f"\nFunction: {args.function}")
    print(f"LSB In: {args.lsb_in}")
    print(f"LSB Out: {args.lsb_out}")
