# FloPoCo Hardware Design Flow Automation

This repository automates the complete hardware design flow from FloPoCo VHDL generation through Vivado synthesis/implementation, simulation, and verification.

## Repository Structure

```
py_scripts/
├── scripts/
│   ├── automation/          # Main automation scripts
│   │   ├── automate_flow.py    # Main orchestration script
│   │   └── call_vivado.py       # Vivado batch mode launcher
│   ├── analysis/            # Analysis and verification scripts
│   │   ├── compute_errors.py
│   │   ├── verify_fixfunction_outputs.py
│   │   └── read_util_report.py
│   ├── visualization/      # Plotting scripts
│   │   └── plot_results.py
│   └── vivado/             # Vivado TCL scripts
│       ├── project_free_flow.tcl
│       └── run_tb.tcl
│
├── hdl/                    # Hardware description files
│   ├── flopoco.vhdl        # Generated VHDL (from FloPoCo)
│   └── tb_FixFunctionByTable.vhdl  # Testbench
│
├── constraints/            # Xilinx Design Constraints
│   └── top.xdc
│
├── data/                   # Data files (CSV)
│   ├── results.csv         # Main results log
│   └── *.csv               # Other data files
│
├── build/                  # Vivado build outputs
│   ├── logs/
│   ├── reports/
│   └── *.dcp
│
├── simulation/             # Simulation outputs
│   └── FixFunctionByTable_outputs.csv
│
└── plots/                  # Generated plots
    ├── epsilon_var/
    ├── lsbIn_var/
    └── lsbOut_var/
```

## Quick Start

### Prerequisites

1. **FloPoCo running in Docker** (or local installation)
2. **Xilinx Vivado** installed and environment sourced

### Setting Up Vivado Environment

Before running the automation flow, you must source the Vivado settings script to set up the environment:

**On Linux/Mac (Bash):**
```bash
source /d/Xilinx/2025.1.1/Vivado/settings64.sh
```

**Note:** Adjust the path to match your Vivado installation location.

### Running the Complete Flow

```bash
# First, source Vivado environment (see above)
source /d/Xilinx/2025.1.1/Vivado/settings64.sh

# Then run the automation
python scripts/automation/automate_flow.py \
    --function 'log(x+0.0001)/log(2)' \
    --lsb-in -6 \
    --lsb-out -6 \
    --part xc7k70tfbg484-3
```

### Plotting Results

```bash
python scripts/visualization/plot_results.py \
    --csv data/results.csv \
    --output plots/
```

## Scripts Overview

### Automation Scripts

- **`automate_flow.py`**: Main orchestration script that:
  - Generates VHDL using FloPoCo
  - Updates testbench and verifier scripts
  - Runs Vivado synthesis and implementation
  - Executes simulation
  - Verifies outputs and computes errors
  - Logs results to CSV

- **`call_vivado.py`**: Launches Vivado in batch mode with a TCL script

### Analysis Scripts

- **`compute_errors.py`**: Calculates detailed errors between simulation outputs and expected values
- **`verify_fixfunction_outputs.py`**: Verifies simulation outputs against mathematical function
- **`read_util_report.py`**: Parses Vivado utilization reports

### Visualization

- **`plot_results.py`**: Generates plots from results CSV (resource comparison, error analysis, etc.)

### Vivado Scripts

- **`project_free_flow.tcl`**: Project-free synthesis and implementation flow
- **`run_tb.tcl`**: Testbench simulation flow

## Requirements

- Python 3.8+
- Xilinx Vivado (must source `settings64.sh` or `settings64.bat` before running)
- FloPoCo (Docker container or local installation)
- Required Python packages: `matplotlib`, `numpy`, `pandas` (optional)

**Important:** The Vivado environment must be sourced in your shell before running the automation flow. This ensures that Vivado tools (`vivado`, `xvhdl`, `xelab`, `xsim`) are available in your PATH.

## Notes

- The `results.csv` file accumulates results from multiple runs

