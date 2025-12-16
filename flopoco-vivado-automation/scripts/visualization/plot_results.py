"""
Plot analysis results from results.csv.

This script creates multiple visualizations to analyze the relationship between
design parameters (lsb_in, lsb_out, epsilon) and outcomes (resources, error).

Usage:
    python plot_results.py [--csv results.csv] [--output plots/]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib numpy")
    raise

# Try to import seaborn for nicer plots (optional)
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    USE_SEABORN = True
except ImportError:
    USE_SEABORN = False
    print("Note: seaborn not available. Using matplotlib defaults. Install with: pip install seaborn")


def load_results(csv_path: Path) -> list[dict]:
    """Load results from CSV file."""
    results = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row["epsilon"] = float(row["epsilon"]) if row["epsilon"] else None
            row["lsb_in"] = int(row["lsb_in"])
            row["lsb_out"] = int(row["lsb_out"])
            # msb_out might not be in old CSV files, so handle it gracefully
            if "msb_out" in row and row["msb_out"]:
                row["msb_out"] = int(row["msb_out"])
            else:
                row["msb_out"] = None
            row["LUTs"] = int(row["LUTs"])
            row["FFs"] = int(row["FFs"])
            row["DSPs"] = int(row["DSPs"])
            row["BRAMs"] = float(row["BRAMs"])
            row["avg_error"] = float(row["avg_error"])
            results.append(row)
    return results


def create_plots(results: list[dict], output_dir: Path, filter_epsilon: float | None = None) -> None:
    """Create all visualization plots.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save plots
        filter_epsilon: If provided, filter results to only this epsilon value
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results:
        print("No results to plot!")
        return
    
    # Filter by epsilon if specified
    if filter_epsilon is not None:
        results = [r for r in results if r["epsilon"] is not None and abs(r["epsilon"] - filter_epsilon) < 1e-10]
        if not results:
            print(f"No results found for epsilon={filter_epsilon}")
            return
        print(f"Filtered to {len(results)} result(s) with epsilon={filter_epsilon}")
    
    # Extract data arrays
    lsb_ins = [r["lsb_in"] for r in results]
    lsb_outs = [r["lsb_out"] for r in results]
    msb_outs = [r["msb_out"] for r in results if r["msb_out"] is not None]
    luts = [r["LUTs"] for r in results]
    ffs = [r["FFs"] for r in results]
    dsps = [r["DSPs"] for r in results]
    brams = [r["BRAMs"] for r in results]
    avg_errors = [r["avg_error"] for r in results]
    epsilons = [r["epsilon"] if r["epsilon"] is not None else 0 for r in results]
    
    # Create configuration labels
    config_labels = [f"lsb_in={li}, lsb_out={lo}" for li, lo in zip(lsb_ins, lsb_outs)]
    
    # 1. Resource comparison (bar chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results))
    width = 0.2
    
    ax.bar(x - 1.5*width, luts, width, label="LUTs", alpha=0.8)
    ax.bar(x - 0.5*width, ffs, width, label="FFs", alpha=0.8)
    ax.bar(x + 0.5*width, dsps, width, label="DSPs", alpha=0.8)
    ax.bar(x + 1.5*width, brams, width, label="BRAMs", alpha=0.8)
    
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Resource Count")
    ax.set_title("Resource Usage Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / "resource_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'resource_comparison.png'}")
    plt.close()
    
    # 3. Error vs LSB positions (scatter/line plot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Average Error vs LSB Positions", fontsize=16, fontweight="bold")
    
    axes[0].scatter(lsb_ins, avg_errors, s=150, alpha=0.7, edgecolors="black", c=lsb_outs, cmap="viridis")
    axes[0].set_xlabel("LSB In")
    axes[0].set_ylabel("Average Error")
    axes[0].set_title("Error vs LSB In (colored by LSB Out)")
    axes[0].grid(True, alpha=0.3)
    cbar0 = plt.colorbar(axes[0].collections[0], ax=axes[0])
    cbar0.set_label("LSB Out")
    for i, label in enumerate(config_labels):
        axes[0].annotate(f"({lsb_ins[i]},{lsb_outs[i]})", 
                        (lsb_ins[i], avg_errors[i]), fontsize=8, alpha=0.7)
    
    axes[1].scatter(lsb_outs, avg_errors, s=150, alpha=0.7, edgecolors="black", c=lsb_ins, cmap="plasma")
    axes[1].set_xlabel("LSB Out")
    axes[1].set_ylabel("Average Error")
    axes[1].set_title("Error vs LSB Out (colored by LSB In)")
    axes[1].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(axes[1].collections[0], ax=axes[1])
    cbar1.set_label("LSB In")
    for i, label in enumerate(config_labels):
        axes[1].annotate(f"({lsb_ins[i]},{lsb_outs[i]})", 
                        (lsb_outs[i], avg_errors[i]), fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / "error_vs_lsb.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'error_vs_lsb.png'}")
    plt.close()
    
    # 3. Total resource usage (stacked bar or pie chart)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Stacked bar chart showing all resources
    x = np.arange(len(results))
    bottom = np.zeros(len(results))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    for i, (resource, color) in enumerate(zip(["LUTs", "FFs", "DSPs", "BRAMs"], colors)):
        values = [r[resource] for r in results]
        axes[0].bar(x, values, bottom=bottom, label=resource, color=color, alpha=0.8)
        bottom += values
    
    axes[0].set_xlabel("Configuration")
    axes[0].set_ylabel("Total Resources")
    axes[0].set_title("Total Resource Usage (Stacked)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(config_labels, rotation=45, ha="right")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")
    
    # Error comparison bar chart
    axes[1].bar(x, avg_errors, alpha=0.7, color="purple", edgecolor="black")
    axes[1].set_xlabel("Configuration")
    axes[1].set_ylabel("Average Error")
    axes[1].set_title("Average Error Comparison")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(config_labels, rotation=45, ha="right")
    axes[1].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / "resource_error_summary.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'resource_error_summary.png'}")
    plt.close()
    
    # 4. MSB Out vs LSB Out with error visualization
    # Filter results that have msb_out data
    results_with_msb = [r for r in results if r["msb_out"] is not None]
    if results_with_msb:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        msb_outs_plot = [r["msb_out"] for r in results_with_msb]
        lsb_outs_plot = [r["lsb_out"] for r in results_with_msb]
        errors_plot = [r["avg_error"] for r in results_with_msb]
        
        # MSB Out vs LSB Out, colored by error
        scatter = ax.scatter(lsb_outs_plot, msb_outs_plot, s=200, alpha=0.7, 
                            c=errors_plot, cmap="viridis", edgecolors="black", linewidth=1.5)
        ax.set_xlabel("LSB Out")
        ax.set_ylabel("MSB Out")
        ax.set_title("MSB Out vs LSB Out (colored by Average Error)")
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Average Error")
        
        # Add annotations with configuration info
        for i, r in enumerate(results_with_msb):
            ax.annotate(f"({r['lsb_in']},{r['lsb_out']})", 
                       (lsb_outs_plot[i], msb_outs_plot[i]), 
                       fontsize=7, alpha=0.6, ha="center")
        
        plt.tight_layout()
        plt.savefig(output_dir / "msb_out_vs_lsb_out.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / 'msb_out_vs_lsb_out.png'}")
        plt.close()
    else:
        print("Note: No msb_out data found, skipping MSB Out vs LSB Out plot")
    
    # 5. MSB Out vs Epsilon plot
    # Filter results that have both msb_out and epsilon data
    results_with_msb_epsilon = [r for r in results if r["msb_out"] is not None and r["epsilon"] is not None]
    if results_with_msb_epsilon:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        msb_outs_plot = [r["msb_out"] for r in results_with_msb_epsilon]
        epsilons_plot = [r["epsilon"] for r in results_with_msb_epsilon]
        errors_plot = [r["avg_error"] for r in results_with_msb_epsilon]
        
        # MSB Out vs Epsilon, colored by error
        scatter = ax.scatter(epsilons_plot, msb_outs_plot, s=200, alpha=0.7, 
                            c=errors_plot, cmap="viridis", edgecolors="black", linewidth=1.5)
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("MSB Out")
        ax.set_title("MSB Out vs Epsilon (colored by Average Error)")
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Average Error")
        
        # Add annotations with configuration info
        for i, r in enumerate(results_with_msb_epsilon):
            ax.annotate(f"({r['lsb_in']},{r['lsb_out']})", 
                       (epsilons_plot[i], msb_outs_plot[i]), 
                       fontsize=7, alpha=0.6, ha="center")
        
        # Use log scale for x-axis if epsilon values span orders of magnitude
        if max(epsilons_plot) / min(epsilons_plot) > 100:
            ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / "msb_out_vs_epsilon.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / 'msb_out_vs_epsilon.png'}")
        plt.close()
    else:
        print("Note: No msb_out and epsilon data found, skipping MSB Out vs Epsilon plot")
    
    # 6. Epsilon vs Error plot
    # Filter results that have epsilon data
    results_with_epsilon = [r for r in results if r["epsilon"] is not None]
    if results_with_epsilon:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epsilons_plot = [r["epsilon"] for r in results_with_epsilon]
        errors_plot = [r["avg_error"] for r in results_with_epsilon]
        lsb_ins_plot = [r["lsb_in"] for r in results_with_epsilon]
        lsb_outs_plot = [r["lsb_out"] for r in results_with_epsilon]
        
        # Epsilon vs Error, colored by lsb_in and lsb_out combination
        # Create a unique identifier for each lsb_in/lsb_out combination for coloring
        unique_configs = {}
        config_id = 0
        config_ids = []
        for li, lo in zip(lsb_ins_plot, lsb_outs_plot):
            config_key = (li, lo)
            if config_key not in unique_configs:
                unique_configs[config_key] = config_id
                config_id += 1
            config_ids.append(unique_configs[config_key])
        
        scatter = ax.scatter(epsilons_plot, errors_plot, s=200, alpha=0.7, 
                            c=config_ids, cmap="tab20", edgecolors="black", linewidth=1.5)
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Average Error")
        ax.set_title("Average Error vs Epsilon")
        ax.grid(True, alpha=0.3)
        
        # Add legend for configurations
        if len(unique_configs) <= 20:  # Only show legend if reasonable number of configs
            handles = []
            labels = []
            for (li, lo), cid in sorted(unique_configs.items(), key=lambda x: x[1]):
                color = plt.cm.tab20(cid / max(1, len(unique_configs) - 1))
                handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=color, markersize=10, 
                                         markeredgecolor='black', markeredgewidth=1.5))
                labels.append(f"lsb_in={li}, lsb_out={lo}")
            ax.legend(handles, labels, loc='best', fontsize=8, ncol=2)
        
        # Add annotations with epsilon values
        for i, r in enumerate(results_with_epsilon):
            ax.annotate(f"ε={r['epsilon']:.2e}", 
                       (epsilons_plot[i], errors_plot[i]), 
                       fontsize=7, alpha=0.6, ha="left", va="bottom")
        
        # Use log scale for x-axis if epsilon values span orders of magnitude
        if max(epsilons_plot) / min(epsilons_plot) > 100:
            ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / "epsilon_vs_error.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / 'epsilon_vs_error.png'}")
        plt.close()
    else:
        print("Note: No epsilon data found, skipping Epsilon vs Error plot")
    
    print(f"\nAll plots saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot analysis results from results.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).parent.parent.parent / "data" / "results.csv",
        help="Path to results CSV file (default: data/results.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent.parent / "plots",
        help="Output directory for plots (default: plots/)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Filter results to a specific epsilon value (optional). If not provided, plots all epsilon vs error data.",
    )
    args = parser.parse_args()
    
    if not args.csv.exists():
        print(f"Error: CSV file not found: {args.csv}")
        return
    
    print(f"Loading results from: {args.csv}")
    results = load_results(args.csv)
    print(f"Loaded {len(results)} result(s)")
    
    if args.epsilon is not None:
        print(f"\nFiltering results for epsilon={args.epsilon}")
    
    print(f"\nCreating plots in: {args.output}")
    create_plots(results, args.output, filter_epsilon=args.epsilon)
    print("\nDone!")


if __name__ == "__main__":
    main()

