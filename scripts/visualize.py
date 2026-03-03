#!/usr/bin/env python3
"""Visualize sweep results: quality vs detection tradeoff + Pareto frontier.

Reads DW_RESULTS_DIR/sweep.csv and generates:
- pareto.png: SSIM vs p-value scatter with Pareto frontier
- summary.png: Heatmap of detection rate by strength × guidance_scale

Usage:
    python scripts/visualize.py
    python scripts/visualize.py --csv results/sweep.csv --output results/
"""

import os
import sys
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = os.environ.get("DW_RESULTS_DIR", str(Path(__file__).resolve().parent.parent / "results"))


def compute_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Find Pareto-optimal points: max SSIM for each level of watermark destruction.

    A point is Pareto-optimal if no other point has both higher SSIM and higher p-value.
    """
    # Sort by p-value descending
    sorted_df = df.sort_values("p_value", ascending=False).reset_index(drop=True)

    pareto = []
    max_ssim = -1

    for _, row in sorted_df.iterrows():
        if row["ssim"] > max_ssim:
            pareto.append(row)
            max_ssim = row["ssim"]

    return pd.DataFrame(pareto)


def plot_pareto(df: pd.DataFrame, output_path: str):
    """Plot SSIM vs p-value scatter with Pareto frontier."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Filter out errors
    valid = df.dropna(subset=["ssim", "p_value"])

    # Scatter plot colored by strength
    scatter = ax.scatter(
        valid["ssim"],
        valid["p_value"],
        c=valid["strength"],
        cmap="viridis",
        alpha=0.6,
        s=40,
        edgecolors="none",
    )

    # Pareto frontier
    # Average across images for each parameter combo
    avg = valid.groupby(["strength", "guidance_scale", "num_inference_steps"]).agg({
        "ssim": "mean",
        "p_value": "mean",
    }).reset_index()

    pareto = compute_pareto_frontier(avg)
    pareto_sorted = pareto.sort_values("ssim")
    ax.plot(
        pareto_sorted["ssim"],
        pareto_sorted["p_value"],
        "r-o",
        linewidth=2,
        markersize=6,
        label="Pareto frontier (avg)",
        zorder=5,
    )

    # Threshold lines
    ax.axhline(y=0.01, color="gray", linestyle="--", alpha=0.5, label="p=0.01 (detection threshold)")
    ax.axhline(y=0.05, color="gray", linestyle=":", alpha=0.5, label="p=0.05")

    # Labels
    ax.set_xlabel("SSIM (image quality, higher = better)", fontsize=12)
    ax.set_ylabel("p-value (higher = watermark more broken)", fontsize=12)
    ax.set_title("Watermark Attack: Quality vs Detection Tradeoff", fontsize=14)
    ax.set_yscale("log")

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Attack strength")

    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Pareto plot saved to {output_path}")
    plt.close()


def plot_heatmap(df: pd.DataFrame, output_path: str):
    """Plot heatmap: detection success rate by strength × guidance_scale."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    valid = df.dropna(subset=["detected"])

    for i, steps in enumerate(sorted(valid["num_inference_steps"].unique())):
        ax = axes[i] if len(axes) > 1 else axes
        subset = valid[valid["num_inference_steps"] == steps]

        # Pivot: fraction of images where watermark was broken
        pivot = subset.pivot_table(
            values="detected",
            index="strength",
            columns="guidance_scale",
            aggfunc=lambda x: 1 - x.mean(),  # Fraction broken (detected=False)
        )

        im = ax.imshow(
            pivot.values,
            cmap="RdYlGn",
            aspect="auto",
            vmin=0,
            vmax=1,
            origin="lower",
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{g:.1f}" for g in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{s:.2f}" for s in pivot.index])
        ax.set_xlabel("Guidance Scale")
        ax.set_ylabel("Strength")
        ax.set_title(f"Watermark Break Rate (steps={steps})")

        # Annotate cells
        for y in range(len(pivot.index)):
            for x in range(len(pivot.columns)):
                val = pivot.values[y, x]
                color = "white" if val > 0.5 else "black"
                ax.text(x, y, f"{val:.0%}", ha="center", va="center",
                        color=color, fontsize=9)

        plt.colorbar(im, ax=ax, label="Break rate")

    plt.suptitle("Attack Success Rate by Hyperparameters", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to {output_path}")
    plt.close()


def print_summary(df: pd.DataFrame):
    """Print summary statistics to stdout."""
    valid = df.dropna(subset=["ssim", "p_value"])

    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    print(f"Total runs: {len(valid)}")
    print(f"Unique images: {valid['image_index'].nunique()}")

    broken = valid[valid["detected"] == False]
    print(f"\nWatermark broken: {len(broken)}/{len(valid)} runs ({len(broken)/len(valid)*100:.1f}%)")

    if len(broken) > 0:
        # Best by quality (highest SSIM among broken)
        avg_broken = broken.groupby(["strength", "guidance_scale", "num_inference_steps"]).agg({
            "ssim": "mean",
            "psnr": "mean",
            "p_value": "mean",
        }).reset_index()

        best_quality = avg_broken.sort_values("ssim", ascending=False).iloc[0]
        print(f"\nBest quality (watermark broken, avg across images):")
        print(f"  strength={best_quality['strength']:.2f}, "
              f"guidance={best_quality['guidance_scale']:.1f}, "
              f"steps={int(best_quality['num_inference_steps'])}")
        print(f"  SSIM={best_quality['ssim']:.3f}, "
              f"PSNR={best_quality['psnr']:.1f}dB, "
              f"p-value={best_quality['p_value']:.4f}")

        # Recommended settings for SynthID transfer
        print(f"\n{'='*60}")
        print("RECOMMENDED SETTINGS FOR SYNTHID TRANSFER:")
        print(f"  strength = {best_quality['strength']:.2f}")
        print(f"  guidance_scale = {best_quality['guidance_scale']:.1f}")
        print(f"  num_inference_steps = {int(best_quality['num_inference_steps'])}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Visualize sweep results")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to sweep.csv")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for plots")
    args = parser.parse_args()

    csv_path = args.csv or os.path.join(RESULTS_DIR, "sweep.csv")
    output_dir = args.output or RESULTS_DIR

    if not os.path.exists(csv_path):
        print(f"No sweep results found at {csv_path}. Run scripts/sweep.py first.")
        sys.exit(1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")

    plot_pareto(df, os.path.join(output_dir, "pareto.png"))
    plot_heatmap(df, os.path.join(output_dir, "heatmap.png"))
    print_summary(df)


if __name__ == "__main__":
    main()
