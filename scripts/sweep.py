#!/usr/bin/env python3
"""Hyperparameter sweep: find optimal attack parameters to break Tree-Ring watermarks.

For each watermarked image × each parameter combination:
1. Run img2img regeneration attack
2. DDIM-invert the attacked image
3. Run Fourier-space detection
4. Record quality metrics (SSIM, PSNR) and detection results (p-value)

Output: DW_RESULTS_DIR/sweep.csv

Usage:
    python scripts/sweep.py                           # Full sweep (60 combos × 10 images)
    python scripts/sweep.py --max-images 3            # Quick test (60 × 3)
    python scripts/sweep.py --strengths 0.3 0.4 0.5   # Custom strength values
"""

import os
import sys
import json
import argparse
import time
from itertools import product

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import pandas as pd
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from watermark.tree_ring import generate_key, create_ring_mask
from watermark.detection import detect_watermark
from watermark.inversion import load_pipeline
from attack.regeneration import regeneration_attack

IMAGE_DIR = os.environ.get("DW_IMAGE_DIR", str(Path(__file__).resolve().parent.parent / "images"))
MODEL_DIR = os.environ.get("DW_MODEL_DIR", str(Path(__file__).resolve().parent.parent / ".models"))
RESULTS_DIR = os.environ.get("DW_RESULTS_DIR", str(Path(__file__).resolve().parent.parent / "results"))

# Default sweep grid
DEFAULT_STRENGTHS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
DEFAULT_GUIDANCE_SCALES = [1.0, 3.0, 7.5]
DEFAULT_STEPS = [20, 50]


def run_sweep(
    max_images: int | None = None,
    strengths: list[float] | None = None,
    guidance_scales: list[float] | None = None,
    step_counts: list[int] | None = None,
    detection_steps: int = 50,
):
    """Run the full hyperparameter sweep.

    Args:
        max_images: Cap on number of images to process (None = all).
        strengths: List of strength values to sweep.
        guidance_scales: List of guidance scale values.
        step_counts: List of num_inference_steps values.
        detection_steps: DDIM steps for inversion/detection.
    """
    if strengths is None:
        strengths = DEFAULT_STRENGTHS
    if guidance_scales is None:
        guidance_scales = DEFAULT_GUIDANCE_SCALES
    if step_counts is None:
        step_counts = DEFAULT_STEPS

    # Load metadata
    meta_path = Path(IMAGE_DIR) / "metadata.json"
    if not meta_path.exists():
        print(f"No metadata found at {meta_path}. Run scripts/generate.py first.")
        sys.exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)

    wm_params = metadata["watermark_params"]
    latent_size = wm_params["latent_size"]

    # Regenerate key and mask from saved params
    key = generate_key(
        latent_size, latent_size,
        radius=wm_params["radius"],
        ring_width=wm_params["ring_width"],
        seed=wm_params["seed"],
    )
    mask = create_ring_mask(
        latent_size, latent_size,
        radius=wm_params["radius"],
        ring_width=wm_params["ring_width"],
    )

    images = metadata["images"]
    if max_images is not None:
        images = images[:max_images]

    combos = list(product(strengths, guidance_scales, step_counts))
    total_runs = len(images) * len(combos)

    print(f"Sweep: {len(images)} images × {len(combos)} combinations = {total_runs} runs")
    print(f"Strengths: {strengths}")
    print(f"Guidance scales: {guidance_scales}")
    print(f"Step counts: {step_counts}")

    # Load pipeline
    print(f"\nLoading SD 2.1 (cache: {MODEL_DIR})...")
    pipe = load_pipeline(cache_dir=MODEL_DIR)

    # Prepare output
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    results = []
    run_count = 0
    start_time = time.time()

    for img_info in images:
        idx = img_info["index"]
        wm_path = img_info["watermarked_path"]
        prompt = img_info["prompt"]

        if not os.path.exists(wm_path):
            print(f"  Image not found: {wm_path}, skipping")
            continue

        wm_image = Image.open(wm_path).convert("RGB")
        print(f"\n--- Image {idx}: {prompt[:50]}... ---")

        for strength, guidance_scale, num_steps in combos:
            run_count += 1
            elapsed = time.time() - start_time
            rate = run_count / elapsed if elapsed > 0 else 0
            eta = (total_runs - run_count) / rate if rate > 0 else 0

            print(f"  [{run_count}/{total_runs}] "
                  f"s={strength:.2f} g={guidance_scale:.1f} n={num_steps} "
                  f"(ETA: {eta/60:.0f}min)", end="", flush=True)

            try:
                # Attack
                attacked, metrics = regeneration_attack(
                    pipe, wm_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    prompt="",
                )

                # Detect watermark in attacked image
                detection = detect_watermark(
                    pipe, attacked, key, mask,
                    num_inference_steps=detection_steps,
                )

                results.append({
                    "image_index": idx,
                    "prompt": prompt,
                    "strength": strength,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_steps,
                    "ssim": metrics["ssim"],
                    "psnr": metrics["psnr"],
                    "p_value": detection["p_value"],
                    "l1_distance": detection["l1_distance"],
                    "detected": detection["detected"],
                })
                print(f" → SSIM={metrics['ssim']:.3f} p={detection['p_value']:.4f} "
                      f"{'DETECTED' if detection['detected'] else 'BROKEN'}")

            except Exception as e:
                print(f" → ERROR: {e}")
                results.append({
                    "image_index": idx,
                    "prompt": prompt,
                    "strength": strength,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_steps,
                    "ssim": None,
                    "psnr": None,
                    "p_value": None,
                    "l1_distance": None,
                    "detected": None,
                    "error": str(e),
                })

            # Save intermediate results after each run
            df = pd.DataFrame(results)
            csv_path = Path(RESULTS_DIR) / "sweep.csv"
            df.to_csv(csv_path, index=False)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Sweep complete: {run_count} runs in {total_time/60:.1f} minutes")
    print(f"Results saved to {csv_path}")

    # Quick summary
    df = pd.DataFrame(results)
    broken = df[df["detected"] == False]
    if len(broken) > 0:
        best = broken.sort_values("ssim", ascending=False).iloc[0]
        print(f"\nBest attack (watermark broken, highest quality):")
        print(f"  strength={best['strength']}, guidance={best['guidance_scale']}, "
              f"steps={best['num_inference_steps']}")
        print(f"  SSIM={best['ssim']:.3f}, PSNR={best['psnr']:.1f}, "
              f"p_value={best['p_value']:.4f}")
    else:
        print("\nNo parameter combination successfully broke the watermark!")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for watermark attack")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Max images to process (default: all)")
    parser.add_argument("--strengths", type=float, nargs="+", default=None)
    parser.add_argument("--guidance-scales", type=float, nargs="+", default=None)
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    parser.add_argument("--detection-steps", type=int, default=50)
    args = parser.parse_args()

    run_sweep(
        max_images=args.max_images,
        strengths=args.strengths,
        guidance_scales=args.guidance_scales,
        step_counts=args.steps,
        detection_steps=args.detection_steps,
    )


if __name__ == "__main__":
    main()
