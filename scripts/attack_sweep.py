#!/usr/bin/env python3
"""Attack sweep: grid-search img2img regeneration params against VideoSeal.

Loads watermarked images from the images dir, applies regeneration attack
at each (strength, guidance_scale) combo, then measures:
  - Image quality: SSIM, PSNR (higher = less distortion)
  - Watermark survival: p-value, bit_accuracy (higher p = weaker watermark)

Usage:
    python scripts/attack_sweep.py                        # default coarse grid
    python scripts/attack_sweep.py --max-images 2         # quick test
    python scripts/attack_sweep.py --embed-strength 0.2   # attack weaker embeddings
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Load .env before any other imports that might read env vars
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from watermark.adapters import create_adapter
from watermark.inversion import load_pipeline
from attack.regeneration import regeneration_attack

# Directories from env vars
MODEL_DIR = os.environ.get("DW_MODEL_DIR", str(Path(__file__).resolve().parent.parent / ".models"))
IMAGE_DIR = os.environ.get("DW_IMAGE_DIR", str(Path(__file__).resolve().parent.parent / "images"))
RESULTS_DIR = os.environ.get("DW_RESULTS_DIR", str(Path(__file__).resolve().parent.parent / "results"))

# Default coarse grid
DEFAULT_STRENGTHS = [0.3, 0.5, 0.7]
DEFAULT_GUIDANCES = [1.0, 7.5]
DEFAULT_STEPS = 50


def find_watermarked_images(
    method: str, dataset: str, embed_strength: float, max_images: int | None = None,
) -> list[Path]:
    """Find watermarked images for a given method/dataset/strength."""
    img_dir = Path(IMAGE_DIR) / method / dataset / f"s{embed_strength}"
    if not img_dir.exists():
        print(f"Image dir not found: {img_dir}")
        sys.exit(1)

    images = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() == ".png" and not p.name.startswith(".")
    )
    if max_images is not None:
        images = images[:max_images]
    return images


def run_sweep(
    attack_strengths: list[float],
    guidance_scales: list[float],
    num_steps: int,
    method: str,
    dataset: str,
    embed_strength: float,
    max_images: int | None,
    seed: int,
) -> Path:
    """Run the attack grid search and write results CSV.

    Returns:
        Path to the output CSV file.
    """
    images = find_watermarked_images(method, dataset, embed_strength, max_images)
    n_combos = len(attack_strengths) * len(guidance_scales)
    n_total = n_combos * len(images)

    print(f"Attack sweep: {method}/{dataset} (embed s={embed_strength})")
    print(f"  {len(images)} images × {n_combos} param combos = {n_total} runs")
    print(f"  Attack strengths: {attack_strengths}")
    print(f"  Guidance scales:  {guidance_scales}")
    print(f"  Inference steps:  {num_steps}")

    # Load SD 2.1 pipeline for the attack
    print(f"\nLoading SD pipeline (cache: {MODEL_DIR})...")
    pipe = load_pipeline(cache_dir=MODEL_DIR)

    # Load VideoSeal adapter for detection
    print("Loading VideoSeal detector...")
    detector = create_adapter(method, seed=seed)

    # Prepare output CSV
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"attack_{method}_{dataset}_s{embed_strength}.csv"

    fieldnames = [
        "image", "embed_strength", "attack_strength", "guidance_scale",
        "num_steps", "ssim", "psnr", "p_value", "bit_accuracy",
        "matching_bits", "detected", "time_s",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        count = 0
        sweep_start = time.time()

        for atk_str in attack_strengths:
            for guidance in guidance_scales:
                for img_path in images:
                    count += 1
                    elapsed = time.time() - sweep_start
                    rate = count / elapsed if elapsed > 1 else 0
                    eta = (n_total - count) / rate if rate > 0 else 0

                    print(
                        f"  [{count}/{n_total}] {img_path.name} "
                        f"atk={atk_str} cfg={guidance} "
                        f"(ETA: {eta:.0f}s)",
                        end="", flush=True,
                    )

                    img = Image.open(img_path).convert("RGB")

                    t0 = time.time()
                    attacked_img, quality = regeneration_attack(
                        pipe, img,
                        strength=atk_str,
                        guidance_scale=guidance,
                        num_inference_steps=num_steps,
                        seed=seed,
                    )
                    attack_time = time.time() - t0

                    # Detect watermark on attacked image
                    detection = detector.extract(attacked_img)

                    row = {
                        "image": img_path.name,
                        "embed_strength": embed_strength,
                        "attack_strength": atk_str,
                        "guidance_scale": guidance,
                        "num_steps": num_steps,
                        "ssim": f"{quality['ssim']:.4f}",
                        "psnr": f"{quality['psnr']:.2f}",
                        "p_value": f"{detection['p_value']:.2e}",
                        "bit_accuracy": f"{detection['bit_accuracy']:.4f}",
                        "matching_bits": detection["matching_bits"],
                        "detected": detection["detected"],
                        "time_s": f"{attack_time:.1f}",
                    }
                    writer.writerow(row)
                    f.flush()

                    status = "DETECTED" if detection["detected"] else "REMOVED"
                    print(
                        f" → {status} "
                        f"(acc={detection['bit_accuracy']:.2%} "
                        f"ssim={quality['ssim']:.3f} "
                        f"{attack_time:.1f}s)"
                    )

    total_time = time.time() - sweep_start
    print(f"\nDone: {count} runs in {total_time:.0f}s ({total_time/count:.1f}s/run)")
    print(f"Results: {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Attack sweep against watermarked images")
    parser.add_argument(
        "--method", type=str, default="videoseal",
        choices=["videoseal", "tree-ring", "stable-signature"],
    )
    parser.add_argument("--dataset", type=str, default="diffusiondb")
    parser.add_argument("--embed-strength", type=float, default=0.3,
                        help="Embedding strength of the watermarked images to attack")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Max images to attack per combo (default: all)")
    parser.add_argument("--attack-strengths", type=float, nargs="+",
                        default=DEFAULT_STRENGTHS)
    parser.add_argument("--guidance-scales", type=float, nargs="+",
                        default=DEFAULT_GUIDANCES)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_sweep(
        attack_strengths=args.attack_strengths,
        guidance_scales=args.guidance_scales,
        num_steps=args.num_steps,
        method=args.method,
        dataset=args.dataset,
        embed_strength=args.embed_strength,
        max_images=args.max_images,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
