#!/usr/bin/env python3
"""Sandwich attack sweep against proxy watermarks (VideoSeal, Tree-Ring, Stable Sig).

Runs sandwich_attack (diffuse → ring-rotate → diffuse) at multiple
(strength_1, strength_2) combos against watermarked images produced by embed.py.
Detection uses the unified adapter interface.

Usage:
    python scripts/attack_sandwich_sweep.py --max-images 2       # quick test
    python scripts/attack_sandwich_sweep.py --method videoseal   # full sweep
    python scripts/attack_sandwich_sweep.py --method tree-ring --dataset div2k
"""

import gc
import os
import sys
import csv
import time
import argparse
from pathlib import Path

import torch

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from watermark.adapters import create_adapter
from watermark.inversion import load_pipeline
from attack.regeneration import sandwich_attack

# Directories from env vars
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = os.environ.get("DW_MODEL_DIR", str(PROJECT_ROOT / ".models"))
IMAGE_DIR = os.environ.get("DW_IMAGE_DIR", str(PROJECT_ROOT / "images"))
RESULTS_DIR = os.environ.get("DW_RESULTS_DIR", str(PROJECT_ROOT / "results"))

# Default sweep grid: (strength_1, strength_2) combos
# Total diffusion budget comparable to single-pass s=0.20–0.40
DEFAULT_COMBOS = [
    (0.10, 0.10),  # total ≈ 0.20
    (0.15, 0.10),  # total ≈ 0.25
    (0.10, 0.15),  # total ≈ 0.25 (asymmetric)
    (0.15, 0.15),  # total ≈ 0.30
    (0.20, 0.15),  # total ≈ 0.35
    (0.20, 0.20),  # total ≈ 0.40
]
DEFAULT_STEPS = 50
GUIDANCE = 1.0


def find_watermarked_images(
    method: str, dataset: str, embed_strength: float, max_images: int | None = None,
) -> list[Path]:
    """Find watermarked images for a given method/dataset/strength."""
    img_dir = Path(IMAGE_DIR) / method / dataset / f"s{embed_strength}"
    if not img_dir.exists():
        print(f"Image dir not found: {img_dir}")
        print(f"Run embed.py first: python scripts/embed.py --method {method} --input-dir datasets/{dataset}")
        sys.exit(1)

    images = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() == ".png" and not p.name.startswith(".")
    )
    if max_images is not None:
        images = images[:max_images]
    return images


def run_sweep(
    combos: list[tuple[float, float]],
    num_rings: int,
    max_angle: float,
    num_steps: int,
    method: str,
    dataset: str,
    embed_strength: float,
    max_images: int | None,
    seed: int,
    tile_size: int,
    overlap: int,
    resume: bool = False,
) -> Path:
    """Run sandwich attack sweep and write results CSV."""
    images = find_watermarked_images(method, dataset, embed_strength, max_images)
    n_total = len(images) * len(combos)

    print(f"Sandwich Attack Sweep: {method}/{dataset} (embed s={embed_strength})")
    print(f"  {len(images)} images × {len(combos)} combos = {n_total} runs")
    print(f"  Combos (s1, s2): {combos}")
    print(f"  Ring: {num_rings} rings, ±{max_angle} deg")
    print(f"  Tile: {tile_size}px, overlap: {overlap}px")
    print(f"  Steps: {num_steps}, guidance: {GUIDANCE}")

    # Load SD 1.5 pipeline (needed for both attack and some detectors)
    print(f"\nLoading SD 1.5 pipeline (cache: {MODEL_DIR})...")
    pipe = load_pipeline(cache_dir=MODEL_DIR)

    # Load detector via adapter
    print(f"Loading {method} detector...")
    detector = create_adapter(method, pipe=pipe, model_dir=MODEL_DIR, seed=seed)

    # Prepare output CSV
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"sandwich_{method}_{dataset}_s{embed_strength}.csv"

    fieldnames = [
        "image", "embed_strength", "strength_1", "strength_2",
        "num_rings", "max_angle", "tile_size", "overlap",
        "ssim", "psnr",
        "p_value", "detected",
    ]
    # Add method-specific detection fields
    if method == "videoseal":
        fieldnames.extend(["bit_accuracy", "matching_bits"])
    elif method == "stable-signature":
        fieldnames.extend(["bit_accuracy", "matching_bits"])
    # tree-ring only has p_value and detected
    fieldnames.append("time_s")

    # Resume support
    done_keys: set[tuple[str, str, str]] = set()
    if resume and csv_path.exists():
        with open(csv_path, newline="") as rf:
            for row in csv.DictReader(rf):
                done_keys.add((row["image"], row["strength_1"], row["strength_2"]))
        print(f"  Resuming: {len(done_keys)} combos already done")

    mode = "a" if (resume and done_keys) else "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        count = 0
        sweep_start = time.time()

        for img_path in images:
            img = Image.open(img_path).convert("RGB")

            for s1, s2 in combos:
                key = (img_path.name, str(s1), str(s2))
                if key in done_keys:
                    continue

                count += 1
                elapsed = time.time() - sweep_start
                rate = count / elapsed if elapsed > 1 else 0
                eta = (n_total - count) / rate if rate > 0 else 0

                print(
                    f"  [{count}/{n_total}] {img_path.name} "
                    f"s1={s1:.2f} ring s2={s2:.2f} "
                    f"(ETA: {eta:.0f}s)",
                    end="", flush=True,
                )

                t0 = time.time()
                attacked_img, quality = sandwich_attack(
                    pipe, img,
                    strength_1=s1,
                    strength_2=s2,
                    num_rings=num_rings,
                    max_angle=max_angle,
                    tile_size=tile_size,
                    overlap=overlap,
                    guidance_scale=GUIDANCE,
                    num_inference_steps=num_steps,
                    seed=seed,
                )
                attack_time = time.time() - t0

                # Detect watermark on attacked image
                detection = detector.extract(attacked_img)
                total_time = time.time() - t0

                row = {
                    "image": img_path.name,
                    "embed_strength": embed_strength,
                    "strength_1": s1,
                    "strength_2": s2,
                    "num_rings": num_rings,
                    "max_angle": max_angle,
                    "tile_size": tile_size,
                    "overlap": overlap,
                    "ssim": f"{quality['ssim']:.4f}",
                    "psnr": f"{quality['psnr']:.2f}",
                    "p_value": f"{detection['p_value']:.2e}",
                    "detected": detection["detected"],
                    "time_s": f"{total_time:.1f}",
                }
                if method in ("videoseal", "stable-signature"):
                    row["bit_accuracy"] = f"{detection['bit_accuracy']:.4f}"
                    row["matching_bits"] = detection["matching_bits"]

                writer.writerow(row)
                f.flush()

                status = "DETECTED" if detection["detected"] else "REMOVED"
                det_detail = (
                    f"acc={detection['bit_accuracy']:.2%}" if "bit_accuracy" in detection
                    else f"p={detection['p_value']:.2e}"
                )
                print(
                    f" -> {status} "
                    f"({det_detail} "
                    f"ssim={quality['ssim']:.3f} "
                    f"{total_time:.1f}s)"
                )

                del attacked_img, detection, quality
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            del img
            gc.collect()

    total_time = time.time() - sweep_start
    print(f"\nDone: {count} sandwich attacks in {total_time:.0f}s")
    print(f"Results: {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Sandwich attack sweep: diffuse -> ring-rotate -> diffuse (proxy watermarks)"
    )
    parser.add_argument(
        "--method", type=str, default="videoseal",
        choices=["videoseal", "tree-ring", "stable-signature"],
    )
    parser.add_argument("--dataset", type=str, default="diffusiondb")
    parser.add_argument("--embed-strength", type=float, default=0.3,
                        help="Embedding strength of the watermarked images to attack")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--combos", type=str, default=None,
        help="Comma-separated s1:s2 pairs, e.g. '0.10:0.10,0.15:0.15'",
    )
    parser.add_argument("--num-rings", type=int, default=15)
    parser.add_argument("--max-angle", type=float, default=1.5)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.combos:
        combos = []
        for pair in args.combos.split(","):
            s1, s2 = pair.strip().split(":")
            combos.append((float(s1), float(s2)))
    else:
        combos = DEFAULT_COMBOS

    run_sweep(
        combos=combos,
        num_rings=args.num_rings,
        max_angle=args.max_angle,
        num_steps=args.num_steps,
        method=args.method,
        dataset=args.dataset,
        embed_strength=args.embed_strength,
        max_images=args.max_images,
        seed=args.seed,
        tile_size=args.tile_size,
        overlap=args.overlap,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
