#!/usr/bin/env python3
"""Ring rotation attack sweep against proxy watermarks.

Pure geometric attack — no diffusion model needed for the attack itself.
Fast: seconds per image on CPU.

Usage:
    python scripts/attack_ring_sweep.py --max-images 2               # quick test
    python scripts/attack_ring_sweep.py --method videoseal            # default grid
    python scripts/attack_ring_sweep.py --method tree-ring --dataset div2k
"""

import gc
import os
import sys
import csv
import time
import argparse
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from watermark.adapters import create_adapter
from attack.ring import ring_rotation_attack
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import numpy as np

# Directories from env vars
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = os.environ.get("DW_MODEL_DIR", str(PROJECT_ROOT / ".models"))
IMAGE_DIR = os.environ.get("DW_IMAGE_DIR", str(PROJECT_ROOT / "images"))
RESULTS_DIR = os.environ.get("DW_RESULTS_DIR", str(PROJECT_ROOT / "results"))

# Default sweep grid: (num_rings, max_angle) combos
DEFAULT_RING_PARAMS = [
    (5, 0.5),
    (5, 1.0),
    (5, 2.0),
    (10, 0.5),
    (10, 1.0),
    (10, 1.5),
    (10, 2.0),
    (15, 1.0),
    (15, 1.5),
    (15, 2.0),
    (20, 1.0),
    (20, 1.5),
    (20, 2.0),
    (25, 1.5),
    (25, 2.0),
]


def compute_quality(original: Image.Image, attacked: Image.Image) -> dict:
    """SSIM and PSNR between original and attacked images."""
    orig_np = np.array(original).astype(np.float64)
    atk_np = np.array(attacked).astype(np.float64)
    if orig_np.shape != atk_np.shape:
        attacked = attacked.resize(original.size, Image.LANCZOS)
        atk_np = np.array(attacked).astype(np.float64)
    ssim = structural_similarity(orig_np, atk_np, channel_axis=2, data_range=255)
    psnr = peak_signal_noise_ratio(orig_np, atk_np, data_range=255)
    return {"ssim": ssim, "psnr": psnr}


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
    ring_params: list[tuple[int, float]],
    method: str,
    dataset: str,
    embed_strength: float,
    max_images: int | None,
    seed: int,
    resume: bool = False,
) -> Path:
    """Run ring rotation attack sweep and write results CSV."""
    images = find_watermarked_images(method, dataset, embed_strength, max_images)
    n_total = len(images) * len(ring_params)

    print(f"Ring Rotation Attack Sweep: {method}/{dataset} (embed s={embed_strength})")
    print(f"  {len(images)} images × {len(ring_params)} param combos = {n_total} runs")
    print(f"  Ring params (rings, angle): {ring_params}")

    # Load detector — Tree-Ring needs the SD pipeline
    pipe = None
    if method in ("tree-ring", "stable-signature"):
        from watermark.inversion import load_pipeline
        print(f"\nLoading SD 1.5 pipeline for {method} detection (cache: {MODEL_DIR})...")
        pipe = load_pipeline(cache_dir=MODEL_DIR)

    print(f"Loading {method} detector...")
    detector = create_adapter(method, pipe=pipe, model_dir=MODEL_DIR, seed=seed)

    # Prepare output CSV
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"ring_{method}_{dataset}_s{embed_strength}.csv"

    fieldnames = [
        "image", "embed_strength", "num_rings", "max_angle",
        "ssim", "psnr",
        "p_value", "detected",
    ]
    if method in ("videoseal", "stable-signature"):
        fieldnames.extend(["bit_accuracy", "matching_bits"])
    fieldnames.append("time_s")

    # Resume support
    done_keys: set[tuple[str, str, str]] = set()
    if resume and csv_path.exists():
        with open(csv_path, newline="") as rf:
            for row in csv.DictReader(rf):
                done_keys.add((row["image"], row["num_rings"], row["max_angle"]))
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

            for num_rings, max_angle in ring_params:
                key = (img_path.name, str(num_rings), str(max_angle))
                if key in done_keys:
                    continue

                count += 1
                elapsed = time.time() - sweep_start
                rate = count / elapsed if elapsed > 1 else 0
                eta = (n_total - count) / rate if rate > 0 else 0

                print(
                    f"  [{count}/{n_total}] {img_path.name} "
                    f"rings={num_rings} angle={max_angle:.1f} "
                    f"(ETA: {eta:.0f}s)",
                    end="", flush=True,
                )

                t0 = time.time()
                attacked_img = ring_rotation_attack(
                    img, num_rings=num_rings, max_angle=max_angle,
                )
                quality = compute_quality(img, attacked_img)
                attack_time = time.time() - t0

                # Detect watermark
                detection = detector.extract(attacked_img)
                total_time = time.time() - t0

                row = {
                    "image": img_path.name,
                    "embed_strength": embed_strength,
                    "num_rings": num_rings,
                    "max_angle": max_angle,
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

            del img
            gc.collect()

    total_time = time.time() - sweep_start
    print(f"\nDone: {count} ring attacks in {total_time:.0f}s")
    print(f"Results: {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Ring rotation attack sweep (proxy watermarks)"
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
        "--ring-params", type=str, default=None,
        help="Comma-separated rings:angle pairs, e.g. '10:1.0,15:1.5'",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.ring_params:
        ring_params = []
        for pair in args.ring_params.split(","):
            r, a = pair.strip().split(":")
            ring_params.append((int(r), float(a)))
    else:
        ring_params = DEFAULT_RING_PARAMS

    run_sweep(
        ring_params=ring_params,
        method=args.method,
        dataset=args.dataset,
        embed_strength=args.embed_strength,
        max_images=args.max_images,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
