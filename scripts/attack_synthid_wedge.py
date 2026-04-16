#!/usr/bin/env python3
"""Wedge attack sweep: noise → ring-rotate → diffusion denoise against SynthID.

Adds pixel noise + ring rotation to destroy the watermark, then uses the
diffusion model purely as a denoiser to reconstruct a clean image.
One diffusion pass (half the cost of the sandwich attack).

Usage:
    python scripts/attack_synthid_wedge.py --max-images 3
    python scripts/attack_synthid_wedge.py
"""

import gc
import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path

import torch

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reverse-SynthID"))

import numpy as np
from PIL import Image
from watermark.inversion import load_pipeline
from attack.regeneration import wedge_attack
from src.extraction.robust_extractor import RobustSynthIDExtractor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "test-synthID" / "corpus"
OUTPUT_DIR = PROJECT_ROOT / "test-synthID"
_env_model_dir = os.environ.get("DW_MODEL_DIR", "")
MODEL_DIR = _env_model_dir if _env_model_dir and Path(_env_model_dir).exists() else str(PROJECT_ROOT / ".models")

# Sweep grid: (noise_sigma, denoise_strength)
DEFAULT_COMBOS = [
    (15, 0.15),
    (25, 0.20),
    (25, 0.30),
    (40, 0.30),
]
DEFAULT_STEPS = 50
GUIDANCE = 1.0


def load_corpus(max_images=None):
    meta_path = CORPUS_DIR / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)
    images = [CORPUS_DIR / e["filename"] for e in metadata
              if e.get("status") == "ok" and (CORPUS_DIR / e["filename"]).exists()]
    return images[:max_images] if max_images else images


def load_detector():
    extractor = RobustSynthIDExtractor()
    corpus_cb = PROJECT_ROOT / "reverse-SynthID" / "artifacts" / "codebook" / "corpus_codebook.pkl"
    prebuilt_cb = PROJECT_ROOT / "reverse-SynthID" / "artifacts" / "codebook" / "robust_codebook.pkl"
    if corpus_cb.exists():
        extractor.load_codebook(str(corpus_cb))
    elif prebuilt_cb.exists():
        extractor.load_codebook(str(prebuilt_cb))
    else:
        raise FileNotFoundError("No codebook found.")
    return extractor


def main():
    parser = argparse.ArgumentParser(description="Wedge attack: noise -> ring -> diffusion denoise")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--combos", type=str, default=None,
                        help="Comma-separated sigma:strength pairs, e.g. '25:0.20,40:0.30'")
    parser.add_argument("--num-rings", type=int, default=15)
    parser.add_argument("--max-angle", type=float, default=1.5)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codebook", type=str, default=None)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.combos:
        combos = []
        for pair in args.combos.split(","):
            sigma, strength = pair.strip().split(":")
            combos.append((float(sigma), float(strength)))
    else:
        combos = DEFAULT_COMBOS

    images = load_corpus(args.max_images)
    n_total = len(images) * len(combos)

    print(f"SynthID Wedge Attack Sweep")
    print(f"  {len(images)} images x {len(combos)} combos = {n_total} runs")
    print(f"  Combos (sigma, strength): {combos}")
    print(f"  Ring: {args.num_rings} rings, +/-{args.max_angle} deg")
    print(f"  Tile: {args.tile_size}px, overlap: {args.overlap}px")
    print(f"  Steps: {args.num_steps}, guidance: {GUIDANCE}")

    print(f"\nLoading SD 1.5 pipeline (cache: {MODEL_DIR})...")
    pipe = load_pipeline(cache_dir=MODEL_DIR)

    print("Loading reverse-SynthID detector...")
    detector = load_detector()

    results_dir = OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "synthid_wedge_sweep.csv"

    fieldnames = [
        "image", "noise_sigma", "denoise_strength", "num_rings", "max_angle",
        "tile_size", "overlap",
        "ssim", "psnr",
        "synthid_confidence", "synthid_detected",
        "synthid_correlation", "synthid_phase_match", "synthid_structure_ratio",
        "time_s",
    ]

    done = set()
    if args.resume and csv_path.exists():
        with open(csv_path, newline="") as rf:
            for row in csv.DictReader(rf):
                done.add((row["image"], row["noise_sigma"], row["denoise_strength"]))
        print(f"  Resuming: {len(done)} already done")

    mode = "a" if (args.resume and done) else "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        count = 0
        t_start = time.time()

        for img_path in images:
            img = Image.open(img_path).convert("RGB")

            for sigma, strength in combos:
                if (img_path.name, str(sigma), str(strength)) in done:
                    continue

                count += 1
                elapsed = time.time() - t_start
                rate = count / elapsed if elapsed > 1 else 0
                eta = (n_total - count) / rate if rate > 0 else 0

                print(f"  [{count}/{n_total}] {img_path.name} "
                      f"sigma={sigma:.0f} denoise={strength:.2f} "
                      f"(ETA: {eta:.0f}s)",
                      end="", flush=True)

                t0 = time.time()
                attacked, quality = wedge_attack(
                    pipe, img,
                    noise_sigma=sigma,
                    num_rings=args.num_rings,
                    max_angle=args.max_angle,
                    denoise_strength=strength,
                    tile_size=args.tile_size,
                    overlap=args.overlap,
                    guidance_scale=GUIDANCE,
                    num_inference_steps=args.num_steps,
                    seed=args.seed,
                )
                attack_time = time.time() - t0

                atk_dir = OUTPUT_DIR / "attacked" / f"wedge_s{sigma:.0f}_d{strength:.2f}"
                atk_dir.mkdir(parents=True, exist_ok=True)
                attacked.save(atk_dir / img_path.name)

                det = detector.detect_array(np.array(attacked))
                total_time = time.time() - t0

                row = {
                    "image": img_path.name,
                    "noise_sigma": sigma,
                    "denoise_strength": strength,
                    "num_rings": args.num_rings,
                    "max_angle": args.max_angle,
                    "tile_size": args.tile_size,
                    "overlap": args.overlap,
                    "ssim": f"{quality['ssim']:.4f}",
                    "psnr": f"{quality['psnr']:.2f}",
                    "synthid_confidence": f"{det.confidence:.4f}",
                    "synthid_detected": det.is_watermarked,
                    "synthid_correlation": f"{det.correlation:.6f}",
                    "synthid_phase_match": f"{det.phase_match:.4f}",
                    "synthid_structure_ratio": f"{det.structure_ratio:.4f}",
                    "time_s": f"{total_time:.1f}",
                }
                writer.writerow(row)
                f.flush()

                status = "DETECTED" if det.is_watermarked else "REMOVED"
                print(f" -> {status} (corr={det.correlation:.4f} "
                      f"ssim={quality['ssim']:.3f} {total_time:.1f}s)")

                del attacked, det
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            del img
            gc.collect()

    print(f"\nDone: {count} wedge attacks in {time.time() - t_start:.0f}s")
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
