#!/usr/bin/env python3
"""Fast noise-ring-denoise attack sweep against SynthID.

No diffusion model needed — runs in seconds per image.

Usage:
    python scripts/attack_synthid_fast.py --max-images 5
    python scripts/attack_synthid_fast.py   # full 50 images
"""

import gc
import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reverse-SynthID"))

import numpy as np
from PIL import Image
from attack.noise_ring import noise_ring_denoise_attack
from src.extraction.robust_extractor import RobustSynthIDExtractor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "test-synthID" / "corpus"
OUTPUT_DIR = PROJECT_ROOT / "test-synthID"

DEFAULT_SIGMAS = [10, 15, 20, 25, 30, 40, 50]
DEFAULT_ANGLES = [1.5]


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
    parser = argparse.ArgumentParser(description="Fast noise-ring-denoise attack sweep")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    parser.add_argument("--angles", type=float, nargs="+", default=DEFAULT_ANGLES)
    parser.add_argument("--num-rings", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    images = load_corpus(args.max_images)
    combos = [(s, a) for s in args.sigmas for a in args.angles]
    n_total = len(images) * len(combos)

    print(f"Fast Noise-Ring-Denoise Attack Sweep")
    print(f"  {len(images)} images x {len(combos)} combos = {n_total} runs")
    print(f"  Sigmas: {args.sigmas}")
    print(f"  Angles: {args.angles}")
    print(f"  Rings: {args.num_rings}")

    print("Loading reverse-SynthID detector...")
    detector = load_detector()

    results_dir = OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "synthid_fast_sweep.csv"

    fieldnames = [
        "image", "noise_sigma", "max_angle", "num_rings",
        "ssim", "psnr",
        "synthid_confidence", "synthid_detected",
        "synthid_correlation", "synthid_phase_match", "synthid_structure_ratio",
        "time_s",
    ]

    done = set()
    if args.resume and csv_path.exists():
        with open(csv_path, newline="") as rf:
            for row in csv.DictReader(rf):
                done.add((row["image"], row["noise_sigma"], row["max_angle"]))
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

            for sigma, angle in combos:
                if (img_path.name, str(sigma), str(angle)) in done:
                    continue

                count += 1
                elapsed = time.time() - t_start
                rate = count / elapsed if elapsed > 1 else 0
                eta = (n_total - count) / rate if rate > 0 else 0

                print(f"  [{count}/{n_total}] {img_path.name} sigma={sigma} angle={angle}",
                      end="", flush=True)

                t0 = time.time()
                attacked, quality = noise_ring_denoise_attack(
                    img, noise_sigma=sigma, num_rings=args.num_rings,
                    max_angle=angle, seed=args.seed,
                )
                attack_time = time.time() - t0

                # Save
                atk_dir = OUTPUT_DIR / "attacked" / f"fast_s{sigma:.0f}_a{angle:.1f}"
                atk_dir.mkdir(parents=True, exist_ok=True)
                attacked.save(atk_dir / img_path.name)

                # Detect
                det = detector.detect_array(np.array(attacked))
                total_time = time.time() - t0

                row = {
                    "image": img_path.name,
                    "noise_sigma": sigma,
                    "max_angle": angle,
                    "num_rings": args.num_rings,
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
                print(f" -> {status} (corr={det.correlation:.4f} ssim={quality['ssim']:.3f} {total_time:.1f}s)")

                del attacked, det
                gc.collect()

            del img

    print(f"\nDone: {count} attacks in {time.time() - t_start:.0f}s")
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
