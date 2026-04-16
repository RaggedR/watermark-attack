#!/usr/bin/env python3
"""Sandwich attack sweep: diffuse → ring-rotate → diffuse against SynthID.

Tests whether interleaving a geometric ring rotation between two lighter
diffusion passes is more effective at removing SynthID than a single
heavier diffusion pass.

Usage:
    python scripts/attack_synthid_sandwich.py --max-images 5   # quick test
    python scripts/attack_synthid_sandwich.py                  # full sweep
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
from attack.regeneration import sandwich_attack
from src.extraction.robust_extractor import RobustSynthIDExtractor

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "test-synthID" / "corpus"
OUTPUT_DIR = PROJECT_ROOT / "test-synthID"
_env_model_dir = os.environ.get("DW_MODEL_DIR", "")
MODEL_DIR = _env_model_dir if _env_model_dir and Path(_env_model_dir).exists() else str(PROJECT_ROOT / ".models")

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


def load_corpus(max_images: int | None = None) -> list[Path]:
    meta_path = CORPUS_DIR / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)
    images = []
    for entry in metadata:
        if entry.get("status") != "ok":
            continue
        img_path = CORPUS_DIR / entry["filename"]
        if img_path.exists():
            images.append(img_path)
    if max_images is not None:
        images = images[:max_images]
    return images


def load_detector(codebook_path: str | None = None) -> RobustSynthIDExtractor:
    extractor = RobustSynthIDExtractor()
    if codebook_path and Path(codebook_path).exists():
        extractor.load_codebook(codebook_path)
    else:
        corpus_cb = PROJECT_ROOT / "reverse-SynthID" / "artifacts" / "codebook" / "corpus_codebook.pkl"
        prebuilt_cb = PROJECT_ROOT / "reverse-SynthID" / "artifacts" / "codebook" / "robust_codebook.pkl"
        if corpus_cb.exists():
            print(f"  Using corpus codebook: {corpus_cb}")
            extractor.load_codebook(str(corpus_cb))
        elif prebuilt_cb.exists():
            print(f"  Using pre-built codebook: {prebuilt_cb}")
            extractor.load_codebook(str(prebuilt_cb))
        else:
            raise FileNotFoundError("No codebook found.")
    return extractor


def run_sweep(
    combos: list[tuple[float, float]],
    num_rings: int,
    max_angle: float,
    num_steps: int,
    max_images: int | None,
    seed: int,
    codebook_path: str | None,
    tile_size: int,
    overlap: int,
    resume: bool = False,
) -> Path:
    images = load_corpus(max_images)
    n_total = len(images) * len(combos)

    print(f"SynthID Sandwich Attack Sweep")
    print(f"  {len(images)} images x {len(combos)} combos = {n_total} runs")
    print(f"  Combos (s1, s2): {combos}")
    print(f"  Ring: {num_rings} rings, +/-{max_angle} deg")
    print(f"  Tile: {tile_size}px, overlap: {overlap}px")
    print(f"  Steps: {num_steps}, guidance: {GUIDANCE}")

    print(f"\nLoading SD 1.5 pipeline (cache: {MODEL_DIR})...")
    pipe = load_pipeline(cache_dir=MODEL_DIR)

    print("Loading reverse-SynthID detector...")
    detector = load_detector(codebook_path)

    results_dir = OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "synthid_sandwich_sweep.csv"

    fieldnames = [
        "image", "strength_1", "strength_2", "num_rings", "max_angle",
        "tile_size", "overlap",
        "ssim", "psnr",
        "synthid_confidence", "synthid_detected",
        "synthid_correlation", "synthid_phase_match", "synthid_structure_ratio",
        "time_s",
    ]

    # Resume support
    done_pairs: set[tuple[str, str, str]] = set()
    if resume and csv_path.exists():
        with open(csv_path, newline="") as rf:
            for row in csv.DictReader(rf):
                done_pairs.add((row["image"], row["strength_1"], row["strength_2"]))
        print(f"  Resuming: {len(done_pairs)} combos already done")

    mode = "a" if (resume and done_pairs) else "w"
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
                if key in done_pairs:
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

                # Save attacked image
                atk_dir = OUTPUT_DIR / "attacked" / f"sandwich_s{s1:.2f}_s{s2:.2f}"
                atk_dir.mkdir(parents=True, exist_ok=True)
                attacked_img.save(atk_dir / img_path.name)

                # Detect watermark
                atk_array = np.array(attacked_img)
                det_result = detector.detect_array(atk_array)
                detect_time = time.time() - t0 - attack_time
                total_time = attack_time + detect_time

                row = {
                    "image": img_path.name,
                    "strength_1": s1,
                    "strength_2": s2,
                    "num_rings": num_rings,
                    "max_angle": max_angle,
                    "tile_size": tile_size,
                    "overlap": overlap,
                    "ssim": f"{quality['ssim']:.4f}",
                    "psnr": f"{quality['psnr']:.2f}",
                    "synthid_confidence": f"{det_result.confidence:.4f}",
                    "synthid_detected": det_result.is_watermarked,
                    "synthid_correlation": f"{det_result.correlation:.6f}",
                    "synthid_phase_match": f"{det_result.phase_match:.4f}",
                    "synthid_structure_ratio": f"{det_result.structure_ratio:.4f}",
                    "time_s": f"{total_time:.1f}",
                }
                writer.writerow(row)
                f.flush()

                status = "DETECTED" if det_result.is_watermarked else "REMOVED"
                print(
                    f" -> {status} "
                    f"(conf={det_result.confidence:.3f} "
                    f"corr={det_result.correlation:.4f} "
                    f"ssim={quality['ssim']:.3f} "
                    f"{total_time:.1f}s)"
                )

                del attacked_img, atk_array, det_result, quality
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
        description="Sandwich attack sweep: diffuse -> ring-rotate -> diffuse"
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--combos", type=str, default=None,
        help="Comma-separated s1:s2 pairs, e.g. '0.10:0.10,0.15:0.15'",
    )
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
            s1, s2 = pair.strip().split(":")
            combos.append((float(s1), float(s2)))
    else:
        combos = DEFAULT_COMBOS

    run_sweep(
        combos=combos,
        num_rings=args.num_rings,
        max_angle=args.max_angle,
        num_steps=args.num_steps,
        max_images=args.max_images,
        seed=args.seed,
        codebook_path=args.codebook,
        tile_size=args.tile_size,
        overlap=args.overlap,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
