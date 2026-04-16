#!/usr/bin/env python3
"""Attack sweep: grid-search img2img regeneration params against SynthID.

Loads SynthID-watermarked corpus images (generated via Gemini API),
attacks them with tiled img2img regeneration at multiple strengths,
and measures watermark survival using reverse-SynthID detection.

Usage:
    python scripts/attack_synthid_sweep.py --max-images 5   # quick test
    python scripts/attack_synthid_sweep.py                  # full 50-image sweep
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

# Load .env before any other imports that might read env vars
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reverse-SynthID"))

import numpy as np
from PIL import Image
from watermark.inversion import load_pipeline
from attack.regeneration import tiled_regeneration_attack
from src.extraction.robust_extractor import RobustSynthIDExtractor

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "test-synthID" / "corpus"
OUTPUT_DIR = PROJECT_ROOT / "test-synthID"
_env_model_dir = os.environ.get("DW_MODEL_DIR", "")
MODEL_DIR = _env_model_dir if _env_model_dir and Path(_env_model_dir).exists() else str(PROJECT_ROOT / ".models")

# Default sweep grid — fine-grained strengths, guidance fixed at 1.0
DEFAULT_STRENGTHS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
DEFAULT_STEPS = 50
GUIDANCE = 1.0  # empty-prompt unconditional — guidance is irrelevant


def load_corpus(max_images: int | None = None) -> list[Path]:
    """Load corpus image paths from metadata.json."""
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
    """Load reverse-SynthID detector with codebook.

    If a corpus-specific codebook exists, use it (more accurate).
    Otherwise fall back to the pre-built one shipped with reverse-SynthID.
    """
    extractor = RobustSynthIDExtractor()

    if codebook_path and Path(codebook_path).exists():
        extractor.load_codebook(codebook_path)
    else:
        # Try corpus-specific codebook first
        corpus_cb = PROJECT_ROOT / "reverse-SynthID" / "artifacts" / "codebook" / "corpus_codebook.pkl"
        prebuilt_cb = PROJECT_ROOT / "reverse-SynthID" / "artifacts" / "codebook" / "robust_codebook.pkl"

        if corpus_cb.exists():
            print(f"  Using corpus codebook: {corpus_cb}")
            extractor.load_codebook(str(corpus_cb))
        elif prebuilt_cb.exists():
            print(f"  Using pre-built codebook: {prebuilt_cb}")
            extractor.load_codebook(str(prebuilt_cb))
        else:
            raise FileNotFoundError(
                "No codebook found. Run codebook extraction first:\n"
                "  python reverse-SynthID/src/extraction/robust_extractor.py "
                "extract test-synthID/corpus/ --output reverse-SynthID/artifacts/codebook/corpus_codebook.pkl"
            )
    return extractor


def detect_original(detector: RobustSynthIDExtractor, img_path: Path) -> dict:
    """Run detection on an original (unattacked) image."""
    result = detector.detect(str(img_path))
    return {
        "confidence": result.confidence,
        "detected": result.is_watermarked,
        "correlation": result.correlation,
        "phase_match": result.phase_match,
        "structure_ratio": result.structure_ratio,
    }


def run_sweep(
    attack_strengths: list[float],
    num_steps: int,
    max_images: int | None,
    seed: int,
    codebook_path: str | None,
    tile_size: int,
    overlap: int,
    resume: bool = False,
) -> Path:
    """Run the SynthID attack sweep and write results CSV."""
    images = load_corpus(max_images)
    n_total = len(images) * len(attack_strengths)

    print(f"SynthID Attack Sweep")
    print(f"  {len(images)} images × {len(attack_strengths)} strengths = {n_total} runs")
    print(f"  Strengths: {attack_strengths}")
    print(f"  Tile: {tile_size}px, overlap: {overlap}px")
    print(f"  Steps: {num_steps}, guidance: {GUIDANCE}")

    # Load SD 1.5 pipeline (float32 — float16 causes dtype mismatch in VAE)
    print(f"\nLoading SD 1.5 pipeline (cache: {MODEL_DIR})...")
    pipe = load_pipeline(cache_dir=MODEL_DIR)

    # Load detector
    print("Loading reverse-SynthID detector...")
    detector = load_detector(codebook_path)

    # Prepare output dirs
    results_dir = OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "synthid_sweep.csv"

    fieldnames = [
        "image", "strength", "tile_size", "overlap",
        "ssim", "psnr",
        "synthid_confidence", "synthid_detected",
        "synthid_correlation", "synthid_phase_match", "synthid_structure_ratio",
        "time_s",
    ]

    # Resume support: track (image, strength) pairs already completed
    done_pairs: set[tuple[str, str]] = set()
    done_originals: set[str] = set()
    if resume and csv_path.exists():
        with open(csv_path, newline="") as rf:
            for row in csv.DictReader(rf):
                done_pairs.add((row["image"], row["strength"]))
                if float(row["strength"]) == 0.0:
                    done_originals.add(row["image"])
        print(f"  Resuming: {len(done_pairs)} (image, strength) pairs already done")

    mode = "a" if (resume and done_pairs) else "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        count = 0
        sweep_start = time.time()

        for img_path in images:
            # Skip original baseline if already recorded
            has_original = img_path.name in done_originals
            # Skip entire image only if ALL strengths are done
            all_done = all(
                (img_path.name, f"{s}") in done_pairs or (img_path.name, str(s)) in done_pairs
                for s in attack_strengths
            ) and has_original
            if all_done:
                continue
            if has_original:
                # Reuse cached original detection — skip to attacks
                print(f"\n  {img_path.name}: original already recorded, checking strengths...")
            else:
                # First: record the original (unattacked) detection baseline
                print(f"\n  {img_path.name}: detecting original...", end="", flush=True)
            if not has_original:
                t0 = time.time()
                orig_det = detect_original(detector, img_path)
                det_time = time.time() - t0
                status = "DETECTED" if orig_det["detected"] else "MISSED"
                print(
                    f" → {status} "
                    f"(conf={orig_det['confidence']:.3f} "
                    f"phase={orig_det['phase_match']:.3f} "
                    f"{det_time:.1f}s)"
                )

                row = {
                    "image": img_path.name,
                    "strength": 0.0,
                    "tile_size": tile_size,
                    "overlap": overlap,
                    "ssim": 1.0,
                    "psnr": float("inf"),
                    "synthid_confidence": f"{orig_det['confidence']:.4f}",
                    "synthid_detected": orig_det["detected"],
                    "synthid_correlation": f"{orig_det['correlation']:.6f}",
                    "synthid_phase_match": f"{orig_det['phase_match']:.4f}",
                    "synthid_structure_ratio": f"{orig_det['structure_ratio']:.4f}",
                    "time_s": f"{det_time:.1f}",
                }
                writer.writerow(row)
                f.flush()

                del orig_det
                gc.collect()

            # Now attack at each strength
            img = Image.open(img_path).convert("RGB")

            for atk_str in attack_strengths:
                # Skip strengths already done for this image
                if (img_path.name, str(atk_str)) in done_pairs:
                    continue
                count += 1
                elapsed = time.time() - sweep_start
                rate = count / elapsed if elapsed > 1 else 0
                eta = (n_total - count) / rate if rate > 0 else 0

                print(
                    f"  [{count}/{n_total}] {img_path.name} "
                    f"s={atk_str:.2f} "
                    f"(ETA: {eta:.0f}s)",
                    end="", flush=True,
                )

                t0 = time.time()
                attacked_img, quality = tiled_regeneration_attack(
                    pipe, img,
                    strength=atk_str,
                    tile_size=tile_size,
                    overlap=overlap,
                    guidance_scale=GUIDANCE,
                    num_inference_steps=num_steps,
                    seed=seed,
                )
                attack_time = time.time() - t0

                # Save attacked image
                atk_dir = OUTPUT_DIR / "attacked" / f"s{atk_str:.2f}"
                atk_dir.mkdir(parents=True, exist_ok=True)
                attacked_img.save(atk_dir / img_path.name)

                # Detect watermark on attacked image
                atk_array = np.array(attacked_img)
                det_result = detector.detect_array(atk_array)
                detect_time = time.time() - t0 - attack_time

                total_time = attack_time + detect_time

                row = {
                    "image": img_path.name,
                    "strength": atk_str,
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
                    f" → {status} "
                    f"(conf={det_result.confidence:.3f} "
                    f"ssim={quality['ssim']:.3f} "
                    f"{total_time:.1f}s)"
                )

                # Free memory between iterations
                del attacked_img, atk_array, det_result, quality
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            # Free source image after all strengths for this image
            del img
            gc.collect()

    total_time = time.time() - sweep_start
    print(f"\nDone: {count} attacks in {total_time:.0f}s")
    print(f"Results: {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Attack sweep against SynthID-watermarked images"
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Max corpus images to attack (default: all 50)",
    )
    parser.add_argument(
        "--strengths", type=float, nargs="+",
        default=DEFAULT_STRENGTHS,
        help="Attack strength values to sweep",
    )
    parser.add_argument("--num-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--codebook", type=str, default=None,
        help="Path to reverse-SynthID codebook (auto-detected if omitted)",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=0,
                        help="Tile overlap in pixels (0 for 1024×1024 images = clean 2×2 grid)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip images already present in the results CSV")
    args = parser.parse_args()

    run_sweep(
        attack_strengths=args.strengths,
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
