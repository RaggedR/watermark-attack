#!/usr/bin/env python3
"""Embed watermarks into existing images using the adapter interface.

Replaces the generate.py workflow for the new approach: watermark existing
images from DIV2K / DiffusionDB datasets rather than generating from prompts.

Usage:
    # VideoSeal (fast — no SD pipeline needed)
    python scripts/embed.py --method videoseal \
        --input-dir "/Volumes/One Touch/diffusion-watermarking/datasets/diffusiondb" \
        --num-images 50 --strengths 0.1 0.2 0.3

    # Stable Signature (needs SD 2.1 for VAE encode/decode)
    python scripts/embed.py --method stable-signature \
        --input-dir "/Volumes/One Touch/diffusion-watermarking/datasets/div2k" \
        --num-images 50

    # Tree-Ring (needs SD 2.1 for DDIM inversion — slow)
    python scripts/embed.py --method tree-ring \
        --input-dir "/Volumes/One Touch/diffusion-watermarking/datasets/diffusiondb" \
        --num-images 10

Output structure:
    images/{method}/{dataset}/s{strength}/0000.png, 0001.png, ...
    images/{method}/{dataset}/metadata.json
"""

import os
import sys
import json
import argparse
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from watermark.adapters import create_adapter

IMAGE_DIR = os.environ.get("DW_IMAGE_DIR", str(Path(__file__).resolve().parent.parent / "images"))
MODEL_DIR = os.environ.get("DW_MODEL_DIR", str(Path(__file__).resolve().parent.parent / ".models"))


def find_images(input_dir: str, num_images: int | None = None) -> list[Path]:
    """Find PNG/JPG images in a directory, sorted by name.

    Args:
        input_dir: Directory to search.
        num_images: Max images to return (None = all).

    Returns:
        List of image paths.
    """
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in exts and p.is_file() and not p.name.startswith(".")
    )
    if num_images is not None:
        images = images[:num_images]
    return images


def embed_images(
    method: str,
    input_dir: str,
    num_images: int | None = None,
    strengths: list[float] | None = None,
    seed: int = 42,
):
    """Embed watermarks into existing images.

    Args:
        method: Watermark method ("tree-ring", "stable-signature", "videoseal").
        input_dir: Directory of source images.
        num_images: Max images to process (None = all).
        strengths: List of embedding strengths to sweep.
        seed: Random seed for watermark key/message generation.
    """
    if strengths is None:
        strengths = [0.3]

    # Find source images
    source_images = find_images(input_dir, num_images)
    if not source_images:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    # Auto-detect dataset name from directory
    dataset = Path(input_dir).name

    print(f"Method: {method}")
    print(f"Dataset: {dataset} ({len(source_images)} images)")
    print(f"Strengths: {strengths}")

    # Load SD pipeline only if needed
    pipe = None
    adapter_temp = create_adapter(method, seed=seed) if method == "videoseal" else None
    if adapter_temp is None:
        # Tree-Ring or Stable Signature — need the pipeline
        from watermark.inversion import load_pipeline
        print(f"Loading SD 2.1 (cache: {MODEL_DIR})...")
        pipe = load_pipeline(cache_dir=MODEL_DIR)

    # Create adapter (with pipeline if needed)
    adapter = create_adapter(method, pipe=pipe, model_dir=MODEL_DIR, seed=seed)

    # Setup output directory
    base_out = Path(IMAGE_DIR) / method / dataset

    # Metadata to save
    meta = {
        "watermark": adapter.metadata(),
        "dataset": dataset,
        "source_dir": str(input_dir),
        "strengths": strengths,
        "images": [],
    }

    start_time = time.time()
    total = len(source_images) * len(strengths)
    count = 0

    for strength in strengths:
        strength_dir = base_out / f"s{strength}"
        strength_dir.mkdir(parents=True, exist_ok=True)

        for i, src_path in enumerate(source_images):
            count += 1
            from PIL import Image
            img = Image.open(src_path).convert("RGB")

            elapsed = time.time() - start_time
            rate = count / elapsed if elapsed > 0 else 0
            eta = (total - count) / rate if rate > 0 else 0

            print(f"  [{count}/{total}] {src_path.name} (s={strength}) "
                  f"ETA: {eta:.0f}s", end="", flush=True)

            try:
                wm_img = adapter.embed(img, strength=strength)
                out_path = strength_dir / f"{i:04d}.png"
                wm_img.save(out_path)
                print(f" → {out_path.name}")

                meta["images"].append({
                    "index": i,
                    "source": str(src_path),
                    "output": str(out_path),
                    "strength": strength,
                })
            except Exception as e:
                print(f" → ERROR: {e}")
                meta["images"].append({
                    "index": i,
                    "source": str(src_path),
                    "output": None,
                    "strength": strength,
                    "error": str(e),
                })

    # Save metadata
    meta_path = base_out / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nDone: {count} images in {total_time:.1f}s")
    print(f"Output: {base_out}")
    print(f"Metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Embed watermarks into existing images"
    )
    parser.add_argument(
        "--method", type=str, required=True,
        choices=["tree-ring", "stable-signature", "videoseal"],
        help="Watermark method",
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory of source images",
    )
    parser.add_argument(
        "--num-images", type=int, default=None,
        help="Max images to process (default: all)",
    )
    parser.add_argument(
        "--strengths", type=float, nargs="+", default=None,
        help="Embedding strengths to sweep (default: [0.3])",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    embed_images(
        method=args.method,
        input_dir=args.input_dir,
        num_images=args.num_images,
        strengths=args.strengths,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
