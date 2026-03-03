#!/usr/bin/env python3
"""Generate watermarked (and baseline unwatermarked) images using Tree-Ring + SD 2.1.

Usage:
    python scripts/generate.py                    # Generate 10 image pairs
    python scripts/generate.py --num-images 20    # Generate 20 pairs
    python scripts/generate.py --prompts-file prompts.txt  # Custom prompts

Output structure in DW_IMAGE_DIR (default: ./images/):
    watermarked/    — images with Tree-Ring watermark embedded
    unwatermarked/  — baseline images from same prompts (different seed)
    metadata.json   — prompts, seeds, key params for each image
"""

import os
import sys
import json
import argparse

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from watermark.tree_ring import generate_key, create_ring_mask, embed_watermark
from watermark.inversion import load_pipeline

IMAGE_DIR = os.environ.get("DW_IMAGE_DIR", str(Path(__file__).resolve().parent.parent / "images"))
MODEL_DIR = os.environ.get("DW_MODEL_DIR", str(Path(__file__).resolve().parent.parent / ".models"))

DEFAULT_PROMPTS = [
    "a photograph of a golden retriever in a park",
    "an oil painting of a mountain landscape at sunset",
    "a digital illustration of a futuristic city skyline",
    "a watercolor painting of flowers in a vase",
    "a photograph of a cat sleeping on a windowsill",
    "an impressionist painting of a bridge over a river",
    "a macro photograph of a butterfly on a leaf",
    "a pencil sketch of an old library interior",
    "a photograph of a coffee cup on a wooden table",
    "a digital art piece of an astronaut floating in space",
]


def generate_images(
    num_images: int = 10,
    prompts: list[str] | None = None,
    num_inference_steps: int = 50,
    wm_seed: int = 42,
    radius: float = 10.0,
    ring_width: float = 3.0,
):
    """Generate paired watermarked/unwatermarked images.

    Args:
        num_images: Number of image pairs to generate.
        prompts: Text prompts (cycles if fewer than num_images).
        num_inference_steps: DDIM steps for generation.
        wm_seed: Seed for watermark key generation.
        radius: Ring radius in Fourier space.
        ring_width: Ring band half-width.
    """
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    # Setup directories
    wm_dir = Path(IMAGE_DIR) / "watermarked"
    nowm_dir = Path(IMAGE_DIR) / "unwatermarked"
    wm_dir.mkdir(parents=True, exist_ok=True)
    nowm_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SD 2.1 (cache: {MODEL_DIR})...")
    pipe = load_pipeline(cache_dir=MODEL_DIR)

    # Determine latent size from pipeline config
    latent_size = pipe.unet.config.sample_size  # 96 for SD 2.1
    print(f"Latent size: {latent_size}x{latent_size}")

    # Generate watermark key and mask
    key = generate_key(latent_size, latent_size, radius, ring_width, seed=wm_seed)
    mask = create_ring_mask(latent_size, latent_size, radius, ring_width)
    print(f"Watermark: radius={radius}, ring_width={ring_width}, "
          f"ring positions={mask.sum().item():.0f}")

    metadata = {
        "watermark_params": {
            "seed": wm_seed,
            "radius": radius,
            "ring_width": ring_width,
            "latent_size": latent_size,
        },
        "num_inference_steps": num_inference_steps,
        "images": [],
    }

    for i in range(num_images):
        prompt = prompts[i % len(prompts)]
        img_seed = 1000 + i
        print(f"\n[{i+1}/{num_images}] Prompt: {prompt[:60]}...")

        # --- Watermarked image ---
        noise = torch.randn(1, 4, latent_size, latent_size,
                            generator=torch.Generator().manual_seed(img_seed))
        watermarked_noise = embed_watermark(noise.clone(), key, mask)

        with torch.no_grad():
            result = pipe(
                prompt,
                latents=watermarked_noise.to(pipe.device, pipe.unet.dtype),
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
            )
        wm_image = result.images[0]
        wm_path = wm_dir / f"{i:04d}.png"
        wm_image.save(wm_path)
        print(f"  Watermarked: {wm_path}")

        # --- Unwatermarked baseline (different seed, same prompt) ---
        generator = torch.Generator().manual_seed(img_seed + 5000)
        with torch.no_grad():
            result = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                generator=generator,
            )
        nowm_image = result.images[0]
        nowm_path = nowm_dir / f"{i:04d}.png"
        nowm_image.save(nowm_path)
        print(f"  Unwatermarked: {nowm_path}")

        metadata["images"].append({
            "index": i,
            "prompt": prompt,
            "seed": img_seed,
            "watermarked_path": str(wm_path),
            "unwatermarked_path": str(nowm_path),
        })

    # Save metadata
    meta_path = Path(IMAGE_DIR) / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")
    print(f"Generated {num_images} watermarked + {num_images} baseline images.")


def main():
    parser = argparse.ArgumentParser(description="Generate watermarked test images")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="File with one prompt per line")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--wm-seed", type=int, default=42)
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--ring-width", type=float, default=3.0)
    args = parser.parse_args()

    prompts = None
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]

    generate_images(
        num_images=args.num_images,
        prompts=prompts,
        num_inference_steps=args.steps,
        wm_seed=args.wm_seed,
        radius=args.radius,
        ring_width=args.ring_width,
    )


if __name__ == "__main__":
    main()
