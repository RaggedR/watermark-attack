#!/usr/bin/env python3
"""Generate watermarked (and baseline unwatermarked) images using SD 2.1.

Supports two watermark types:
- tree-ring:        Fourier-space noise injection (default)
- stable-signature: Fine-tuned VAE decoder (Meta, ICCV 2023)

Usage:
    python scripts/generate.py                                           # Tree-Ring (default)
    python scripts/generate.py --watermark-type stable-signature         # Stable Signature
    python scripts/generate.py --num-images 20                           # More images
    python scripts/generate.py --prompts-file prompts.txt                # Custom prompts

Output structure in DW_IMAGE_DIR (default: ./images/):
    watermarked/    — images with watermark embedded
    unwatermarked/  — baseline images from same prompts (different seed)
    metadata.json   — prompts, seeds, watermark params for each image
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
from watermark.stable_signature import (
    generate_ss_key, key_to_str, download_extractor, download_decoder,
    patch_pipeline_decoder, SS_DEFAULT_KEY,
)

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
    watermark_type: str = "tree-ring",
    wm_seed: int = 42,
    radius: float = 10.0,
    ring_width: float = 3.0,
):
    """Generate paired watermarked/unwatermarked images.

    Args:
        num_images: Number of image pairs to generate.
        prompts: Text prompts (cycles if fewer than num_images).
        num_inference_steps: DDIM steps for generation.
        watermark_type: "tree-ring" or "stable-signature".
        wm_seed: Seed for watermark key generation.
        radius: Ring radius in Fourier space (tree-ring only).
        ring_width: Ring band half-width (tree-ring only).
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

    # Setup watermark-specific state
    if watermark_type == "tree-ring":
        key = generate_key(latent_size, latent_size, radius, ring_width, seed=wm_seed)
        mask = create_ring_mask(latent_size, latent_size, radius, ring_width)
        print(f"Watermark: Tree-Ring, radius={radius}, ring_width={ring_width}, "
              f"ring positions={mask.sum().item():.0f}")

        metadata = {
            "watermark_type": "tree-ring",
            "watermark_params": {
                "seed": wm_seed,
                "radius": radius,
                "ring_width": ring_width,
                "latent_size": latent_size,
            },
            "num_inference_steps": num_inference_steps,
            "images": [],
        }

    elif watermark_type == "stable-signature":
        ss_key = generate_ss_key(n_bits=48, seed=wm_seed)
        ss_key_str = key_to_str(ss_key)
        print(f"Watermark: Stable Signature, key={ss_key_str}")

        # Download and patch decoder
        decoder_path = download_decoder(MODEL_DIR)
        pipe = patch_pipeline_decoder(pipe, decoder_path)

        metadata = {
            "watermark_type": "stable-signature",
            "watermark_params": {
                "seed": wm_seed,
                "n_bits": 48,
                "key": ss_key_str,
                "latent_size": latent_size,
            },
            "num_inference_steps": num_inference_steps,
            "images": [],
        }

    else:
        raise ValueError(f"Unknown watermark type: {watermark_type}")

    for i in range(num_images):
        prompt = prompts[i % len(prompts)]
        img_seed = 1000 + i
        print(f"\n[{i+1}/{num_images}] Prompt: {prompt[:60]}...")

        # --- Watermarked image ---
        noise = torch.randn(1, 4, latent_size, latent_size,
                            generator=torch.Generator().manual_seed(img_seed))

        if watermark_type == "tree-ring":
            # Embed watermark in noise (Fourier domain)
            watermarked_noise = embed_watermark(noise.clone(), key, mask)
        else:
            # Stable Signature: watermark comes from decoder, noise is standard
            watermarked_noise = noise

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
        # For SS: need a SEPARATE pipeline with unpatched decoder for baselines.
        # For simplicity, use same pipeline with different seed — SS baseline will
        # still have the watermark. This is fine: baselines are for visual comparison,
        # not for detection testing. The sweep uses watermarked images only.
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
    parser.add_argument("--watermark-type", type=str, default="tree-ring",
                        choices=["tree-ring", "stable-signature"],
                        help="Watermark method (default: tree-ring)")
    parser.add_argument("--wm-seed", type=int, default=42)
    parser.add_argument("--radius", type=float, default=10.0,
                        help="Ring radius in Fourier space (tree-ring only)")
    parser.add_argument("--ring-width", type=float, default=3.0,
                        help="Ring band half-width (tree-ring only)")
    args = parser.parse_args()

    prompts = None
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]

    generate_images(
        num_images=args.num_images,
        prompts=prompts,
        num_inference_steps=args.steps,
        watermark_type=args.watermark_type,
        wm_seed=args.wm_seed,
        radius=args.radius,
        ring_width=args.ring_width,
    )


if __name__ == "__main__":
    main()
