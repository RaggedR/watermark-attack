#!/usr/bin/env python3
"""Generate SynthID-watermarked images via the Gemini API.

Step 1: Ask Gemini to generate diverse image prompts.
Step 2: Use those prompts to generate images (which carry SynthID automatically).

Usage:
    # Generate 50 images (default)
    python scripts/generate_synthid.py

    # Generate 10 images to a custom directory
    python scripts/generate_synthid.py --num-images 10 --output-dir test-synthID/corpus

    # Use existing prompts file (skip generation)
    python scripts/generate_synthid.py --prompts-file prompts.json
"""

import argparse
import json
import time
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types
from PIL import Image


def generate_prompts(client, num_prompts: int) -> list[str]:
    """Ask Gemini to generate diverse image prompts."""
    meta_prompt = f"""Generate exactly {num_prompts} diverse image generation prompts.
Requirements:
- Each prompt should describe a different scene, subject, or concept
- Mix of: people, landscapes, animals, objects, abstract art, architecture, food, vehicles, fantasy, sci-fi
- Mix of styles: photorealistic, painting, illustration, 3D render, watercolor, sketch
- Each prompt should be 1-2 sentences, specific and vivid
- No duplicates or near-duplicates

Return ONLY a JSON array of strings, no other text. Example:
["A photorealistic close-up of a barn owl perched on a mossy branch at dusk", "An oil painting of a bustling Tokyo street market in the rain"]"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=meta_prompt,
    )

    # Parse JSON from response
    text = response.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    prompts = json.loads(text)

    if len(prompts) != num_prompts:
        print(f"  Warning: requested {num_prompts} prompts, got {len(prompts)}")

    return prompts


def generate_image(client, prompt: str, aspect_ratio: str = "1:1") -> Image.Image | None:
    """Generate a single image via Gemini. Returns PIL Image or None on failure."""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                ),
            ),
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data:
                return Image.open(BytesIO(part.inline_data.data))

        print(f"  No image data in response")
        return None

    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate SynthID-watermarked images via Gemini API"
    )
    parser.add_argument(
        "--num-images", type=int, default=50,
        help="Number of images to generate (default: 50)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("test-synthID/corpus"),
        help="Output directory for images (default: test-synthID/corpus)",
    )
    parser.add_argument(
        "--prompts-file", type=Path, default=None,
        help="JSON file with pre-generated prompts (skip prompt generation)",
    )
    parser.add_argument(
        "--aspect-ratio", type=str, default="1:1",
        help="Aspect ratio for generated images (default: 1:1)",
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds between API calls to respect rate limits (default: 2.0)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client()

    # Step 1: Get prompts
    if args.prompts_file and args.prompts_file.exists():
        print(f"Loading prompts from {args.prompts_file}")
        prompts = json.loads(args.prompts_file.read_text())
    else:
        print(f"Generating {args.num_images} prompts via Gemini...")
        prompts = generate_prompts(client, args.num_images)
        # Save prompts for reproducibility
        prompts_path = args.output_dir / "prompts.json"
        prompts_path.write_text(json.dumps(prompts, indent=2))
        print(f"  Saved prompts to {prompts_path}")

    prompts = prompts[:args.num_images]

    # Step 2: Generate images
    print(f"\nGenerating {len(prompts)} images...")
    metadata = []
    succeeded = 0
    failed = 0

    for i, prompt in enumerate(prompts):
        filename = f"{i:04d}.png"
        out_path = args.output_dir / filename

        # Skip if already exists
        if out_path.exists():
            print(f"  [{i+1}/{len(prompts)}] {filename} already exists, skipping")
            succeeded += 1
            metadata.append({"index": i, "filename": filename, "prompt": prompt, "status": "skipped"})
            continue

        print(f"  [{i+1}/{len(prompts)}] {prompt[:60]}...")
        img = generate_image(client, prompt, aspect_ratio=args.aspect_ratio)

        if img is not None:
            img.save(out_path, format="PNG")
            succeeded += 1
            metadata.append({
                "index": i,
                "filename": filename,
                "prompt": prompt,
                "size": list(img.size),
                "status": "ok",
            })
            print(f"           → {filename} ({img.size[0]}x{img.size[1]})")
        else:
            failed += 1
            metadata.append({"index": i, "filename": filename, "prompt": prompt, "status": "failed"})

        # Rate limit courtesy
        if i < len(prompts) - 1:
            time.sleep(args.delay)

    # Save metadata
    meta_path = args.output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nDone: {succeeded} succeeded, {failed} failed")
    print(f"Images: {args.output_dir}/")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
