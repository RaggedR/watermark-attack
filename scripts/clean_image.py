#!/usr/bin/env python3
"""Strip metadata and remove visible watermarks from AI-generated images.

Designed for images from Google Gemini / Imagen which include:
  1. C2PA provenance metadata in a non-standard PNG chunk (caBX)
  2. A visible diamond/sparkle badge in the bottom-right corner

Usage:
    # Clean a single image
    python scripts/clean_image.py input.png output.png

    # Clean all images in a directory
    python scripts/clean_image.py input_dir/ output_dir/

    # Adjust inpaint region size (default: 90px from corner)
    python scripts/clean_image.py input.png output.png --badge-size 120

    # Skip visible watermark removal (metadata strip only)
    python scripts/clean_image.py input.png output.png --metadata-only
"""

import argparse
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# PNG chunks that carry image data (everything else is metadata)
ESSENTIAL_CHUNKS = {"IHDR", "PLTE", "IDAT", "IEND"}


def strip_metadata(input_path: Path, output_path: Path) -> dict:
    """Re-write a PNG keeping only essential chunks.

    Strips caBX (C2PA), tEXt, iTXt, zTXt, eXIf, and any other
    ancillary/non-critical chunks.

    Returns:
        Dict with stripped chunk names and their sizes.
    """
    stripped = {}

    with open(input_path, "rb") as f:
        signature = f.read(8)
        chunks = []
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            length = struct.unpack(">I", header[:4])[0]
            chunk_type = header[4:8].decode("ascii", errors="replace")
            data = f.read(length)
            crc = f.read(4)

            if chunk_type in ESSENTIAL_CHUNKS:
                chunks.append((header, data, crc))
            else:
                stripped[chunk_type] = length

    with open(output_path, "wb") as f:
        f.write(signature)
        for header, data, crc in chunks:
            f.write(header)
            f.write(data)
            f.write(crc)

    return stripped


def remove_badge(image: np.ndarray, badge_size: int = 90) -> np.ndarray:
    """Inpaint over the bottom-right corner badge.

    Uses OpenCV's Telea inpainting algorithm to fill the badge region
    with content that matches the surrounding area.

    Args:
        image: BGR image array (OpenCV format).
        badge_size: Size in pixels of the square region to inpaint
            from the bottom-right corner.

    Returns:
        Image with badge region inpainted.
    """
    h, w = image.shape[:2]

    # Create a mask: white = region to inpaint
    mask = np.zeros((h, w), dtype=np.uint8)

    # The Gemini badge sits in the bottom-right corner.
    # Use a circular mask centered slightly inward from the corner
    # for more natural inpainting than a hard rectangle.
    center_x = w - badge_size // 2 - 10
    center_y = h - badge_size // 2 - 10
    radius = badge_size // 2
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    # Inpaint using Telea's method (good for textured regions)
    result = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    return result


def clean_image(
    input_path: Path,
    output_path: Path,
    badge_size: int = 90,
    metadata_only: bool = False,
) -> dict:
    """Full cleaning pipeline: strip metadata + remove visible badge.

    Args:
        input_path: Source image.
        output_path: Cleaned output image.
        badge_size: Size of corner badge to inpaint (pixels).
        metadata_only: If True, skip badge removal.

    Returns:
        Dict summarizing what was done.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"input": str(input_path)}

    if metadata_only:
        # Just strip metadata at the PNG chunk level
        stripped = strip_metadata(input_path, output_path)
        report["stripped_chunks"] = stripped
        report["badge_removed"] = False
    else:
        # Load image, remove badge, then save without metadata
        img_pil = Image.open(input_path).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        img_cv = remove_badge(img_cv, badge_size=badge_size)

        # Save via PIL (clean, no metadata)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        Image.fromarray(img_rgb).save(output_path, format="PNG")

        # Also report what metadata would have been stripped
        stripped = {}
        with open(input_path, "rb") as f:
            f.read(8)  # signature
            while True:
                header = f.read(8)
                if len(header) < 8:
                    break
                length = struct.unpack(">I", header[:4])[0]
                chunk_type = header[4:8].decode("ascii", errors="replace")
                f.read(length + 4)  # data + crc
                if chunk_type not in ESSENTIAL_CHUNKS:
                    stripped[chunk_type] = length

        report["stripped_chunks"] = stripped
        report["badge_removed"] = True
        report["badge_size"] = badge_size

    # Report file size change
    report["input_size"] = input_path.stat().st_size
    report["output_size"] = output_path.stat().st_size

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Strip metadata and remove visible watermarks from AI-generated images"
    )
    parser.add_argument("input", type=Path, help="Input image or directory")
    parser.add_argument("output", type=Path, help="Output image or directory")
    parser.add_argument(
        "--badge-size", type=int, default=90,
        help="Size of corner badge to inpaint in pixels (default: 90)",
    )
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Only strip metadata, don't remove visible badge",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        # Process all images in directory
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        images = sorted(
            p for p in args.input.iterdir()
            if p.suffix.lower() in exts and not p.name.startswith(".")
        )
        if not images:
            print(f"No images found in {args.input}")
            sys.exit(1)

        args.output.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            out_path = args.output / img_path.name
            report = clean_image(
                img_path, out_path,
                badge_size=args.badge_size,
                metadata_only=args.metadata_only,
            )
            chunks = report["stripped_chunks"]
            chunk_str = ", ".join(f"{k}({v}B)" for k, v in chunks.items()) or "none"
            size_delta = report["input_size"] - report["output_size"]
            print(f"  {img_path.name} → {out_path.name}  "
                  f"stripped=[{chunk_str}]  saved {size_delta:,}B  "
                  f"badge={'yes' if report['badge_removed'] else 'no'}")
    else:
        # Single image
        report = clean_image(
            args.input, args.output,
            badge_size=args.badge_size,
            metadata_only=args.metadata_only,
        )
        chunks = report["stripped_chunks"]
        size_delta = report["input_size"] - report["output_size"]

        print(f"Input:  {args.input} ({report['input_size']:,} bytes)")
        print(f"Output: {args.output} ({report['output_size']:,} bytes)")
        print(f"Size reduction: {size_delta:,} bytes")
        if chunks:
            print(f"Stripped chunks:")
            for name, size in chunks.items():
                print(f"  {name}: {size:,} bytes")
        if report["badge_removed"]:
            print(f"Badge inpainted: {report['badge_size']}px from corner")


if __name__ == "__main__":
    main()
