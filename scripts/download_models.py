#!/usr/bin/env python3
"""Download Stable Signature model weights.

Usage:
    python scripts/download_models.py                    # Download both (extractor + decoder)
    python scripts/download_models.py --extractor-only   # Just the 1.2MB extractor
    python scripts/download_models.py --decoder-only     # Just the 198MB decoder

Models are saved to DW_MODEL_DIR/stable_signature/ (default: .models/stable_signature/).
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from watermark.stable_signature import download_extractor, download_decoder


def main():
    parser = argparse.ArgumentParser(description="Download Stable Signature model weights")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--extractor-only", action="store_true",
                       help="Download only the extractor (1.2MB)")
    group.add_argument("--decoder-only", action="store_true",
                       help="Download only the decoder (198MB)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Model directory (default: DW_MODEL_DIR or .models/)")
    args = parser.parse_args()

    model_dir = args.model_dir or os.environ.get(
        "DW_MODEL_DIR",
        str(Path(__file__).resolve().parent.parent / ".models"),
    )

    if args.decoder_only:
        path = download_decoder(model_dir)
        print(f"Decoder ready: {path}")
    elif args.extractor_only:
        path = download_extractor(model_dir)
        print(f"Extractor ready: {path}")
    else:
        ext_path = download_extractor(model_dir)
        print(f"Extractor ready: {ext_path}")
        dec_path = download_decoder(model_dir)
        print(f"Decoder ready: {dec_path}")
        print("\nAll Stable Signature models downloaded.")


if __name__ == "__main__":
    main()
