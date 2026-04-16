#!/usr/bin/env python3
"""Download DIV2K and DiffusionDB datasets to external drive.

Usage:
    python scripts/download_datasets.py --output-dir "/Volumes/One Touch/diffusion-watermarking/datasets"
    python scripts/download_datasets.py --div2k-only
    python scripts/download_datasets.py --diffusiondb-only
"""

import argparse
from pathlib import Path


def download_div2k(output_dir: Path):
    """Download DIV2K high-res images (800 train + 100 val = 900 images).

    Downloads directly from ETH Zurich (official source) as zip files.
    """
    import shutil
    import subprocess
    import zipfile

    div2k_dir = output_dir / "div2k"
    div2k_dir.mkdir(parents=True, exist_ok=True)

    existing = list(div2k_dir.glob("*.png"))
    if len(existing) >= 900:
        print(f"DIV2K: already have {len(existing)} images, skipping")
        return

    base_url = "https://data.vision.ee.ethz.ch/cvl/DIV2K"
    zips = [
        ("DIV2K_train_HR.zip", "DIV2K_train_HR"),
        ("DIV2K_valid_HR.zip", "DIV2K_valid_HR"),
    ]

    tmp_dir = output_dir / "_div2k_tmp"
    tmp_dir.mkdir(exist_ok=True)

    for zip_name, inner_dir in zips:
        zip_path = tmp_dir / zip_name
        if not zip_path.exists():
            url = f"{base_url}/{zip_name}"
            print(f"  Downloading {zip_name}...")
            subprocess.run(
                ["curl", "-L", "-o", str(zip_path), url],
                check=True,
            )

        print(f"  Extracting {zip_name}...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_dir)

        # Move PNGs into div2k_dir with sequential naming
        src_dir = tmp_dir / inner_dir
        for png in sorted(src_dir.glob("*.png")):
            dest = div2k_dir / png.name
            shutil.move(str(png), str(dest))

    # Clean up temp files (use subprocess on exFAT where shutil.rmtree can fail)
    subprocess.run(["rm", "-rf", str(tmp_dir)], check=False)

    final_count = len(list(div2k_dir.glob("*.png")))
    print(f"DIV2K: {final_count} images saved to {div2k_dir}")


def download_diffusiondb(output_dir: Path):
    """Download ~1000 DiffusionDB images (one partition of SD-generated images).

    Downloads a single zip partition directly from the HuggingFace repo,
    bypassing the legacy dataset script that newer datasets versions reject.
    """
    import subprocess
    import zipfile

    from PIL import Image

    ddb_dir = output_dir / "diffusiondb"
    ddb_dir.mkdir(parents=True, exist_ok=True)

    existing = list(ddb_dir.glob("*.png"))
    if len(existing) >= 1000:
        print(f"DiffusionDB: already have {len(existing)} images, skipping")
        return

    print("Downloading DiffusionDB partition from HuggingFace...")
    zip_path = output_dir / "_diffusiondb_tmp.zip"
    if not zip_path.exists():
        # Download one partition (~1000 images, ~375 MB)
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id="poloclub/diffusiondb",
            filename="images/part-000001.zip",
            repo_type="dataset",
        )
        # Copy to our location (hf_hub caches in ~/.cache)
        subprocess.run(["cp", downloaded, str(zip_path)], check=True)

    print("  Extracting images...")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        for i, member in enumerate(sorted(members)):
            dest = ddb_dir / f"{i:04d}.png"
            if dest.exists():
                continue
            with zf.open(member) as src:
                img = Image.open(src)
                img.save(dest, "PNG")
            if (i + 1) % 200 == 0:
                print(f"    Saved {i + 1}/{len(members)}")

    # Clean up zip
    zip_path.unlink(missing_ok=True)

    final_count = len(list(ddb_dir.glob("*.png")))
    print(f"DiffusionDB: {final_count} images saved to {ddb_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download image datasets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Volumes/One Touch/diffusion-watermarking/datasets"),
        help="Where to save datasets",
    )
    parser.add_argument(
        "--div2k-only", action="store_true", help="Only download DIV2K"
    )
    parser.add_argument(
        "--diffusiondb-only", action="store_true", help="Only download DiffusionDB"
    )
    args = parser.parse_args()

    if not args.output_dir.parent.exists():
        print(f"Error: {args.output_dir.parent} does not exist. Is the drive mounted?")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    if not args.diffusiondb_only:
        download_div2k(args.output_dir)

    if not args.div2k_only:
        download_diffusiondb(args.output_dir)

    print("\n--- Summary ---")
    for subdir in sorted(args.output_dir.iterdir()):
        if subdir.is_dir():
            count = len(list(subdir.glob("*.png")))
            print(f"  {subdir.name}: {count} images")


if __name__ == "__main__":
    main()
