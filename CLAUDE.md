# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Research pipeline for evaluating watermark robustness against diffusion-based attacks.
The diffusion model is the **weapon** (img2img regeneration), not the watermark.
We tune attack hyperparameters against proxy watermarks (Tree-Ring, Stable Signature, VideoSeal, watermark-rs),
then transfer optimal settings to attack SynthID (Google DeepMind, closed-source).

## Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Tests (fast — no model downloads needed)
pytest

# Tests (slow — requires SD 2.1 cached in .models/)
pytest -m slow

# Single test file
pytest tests/test_tree_ring.py -v

# Download Stable Signature weights (1.2MB extractor + 198MB decoder)
python scripts/download_models.py

# Generate watermarked images from prompts
python scripts/generate.py --watermark-type tree-ring --num-images 10
python scripts/generate.py --watermark-type stable-signature --num-images 10

# Hyperparameter sweep (attack watermarked images)
python scripts/sweep.py --max-images 3   # quick test
python scripts/sweep.py                  # full sweep (60 combos × all images)
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DW_MODEL_DIR` | `./.models/` | Pretrained model cache (SD 2.1 is ~5GB) |
| `DW_IMAGE_DIR` | `./images/` | Generated/watermarked image output |
| `DW_RESULTS_DIR` | `./results/` | Sweep CSV results and plots |

## Architecture

### Watermark methods (`watermark/`)

- **Tree-Ring** (`tree_ring.py`, `detection.py`): Fourier-domain noise injection. Zero-bit (presence/absence). Detection requires DDIM inversion (~30s/image).
- **Stable Signature** (`stable_signature.py`, `detection_ss.py`): Fine-tuned VAE decoder embeds fixed 48-bit key. Detection via HiDDeN CNN extractor (~0.1s/image). Downloads models from `dl.fbaipublicfiles.com`.
- **DDIM utilities** (`inversion.py`): Shared encode/invert/reconstruct functions. `load_pipeline()` returns SD 2.1 with DDIM scheduler on MPS or CPU.

### Attack (`attack/regeneration.py`)

Single attack method: img2img regeneration. Wraps any SD pipeline as `StableDiffusionImg2ImgPipeline`, sweeps `strength`, `guidance_scale`, `num_inference_steps`. Returns attacked image + SSIM/PSNR metrics.

### Scripts (`scripts/`)

`generate.py` → produce watermarked images → `sweep.py` → grid search attack params → `visualize.py` → Pareto plots.
All scripts read `metadata.json` to auto-detect watermark type and switch detectors.

### Test data

1000 OCR document PNGs at `~/data/diffusion/ocr_quality_pngs/` (1.0 GB, 300 DPI).
Used for existing-image watermarking workflows (not generated from prompts).

## Hardware Notes

- **Apple Silicon (MPS)**: Primary target. All entry points set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **FFT ops**: Fall back to CPU automatically (MPS doesn't support FFT). Negligible overhead on 96×96 latents.
- **SD 2.1 latent size**: 96×96 (produces 768×768 images). Hardcoded in several places.

## Testing Conventions

- Fast tests use mocked models and synthetic data.
- Slow tests marked `@pytest.mark.slow` (configured in `pytest.ini`). Skipped automatically if SD 2.1 not cached.
- Tests that need the model check for `.models/` existence and `pytest.skip()` if absent.
