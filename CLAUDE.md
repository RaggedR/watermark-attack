# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Research pipeline for evaluating watermark robustness against diffusion-based attacks.
The diffusion model is the **weapon** (img2img regeneration), not the watermark.
We tune attack hyperparameters against proxy watermarks (VideoSeal, Tree-Ring, Stable Signature),
then transfer optimal settings to attack SynthID (Google DeepMind, closed-source).

## Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Tests (fast — no model downloads needed)
pytest

# Tests (slow — requires models cached)
pytest -m slow

# Single test file
pytest tests/test_adapters.py -v

# Embed watermarks into existing images (primary workflow)
python scripts/embed.py --method videoseal --input-dir datasets/diffusiondb --num-images 5
python scripts/embed.py --method tree-ring --input-dir datasets/div2k --num-images 5

# Attack sweep — diffusion only (grid-search img2img params)
python scripts/attack_sweep.py --max-images 3    # quick test
python scripts/attack_sweep.py                   # full sweep

# Attack sweep — ring rotation only (no diffusion model needed for attack)
python scripts/attack_ring_sweep.py --max-images 3
python scripts/attack_ring_sweep.py --method tree-ring --dataset div2k

# Attack sweep — sandwich (diffuse → ring → diffuse)
python scripts/attack_sandwich_sweep.py --max-images 2
python scripts/attack_sandwich_sweep.py --method videoseal --combos '0.15:0.15,0.20:0.20'

# SynthID-specific pipeline
python scripts/generate_synthid.py --num-images 50   # requires GOOGLE_API_KEY
python scripts/attack_synthid_sweep.py --max-images 2 --strengths 0.10 0.30
python scripts/attack_synthid_sandwich.py --max-images 2

# Legacy: generate watermarked images from prompts
python scripts/generate.py --watermark-type tree-ring --num-images 10
python scripts/sweep.py --max-images 3

# Download Stable Signature weights
python scripts/download_models.py
python scripts/download_models.py --extractor-only   # just the 1.2MB extractor

# Visualise results
python scripts/visualize.py
```

## Environment Variables

Configured in `.env` (loaded via `python-dotenv`). Heavy assets live on an external drive.

| Variable | Default | Purpose |
|----------|---------|---------|
| `DW_MODEL_DIR` | `./.models/` | Pretrained model cache (SD 1.5 is ~2GB) |
| `DW_IMAGE_DIR` | `./images/` | Watermarked image output |
| `DW_RESULTS_DIR` | `./results/` | Sweep CSV results and plots |
| `GOOGLE_API_KEY` | — | Required for SynthID generation via Gemini API |

Typical `.env` pointing at an external drive:
```
DW_MODEL_DIR=/Volumes/One Touch/diffusion-watermarking/models
DW_IMAGE_DIR=/Volumes/One Touch/diffusion-watermarking/images
DW_RESULTS_DIR=/Volumes/One Touch/diffusion-watermarking/results
```

Local `ckpts/` is a symlink → external `models/` (VideoSeal hardcodes `ckpts/` in CWD).

## Architecture

### Core design: Adapter pattern

All watermark methods implement `WatermarkAdapter` (ABC in `watermark/adapters.py`):

```
WatermarkAdapter
├── embed(image, strength) → Image
├── extract(image) → {p_value, detected, bit_accuracy, ...}
└── metadata() → dict

Implementations:
├── VideoSealAdapter   — 256-bit neural encoder, ~0.5s detect
├── TreeRingAdapter    — 0-bit Fourier noise, ~30s detect (DDIM inversion)
└── StableSignatureAdapter — 48-bit learned VAE, ~0.1s detect (CNN)
```

Factory: `create_adapter("videoseal"|"tree-ring"|"stable-signature", pipe=..., seed=...)`.

### Pipeline flow

```
embed.py → watermarks existing images using an adapter
    ↓
attack_sweep.py → grid-searches img2img regeneration params
    ↓
visualize.py → plots Pareto frontiers (quality vs removal)
```

Scripts read `metadata.json` (written by embed.py) to auto-detect watermark type and route to the correct detector.

### Watermark methods (`watermark/`)

- **VideoSeal** (`videoseal_wm.py`): Meta's neural encoder/decoder. 256-bit payload. Closest open-source proxy to SynthID. Checkpoint auto-downloads to `ckpts/` (~260MB). Detector output shape: `(B, 1+nbits)` — first element is presence indicator.
- **Tree-Ring** (`tree_ring.py`, `detection.py`): Fourier-domain noise injection. Zero-bit (presence/absence). Detection requires DDIM inversion.
- **Stable Signature** (`stable_signature.py`, `detection_ss.py`): Fine-tuned VAE decoder embeds fixed 48-bit key. Detection via HiDDeN CNN extractor. Downloads models from `dl.fbaipublicfiles.com`.
- **DDIM utilities** (`inversion.py`): Shared encode/invert/reconstruct functions. `load_pipeline()` returns SD 1.5 with DDIM scheduler on MPS or CPU.

### Attack (`attack/regeneration.py`)

Single attack method: img2img regeneration. `regeneration_attack()` wraps any SD pipeline as `StableDiffusionImg2ImgPipeline`, sweeps `strength`, `guidance_scale`, `num_inference_steps`. Returns attacked image + SSIM/PSNR metrics.

`tiled_regeneration_attack()` handles images larger than 512×512 by splitting into overlapping patches with linear feathering.

### Fragile components

- **SS decoder patching**: `stable_signature.py` remaps LDM-format decoder keys to diffusers format (`_remap_ldm_decoder_keys`). Regex-based; breaks if HuggingFace restructures VAE keys.
- **VideoSeal card path**: `videoseal_wm.py` resolves model card paths relative to `videoseal.__file__`; assumes pip package structure.
- **`configs/attenuation.yaml`**: Must exist in CWD for VideoSeal (not shipped with pip wheel).

## Hardware Notes

- **Apple Silicon (MPS)**: Primary target. All entry points set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **FFT ops**: Fall back to CPU automatically (MPS doesn't support FFT).
- **Float32 required**: float16 on MPS causes NaN in VAE decoder.
- **Attack model**: SD 1.5 at 512×512 (~2s/step on MPS). SD 2.1 was removed from HuggingFace and was unusably slow on MPS.

## Testing Conventions

- Fast tests use mocked models and synthetic data — run with `pytest` (no flags).
- Slow tests marked `@pytest.mark.slow` (configured in `pytest.ini`).
- Key test files: `test_adapters.py` (adapter factory + properties), `test_videoseal.py` (tensor conversion + binomial math), `test_tree_ring.py` (key generation + ring masks).
