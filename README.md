# Diffusion Watermarking — Attack Robustness Pipeline

> **Status: Work in progress.** The core attack pipeline works against two proxy watermarks. The full multi-method comparison and SynthID transfer attack are not yet implemented.

## Goal

Find diffusion model hyperparameters that reliably remove invisible watermarks from images while preserving visual quality. The diffusion model is the **weapon** — we use img2img regeneration as an attack, not as the watermarking method.

The end target is [SynthID](https://deepmind.google/technologies/synthid/) (Google DeepMind), whose detector is closed-source. We tune attacks against open-source proxy watermarks and transfer the optimal parameters to SynthID.

## Approach

```
watermarked image → img2img regeneration attack → detect watermark → sweep parameters
```

We grid-search over three attack hyperparameters — `strength`, `guidance_scale`, and `num_inference_steps` — using Stable Diffusion as the attack model. For each combination we measure:

- **Image quality**: SSIM and PSNR (higher = less distortion)
- **Watermark survival**: detection p-value or bit accuracy (lower = better attack)

The sweet spot is a Pareto frontier: maximum watermark removal with minimum quality loss.

## Proxy Watermarks

| Method | Type | Bits | Detection | Paper |
|--------|------|------|-----------|-------|
| **Tree-Ring** | Fourier noise injection | 0 (presence only) | DDIM inversion + FFT (~30s/image) | [Wen et al., 2023](https://arxiv.org/abs/2305.20030) |
| **Stable Signature** | Learned VAE decoder | 48-bit fixed key | HiDDeN CNN extractor (~0.1s/image) | [Fernandez et al., ICCV 2023](https://arxiv.org/abs/2303.15435) |

Planned additions:
- **VideoSeal/PixelSeal** (Meta) — 256-bit neural watermark, closest open-source analogue to SynthID
- **watermark-rs** — classical signal-processing watermark (robustness baseline)

## What Works Today

- Generate watermarked images with Tree-Ring or Stable Signature via Stable Diffusion 2.1
- Run img2img regeneration attacks with configurable parameters
- Detect watermark survival using method-specific detectors
- Hyperparameter sweep with CSV output and Pareto frontier plots

## What's Next

1. **Unified adapter layer** — common `embed()` / `extract()` interface across all watermark methods
2. **Post-hoc watermarking** — shift from generating watermarked images to watermarking existing images (1000-image test set)
3. **VideoSeal integration** — add Meta's neural watermark as the primary SynthID proxy
4. **Cross-method comparison** — which watermark is hardest to break? Do optimal attack params transfer?
5. **SynthID transfer attack** — apply tuned parameters to SynthID-watermarked images

## Quick Start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate watermarked images
python scripts/generate.py --watermark-type tree-ring --num-images 10

# Run attack sweep
python scripts/sweep.py --max-images 3

# Run tests (fast, no GPU required)
pytest
```

## Hardware

Developed on Apple Silicon (MPS). FFT operations fall back to CPU automatically. SD 2.1 weights (~5GB) are cached in `.models/`.

## References

- [Tree-Ring Watermarks (Wen et al., 2023)](https://arxiv.org/abs/2305.20030)
- [The Stable Signature (Fernandez et al., ICCV 2023)](https://arxiv.org/abs/2303.15435)
- [SynthID (Google DeepMind)](https://deepmind.google/technologies/synthid/)
