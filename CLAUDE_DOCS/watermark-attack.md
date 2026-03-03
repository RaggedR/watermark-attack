# Feature: Diffusion Watermark Breaking Pipeline
> Hyperparameter tuning pipeline to find optimal img2img regeneration parameters that break Tree-Ring watermarks while preserving image quality.

## Overview
This project implements a pipeline to break diffusion model watermarks using img2img regeneration attacks. The strategy:

1. **Proxy target**: Tune attack parameters against Tree-Ring (open-source, local detector) since SynthID's detector is closed-source
2. **Generate**: Create watermarked images using Tree-Ring + Stable Diffusion 2.1
3. **Attack**: Apply img2img regeneration at various strength/guidance/step settings
4. **Detect**: DDIM-invert the attacked image and check if the watermark survives in Fourier space
5. **Sweep**: Grid search over hyperparameters, recording quality (SSIM/PSNR) vs detection (p-value)
6. **Transfer**: Apply winning parameters to SynthID images, verify via Gemini

### Why Tree-Ring as proxy
Tree-Ring is the most widely studied Fourier-space watermark, used as a benchmark in NeurIPS 2024 and USENIX 2025 attack papers. It's arguably harder to break than SynthID since the attacker can't use gradient-based guidance without the detector.

### MPS + FFT Hybrid
UNet/VAE run on MPS (Apple Silicon GPU). FFT operations fall back to CPU since MPS doesn't support FFT — this is fine since FFT on 64x64 latents is microseconds.

## Resources
- [Tree-Ring Paper](https://arxiv.org/abs/2305.20030)
- [SynthID Blog](https://deepmind.google/technologies/synthid/)

## Assets
| File | Purpose |
|------|---------|
| `watermark/tree_ring.py` | Key generation + Fourier embedding |
| `watermark/inversion.py` | DDIM inversion (modern diffusers) |
| `watermark/detection.py` | Fourier-space detection + p-value |
| `attack/regeneration.py` | img2img regeneration attack |
| `scripts/generate.py` | Generate watermarked test images |
| `scripts/sweep.py` | Hyperparameter sweep |
| `scripts/visualize.py` | Pareto frontier plots |
