# Feature: Diffusion Watermark Breaking Pipeline
> Hyperparameter tuning pipeline to find optimal img2img regeneration parameters that break diffusion watermarks while preserving image quality. Supports Tree-Ring and Stable Signature as proxy benchmarks.

## Overview
This project implements a pipeline to break diffusion model watermarks using img2img regeneration attacks. The strategy:

1. **Proxy targets**: Tune attack parameters against Tree-Ring and Stable Signature (open-source, local detectors) since SynthID's detector is closed-source
2. **Generate**: Create watermarked images using SD 2.1 with either watermark type
3. **Attack**: Apply img2img regeneration at various strength/guidance/step settings
4. **Detect**: Check if watermark survives (DDIM inversion for Tree-Ring, CNN for Stable Signature)
5. **Sweep**: Grid search over hyperparameters, recording quality (SSIM/PSNR) vs detection (p-value)
6. **Transfer**: Apply winning parameters to SynthID images, verify via Gemini

### Two watermark benchmarks

| | Tree-Ring | Stable Signature |
|---|---|---|
| **Paper** | Wen et al., 2023 | Fernandez et al., ICCV 2023 |
| **Where watermark lives** | Initial noise (Fourier domain) | VAE decoder weights (learned) |
| **Generation** | Modify noise → standard decoder | Standard noise → patched decoder |
| **Detection** | DDIM inversion (50 UNet passes) + FFT | CNN forward pass (one shot) |
| **Detection speed** | ~30s per image | ~0.1s per image |
| **Closer to SynthID?** | No (Fourier injection) | Yes (learned decoder modification) |

### Why two benchmarks
- **Tree-Ring** is the most widely studied Fourier-space watermark (NeurIPS 2024, USENIX 2025). Good baseline but architecturally different from SynthID.
- **Stable Signature** uses a learned decoder modification, which is architecturally closer to SynthID. If the same `(strength, guidance_scale, steps)` triple breaks both, that's strong evidence it'll break SynthID too.

### MPS + FFT Hybrid
UNet/VAE run on MPS (Apple Silicon GPU). FFT operations fall back to CPU since MPS doesn't support FFT — this is fine since FFT on 96x96 latents is microseconds.

## Resources
- [Tree-Ring Paper](https://arxiv.org/abs/2305.20030)
- [Stable Signature Paper](https://arxiv.org/abs/2303.15435)
- [SynthID Blog](https://deepmind.google/technologies/synthid/)

## Assets
| File | Purpose |
|------|---------|
| `watermark/tree_ring.py` | Tree-Ring key generation + Fourier embedding |
| `watermark/detection.py` | Fourier-space detection + p-value |
| `watermark/stable_signature.py` | Stable Signature key management, model downloads, decoder patching |
| `watermark/detection_ss.py` | HiDDeN extractor detection + binomial test |
| `watermark/inversion.py` | DDIM inversion (shared by Tree-Ring detection) |
| `attack/regeneration.py` | img2img regeneration attack (watermark-agnostic) |
| `scripts/generate.py` | Generate watermarked test images (`--watermark-type` flag) |
| `scripts/sweep.py` | Hyperparameter sweep (auto-detects watermark type from metadata) |
| `scripts/download_models.py` | Download Stable Signature weights |
| `scripts/visualize.py` | Pareto frontier plots |
