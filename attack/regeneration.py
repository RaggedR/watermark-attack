"""Regeneration attack: img2img to destroy watermarks while preserving quality.

The key insight: running an image through img2img at moderate strength adds
enough diffusion noise to destroy the watermark's Fourier signature, while
the denoising process reconstructs a visually similar image.
"""

import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def _compute_quality_metrics(original: Image.Image, attacked: Image.Image) -> dict:
    """Compute SSIM and PSNR between original and attacked images."""
    orig_np = np.array(original).astype(np.float64)
    atk_np = np.array(attacked).astype(np.float64)

    # Ensure same size
    if orig_np.shape != atk_np.shape:
        attacked = attacked.resize(original.size, Image.LANCZOS)
        atk_np = np.array(attacked).astype(np.float64)

    ssim = structural_similarity(orig_np, atk_np, channel_axis=2, data_range=255)
    psnr = peak_signal_noise_ratio(orig_np, atk_np, data_range=255)

    return {"ssim": ssim, "psnr": psnr}


def regeneration_attack(
    pipe,
    image: Image.Image,
    strength: float = 0.3,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    prompt: str = "",
    seed: int = 0,
) -> tuple[Image.Image, dict]:
    """Apply img2img regeneration attack to destroy watermark.

    Args:
        pipe: StableDiffusionPipeline (will be wrapped as img2img internally).
        image: Watermarked image to attack.
        strength: Fraction of diffusion steps to apply (0 = no change, 1 = full regeneration).
        guidance_scale: Classifier-free guidance weight.
        num_inference_steps: Total DDIM steps.
        prompt: Text prompt (empty = unconditional).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (attacked_image, metrics_dict) where metrics_dict has ssim and psnr.
    """
    # Create img2img pipeline from the same components
    img2img = StableDiffusionImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    # Resize input to pipeline's expected resolution
    target_size = pipe.unet.config.sample_size * pipe.vae_scale_factor
    image_resized = image.resize((target_size, target_size), Image.LANCZOS)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    with torch.no_grad():
        result = img2img(
            prompt=prompt,
            image=image_resized,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

    attacked = result.images[0]
    metrics = _compute_quality_metrics(image_resized, attacked)

    return attacked, metrics
