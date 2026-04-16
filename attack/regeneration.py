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

    # Resize to pipeline's expected resolution, preserving aspect ratio
    # by rounding each dimension to the nearest multiple of vae_scale_factor.
    original_size = image.size  # (W, H)
    scale_factor = pipe.vae_scale_factor
    target_w = round(image.width / scale_factor) * scale_factor
    target_h = round(image.height / scale_factor) * scale_factor
    image_resized = image.resize((target_w, target_h), Image.LANCZOS)

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
    del result, img2img, image_resized

    # Resize back to original dimensions
    if attacked.size != original_size:
        attacked = attacked.resize(original_size, Image.LANCZOS)

    metrics = _compute_quality_metrics(image, attacked)

    return attacked, metrics


def _make_feather_mask(tile_size: int, overlap: int) -> np.ndarray:
    """Create a linear feathering mask for tile blending.

    The mask is 1.0 in the interior and linearly ramps from 0→1
    across the overlap zone on each edge.  When overlapping tiles
    are summed with their masks, the overlap zone blends smoothly.
    """
    mask = np.ones((tile_size, tile_size), dtype=np.float32)
    if overlap <= 0:
        return mask

    ramp = np.linspace(0, 1, overlap, dtype=np.float32)
    # left / right
    mask[:, :overlap] *= ramp[np.newaxis, :]
    mask[:, -overlap:] *= ramp[np.newaxis, ::-1]
    # top / bottom
    mask[:overlap, :] *= ramp[:, np.newaxis]
    mask[-overlap:, :] *= ramp[::-1, np.newaxis]

    return mask


def tiled_regeneration_attack(
    pipe,
    image: Image.Image,
    strength: float = 0.3,
    tile_size: int = 512,
    overlap: int = 64,
    guidance_scale: float = 1.0,
    num_inference_steps: int = 50,
    prompt: str = "",
    seed: int = 0,
) -> tuple[Image.Image, dict]:
    """Attack a large image by splitting into overlapping tiles.

    Each tile is attacked independently via ``regeneration_attack`` at
    the pipeline's native resolution, then reassembled with linear
    feathering in the overlap zones to avoid visible seams.

    Args:
        pipe: StableDiffusionPipeline (passed through to regeneration_attack).
        image: Input image (any size ≥ tile_size).
        strength: img2img noise strength.
        tile_size: Tile edge length in pixels (should match pipeline's native res).
        overlap: Overlap in pixels between adjacent tiles.
        guidance_scale: CFG weight.
        num_inference_steps: DDIM steps per tile.
        prompt: Text prompt.
        seed: Base seed (incremented per tile for diversity).

    Returns:
        Tuple of (attacked_image, metrics_dict) with whole-image SSIM/PSNR.
    """
    W, H = image.size
    stride = tile_size - overlap

    # Compute tile grid positions — use linspace-style placement so
    # we get exactly ceil(dim/stride) tiles without degenerate near-
    # total-overlap edge tiles.
    def _tile_positions(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        n_tiles = max(1, -(-((length - tile_size)) // stride) + 1)  # ceil division
        if n_tiles == 1:
            return [0]
        # Evenly space n_tiles across the dimension
        positions = [round(i * (length - tile_size) / (n_tiles - 1)) for i in range(n_tiles)]
        return positions

    xs = _tile_positions(W)
    ys = _tile_positions(H)

    feather = _make_feather_mask(tile_size, overlap)
    # Expand mask to 3 channels
    feather_3ch = feather[:, :, np.newaxis]

    canvas = np.zeros((H, W, 3), dtype=np.float64)
    weight = np.zeros((H, W, 1), dtype=np.float64)

    for tile_idx, (x, y) in enumerate([(x, y) for y in ys for x in xs]):
        tile = image.crop((x, y, x + tile_size, y + tile_size))
        attacked_tile, _ = regeneration_attack(
            pipe,
            tile,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            prompt=prompt,
            seed=seed + tile_idx,
        )
        tile_np = np.array(attacked_tile).astype(np.float64)
        canvas[y : y + tile_size, x : x + tile_size] += tile_np * feather_3ch
        weight[y : y + tile_size, x : x + tile_size] += feather_3ch

    # Normalise by accumulated weights
    np.divide(canvas, weight, out=canvas, where=weight > 0)
    attacked = Image.fromarray(canvas.clip(0, 255).astype(np.uint8))

    metrics = _compute_quality_metrics(image, attacked)
    return attacked, metrics


def sandwich_attack(
    pipe,
    image: Image.Image,
    strength_1: float = 0.15,
    strength_2: float = 0.15,
    num_rings: int = 15,
    max_angle: float = 1.5,
    tile_size: int = 512,
    overlap: int = 0,
    guidance_scale: float = 1.0,
    num_inference_steps: int = 50,
    seed: int = 0,
) -> tuple[Image.Image, dict]:
    """Sandwich attack: diffuse → ring-rotate → diffuse.

    Two tiled img2img passes with a concentric ring rotation between them.
    The ring rotation disrupts spatial/frequency coherence that survives
    the first diffusion pass, and the second pass smooths rotation artifacts.

    Args:
        pipe: StableDiffusionPipeline.
        image: Watermarked input image.
        strength_1: img2img strength for the first diffusion pass.
        strength_2: img2img strength for the second diffusion pass.
        num_rings: Number of concentric rings for rotation.
        max_angle: Ring rotation angle in degrees (alternates ±).
        tile_size: Tile edge length for tiled regeneration.
        overlap: Tile overlap in pixels.
        guidance_scale: CFG weight.
        num_inference_steps: DDIM steps per tile per pass.
        seed: Base seed.

    Returns:
        Tuple of (attacked_image, metrics_dict) with whole-image SSIM/PSNR
        measured against the original input.
    """
    from attack.ring import ring_rotation_attack

    # Pass 1: diffusion
    after_diffuse_1, _ = tiled_regeneration_attack(
        pipe, image,
        strength=strength_1,
        tile_size=tile_size,
        overlap=overlap,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )

    # Pass 2: ring rotation
    after_ring = ring_rotation_attack(
        after_diffuse_1,
        num_rings=num_rings,
        max_angle=max_angle,
    )
    del after_diffuse_1

    # Pass 3: diffusion again (different seed to avoid correlation)
    after_diffuse_2, _ = tiled_regeneration_attack(
        pipe, after_ring,
        strength=strength_2,
        tile_size=tile_size,
        overlap=overlap,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed + 1000,
    )
    del after_ring

    metrics = _compute_quality_metrics(image, after_diffuse_2)
    return after_diffuse_2, metrics


def wedge_attack(
    pipe,
    image: Image.Image,
    noise_sigma: float = 25.0,
    num_rings: int = 15,
    max_angle: float = 1.5,
    denoise_strength: float = 0.3,
    tile_size: int = 512,
    overlap: int = 0,
    guidance_scale: float = 1.0,
    num_inference_steps: int = 50,
    seed: int = 0,
) -> tuple[Image.Image, dict]:
    """Wedge attack: add noise → ring-rotate → diffusion denoise.

    Adds Gaussian noise in pixel space to corrupt the watermark signal,
    applies concentric ring rotation to scramble spatial/frequency patterns,
    then uses the diffusion model to denoise and reconstruct a clean image.
    One diffusion pass instead of two.

    Args:
        pipe: StableDiffusionPipeline.
        image: Watermarked input image.
        noise_sigma: Gaussian noise std dev (0–255 scale).
        num_rings: Number of concentric rings for rotation.
        max_angle: Ring rotation angle in degrees (alternates ±).
        denoise_strength: img2img strength for the denoising pass.
        tile_size: Tile edge length for tiled regeneration.
        overlap: Tile overlap in pixels.
        guidance_scale: CFG weight.
        num_inference_steps: DDIM steps per tile.
        seed: Base seed.

    Returns:
        (attacked_image, metrics_dict) with SSIM/PSNR vs original.
    """
    from attack.ring import ring_rotation_attack

    # Step 1: Add Gaussian noise in pixel space
    rng = np.random.RandomState(seed)
    img_array = np.array(image).astype(np.float64)
    noisy = img_array + rng.normal(0, noise_sigma, img_array.shape)
    noisy_img = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))

    # Step 2: Ring rotation
    rotated_img = ring_rotation_attack(noisy_img, num_rings=num_rings, max_angle=max_angle)
    del noisy_img

    # Step 3: Diffusion model denoises and reconstructs
    denoised, _ = tiled_regeneration_attack(
        pipe, rotated_img,
        strength=denoise_strength,
        tile_size=tile_size,
        overlap=overlap,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
    del rotated_img

    metrics = _compute_quality_metrics(image, denoised)
    return denoised, metrics
