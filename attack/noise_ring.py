"""Fast noise-ring-denoise attack.

No diffusion model needed. The pipeline:
  1. Add Gaussian noise (corrupts watermark signal)
  2. Ring-rotate (scrambles spatial/frequency coherence)
  3. Denoise with non-local means (removes noise, smooths artifacts)

Runs in seconds per image on CPU.
"""

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage.restoration import denoise_nl_means, estimate_sigma
from attack.ring import ring_rotation_attack


def noise_ring_denoise_attack(
    image: Image.Image,
    noise_sigma: float = 25.0,
    num_rings: int = 15,
    max_angle: float = 1.5,
    denoise_h: float | None = None,
    seed: int = 42,
) -> tuple[Image.Image, dict]:
    """Add noise → ring rotate → denoise.

    Args:
        image: Input PIL image (RGB).
        noise_sigma: Gaussian noise std dev (0–255 scale). Higher = more destruction.
        num_rings: Number of concentric rings for rotation.
        max_angle: Ring rotation angle in degrees (alternates ±).
        denoise_h: NLM filter strength. None = auto (1.2 × estimated sigma).
        seed: RNG seed for reproducible noise.

    Returns:
        (attacked_image, metrics_dict) with SSIM/PSNR vs original.
    """
    rng = np.random.RandomState(seed)
    orig_array = np.array(image).astype(np.float64)

    # Step 1: Add Gaussian noise
    noisy = orig_array + rng.normal(0, noise_sigma, orig_array.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy)

    # Step 2: Ring rotation
    rotated_img = ring_rotation_attack(noisy_img, num_rings=num_rings, max_angle=max_angle)
    rotated = np.array(rotated_img)

    # Step 3: Denoise with non-local means
    sigma_est = estimate_sigma(rotated, channel_axis=2, average_sigmas=True)
    if denoise_h is None:
        denoise_h = 1.2 * sigma_est
    denoised = denoise_nl_means(
        rotated, h=denoise_h, sigma=sigma_est,
        fast_mode=True, patch_size=5, patch_distance=6,
        channel_axis=2,
    )
    result = (np.clip(denoised, 0, 1) * 255).astype(np.uint8)
    result_img = Image.fromarray(result)

    # Metrics vs original
    orig_f = orig_array / 255.0
    result_f = result.astype(np.float64) / 255.0
    ssim = structural_similarity(orig_f, result_f, channel_axis=2, data_range=1.0)
    psnr = peak_signal_noise_ratio(orig_f, result_f, data_range=1.0)

    return result_img, {"ssim": ssim, "psnr": psnr, "noise_sigma": noise_sigma, "sigma_est": sigma_est}
