"""Fourier-space watermark detection with statistical testing.

Two entry points:
- detect_in_noise(): For testing — operates directly on a noise tensor.
- detect_watermark(): Full pipeline — takes an image, DDIM-inverts it, then detects.
"""

import torch
import numpy as np
from scipy import stats
from PIL import Image


def _compute_fourier_distance(
    noise: torch.Tensor,
    key: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """Compute L1 distance between noise's Fourier coefficients and the key at ring positions.

    Args:
        noise: Noise tensor, shape (B, C, H, W).
        key: Watermark key, shape (H, W).
        mask: Binary ring mask, shape (H, W).

    Returns:
        Dict with l1_distance, p_value, detected (bool).
    """
    B, C, H, W = noise.shape
    mask_bool = mask.cpu().bool()
    key_cpu = key.cpu()
    n_positions = mask_bool.sum().item()

    # Collect L1 distances across all channels
    all_diffs = []
    for b in range(B):
        for c in range(C):
            channel = noise[b, c].cpu().float()
            freq = torch.fft.fftshift(torch.fft.fft2(channel))
            extracted = freq.real[mask_bool]
            expected = key_cpu[mask_bool]
            diffs = (extracted - expected).abs()
            all_diffs.append(diffs)

    all_diffs = torch.cat(all_diffs)
    mean_l1 = all_diffs.mean().item()

    # Statistical test: under H0 (no watermark), the extracted Fourier coefficients
    # at ring positions are independent Gaussians. The L1 distance from the key
    # follows a folded normal distribution. We use a one-sample t-test comparing
    # the observed differences to the expected mean under H0.
    #
    # Under H0: each diff_i ~ |N(0, sigma)| (half-normal) where sigma ≈ 1
    # Under H1 (watermark present): diff_i ≈ 0
    #
    # We test whether the mean L1 is significantly lower than expected under H0.
    # Expected mean of half-normal with sigma=1: sqrt(2/pi) ≈ 0.798
    expected_mean_h0 = np.sqrt(2 / np.pi)
    diffs_np = all_diffs.numpy()

    # One-sample t-test: is the observed mean significantly below H0 expectation?
    t_stat, two_sided_p = stats.ttest_1samp(diffs_np, expected_mean_h0)

    # We want a one-sided test (is mean L1 *lower* than expected?)
    if t_stat < 0:
        p_value = two_sided_p / 2  # One-sided: mean is below H0
    else:
        p_value = 1 - two_sided_p / 2  # Mean is above H0 — no watermark

    detected = bool(p_value < 0.01)

    return {
        "l1_distance": mean_l1,
        "p_value": float(p_value),
        "detected": detected,
        "t_statistic": t_stat,
        "n_positions": n_positions * C * B,
    }


def detect_in_noise(
    noise: torch.Tensor,
    key: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """Detect watermark directly in a noise tensor (no model needed).

    Use this for unit testing the detection math without loading SD.

    Args:
        noise: Noise tensor, shape (B, C, H, W).
        key: Watermark key, shape (H, W).
        mask: Binary ring mask, shape (H, W).

    Returns:
        Detection result dict with p_value, l1_distance, detected.
    """
    return _compute_fourier_distance(noise, key, mask)


def detect_watermark(
    pipe,
    image: Image.Image,
    key: torch.Tensor,
    mask: torch.Tensor,
    num_inference_steps: int = 50,
) -> dict:
    """Full detection pipeline: DDIM-invert image → detect in Fourier space.

    Args:
        pipe: Loaded StableDiffusionPipeline.
        image: Image to test for watermark.
        key: Watermark key, shape (H, W).
        mask: Binary ring mask, shape (H, W).
        num_inference_steps: Steps for DDIM inversion.

    Returns:
        Detection result dict with p_value, l1_distance, detected.
    """
    from watermark.inversion import ddim_invert

    inverted = ddim_invert(pipe, image, num_inference_steps=num_inference_steps)
    return _compute_fourier_distance(inverted, key, mask)
