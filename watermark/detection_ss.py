"""Stable Signature detection — HiDDeN extractor + binomial test.

Detection pipeline (no diffusion model needed):
1. ImageNet-normalize image → (1, 3, H, W) tensor
2. extractor(tensor) → (1, 48) logits
3. Threshold at 0 → 48 predicted bits
4. Compare to key → count matching bits → binomial test → p-value

This is orders of magnitude faster than Tree-Ring detection (one CNN forward pass
vs 50 DDIM inversion steps).
"""

import torch
import numpy as np
from PIL import Image
from scipy.stats import binomtest

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_extractor(path: str) -> torch.jit.ScriptModule:
    """Load the HiDDeN extractor from a TorchScript file.

    Args:
        path: Path to dec_48b_whit.torchscript.pt

    Returns:
        TorchScript module ready for inference.
    """
    extractor = torch.jit.load(path, map_location=DEVICE)
    extractor.eval()
    return extractor


def _normalize_image(image: torch.Tensor | Image.Image | np.ndarray) -> torch.Tensor:
    """Convert image to ImageNet-normalized (1, 3, H, W) float32 tensor.

    Accepts:
        - torch.Tensor: (1, 3, H, W) or (3, H, W) — values in [0, 1] or already normalized
        - PIL.Image: RGB image
        - np.ndarray: (H, W, 3) uint8

    Returns:
        Normalized tensor of shape (1, 3, H, W).
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    elif isinstance(image, np.ndarray):
        img = image.astype(np.float32) / 255.0 if image.max() > 1.0 else image.astype(np.float32)
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        tensor = image.float()
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        # If values look like [0, 255] range, normalize
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # ImageNet normalization
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor


def _binomial_p_value(matching_bits: int, total_bits: int) -> float:
    """Compute one-sided p-value: probability of seeing ≥ matching_bits matches by chance.

    Under H0 (no watermark): each bit matches with probability 0.5, independently.

    Args:
        matching_bits: Number of bits matching the key.
        total_bits: Total number of bits.

    Returns:
        p-value (probability of this many or more matches under H0).
    """
    result = binomtest(matching_bits, total_bits, 0.5, alternative="greater")
    return float(result.pvalue)


def detect_ss(
    extractor,
    image: torch.Tensor | Image.Image | np.ndarray,
    key: list[int],
    threshold: float = 0.01,
) -> dict:
    """Detect Stable Signature watermark in an image.

    Args:
        extractor: Loaded HiDDeN extractor (TorchScript or callable).
        image: Image to test — PIL, numpy, or torch tensor.
        key: Expected watermark key as list of 0/1 ints.
        threshold: p-value threshold for detection (default 0.01).

    Returns:
        Dict with: p_value, detected, bit_accuracy, matching_bits.
    """
    tensor = _normalize_image(image)
    try:
        device = next(extractor.parameters()).device
        if not isinstance(device, torch.device):
            device = "cpu"
    except (StopIteration, AttributeError, TypeError):
        device = "cpu"
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = extractor(tensor)  # (1, n_bits)

    # Threshold at 0: positive → 1, negative → 0
    predicted_bits = (logits[0] > 0).cpu().tolist()
    n_bits = len(key)

    # Count matching bits
    matching = sum(1 for pred, expected in zip(predicted_bits, key) if pred == expected)

    p_value = _binomial_p_value(matching, n_bits)
    detected = p_value < threshold

    return {
        "p_value": p_value,
        "detected": detected,
        "bit_accuracy": matching / n_bits,
        "matching_bits": matching,
    }
