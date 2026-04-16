"""VideoSeal watermark — neural encoder/decoder for 256-bit embedding.

VideoSeal (Meta, 2024) uses learned CNN encoder + decoder for image watermarking.
Unlike Stable Signature (which bakes the watermark into the VAE decoder), VideoSeal
operates post-hoc on any image — making it closer to SynthID's approach.

Architecture:
    Embed:  image + message → neural encoder → watermarked image
    Detect: image → neural decoder → 256 logits → threshold → binomial test

This is the best open-source proxy for SynthID (136-bit neural watermark).
"""

import os
import torch
import numpy as np
from PIL import Image
from scipy.stats import binomtest

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Cache loaded models to avoid reloading
_model_cache: dict[str, object] = {}


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to (1, 3, H, W) float32 tensor in [0, 1]."""
    img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert (1, 3, H, W) or (3, H, W) float tensor to PIL image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    img_np = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((img_np * 255).round().astype(np.uint8))


def generate_message(n_bits: int = 256, seed: int = 42) -> torch.Tensor:
    """Generate a deterministic binary message tensor.

    Args:
        n_bits: Number of bits in the message.
        seed: Random seed for reproducibility.

    Returns:
        Binary tensor of shape (1, n_bits) with values in {0, 1}.
    """
    rng = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (1, n_bits), generator=rng)


def _binomial_p_value(matching_bits: int, total_bits: int) -> float:
    """Compute one-sided p-value for bit match count.

    Under H0 (no watermark): each bit matches with probability 0.5.
    """
    result = binomtest(matching_bits, total_bits, 0.5, alternative="greater")
    return float(result.pvalue)


def load_videoseal_model(model_name: str = "videoseal") -> object:
    """Load a VideoSeal model, with caching and MPS fallback.

    Args:
        model_name: Model identifier — "videoseal" maps to the videoseal_1.0 card.

    Returns:
        Loaded VideoSeal model on the appropriate device.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    import videoseal
    from pathlib import Path

    # The pip package uses relative paths for card lookup, which breaks
    # when CWD != the videoseal repo. Resolve the card path ourselves.
    if model_name == "videoseal":
        card_path = Path(videoseal.__file__).parent / "cards" / "videoseal_1.0.yaml"
        model = videoseal.load(card_path)
    else:
        model = videoseal.load(model_name)

    try:
        model = model.to(DEVICE)
    except RuntimeError:
        # MPS fallback — some ops may not be supported
        model = model.to("cpu")

    model.eval()
    _model_cache[model_name] = model
    return model


def embed_videoseal(
    image: Image.Image,
    model_name: str = "videoseal",
    message: torch.Tensor | None = None,
    scaling_w: float = 0.3,
) -> tuple[Image.Image, torch.Tensor]:
    """Embed a VideoSeal watermark into an image.

    Args:
        image: Input PIL image.
        model_name: VideoSeal model identifier.
        message: Binary message tensor (1, n_bits). If None, generates default.
        scaling_w: Watermark strength multiplier (0.0–0.5 typical range).

    Returns:
        Tuple of (watermarked PIL image, message tensor used).
    """
    model = load_videoseal_model(model_name)

    if message is None:
        message = generate_message()

    img_tensor = _pil_to_tensor(image).to(next(model.parameters()).device)
    msg = message.to(img_tensor.device)

    # scaling_w lives on the blender, not the model itself
    orig_scaling = model.blender.scaling_w
    model.blender.scaling_w = scaling_w

    with torch.no_grad():
        outputs = model.embed(img_tensor, msg, is_video=False)

    model.blender.scaling_w = orig_scaling

    watermarked_tensor = outputs["imgs_w"]
    return _tensor_to_pil(watermarked_tensor), message


def detect_videoseal(
    image: Image.Image,
    model_name: str = "videoseal",
    expected_message: torch.Tensor | None = None,
    threshold: float = 0.01,
) -> dict:
    """Detect VideoSeal watermark in an image.

    Args:
        image: Image to test.
        model_name: VideoSeal model identifier.
        expected_message: Expected message for bit accuracy. If None, generates default.
        threshold: p-value threshold for detection.

    Returns:
        Dict with: p_value, detected, bit_accuracy, matching_bits.
    """
    model = load_videoseal_model(model_name)

    if expected_message is None:
        expected_message = generate_message()

    img_tensor = _pil_to_tensor(image).to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model.detect(img_tensor)

    # outputs["preds"] shape is (1, 1+n_bits): first element is a
    # watermark-presence indicator, remaining are message bit logits.
    preds = outputs["preds"]
    predicted_bits = (preds[0][1:] > 0).cpu().tolist()
    expected_bits = expected_message[0].tolist()

    n_bits = len(expected_bits)
    matching = sum(
        1 for pred, exp in zip(predicted_bits, expected_bits) if pred == exp
    )

    p_value = _binomial_p_value(matching, n_bits)

    return {
        "p_value": p_value,
        "detected": p_value < threshold,
        "bit_accuracy": matching / n_bits,
        "matching_bits": matching,
    }
