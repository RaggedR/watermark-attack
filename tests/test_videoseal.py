"""Tests for VideoSeal watermark wrapper.

Fast tests cover tensor conversion, message generation, and binomial math.
Slow tests require the VideoSeal model download (~300MB).
"""

import torch
import numpy as np
import pytest
from PIL import Image

from watermark.videoseal_wm import (
    _pil_to_tensor,
    _tensor_to_pil,
    generate_message,
    _binomial_p_value,
)


class TestTensorConversions:
    """PIL ↔ tensor round-trips."""

    def test_pil_to_tensor_shape(self):
        img = Image.new("RGB", (64, 48), color=(128, 64, 32))
        tensor = _pil_to_tensor(img)
        assert tensor.shape == (1, 3, 48, 64)

    def test_pil_to_tensor_range(self):
        img = Image.new("RGB", (32, 32), color=(255, 0, 128))
        tensor = _pil_to_tensor(img)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_tensor_to_pil_shape(self):
        tensor = torch.rand(1, 3, 48, 64)
        img = _tensor_to_pil(tensor)
        assert img.size == (64, 48)
        assert img.mode == "RGB"

    def test_tensor_to_pil_3d(self):
        """Accepts (3, H, W) without batch dimension."""
        tensor = torch.rand(3, 32, 32)
        img = _tensor_to_pil(tensor)
        assert img.size == (32, 32)

    def test_roundtrip(self):
        """PIL → tensor → PIL preserves pixel values (within quantization)."""
        img = Image.new("RGB", (32, 32), color=(100, 150, 200))
        tensor = _pil_to_tensor(img)
        img2 = _tensor_to_pil(tensor)
        np.testing.assert_array_equal(np.array(img), np.array(img2))

    def test_tensor_clamp(self):
        """Out-of-range tensor values get clamped to [0, 1]."""
        # Shape (3, 2, 2): 3 channels, 2x2 image
        tensor = torch.tensor([
            [[-0.5, 1.5], [0.3, 0.7]],
            [[ 0.5, 0.5], [0.5, 0.5]],
            [[ 0.5, 0.5], [0.5, 0.5]],
        ])
        img = _tensor_to_pil(tensor)
        pixels = np.array(img)
        assert pixels.min() >= 0
        assert pixels.max() <= 255


class TestMessageGeneration:
    """Deterministic binary message generation."""

    def test_shape(self):
        msg = generate_message(n_bits=256)
        assert msg.shape == (1, 256)

    def test_binary(self):
        msg = generate_message(n_bits=256)
        assert set(msg[0].tolist()).issubset({0, 1})

    def test_deterministic(self):
        msg1 = generate_message(n_bits=256, seed=42)
        msg2 = generate_message(n_bits=256, seed=42)
        assert torch.equal(msg1, msg2)

    def test_different_seeds(self):
        msg1 = generate_message(n_bits=256, seed=42)
        msg2 = generate_message(n_bits=256, seed=99)
        assert not torch.equal(msg1, msg2)

    def test_custom_bits(self):
        msg = generate_message(n_bits=128)
        assert msg.shape == (1, 128)

    def test_roughly_balanced(self):
        """256-bit message should have roughly equal 0s and 1s."""
        msg = generate_message(n_bits=256, seed=42)
        ones = msg.sum().item()
        assert 80 < ones < 176  # very loose — just sanity check


class TestBinomialMath:
    """Binomial p-value calculations."""

    def test_perfect_match(self):
        """256/256 matching bits → extremely small p-value."""
        p = _binomial_p_value(256, 256)
        assert p < 1e-50

    def test_random_match(self):
        """128/256 matching bits (chance level) → p ≈ 0.5."""
        p = _binomial_p_value(128, 256)
        assert 0.4 < p < 0.6

    def test_strong_match(self):
        """200/256 matching bits → very significant."""
        p = _binomial_p_value(200, 256)
        assert p < 1e-10

    def test_below_chance(self):
        """50/256 matching bits → p close to 1."""
        p = _binomial_p_value(50, 256)
        assert p > 0.99


class TestVideoSealRealModel:
    """Integration tests requiring the actual VideoSeal model."""

    @pytest.mark.slow
    def test_embed_detect_roundtrip(self):
        """Embed watermark → detect it in watermarked image."""
        from watermark.videoseal_wm import embed_videoseal, detect_videoseal

        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        msg = generate_message(n_bits=256, seed=42)

        wm_img, used_msg = embed_videoseal(img, message=msg, scaling_w=0.3)
        result = detect_videoseal(wm_img, expected_message=msg)

        assert result["detected"], f"Watermark not detected, p={result['p_value']}"
        assert result["bit_accuracy"] > 0.8

    @pytest.mark.slow
    def test_unwatermarked_not_detected(self):
        """Clean image should not be detected as watermarked."""
        from watermark.videoseal_wm import detect_videoseal

        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        msg = generate_message(n_bits=256, seed=42)

        result = detect_videoseal(img, expected_message=msg)
        assert not result["detected"], f"False positive, p={result['p_value']}"

    @pytest.mark.slow
    def test_embed_preserves_size(self):
        """Watermarked image has same dimensions as input."""
        from watermark.videoseal_wm import embed_videoseal

        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        wm_img, _ = embed_videoseal(img, scaling_w=0.2)
        assert wm_img.size == img.size
