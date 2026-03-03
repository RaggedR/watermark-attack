"""Tests for Tree-Ring watermark key generation and Fourier embedding."""

import os
import torch
import pytest

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from watermark.tree_ring import generate_key, create_ring_mask, embed_watermark


class TestRingMask:
    """Test Fourier-space ring mask generation."""

    def test_mask_shape(self):
        mask = create_ring_mask(64, 64, radius=10, ring_width=3)
        assert mask.shape == (64, 64)

    def test_mask_is_binary(self):
        mask = create_ring_mask(64, 64, radius=10, ring_width=3)
        assert torch.all((mask == 0) | (mask == 1))

    def test_mask_is_ring_shaped(self):
        """Mask should only have nonzero values in a ring region."""
        h, w = 64, 64
        mask = create_ring_mask(h, w, radius=10, ring_width=3)
        cy, cx = h // 2, w // 2
        ys, xs = torch.where(mask > 0)
        dists = torch.sqrt((ys.float() - cy) ** 2 + (xs.float() - cx) ** 2)
        # All nonzero positions should be within [radius - ring_width, radius + ring_width]
        assert torch.all(dists >= 10 - 3)
        assert torch.all(dists <= 10 + 3)

    def test_mask_nonzero_count(self):
        """Ring mask should have a reasonable number of nonzero elements."""
        mask = create_ring_mask(64, 64, radius=10, ring_width=3)
        count = mask.sum().item()
        # A ring of radius 10 ± 3 on a 64x64 grid should have roughly 100-400 pixels
        assert 50 < count < 500


class TestKeyGeneration:
    """Test watermark key generation."""

    def test_key_deterministic(self):
        """Same seed should produce same key."""
        k1 = generate_key(64, 64, radius=10, ring_width=3, seed=42)
        k2 = generate_key(64, 64, radius=10, ring_width=3, seed=42)
        assert torch.allclose(k1, k2)

    def test_key_different_seeds(self):
        """Different seeds should produce different keys."""
        k1 = generate_key(64, 64, radius=10, ring_width=3, seed=42)
        k2 = generate_key(64, 64, radius=10, ring_width=3, seed=99)
        assert not torch.allclose(k1, k2)

    def test_key_shape(self):
        key = generate_key(64, 64, radius=10, ring_width=3, seed=42)
        assert key.shape == (64, 64)

    def test_key_only_nonzero_in_ring(self):
        """Key values should be zero outside the ring mask."""
        key = generate_key(64, 64, radius=10, ring_width=3, seed=42)
        mask = create_ring_mask(64, 64, radius=10, ring_width=3)
        outside = key[mask == 0]
        assert torch.all(outside == 0)


class TestEmbedWatermark:
    """Test watermark embedding into noise tensors."""

    def test_embed_modifies_noise(self):
        """Embedding should change the noise tensor."""
        noise = torch.randn(1, 4, 64, 64)
        key = generate_key(64, 64, radius=10, ring_width=3, seed=42)
        mask = create_ring_mask(64, 64, radius=10, ring_width=3)
        watermarked = embed_watermark(noise.clone(), key, mask)
        assert not torch.allclose(noise, watermarked)

    def test_embed_preserves_shape(self):
        noise = torch.randn(1, 4, 64, 64)
        key = generate_key(64, 64, radius=10, ring_width=3, seed=42)
        mask = create_ring_mask(64, 64, radius=10, ring_width=3)
        watermarked = embed_watermark(noise.clone(), key, mask)
        assert watermarked.shape == noise.shape

    def test_roundtrip_fourier(self):
        """Embed key into noise, then extract from Fourier domain — should recover key."""
        noise = torch.randn(1, 4, 64, 64)
        key = generate_key(64, 64, radius=10, ring_width=3, seed=42)
        mask = create_ring_mask(64, 64, radius=10, ring_width=3)
        watermarked = embed_watermark(noise.clone(), key, mask)

        # Extract: FFT each channel, check ring positions match key
        for c in range(4):
            freq = torch.fft.fftshift(torch.fft.fft2(watermarked[0, c]))
            extracted = freq.real[mask == 1]
            expected = key[mask == 1]
            # The real part at ring positions should match the key
            assert torch.allclose(extracted, expected, atol=1e-4), (
                f"Channel {c}: max diff = {(extracted - expected).abs().max().item()}"
            )


class TestDetectionIntegration:
    """Integration test: embed watermark → detect in Fourier space (no model needed)."""

    def test_watermarked_noise_detected(self):
        """Watermark embedded directly in noise should be perfectly detectable."""
        from watermark.detection import detect_in_noise

        noise = torch.randn(1, 4, 64, 64)
        key = generate_key(64, 64, radius=10, ring_width=3, seed=42)
        mask = create_ring_mask(64, 64, radius=10, ring_width=3)
        watermarked = embed_watermark(noise.clone(), key, mask)

        result = detect_in_noise(watermarked, key, mask)
        assert result["p_value"] < 0.01, f"p_value too high: {result['p_value']}"
        assert result["detected"] is True

    def test_unwatermarked_noise_not_detected(self):
        """Plain random noise should not trigger detection."""
        from watermark.detection import detect_in_noise

        noise = torch.randn(1, 4, 64, 64)
        key = generate_key(64, 64, radius=10, ring_width=3, seed=42)
        mask = create_ring_mask(64, 64, radius=10, ring_width=3)

        result = detect_in_noise(noise, key, mask)
        assert result["p_value"] > 0.01, f"False positive: p_value = {result['p_value']}"
        assert result["detected"] is False
