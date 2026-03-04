"""Tests for Stable Signature watermark — key management and CNN-based detection.

Fast tests use mocked extractors and need no model downloads.
Slow tests (marked @pytest.mark.slow) need the 1.2MB extractor weights.
"""

import os
import torch
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# ── Key generation ──────────────────────────────────────────────────────

class TestKeyGeneration:
    """Test 48-bit watermark key generation."""

    def test_key_length(self):
        from watermark.stable_signature import generate_ss_key
        key = generate_ss_key(n_bits=48, seed=42)
        assert len(key) == 48

    def test_key_is_binary(self):
        from watermark.stable_signature import generate_ss_key
        key = generate_ss_key(n_bits=48, seed=42)
        assert all(b in (0, 1) for b in key)

    def test_key_deterministic(self):
        from watermark.stable_signature import generate_ss_key
        k1 = generate_ss_key(n_bits=48, seed=42)
        k2 = generate_ss_key(n_bits=48, seed=42)
        assert k1 == k2

    def test_key_different_seeds(self):
        from watermark.stable_signature import generate_ss_key
        k1 = generate_ss_key(n_bits=48, seed=42)
        k2 = generate_ss_key(n_bits=48, seed=99)
        assert k1 != k2

    def test_default_key_format(self):
        """The known key baked into sd2_decoder.pth has correct format."""
        from watermark.stable_signature import SS_DEFAULT_KEY
        assert len(SS_DEFAULT_KEY) == 48
        assert all(b in (0, 1) for b in SS_DEFAULT_KEY)


# ── Key serialization ───────────────────────────────────────────────────

class TestKeySerialization:
    """Test key ↔ string roundtrip."""

    def test_key_to_str(self):
        from watermark.stable_signature import key_to_str
        assert key_to_str([1, 0, 1, 1]) == "1011"

    def test_str_to_key(self):
        from watermark.stable_signature import str_to_key
        assert str_to_key("1011") == [1, 0, 1, 1]

    def test_roundtrip(self):
        from watermark.stable_signature import generate_ss_key, key_to_str, str_to_key
        key = generate_ss_key(n_bits=48, seed=42)
        assert str_to_key(key_to_str(key)) == key

    def test_roundtrip_str(self):
        from watermark.stable_signature import key_to_str, str_to_key
        s = "111010110101000001010111010011010100010000100111"
        assert key_to_str(str_to_key(s)) == s


# ── Detection with mock extractor ───────────────────────────────────────

class TestDetectionMocked:
    """Test detection logic using mocked CNN extractor outputs."""

    def _make_mock_extractor(self, logits: torch.Tensor):
        """Create a mock extractor that returns fixed logits."""
        mock = MagicMock()
        mock.return_value = logits
        mock.eval = MagicMock(return_value=mock)
        mock.to = MagicMock(return_value=mock)
        return mock

    def test_perfect_match_detected(self):
        """48/48 matching bits → very low p-value."""
        from watermark.detection_ss import detect_ss

        key = [1, 0, 1, 1, 0] * 9 + [1, 0, 1]  # 48 bits
        # Logits that, when thresholded at 0, match key perfectly
        # bit=1 → positive logit, bit=0 → negative logit
        logits = torch.tensor([[10.0 if b == 1 else -10.0 for b in key]])
        extractor = self._make_mock_extractor(logits)

        image = torch.randn(1, 3, 256, 256)
        result = detect_ss(extractor, image, key)

        assert result["bit_accuracy"] == 1.0
        assert result["matching_bits"] == 48
        assert result["p_value"] < 0.01
        assert result["detected"] is True

    def test_random_bits_not_detected(self):
        """~24/48 matching bits → p-value near 0.5."""
        from watermark.detection_ss import detect_ss

        key = [1] * 48  # All ones
        # Random logits — roughly half will match
        torch.manual_seed(123)
        logits = torch.randn(1, 48)
        extractor = self._make_mock_extractor(logits)

        image = torch.randn(1, 3, 256, 256)
        result = detect_ss(extractor, image, key)

        # Should NOT detect — random bits match ~50%
        assert result["p_value"] > 0.01
        assert result["detected"] is False

    def test_zero_match_not_detected(self):
        """0/48 matching bits → p-value ≈ 1.0."""
        from watermark.detection_ss import detect_ss

        key = [1] * 48  # All ones
        logits = torch.tensor([[-10.0] * 48])  # All predict 0 → 0/48 match
        extractor = self._make_mock_extractor(logits)

        image = torch.randn(1, 3, 256, 256)
        result = detect_ss(extractor, image, key)

        assert result["matching_bits"] == 0
        assert result["p_value"] > 0.99
        assert result["detected"] is False


# ── Binomial math spot-checks ────────────────────────────────────────────

class TestBinomialMath:
    """Verify the binomial test gives expected p-values."""

    def test_48_of_48(self):
        """All bits match → p ≈ 3.55e-15."""
        from watermark.detection_ss import _binomial_p_value
        p = _binomial_p_value(48, 48)
        assert p < 1e-12

    def test_24_of_48(self):
        """Half bits match → p ≈ 0.5+."""
        from watermark.detection_ss import _binomial_p_value
        p = _binomial_p_value(24, 48)
        assert 0.4 < p < 0.7  # Not significant

    def test_36_of_48(self):
        """75% match → significant."""
        from watermark.detection_ss import _binomial_p_value
        p = _binomial_p_value(36, 48)
        assert p < 0.001

    def test_0_of_48(self):
        """No bits match → p = 1.0."""
        from watermark.detection_ss import _binomial_p_value
        p = _binomial_p_value(0, 48)
        assert p > 0.99


# ── Slow tests (need extractor download) ────────────────────────────────

@pytest.mark.slow
class TestExtractorReal:
    """Tests that require downloading the 1.2MB extractor model."""

    def test_extractor_loads(self):
        """Extractor loads and produces (1, 48) output."""
        from watermark.stable_signature import download_extractor
        from watermark.detection_ss import load_extractor

        model_dir = os.environ.get(
            "DW_MODEL_DIR",
            os.path.join(os.path.dirname(__file__), "..", ".models"),
        )
        path = download_extractor(model_dir)
        extractor = load_extractor(path)

        # Random image → extractor should produce 48 logits
        device = next(extractor.parameters()).device
        dummy = torch.randn(1, 3, 256, 256, device=device)
        with torch.no_grad():
            out = extractor(dummy)
        assert out.shape == (1, 48)

    def test_random_image_not_detected(self):
        """A random noise image should not match any key."""
        from watermark.stable_signature import download_extractor, generate_ss_key
        from watermark.detection_ss import load_extractor, detect_ss

        model_dir = os.environ.get(
            "DW_MODEL_DIR",
            os.path.join(os.path.dirname(__file__), "..", ".models"),
        )
        path = download_extractor(model_dir)
        extractor = load_extractor(path)

        key = generate_ss_key(n_bits=48, seed=42)
        dummy = torch.randn(1, 3, 256, 256)
        result = detect_ss(extractor, dummy, key)

        assert result["detected"] is False
        assert result["p_value"] > 0.01
