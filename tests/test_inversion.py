"""Tests for DDIM inversion.

These tests require the Stable Diffusion model (~5GB download).
They are marked with @pytest.mark.slow and skipped by default.
Run with: pytest -m slow
"""

import os
import pytest
import torch

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from watermark.inversion import load_pipeline, ddim_invert, ddim_reconstruct

MODEL_DIR = os.environ.get("DW_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", ".models"))
MODEL_ID = "stabilityai/stable-diffusion-2-1"


def has_model():
    """Check if SD model is available (either cached or downloadable)."""
    try:
        # Check if model is cached locally
        cache_path = os.path.join(MODEL_DIR, "models--stabilityai--stable-diffusion-2-1")
        return os.path.exists(cache_path)
    except Exception:
        return False


skip_no_model = pytest.mark.skipif(
    not has_model(),
    reason="SD 2.1 model not cached. Run scripts/generate.py first to download."
)


@pytest.mark.slow
@skip_no_model
class TestDDIMInversion:
    """Test DDIM inversion fidelity. Requires SD 2.1 model."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        return load_pipeline(MODEL_ID, cache_dir=MODEL_DIR)

    def test_inversion_roundtrip(self, pipeline):
        """Invert an image to noise, reconstruct — PSNR should be > 25dB."""
        from PIL import Image
        import numpy as np
        from skimage.metrics import peak_signal_noise_ratio

        # Generate a simple test image via the pipeline
        generator = torch.Generator().manual_seed(42)
        with torch.no_grad():
            result = pipeline(
                "a photograph of a cat",
                num_inference_steps=50,
                generator=generator,
            )
        original = result.images[0]

        # Invert
        inverted_latent = ddim_invert(pipeline, original, num_inference_steps=50)
        assert inverted_latent.shape == (1, 4, 96, 96)  # SD 2.1 produces 768px → 96 latent

        # Reconstruct
        reconstructed = ddim_reconstruct(pipeline, inverted_latent, num_inference_steps=50)

        # Compare
        orig_np = np.array(original).astype(float)
        recon_np = np.array(reconstructed).astype(float)
        psnr = peak_signal_noise_ratio(orig_np, recon_np, data_range=255)
        assert psnr > 25, f"PSNR too low: {psnr:.1f}dB"

    def test_inverted_latent_dtype(self, pipeline):
        """Inverted latent should be float32."""
        from PIL import Image

        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            result = pipeline(
                "a red square",
                num_inference_steps=20,
                generator=generator,
            )
        original = result.images[0]
        inverted = ddim_invert(pipeline, original, num_inference_steps=20)
        assert inverted.dtype == torch.float32
