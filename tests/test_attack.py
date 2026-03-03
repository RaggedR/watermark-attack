"""Tests for the regeneration attack pipeline.

These tests require the Stable Diffusion model (~5GB download).
They are marked with @pytest.mark.slow and skipped by default.
Run with: pytest -m slow
"""

import os
import pytest
import torch
import numpy as np

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from watermark.tree_ring import generate_key, create_ring_mask, embed_watermark

MODEL_DIR = os.environ.get("DW_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", ".models"))
MODEL_ID = "stabilityai/stable-diffusion-2-1"


def has_model():
    cache_path = os.path.join(MODEL_DIR, "models--stabilityai--stable-diffusion-2-1")
    return os.path.exists(cache_path)


skip_no_model = pytest.mark.skipif(
    not has_model(),
    reason="SD 2.1 model not cached. Run scripts/generate.py first to download."
)


@pytest.mark.slow
@skip_no_model
class TestRegenerationAttack:
    """End-to-end attack test: watermark → attack → detect."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        from watermark.inversion import load_pipeline
        return load_pipeline(MODEL_ID, cache_dir=MODEL_DIR)

    def test_attack_breaks_watermark(self, pipeline):
        """
        Generate watermarked image → attack at strength=0.5 → detect.
        The watermark should no longer be detectable (p_value > 0.05).
        """
        from watermark.detection import detect_watermark as full_detect
        from attack.regeneration import regeneration_attack

        # Generate watermarked image
        key = generate_key(96, 96, radius=10, ring_width=3, seed=42)
        mask = create_ring_mask(96, 96, radius=10, ring_width=3)

        # Create watermarked noise and generate image
        noise = torch.randn(1, 4, 96, 96)
        watermarked_noise = embed_watermark(noise.clone(), key, mask)

        generator = torch.Generator().manual_seed(42)
        with torch.no_grad():
            result = pipeline(
                "a landscape painting",
                latents=watermarked_noise.to(pipeline.device, pipeline.unet.dtype),
                num_inference_steps=50,
                generator=generator,
            )
        watermarked_image = result.images[0]

        # Attack
        attacked_image, metrics = regeneration_attack(
            pipeline,
            watermarked_image,
            strength=0.5,
            guidance_scale=7.5,
            num_inference_steps=50,
        )

        # Verify quality is reasonable
        assert metrics["ssim"] > 0.3, f"Quality too low: SSIM={metrics['ssim']:.3f}"

        # Detect watermark in attacked image
        result = full_detect(pipeline, attacked_image, key, mask, num_inference_steps=50)
        assert result["p_value"] > 0.05, (
            f"Watermark survived attack: p_value={result['p_value']:.4f}"
        )

    def test_weak_attack_preserves_watermark(self, pipeline):
        """
        Attack at very low strength (0.05) should preserve the watermark.
        """
        from watermark.detection import detect_watermark as full_detect
        from attack.regeneration import regeneration_attack

        key = generate_key(96, 96, radius=10, ring_width=3, seed=42)
        mask = create_ring_mask(96, 96, radius=10, ring_width=3)

        noise = torch.randn(1, 4, 96, 96)
        watermarked_noise = embed_watermark(noise.clone(), key, mask)

        generator = torch.Generator().manual_seed(42)
        with torch.no_grad():
            result = pipeline(
                "a forest scene",
                latents=watermarked_noise.to(pipeline.device, pipeline.unet.dtype),
                num_inference_steps=50,
                generator=generator,
            )
        watermarked_image = result.images[0]

        # Very weak attack
        attacked_image, metrics = regeneration_attack(
            pipeline,
            watermarked_image,
            strength=0.05,
            guidance_scale=7.5,
            num_inference_steps=50,
        )

        # Quality should be very high
        assert metrics["ssim"] > 0.85, f"Weak attack changed too much: SSIM={metrics['ssim']:.3f}"

        # Watermark should still be detectable
        result = full_detect(pipeline, attacked_image, key, mask, num_inference_steps=50)
        assert result["p_value"] < 0.05, (
            f"Watermark lost with weak attack: p_value={result['p_value']:.4f}"
        )

    def test_attack_returns_quality_metrics(self, pipeline):
        """Attack should return SSIM and PSNR."""
        from attack.regeneration import regeneration_attack
        from PIL import Image

        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            result = pipeline("a blue sky", num_inference_steps=20, generator=generator)
        image = result.images[0]

        attacked, metrics = regeneration_attack(
            pipeline, image, strength=0.3, num_inference_steps=20
        )

        assert "ssim" in metrics
        assert "psnr" in metrics
        assert isinstance(attacked, Image.Image)
        assert 0 <= metrics["ssim"] <= 1
        assert metrics["psnr"] > 0
