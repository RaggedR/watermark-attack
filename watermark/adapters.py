"""Unified watermark adapter interface.

Abstracts over Tree-Ring, Stable Signature, and VideoSeal so that
scripts/embed.py and scripts/sweep.py can treat all methods identically.

Each adapter wraps:
    embed(image, strength) → watermarked PIL image
    extract(image)         → dict with p_value, detected, ...
    metadata()             → dict for serialization to metadata.json
"""

import os
from abc import ABC, abstractmethod
from PIL import Image


class WatermarkAdapter(ABC):
    """Common interface for all watermark methods."""

    name: str           # "tree-ring", "stable-signature", "videoseal"
    capacity_bits: int  # 0 for Tree-Ring, 48 for SS, 256 for VideoSeal
    needs_pipeline: bool  # Whether embed() needs the SD pipeline

    @abstractmethod
    def embed(self, image: Image.Image, strength: float = 1.0) -> Image.Image:
        """Embed watermark into an existing image.

        Args:
            image: Source PIL image.
            strength: Embedding strength (interpretation varies by method).

        Returns:
            Watermarked PIL image.
        """

    @abstractmethod
    def extract(self, image: Image.Image) -> dict:
        """Extract/detect watermark from an image.

        Returns:
            Dict always containing at least: p_value (float), detected (bool).
            May also include: bit_accuracy, matching_bits, l1_distance, etc.
        """

    @abstractmethod
    def metadata(self) -> dict:
        """Return serializable dict of watermark configuration for metadata.json."""


class TreeRingAdapter(WatermarkAdapter):
    """Tree-Ring: Fourier-space noise injection.

    Embed: DDIM-invert image → inject key in Fourier domain → DDIM forward.
    Detect: DDIM-invert image → check Fourier ring positions → t-test.

    Both operations require the full SD pipeline (~30s per image).
    """

    name = "tree-ring"
    capacity_bits = 0
    needs_pipeline = True

    def __init__(self, pipe, seed: int = 42, radius: float = 10.0, ring_width: float = 3.0):
        from watermark.tree_ring import generate_key, create_ring_mask

        self.pipe = pipe
        self.seed = seed
        self.radius = radius
        self.ring_width = ring_width

        latent_size = pipe.unet.config.sample_size  # 96 for SD 2.1
        self.latent_size = latent_size
        self.key = generate_key(latent_size, latent_size, radius, ring_width, seed=seed)
        self.mask = create_ring_mask(latent_size, latent_size, radius, ring_width)

    def embed(self, image: Image.Image, strength: float = 1.0) -> Image.Image:
        """DDIM-invert → inject Fourier key → DDIM-regenerate."""
        import torch
        from watermark.tree_ring import embed_watermark
        from watermark.inversion import ddim_invert, ddim_reconstruct

        # Invert to recover noise estimate x_T
        inverted = ddim_invert(self.pipe, image, num_inference_steps=50)

        # Inject watermark key in Fourier domain
        watermarked_noise = embed_watermark(inverted, self.key, self.mask)

        # Regenerate image from watermarked noise
        return ddim_reconstruct(self.pipe, watermarked_noise, num_inference_steps=50)

    def extract(self, image: Image.Image) -> dict:
        from watermark.detection import detect_watermark
        return detect_watermark(self.pipe, image, self.key, self.mask)

    def metadata(self) -> dict:
        return {
            "watermark_type": self.name,
            "seed": self.seed,
            "radius": self.radius,
            "ring_width": self.ring_width,
            "latent_size": self.latent_size,
        }


class StableSignatureAdapter(WatermarkAdapter):
    """Stable Signature: fine-tuned VAE decoder embeds a fixed 48-bit key.

    Embed: VAE-encode image → decode with patched decoder → watermarked pixels.
    Detect: CNN extractor → threshold logits → binomial test.

    Embed needs the SD pipeline (for its VAE); detect only needs the 1.2MB extractor.
    """

    name = "stable-signature"
    capacity_bits = 48
    needs_pipeline = True

    def __init__(self, pipe, model_dir: str | None = None, seed: int = 42):
        import torch
        from watermark.stable_signature import (
            generate_ss_key, download_extractor, download_decoder,
            patch_pipeline_decoder, SS_DEFAULT_KEY,
        )
        from watermark.detection_ss import load_extractor

        self.pipe = pipe
        self.seed = seed

        if model_dir is None:
            model_dir = os.environ.get(
                "DW_MODEL_DIR",
                os.path.join(os.path.dirname(__file__), "..", ".models"),
            )
        self.model_dir = model_dir

        # Download and load extractor
        ext_path = download_extractor(model_dir)
        self.extractor = load_extractor(ext_path)

        # Patch pipeline decoder
        decoder_path = download_decoder(model_dir)
        patch_pipeline_decoder(pipe, decoder_path)

        # Use the default key baked into the decoder
        self.key = SS_DEFAULT_KEY

    def embed(self, image: Image.Image, strength: float = 1.0) -> Image.Image:
        """VAE encode → decode with patched decoder.

        The strength parameter is ignored — the watermark strength is
        baked into the fine-tuned decoder weights.
        """
        import torch
        import numpy as np
        from watermark.inversion import encode_image, DTYPE

        latent = encode_image(self.pipe, image)

        # Decode with patched VAE (watermark comes from decoder weights)
        with torch.no_grad():
            decoded = self.pipe.vae.decode(
                latent / self.pipe.vae.config.scaling_factor
            ).sample

        # Convert to PIL
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = decoded.cpu().permute(0, 2, 3, 1).numpy()[0]
        decoded = (decoded * 255).round().astype(np.uint8)
        return Image.fromarray(decoded)

    def extract(self, image: Image.Image) -> dict:
        from watermark.detection_ss import detect_ss
        return detect_ss(self.extractor, image, self.key)

    def metadata(self) -> dict:
        from watermark.stable_signature import key_to_str
        return {
            "watermark_type": self.name,
            "seed": self.seed,
            "n_bits": self.capacity_bits,
            "key": key_to_str(self.key),
        }


class VideoSealAdapter(WatermarkAdapter):
    """VideoSeal: neural encoder/decoder for 256-bit embedding.

    Embed: image + message → neural encoder → watermarked image (fast, no SD).
    Detect: image → neural decoder → 256 logits → binomial test (fast).

    Closest open-source proxy for SynthID.
    """

    name = "videoseal"
    capacity_bits = 256
    needs_pipeline = False

    def __init__(self, model_name: str = "videoseal", seed: int = 42):
        self.model_name = model_name
        self.seed = seed
        # Lazy-load model on first use
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from watermark.videoseal_wm import load_videoseal_model
            self._model = load_videoseal_model(self.model_name)

    def embed(self, image: Image.Image, strength: float = 0.3) -> Image.Image:
        """Neural watermark embedding. Strength maps to scaling_w."""
        from watermark.videoseal_wm import embed_videoseal, generate_message

        msg = generate_message(n_bits=self.capacity_bits, seed=self.seed)
        wm_image, _ = embed_videoseal(
            image, model_name=self.model_name, message=msg, scaling_w=strength,
        )
        return wm_image

    def extract(self, image: Image.Image) -> dict:
        from watermark.videoseal_wm import detect_videoseal, generate_message

        msg = generate_message(n_bits=self.capacity_bits, seed=self.seed)
        return detect_videoseal(
            image, model_name=self.model_name, expected_message=msg,
        )

    def metadata(self) -> dict:
        from watermark.videoseal_wm import generate_message
        msg = generate_message(n_bits=self.capacity_bits, seed=self.seed)
        return {
            "watermark_type": self.name,
            "seed": self.seed,
            "n_bits": self.capacity_bits,
            "model_name": self.model_name,
        }


def create_adapter(
    method: str,
    pipe=None,
    model_dir: str | None = None,
    seed: int = 42,
) -> WatermarkAdapter:
    """Factory function to create a watermark adapter.

    Args:
        method: One of "tree-ring", "stable-signature", "videoseal".
        pipe: StableDiffusionPipeline (required for tree-ring and stable-signature).
        model_dir: Model cache directory (for stable-signature weights).
        seed: Random seed for key/message generation.

    Returns:
        Configured WatermarkAdapter instance.

    Raises:
        ValueError: If method is unknown or pipeline is missing when required.
    """
    if method == "tree-ring":
        if pipe is None:
            raise ValueError("Tree-Ring adapter requires a StableDiffusionPipeline (pipe)")
        return TreeRingAdapter(pipe, seed=seed)

    elif method == "stable-signature":
        if pipe is None:
            raise ValueError("Stable Signature adapter requires a StableDiffusionPipeline (pipe)")
        return StableSignatureAdapter(pipe, model_dir=model_dir, seed=seed)

    elif method == "videoseal":
        return VideoSealAdapter(seed=seed)

    else:
        raise ValueError(
            f"Unknown watermark method: {method!r}. "
            f"Choose from: tree-ring, stable-signature, videoseal"
        )
