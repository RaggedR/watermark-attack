"""Tests for the watermark adapter interface.

All fast — uses mocked models, no downloads needed.
"""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from watermark.adapters import (
    WatermarkAdapter,
    TreeRingAdapter,
    StableSignatureAdapter,
    VideoSealAdapter,
    create_adapter,
)


class TestAdapterProperties:
    """Verify adapter class attributes match the spec."""

    def test_tree_ring_properties(self):
        assert TreeRingAdapter.name == "tree-ring"
        assert TreeRingAdapter.capacity_bits == 0
        assert TreeRingAdapter.needs_pipeline is True

    def test_stable_signature_properties(self):
        assert StableSignatureAdapter.name == "stable-signature"
        assert StableSignatureAdapter.capacity_bits == 48
        assert StableSignatureAdapter.needs_pipeline is True

    def test_videoseal_properties(self):
        assert VideoSealAdapter.name == "videoseal"
        assert VideoSealAdapter.capacity_bits == 256
        assert VideoSealAdapter.needs_pipeline is False

    def test_all_are_watermark_adapters(self):
        assert issubclass(TreeRingAdapter, WatermarkAdapter)
        assert issubclass(StableSignatureAdapter, WatermarkAdapter)
        assert issubclass(VideoSealAdapter, WatermarkAdapter)


class TestFactory:
    """Test create_adapter factory function."""

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown watermark method"):
            create_adapter("nonexistent")

    def test_tree_ring_requires_pipe(self):
        with pytest.raises(ValueError, match="requires a StableDiffusionPipeline"):
            create_adapter("tree-ring")

    def test_stable_signature_requires_pipe(self):
        with pytest.raises(ValueError, match="requires a StableDiffusionPipeline"):
            create_adapter("stable-signature")

    def test_videoseal_no_pipe_needed(self):
        """VideoSeal adapter can be created without a pipeline."""
        adapter = create_adapter("videoseal")
        assert isinstance(adapter, VideoSealAdapter)
        assert adapter.name == "videoseal"

    def test_videoseal_seed(self):
        adapter = create_adapter("videoseal", seed=99)
        assert adapter.seed == 99


class TestVideoSealAdapter:
    """Test VideoSealAdapter behavior (no real model needed)."""

    def test_metadata_fields(self):
        adapter = VideoSealAdapter(seed=42)
        meta = adapter.metadata()
        assert meta["watermark_type"] == "videoseal"
        assert meta["seed"] == 42
        assert meta["n_bits"] == 256
        assert "model_name" in meta

    def test_lazy_model_loading(self):
        """Model should not be loaded at construction time."""
        adapter = VideoSealAdapter()
        assert adapter._model is None


class TestTreeRingAdapter:
    """Test TreeRingAdapter with mocked pipeline."""

    @pytest.fixture
    def mock_pipe(self):
        pipe = MagicMock()
        pipe.unet.config.sample_size = 96
        pipe.vae_scale_factor = 8
        return pipe

    def test_init_creates_key_and_mask(self, mock_pipe):
        adapter = TreeRingAdapter(mock_pipe, seed=42, radius=10.0, ring_width=3.0)
        assert adapter.key.shape == (96, 96)
        assert adapter.mask.shape == (96, 96)
        assert adapter.latent_size == 96

    def test_metadata_fields(self, mock_pipe):
        adapter = TreeRingAdapter(mock_pipe, seed=42)
        meta = adapter.metadata()
        assert meta["watermark_type"] == "tree-ring"
        assert meta["seed"] == 42
        assert meta["radius"] == 10.0
        assert meta["ring_width"] == 3.0
        assert meta["latent_size"] == 96

    def test_factory_creates_tree_ring(self, mock_pipe):
        adapter = create_adapter("tree-ring", pipe=mock_pipe, seed=42)
        assert isinstance(adapter, TreeRingAdapter)


class TestStableSignatureAdapter:
    """Test StableSignatureAdapter with mocked pipeline and models."""

    @pytest.fixture
    def mock_pipe(self):
        pipe = MagicMock()
        pipe.unet.config.sample_size = 96
        pipe.vae_scale_factor = 8
        pipe.device = "cpu"
        pipe.vae.dtype = "float32"
        return pipe

    @patch("watermark.stable_signature.download_decoder")
    @patch("watermark.stable_signature.patch_pipeline_decoder")
    @patch("watermark.detection_ss.load_extractor")
    @patch("watermark.stable_signature.download_extractor")
    def test_init_downloads_models(
        self, mock_dl_ext, mock_load_ext, mock_patch, mock_dl_dec, mock_pipe
    ):
        mock_dl_ext.return_value = "/fake/extractor.pt"
        mock_dl_dec.return_value = "/fake/decoder.pth"
        mock_load_ext.return_value = MagicMock()

        adapter = StableSignatureAdapter(mock_pipe, model_dir="/fake/models")
        mock_dl_ext.assert_called_once_with("/fake/models")
        mock_dl_dec.assert_called_once_with("/fake/models")
        mock_patch.assert_called_once()

    @patch("watermark.stable_signature.download_decoder")
    @patch("watermark.stable_signature.patch_pipeline_decoder")
    @patch("watermark.detection_ss.load_extractor")
    @patch("watermark.stable_signature.download_extractor")
    def test_metadata_fields(
        self, mock_dl_ext, mock_load_ext, mock_patch, mock_dl_dec, mock_pipe
    ):
        mock_dl_ext.return_value = "/fake/extractor.pt"
        mock_dl_dec.return_value = "/fake/decoder.pth"
        mock_load_ext.return_value = MagicMock()

        adapter = StableSignatureAdapter(mock_pipe, model_dir="/fake/models")
        meta = adapter.metadata()
        assert meta["watermark_type"] == "stable-signature"
        assert meta["n_bits"] == 48
        assert "key" in meta
