"""Stable Signature watermark — key management, model downloads, decoder patching.

Stable Signature (Meta, ICCV 2023) fine-tunes the VAE decoder to embed a binary
watermark. Detection uses a separate CNN extractor (HiDDeN architecture).

Architecture:
    Generation: standard noise → patched VAE decoder → watermarked pixels
    Detection:  image → CNN extractor → 48 logits → threshold → binomial test

Key difference from Tree-Ring: the watermark is in the decoder weights, not the noise.
"""

import os
import re
import torch
import urllib.request
from pathlib import Path

# The known 48-bit key baked into sd2_decoder.pth
SS_DEFAULT_KEY = [
    1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1,
    0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0,
    0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1,
]

SS_MODEL_BASE_URL = "https://dl.fbaipublicfiles.com/ssl_watermarking"


# ── Key management ──────────────────────────────────────────────────────

def generate_ss_key(n_bits: int = 48, seed: int = 42) -> list[int]:
    """Generate a deterministic binary watermark key.

    Args:
        n_bits: Number of bits in the key.
        seed: Random seed for reproducibility.

    Returns:
        List of n_bits integers, each 0 or 1.
    """
    rng = torch.Generator().manual_seed(seed)
    return [int(torch.randint(0, 2, (1,), generator=rng).item()) for _ in range(n_bits)]


def key_to_str(key: list[int]) -> str:
    """Convert key list to bit string: [1,0,1,...] → '101...'."""
    return "".join(str(b) for b in key)


def str_to_key(s: str) -> list[int]:
    """Convert bit string to key list: '101...' → [1,0,1,...]."""
    return [int(c) for c in s]


# ── Model downloads ─────────────────────────────────────────────────────

def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def download_extractor(model_dir: str | None = None) -> str:
    """Download the HiDDeN extractor (1.2MB TorchScript model).

    Args:
        model_dir: Directory to store models. Defaults to .models/stable_signature/.

    Returns:
        Path to the downloaded extractor file.
    """
    if model_dir is None:
        model_dir = os.environ.get(
            "DW_MODEL_DIR",
            os.path.join(os.path.dirname(__file__), "..", ".models"),
        )
    ss_dir = _ensure_dir(os.path.join(model_dir, "stable_signature"))
    path = os.path.join(ss_dir, "dec_48b_whit.torchscript.pt")

    if not os.path.exists(path):
        url = f"{SS_MODEL_BASE_URL}/dec_48b_whit.torchscript.pt"
        print(f"Downloading extractor ({url})...")
        urllib.request.urlretrieve(url, path)
        print(f"  Saved to {path} ({os.path.getsize(path) / 1024:.0f} KB)")

    return path


def download_decoder(model_dir: str | None = None) -> str:
    """Download the fine-tuned SD2 VAE decoder (198MB).

    Args:
        model_dir: Directory to store models. Defaults to .models/stable_signature/.

    Returns:
        Path to the downloaded decoder file.
    """
    if model_dir is None:
        model_dir = os.environ.get(
            "DW_MODEL_DIR",
            os.path.join(os.path.dirname(__file__), "..", ".models"),
        )
    ss_dir = _ensure_dir(os.path.join(model_dir, "stable_signature"))
    path = os.path.join(ss_dir, "sd2_decoder.pth")

    if not os.path.exists(path):
        url = f"{SS_MODEL_BASE_URL}/sd2_decoder.pth"
        print(f"Downloading decoder ({url})...")
        print("  This is ~198MB, may take a few minutes...")
        urllib.request.urlretrieve(url, path)
        print(f"  Saved to {path} ({os.path.getsize(path) / 1024 / 1024:.0f} MB)")

    return path


# ── LDM → Diffusers key remapping ──────────────────────────────────────

def _remap_ldm_decoder_keys(state_dict: dict) -> dict:
    """Remap LDM-format VAE decoder keys to diffusers format.

    LDM uses:  decoder.up.{block_id}.block.{layer_id}.{conv/norm}
    Diffusers: decoder.up_blocks.{block_id}.resnets.{layer_id}.{conv/norm}

    Only remaps decoder keys; encoder keys are left unchanged.
    """
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k

        # decoder.up.{i}.block.{j} → decoder.up_blocks.{i}.resnets.{j}
        new_key = re.sub(
            r"^decoder\.up\.(\d+)\.block\.(\d+)\.",
            r"decoder.up_blocks.\1.resnets.\2.",
            new_key,
        )

        # decoder.up.{i}.upsample → decoder.up_blocks.{i}.upsamplers.0
        new_key = re.sub(
            r"^decoder\.up\.(\d+)\.upsample\.",
            r"decoder.up_blocks.\1.upsamplers.0.",
            new_key,
        )

        # decoder.mid.block_{1,2} → decoder.mid_block.resnets.{0,1}
        new_key = re.sub(
            r"^decoder\.mid\.block_1\.",
            r"decoder.mid_block.resnets.0.",
            new_key,
        )
        new_key = re.sub(
            r"^decoder\.mid\.block_2\.",
            r"decoder.mid_block.resnets.1.",
            new_key,
        )

        # decoder.mid.attn_1 → decoder.mid_block.attentions.0
        new_key = re.sub(
            r"^decoder\.mid\.attn_1\.",
            r"decoder.mid_block.attentions.0.",
            new_key,
        )

        # Attention key remapping within mid block
        # q/k/v/proj_out stay the same name in diffusers
        # but norm → group_norm
        new_key = new_key.replace(
            "mid_block.attentions.0.norm.",
            "mid_block.attentions.0.group_norm.",
        )

        # decoder.norm_out → decoder.conv_norm_out
        new_key = new_key.replace(
            "decoder.norm_out.",
            "decoder.conv_norm_out.",
        )

        # ResNet block internal remapping:
        # norm1 → norm1, norm2 → norm2 (same)
        # conv1 → conv1, conv2 → conv2 (same)
        # nin_shortcut → conv_shortcut
        new_key = new_key.replace(".nin_shortcut.", ".conv_shortcut.")

        # Only keep decoder keys (drop encoder keys from the checkpoint)
        if new_key.startswith("decoder.") or new_key.startswith("quant_conv.") or new_key.startswith("post_quant_conv."):
            new_sd[new_key] = v

    return new_sd


def patch_pipeline_decoder(pipe, decoder_path: str):
    """Replace the pipeline's VAE decoder weights with Stable Signature's fine-tuned decoder.

    Args:
        pipe: StableDiffusionPipeline with standard VAE.
        decoder_path: Path to sd2_decoder.pth (LDM format).

    Returns:
        The same pipeline, with patched decoder.
    """
    raw_sd = torch.load(decoder_path, map_location="cpu", weights_only=True)
    remapped = _remap_ldm_decoder_keys(raw_sd)

    # Load only decoder keys into the VAE (strict=False — encoder keys untouched)
    missing, unexpected = pipe.vae.load_state_dict(remapped, strict=False)

    # All missing keys should be encoder-side; unexpected should be empty
    decoder_missing = [k for k in missing if k.startswith("decoder.")]
    if decoder_missing:
        print(f"WARNING: {len(decoder_missing)} decoder keys missing after remapping:")
        for k in decoder_missing[:10]:
            print(f"  {k}")

    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys:")
        for k in unexpected[:10]:
            print(f"  {k}")

    # Quick sanity check: decoder should produce non-NaN output
    with torch.no_grad():
        dummy_latent = torch.randn(1, 4, 8, 8).to(pipe.device, pipe.vae.dtype)
        test_out = pipe.vae.decode(dummy_latent).sample
        if torch.isnan(test_out).any():
            raise RuntimeError(
                "Decoder produces NaN after patching — key remapping likely failed. "
                "Try download_decoder() again or use Fallback A (omegaconf + LDM loader)."
            )

    print("Stable Signature decoder patched successfully.")
    return pipe
