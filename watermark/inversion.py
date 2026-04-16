"""DDIM inversion using modern diffusers API.

Given an image, encodes it to a latent and iteratively inverts the diffusion
process to recover an estimate of the initial noise x_T. This inverted noise
can then be analyzed in Fourier space for watermark detection.
"""

import os
import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DDIMInverseScheduler,
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# MPS for UNet/VAE; FFT falls back to CPU automatically
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float32  # float16 causes NaN in VAE on MPS; float32 is fast enough with SD 1.5


def load_pipeline(
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    cache_dir: str | None = None,
    low_memory: bool = False,
) -> StableDiffusionPipeline:
    """Load SD pipeline with DDIM scheduler configured for inversion.

    Args:
        model_id: HuggingFace model ID. Default is SD 1.5 (512x512, fast on MPS).
            For SD 2.1 (768x768), use "sd2-community/stable-diffusion-2-1".
        cache_dir: Where to cache model weights. Defaults to DW_MODEL_DIR env var.
        low_memory: If True, load UNet and text encoder in float16 (saves ~2 GB)
            while keeping VAE in float32 (float16 VAE causes NaN on MPS).

    Returns:
        StableDiffusionPipeline ready for generation and inversion.
    """
    if cache_dir is None:
        cache_dir = os.environ.get(
            "DW_MODEL_DIR",
            os.path.join(os.path.dirname(__file__), "..", ".models"),
        )

    load_dtype = torch.float16 if low_memory else DTYPE

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=load_dtype,
        cache_dir=cache_dir,
        safety_checker=None,
        requires_safety_checker=False,
    )

    if low_memory:
        # Keep VAE in float32 to avoid NaN on MPS decoder
        pipe.vae = pipe.vae.to(dtype=torch.float32)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe


def encode_image(pipe: StableDiffusionPipeline, image: Image.Image) -> torch.Tensor:
    """Encode a PIL image to a VAE latent.

    Args:
        pipe: Loaded SD pipeline.
        image: Input image (will be resized to pipeline's expected size).

    Returns:
        Latent tensor of shape (1, 4, H/8, W/8).
    """
    # Resize to match pipeline's expected resolution
    target_size = pipe.unet.config.sample_size * pipe.vae_scale_factor
    image = image.resize((target_size, target_size), Image.LANCZOS)

    # Convert to tensor [-1, 1]
    img_np = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    img_tensor = (img_tensor * 2.0 - 1.0).to(pipe.device, DTYPE)

    # VAE encode
    with torch.no_grad():
        latent_dist = pipe.vae.encode(img_tensor)
        latent = latent_dist.latent_dist.mean * pipe.vae.config.scaling_factor

    return latent


def ddim_invert(
    pipe: StableDiffusionPipeline,
    image: Image.Image,
    num_inference_steps: int = 50,
    prompt: str = "",
) -> torch.Tensor:
    """DDIM-invert an image back to its estimated initial noise x_T.

    Uses the unconditional (empty prompt) trajectory for inversion, which
    works well for watermark detection since we only need a rough recovery
    of the noise's Fourier structure.

    Args:
        pipe: Loaded SD pipeline.
        image: Image to invert.
        num_inference_steps: Number of DDIM steps for inversion.
        prompt: Text prompt (empty for unconditional inversion).

    Returns:
        Estimated initial noise x_T, shape (1, 4, H/8, W/8).
    """
    # Encode image to latent
    latent = encode_image(pipe, image)

    # Set up inverse scheduler
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inverse_scheduler.set_timesteps(num_inference_steps, device=pipe.device)
    timesteps = inverse_scheduler.timesteps

    # Get text embeddings for guidance
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(pipe.device))[0]

    # Also get unconditional embeddings
    uncond_inputs = pipe.tokenizer(
        "",
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(uncond_inputs.input_ids.to(pipe.device))[0]

    # Concat for classifier-free guidance (not used with scale=1, but keeps API consistent)
    embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Iterative DDIM inversion: x_0 → x_1 → ... → x_T
    x = latent
    with torch.no_grad():
        for t in timesteps:
            # Predict noise at this timestep
            model_input = torch.cat([x, x])  # unconditional + conditional
            noise_pred = pipe.unet(model_input, t, encoder_hidden_states=embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # No guidance for inversion (scale = 1.0)
            noise_pred = noise_pred_uncond

            # Inverse DDIM step: go from x_t to x_{t+1}
            x = inverse_scheduler.step(noise_pred, t, x).prev_sample

    return x.float()


def ddim_reconstruct(
    pipe: StableDiffusionPipeline,
    latent: torch.Tensor,
    num_inference_steps: int = 50,
    prompt: str = "",
) -> Image.Image:
    """Reconstruct an image from a noise latent via DDIM forward pass.

    Used to verify inversion quality: invert(image) → latent → reconstruct → compare.

    Args:
        pipe: Loaded SD pipeline.
        latent: Initial noise latent x_T.
        num_inference_steps: Number of DDIM steps.
        prompt: Text prompt.

    Returns:
        Reconstructed PIL image.
    """
    pipe.scheduler.set_timesteps(num_inference_steps, device=pipe.device)

    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(pipe.device))[0]

    uncond_inputs = pipe.tokenizer(
        "",
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(uncond_inputs.input_ids.to(pipe.device))[0]

    embeddings = torch.cat([uncond_embeddings, text_embeddings])

    x = latent.to(pipe.device, DTYPE)
    with torch.no_grad():
        for t in pipe.scheduler.timesteps:
            model_input = torch.cat([x, x])
            noise_pred = pipe.unet(model_input, t, encoder_hidden_states=embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond
            x = pipe.scheduler.step(noise_pred, t, x).prev_sample

    # Decode latent to image
    x = x / pipe.vae.config.scaling_factor
    with torch.no_grad():
        decoded = pipe.vae.decode(x).sample

    # Convert to PIL
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.cpu().permute(0, 2, 3, 1).numpy()[0]
    decoded = (decoded * 255).round().astype(np.uint8)
    return Image.fromarray(decoded)
