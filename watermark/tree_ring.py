"""Tree-Ring watermark: key generation and Fourier-space embedding.

Implements the core watermark from "Tree-Ring Watermarks: Fingerprints for
Diffusion Images that are Invisible and Robust" (Wen et al., 2023).

The idea: inject a known pattern at specific ring positions in the Fourier
transform of the initial noise tensor. After diffusion, the pattern persists
in the generated image and can be recovered via DDIM inversion + FFT.
"""

import torch


def create_ring_mask(
    h: int, w: int, radius: float = 10.0, ring_width: float = 3.0
) -> torch.Tensor:
    """Create a binary mask selecting ring positions in Fourier space.

    The mask is centered at (h//2, w//2) — matching torch.fft.fftshift convention.
    Pixels whose distance from center falls in [radius - ring_width, radius + ring_width]
    are set to 1; everything else is 0.

    Args:
        h, w: Spatial dimensions of the latent.
        radius: Center radius of the ring (in pixels).
        ring_width: Half-width of the ring band.

    Returns:
        Binary mask of shape (h, w).
    """
    cy, cx = h // 2, w // 2
    ys = torch.arange(h, dtype=torch.float32) - cy
    xs = torch.arange(w, dtype=torch.float32) - cx
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dist = torch.sqrt(yy**2 + xx**2)
    mask = ((dist >= radius - ring_width) & (dist <= radius + ring_width)).float()
    return mask


def generate_key(
    h: int,
    w: int,
    radius: float = 10.0,
    ring_width: float = 3.0,
    seed: int = 42,
) -> torch.Tensor:
    """Generate a deterministic watermark key for the ring positions.

    Values are standard Gaussian at ring positions, zero elsewhere.

    Args:
        h, w: Spatial dimensions.
        radius, ring_width: Ring geometry.
        seed: RNG seed for reproducibility.

    Returns:
        Key tensor of shape (h, w) — nonzero only at ring positions.
    """
    mask = create_ring_mask(h, w, radius, ring_width)
    gen = torch.Generator().manual_seed(seed)
    values = torch.randn(h, w, generator=gen)

    # Enforce Hermitian symmetry: key[y,x] = key[mirror(y,x)]
    # In the shifted FFT domain, the mirror of index y is (H - y) % H.
    # This ensures that embedding the key with zero imaginary part produces
    # a purely real contribution after IFFT, enabling exact roundtrip recovery.
    mirror_y = (h - torch.arange(h)) % h
    mirror_x = (w - torch.arange(w)) % w
    yy, xx = torch.meshgrid(mirror_y, mirror_x, indexing="ij")
    values_mirror = values[yy, xx]
    values = (values + values_mirror) / 2

    key = values * mask
    return key


def embed_watermark(
    noise: torch.Tensor,
    key: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Embed watermark key into a noise tensor via Fourier-space injection.

    For each channel of the noise tensor:
    1. FFT → shift to center
    2. Replace complex value at ring positions with key + 0j
    3. Shift back → IFFT

    This directly stamps the key into the frequency domain. After diffusion,
    DDIM inversion recovers an approximation of this noise, and the key
    can be read back from the ring positions.

    Args:
        noise: Initial noise tensor, shape (B, C, H, W).
        key: Watermark key, shape (H, W). Nonzero only at ring positions.
        mask: Binary ring mask, shape (H, W).

    Returns:
        Modified noise tensor with watermark embedded.
    """
    B, C, H, W = noise.shape
    watermarked = noise.clone()

    # Ensure key and mask are on CPU for FFT operations
    key_cpu = key.cpu()
    mask_cpu = mask.cpu()
    mask_bool = mask_cpu.bool()

    for b in range(B):
        for c in range(C):
            channel = watermarked[b, c].cpu().float()
            # Forward FFT + center shift
            freq = torch.fft.fftshift(torch.fft.fft2(channel))
            # Inject key at ring positions (replace both real and imaginary)
            real = freq.real.clone()
            imag = freq.imag.clone()
            real[mask_bool] = key_cpu[mask_bool]
            imag[mask_bool] = 0.0
            freq = torch.complex(real, imag)
            # Inverse shift + IFFT
            channel_out = torch.fft.ifft2(torch.fft.ifftshift(freq)).real
            watermarked[b, c] = channel_out.to(noise.device)

    return watermarked
