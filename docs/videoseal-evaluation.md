# VideoSeal Watermark Evaluation

## What is VideoSeal?

**VideoSeal** is a neural image/video watermarking system developed by **Meta (FAIR)**. It embeds an invisible binary message into images using end-to-end learned neural networks — both the encoder (embedder) and decoder (extractor) are CNNs trained jointly.

This makes it architecturally closer to Google's **SynthID** than any other open-source watermark. SynthID is our eventual attack target but is closed-source; VideoSeal at 256 bits is our best proxy.

### How it differs from our other watermarks

| | VideoSeal | Tree-Ring | Stable Signature |
|---|---|---|---|
| **Type** | Learned neural encoder/decoder | Fourier noise injection | Fine-tuned VAE decoder |
| **Capacity** | 256 bits | 0 bits (presence only) | 48 bits (fixed key) |
| **Approach** | Post-hoc (any image) | Generation-time (noise) | Generation-time (decoder) |
| **Needs SD pipeline** | No | Yes | Yes |
| **Embed speed** | ~0.3s/image | ~30s/image | ~1s/image |
| **Detect speed** | ~0.1s/image | ~30s/image | ~0.1s/image |

---

## Model Architecture

We use the **videoseal_1.0** model card (checkpoint `y_256b_img.pth`, ~260 MB).

### Embedder: `unet_small2_yuv_quant`

Operates in YUV colour space — embeds the watermark only in the Y (luma) channel to exploit the human visual system's lower sensitivity to luma changes at high frequencies.

```
Input:  Y channel of image (1, 1, 256, 256)
        + 256-bit binary message
Output: Watermark residual (1, 1, 256, 256)

UNet: 8 blocks, z_channels=16, mults=[1, 2, 4, 8]
      batch normalization, ReLU, tanh final activation
Message processor: binary+concat, nbits=16 internal, hidden_size=32
```

The residual is **additively blended** with the original image:
```
watermarked = scaling_i * original + scaling_w * residual
```
where `scaling_i = 1.0` and `scaling_w` controls watermark strength.

A **JND (Just Noticeable Difference) attenuator** suppresses the watermark in perceptually sensitive regions.

### Extractor: `convnext_tiny`

```
ConvNeXt encoder: depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]
Pixel decoder:    embed_dim=768, nbits=16 internal
Output:           (1, 257) — 1 presence indicator + 256 message bit logits
```

The first output (index 0) is a watermark-presence score. The remaining 256 values are logits for each message bit (positive → 1, negative → 0).

---

## What We Embedded

### Message

A **deterministic 256-bit pseudorandom binary sequence**, generated with `torch.randint(0, 2, (1, 256))` seeded at `seed=42`.

```
First 32 bits: [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
Distribution:  126 ones / 130 zeros (roughly balanced)
```

This is a fixed test message — **the same 256 bits were embedded into every test image**. Detection checks whether this exact message can be recovered. In a real deployment, these 256 bits would carry provenance metadata (e.g. hashed creator ID + token ID + ECC), and each image could carry a unique message.

### Embedding Strengths

We tested two `scaling_w` values:

| `scaling_w` | Description |
|---|---|
| **0.2** | Default (from model card). Designed for imperceptibility. |
| **0.3** | 50% stronger. More robust to attacks, slightly more visible. |

---

## Test Images

**Dataset:** DiffusionDB — 1,000 Stable Diffusion-generated images (512×512 PNG).

This is the target domain for our attack research: AI-generated images are what SynthID watermarks in practice. We used the first 5 images for this evaluation.

| Image | Dimensions | Source |
|---|---|---|
| `0000.png` | 512×512 | DiffusionDB partition 0 |
| `0001.png` | 512×512 | DiffusionDB partition 0 |
| `0002.png` | 512×768 | DiffusionDB partition 0 |
| `0003.png` | 512×512 | DiffusionDB partition 0 |
| `0004.png` | 512×512 | DiffusionDB partition 0 |

Watermarked images preserve the original dimensions exactly.

---

## Image Quality Degradation

### scaling_w = 0.2 (default strength)

| Image | SSIM | PSNR |
|---|---|---|
| 0000 | 0.9955 | 43.3 dB |
| 0001 | 0.9946 | 44.8 dB |
| 0002 | 0.9957 | 44.9 dB |
| 0003 | 0.9957 | 46.2 dB |
| 0004 | 0.9954 | 45.7 dB |
| **Average** | **0.9954** | **45.0 dB** |

### scaling_w = 0.3 (stronger)

| Image | SSIM | PSNR |
|---|---|---|
| 0000 | 0.9903 | 39.9 dB |
| 0001 | 0.9891 | 41.4 dB |
| 0002 | 0.9912 | 41.5 dB |
| 0003 | 0.9923 | 43.0 dB |
| 0004 | 0.9905 | 42.3 dB |
| **Average** | **0.9907** | **41.6 dB** |

**Interpretation:**
- SSIM > 0.99 at both strengths — the watermark is effectively invisible.
- PSNR 41–45 dB is well above the "visually lossless" threshold (~38 dB).
- Increasing `scaling_w` from 0.2 to 0.3 costs ~3.4 dB PSNR (still excellent quality).

---

## Detection Results

### True Positives (watermarked images)

Detection threshold: p-value < 0.01 (binomial test, one-sided).

#### scaling_w = 0.2

| Image | Bit Accuracy | Matching Bits | p-value | Detected |
|---|---|---|---|---|
| 0000 | 100.0% | 256/256 | 8.64e-78 | Yes |
| 0001 | 99.2% | 254/256 | 2.84e-73 | Yes |
| 0002 | 100.0% | 256/256 | 8.64e-78 | Yes |
| 0003 | 100.0% | 256/256 | 8.64e-78 | Yes |
| 0004 | 100.0% | 256/256 | 8.64e-78 | Yes |
| **Total** | **99.8%** | | | **5/5 (100%)** |

#### scaling_w = 0.3

| Image | Bit Accuracy | Matching Bits | p-value | Detected |
|---|---|---|---|---|
| 0000 | 100.0% | 256/256 | 8.64e-78 | Yes |
| 0001 | 100.0% | 256/256 | 8.64e-78 | Yes |
| 0002 | 100.0% | 256/256 | 8.64e-78 | Yes |
| 0003 | 100.0% | 256/256 | 8.64e-78 | Yes |
| 0004 | 100.0% | 256/256 | 8.64e-78 | Yes |
| **Total** | **100.0%** | | | **5/5 (100%)** |

### False Positives (unwatermarked images)

#### DiffusionDB — in-domain (same dataset, no watermark)

| Image | Bit Accuracy | p-value | Detected |
|---|---|---|---|
| 0000 | 55.1% | 5.90e-02 | No |
| 0001 | 51.2% | 3.77e-01 | No |
| 0002 | 52.7% | 2.08e-01 | No |
| 0003 | 52.0% | 2.87e-01 | No |
| 0004 | 48.0% | 7.54e-01 | No |
| **Total** | **51.8%** | | **0/5 (0%)** |

#### DIV2K — out-of-domain (natural photos, ~2040px, no watermark)

| Image | Resolution | Bit Accuracy | p-value | Detected |
|---|---|---|---|---|
| 0001 | 2040×1404 | 51.2% | 3.77e-01 | No |
| 0002 | 2040×1848 | 47.3% | 8.26e-01 | No |
| 0003 | 2040×1356 | 52.0% | 2.87e-01 | No |
| 0004 | 2040×1344 | 47.3% | 8.26e-01 | No |
| 0005 | 1608×2040 | 56.2% | 2.62e-02 | No |
| **Total** | | **50.8%** | | **0/5 (0%)** |

---

## Summary

| Metric | scaling_w=0.2 | scaling_w=0.3 |
|---|---|---|
| **True positive rate** | 100% (5/5) | 100% (5/5) |
| **False positive rate** | 0% (0/10) | 0% (0/10) |
| **Avg bit accuracy (watermarked)** | 99.8% | 100.0% |
| **Avg bit accuracy (clean)** | 51.8% | 51.8% |
| **Avg SSIM** | 0.9954 | 0.9907 |
| **Avg PSNR** | 45.0 dB | 41.6 dB |
| **p-value separation** | 10^-74 vs 10^-1 | 10^-78 vs 10^-1 |

**Note:** These results are pre-attack. The watermark has not yet been subjected to our img2img regeneration attack. The purpose of this evaluation is to confirm the watermark embeds and detects correctly before we try to break it.

---

## Detection Method

Detection uses a **one-sided binomial test**:

1. Extract 256 bit logits from the ConvNeXt detector
2. Threshold at 0 → predicted bits
3. Count matching bits against expected message
4. Under H0 (no watermark): each bit matches with p=0.5 independently
5. p-value = P(matching ≥ observed | H0)
6. Reject H0 (declare watermark detected) if p < 0.01

This is identical to the statistical framework used for Stable Signature detection (but with 256 bits instead of 48, giving much more statistical power).

---

## Reproduction

```bash
# Embed watermarks
python scripts/embed.py --method videoseal \
  --input-dir "/Volumes/One Touch/diffusion-watermarking/datasets/diffusiondb" \
  --num-images 5 --strengths 0.2 0.3

# Verify detection (Python)
from watermark.videoseal_wm import detect_videoseal, generate_message
from PIL import Image
msg = generate_message(n_bits=256, seed=42)
img = Image.open("images/videoseal/diffusiondb/s0.3/0000.png")
result = detect_videoseal(img, expected_message=msg)
print(result)  # {'p_value': 8.64e-78, 'detected': True, 'bit_accuracy': 1.0, 'matching_bits': 256}
```
