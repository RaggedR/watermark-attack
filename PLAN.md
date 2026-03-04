# Plan: Watermark Robustness Pipeline + VideoSeal Integration

## Context

The current codebase generates images from prompts and watermarks them during generation.
The new pipeline watermarks **existing images** (1000 OCR PNGs at `~/data/diffusion/ocr_quality_pngs/`)
using multiple methods, then attacks them with diffusion img2img to find optimal removal parameters.

The end goal: tune a diffusion attack that transfers to SynthID.

## Architecture Change

**Before:** `prompt → SD generate with watermark → attack → detect`
**After:**  `existing PNG → watermark embed → attack → extract message → report`

The pipeline is **watermark-model agnostic**: you pick a method, pick N images, and the
same sweep/report machinery runs regardless.

---

## Phase 0: Add VideoSeal dependency

1. Clone `facebookresearch/videoseal` into a local path (or add as a dependency)
2. Add to `requirements.txt`: videoseal deps (timm, einops, omegaconf, lpips, pytorch_msssim, av, decord, PyWavelets)
3. Verify `videoseal.load("pixelseal")` works on MPS/CPU
4. Write a quick smoke test

## Phase 1: Watermark adapter layer (`watermark/adapters.py`)

Create a unified interface so the pipeline doesn't care which method is used:

```python
class WatermarkAdapter:
    """Common interface for all watermark methods."""
    name: str
    capacity_bits: int

    def embed(self, image: PIL.Image, message: bytes, strength: float) -> PIL.Image:
        """Embed message into image at given strength. Returns watermarked image."""

    def extract(self, image: PIL.Image) -> dict:
        """Extract message from image. Returns {message, bit_accuracy, confidence, detected}."""

    def encode_message(self, wallet: str, token_id: int) -> bytes/tensor:
        """Encode wallet:tokenId into the method's native format."""

    def decode_message(self, raw) -> tuple[str, int]:
        """Decode native format back to (wallet, token_id)."""
```

Concrete adapters:

| Adapter | Embed | Extract | Message encoding |
|---------|-------|---------|-----------------|
| `VideoSealAdapter` | `model.embed()` on tensor | `model.detect()` → bits | 256-bit binary (wallet hash + token) |
| `StableSignatureAdapter` | VAE encode → patched decode | HiDDeN extractor → 48 bits | 48-bit fixed key (no custom message) |
| `TreeRingAdapter` | DDIM invert → modify noise → regenerate | DDIM invert → FFT check | Zero-bit (presence only) |
| `WatermarkRsAdapter` | Shell out to watermark-rs binary | Shell out to watermark-rs extract | Native string `wallet:tokenId` |

**Note on Tree-Ring:** Since it requires regeneration through the diffusion pipeline,
it will change the image more than the other methods. This is expected and part of the
comparison.

**Note on Stable Signature:** The 48-bit key is fixed in the pretrained decoder.
We can't embed a custom wallet:tokenId. We measure bit accuracy after attack instead.

**Note on message encoding for VideoSeal (256 bits):**
- Bits 0-159: SHA-1 of wallet address (160 bits)
- Bits 160-191: token_id as uint32 (32 bits)
- Bits 192-255: CRC / ECC (64 bits error correction)

## Phase 2: Embedding script (`scripts/embed.py`)

New script replacing `scripts/generate.py` for existing-image workflows:

```
python scripts/embed.py \
  --method videoseal \
  --input-dir ~/data/diffusion/ocr_quality_pngs \
  --num-images 50 \
  --strengths 0.1 0.2 0.3 0.5 \
  --wallet "0x742d35Cc6634C0532925a3b844Bc9e7595f6fE0E" \
  --token-id 42
```

Output structure:
```
images/
  videoseal/
    s0.1/
      0000.png, 0001.png, ...
    s0.2/
      ...
  metadata.json   # method, strengths, message, source paths, etc.
```

Steps:
1. Load N images from input dir
2. For each strength level:
   - Embed watermark using the chosen adapter
   - Save watermarked image
3. Save metadata.json with all parameters

## Phase 3: Verification script (`scripts/verify.py`)

Confirm watermarks are detectable before running the attack sweep:

```
python scripts/verify.py
```

Reads metadata.json, runs extract on every watermarked image, reports:
- Per-image bit accuracy
- Per-strength average bit accuracy
- Any failures (message not recoverable)

## Phase 4: Attack sweep (`scripts/sweep.py` — refactor existing)

Refactor the existing sweep to work with the new adapter-based pipeline:

```
python scripts/sweep.py \
  --attack-model stabilityai/stable-diffusion-2-1 \
  --strengths 0.1 0.2 0.3 0.4 0.5 \
  --guidance-scales 1.0 3.0 7.5 \
  --steps 20 50
```

For each watermarked image × each attack parameter combination:
1. Run img2img regeneration attack
2. Extract watermark from attacked image using adapter
3. Record: SSIM, PSNR, bit_accuracy, message_recovered (bool), p_value

Output: `results/sweep_{method}_{model}.csv`

**Multiple attack models:** The `--attack-model` flag lets you test
SD 2.1, SDXL, etc. as the attacker (the "model agnostic" part).

## Phase 5: Report generation (`scripts/report.py`)

Generate a summary report:
- Table: method × attack_strength → bit_accuracy (averaged)
- Pareto plot: SSIM vs bit_accuracy (find the sweet spot)
- Heatmaps: strength × guidance_scale → detection rate
- Recommended attack parameters per method
- Cross-method comparison: which watermark is hardest to break?

Output: `results/report.md` + PNG plots

## Phase 6: Tests

- `tests/test_adapters.py` — unit tests for each adapter (embed/extract round-trip)
- `tests/test_videoseal.py` — VideoSeal-specific integration test
- `tests/test_embed_script.py` — end-to-end test of the embedding pipeline
- Update existing tests for refactored sweep.py

---

## File changes summary

| File | Action |
|------|--------|
| `watermark/adapters.py` | **NEW** — unified adapter interface + 4 implementations |
| `watermark/videoseal.py` | **NEW** — VideoSeal/PixelSeal wrapper |
| `scripts/embed.py` | **NEW** — embed watermarks into existing images |
| `scripts/verify.py` | **NEW** — verify watermark extraction on clean images |
| `scripts/sweep.py` | **MODIFY** — refactor to use adapters |
| `scripts/report.py` | **NEW** — generate analysis report |
| `watermark/__init__.py` | **MODIFY** — add new exports |
| `requirements.txt` | **MODIFY** — add videoseal dependencies |
| `tests/test_adapters.py` | **NEW** — adapter unit tests |
| `tests/test_videoseal.py` | **NEW** — VideoSeal smoke test |
| `CLAUDE_DOCS/watermark-attack.md` | **MODIFY** — update docs |

## Open questions for Robin

1. **watermark-rs**: Is the binary compiled? Where is the executable?
   Need to know the exact CLI for embed + extract.
2. **Attack models**: Start with SD 2.1 only, or add SDXL from the start?
3. **N**: How many images for the first run? (50? 100? All 1000?)
4. **Strengths for embedding**: What range? (VideoSeal default scaling_w is 0.2)
