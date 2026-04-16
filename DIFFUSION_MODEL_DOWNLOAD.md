# Re-downloading Model Weights

The `.models/` directory was deleted to save ~4 GB of disk space. All weights are re-downloadable.

## What was in `.models/`

| Model | Size | Source |
|-------|------|--------|
| Stable Diffusion v1.5 (HuggingFace cache) | ~4 GB | `stable-diffusion-v1-5/stable-diffusion-v1-5` on HuggingFace |
| Stable Signature extractor | ~1.2 MB | Meta's Stable Signature repo (auto-downloaded) |
| Stable Signature decoder (`dec_48b_whit.torchscript.pt`) | ~198 MB | Meta's Stable Signature repo (auto-downloaded) |

**Total: ~4 GB**

## How to re-download

### Stable Diffusion v1.5 (~4 GB)

Downloaded automatically on first use by any script that calls `load_pipeline()` (e.g. `scripts/generate.py`, `scripts/embed.py`). It uses HuggingFace `diffusers` and caches to `.models/`.

Or trigger it manually:

```python
from watermark.inversion import load_pipeline
pipe = load_pipeline()  # downloads to .models/
```

### Stable Signature weights (~200 MB)

```bash
python scripts/download_models.py
```

Or download only what you need:

```bash
python scripts/download_models.py --extractor-only   # 1.2 MB
python scripts/download_models.py --decoder-only      # 198 MB
```

### Custom model directory

Set `DW_MODEL_DIR` to cache weights elsewhere:

```bash
export DW_MODEL_DIR=/path/to/models
python scripts/download_models.py
```
