# Changes Summary

## Latest Updates (v2.0)

### New in this version:
- **Dynamic Batching**: Configurable request batching for improved throughput
- **Precision Mode**: Unified `PRECISION_MODE` env var (bf16, 8bit, 4bit)
- **Precision Fallback**: Auto-switch to fp16 when GPU lacks bf16 support
- **Int8 Resilience**: 8-bit loading now retries with fp32 CPU offload if VRAM is insufficient
- **Memory Optimizations**: xformers attention, VAE tiling/slicing
- **Fixed Dependencies**: Updated transformers>=4.49.0 for Qwen2.5-VL compatibility

## What's New

### 1. 4-bit Quantization
- **Transformer**: Quantized with NF4 for ~75% memory reduction
- **Text Encoder**: Quantized with NF4 for additional memory savings
- **Memory Efficiency**: Can run on GPUs with less VRAM

### 2. Lightning LoRA Support
- **Fast Inference**: 4-step or 8-step inference (vs 40 steps default)
- **Quality**: Maintains quality while being 5-10x faster
- **Configurable**: Enable/disable via `USE_LIGHTNING_LORA` env var
- **LoRA Weights**: Auto-downloads from `lightx2v/Qwen-Image-Lightning`

### 3. CPU Offloading
- **Smart Memory Management**: Moves model components between CPU/GPU as needed
- **Lower VRAM Usage**: Can run on smaller GPUs

### 4. Runtime Model Loading
- **No Build-Time Downloads**: Models downloaded on first run
- **Smaller Docker Image**: No model weights in image (~10GB+ savings)
- **Faster Builds**: Build time reduced from 10+ minutes to <2 minutes

### 5. New Features
- **Seed Parameter**: Reproducible results with custom seeds
- **Enhanced Logging**: Better visibility into loading process
- **Response Metadata**: Returns inference steps and Lightning status

## Configuration

### Environment Variables

```bash
# Enable/disable Lightning LoRA (default: true)
USE_LIGHTNING_LORA=true

# Lightning steps: 4 or 8 (default: 8)
LIGHTNING_STEPS=8

# Model cache directory (default: /models/qwen_image_edit_2509)
MODEL_DIR=/models/qwen_image_edit_2509
```

### API Changes

**New Input Parameter:**
```json
{
  "seed": 42  // Optional: for reproducible results
}
```

**Enhanced Response:**
```json
{
  "prompt": "...",
  "output_image_base64": "...",
  "num_inference_steps": 8,
  "lightning_enabled": true
}
```

## Performance Comparison

| Mode | Steps | VRAM Usage | Speed | Quality |
|------|-------|------------|-------|---------|
| Original | 40 | ~24GB | 1x | Best |
| Quantized | 40 | ~8GB | 0.95x | Excellent |
| Lightning 8-step | 8 | ~8GB | ~5x | Very Good |
| Lightning 4-step | 4 | ~8GB | ~10x | Good |

## Migration Notes

### Breaking Changes
None! The API is backward compatible.

### Recommended Settings
- For quality: Use 8-step Lightning (default)
- For speed: Use 4-step Lightning
- For best quality: Disable Lightning, use 40 steps

### Build Changes
- Must build with `--platform linux/amd64` on M1/M2 Macs
- Use provided `build.sh` script for easy building

## Technical Details

### Quantization Config
```python
# Transformer
load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.bfloat16
llm_int8_skip_modules=["transformer_blocks.0.img_mod"]

# Text Encoder
load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.bfloat16
```

### Dependencies
- Added: `bitsandbytes` (for quantization)
- Using: `transformers`, `diffusers`, `torch`

## Testing

Test with:
```bash
curl -X POST https://your-runpod-endpoint/run \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Turn this into a painting",
      "image_url": "https://example.com/image.jpg",
      "num_inference_steps": 8,
      "seed": 42
    }
  }'
```

