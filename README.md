# qwen-image-edit-runpod

# Qwen Image Edit on Runpod Hub

Deploy the **Qwen/Qwen-Image-Edit-2509** model as a serverless image editing tool on Runpod Hub.

### ✨ Features

- **4-bit Quantization**: Memory-efficient inference with NF4 quantization
- **Lightning LoRA**: Optional fast inference (4 or 8 steps vs 40 steps)
- **CPU Offloading**: Efficient memory management for large models
- **Runtime Model Loading**: Models downloaded on-demand, not baked into image

### 🔧 Usage

This model edits input images according to a text prompt.

**Example Input:**

[![Runpod](https://api.runpod.io/badge/felipejfc/qwen-image-edit-runpod)](https://console.runpod.io/hub/felipejfc/qwen-image-edit-runpod)

```json
{
  "prompt": "Turn this cat into a dog",
  "image_url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png",
  "num_inference_steps": 8,
  "true_cfg_scale": 4.0,
  "seed": 42
}
```

### 📝 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | "Enhance the image" | Text description of desired edit |
| `image_url` | string | **required** | URL of input image |
| `num_inference_steps` | int | 8 (with Lightning) or 40 | Number of denoising steps |
| `true_cfg_scale` | float | 4.0 | Classifier-free guidance scale |
| `negative_prompt` | string | null | What to avoid in the image |
| `seed` | int | random | Random seed for reproducibility |

### ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_LIGHTNING_LORA` | `true` | Enable Lightning LoRA for fast inference |
| `LIGHTNING_STEPS` | `8` | Steps for Lightning mode (4 or 8) |
| `MODEL_DIR` | `/models/qwen_image_edit_2509` | Cache directory for models |

### 🏗️ Building

Build for x86_64/amd64 (required for RunPod GPUs):

```bash
./build.sh
```

Or manually:

```bash
docker buildx build --platform linux/amd64 -f .runpod/Dockerfile -t your-image:tag --load .
```
