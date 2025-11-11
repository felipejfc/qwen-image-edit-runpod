# qwen-image-edit-runpod

# Qwen Image Edit on Runpod Hub

Deploy the **Qwen/Qwen-Image-Edit-2509** model as a serverless image editing tool on Runpod Hub.

### ✨ Features

- **Configurable Precision**: Run in 8-bit, 4-bit, or full bf16 for maximum quality
- **Lightning LoRA**: Optional fast inference (4 or 8 steps vs 40 steps)
- **Dynamic Batching**: Batch multiple requests for improved throughput
- **CPU Offloading**: Efficient memory management for large models
- **Runtime Model Loading**: Models downloaded on-demand, not baked into image

### 🔧 Usage

This model edits input images according to a text prompt. It can run in two modes:

#### 🚀 Standalone FastAPI Mode

Run as a standalone API server with direct HTTP endpoints:

```bash
STANDALONE_MODE=1 API_HOST=0.0.0.0 API_PORT=8000 python handler.py
```

**API Endpoints:**
- `POST /run` - Submit inference request (async-compatible)
- `POST /runsync` - Alias for `/run` (RunPod compatibility)
- `GET /health` - Health check endpoint

**Example Request:**

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Turn this cat into a dog",
      "image_url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png",
      "num_inference_steps": 8,
      "true_cfg_scale": 4.0,
      "seed": 42
    }
  }'
```

#### ☁️ RunPod Serverless Mode

Deploy to RunPod Hub for auto-scaling serverless execution:

[![Runpod](https://api.runpod.io/badge/felipejfc/qwen-image-edit-runpod)](https://console.runpod.io/hub/felipejfc/qwen-image-edit-runpod)

```json
{
  "input": {
    "prompt": "Turn this cat into a dog",
    "image_url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png",
    "num_inference_steps": 8,
    "true_cfg_scale": 4.0,
    "seed": 42
  }
}
```

### 📝 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | "Enhance the image" | Text description of desired edit |
| `image_url` | string | **required** | URL of input image |
| `num_inference_steps` | int | 8 (with Lightning) or 40 | Number of denoising steps |
| `true_cfg_scale` | float | 4.0 | CFG (Classifier-Free Guidance) scale - controls prompt adherence. Higher values (7-15) follow prompt more strictly, lower values (1-3) allow more creativity |
| `negative_prompt` | string | null | What to avoid in the image |
| `seed` | int | random | Random seed for reproducibility |

### ⚙️ Environment Variables

#### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `PRECISION_MODE` | `8bit` | Precision selection: `8bit`, `4bit`, or `bf16` |
| `USE_LIGHTNING_LORA` | `true` | Enable Lightning LoRA for fast inference |
| `LIGHTNING_STEPS` | `8` | Steps for Lightning mode (4 or 8) |
| `DEFAULT_CFG_SCALE` | `4.0` | Default CFG (Classifier-Free Guidance) scale. Higher (7-15) = stricter prompt adherence, lower (1-3) = more creative freedom |
| `BATCH_SIZE` | `4` | Maximum number of requests per batch |
| `BATCH_TIMEOUT_MS` | `40` | Wait time in milliseconds before processing partial batch |
| `MODEL_DIR` | `/models/qwen_image_edit_2509` | Cache directory for models |

#### Standalone FastAPI Mode

| Variable | Default | Description |
|----------|---------|-------------|
| `STANDALONE_MODE` | `0` | Enable FastAPI server mode (set to `1`) |
| `API_HOST` | `0.0.0.0` | Host address for FastAPI server |
| `API_PORT` | `8000` | Port for FastAPI server |
| `API_WORKERS` | `1` | Number of uvicorn worker processes |

Note: If `PRECISION_MODE=bf16` but the GPU does not support bfloat16, the handler automatically falls back to fp16 and logs a warning. When running in 8-bit mode, the loader first tries to keep everything on the GPU and, if that fails due to limited VRAM, it transparently retries with fp32 CPU offload for the remaining modules.

### 🔄 Dynamic Batching

The handler supports dynamic batching to maximize GPU utilization and throughput:

- **How it works**: Requests arriving within `BATCH_TIMEOUT_MS` are collected and processed together
- **Max batch size**: Controlled by `BATCH_SIZE` - when reached, processes immediately
- **Timeout**: If timeout expires before batch fills, processes partial batch
- **Smart bucketing**: Requests are grouped by image dimensions, inference steps, and CFG scale for optimal batching
- **Thread-safe**: Multiple concurrent requests are safely handled and batched
- **Works in all modes**: Batching is available in both RunPod serverless and standalone FastAPI modes

#### Batching Performance

| Configuration | Use Case | Throughput |
|---------------|----------|------------|
| `BATCH_SIZE=1` | Lowest latency, single requests | 1x baseline |
| `BATCH_SIZE=4, BATCH_TIMEOUT_MS=40` | Balanced throughput/latency | 2-3x baseline |
| `BATCH_SIZE=8, BATCH_TIMEOUT_MS=100` | Maximum throughput | 3-5x baseline |

Example configurations:

```bash
# High throughput API server (recommended for production)
STANDALONE_MODE=1 BATCH_SIZE=4 BATCH_TIMEOUT_MS=40 python handler.py

# Maximum throughput (for heavy load)
STANDALONE_MODE=1 BATCH_SIZE=8 BATCH_TIMEOUT_MS=100 python handler.py

# Low latency (single request optimization)
STANDALONE_MODE=1 BATCH_SIZE=1 python handler.py
```

**Note**: FastAPI's async architecture combined with the batching system allows multiple concurrent requests to be efficiently grouped and processed together, maximizing GPU utilization without blocking.

### 🧪 Testing Standalone Mode

Test the API with both `image_url` and `image_base64` inputs:

```bash
# Using image URL
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Make the sky more dramatic",
      "image_url": "https://example.com/image.jpg",
      "num_inference_steps": 8,
      "seed": 12345
    }
  }'

# Using base64 encoded image
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": {
      \"prompt\": \"Add vibrant colors\",
      \"image_base64\": \"$(base64 -i input.jpg)\",
      \"num_inference_steps\": 8
    }
  }"

# Health check
curl http://localhost:8000/health
```

### 🏗️ Building

Build for x86_64/amd64 (required for RunPod GPUs):

```bash
./build.sh
```

Or manually:

```bash
docker buildx build --platform linux/amd64 -f .runpod/Dockerfile -t your-image:tag --load .
```

### 🔧 Troubleshooting

**Import Errors (GradientCheckpointingLayer, etc.)**
- Ensure `transformers>=4.49.0` is installed
- The Dockerfile now specifies minimum versions for compatibility
- Rebuild your Docker image to get updated dependencies

**RuntimeError: operator torchvision::nms does not exist**
- Install `torchvision>=0.20.0` (included in Dockerfile/requirements)
- Rebuild image to ensure torchvision matches the bundled torch version
- If using a custom base image, make sure `torchvision` is compiled for your CUDA toolkit

**Out of Memory Errors**
- Try `PRECISION_MODE=4bit` for lowest memory usage (~6GB VRAM)
- Enable batching with smaller batch sizes: `BATCH_SIZE=2`
- Default `PRECISION_MODE=8bit` uses ~8-10GB VRAM

**Slow Performance**
- Enable Lightning LoRA: `USE_LIGHTNING_LORA=true LIGHTNING_STEPS=8`
- For throughput, enable batching: `BATCH_SIZE=4 BATCH_TIMEOUT_MS=200`
- Ensure xformers is properly installed for efficient attention
