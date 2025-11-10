# Standalone FastAPI Mode

This document describes the standalone FastAPI mode added to the Qwen Image Edit handler.

## Overview

The handler now supports running as a standalone FastAPI server, allowing direct HTTP API access without requiring RunPod's serverless infrastructure. This mode is optimized for high-throughput image generation workflows and maintains full compatibility with the fotoapp backend's API expectations.

## Key Features

✅ **Full API Compatibility** - Matches the exact API format expected by `fotoapp/backend/internal/imagegen/runpod_provider.go`  
✅ **Batching Preserved** - Uses the same `BatchMux` system for maximum GPU utilization  
✅ **High Throughput** - FastAPI's async architecture + smart batching = optimal performance  
✅ **Flexible Input** - Supports both `image_url` and `image_base64` inputs  
✅ **Health Checks** - Built-in `/health` endpoint for monitoring  

## Quick Start

```bash
# Start the FastAPI server
STANDALONE_MODE=1 python handler.py

# With custom configuration
STANDALONE_MODE=1 \
  API_HOST=0.0.0.0 \
  API_PORT=8000 \
  BATCH_SIZE=4 \
  BATCH_TIMEOUT_MS=40 \
  python handler.py
```

## API Endpoints

### POST /run
Submit an image editing request. Returns the edited image as base64.

**Request Format:**
```json
{
  "input": {
    "prompt": "Make the sky more dramatic",
    "image_url": "https://example.com/image.jpg",
    "num_inference_steps": 8,
    "true_cfg_scale": 4.0,
    "seed": 12345,
    "negative_prompt": "blurry, low quality"
  }
}
```

**Alternative with base64:**
```json
{
  "input": {
    "prompt": "Add vibrant colors",
    "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
    "num_inference_steps": 8
  }
}
```

**Response Format:**
```json
{
  "image": "iVBORw0KGgoAAAANSUhEUg...",
  "prompt": "Make the sky more dramatic",
  "num_inference_steps": 8,
  "lightning_enabled": true
}
```

### POST /runsync
Alias for `/run` - maintains compatibility with RunPod's endpoint patterns.

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "ready",
  "gpu_available": true
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STANDALONE_MODE` | `0` | Set to `1` to enable FastAPI server |
| `API_HOST` | `0.0.0.0` | Host address to bind to |
| `API_PORT` | `8000` | Port to listen on |
| `API_WORKERS` | `1` | Number of uvicorn workers |

## Batching Configuration

The standalone mode uses the same batching system as RunPod serverless mode:

| Config | Latency | Throughput | Best For |
|--------|---------|------------|----------|
| `BATCH_SIZE=1` | Lowest | 1x | Single requests, development |
| `BATCH_SIZE=4, BATCH_TIMEOUT_MS=40` | Medium | 2-3x | Production (recommended) |
| `BATCH_SIZE=8, BATCH_TIMEOUT_MS=100` | Higher | 3-5x | High load scenarios |

### How Batching Works

1. **Request arrives** → Added to bucket based on (width, height, steps, cfg_scale)
2. **Bucket fills or timeout** → Batch is processed on GPU
3. **Results returned** → Each request gets its individual result
4. **Concurrent requests** → FastAPI handles multiple concurrent requests, the batching system groups them intelligently

### Example: High Throughput Setup

```bash
STANDALONE_MODE=1 \
  BATCH_SIZE=4 \
  BATCH_TIMEOUT_MS=40 \
  USE_LIGHTNING_LORA=true \
  LIGHTNING_STEPS=8 \
  PRECISION_MODE=8bit \
  python handler.py
```

This configuration:
- Batches up to 4 requests together
- Waits max 40ms for batch to fill
- Uses Lightning LoRA for fast inference (8 steps)
- Runs in 8-bit mode for memory efficiency
- Can handle 2-3x more throughput than single-request mode

## Compatibility with Fotoapp Backend

The API format is designed to be 100% compatible with the fotoapp backend's `runpod_provider.go`:

| Go Field | API Field | Notes |
|----------|-----------|-------|
| `input.prompt` | `input.prompt` | Text prompt |
| `input.image_url` | `input.image_url` | Public image URL |
| `input.image_base64` | `input.image_base64` | Base64 encoded image |
| `input.seed` | `input.seed` | Random seed |
| `input.width` | `input.width` | Target width (optional) |
| `input.height` | `input.height` | Target height (optional) |
| `input.num_inference_steps` | `input.num_inference_steps` | Steps |
| `input.true_cfg_scale` | `input.true_cfg_scale` | CFG scale |
| `input.negative_prompt` | `input.negative_prompt` | Negative prompt |

**Response field**: The response uses `"image"` as the key (not `output_image_base64`), which matches the Go parser's expected format in `decodeImagePayload`.

## Testing

```bash
# Test with image URL
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Make it more artistic",
      "image_url": "https://example.com/test.jpg",
      "num_inference_steps": 8
    }
  }'

# Test with base64
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": {
      \"prompt\": \"Enhance colors\",
      \"image_base64\": \"$(base64 -i test.jpg | tr -d '\n')\",
      \"num_inference_steps\": 8,
      \"seed\": 42
    }
  }"

# Health check
curl http://localhost:8000/health
```

## Performance Tips

1. **Enable Lightning LoRA**: `USE_LIGHTNING_LORA=true LIGHTNING_STEPS=8` for 5x faster inference
2. **Optimize batch settings**: Start with `BATCH_SIZE=4 BATCH_TIMEOUT_MS=40` and adjust based on load
3. **Use 8-bit mode**: `PRECISION_MODE=8bit` for good balance of speed and memory
4. **Monitor GPU**: Watch GPU utilization to ensure batching is effective
5. **Load test**: Use tools like `wrk` or `locust` to find optimal batch settings

## Architecture

```
┌─────────────────┐
│  FastAPI App    │
│  (async)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  BatchMux       │
│  (thread-safe)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Bucket System  │
│  (by size/cfg)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GPU Pipeline   │
│  (batched)      │
└─────────────────┘
```

## Comparison: Standalone vs RunPod Serverless

| Feature | Standalone | RunPod Serverless |
|---------|------------|-------------------|
| Setup | Simple, just run script | Requires RunPod account |
| Scaling | Manual (add instances) | Auto-scaling |
| Batching | ✅ Supported | ✅ Supported |
| Costs | Your infrastructure | Pay per second |
| Latency | Direct (no cold starts) | Possible cold starts |
| Best For | Self-hosted, high control | Cloud deployment, variable load |

## Dependencies

Added to `requirements.txt`:
- `fastapi>=0.104.0` - Web framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `pydantic>=2.0.0` - Request/response validation

These are only loaded when `STANDALONE_MODE=1`, so they don't affect RunPod serverless mode.

