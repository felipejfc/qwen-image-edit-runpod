# RunPod Batching Test Guide

This guide explains how to use `test_runpod_batching.py` to test your deployed RunPod endpoint with concurrent load.

## Quick Start

### 1. Configure the Test

You can configure the test in two ways:

#### Option A: Environment Variables (Recommended)

```bash
# Set configuration via environment variables
export API_URL="https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
export API_KEY="YOUR_RUNPOD_API_KEY"
export CONCURRENT_REQUESTS="20"

# Then run the test
python test_runpod_batching.py
```

For standalone mode:
```bash
export API_URL="http://localhost:8888/run"
export API_KEY=""  # No API key needed for local
python test_runpod_batching.py
```

#### Option B: Edit Script

Open `test_runpod_batching.py` and update the configuration section:

```python
# Edit these values in the script
API_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
API_KEY = "YOUR_RUNPOD_API_KEY"
CONCURRENT_REQUESTS = 20

# Prompt templates (FILL THESE IN)
PROMPT_TEMPLATES = [
    "YOUR_PROMPT_1",  # e.g., "Make the colors more vibrant"
    "YOUR_PROMPT_2",  # e.g., "Add dramatic lighting"
    "YOUR_PROMPT_3",  # e.g., "Convert to watercolor style"
    "YOUR_PROMPT_4",  # e.g., "Make it vintage"
    "YOUR_PROMPT_5",  # e.g., "Enhance details"
]
```

### 2. Run the Test

```bash
python test_runpod_batching.py
```

The script will display your configuration before running.

## What It Does

The test script:

1. **Generates shuffled requests**: Creates `CONCURRENT_REQUESTS` number of requests, cycling through your 5 prompts and shuffling them
2. **Sends concurrently**: All requests are sent at once using asyncio
3. **Tracks timing**: Measures individual request duration and total time
4. **Verifies responses**: Checks that each request gets back the correct prompt (ensures no cross-talk)
5. **Saves images locally**: Automatically saves all result images to `test_results/TIMESTAMP/`
6. **Analyzes batching**: Shows timing patterns that indicate batching behavior

## Example Output

```
================================================================================
Configuration
================================================================================
API_URL:             https://api.runpod.ai/v2/abc123/runsync
API_KEY:             ***xyz1
CONCURRENT_REQUESTS: 20
IMAGE_URL:           https://midias.correiobraziliense.com.br/_midias/png/202...

================================================================================
RunPod API Batching Test
================================================================================

Configuration:
  API URL:         https://api.runpod.ai/v2/abc123/runsync
  Concurrent reqs: 20
  Prompt variants: 5
  Image URL:       https://...
  Steps:           8
  CFG Scale:       4.0
  Output dir:      test_results/20241110_143052

Generated 20 shuffled requests
Sending concurrent requests...

================================================================================
Results
================================================================================
Request   0: ✓ 💾  3.45s | Image: 1,234,567 bytes | Prompt: Put girl in a white bikini...
Request   1: ✓ 💾  3.47s | Image: 1,235,789 bytes | Prompt: Put girl in a red bikini...
Request   2: ✓ 💾  3.46s | Image: 1,233,456 bytes | Prompt: Make her dress white...
...

================================================================================
Summary
================================================================================

Overall:
  Total requests:  20
  Successful:      20 (100.0%)
  Failed:          0
  All matched:     ✓ YES
  Images saved:    20
  Total time:      6.82s

Timing:
  Avg time/req:    3.46s
  Min time:        3.42s
  Max time:        3.51s
  Throughput:      2.93 req/s

Per-Prompt Timing:
  Put girl in a white bikini             : 3.45s avg (4 reqs)
  Put girl in a red bikini               : 3.47s avg (4 reqs)
  Make her dress white                   : 3.46s avg (4 reqs)
  Make her wear a white hat              : 3.44s avg (4 reqs)
  Make her wear a red hat                : 3.48s avg (4 reqs)

✓ SUCCESS: All requests completed successfully with correct mapping!
  Batching system is working as expected.

💾 Images saved to: test_results/20241110_143052/
   Total images: 20
================================================================================
```

## Saved Images

All result images are automatically saved to a timestamped directory:

```
test_results/
└── 20241110_143052/           # Timestamp when test was run
    ├── req_000_Put_girl_in_a_white_bikini.png
    ├── req_001_Put_girl_in_a_red_bikini.png
    ├── req_002_Make_her_dress_white.png
    ├── req_003_Make_her_wear_a_white_hat.png
    ├── req_004_Make_her_wear_a_red_hat.png
    ├── ...
    └── req_019_Put_girl_in_a_white_bikini.png
```

**Filename format**: `req_{id:03d}_{prompt_snippet}.png`
- `req_000`: Request ID (padded to 3 digits)
- `Put_girl_in_a_white_bikini`: First 30 chars of prompt (sanitized)
- `.png`: Image format

**Legend**:
- ✓ = Response matched request (correct mapping)
- 💾 = Image successfully saved
- ⚠ = Image not saved (error occurred)

## Understanding Results

### Batching Indicators

**Good batching** (what you want to see):
- Similar timing across all requests (3.4-3.5s in example)
- Total time much less than sum of individual times
- High throughput (multiple requests per second)

**Poor batching** (if you see this):
- Wide variation in timing (2s, 5s, 8s, etc.)
- Total time ≈ sum of individual times
- Low throughput (<1 req/s)

### Expected Performance

With `BATCH_SIZE=4` and `BATCH_TIMEOUT_MS=40`:

| Concurrent Requests | Expected Batches | Expected Throughput |
|---------------------|------------------|---------------------|
| 1-3                 | 1 batch of 1-3   | ~1.5x single req    |
| 4-7                 | 1-2 batches      | 2-3x single req     |
| 8-16                | 2-4 batches      | 3-4x single req     |
| 20+                 | 5+ batches       | 3-5x single req     |

### Common Issues

**All requests timeout:**
- Check your API_URL is correct
- Check your API_KEY is valid
- Check your RunPod endpoint is deployed and running

**Responses don't match prompts:**
- This is a bug - batching response mapping is incorrect
- File an issue with the output

**Very slow (>10s per request):**
- Your RunPod instance might be under-powered
- Consider using Lightning LoRA: `USE_LIGHTNING_LORA=true LIGHTNING_STEPS=8`
- Check GPU utilization in RunPod logs

## Configuration Options

All configuration can be set via environment variables or by editing the script.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_URL` | `https://localhost:8888/run` | API endpoint URL |
| `API_KEY` | `""` | API authentication key (if required) |
| `CONCURRENT_REQUESTS` | `20` | Number of concurrent requests to send |

Example:
```bash
export API_URL="https://api.runpod.ai/v2/abc123/runsync"
export API_KEY="your-api-key-here"
export CONCURRENT_REQUESTS="50"
python test_runpod_batching.py
```

### Concurrent Requests

```python
CONCURRENT_REQUESTS = 20  # Test with 20 concurrent requests
```

Recommended values:
- **Light load**: 5-10 requests
- **Medium load**: 20-50 requests  
- **Heavy load**: 100-200 requests

### Prompts

Provide 5 different prompts to test variety:

```python
PROMPT_TEMPLATES = [
    "Make the colors more vibrant and saturated",
    "Add dramatic lighting and shadows",
    "Convert to a watercolor painting style",
    "Make it look like a vintage photograph",
    "Enhance details and add cinematic color grading",
]
```

The script will cycle through these prompts across all concurrent requests, ensuring variety while maintaining testability.

### Inference Parameters

```python
NUM_INFERENCE_STEPS = 8      # 4 or 8 with Lightning, 40 without
TRUE_CFG_SCALE = 4.0         # Guidance scale (higher = more prompt adherence)
```

## Advanced Usage

### Test Different Batch Sizes

Update your RunPod endpoint environment variables and re-test:

```bash
# Test with different batch sizes
BATCH_SIZE=2 BATCH_TIMEOUT_MS=50   # Small batches
BATCH_SIZE=4 BATCH_TIMEOUT_MS=40   # Balanced (recommended)
BATCH_SIZE=8 BATCH_TIMEOUT_MS=100  # Large batches
```

### Load Testing

For serious load testing, use the script with higher concurrency:

```python
CONCURRENT_REQUESTS = 100  # Heavy concurrent load
```

Watch your RunPod GPU utilization to ensure batching is effective.

### Custom Images

Replace the `IMAGE_URL` with your own test image:

```python
IMAGE_URL = "https://your-domain.com/your-test-image.jpg"
```

## Troubleshooting

### Script won't run

Install dependencies:
```bash
pip install aiohttp
```

### Connection errors

Check your endpoint URL format:
```
Correct: https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync
Wrong:   https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run
```

### Authentication errors

Make sure your API key is set:
```python
API_KEY = "your-actual-runpod-api-key"
```

Or use environment variable:
```bash
export RUNPOD_API_KEY="your-actual-runpod-api-key"
```

Then in script:
```python
import os
API_KEY = os.environ.get("RUNPOD_API_KEY", "")
```

## Next Steps

After confirming batching works:

1. **Optimize batch size**: Test different `BATCH_SIZE` values
2. **Tune timeout**: Adjust `BATCH_TIMEOUT_MS` based on arrival patterns
3. **Monitor production**: Set up logging to track batch efficiency
4. **Scale up**: Add more instances if single-instance batching isn't enough

Happy testing! 🚀

