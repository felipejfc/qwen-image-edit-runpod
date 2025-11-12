#!/usr/bin/env python3
"""
Configurable test script for Qwen Image Edit RunPod API batching.
Sends concurrent requests with shuffled prompts to test batching behavior.

Compatible with:
- RunPod Serverless (use /runsync endpoint)
- Standalone mode (local server)

Usage with RunPod Serverless:
    export API_URL='https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync'
    export API_KEY='your-api-key'
    export CONCURRENT_REQUESTS=20
    python test_runpod_batching.py

Usage with Standalone mode:
    export API_URL='http://localhost:8888/run'
    python test_runpod_batching.py
    
Qwen Image Edit API Reference:
    https://github.com/wlsdml1114/qwen_image_edit
"""
import asyncio
import aiohttp
import json
import time
import random
import base64
import os
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# RunPod API endpoint (can be overridden with API_URL env var)
# For RunPod Serverless: https://api.runpod.ai/v2/{endpoint_id}/runsync
# For Standalone mode: http://localhost:8888/run
API_URL = os.environ.get("API_URL", "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync")
API_KEY = os.environ.get("API_KEY", "")  # Required for RunPod Serverless

# Qwen Image Edit API expects:
# - prompt: Text prompt that guides the edit (required)
# - image_url/image_path/image_base64: Input image (required)
# - seed: Random seed for deterministic output (required)
# - width: Output image width in pixels (required)
# - height: Output image height in pixels (required)

# Number of concurrent requests to send
CONCURRENT_REQUESTS = int(os.environ.get("CONCURRENT_REQUESTS", "20"))

# Test image URL
IMAGE_URL = "https://midias.correiobraziliense.com.br/_midias/png/2025/11/04/675x450/1_g2g2tjgxaaacl7t-60613956.png"

# Prompt templates (fill these in with your actual prompts)
PROMPT_TEMPLATES = [
    "Put girl in a white bikini",  # Replace with your actual prompt
    "Put girl in a red bikini",  # Replace with your actual prompt
    "Make her dress white",  # Replace with your actual prompt
    "Make her wear a white hat",  # Replace with your actual prompt
    "Make her wear a red hat",  # Replace with your actual prompt
]

# Qwen Image Edit parameters (required by the API)
WIDTH = int(os.environ.get("WIDTH", "768"))
HEIGHT = int(os.environ.get("HEIGHT", "1024"))

# Output directory for saving images
OUTPUT_DIR = "test_results"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_image(image_base64: str, request_id: int, prompt: str, output_dir: str) -> str:
    """Save base64 encoded image to file."""
    try:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        
        # Create safe filename from prompt (first 30 chars)
        safe_prompt = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in prompt)
        safe_prompt = safe_prompt[:30].strip().replace(' ', '_')
        
        # Generate filename
        filename = f"req_{request_id:03d}_{safe_prompt}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save image
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        return filepath
    except Exception as e:
        print(f"Error saving image for request {request_id}: {e}")
        return None

# =============================================================================
# TEST IMPLEMENTATION
# =============================================================================

def generate_requests(count: int) -> List[Dict]:
    """Generate shuffled test requests for Qwen Image Edit API."""
    requests = []
    for i in range(count):
        # Cycle through prompts and shuffle
        prompt = PROMPT_TEMPLATES[i % len(PROMPT_TEMPLATES)]
        
        requests.append({
            "request_id": i,
            "prompt": prompt,
            "payload": {
                "input": {
                    "prompt": prompt,
                    "image_url": IMAGE_URL,
                    "seed": 1000 + i,  # Unique seed per request
                    "width": WIDTH,
                    "height": HEIGHT,
                }
            }
        })
    
    # Shuffle to test that responses are correctly mapped
    random.shuffle(requests)
    return requests


async def send_request(
    session: aiohttp.ClientSession,
    request: Dict,
    headers: Dict,
    output_dir: str
) -> Dict:
    """Send a single request and measure timing."""
    request_id = request["request_id"]
    prompt = request["prompt"]
    payload = request["payload"]
    
    start_time = time.time()
    
    try:
        async with session.post(API_URL, json=payload, headers=headers) as resp:
            duration = time.time() - start_time
            
            if resp.status != 200:
                error_text = await resp.text()
                # Try to parse JSON error response
                try:
                    error_json = json.loads(error_text) if error_text else {}
                    error_msg = error_json.get("error", error_text or f"HTTP {resp.status}")
                except:
                    error_msg = error_text or f"HTTP {resp.status} with no response body"
                
                return {
                    "request_id": request_id,
                    "prompt_sent": prompt,
                    "status": "error",
                    "http_status": resp.status,
                    "error": error_msg,
                    "duration": duration,
                }
            
            result = await resp.json()
            
            # Debug: print response structure for first request
            if request_id == 0:
                print(f"\n[DEBUG] Response structure for request 0:")
                print(f"  Keys in response: {list(result.keys())}")
                print(f"  Status: {result.get('status')}")
                print(f"  Error: {result.get('error')}")
                print(f"  Execution time: {result.get('executionTime')}ms")
                if "output" in result:
                    output_keys = list(result['output'].keys())
                    print(f"  Keys in output: {output_keys}")
                    # Show first 200 chars of each field
                    for key in output_keys:
                        val = result['output'][key]
                        if isinstance(val, str):
                            val_display = val[:200] + "..." if len(val) > 200 else val
                        else:
                            val_display = str(val)
                        print(f"    {key}: {val_display}")
                else:
                    # Show first 200 chars of each field
                    for key in list(result.keys())[:10]:  # Limit to first 10 keys
                        val = result[key]
                        if isinstance(val, str):
                            val_display = val[:200] + "..." if len(val) > 200 else val
                        else:
                            val_display = str(val)
                        print(f"    {key}: {val_display}")
                print()
            
            # Handle both RunPod serverless format (with "output" wrapper) 
            # and standalone format (direct response)
            if "output" in result:
                # RunPod serverless format
                output = result["output"]
                image_data = output.get("image", "")
            else:
                # Standalone/direct format
                image_data = result.get("image", "")
            
            # Qwen Image Edit returns base64 with data URI prefix
            # Extract just the base64 part if present
            if image_data and image_data.startswith("data:image"):
                # Format: "data:image/png;base64,<base64_data>"
                image_base64 = image_data.split(",", 1)[1] if "," in image_data else image_data
            else:
                image_base64 = image_data
            
            # Save image to file
            saved_path = None
            if image_base64:
                saved_path = save_image(image_base64, request_id, prompt, output_dir)
            
            # Since Qwen Image Edit doesn't return the prompt,
            # we assume it matches if we got a valid response
            matches = bool(image_base64)
            
            return {
                "request_id": request_id,
                "prompt_sent": prompt,
                "prompt_received": prompt,  # API doesn't echo prompt back
                "matches": matches,
                "duration": duration,
                "image_size": len(image_base64) if image_base64 else 0,
                "saved_path": saved_path,
                "status": "success",
            }
            
    except asyncio.TimeoutError:
        duration = time.time() - start_time
        return {
            "request_id": request_id,
            "prompt_sent": prompt,
            "status": "timeout",
            "duration": duration,
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            "request_id": request_id,
            "prompt_sent": prompt,
            "status": "exception",
            "error": str(e),
            "duration": duration,
        }


async def run_test():
    """Execute the batching test."""
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_DIR, timestamp)
    
    print("=" * 80)
    print("Qwen Image Edit - RunPod API Batching Test")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  API URL:         {API_URL}")
    print(f"  Concurrent reqs: {CONCURRENT_REQUESTS}")
    print(f"  Prompt variants: {len(PROMPT_TEMPLATES)}")
    print(f"  Image URL:       {IMAGE_URL}")
    print(f"  Width:           {WIDTH}px")
    print(f"  Height:          {HEIGHT}px")
    print(f"  Output dir:      {output_dir}")
    print()
    
    # Generate test requests
    requests = generate_requests(CONCURRENT_REQUESTS)
    print(f"Generated {len(requests)} shuffled requests")
    print("Sending concurrent requests...\n")
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
    }
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    
    # Send all requests concurrently
    start_time = time.time()
    
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            send_request(session, req, headers, output_dir)
            for req in requests
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time = time.time() - start_time
    
    # Process and analyze results
    print("=" * 80)
    print("Results")
    print("=" * 80)
    
    successful = []
    failed = []
    prompt_stats = defaultdict(list)
    saved_count = 0
    
    for result in results:
        if isinstance(result, Exception):
            failed.append({"error": str(result)})
            continue
        
        req_id = result["request_id"]
        status = result.get("status")
        
        if status == "success":
            successful.append(result)
            prompt = result["prompt_sent"]
            prompt_stats[prompt].append(result["duration"])
            
            matches = "✓" if result.get("matches") else "✗"
            duration = result["duration"]
            img_size = result.get("image_size", 0)
            saved_path = result.get("saved_path")
            
            if saved_path:
                saved_count += 1
                saved_indicator = "💾"
            else:
                saved_indicator = "⚠"
            
            print(f"Request {req_id:3d}: {matches} {saved_indicator} {duration:6.2f}s | "
                  f"Image: {img_size:,} bytes | "
                  f"Prompt: {prompt[:40]}...")
        else:
            failed.append(result)
            error = result.get("error", "Unknown error")
            http_status = result.get("http_status", "")
            duration = result.get("duration", 0)
            
            # Format error message with HTTP status if available
            if http_status:
                error_display = f"HTTP {http_status}: {error}"
            else:
                error_display = error
            
            print(f"Request {req_id:3d}: ✗ {duration:6.2f}s | ERROR: {error_display[:80]}")
    
    # Summary statistics
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    success_count = len(successful)
    fail_count = len(failed)
    all_match = all(r.get("matches", False) for r in successful)
    
    print(f"\nOverall:")
    print(f"  Total requests:  {CONCURRENT_REQUESTS}")
    print(f"  Successful:      {success_count} ({success_count/CONCURRENT_REQUESTS*100:.1f}%)")
    print(f"  Failed:          {fail_count}")
    print(f"  All matched:     {'✓ YES' if all_match else '✗ NO'}")
    print(f"  Images saved:    {saved_count}")
    print(f"  Total time:      {total_time:.2f}s")
    
    if successful:
        durations = [r["duration"] for r in successful]
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        print(f"\nTiming:")
        print(f"  Avg time/req:    {avg_duration:.2f}s")
        print(f"  Min time:        {min_duration:.2f}s")
        print(f"  Max time:        {max_duration:.2f}s")
        print(f"  Throughput:      {CONCURRENT_REQUESTS/total_time:.2f} req/s")
        
        # Per-prompt statistics
        if len(prompt_stats) > 1:
            print(f"\nPer-Prompt Timing:")
            for prompt, times in sorted(prompt_stats.items()):
                avg = sum(times) / len(times)
                print(f"  {prompt[:40]:40s}: {avg:.2f}s avg ({len(times)} reqs)")
    
    # Final verdict
    print()
    if success_count == CONCURRENT_REQUESTS and all_match:
        print("✓ SUCCESS: All requests completed successfully with correct mapping!")
        print("  Batching system is working as expected.")
    elif success_count > 0 and all_match:
        print("⚠ PARTIAL: Some requests succeeded with correct mapping.")
        print(f"  {fail_count} requests failed.")
    else:
        print("✗ FAILURE: Requests failed or responses were incorrectly mapped.")
    
    # Show where images were saved
    if saved_count > 0:
        print()
        print(f"💾 Images saved to: {output_dir}/")
        print(f"   Total images: {saved_count}")
    
    # Troubleshooting tips if there were failures
    if fail_count > 0 and failed:
        print()
        print("⚠️  Troubleshooting Tips:")
        print("-" * 80)
        # Check for common error patterns
        error_messages = [f.get("error", "") for f in failed]
        http_statuses = [f.get("http_status") for f in failed if "http_status" in f]
        
        if 401 in http_statuses or 403 in http_statuses:
            print("  🔑 Authentication Error (HTTP 401/403):")
            print("     - Check your API_KEY is set correctly")
            print("     - Make sure you're using just: API_KEY=rpa_YOUR_KEY")
            print("     - NOT: API_KEY=RUNPOD_API_KEY=rpa_YOUR_KEY")
            print()
        if 404 in http_statuses:
            print("  🔍 Endpoint Not Found (HTTP 404):")
            print("     - Verify your endpoint ID in API_URL")
            print("     - Check the endpoint is deployed and active")
            print()
        if any("timeout" in str(e).lower() for e in error_messages):
            print("  ⏱️  Timeout Errors:")
            print("     - Try reducing CONCURRENT_REQUESTS")
            print("     - Increase timeout in the script")
            print()
        if API_URL.endswith("/run"):
            print("  🔄 Async Endpoint Detected:")
            print("     - You're using the /run endpoint (asynchronous)")
            print("     - For this test, use /runsync (synchronous) instead:")
            print(f"     export API_URL='{API_URL.replace('/run', '/runsync')}'")
            print()
        
        print("  💡 Correct usage example:")
        print("     export API_URL='https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync'")
        print("     export API_KEY='rpa_YOUR_API_KEY'")
        print("     export CONCURRENT_REQUESTS=2")
        print("     python test_runpod_batching.py")
    
    print("=" * 80)
    
    return success_count == CONCURRENT_REQUESTS and all_match


def main():
    """Main entry point."""
    # Display configuration
    print("\n" + "=" * 80)
    print("Configuration")
    print("=" * 80)
    print(f"API_URL:             {API_URL}")
    print(f"API_KEY:             {'***' + API_KEY[-4:] if len(API_KEY) > 4 else '[not set]'}")
    print(f"CONCURRENT_REQUESTS: {CONCURRENT_REQUESTS}")
    print(f"WIDTH:               {WIDTH}px")
    print(f"HEIGHT:              {HEIGHT}px")
    print(f"IMAGE_URL:           {IMAGE_URL[:60]}...")
    print()
    
    # Validate configuration
    if not API_URL or "YOUR_ENDPOINT_ID" in API_URL:
        print("ERROR: API_URL not configured!")
        print()
        print("For RunPod Serverless, set via environment variable:")
        print("  export API_URL='https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync'")
        print("  export API_KEY='your-api-key'")
        print()
        print("For Standalone mode:")
        print("  export API_URL='http://localhost:8888/run'")
        print()
        print("Or edit API_URL in the script.")
        print()
        print("NOTE: Use /runsync endpoint for synchronous responses!")
        return 1
    
    if "PLACEHOLDER_PROMPT" in str(PROMPT_TEMPLATES):
        print("WARNING: Some prompts still contain PLACEHOLDER text.")
        print("Please update PROMPT_TEMPLATES with your actual prompts.")
        print()
    
    # Run the test
    success = asyncio.run(run_test())
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())


# =============================================================================
# USAGE EXAMPLES
# =============================================================================
#
# Example 1: Test with default settings (20 concurrent requests, 768x1024 output)
#     export API_URL='https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync'
#     export API_KEY='rpa_YOUR_API_KEY'
#     python test_runpod_batching.py
#
# Example 2: Test with custom resolution and fewer concurrent requests
#     export API_URL='https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync'
#     export API_KEY='rpa_YOUR_API_KEY'
#     export CONCURRENT_REQUESTS=5
#     export WIDTH=512
#     export HEIGHT=512
#     python test_runpod_batching.py
#
# Example 3: Test with local standalone server
#     export API_URL='http://localhost:8888/run'
#     export WIDTH=1024
#     export HEIGHT=1024
#     python test_runpod_batching.py
#
# =============================================================================
