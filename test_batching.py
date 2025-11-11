#!/usr/bin/env python3
"""
Test script to verify batching and per-request response mapping.
This sends multiple concurrent requests and verifies each gets the correct result.
"""
import asyncio
import aiohttp
import json
import time
from typing import List, Dict

API_URL = "http://localhost:8000"

# Test with different prompts to verify correct mapping
TEST_REQUESTS = [
    {
        "input": {
            "prompt": f"Test request {i}: Make it style {i}",
            "image_url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png",
            "num_inference_steps": 8,
            "seed": 1000 + i,  # Unique seed per request
        }
    }
    for i in range(10)  # Send 10 concurrent requests
]


async def send_request(session: aiohttp.ClientSession, request: Dict, request_id: int) -> Dict:
    """Send a single request and track timing."""
    start = time.time()
    prompt = request["input"]["prompt"]
    
    try:
        async with session.post(f"{API_URL}/run", json=request) as resp:
            if resp.status != 200:
                error = await resp.text()
                return {
                    "request_id": request_id,
                    "prompt_sent": prompt,
                    "status": "error",
                    "error": error,
                    "duration": time.time() - start,
                }
            
            result = await resp.json()
            duration = time.time() - start
            
            # Verify prompt matches
            prompt_returned = result.get("prompt", "")
            matches = prompt == prompt_returned
            
            return {
                "request_id": request_id,
                "prompt_sent": prompt,
                "prompt_received": prompt_returned,
                "matches": matches,
                "duration": duration,
                "image_size": len(result.get("image", "")),
                "steps": result.get("num_inference_steps"),
            }
    except Exception as e:
        return {
            "request_id": request_id,
            "prompt_sent": prompt,
            "status": "exception",
            "error": str(e),
            "duration": time.time() - start,
        }


async def test_concurrent_requests():
    """Send multiple concurrent requests and verify responses."""
    print("=" * 80)
    print("Testing concurrent request batching and response mapping")
    print("=" * 80)
    print(f"\nSending {len(TEST_REQUESTS)} concurrent requests...\n")
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        # Send all requests concurrently
        tasks = [
            send_request(session, req, i)
            for i, req in enumerate(TEST_REQUESTS)
        ]
        results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    # Analyze results
    print("Results:")
    print("-" * 80)
    
    all_match = True
    total_duration = 0
    successful = 0
    
    for result in results:
        req_id = result["request_id"]
        matches = result.get("matches", False)
        duration = result["duration"]
        status = "✓" if matches else "✗"
        
        print(f"Request {req_id:2d}: {status} {duration:.2f}s", end="")
        
        if matches:
            successful += 1
            total_duration += duration
            img_size = result["image_size"]
            print(f" | Image: {img_size:,} bytes")
        else:
            all_match = False
            print(f" | ERROR: {result.get('error', 'Prompt mismatch')}")
    
    print("-" * 80)
    print(f"\nSummary:")
    print(f"  Total requests:  {len(TEST_REQUESTS)}")
    print(f"  Successful:      {successful}")
    print(f"  Failed:          {len(TEST_REQUESTS) - successful}")
    print(f"  All matched:     {'✓ YES' if all_match else '✗ NO'}")
    print(f"  Total time:      {total_time:.2f}s")
    print(f"  Avg time/req:    {total_duration/successful:.2f}s" if successful > 0 else "  N/A")
    print(f"  Throughput:      {len(TEST_REQUESTS)/total_time:.2f} req/s")
    
    if all_match and successful == len(TEST_REQUESTS):
        print("\n✓ SUCCESS: All requests received correct responses!")
        print("  Batching and per-request response mapping is working correctly.")
    else:
        print("\n✗ FAILURE: Some requests did not receive correct responses!")
        print("  Check the server logs and results above.")
    
    print("=" * 80)
    
    return all_match and successful == len(TEST_REQUESTS)


async def test_health():
    """Test health endpoint."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_URL}/health") as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"Health check: {result}")
                    return True
                else:
                    print(f"Health check failed: {resp.status}")
                    return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False


async def main():
    print("\nChecking API health...")
    if not await test_health():
        print("\n✗ API not reachable. Make sure the server is running:")
        print("  STANDALONE_MODE=1 BATCH_SIZE=4 BATCH_TIMEOUT_MS=40 python handler.py")
        return
    
    print("\n")
    success = await test_concurrent_requests()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)

