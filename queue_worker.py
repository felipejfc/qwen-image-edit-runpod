"""
Queue worker for processing jobs from Redis queue.

Uses a 3-stage pipeline architecture to maximize GPU utilization:
1. Prefetch Thread: Pops jobs from Redis, downloads images from S3
2. Inference Thread: Runs GPU inference (never blocks on I/O)
3. Upload Thread: Uploads results to S3, updates DynamoDB

This allows GPU to process job N while prefetch downloads N+1 and upload handles N-1.
"""
import base64
import json
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

import redis

from config import (
    REDIS_URL,
    REDIS_QUEUE_NAME,
    REDIS_BRPOP_TIMEOUT,
    STALE_JOB_TIMEOUT_SECONDS,
    PREFETCH_DEPTH,
    READY_QUEUE_SIZE,
    RESULT_QUEUE_SIZE,
    INTERNAL_QUEUE_TIMEOUT,
)
from job_store import (
    get_job,
    update_job_running,
    update_job_succeeded,
    update_job_failed,
    update_job_canceled,
    JobStatus,
)
from storage import download_image, upload_image

# Global state
_redis_client: Optional[redis.Redis] = None
_redis_pubsub_client: Optional[redis.Redis] = None
_shutdown_event = threading.Event()


@dataclass
class PreparedJob:
    """A job that has been fetched and is ready for inference."""
    job: JobStatus
    image_bytes: bytes
    content_type: str


@dataclass
class CompletedJob:
    """A job that has completed inference and is ready for upload."""
    job: JobStatus
    result_bytes: bytes
    success: bool
    error_code: Optional[str] = None
    error_msg: Optional[str] = None


def get_redis_client() -> redis.Redis:
    """Get or create the Redis client singleton for queue operations."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_timeout=30,
            socket_connect_timeout=10,
            retry_on_timeout=True,
        )
    return _redis_client


def get_redis_pubsub_client() -> redis.Redis:
    """Get or create the Redis client singleton for pub/sub operations."""
    global _redis_pubsub_client
    if _redis_pubsub_client is None:
        _redis_pubsub_client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_timeout=30,
            socket_connect_timeout=10,
        )
    return _redis_pubsub_client


def publish_job_status(job_id: str, status: str, output_url: Optional[str] = None,
                       error_code: Optional[str] = None, error_msg: Optional[str] = None):
    """Publish job status update to Redis pub/sub for SSE notifications."""
    try:
        client = get_redis_pubsub_client()
        payload = {
            "job_id": job_id,
            "status": status,
        }
        if output_url:
            payload["output_url"] = output_url
        if error_code:
            payload["error_code"] = error_code
        if error_msg:
            payload["error_msg"] = error_msg
        
        channel = f"job:{job_id}"
        client.publish(channel, json.dumps(payload))
    except Exception as e:
        print(f"[Worker] Failed to publish status update: {e}")


def is_job_stale(job: JobStatus) -> bool:
    """Check if a job is too old to process."""
    if not job.created_at:
        return False
    now = datetime.now(timezone.utc)
    created = job.created_at if job.created_at.tzinfo else job.created_at.replace(tzinfo=timezone.utc)
    age = now - created
    return age > timedelta(seconds=STALE_JOB_TIMEOUT_SECONDS)


def format_prompt(job: JobStatus) -> str:
    """Format the prompt, applying style formatting if needed."""
    final_prompt = job.prompt.strip()
    if job.style_code and job.style_code.strip():
        final_prompt = f"Change the clothing style to: {job.prompt}, keep all the rest of the image consistent"
    return final_prompt


# =============================================================================
# STAGE 1: PREFETCH THREAD
# =============================================================================

def prefetch_worker(ready_queue: queue.Queue, worker_id: int = 0):
    """
    Prefetch thread: pops jobs from Redis, downloads images, queues for inference.
    
    This thread handles all the I/O before inference:
    - BRPOP from Redis queue
    - Get job metadata from DynamoDB
    - Download source image from S3
    - Mark job as running
    - Put prepared job into ready_queue
    """
    print(f"[Prefetch-{worker_id}] Starting prefetch worker")
    client = get_redis_client()
    
    while not _shutdown_event.is_set():
        try:
            # Check if ready queue has space (don't prefetch too far ahead)
            if ready_queue.qsize() >= PREFETCH_DEPTH:
                time.sleep(0.1)
                continue
            
            # Pop job from Redis queue
            result = client.brpop(REDIS_QUEUE_NAME, timeout=REDIS_BRPOP_TIMEOUT)
            if result is None:
                continue
            
            _, job_id = result
            print(f"[Prefetch-{worker_id}] Fetching job {job_id}")
            
            # Get job details from DynamoDB
            job = get_job(job_id)
            if not job:
                print(f"[Prefetch-{worker_id}] Job {job_id} not found in DynamoDB, skipping")
                continue
            
            # Check if job is stale
            if is_job_stale(job):
                print(f"[Prefetch-{worker_id}] Job {job_id} is stale, canceling")
                update_job_canceled(job_id, "stale_job", f"Job older than {STALE_JOB_TIMEOUT_SECONDS}s")
                publish_job_status(job_id, "canceled", error_code="stale_job")
                continue
            
            # Download source image from S3
            try:
                image_bytes, content_type = download_image(job.source_key)
                print(f"[Prefetch-{worker_id}] Downloaded {len(image_bytes)} bytes for job {job_id}")
            except Exception as e:
                error_msg = f"Failed to download source image: {e}"
                print(f"[Prefetch-{worker_id}] {error_msg}")
                update_job_failed(job_id, "download_failed", error_msg)
                publish_job_status(job_id, "failed", error_code="download_failed", error_msg=error_msg)
                continue
            
            # Mark job as running
            update_job_running(job_id)
            publish_job_status(job_id, "running")
            
            # Put prepared job into ready queue
            prepared = PreparedJob(job=job, image_bytes=image_bytes, content_type=content_type)
            
            # Use timeout to allow checking shutdown event
            while not _shutdown_event.is_set():
                try:
                    ready_queue.put(prepared, timeout=INTERNAL_QUEUE_TIMEOUT)
                    break
                except queue.Full:
                    continue
            
        except redis.ConnectionError as e:
            print(f"[Prefetch-{worker_id}] Redis connection error: {e}")
            time.sleep(5)
        except Exception as e:
            print(f"[Prefetch-{worker_id}] Error in prefetch loop: {e}")
            time.sleep(1)
    
    print(f"[Prefetch-{worker_id}] Prefetch worker stopped")


# =============================================================================
# STAGE 2: INFERENCE THREAD
# =============================================================================

def inference_worker(
    handler_func: Callable[[Dict[str, Any]], Dict[str, Any]],
    ready_queue: queue.Queue,
    result_queue: queue.Queue,
    worker_id: int = 0
):
    """
    Inference thread: takes prepared jobs, runs GPU inference, queues results.
    
    This thread should NEVER block on I/O - it only does GPU work.
    - Get prepared job from ready_queue (image already downloaded)
    - Run inference
    - Put result into result_queue (upload happens elsewhere)
    """
    print(f"[Inference-{worker_id}] Starting inference worker")
    
    while not _shutdown_event.is_set():
        try:
            # Get next prepared job (with timeout to check shutdown)
            try:
                prepared = ready_queue.get(timeout=INTERNAL_QUEUE_TIMEOUT)
            except queue.Empty:
                continue
            
            job = prepared.job
            print(f"[Inference-{worker_id}] Processing job {job.id}")
            
            # Prepare inference event
            final_prompt = format_prompt(job)
            event = {
                "input": {
                    "prompt": final_prompt,
                    "image_base64": base64.b64encode(prepared.image_bytes).decode("utf-8"),
                }
            }
            
            # Run inference
            try:
                result = handler_func(event)
            except Exception as e:
                error_msg = f"Image generation failed: {e}"
                print(f"[Inference-{worker_id}] {error_msg}")
                completed = CompletedJob(
                    job=job,
                    result_bytes=b"",
                    success=False,
                    error_code="generation_failed",
                    error_msg=error_msg,
                )
                _put_result(result_queue, completed)
                continue
            
            # Check for error in result
            if "error" in result:
                error_msg = f"Image generation failed: {result.get('error', 'Unknown error')}"
                print(f"[Inference-{worker_id}] {error_msg}")
                completed = CompletedJob(
                    job=job,
                    result_bytes=b"",
                    success=False,
                    error_code="generation_failed",
                    error_msg=error_msg,
                )
                _put_result(result_queue, completed)
                continue
            
            # Decode result image
            result_base64 = result.get("image", "")
            if not result_base64:
                error_msg = "Image generation returned empty result"
                print(f"[Inference-{worker_id}] {error_msg}")
                completed = CompletedJob(
                    job=job,
                    result_bytes=b"",
                    success=False,
                    error_code="generation_failed",
                    error_msg=error_msg,
                )
                _put_result(result_queue, completed)
                continue
            
            result_bytes = base64.b64decode(result_base64)
            print(f"[Inference-{worker_id}] Completed inference for job {job.id}, {len(result_bytes)} bytes")
            
            # Queue successful result for upload
            completed = CompletedJob(job=job, result_bytes=result_bytes, success=True)
            _put_result(result_queue, completed)
            
        except Exception as e:
            print(f"[Inference-{worker_id}] Error in inference loop: {e}")
            time.sleep(0.1)
    
    print(f"[Inference-{worker_id}] Inference worker stopped")


def _put_result(result_queue: queue.Queue, completed: CompletedJob):
    """Put a result into the result queue, respecting shutdown."""
    while not _shutdown_event.is_set():
        try:
            result_queue.put(completed, timeout=INTERNAL_QUEUE_TIMEOUT)
            return
        except queue.Full:
            continue


# =============================================================================
# STAGE 3: UPLOAD THREAD
# =============================================================================

def upload_worker(result_queue: queue.Queue, worker_id: int = 0):
    """
    Upload thread: takes completed jobs, uploads to S3, updates DynamoDB.
    
    This thread handles all the I/O after inference:
    - Get completed job from result_queue
    - Upload result image to S3
    - Update job status in DynamoDB
    - Publish completion status to Redis
    """
    print(f"[Upload-{worker_id}] Starting upload worker")
    
    while not _shutdown_event.is_set():
        try:
            # Get next completed job (with timeout to check shutdown)
            try:
                completed = result_queue.get(timeout=INTERNAL_QUEUE_TIMEOUT)
            except queue.Empty:
                continue
            
            job = completed.job
            
            if not completed.success:
                # Job failed during inference, just update status
                print(f"[Upload-{worker_id}] Recording failure for job {job.id}")
                update_job_failed(job.id, completed.error_code or "unknown", completed.error_msg or "Unknown error")
                publish_job_status(job.id, "failed", error_code=completed.error_code, error_msg=completed.error_msg)
                continue
            
            # Upload result to S3
            output_key = f"gen/{job.user_id}/output/{job.id}.jpg"
            print(f"[Upload-{worker_id}] Uploading result for job {job.id}")
            
            try:
                output_url = upload_image(output_key, completed.result_bytes, "image/jpeg")
            except Exception as e:
                error_msg = f"Failed to upload result: {e}"
                print(f"[Upload-{worker_id}] {error_msg}")
                update_job_failed(job.id, "upload_failed", error_msg)
                publish_job_status(job.id, "failed", error_code="upload_failed", error_msg=error_msg)
                continue
            
            # Update job with success
            update_job_succeeded(job.id, output_key, output_url)
            publish_job_status(job.id, "succeeded", output_url=output_url)
            
            print(f"[Upload-{worker_id}] Successfully completed job {job.id}")
            
        except Exception as e:
            print(f"[Upload-{worker_id}] Error in upload loop: {e}")
            time.sleep(0.1)
    
    print(f"[Upload-{worker_id}] Upload worker stopped")


# =============================================================================
# GRACEFUL SHUTDOWN
# =============================================================================

def drain_queues(ready_queue: queue.Queue, result_queue: queue.Queue, 
                 handler_func: Callable[[Dict[str, Any]], Dict[str, Any]],
                 timeout_seconds: float = 30.0):
    """
    Drain in-flight jobs during graceful shutdown.
    
    - Process any jobs already in ready_queue
    - Upload any results already in result_queue
    """
    print(f"[Shutdown] Draining queues (timeout: {timeout_seconds}s)...")
    start_time = time.time()
    
    # First, drain result queue (already processed jobs)
    while not result_queue.empty() and (time.time() - start_time) < timeout_seconds:
        try:
            completed = result_queue.get_nowait()
            job = completed.job
            
            if completed.success:
                output_key = f"gen/{job.user_id}/output/{job.id}.jpg"
                try:
                    output_url = upload_image(output_key, completed.result_bytes, "image/jpeg")
                    update_job_succeeded(job.id, output_key, output_url)
                    publish_job_status(job.id, "succeeded", output_url=output_url)
                    print(f"[Shutdown] Uploaded pending result for job {job.id}")
                except Exception as e:
                    print(f"[Shutdown] Failed to upload job {job.id}: {e}")
                    update_job_failed(job.id, "upload_failed", str(e))
            else:
                update_job_failed(job.id, completed.error_code or "shutdown", completed.error_msg or "Worker shutdown")
        except queue.Empty:
            break
    
    # Then, process any jobs in ready queue
    while not ready_queue.empty() and (time.time() - start_time) < timeout_seconds:
        try:
            prepared = ready_queue.get_nowait()
            job = prepared.job
            print(f"[Shutdown] Processing ready job {job.id}")
            
            # Run inference
            final_prompt = format_prompt(job)
            event = {
                "input": {
                    "prompt": final_prompt,
                    "image_base64": base64.b64encode(prepared.image_bytes).decode("utf-8"),
                }
            }
            
            try:
                result = handler_func(event)
                if "error" not in result and result.get("image"):
                    result_bytes = base64.b64decode(result["image"])
                    output_key = f"gen/{job.user_id}/output/{job.id}.jpg"
                    output_url = upload_image(output_key, result_bytes, "image/jpeg")
                    update_job_succeeded(job.id, output_key, output_url)
                    publish_job_status(job.id, "succeeded", output_url=output_url)
                    print(f"[Shutdown] Completed job {job.id}")
                else:
                    update_job_failed(job.id, "generation_failed", "Failed during shutdown")
            except Exception as e:
                print(f"[Shutdown] Failed to process job {job.id}: {e}")
                update_job_failed(job.id, "shutdown_error", str(e))
        except queue.Empty:
            break
    
    elapsed = time.time() - start_time
    print(f"[Shutdown] Queue draining completed in {elapsed:.1f}s")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def start_queue_worker(handler_func: Callable[[Dict[str, Any]], Dict[str, Any]]):
    """
    Start the queue worker with 3-stage pipeline.
    
    Pipeline architecture:
    1. Prefetch Thread(s): Pop jobs from Redis, download images
    2. Inference Thread: Run GPU inference (main thread)
    3. Upload Thread(s): Upload results to S3, update DynamoDB
    """
    print(f"[QueueWorker] Starting 3-stage pipeline worker")
    print(f"[QueueWorker] Redis URL: {REDIS_URL}")
    print(f"[QueueWorker] Queue: {REDIS_QUEUE_NAME}")
    print(f"[QueueWorker] Pipeline config: prefetch_depth={PREFETCH_DEPTH}, ready_queue={READY_QUEUE_SIZE}, result_queue={RESULT_QUEUE_SIZE}")
    
    # Test Redis connection
    try:
        client = get_redis_client()
        client.ping()
        print("[QueueWorker] Redis connection successful")
    except Exception as e:
        print(f"[QueueWorker] Failed to connect to Redis: {e}")
        sys.exit(1)
    
    # Create internal queues
    ready_queue: queue.Queue[PreparedJob] = queue.Queue(maxsize=READY_QUEUE_SIZE)
    result_queue: queue.Queue[CompletedJob] = queue.Queue(maxsize=RESULT_QUEUE_SIZE)
    
    # Track threads
    threads = []
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n[QueueWorker] Received signal {signum}, initiating graceful shutdown...")
        _shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start prefetch thread(s)
    # Usually 1-2 prefetch threads are enough
    num_prefetch = min(PREFETCH_DEPTH, 2)
    for i in range(num_prefetch):
        t = threading.Thread(target=prefetch_worker, args=(ready_queue, i), daemon=True)
        t.start()
        threads.append(("prefetch", t))
    
    # Start upload thread(s)
    # Usually 1-2 upload threads are enough
    num_upload = 2
    for i in range(num_upload):
        t = threading.Thread(target=upload_worker, args=(result_queue, i), daemon=True)
        t.start()
        threads.append(("upload", t))
    
    print(f"[QueueWorker] Started {num_prefetch} prefetch thread(s) and {num_upload} upload thread(s)")
    print(f"[QueueWorker] Running inference on main thread")
    
    # Run inference on main thread (this is where the GPU work happens)
    # This ensures the GPU-heavy work happens on the main thread
    try:
        inference_worker(handler_func, ready_queue, result_queue, worker_id=0)
    except Exception as e:
        print(f"[QueueWorker] Inference worker error: {e}")
        _shutdown_event.set()
    
    # Graceful shutdown - drain queues
    print("[QueueWorker] Draining in-flight jobs...")
    drain_queues(ready_queue, result_queue, handler_func, timeout_seconds=60.0)
    
    # Wait for threads to finish
    print("[QueueWorker] Waiting for worker threads to finish...")
    for name, t in threads:
        t.join(timeout=5.0)
        if t.is_alive():
            print(f"[QueueWorker] Warning: {name} thread did not stop cleanly")
    
    print("[QueueWorker] Queue worker stopped")


def get_queue_depth() -> int:
    """Get the current queue depth."""
    try:
        client = get_redis_client()
        return client.llen(REDIS_QUEUE_NAME)
    except Exception:
        return -1


# =============================================================================
# LEGACY SINGLE-THREADED MODE (for testing/debugging)
# =============================================================================

def process_job(handler_func: Callable[[Dict[str, Any]], Dict[str, Any]], job_id: str) -> bool:
    """
    Process a single job (legacy single-threaded mode for testing).
    
    This is the original implementation, kept for testing and debugging.
    """
    print(f"[Worker] Processing job {job_id}")
    
    try:
        job = get_job(job_id)
        if not job:
            print(f"[Worker] Job {job_id} not found")
            return False
        
        if is_job_stale(job):
            print(f"[Worker] Job {job_id} is stale, canceling")
            update_job_canceled(job_id, "stale_job", f"Job older than {STALE_JOB_TIMEOUT_SECONDS}s")
            publish_job_status(job_id, "canceled", error_code="stale_job")
            return False
        
        update_job_running(job_id)
        publish_job_status(job_id, "running")
        
        try:
            image_bytes, content_type = download_image(job.source_key)
        except Exception as e:
            error_msg = f"Failed to download source image: {e}"
            update_job_failed(job_id, "download_failed", error_msg)
            publish_job_status(job_id, "failed", error_code="download_failed", error_msg=error_msg)
            return False
        
        final_prompt = format_prompt(job)
        event = {
            "input": {
                "prompt": final_prompt,
                "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
            }
        }
        
        try:
            result = handler_func(event)
        except Exception as e:
            error_msg = f"Image generation failed: {e}"
            update_job_failed(job_id, "generation_failed", error_msg)
            publish_job_status(job_id, "failed", error_code="generation_failed", error_msg=error_msg)
            return False
        
        if "error" in result:
            error_msg = f"Image generation failed: {result.get('error', 'Unknown error')}"
            update_job_failed(job_id, "generation_failed", error_msg)
            publish_job_status(job_id, "failed", error_code="generation_failed", error_msg=error_msg)
            return False
        
        result_base64 = result.get("image", "")
        if not result_base64:
            error_msg = "Image generation returned empty result"
            update_job_failed(job_id, "generation_failed", error_msg)
            publish_job_status(job_id, "failed", error_code="generation_failed", error_msg=error_msg)
            return False
        
        result_bytes = base64.b64decode(result_base64)
        output_key = f"gen/{job.user_id}/output/{job_id}.jpg"
        
        try:
            output_url = upload_image(output_key, result_bytes, "image/jpeg")
        except Exception as e:
            error_msg = f"Failed to upload result: {e}"
            update_job_failed(job_id, "upload_failed", error_msg)
            publish_job_status(job_id, "failed", error_code="upload_failed", error_msg=error_msg)
            return False
        
        update_job_succeeded(job_id, output_key, output_url)
        publish_job_status(job_id, "succeeded", output_url=output_url)
        
        print(f"[Worker] Successfully completed job {job_id}")
        return True
        
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        try:
            update_job_failed(job_id, "internal_error", error_msg)
            publish_job_status(job_id, "failed", error_code="internal_error", error_msg=error_msg)
        except Exception:
            pass
        return False
