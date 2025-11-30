"""Main entry point and RunPod handler."""
import os
import time
from typing import Dict, Any
import runpod

from config import STANDALONE_MODE, QUEUE_WORKER_MODE
from model_loader import load_pipeline
from inference import run_inference
from utils import handle_error

# Load pipeline at module level
pipe = load_pipeline()


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod serverless handler."""
    try:
        return run_inference(pipe, event)
    except Exception as e:
        return handle_error(e)


if __name__ == "__main__":
    if QUEUE_WORKER_MODE:
        print("[Mode] Queue worker (static pod)")
        from queue_worker import start_queue_worker
        start_queue_worker(handler)
    elif STANDALONE_MODE:
        print("[Mode] FastAPI standalone server")
        from server import start_server
        start_server(handler)
    elif os.getenv("RUNPOD_SERVERLESS", ""):
        print("[Mode] RunPod serverless")
        runpod.serverless.start({"handler": handler})
    else:
        print("[Mode] Local mode - set RUNPOD_SERVERLESS=1, STANDALONE_MODE=1, or QUEUE_WORKER_MODE=1")
        print("Keeping alive…")
        while True:
            time.sleep(3600)
