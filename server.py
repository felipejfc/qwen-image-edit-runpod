"""FastAPI standalone server."""
import os
import torch
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from utils import handle_error


class RunpodInput(BaseModel):
    prompt: str
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    seed: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    num_inference_steps: Optional[int] = None
    true_cfg_scale: Optional[float] = None
    negative_prompt: Optional[str] = None


class RunpodRequest(BaseModel):
    input: RunpodInput


def create_app(handler_func):
    """Create FastAPI app with the given handler function."""
    app = FastAPI(title="Qwen Image Edit API", version="1.0")
    
    @app.get("/health")
    async def health():
        return {"status": "ready", "gpu_available": torch.cuda.is_available()}
    
    @app.post("/run")
    async def run_inference(request: RunpodRequest):
        try:
            inp = request.input
            if not inp.image_url and not inp.image_base64:
                raise HTTPException(status_code=400, detail="Either image_url or image_base64 required")
            
            event = {"input": {
                "prompt": inp.prompt,
                "image_url": inp.image_url or f"data:image/png;base64,{inp.image_base64}",
            }}
            
            if inp.seed is not None:
                event["input"]["seed"] = inp.seed
            if inp.num_inference_steps is not None:
                event["input"]["num_inference_steps"] = inp.num_inference_steps
            if inp.true_cfg_scale is not None:
                event["input"]["true_cfg_scale"] = inp.true_cfg_scale
            if inp.negative_prompt is not None:
                event["input"]["negative_prompt"] = inp.negative_prompt
            
            result = handler_func(event)
            
            if "error" in result:
                raise HTTPException(status_code=500, detail=result)
            
            return JSONResponse(content=result)
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=handle_error(e))
    
    @app.post("/runsync")
    async def run_sync(request: RunpodRequest):
        """Alias for /run - maintains compatibility with RunPod endpoint patterns"""
        return await run_inference(request)
    
    return app


def start_server(handler_func):
    """Start the FastAPI server."""
    app = create_app(handler_func)
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    workers = int(os.environ.get("API_WORKERS", "1"))
    print(f"[Server] Starting on {host}:{port} (workers={workers})")
    uvicorn.run(app, host=host, port=port, workers=workers, log_level="info")

