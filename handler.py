import os, io, base64, time
from typing import Dict, Any, Optional
from PIL import Image

import torch
import torch.nn as nn
import runpod
from transformers import (
    BitsAndBytesConfig as HF_BNB,
    Qwen2_5_VLForConditionalGeneration,  # correct text-encoder class
)
from diffusers import (
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
    BitsAndBytesConfig as DF_BNB,
)
from diffusers.utils import load_image

# FastAPI imports (conditional)
STANDALONE_MODE = bool(int(os.environ.get("STANDALONE_MODE", "0")))
if STANDALONE_MODE:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn

# ====================
# Env config
# ====================
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-Edit-2509")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/qwen_image_edit_2509")
LOCAL_ONLY = bool(int(os.environ.get("LOCAL_ONLY", "0")))

# Precision: bf16 | 8bit | 4bit
PRECISION_MODE = os.environ.get("PRECISION_MODE", "bf16").strip().lower()
if PRECISION_MODE not in {"bf16", "8bit", "4bit"}:
    raise ValueError("PRECISION_MODE must be one of: bf16 | 8bit | 4bit")

# Which parts to quantize in 8/4-bit: "text" or "all" (quantize both text encoder + transformer, like reference code)
QUANTIZE_COMPONENTS = os.environ.get("QUANTIZE_COMPONENTS", "all").strip().lower()
if QUANTIZE_COMPONENTS not in {"text", "all"}:
    raise ValueError("QUANTIZE_COMPONENTS must be 'text' or 'all'")

# bitsandbytes INT8 CPU offload (ignored for 4-bit UNet — not supported)
INT8_CPU_OFFLOAD = bool(int(os.environ.get("INT8_CPU_OFFLOAD", "0")))

# bnb compute dtype in quant modes.
BNB_COMPUTE = os.environ.get("BNB_COMPUTE", "bf16").strip().lower()
if BNB_COMPUTE not in {"fp16", "bf16"}:
    raise ValueError("BNB_COMPUTE must be 'fp16' or 'bf16'")

# LoRA
USE_LIGHTNING_LORA = os.environ.get("USE_LIGHTNING_LORA", "true").lower() == "true"
LIGHTNING_STEPS = int(os.environ.get("LIGHTNING_STEPS", "8"))  # 4 or 8

# CFG (Classifier-Free Guidance) - controls how strictly the model follows the prompt
# Higher values (7-15) = stricter prompt adherence, Lower values (1-3) = more creative freedom
DEFAULT_CFG_SCALE = float(os.environ.get("DEFAULT_CFG_SCALE", "4.0"))

# Optional placement cap for accelerate-style loaders (string like "30GiB")
GPU_MAX_GIB = os.environ.get("GPU_MAX_GIB", None)
max_memory = None
if GPU_MAX_GIB:
    max_memory = {0: str(GPU_MAX_GIB), "cpu": "128GiB"}

# ====================
# Torch / dtype
# ====================
torch.backends.cuda.matmul.allow_tf32 = True
# BF16 for H100 fast path; align pipeline dtype with bnb compute when offloading
_bf16_ok = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
PIPELINE_DTYPE = torch.bfloat16 if BNB_COMPUTE == "bf16" and _bf16_ok else torch.float16

if BNB_COMPUTE == "bf16" and not _bf16_ok:
    print("[Init] Warning: GPU reports no BF16 support; falling back to fp16 compute for quantized modules.")
    BNB_COMPUTE = "fp16"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

print(f"[Init] PRECISION_MODE={PRECISION_MODE}, QUANTIZE_COMPONENTS={QUANTIZE_COMPONENTS}, "
      f"INT8_CPU_OFFLOAD={INT8_CPU_OFFLOAD}, BNB_COMPUTE={BNB_COMPUTE}")
print(f"[Init] pipeline dtype={PIPELINE_DTYPE}, bf16_supported={torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None}")
print(f"[Init] DEFAULT_CFG_SCALE={DEFAULT_CFG_SCALE}")

# ====================
# BitsAndBytes configs
# ====================
def _bnb_compute_dtype():
    return PIPELINE_DTYPE

def _bnb_cfg_8bit_hf():
    return HF_BNB(load_in_8bit=True)

def _bnb_cfg_4bit_hf():
    return HF_BNB(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_bnb_compute_dtype(),
    )

def _bnb_cfg_8bit_df():
    return DF_BNB(
        load_in_8bit=True,
        llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
    )

def _bnb_cfg_4bit_df():
    return DF_BNB(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_bnb_compute_dtype(),
        llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
    )

# ====================
# Build / load pipeline
# ====================
print(f"[Init] Loading components: {MODEL_ID}")

pipe = None
text_encoder = None
transformer = None

# Helper: load quantized components (can force device + max_memory)
def _load_text_encoder_quant(prec_mode: str):
    q = _bnb_cfg_8bit_hf() if prec_mode == "8bit" else _bnb_cfg_4bit_hf()
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder",
        quantization_config=q,
        torch_dtype=PIPELINE_DTYPE,  # Critical: match working reference code
        cache_dir=MODEL_DIR,
        local_files_only=LOCAL_ONLY,
        trust_remote_code=True,
        max_memory=max_memory if max_memory else None,
    )

def _load_transformer_quant(prec_mode: str):
    q = _bnb_cfg_8bit_df() if prec_mode == "8bit" else _bnb_cfg_4bit_df()
    return QwenImageTransformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        quantization_config=q,
        torch_dtype=PIPELINE_DTYPE,  # Critical: match working reference code
        cache_dir=MODEL_DIR,
        local_files_only=LOCAL_ONLY,
        max_memory=max_memory if max_memory else None,
    )

# Fast path (H100 etc.): GPU-first assembly
if PRECISION_MODE == "bf16":
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=PIPELINE_DTYPE,
        cache_dir=MODEL_DIR,
        local_files_only=LOCAL_ONLY,
    )
    pipe.to("cuda")

else:
    # Quant path: build quantized parts FIRST, move to CPU, then assemble (matches reference code)
    print(f"[Init] Loading quantized text_encoder ({PRECISION_MODE})...")
    text_encoder = _load_text_encoder_quant(PRECISION_MODE)
    if INT8_CPU_OFFLOAD:
        text_encoder = text_encoder.to("cpu")
    
    print(f"[Init] Loading quantized transformer ({PRECISION_MODE})...")
    if QUANTIZE_COMPONENTS == "all":
        transformer = _load_transformer_quant(PRECISION_MODE)
        if INT8_CPU_OFFLOAD:
            transformer = transformer.to("cpu")
    
    print("[Init] Assembling pipeline...")
    fp_kwargs = dict(
        torch_dtype=PIPELINE_DTYPE,
        cache_dir=MODEL_DIR,
        local_files_only=LOCAL_ONLY,
        text_encoder=text_encoder,
    )
    if transformer is not None:
        fp_kwargs["transformer"] = transformer

    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL_ID, **fp_kwargs)

    # Enable CPU offload (like reference code) to swap quantized components dynamically
    if INT8_CPU_OFFLOAD:
        print("[Init] Enabling model CPU offload...")
        pipe.enable_model_cpu_offload()
    else:
        print("[Init] Moving pipeline to CUDA...")
        pipe.to("cuda")


def _align_bias_dtypes(root: nn.Module, target_dtype: torch.dtype):
    other = torch.float16 if target_dtype == torch.bfloat16 else torch.bfloat16
    for module in root.modules():
        bias = getattr(module, "bias", None)
        if isinstance(bias, nn.Parameter) and bias.dtype == other:
            module.bias = nn.Parameter(bias.data.to(target_dtype), requires_grad=bias.requires_grad)


# Ensure no fp16/bf16 bias mismatches when running quantized modes
if isinstance(pipe, nn.Module):
    _align_bias_dtypes(pipe, PIPELINE_DTYPE)

# Memory savers
pipe.enable_attention_slicing()
if hasattr(pipe, "vae") and pipe.vae is not None:
    if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()
    if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()

# Try xFormers only if installed/matching
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("[Init] xFormers attention enabled.")
except Exception as e:
    print(f"[Init] xFormers not enabled: {e}")

# LoRA (like your snippet)
if USE_LIGHTNING_LORA:
    try:
        print(f"[Init] Loading Lightning LoRA ({LIGHTNING_STEPS} steps)")
        # Try common filenames; adjust if your repo uses different names
        if LIGHTNING_STEPS == 8:
            pipe.load_lora_weights(
                "lightx2v/Qwen-Image-Lightning",
                weight_name="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors",
            )
        elif LIGHTNING_STEPS == 4:
            pipe.load_lora_weights(
                "lightx2v/Qwen-Image-Lightning",
                weight_name="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
            )
        else:
            print("[Init] Unsupported LIGHTNING_STEPS; skipping LoRA.")
            USE_LIGHTNING_LORA = False
        print("[Init] Lightning ready.")
    except Exception as e:
        print(f"[Init] Lightning LoRA load failed: {e}")
        USE_LIGHTNING_LORA = False

print("[Init] Pipeline ready.")

# ====================
# Helpers
# ====================
def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

def _default_steps() -> int:
    return LIGHTNING_STEPS if USE_LIGHTNING_LORA else 40

# ====================
# Direct inference (no batching)
# ====================
def _run_inference(event: Dict[str, Any]) -> Dict[str, Any]:
    inp = event["input"]
    
    # Load and prepare image
    img = load_image(inp["image_url"]).convert("RGB")
    
    # Get parameters
    prompt = inp.get("prompt", "Enhance the image")
    negative_prompt = inp.get("negative_prompt", None)
    steps = int(inp.get("num_inference_steps", _default_steps()))
    cfg = float(inp.get("true_cfg_scale", DEFAULT_CFG_SCALE))
    
    # Prepare generator if seed provided
    generator = None
    if "seed" in inp:
        generator = torch.Generator(device="cuda").manual_seed(int(inp["seed"]))
    
    # Run inference (ensure inputs match pipeline dtype)
    with torch.inference_mode():
        out = pipe(
            image=img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            true_cfg_scale=cfg,
            num_images_per_prompt=1,
            generator=generator,
        )
    
    return {
        "image": _to_b64(out.images[0]),
        "prompt": prompt,
        "num_inference_steps": steps,
        "lightning_enabled": bool(USE_LIGHTNING_LORA),
    }

# ====================
# Error helper
# ====================
def _handle_error(e: Exception) -> Dict[str, Any]:
    info = None
    try:
        info = {
            "device_count": torch.cuda.device_count(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.device_count() else None,
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None,
        }
    except Exception:
        pass
    return {"error": str(e), "gpu_info": info}

# ====================
# Serverless handler
# ====================
def handler(event):
    try:
        return _run_inference(event)
    except Exception as e:
        return _handle_error(e)

# ====================
# FastAPI standalone mode
# ====================
if STANDALONE_MODE:
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

    class RunpodOutput(BaseModel):
        image: str
        prompt: str
        num_inference_steps: int
        lightning_enabled: bool

    app = FastAPI(title="Qwen Image Edit API", version="1.0")

    @app.get("/health")
    async def health():
        return {"status": "ready", "gpu_available": torch.cuda.is_available()}

    @app.post("/run")
    async def run_inference(request: RunpodRequest):
        try:
            # Convert FastAPI request to handler format
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
            
            # Run through batching system
            result = handler(event)
            
            if "error" in result:
                raise HTTPException(status_code=500, detail=result)
            
            return JSONResponse(content=result)
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=_handle_error(e))

    @app.post("/runsync")
    async def run_sync(request: RunpodRequest):
        """Alias for /run - maintains compatibility with RunPod endpoint patterns"""
        return await run_inference(request)

    def start_server():
        host = os.environ.get("API_HOST", "0.0.0.0")
        port = int(os.environ.get("API_PORT", "8000"))
        workers = int(os.environ.get("API_WORKERS", "1"))
        print(f"[FastAPI] Starting server on {host}:{port} (workers={workers})")
        uvicorn.run(app, host=host, port=port, workers=workers, log_level="info")

# ====================
# Entry point
# ====================
if STANDALONE_MODE:
    print("[Mode] FastAPI standalone server")
    start_server()
elif os.getenv("RUNPOD_SERVERLESS", ""):
    print("[Mode] RunPod serverless")
    runpod.serverless.start({"handler": handler})
else:
    print("[Mode] Local mode - set RUNPOD_SERVERLESS=1 for serverless or STANDALONE_MODE=1 for API server")
    print("Keeping alive…")
    while True:
        time.sleep(3600)
