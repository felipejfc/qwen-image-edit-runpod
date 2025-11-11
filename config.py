"""Configuration and environment variables."""
import os
import torch

# Model configuration
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-Edit-2509")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/qwen_image_edit_2509")
LOCAL_ONLY = bool(int(os.environ.get("LOCAL_ONLY", "0")))

# Precision mode: bf16 | 8bit | 4bit
PRECISION_MODE = os.environ.get("PRECISION_MODE", "bf16").strip().lower()
if PRECISION_MODE not in {"bf16", "8bit", "4bit"}:
    raise ValueError("PRECISION_MODE must be one of: bf16 | 8bit | 4bit")

# Quantize components: "text" or "all"
QUANTIZE_COMPONENTS = os.environ.get("QUANTIZE_COMPONENTS", "all").strip().lower()
if QUANTIZE_COMPONENTS not in {"text", "all"}:
    raise ValueError("QUANTIZE_COMPONENTS must be 'text' or 'all'")

# bitsandbytes settings
INT8_CPU_OFFLOAD = bool(int(os.environ.get("INT8_CPU_OFFLOAD", "0")))
BNB_COMPUTE = os.environ.get("BNB_COMPUTE", "bf16").strip().lower()
if BNB_COMPUTE not in {"fp16", "bf16"}:
    raise ValueError("BNB_COMPUTE must be 'fp16' or 'bf16'")

# LoRA configuration
USE_LIGHTNING_LORA = os.environ.get("USE_LIGHTNING_LORA", "true").lower() == "true"
LIGHTNING_STEPS = int(os.environ.get("LIGHTNING_STEPS", "8"))

# CFG scale default
DEFAULT_CFG_SCALE = float(os.environ.get("DEFAULT_CFG_SCALE", "4.0"))

# Memory management
GPU_MAX_GIB = os.environ.get("GPU_MAX_GIB", None)
MAX_MEMORY = {0: str(GPU_MAX_GIB), "cpu": "128GiB"} if GPU_MAX_GIB else None

# Standalone mode
STANDALONE_MODE = bool(int(os.environ.get("STANDALONE_MODE", "0")))

# Torch configuration
torch.backends.cuda.matmul.allow_tf32 = True
_bf16_ok = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
PIPELINE_DTYPE = torch.bfloat16 if BNB_COMPUTE == "bf16" and _bf16_ok else torch.float16

if BNB_COMPUTE == "bf16" and not _bf16_ok:
    print("[Config] Warning: GPU reports no BF16 support; falling back to fp16.")
    BNB_COMPUTE = "fp16"

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

print(f"[Config] PRECISION_MODE={PRECISION_MODE}, QUANTIZE_COMPONENTS={QUANTIZE_COMPONENTS}")
print(f"[Config] INT8_CPU_OFFLOAD={INT8_CPU_OFFLOAD}, BNB_COMPUTE={BNB_COMPUTE}")
print(f"[Config] PIPELINE_DTYPE={PIPELINE_DTYPE}, DEFAULT_CFG_SCALE={DEFAULT_CFG_SCALE}")

