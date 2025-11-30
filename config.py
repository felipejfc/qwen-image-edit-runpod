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

# Queue worker mode - for static pod deployment
QUEUE_WORKER_MODE = bool(int(os.environ.get("QUEUE_WORKER_MODE", "0")))

# Redis configuration (for queue worker mode)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
REDIS_QUEUE_NAME = os.environ.get("REDIS_QUEUE_NAME", "queue:runpod_generation")
REDIS_BRPOP_TIMEOUT = int(os.environ.get("REDIS_BRPOP_TIMEOUT", "5"))

# AWS configuration (for queue worker mode)
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# DynamoDB configuration
DYNAMODB_JOBS_TABLE = os.environ.get("DYNAMODB_JOBS_TABLE", "jobs")
DYNAMODB_ENDPOINT_URL = os.environ.get("DYNAMODB_ENDPOINT_URL", None)  # For local testing

# S3 configuration
S3_BUCKET_NAME = os.environ.get("BUCKET_NAME", "")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", None)  # For R2/MinIO
S3_PUBLIC_URL = os.environ.get("S3_PUBLIC_URL", "")  # Public URL prefix for generated images

# Worker configuration
WORKER_CONCURRENCY = int(os.environ.get("WORKER_CONCURRENCY", "1"))
STALE_JOB_TIMEOUT_SECONDS = int(os.environ.get("STALE_JOB_TIMEOUT", "60"))

# Pipeline configuration (for GPU optimization)
# Prefetch thread will keep this many jobs ready with downloaded images
PREFETCH_DEPTH = int(os.environ.get("PREFETCH_DEPTH", "10"))
# Maximum jobs waiting in the ready queue (with images in memory)
READY_QUEUE_SIZE = int(os.environ.get("READY_QUEUE_SIZE", "10"))
# Maximum results waiting to be uploaded
RESULT_QUEUE_SIZE = int(os.environ.get("RESULT_QUEUE_SIZE", "5"))
# Timeout for internal queue operations (seconds)
INTERNAL_QUEUE_TIMEOUT = float(os.environ.get("INTERNAL_QUEUE_TIMEOUT", "1.0"))

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

if QUEUE_WORKER_MODE:
    print(f"[Config] QUEUE_WORKER_MODE enabled")
    print(f"[Config] REDIS_URL={REDIS_URL}, QUEUE={REDIS_QUEUE_NAME}")
    print(f"[Config] DYNAMODB_JOBS_TABLE={DYNAMODB_JOBS_TABLE}")
    print(f"[Config] S3_BUCKET_NAME={S3_BUCKET_NAME}")
    print(f"[Config] WORKER_CONCURRENCY={WORKER_CONCURRENCY}")
    print(f"[Config] Pipeline: PREFETCH_DEPTH={PREFETCH_DEPTH}, READY_QUEUE={READY_QUEUE_SIZE}, RESULT_QUEUE={RESULT_QUEUE_SIZE}")

