# ---------- Runtime image ----------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf_cache \
    HUGGINGFACE_HUB_CACHE=/opt/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1

RUN pip install \
    "diffusers[torch]>=0.31.0" \
    "transformers>=4.45.0" \
    "accelerate>=0.34.0" \
    "safetensors>=0.4.5" \
    "huggingface_hub[hf-transfer]>=0.25.2" \
    "runpod>=1.6.2" \
    "Pillow>=10.3.0" \
    "requests>=2.31.0"

WORKDIR /app
COPY handler.py /app/handler.py

# Create model cache directory for runtime downloads
RUN mkdir -p /models/qwen_image_edit_2509 /opt/hf_cache

CMD ["python3", "-u", "/app/handler.py"]
