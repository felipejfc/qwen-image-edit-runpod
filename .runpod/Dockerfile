# ---------- Stage 1: download weights without loading the model ----------
    FROM python:3.11-slim AS downloader

    ENV HF_HOME=/tmp/hf_cache \
        HUGGINGFACE_HUB_CACHE=/tmp/hf_cache \
        HF_HUB_ENABLE_HF_TRANSFER=1 \
        PIP_NO_CACHE_DIR=1
    
    RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && \
        rm -rf /var/lib/apt/lists/*
    
    # Only need hub client + hf-transfer here (no torch/cuda!)
    RUN pip install "huggingface_hub[hf-transfer]>=0.25.2"
    
    # Stream the whole repo to /models without loading it into memory
    RUN python - <<'PY'
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="Qwen/Qwen-Image-Edit-2509",
        local_dir="/models/qwen_image_edit_2509",
        local_dir_use_symlinks=False,   # real files (faster start in final image)
        allow_patterns=["*"],           # or narrow if you want to prune
        max_workers=4,                  # parallel file fetch
    )
    PY
    
    # ---------- Stage 2: runtime with CUDA + PyTorch + diffusers ----------
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
    
    # CUDA 12.1 wheels
    RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1+cu121 torchvision==0.19.1+cu121
    
    # Runtime libs
    RUN pip install \
        "diffusers[torch]>=0.31.0" \
        "transformers>=4.45.0" \
        "accelerate>=0.34.0" \
        "safetensors>=0.4.5" \
        "huggingface_hub[hf-transfer]>=0.25.2" \
        "runpod>=1.6.2" \
        "Pillow>=10.3.0" \
        "requests>=2.31.0"
    
    # Copy the downloaded model files in a single layer
    COPY --from=downloader /models/qwen_image_edit_2509 /models/qwen_image_edit_2509
    
    # Your app
    WORKDIR /app
    COPY handler.py /app/handler.py
    
    # Make sure cache dir exists (optional if you only use local dir)
    RUN mkdir -p /opt/hf_cache
    
    CMD ["python3", "-u", "/app/handler.py"]
    