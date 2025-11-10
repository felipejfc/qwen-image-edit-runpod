FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204

# Install dependencies with specific versions for Qwen2.5-VL compatibility
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        transformers>=4.49.0 \
        accelerate>=1.2.1 \
        safetensors>=0.4.0 \
        pillow>=10.0.0 \
        runpod \
        hf_transfer \
        bitsandbytes>=0.45.0 \
        peft>=0.14.0 \
        fastapi>=0.104.0 \
        uvicorn[standard]>=0.24.0 \
        pydantic>=2.0.0 \
        aiohttp>=3.9.0 \
        git+https://github.com/huggingface/diffusers && \
    pip install --no-cache-dir \
        torchvision>=0.20.0 \
        --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY handler.py /app/handler.py

# Create model cache directory for runtime downloads
RUN mkdir -p /models/qwen_image_edit_2509 /opt/hf_cache
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

CMD ["python", "handler.py"]
