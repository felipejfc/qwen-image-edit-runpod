# Use a modern Runpod PyTorch base image
FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204

# Install dependencies
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors pillow runpod hf_transfer bitsandbytes huggingface_hub git+https://github.com/huggingface/diffusers

# Pre-download model files without loading into memory
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen-Image-Edit-2509', ignore_patterns=['*.ckpt', '*.bin'])"

# Copy handler file
WORKDIR /app
COPY handler.py .

# Set entrypoint
CMD ["python", "handler.py"]
