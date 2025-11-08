# Use a modern Runpod PyTorch base image
FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204

# Set environment variables for Hugging Face cache
# During build, use /root/.cache; at runtime, /workspace/cache will be used
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface

# Install dependencies
RUN pip install --no-cache-dir transformers accelerate safetensors pillow runpod hf_transfer bitsandbytes git+https://github.com/huggingface/diffusers

# Pre-download the model during build to avoid runtime download
RUN python -c "from diffusers import DiffusionPipeline; import torch; DiffusionPipeline.from_pretrained('Qwen/Qwen-Image-Edit-2509', torch_dtype=torch.float16, use_safetensors=True)"

# Copy handler file
WORKDIR /app
COPY handler.py .

# Set entrypoint
CMD ["python", "handler.py"]
