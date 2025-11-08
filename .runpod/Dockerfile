FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204

RUN pip install --no-cache-dir diffusers transformers accelerate safetensors pillow runpod hf_transfer bitsandbytes git+https://github.com/huggingface/diffusers

WORKDIR /app
COPY handler.py /app/handler.py

# Create model cache directory for runtime downloads
RUN mkdir -p /models/qwen_image_edit_2509 /opt/hf_cache

CMD ["python3", "-u", "/app/handler.py"]
