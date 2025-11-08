import os
import io
import base64
import runpod
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline  # <-- custom pipeline
from diffusers.utils import load_image

# Recommended on RunPod: persist cache on a mounted volume to avoid re-downloads
os.environ.setdefault("HF_HOME", "/runpod-volume/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/runpod-volume/hf_cache")

# 1) Load the *right* pipeline, in BF16, all on the single GPU
#    - trust_remote_code helps when the custom pipeline lives in the repo
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
pipe.to("cuda")  # don't use device_map="auto" here

# Optional memory/latency knobs (safe on 24–48GB cards)
pipe.enable_attention_slicing()
# If you installed xFormers or flash-attn, you can enable them as well:
# pipe.enable_xformers_memory_efficient_attention()

def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def handler(event):
    try:
        inp = event["input"]
        prompt = inp.get("prompt", "Enhance the image")
        image_url = inp.get("image_url")
        if not image_url:
            return {"error": "Missing 'image_url'."}

        # Load image (URL or data: URL)
        image = load_image(image_url).convert("RGB")

        # Basic run (match model card defaults)
        out = pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=int(inp.get("num_inference_steps", 40)),
            true_cfg_scale=float(inp.get("true_cfg_scale", 4.0)),
            negative_prompt=inp.get("negative_prompt", None),
            num_images_per_prompt=1,
        )
        return {
            "prompt": prompt,
            "output_image_base64": _to_b64(out.images[0]),
        }
    except Exception as e:
        # Helpful GPU diagnostics when things go wrong
        try:
            gpu = torch.cuda.get_device_name(0)
            mem_total = torch.cuda.get_device_properties(0).total_memory
            mem_alloc = torch.cuda.memory_allocated(0)
            mem_res = torch.cuda.memory_reserved(0)
            gpu_info = {
                "gpu": gpu,
                "mem_total": mem_total,
                "mem_alloc": mem_alloc,
                "mem_reserved": mem_res,
                "torch_bf16": torch.cuda.is_bf16_supported(),
            }
        except Exception:
            gpu_info = None
        return {"error": str(e), "gpu_info": gpu_info}

runpod.serverless.start({"handler": handler})
