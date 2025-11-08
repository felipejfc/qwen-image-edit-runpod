# handler.py
import os, io, base64, runpod, torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image


MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/qwen_image_edit_2509")

# Download and load model at runtime
print(f"Loading model from {MODEL_ID}...")
print(f"Cache directory: {MODEL_DIR}")

pipe = QwenImageEditPlusPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    cache_dir=MODEL_DIR,
)
print("Model loaded successfully!")
pipe.to("cuda")
pipe.enable_attention_slicing()            # already had this

# memory savers like ComfyUI:
# VAE-level memory savers (note: on the VAE, not pipeline)
if hasattr(pipe, "vae") and pipe.vae is not None:
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

def _to_b64(img):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def handler(event):
    try:
        inp = event["input"]
        prompt = inp.get("prompt", "Enhance the image")
        image_url = inp.get("image_url")
        if not image_url:
            return {"error": "Missing 'image_url'."}
        img = load_image(image_url).convert("RGB")

        out = pipe(
            image=img,
            prompt=prompt,
            num_inference_steps=int(inp.get("num_inference_steps", 40)),
            true_cfg_scale=float(inp.get("true_cfg_scale", 4.0)),
            negative_prompt=inp.get("negative_prompt"),
            num_images_per_prompt=1,
        )
        return {"prompt": prompt, "output_image_base64": _to_b64(out.images[0])}
    except Exception as e:
        info = None
        try:
            info = {
                "device_count": torch.cuda.device_count(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.device_count() else None,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        except: pass
        return {"error": str(e), "gpu_info": info}

runpod.serverless.start({"handler": handler})
