"""Inference logic."""
import torch
import base64
import io
from typing import Dict, Any
from PIL import Image
from diffusers.utils import load_image

from utils import to_base64, default_steps
from config import DEFAULT_CFG_SCALE, USE_LIGHTNING_LORA


def run_inference(pipe, event: Dict[str, Any]) -> Dict[str, Any]:
    """Execute image editing inference."""
    inp = event["input"]
    
    # Load and prepare image
    # Support both base64 and URL inputs
    if "image_base64" in inp and inp["image_base64"]:
        # Decode base64 image
        img_data = base64.b64decode(inp["image_base64"])
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
    elif "image_url" in inp and inp["image_url"]:
        img = load_image(inp["image_url"]).convert("RGB")
    else:
        raise ValueError("Either 'image_base64' or 'image_url' must be provided")
    
    # Get parameters
    prompt = inp.get("prompt", "Enhance the image")
    negative_prompt = inp.get("negative_prompt", None)
    steps = int(inp.get("num_inference_steps", default_steps()))
    cfg = float(inp.get("true_cfg_scale", DEFAULT_CFG_SCALE))
    
    # Prepare generator if seed provided
    generator = None
    if "seed" in inp:
        generator = torch.Generator(device="cuda").manual_seed(int(inp["seed"]))
    
    # Run inference
    with torch.inference_mode():
        out = pipe(
            image=img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            true_cfg_scale=cfg,
            num_images_per_prompt=1,
            generator=generator,
        )
    
    return {
        "image": to_base64(out.images[0]),
        "prompt": prompt,
        "num_inference_steps": steps,
        "lightning_enabled": bool(USE_LIGHTNING_LORA),
    }

