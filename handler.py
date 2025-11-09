# handler.py
import os, io, base64, runpod, torch
from PIL import Image
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from diffusers.utils import load_image

MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/qwen_image_edit_2509")
TORCH_DTYPE = torch.bfloat16

# Configuration for Lightning LoRA (faster inference)
USE_LIGHTNING_LORA = os.environ.get("USE_LIGHTNING_LORA", "true").lower() == "true"
LIGHTNING_STEPS = int(os.environ.get("LIGHTNING_STEPS", "8"))  # 4 or 8

print(f"Loading model from {MODEL_ID}...")
print(f"Cache directory: {MODEL_DIR}")
print(f"Using 4-bit quantization for memory efficiency")
print(f"Lightning LoRA: {USE_LIGHTNING_LORA} (steps: {LIGHTNING_STEPS})")

# Load transformer with 4-bit quantization
print("Loading quantized transformer...")
transformer_quant_config = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=TORCH_DTYPE,
    llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
)

transformer = QwenImageTransformer2DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer",
    quantization_config=transformer_quant_config,
    torch_dtype=TORCH_DTYPE,
    cache_dir=MODEL_DIR,
)
transformer = transformer.to("cpu")

# Load text encoder with 4-bit quantization
print("Loading quantized text encoder...")
text_encoder_quant_config = TransformersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=TORCH_DTYPE,
)

text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    subfolder="text_encoder",
    quantization_config=text_encoder_quant_config,
    torch_dtype=TORCH_DTYPE,
    cache_dir=MODEL_DIR,
)
text_encoder = text_encoder.to("cpu")

# Create pipeline with quantized components
print("Creating pipeline...")
pipe = QwenImageEditPlusPipeline.from_pretrained(
    MODEL_ID,
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=TORCH_DTYPE,
    cache_dir=MODEL_DIR,
)

# Optionally load Lightning LoRA weights for faster inference
if USE_LIGHTNING_LORA:
    try:
        print(f"Loading Lightning LoRA for {LIGHTNING_STEPS}-step inference...")
        if LIGHTNING_STEPS == 8:
            pipe.load_lora_weights(
                "lightx2v/Qwen-Image-Lightning",
                weight_name="Qwen-Image-Lightning-8steps-V2.0.safetensors"
            )
            print("✓ 8-step Lightning LoRA loaded successfully")
        elif LIGHTNING_STEPS == 4:
            pipe.load_lora_weights(
                "lightx2v/Qwen-Image-Lightning",
                weight_name="Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors"
            )
            print("✓ 4-step Lightning LoRA loaded successfully")
        else:
            print(f"Warning: LIGHTNING_STEPS={LIGHTNING_STEPS} not supported (use 4 or 8), skipping LoRA")
            USE_LIGHTNING_LORA = False
    except Exception as e:
        print(f"Warning: Failed to load Lightning LoRA: {e}")
        print("Continuing without Lightning LoRA...")
        USE_LIGHTNING_LORA = False

# Enable model CPU offload for memory efficiency
print("Enabling CPU offload...")
pipe.enable_model_cpu_offload()

print("Model loaded successfully!")

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

        # Use Lightning steps if enabled, otherwise use provided steps
        default_steps = LIGHTNING_STEPS if USE_LIGHTNING_LORA else 40
        num_steps = int(inp.get("num_inference_steps", default_steps))
        
        # Create generator for reproducibility if seed is provided
        generator = None
        if "seed" in inp:
            generator = torch.Generator(device="cuda").manual_seed(int(inp["seed"]))

        out = pipe(
            image=img,
            prompt=prompt,
            num_inference_steps=num_steps,
            true_cfg_scale=float(inp.get("true_cfg_scale", 4.0)),
            negative_prompt=inp.get("negative_prompt"),
            num_images_per_prompt=1,
            generator=generator,
        )
        return {
            "prompt": prompt,
            "output_image_base64": _to_b64(out.images[0]),
            "num_inference_steps": num_steps,
            "lightning_enabled": USE_LIGHTNING_LORA,
        }
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
