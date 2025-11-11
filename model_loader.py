"""Model loading and pipeline setup."""
import torch
import torch.nn as nn
from transformers import BitsAndBytesConfig as HF_BNB, Qwen2_5_VLForConditionalGeneration
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel, BitsAndBytesConfig as DF_BNB

from config import (
    MODEL_ID, MODEL_DIR, LOCAL_ONLY, PRECISION_MODE, QUANTIZE_COMPONENTS,
    INT8_CPU_OFFLOAD, BNB_COMPUTE, PIPELINE_DTYPE, MAX_MEMORY,
    USE_LIGHTNING_LORA, LIGHTNING_STEPS
)


def _bnb_compute_dtype():
    return PIPELINE_DTYPE


def _bnb_cfg_8bit_hf():
    return HF_BNB(load_in_8bit=True)


def _bnb_cfg_4bit_hf():
    return HF_BNB(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_bnb_compute_dtype(),
    )


def _bnb_cfg_8bit_df():
    return DF_BNB(
        load_in_8bit=True,
        llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
    )


def _bnb_cfg_4bit_df():
    return DF_BNB(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_bnb_compute_dtype(),
        llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
    )


def _load_text_encoder_quant(prec_mode: str):
    q = _bnb_cfg_8bit_hf() if prec_mode == "8bit" else _bnb_cfg_4bit_hf()
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder",
        quantization_config=q,
        torch_dtype=PIPELINE_DTYPE,
        cache_dir=MODEL_DIR,
        local_files_only=LOCAL_ONLY,
        trust_remote_code=True,
        max_memory=MAX_MEMORY,
    )


def _load_transformer_quant(prec_mode: str):
    q = _bnb_cfg_8bit_df() if prec_mode == "8bit" else _bnb_cfg_4bit_df()
    return QwenImageTransformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        quantization_config=q,
        torch_dtype=PIPELINE_DTYPE,
        cache_dir=MODEL_DIR,
        local_files_only=LOCAL_ONLY,
        max_memory=MAX_MEMORY,
    )


def _align_bias_dtypes(root: nn.Module, target_dtype: torch.dtype):
    other = torch.float16 if target_dtype == torch.bfloat16 else torch.bfloat16
    for module in root.modules():
        bias = getattr(module, "bias", None)
        if isinstance(bias, nn.Parameter) and bias.dtype == other:
            module.bias = nn.Parameter(bias.data.to(target_dtype), requires_grad=bias.requires_grad)


def load_pipeline():
    """Load and configure the image edit pipeline."""
    print(f"[ModelLoader] Loading components: {MODEL_ID}")
    
    pipe = None
    
    if PRECISION_MODE == "bf16":
        # Fast path: GPU-first assembly
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=PIPELINE_DTYPE,
            cache_dir=MODEL_DIR,
            local_files_only=LOCAL_ONLY,
        )
        pipe.to("cuda")
    else:
        # Quant path: build quantized components first
        print(f"[ModelLoader] Loading quantized text_encoder ({PRECISION_MODE})...")
        text_encoder = _load_text_encoder_quant(PRECISION_MODE)
        if INT8_CPU_OFFLOAD:
            text_encoder = text_encoder.to("cpu")
        
        transformer = None
        if QUANTIZE_COMPONENTS == "all":
            print(f"[ModelLoader] Loading quantized transformer ({PRECISION_MODE})...")
            transformer = _load_transformer_quant(PRECISION_MODE)
            if INT8_CPU_OFFLOAD:
                transformer = transformer.to("cpu")
        
        print("[ModelLoader] Assembling pipeline...")
        fp_kwargs = dict(
            torch_dtype=PIPELINE_DTYPE,
            cache_dir=MODEL_DIR,
            local_files_only=LOCAL_ONLY,
            text_encoder=text_encoder,
        )
        if transformer is not None:
            fp_kwargs["transformer"] = transformer
        
        pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL_ID, **fp_kwargs) 
    
    # Align bias dtypes
    if isinstance(pipe, nn.Module):
        _align_bias_dtypes(pipe, PIPELINE_DTYPE)
    
    # Memory optimizations
    pipe.enable_attention_slicing()
    if hasattr(pipe, "vae") and pipe.vae is not None:
        if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()
        if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
    
    # Try xFormers
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[ModelLoader] xFormers attention enabled.")
    except Exception as e:
        print(f"[ModelLoader] xFormers not enabled: {e}")
    
    # Load LoRA
    if USE_LIGHTNING_LORA:
        try:
            print(f"[ModelLoader] Loading Lightning LoRA ({LIGHTNING_STEPS} steps)")
            if LIGHTNING_STEPS == 8:
                pipe.load_lora_weights(
                    "lightx2v/Qwen-Image-Lightning",
                    weight_name="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors",
                )
            elif LIGHTNING_STEPS == 4:
                pipe.load_lora_weights(
                    "lightx2v/Qwen-Image-Lightning",
                    weight_name="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
                )
            else:
                print("[ModelLoader] Unsupported LIGHTNING_STEPS; skipping LoRA.")
            print("[ModelLoader] Lightning ready.")
        except Exception as e:
            print(f"[ModelLoader] Lightning LoRA load failed: {e}")
    
    print("[ModelLoader] Pipeline ready.")

    if INT8_CPU_OFFLOAD:
        print("[ModelLoader] Enabling model CPU offload...")
        pipe.enable_model_cpu_offload()
    else:
        print("[ModelLoader] Moving pipeline to CUDA...")
        pipe.to("cuda")

    return pipe

