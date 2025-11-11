"""Utility functions."""
import io
import base64
import torch
from PIL import Image
from typing import Dict, Any

from config import USE_LIGHTNING_LORA, LIGHTNING_STEPS


def to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def default_steps() -> int:
    """Get default inference steps based on LoRA configuration."""
    return LIGHTNING_STEPS if USE_LIGHTNING_LORA else 40


def handle_error(e: Exception) -> Dict[str, Any]:
    """Format error response with GPU info."""
    info = None
    try:
        info = {
            "device_count": torch.cuda.device_count(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.device_count() else None,
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None,
        }
    except Exception:
        pass
    return {"error": str(e), "gpu_info": info}

