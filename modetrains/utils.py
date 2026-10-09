"""Helpers: hardware detection, VRAM estimates, generic speedups.

Copyright (c) 2026 salim-slimani.
"""
from __future__ import annotations
import os, random, platform


def set_seed(seed: int = 3407):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_device_info() -> dict:
    info = {"platform": platform.system(), "cpu": os.cpu_count()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            prop = torch.cuda.get_device_properties(0)
            info["gpu_vram_gb"] = round(prop.total_memory / 1e9, 2)
            info["bf16_supported"] = prop.major >= 8
        else:
            info["gpu_name"] = None
    except Exception as e:
        info["torch"] = f"not-installed ({e})"
        info["cuda_available"] = False
    return info


def pick_dtype(prefer: str | None = None) -> str:
    """Auto dtype: bfloat16 when supported, else float16."""
    if prefer in ("bfloat16", "float16"):
        return prefer
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            return "bfloat16"
    except Exception:
        pass
    return "float16"


def estimate_vram(model_name: str, load_in_4bit: bool, max_seq: int, lora_r: int = 16) -> dict:
    """Rough empirical estimate (7-8B class) — check before loading."""
    base = 4.0 if load_in_4bit else 14.0  # GB for a ~7-8B model
    seq_factor = max_seq / 2048.0
    lora_factor = 0.3 + (lora_r / 64.0)
    train_gb = round(base + 2.5 * seq_factor + lora_factor, 2)
    infer_gb = round(base * 0.7 + 0.5 * seq_factor, 2)
    return {"train_need_gb": train_gb, "infer_need_gb": infer_gb,
            "note": "Rough estimate for 7-8B models. Larger models need more."}


def auto_batch_size(vram_gb: float | None, seq_len: int) -> tuple[int, int]:
    """Guess (batch, accum) from the available VRAM."""
    if vram_gb is None or vram_gb < 8:
        return (1, 8) if seq_len > 2048 else (2, 4)
    if vram_gb < 16:
        return (2, 4)
    if vram_gb < 24:
        return (4, 4)
    return (4, 2)


def enable_speedups(use_torch_compile: bool = False, model=None):
    """Enable generic PyTorch speedups (matmul precision + TF32 + optional compile)."""
    try:
        import torch
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
        if use_torch_compile and model is not None:
            try:
                model = torch.compile(model, mode="max-autotune", fullgraph=False)
            except Exception:
                pass
        return model
    except Exception:
        return model
