"""modetrains — مكتبة تدريب LLM سريعة وموفرة للذاكرة (شبيهة بـ Unsloth)."""
from .config import ModeTrainsConfig, LoraConfig, QuantConfig
from .models import FastModel
from .data import format_alpaca, format_chat, pack_dataset, load_chat_dataset
from .trainers import train_sft, train_dpo
from .export import save_merged, push_to_hub_helper
from .infer import generate, chat
from .utils import (
    get_device_info,
    estimate_vram,
    auto_batch_size,
    enable_speedups,
    set_seed,
)

__version__ = "0.1.0"
__all__ = [
    "ModeTrainsConfig", "LoraConfig", "QuantConfig",
    "FastModel",
    "format_alpaca", "format_chat", "pack_dataset", "load_chat_dataset",
    "train_sft", "train_dpo",
    "save_merged", "push_to_hub_helper",
    "generate", "chat",
    "get_device_info", "estimate_vram", "auto_batch_size",
    "enable_speedups", "set_seed",
]
