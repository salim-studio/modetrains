"""Fast model loading: 4-bit + Flash/SDPA + RoPE + gradient checkpointing + Liger.

Copyright (c) 2026 salim-slimani.
"""
from __future__ import annotations
from .config import ModeTrainsConfig
from .utils import pick_dtype, enable_speedups


def _resolve_attn(cfg: ModeTrainsConfig) -> str | None:
    if cfg.attn_implementation != "auto":
        return None if cfg.attn_implementation == "eager" else cfg.attn_implementation
    if not cfg.use_flash_attn:
        return "sdpa"
    # auto: prefer flash_attention_2, fall back to sdpa
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def _build_bnb_config(cfg: ModeTrainsConfig):
    if not cfg.load_in_4bit and cfg.quant.mode == "none":
        return None
    try:
        from transformers import BitsAndBytesConfig
        import torch
        dtype = torch.bfloat16 if pick_dtype(cfg.dtype) == "bfloat16" else torch.float16
        if cfg.quant.mode in ("4bit",) or cfg.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type=cfg.quant.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=cfg.quant.use_double_quant,
            )
        if cfg.quant.mode == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True,
                                      llm_int8_threshold=cfg.quant.llm_int8_threshold)
    except Exception:
        pass
    return None


class FastModel:
    """Primary entry point: FastModel.from_pretrained + get_peft_model."""

    @staticmethod
    def from_pretrained(cfg: ModeTrainsConfig):
        """Load a model + tokenizer ready for fast training.

        Returns: (model, tokenizer)
        """
        cfg.validate()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        dtype = torch.bfloat16 if pick_dtype(cfg.dtype) == "bfloat16" else torch.float16
        attn = _resolve_attn(cfg)
        bnb = _build_bnb_config(cfg)

        kwargs = dict(device_map=cfg.device_map, trust_remote_code=True)
        if bnb is not None:
            kwargs["quantization_config"] = bnb
        else:
            kwargs["torch_dtype"] = dtype
        if attn is not None:
            kwargs["attn_implementation"] = attn
        if cfg.rope_scaling:
            kwargs["rope_scaling"] = cfg.rope_scaling

        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        if getattr(model.config, "model_max_length", None):
            try:
                model.config.model_max_length = cfg.max_seq_length
            except Exception:
                pass

        # Speedup 1: memory-efficient gradient checkpointing
        if cfg.gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                try:
                    model.gradient_checkpointing_enable()
                except Exception:
                    pass
            model.config.use_cache = False
        else:
            model.config.use_cache = True

        # Speedup 2: fused Liger kernels (RMSNorm / RoPE / SwiGLU) when installed
        if cfg.use_liger_kernel:
            try:
                from liger_kernel.transformers import _apply_liger_kernel
                _apply_liger_kernel(model_type=getattr(model.config, "model_type", ""))
            except Exception:
                try:
                    import liger_kernel.transformers as _lk  # noqa: F401
                    from liger_kernel.transformers import monkey_patch as _mp
                    if hasattr(_mp, "apply_liger_kernel_to_llama"):
                        _mp.apply_liger_kernel_to_llama()
                except Exception:
                    pass

        # Speedup 3: generic torch ops
        enable_speedups(cfg.use_torch_compile, model=None)
        if cfg.use_better_transformer:
            try:
                from optimum.bettertransformer import BetterTransformer
                model = BetterTransformer.transform(model, keep_original_model=False)
            except Exception:
                pass

        # Record the effective precision
        try:
            cfg.bf16 = (dtype == torch.bfloat16)
            cfg.fp16 = (dtype == torch.float16)
        except Exception:
            pass
        return model, tokenizer

    @staticmethod
    def get_peft_model(model, cfg: ModeTrainsConfig):
        """Attach a fast LoRA adapter (supports rsLoRA + LoftQ)."""
        from peft import LoraConfig as _LC, get_peft_model as _get, prepare_model_for_kbit_training

        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            try:
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=(cfg.lora.use_gradient_checkpointing != "false"))
            except Exception:
                pass

        peft_cfg = _LC(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            target_modules=list(cfg.lora.target_modules),
            lora_dropout=cfg.lora.dropout,
            bias=cfg.lora.bias,
            task_type="CAUSAL_LM",
            use_rslora=cfg.lora.use_rslora,
            loftq_config=cfg.lora.loftq_config,
        )
        model = _get(model, peft_cfg)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
        return model

    @staticmethod
    def for_inference(model):
        """Switch a trained model to inference mode (KV-cache + eval)."""
        try:
            model.eval()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = True
            if hasattr(model, "gradient_checkpointing_disable"):
                try:
                    model.gradient_checkpointing_disable()
                except Exception:
                    pass
        except Exception:
            pass
        return model
