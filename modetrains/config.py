"""إعدادات modetrains — كل الخيارات في مكان واحد."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QuantConfig:
    """إعدادات التكميم (Quantization)."""
    mode: str = "4bit"  # "4bit" | "8bit" | "none" | "fp8"
    bnb_4bit_compute_dtype: str = "bfloat16"  # bfloat16 | float16
    bnb_4bit_quant_type: str = "nf4"  # nf4 | fp4
    use_double_quant: bool = True
    llm_int8_threshold: float = 6.0

    def validate(self):
        assert self.mode in ("4bit", "8bit", "none", "fp8"), f"quant mode غير معروف: {self.mode}"


@dataclass
class LoraConfig:
    """إعدادات LoRA / QLoRA."""
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_rslora: bool = True
    use_gradient_checkpointing: str = "unsloth"  # unsloth | true | false
    loftq_config: Optional[dict] = None

    def validate(self):
        assert self.r > 0 and self.alpha > 0


@dataclass
class ModeTrainsConfig:
    """الإعداد الرئيسي للتدريب السريع."""
    model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    max_seq_length: int = 2048
    dtype: Optional[str] = None  # auto: bfloat16 إن توفر وإلا float16
    load_in_4bit: bool = True
    attn_implementation: str = "auto"  # auto | flash_attention_2 | sdpa | eager
    rope_scaling: Optional[dict] = None  # مثال: {"type":"linear","factor":4.0}
    device_map: str = "auto"
    # تسريع
    use_flash_attn: bool = True
    use_liger_kernel: bool = True  # إن توفرت مكتبة liger-kernel
    use_torch_compile: bool = False
    use_better_transformer: bool = False
    gradient_checkpointing: bool = True
    # تدريب
    per_device_batch: int = 2
    grad_accum: int = 4
    lr: float = 2e-4
    warmup_steps: int = 5
    max_steps: int = 60
    optim: str = "adamw_8bit"  # adamw_8bit | adamw_torch | paged_adamw_8bit | lion
    weight_decay: float = 0.01
    lr_scheduler: str = "linear"
    seed: int = 3407
    output_dir: str = "outputs"
    packing: bool = True  # دمج عدة أمثلة في سياق واحد = أسرع 2-3x
    dataset_num_proc: int = 2
    fp16: bool = False
    bf16: Optional[bool] = None  # auto
    logging_steps: int = 1
    save_steps: int = 100
    eval_steps: Optional[int] = None

    quant: QuantConfig = field(default_factory=QuantConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)

    def validate(self):
        self.quant.validate()
        self.lora.validate()
        assert self.max_seq_length >= 256

    @classmethod
    def fast_default(cls, model_name: str, max_seq_length: int = 2048) -> "ModeTrainsConfig":
        """وصفة جاهزة لأسرع تدريب موفر للذاكرة."""
        return cls(model_name=model_name, max_seq_length=max_seq_length,
                   load_in_4bit=True, packing=True, gradient_checkpointing=True,
                   per_device_batch=2, grad_accum=4, optim="adamw_8bit")
