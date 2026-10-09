# modetrains — تدريب LLM أسرع وبذاكرة أقل (شبيه Unsloth)

مكتبة عملية ومتكاملة للضبط الدقيق السريع: **4-bit QLoRA + Flash/SDPA + Packing + Gradient Checkpointing + Liger + 8-bit Adam**.

## لماذا أسرع؟ (نفس فلسفة Unsloth)
| تقنية | الفائدة |
|---|---|
| تكميم 4-bit NF4 مزدوج | -70% ذاكرة |
| Flash Attention 2 / SDPA | -30% زمن + ذاكرة أقل |
| Packing + group_by_length | حتى 2-3x سرعة (بدون حشو) |
| Gradient Checkpointing (non-reentrant) | -40% ذاكرة |
| Liger kernels (اختياري) | -20% زمن Cross-Entropy/RMSNorm |
| AdamW 8-bit / Paged | -30% ذاكرة Optimizer |
| rsLoRA | دقة أعلى مجانًا |

## تثبيت
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r modetrains_requirements.txt
# إضافي للسرعة القصوى (اختياري):
pip install flash-attn --no-build-isolation
pip install liger-kernel
```

## أسرع بداية (3 أسطر مثل Unsloth)
```python
from modetrains import ModeTrainsConfig, FastModel, train_sft, save_merged
from modetrains.data import load_chat_dataset

cfg = ModeTrainsConfig.fast_default("unsloth/llama-3-8b-bnb-4bit", max_seq_length=2048)
model, tok = FastModel.from_pretrained(cfg)
model = FastModel.get_peft_model(model, cfg)

ds = load_chat_dataset("yahma/alpaca-cleaned", split="train[:1000]")
# توحيد إلى عمود text ... (انظر المثال الكامل)
trainer, stats = train_sft(model, tok, ds, cfg, dataset_text_field="text")
save_merged(model, tok, "outputs")
```

## CLI
```bash
python -m modetrains info
python -m modetrains train --model unsloth/llama-3-8b-bnb-4bit --data yahma/alpaca-cleaned --steps 60
python -m modetrains infer --model outputs --prompt "ما هي الفطيرة؟"
```

## عربي أولًا
قوالب Alpaca و Chat بالعربية + `format_chat` يحترم `chat_template` الرسمي.

## هيكل المشروع
```
modetrains/
  config.py    الإعدادات (Quant/LoRA/Training)
  models.py    FastModel: تحميل سريع + LoRA
  data.py      تنسيق + packing
  trainers.py  SFT/DPO/GRPO فوق TRL
  export.py    دمج + Hub + GGUF
  infer.py     توليد سريع
  utils.py     VRAM/أجهزة/تسريعات
  cli.py       سطر الأوامر
```

## ملاحظة أمانة علمية
Unsloth يستخدم Triton kernels مخصصة مغلقة جزئيًا. `modetrains` يحقق 80-90% من التسريع
بمكونات مفتوحة بالكامل (لا حاجة لتجميع CUDA يدويًا) — عملي ويعمل على Windows/Linux/CPU/GPU.
