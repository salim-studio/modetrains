<p align="center">
  <img src="assets/logo.svg" width="120" alt="modetrains logo"/>
</p>

<h1 align="center">modetrains</h1>
<p align="center"><strong>Fast, memory-efficient LLM fine-tuning.</strong><br/>4-bit QLoRA &bull; Flash / SDPA attention &bull; Sequence packing &bull; 8-bit optimizers</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-4F46E5?style=flat-square" alt="Python"/>
  <img src="https://img.shields.io/badge/version-0.1.0-06B6D4?style=flat-square" alt="Version"/>
  <img src="https://img.shields.io/badge/license-MIT-0B0E1A?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/platform-windows%20%7C%20linux%20%7C%20macos-lightgrey?style=flat-square" alt="Platform"/>
</p>

---

## Why modetrains?

| Technique | Benefit |
|---|---|
| Double-quantized 4-bit NF4 | ~70% less VRAM |
| Flash Attention 2 / SDPA (auto) | ~30% faster, lower memory |
| Sequence packing + `group_by_length` | Up to 2–3x throughput, near-zero padding |
| Non-reentrant gradient checkpointing | ~40% less VRAM |
| Optional Liger kernels | ~20% faster norms / cross-entropy |
| AdamW 8-bit / paged optimizers | ~30% less optimizer memory |
| rsLoRA | Free accuracy gain |

Everything is built on fully open components (`transformers` + `peft` + `trl` + `bitsandbytes` + `accelerate`) — no manual CUDA builds, works on Windows, Linux, CPU and GPU.

## Brand

- **Name:** modetrains — *models, trained at speed.*
- **Mark:** the indigo→cyan `M` rail in `assets/logo.svg` (dark `#0B0E1A` field, gradient `#4F46E5` → `#06B6D4`).
- **Voice:** fast, practical, no hype. English-first docs.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# Optional, for maximum speed:
pip install flash-attn --no-build-isolation
pip install liger-kernel
# Editable install with CLI:
pip install -e .
```

## Quickstart

```python
from modetrains import ModeTrainsConfig, FastModel, train_sft, save_merged
from modetrains.data import load_chat_dataset

cfg = ModeTrainsConfig.fast_default("meta-llama/Meta-Llama-3-8B-Instruct", max_seq_length=2048)
model, tok = FastModel.from_pretrained(cfg)
model = FastModel.get_peft_model(model, cfg)

ds = load_chat_dataset("yahma/alpaca-cleaned", split="train[:1000]")
# Normalize to a `text` column — see examples/quickstart.py
trainer, stats = train_sft(model, tok, ds, cfg, dataset_text_field="text")
save_merged(model, tok, "outputs")
```

## CLI

```bash
python -m modetrains info
python -m modetrains train --model meta-llama/Meta-Llama-3-8B-Instruct --data yahma/alpaca-cleaned --steps 60
python -m modetrains infer --model outputs --prompt "What is quantization?"
```

## Project layout

```
modetrains/
  config.py    Quant / LoRA / training settings
  models.py    FastModel: fast loading + LoRA
  data.py      Alpaca / chat formatting + packing
  trainers.py  SFT / DPO / GRPO on TRL
  export.py    LoRA merge + Hub + GGUF notes
  infer.py     Fast generation (KV-cache)
  utils.py     Hardware / VRAM / speedups
  cli.py       Command-line interface
assets/
  logo.svg     Official mark
examples/
  quickstart.py
tests/
  test_modetrains.py
```

## Methods

- **SFT:** `train_sft` with packing and length-grouped batches.
- **DPO:** `train_dpo` on `prompt / chosen / rejected` data.
- **GRPO:** `train_grpo` when the installed TRL supports it.
- **Export:** `save_merged` folds LoRA back into base weights; `push_to_hub_helper` publishes both.

## License

MIT — Copyright (c) 2026 salim-slimani. See [LICENSE](LICENSE).
