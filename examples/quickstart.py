"""End-to-end SFT example (English).

Copyright (c) 2026 salim-slimani.
"""
from datasets import Dataset

from modetrains import ModeTrainsConfig, FastModel, train_sft, save_merged, generate
from modetrains.data import format_alpaca

cfg = ModeTrainsConfig.fast_default("meta-llama/Meta-Llama-3-8B-Instruct", max_seq_length=2048)
cfg.max_steps = 60

model, tok = FastModel.from_pretrained(cfg)
model = FastModel.get_peft_model(model, cfg)

# Tiny demo dataset
rows = [
    {"instruction": "Explain global warming simply", "input": "", "output": "The Earth warms as greenhouse gases trap heat..."},
    {"instruction": "Summarize", "input": "A long text...", "output": "The summary..."},
]
texts = [format_alpaca(r["instruction"], r["input"], r["output"]) for r in rows]
ds = Dataset.from_dict({"text": texts * 50})

trainer, stats = train_sft(model, tok, ds, cfg, dataset_text_field="text")
print(stats)
save_merged(model, tok, "outputs-mt")
print(generate(model, tok, "What is artificial intelligence?"))
