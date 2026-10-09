"""مثال شامل: تدريب عربي سريع بأسلوب Unsloth."""
from modetrains import ModeTrainsConfig, FastModel, train_sft, save_merged, generate
from modetrains.data import format_alpaca

cfg = ModeTrainsConfig.fast_default("unsloth/llama-3-8b-bnb-4bit", max_seq_length=2048)
cfg.max_steps = 60

model, tok = FastModel.from_pretrained(cfg)
model = FastModel.get_peft_model(model, cfg)

# بيانات عربية صغيرة للتجربة
from datasets import Dataset
rows = [
    {"instruction": "اشرح الاحتباس الحراري ببساطة", "input": "", "output": "هو ارتفاع حرارة الأرض بسبب الغازات الدفيئة..."},
    {"instruction": "لخص", "input": "النص طويل...", "output": "الخلاصة..."},
]
texts = [format_alpaca(r["instruction"], r["input"], r["output"]) for r in rows]
ds = Dataset.from_dict({"text": texts * 50})

trainer, stats = train_sft(model, tok, ds, cfg, dataset_text_field="text")
print(stats)
save_merged(model, tok, "outputs-mt")
print(generate(model, tok, "ما هو الذكاء الاصطناعي؟"))
