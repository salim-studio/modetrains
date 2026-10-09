"""التصدير: دمج LoRA + حفظ + (اختياري) GGUF + دفع إلى Hub."""
from __future__ import annotations
import os


def save_merged(model, tokenizer, out_dir: str, merge: bool = True):
    """حفظ النموذج. merge=True يدمج LoRA في الأوزان الأساسية."""
    os.makedirs(out_dir, exist_ok=True)
    m = model
    if merge:
        try:
            m = model.merge_and_unload()
        except Exception:
            m = model
    m.save_pretrained(out_dir)
    try:
        tokenizer.save_pretrained(out_dir)
    except Exception:
        pass
    return out_dir


def push_to_hub_helper(model, tokenizer, repo_id: str, merge: bool = True, private: bool = False):
    m = model
    if merge:
        try:
            m = model.merge_and_unload()
        except Exception:
            pass
    m.push_to_hub(repo_id, private=private)
    try:
        tokenizer.push_to_hub(repo_id, private=private)
    except Exception:
        pass
    return repo_id


def export_gguf_note(out_dir: str, quant: str = "q4_k_m"):
    """GGUF يحتاج llama.cpp. نطبع الأمر الجاهز بدل كسر التثبيت."""
    print(f"[modetrains] للتحويل إلى GGUF ({quant}):")
    print(f"  python -m llama_cpp.convert --outtype {quant} {out_dir} --outfile {out_dir}/model.gguf")
    print("  ثم: ollama create mymodel -f Modelfile  (أو استخدم llama.cpp مباشرة)")
