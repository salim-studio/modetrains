"""Export: merge LoRA adapters, save locally, push to the Hub, GGUF notes.

Copyright (c) 2026 salim-slimani.
"""
from __future__ import annotations
import os


def save_merged(model, tokenizer, out_dir: str, merge: bool = True):
    """Save the model. merge=True folds LoRA weights into the base model."""
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
    """Merge (optionally) and push model + tokenizer to the Hugging Face Hub."""
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
    """GGUF conversion needs llama.cpp — print the ready-to-run command."""
    print(f"[modetrains] Convert to GGUF ({quant}):")
    print(f"  python -m llama_cpp.convert --outtype {quant} {out_dir} --outfile {out_dir}/model.gguf")
    print("  Then: ollama create mymodel -f Modelfile  (or use llama.cpp directly)")
