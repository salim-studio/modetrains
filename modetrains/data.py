"""تحضير البيانات: Alpaca / Chat / Packing — يدعم العربية."""
from __future__ import annotations
from typing import List, Dict, Any

ALPACA_PROMPT = """أدناه تعليمات تصف مهمة. اكتب ردًا مناسبًا يكمل الطلب.

### التعليمات:
{}

### المدخلات:
{}

### الرد:
{}"""

ALPACA_NO_INPUT = """أدناه تعليمات تصف مهمة. اكتب ردًا مناسبًا يكمل الطلب.

### التعليمات:
{}

### الرد:
{}"""


def format_alpaca(instruction: str, inp: str = "", output: str = "") -> str:
    if inp.strip():
        return ALPACA_PROMPT.format(instruction, inp, output)
    return ALPACA_NO_INPUT.format(instruction, output)


def format_chat(messages: List[Dict[str, str]], tokenizer=None, add_generation_prompt: bool = False) -> str:
    """messages = [{"role":"user"/"assistant"/"system", "content": "..."}].
    يستخدم chat_template إن توفر وإلا تنسيق بسيط."""
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=add_generation_prompt)
        except Exception:
            pass
    parts = []
    for m in messages:
        r, c = m.get("role", "user"), m.get("content", "")
        if r == "system":
            parts.append(f"<system>{c}</system>")
        elif r == "assistant":
            parts.append(f"<assistant>{c}</assistant>")
        else:
            parts.append(f"<user>{c}</user>")
    if add_generation_prompt:
        parts.append("<assistant>")
    return "\n".join(parts)


def to_text_rows(rows: List[Dict[str, Any]], mode: str = "alpaca", tokenizer=None) -> List[str]:
    out = []
    for r in rows:
        if mode == "alpaca":
            out.append(format_alpaca(r.get("instruction", ""), r.get("input", ""), r.get("output", "")))
        elif mode == "chat":
            out.append(format_chat(r.get("messages", []), tokenizer))
        elif mode == "text":
            out.append(r.get("text", ""))
        else:
            raise ValueError(f"mode غير معروف: {mode}")
    return out


def load_chat_dataset(hf_name_or_path: str, split: str = "train", text_field: str = "text"):
    """تحميل dataset من HuggingFace أو ملف محلي (json/jsonl/csv)."""
    from datasets import load_dataset, load_from_disk
    import os
    if os.path.isdir(hf_name_or_path):
        try:
            return load_from_disk(hf_name_or_path)
        except Exception:
            pass
    if hf_name_or_path.endswith((".json", ".jsonl", ".csv")):
        ext = "csv" if hf_name_or_path.endswith(".csv") else "json"
        return load_dataset(ext, data_files=hf_name_or_path, split=split)
    return load_dataset(hf_name_or_path, split=split)


def pack_dataset(tokenized, seq_length: int = 2048, tokenizer=None):
    """دمج الرموز (packing) لملء السياق بالكامل — أسرع 2x وأقل حشو.

    tokenized: dict فيه input_ids (list of lists). يعيد dataset جاهز.
    يستخدم Batched packing بسيط وفعال بدون اعتماديات إضافية.
    """
    import torch
    from torch.utils.data import Dataset as _D

    ids = tokenized["input_ids"]
    # تسطيح ثم تقطيع
    flat = [t for seq in ids for t in seq]
    # إسقاط الباقي
    n = (len(flat) // seq_length) * seq_length
    flat = flat[:n]

    class _Packed(_D):
        def __len__(self): return max(1, len(flat) // seq_length)
        def __getitem__(self, i):
            s = flat[i*seq_length:(i+1)*seq_length]
            x = torch.tensor(s, dtype=torch.long)
            return {"input_ids": x, "labels": x.clone(),
                    "attention_mask": torch.ones_like(x)}
    return _Packed()
