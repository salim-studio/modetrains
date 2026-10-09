"""استدلال سريع: generate + chat مع KV-cache و torch.compile اختياري."""
from __future__ import annotations


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256,
             temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True) -> str:
    try:
        import torch
    except Exception as e:
        raise ImportError("torch غير مثبت. ثبّت: pip install torch") from e
    from .models import FastModel
    model = FastModel.for_inference(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        pass
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_p=top_p,
                             do_sample=do_sample, use_cache=True,
                             pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # أعد فقط التكملة
    if text.startswith(prompt):
        return text[len(prompt):].lstrip()
    return text


def chat(model, tokenizer, messages, **kw) -> str:
    from .data import format_chat
    prompt = format_chat(messages, tokenizer, add_generation_prompt=True)
    return generate(model, tokenizer, prompt, **kw)
