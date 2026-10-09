"""Lightweight logic tests — no GPU / torch required.

Copyright (c) 2026 salim-slimani.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modetrains.config import ModeTrainsConfig
from modetrains.data import format_alpaca, format_chat, to_text_rows
from modetrains.utils import estimate_vram, auto_batch_size, pick_dtype, set_seed

def test_config():
    c = ModeTrainsConfig.fast_default("some-model", 2048)
    c.validate()
    assert c.packing and c.load_in_4bit

def test_alpaca():
    t = format_alpaca("Do X", "", "Done")
    assert "Instruction" in t and "Done" in t
    t2 = format_alpaca("Summarize", "some text", "summary")
    assert "Input" in t2

def test_chat():
    msgs = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    t = format_chat(msgs, tokenizer=None, add_generation_prompt=True)
    assert "Hello" in t and "<assistant>" in t

def test_rows():
    rows = [{"instruction": "i", "input": "", "output": "o"}]
    assert len(to_text_rows(rows, "alpaca")) == 1

def test_utils():
    set_seed(1)
    e = estimate_vram("7B", True, 2048)
    assert e["train_need_gb"] > 0
    b, a = auto_batch_size(8, 2048)
    assert b >= 1
    assert pick_dtype("float16") == "float16"

if __name__ == "__main__":
    for fn in [test_config, test_alpaca, test_chat, test_rows, test_utils]:
        fn(); print(f"PASS {fn.__name__}")
    print("ALL MODETRAINS TESTS PASSED")
