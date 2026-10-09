"""اختبارات خفيفة بدون GPU/torch — تتحقق من المنطق فقط."""
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
    t = format_alpaca("افعل X", "", "تم")
    assert "التعليمات" in t and "تم" in t
    t2 = format_alpaca("لخص", "نص", "خلاصة")
    assert "المدخلات" in t2

def test_chat():
    msgs = [{"role": "user", "content": "مرحبا"}, {"role": "assistant", "content": "أهلا"}]
    t = format_chat(msgs, tokenizer=None, add_generation_prompt=True)
    assert "مرحبا" in t and "<assistant>" in t

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
