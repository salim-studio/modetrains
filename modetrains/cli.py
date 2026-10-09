"""سطر أوامر modetrains: train / infer / info."""
from __future__ import annotations
import argparse, json
from .config import ModeTrainsConfig
from .utils import get_device_info, estimate_vram


def cmd_info(_a):
    info = get_device_info()
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(json.dumps(estimate_vram("7B", True, 2048), ensure_ascii=False, indent=2))


def cmd_train(a):
    from .models import FastModel
    from .data import load_chat_dataset
    from .trainers import train_sft
    from .export import save_merged
    cfg = ModeTrainsConfig(model_name=a.model, max_seq_length=a.max_seq,
                           max_steps=a.steps, output_dir=a.out,
                           per_device_batch=a.batch, grad_accum=a.accum)
    print(f"[modetrains] تحميل {cfg.model_name} ...")
    model, tok = FastModel.from_pretrained(cfg)
    model = FastModel.get_peft_model(model, cfg)
    print(f"[modetrains] dataset: {a.data}")
    ds = load_chat_dataset(a.data, split=a.split)
    # توحيد عمود text إن لزم
    if "text" not in ds.column_names:
        def _map(ex):
            if "messages" in ex:
                from .data import format_chat
                return {"text": format_chat(ex["messages"], tok)}
            instr = ex.get("instruction", ex.get("prompt", ""))
            outp = ex.get("output", ex.get("completion", ex.get("response", "")))
            from .data import format_alpaca
            return {"text": format_alpaca(instr, ex.get("input", ""), outp)}
        ds = ds.map(_map, num_proc=cfg.dataset_num_proc)
    trainer, stats = train_sft(model, tok, ds, cfg, dataset_text_field="text")
    print(stats)
    save_merged(model, tok, a.out)
    print(f"[modetrains] تم الحفظ في {a.out}")


def cmd_infer(a):
    from .models import FastModel
    from .infer import generate
    cfg = ModeTrainsConfig(model_name=a.model, max_seq_length=2048, load_in_4bit=not a.full)
    model, tok = FastModel.from_pretrained(cfg)
    print(generate(model, tok, a.prompt, max_new_tokens=a.tokens))


def build_parser():
    p = argparse.ArgumentParser(prog="modetrains", description="Fast LLM training (Unsloth-like)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("info"); s.set_defaults(fn=cmd_info)
    t = sub.add_parser("train"); t.set_defaults(fn=cmd_train)
    t.add_argument("--model", required=True); t.add_argument("--data", required=True)
    t.add_argument("--split", default="train"); t.add_argument("--out", default="outputs")
    t.add_argument("--max-seq", type=int, default=2048); t.add_argument("--steps", type=int, default=60)
    t.add_argument("--batch", type=int, default=2); t.add_argument("--accum", type=int, default=4)
    i = sub.add_parser("infer"); i.set_defaults(fn=cmd_infer)
    i.add_argument("--model", required=True); i.add_argument("--prompt", required=True)
    i.add_argument("--tokens", type=int, default=256); i.add_argument("--full", action="store_true")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)

if __name__ == "__main__":
    main()
