"""التدريب: SFT + DPO فوق TRL مع إعدادات السرعة."""
from __future__ import annotations
from .config import ModeTrainsConfig


def _training_args(cfg: ModeTrainsConfig):
    from transformers import TrainingArguments
    bf16 = cfg.bf16
    if bf16 is None:
        try:
            import torch
            bf16 = torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8
        except Exception:
            bf16 = False
    return TrainingArguments(
        per_device_train_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        learning_rate=cfg.lr,
        fp16=(cfg.fp16 and not bf16),
        bf16=bool(bf16),
        logging_steps=cfg.logging_steps,
        optim=cfg.optim,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler,
        seed=cfg.seed,
        output_dir=cfg.output_dir,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        group_by_length=True,  # ترتيب حسب الطول = أقل حشو = أسرع
        dataloader_num_workers=2,
        report_to="none",
    )


def train_sft(model, tokenizer, dataset, cfg: ModeTrainsConfig, **sft_kwargs):
    """تدريب SFT سريع. dataset يجب أن يحوي عمود text (أو حسب text_field)."""
    from trl import SFTTrainer
    args = _training_args(cfg)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=args,
        max_seq_length=cfg.max_seq_length,
        packing=bool(cfg.packing),
        dataset_num_proc=cfg.dataset_num_proc,
        **sft_kwargs,
    )
    tr = trainer.train()
    return trainer, tr


def train_dpo(model, tokenizer, dataset, cfg: ModeTrainsConfig, beta: float = 0.1):
    """تدريب DPO (chosen/rejected). dataset: أعمدة prompt/chosen/rejected."""
    from trl import DPOTrainer, DPOConfig
    from transformers import TrainingArguments  # noqa
    args = DPOConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        logging_steps=cfg.logging_steps,
        bf16=(cfg.bf16 if cfg.bf16 is not None else False),
        optim=cfg.optim,
        seed=cfg.seed,
        beta=beta,
        report_to="none",
    )
    trainer = DPOTrainer(model=model, ref_model=None, args=args,
                         train_dataset=dataset, processing_class=tokenizer)
    tr = trainer.train()
    return trainer, tr


def train_grpo(model, tokenizer, dataset, cfg: ModeTrainsConfig, reward_fn=None, **kw):
    """GRPO إن توفر في نسخة TRL، وإلا رسالة واضحة."""
    try:
        from trl import GRPOTrainer, GRPOConfig
    except Exception as e:
        raise ImportError("نسخة TRL الحالية لا تدعم GRPO. حدّث trl>=0.9") from e
    args = GRPOConfig(output_dir=cfg.output_dir, max_steps=cfg.max_steps,
                      per_device_train_batch_size=cfg.per_device_batch,
                      gradient_accumulation_steps=cfg.grad_accum,
                      learning_rate=cfg.lr, logging_steps=cfg.logging_steps,
                      report_to="none", **kw)
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=reward_fn,
                          args=args, train_dataset=dataset)
    return trainer, trainer.train()
