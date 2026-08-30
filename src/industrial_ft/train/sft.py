"""SFT 训练入口：TRL SFTTrainer + LoRA（5090 加速版，适配 TRL 1.x）。

用法：
    python -m industrial_ft.train.sft --config configs/train/lora_qwen7b_industrial.yaml
    # 或 bash scripts/train.sh [config]
    # 断点续训：python -m industrial_ft.train.sft --config <cfg> --resume

一切参数来自 yaml 配置，代码不硬编码实验参数。
加速开关（见 docs/训练方案.md §5.2）：torch_compile / optim / dataloader_num_workers /
attn_implementation（默认 sdpa，勿用 flash_attention_2——Blackwell sm_120 不支持）。

TRL 1.x API 注意（与 0.10~0.17 的差异）：
- warmup_ratio → warmup_steps
- max_seq_length → max_length
- SFTTrainer(tokenizer=...) → processing_class
"""
from __future__ import annotations

import argparse

import yaml
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from industrial_ft.data.loader import load_train_dataset
from industrial_ft.utils.seed import set_seed


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_lora_config(cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        task_type="CAUSAL_LM",
    )


def build_model_and_tokenizer(cfg: dict):
    model_name = cfg["model_name_or_path"]
    trust_remote_code = cfg.get("trust_remote_code", False)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if cfg.get("qlora", False):
        # 注意：5090 (sm_120) 上 NF4 有性能惩罚且依赖版本敏感，默认不用（见方案 §4.2）
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )

    # 注意力实现：默认 sdpa（PyTorch 2.7+ 对 Blackwell 支持良好）；
    # 严禁 flash_attention_2（sm_120 不兼容，会直接报错）
    attn_implementation = cfg.get("attn_implementation", "sdpa")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        quantization_config=quantization_config,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT with TRL (LoRA/QLoRA)")
    parser.add_argument("--config", required=True, help="path to yaml config under configs/train/")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 output_dir 内最新 checkpoint 断点续训（自动找 checkpoint-* 目录）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    model, tokenizer = build_model_and_tokenizer(cfg)
    train_dataset = load_train_dataset(
        dataset=cfg.get("dataset", "gsm8k"),
        revision=cfg.get("dataset_revision", "main"),
        max_samples=cfg.get("max_train_samples"),
    )

    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        # TRL 1.x：warmup_ratio 已移除，改用 warmup_steps（config 默认 30 ≈ 3% @ 全局 batch 16）
        warmup_steps=cfg.get("warmup_steps", 30),
        num_train_epochs=cfg.get("num_train_epochs", 3.0),
        weight_decay=cfg.get("weight_decay", 0.01),
        bf16=cfg.get("bf16", False),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 2),
        report_to=cfg.get("report_to", "none"),
        run_name=cfg.get("run_name"),
        # TRL 1.x：max_seq_length 已改名为 max_length
        max_length=cfg.get("max_length", 2048),
        # TRL 1.x：数据集直接提供 prompt/completion 字段（见 loader.py），无需 dataset_text_field
        seed=cfg.get("seed", 42),
        # ---- 5090 加速开关（docs/训练方案.md §5.2）----
        torch_compile=cfg.get("torch_compile", False),
        optim=cfg.get("optim", "adamw_torch"),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,  # TRL 1.x：原 tokenizer= 参数已改名
        peft_config=build_lora_config(cfg),
    )

    if args.resume:
        resume_from_checkpoint = True  # 自动定位 output_dir 内最新 checkpoint
        print("[sft] resume: 从 output_dir 最新 checkpoint 续训")
    else:
        resume_from_checkpoint = None

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(cfg["output_dir"])
    print(f"Training done. Adapter saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
