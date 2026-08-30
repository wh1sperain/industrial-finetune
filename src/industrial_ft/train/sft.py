"""SFT 训练入口：TRL SFTTrainer + LoRA/QLoRA。

用法：
    python -m industrial_ft.train.sft --config configs/train/lora_qwen7b_industrial.yaml
    # 或 bash scripts/train.sh [config]

一切参数来自 yaml 配置，代码不硬编码实验参数。
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
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        quantization_config=quantization_config,
        device_map="auto",
    )
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT with TRL (LoRA/QLoRA)")
    parser.add_argument("--config", required=True, help="path to yaml config under configs/train/")
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
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        num_train_epochs=cfg.get("num_train_epochs", 3.0),
        bf16=cfg.get("bf16", False),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 2),
        report_to=cfg.get("report_to", "none"),
        run_name=cfg.get("run_name"),
        max_seq_length=cfg.get("max_seq_length", 1024),
        dataset_text_field="text",
        seed=cfg.get("seed", 42),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        peft_config=build_lora_config(cfg),
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"Training done. Adapter saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
