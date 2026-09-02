from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from industry_instruction.utils.config import load_eval_config
from industry_instruction.data.loader import normalize_documents, format_sample

def generate_batch(
    model, tokenizer, prompts: list[str], max_new_tokens: int, max_length: int
) -> list[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    results = []
    for i, out in enumerate(outputs):
        gen = out[inputs["input_ids"].shape[1] :]
        results.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    return results

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model (TRL 1.x project)")
    parser.add_argument("--experiment", required=True, help="configs/experiments/industry_instruction/ 下的配置名")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="不加载 adapter，评测基座模型",
    )
    args = parser.parse_args()
    
    cfg = load_eval_config(args.experiment)
    model_name = cfg["model_name_or_path"]
    adapter_path = cfg.get("adapter_path") or cfg["adapter_dir"]
    eval_dataset = cfg["eval_dataset"]
    eval_revision = cfg.get("eval_dataset_revision", "main")
    max_new_tokens = cfg.get("max_new_tokens", 256)
    batch_size = cfg.get("batch_size", 8)
    max_length = cfg.get("max_length", 2048)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float32,
        attn_implementation="sdpa",
        device_map="auto",
    )
    if not args.baseline:
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"[eval] adapter loaded: {adapter_path}")
    else:
        print("[eval] baseline 模式")
    model.eval()

    ds_dict = load_dataset(eval_dataset, split="test", revision=eval_revision)
    ds = []
    for sample in ds_dict:
        formatted_sample = format_sample(sample)
        ds.append(formatted_sample)
    
    prompts = [r.get("prompt", "") for r in ds]
    answers = [r.get("completion", "") for r in ds]
    documents = [r.get("documents", "") for r in ds_dict]
    

    predictions: list[str] = []
    for i in range(0, len(prompts), batch_size):
        preds = generate_batch(model, tokenizer, prompts[i : i + batch_size], max_new_tokens, max_length)
        predictions.extend(preds)
        print(f"[eval] {min(i + batch_size, len(prompts))}/{len(prompts)}")


    out_dir = Path(cfg.get("output_dir", adapter_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = []
    if args.baseline:
        tag.append("baseline")
    suffix = "_" + "_".join(tag) if tag else ""

    rows = [
        {"prompt": p, "answer": a, "prediction": pred, "document": doc}
        for p, a, pred, doc in zip(prompts, answers, predictions, documents)
    ]
    out_file = out_dir / f"eval_predictions{suffix}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n预测结果已保存: {out_file}")

    


if __name__ == "__main__":
    main()