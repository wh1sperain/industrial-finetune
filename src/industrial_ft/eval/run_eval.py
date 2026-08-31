"""评测入口：加载 LoRA adapter，在官方评测集上生成并计算指标。

用法：
    # Probe 快速验证（50 条，M1 先用这个）：
    python -m industrial_ft.eval.run_eval --config configs/eval/industrial_benchmark.yaml --probe
    # 正式 Eval（301 条）：
    python -m industrial_ft.eval.run_eval --config configs/eval/industrial_benchmark.yaml
    # 基座对照（不加载 adapter，测 zero-shot 增量）：
    python -m industrial_ft.eval.run_eval --config configs/eval/industrial_benchmark.yaml --baseline

指标（M1 简化版；企业方确认工业口径后可扩展 compute_metrics）：
- 非空率  ：模型输出非空比例（链路有效性）
- EM      ：规范化后完全匹配率
- contain ：模型输出包含标准答案的比例（仅统计标准答案长度 ≥4 的样本，防短答案误判）
- 分模块  ：按 source 六类分别统计（定位能力短板）
- 预测结果保存到 JSONL，供人工抽查
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from industrial_ft.utils.config import load_config


def normalize(text: str) -> str:
    """规范化：去空白与标点，用于严格匹配。"""
    text = re.sub(r"\s+", "", text or "")
    text = re.sub(r"[，。！？、；：""''（）()《》【】\-—\s]", "", text)
    return text


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    """批量生成（SDPA + bf16），返回回答文本列表。"""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
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


def compute_metrics(predictions: list[str], references: list[str], sources: list[str]) -> dict:
    """计算非空率 / EM / contain 及六模块分项。"""
    total = len(references)
    non_empty = sum(1 for p in predictions if p.strip())
    em = contain = 0
    per_module = defaultdict(lambda: {"total": 0, "em": 0, "contain": 0})

    for pred, ref, src in zip(predictions, references, sources):
        p_norm, r_norm = normalize(pred), normalize(ref)
        if r_norm and r_norm == p_norm:
            em += 1
        if len(r_norm) >= 4 and r_norm in p_norm:
            contain += 1
        m = per_module[src or "unknown"]
        m["total"] += 1
        if r_norm and r_norm == p_norm:
            m["em"] += 1
        if len(r_norm) >= 4 and r_norm in p_norm:
            m["contain"] += 1

    metrics = {
        "total": total,
        "non_empty": non_empty / total if total else 0,
        "em": em / total if total else 0,
        "contain": contain / total if total else 0,
    }
    metrics["per_module"] = {
        src: {
            "total": m["total"],
            "em": m["em"] / m["total"] if m["total"] else 0,
            "contain": m["contain"] / m["total"] if m["total"] else 0,
        }
        for src, m in sorted(per_module.items())
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model (TRL 1.x project)")
    parser.add_argument("--config", required=True, help="path to yaml config under configs/eval/")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="快速验证模式：用 Probe 集（50 条）代替完整 Eval 集",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="不加载 adapter，评测基座模型（zero-shot 对照）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model_name_or_path"]
    # 模型路径/输出目录由 utils.config 统一注入（单一事实源：configs/models.yaml）
    adapter_path = cfg.get("adapter_path") or cfg["adapter_dir"]
    eval_dataset = cfg["eval_dataset"]
    eval_revision = cfg.get("eval_dataset_revision", "main")
    max_new_tokens = cfg.get("max_new_tokens", 256)
    batch_size = cfg.get("batch_size", 8)

    if args.probe:
        eval_dataset = cfg["probe_dataset"]
        print(f"[eval] PROBE 模式: {eval_dataset}")
    else:
        print(f"[eval] 完整 Eval 模式: {eval_dataset}@{eval_revision}")

    # 1. 加载模型（bf16 + SDPA，与训练一致；严禁 flash_attention_2）
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    if not args.baseline:
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"[eval] adapter loaded: {adapter_path}")
    else:
        print("[eval] baseline 模式: 基座 zero-shot（不加载 adapter）")
    model.eval()

    # 2. 加载评测集（prompt / response / source 三字段）
    ds = load_dataset(eval_dataset, split="train")
    prompts = [f"问：{r['prompt']}" for r in ds]
    references = [r["response"] for r in ds]
    sources = [r.get("source", "unknown") for r in ds]
    print(f"[eval] samples: {len(ds)}")

    # 3. 分批生成
    predictions: list[str] = []
    for i in range(0, len(prompts), batch_size):
        preds = generate_batch(model, tokenizer, prompts[i : i + batch_size], max_new_tokens)
        predictions.extend(preds)
        print(f"[eval] {min(i + batch_size, len(prompts))}/{len(prompts)}")

    # 4. 指标
    metrics = compute_metrics(predictions, references, sources)
    print("\n===== 评测结果 =====")
    print(f"总样本: {metrics['total']}")
    print(f"非空率 : {metrics['non_empty']:.2%}")
    print(f"EM     : {metrics['em']:.2%}")
    print(f"contain: {metrics['contain']:.2%}")
    print("\n--- 分模块 ---")
    for src, m in metrics["per_module"].items():
        print(f"{src:<15} n={m['total']:<4} EM={m['em']:.2%}  contain={m['contain']:.2%}")

    # 5. 保存预测（供人工抽查 / 企业方指标复核）
    out_dir = Path(cfg.get("output_dir", adapter_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = []
    if args.probe:
        tag.append("probe")
    if args.baseline:
        tag.append("baseline")
    out_file = out_dir / ("eval_predictions_" + "_".join(tag) + ".jsonl" if tag else "eval_predictions.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for p, r, pred, src in zip(prompts, references, predictions, sources):
            f.write(
                json.dumps(
                    {"prompt": p, "reference": r, "prediction": pred, "source": src},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"\n预测结果已保存: {out_file}")


if __name__ == "__main__":
    main()
