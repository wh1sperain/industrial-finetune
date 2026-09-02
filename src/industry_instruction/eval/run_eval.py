from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from industry_instruction.data.loader import format_sample
from industry_instruction.utils.config import load_eval_config

SYSTEM_PROMPT = (
    'You are a helpful assistant. You must output your answer strictly as valid JSON '
    'in the format {"answer": ["choice"]}.'
)


def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_batch(
    model, tokenizer, chat_texts: list[str], max_new_tokens: int, max_length: int
) -> list[str]:
    inputs = tokenizer(
        chat_texts,
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


def parse_answer(text: str):
    """从模型输出中提取 JSON 对象或 list，返回 answer 字段或 None。"""
    match = re.search(r"(\{.*?\}|\[.*?\])", text, flags=re.DOTALL)
    if not match:
        return None
    candidate = match.group(1)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(candidate)
        except Exception:
            return None
    if isinstance(parsed, dict):
        return parsed.get("answer")
    if isinstance(parsed, list):
        return parsed
    return parsed


def normalize_answer_set(answer):
    if answer is None:
        return set()
    if isinstance(answer, str):
        return {answer.strip()}
    if isinstance(answer, (list, tuple)):
        return {str(x).strip() for x in answer}
    return {str(answer).strip()}


def compute_set_metrics(pred_set, ref_set):
    if not ref_set:
        return 0.0, 0.0, 0.0
    intersection = len(pred_set & ref_set)
    union = len(pred_set | ref_set)
    jaccard = intersection / union if union > 0 else 0.0
    precision = intersection / len(pred_set) if pred_set else 0.0
    recall = intersection / len(ref_set) if ref_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    set_match = 1.0 if pred_set == ref_set else 0.0
    return set_match, jaccard, f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate industry_instruction fine-tuned model")
    parser.add_argument(
        "--experiment",
        required=True,
        help="configs/experiments/industry_instruction/ 下的配置名",
    )
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
    eval_dataset_name = cfg.get("eval_dataset_name") or cfg.get("dataset_name")
    eval_revision = cfg.get("eval_dataset_revision", "main")
    max_new_tokens = cfg.get("max_new_tokens", 256)
    batch_size = cfg.get("batch_size", 8)
    max_length = cfg.get("max_length", 2048)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # decoder-only 批量生成必须 left padding，否则 pad token 在末尾会影响生成
    tokenizer.padding_side = "left"

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

    if not eval_dataset_name:
        raise SystemExit("[eval] 配置缺少 eval_dataset_name（HF 数据集 config name）")
    ds = load_dataset(
        eval_dataset,
        name=eval_dataset_name,
        split="test",
        revision=eval_revision,
    )
    print(f"[eval] 评测集: {eval_dataset} / {eval_dataset_name} / test，样本数 {len(ds)}")

    records = []
    for sample in ds:
        prompt = format_sample(sample)["prompt"]
        chat_text = build_chat_prompt(tokenizer, prompt)
        records.append(
            {
                "prompt": prompt,
                "chat_text": chat_text,
                "reference": sample.get("answer", []),
                "documents": sample.get("documents", []),
            }
        )

    chat_texts = [r["chat_text"] for r in records]
    predictions: list[str] = []
    for i in range(0, len(chat_texts), batch_size):
        preds = generate_batch(
            model,
            tokenizer,
            chat_texts[i : i + batch_size],
            max_new_tokens,
            max_length,
        )
        predictions.extend(preds)
        print(f"[eval] {min(i + batch_size, len(chat_texts))}/{len(chat_texts)}")

    for record, pred in zip(records, predictions):
        parsed = parse_answer(pred)
        pred_set = normalize_answer_set(parsed)
        ref_set = normalize_answer_set(record["reference"])
        set_match, jaccard, f1 = compute_set_metrics(pred_set, ref_set)
        record.update(
            {
                "prediction_raw": pred,
                "prediction_parsed": parsed,
                "set_match": set_match,
                "jaccard": jaccard,
                "f1": f1,
            }
        )

    out_dir = Path(cfg.get("output_dir", adapter_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = []
    if args.baseline:
        tag.append("baseline")
    suffix = "_" + "_".join(tag) if tag else ""

    rows = [
        {
            "prompt": r["prompt"],
            "reference": r["reference"],
            "prediction": r["prediction_raw"],
            "prediction_parsed": r["prediction_parsed"],
            "document": r["documents"],
            "set_match": r["set_match"],
            "jaccard": r["jaccard"],
            "f1": r["f1"],
        }
        for r in records
    ]
    out_file = out_dir / f"eval_predictions{suffix}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n预测结果已保存: {out_file}")

    n = len(records)
    if n:
        print("\n===== Set-Match / Jaccard / F1 =====")
        print(f"Set-Match Accuracy: {sum(r['set_match'] for r in records) / n:.2%}")
        print(f"Jaccard Similarity: {sum(r['jaccard'] for r in records) / n:.2%}")
        print(f"F1               : {sum(r['f1'] for r in records) / n:.2%}")


if __name__ == "__main__":
    main()
