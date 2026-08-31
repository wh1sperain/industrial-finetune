"""评测入口：加载 LoRA adapter，在官方评测集上生成并计算指标。

用法：
    # Probe 快速验证（50 条，M1 先用这个）：
    python -m industrial_ft.eval.run_eval --config configs/eval/industrial_benchmark.yaml --probe
    # 正式 Eval（301 条）：
    python -m industrial_ft.eval.run_eval --config configs/eval/industrial_benchmark.yaml
    # 基座对照（不加载 adapter，测 zero-shot 增量）：
    python -m industrial_ft.eval.run_eval --config configs/eval/industrial_benchmark.yaml --baseline
    # 调整人工抽查条数（默认每模块 3 条，0 关闭）：
    python -m industrial_ft.eval.run_eval --config ... --probe --review 3

指标（分层设计）：
- L0 链路检查：非空率 / EM / contain（仅诊断用，长文生成任务参考性低）
- L1 术语覆盖率：从标准答案自动提取关键术语（引号术语/书名号规范/英文缩写），
  统计模型输出包含比例——长文问答的主指标，可解释、零依赖
- L2 语义相似度：M2 预留（embedding cosine，需 sentence-transformers）
- L3 人工抽查：每模块抽取 N 条生成 markdown 清单，双方人工判定（答对/部分/跑题）

产物：eval_predictions_*.jsonl（全量预测）+ eval_review_*.md（人工抽查清单）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from industrial_ft.utils.config import load_config
from industrial_ft.utils.metrics import build_review_markdown, compute_metrics


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
    parser.add_argument(
        "--review",
        type=int,
        default=3,
        help="L3 人工抽查：每模块抽取条数（默认 3，0 关闭）",
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

    # 2. 加载评测集：按模块分为 6 个 split（concept/forum/law/safety/specifications/synthesis）
    #    合并为一个数据集，split 名即模块名（source），直接用于分模块统计
    ds_dict = load_dataset(eval_dataset)
    ds = concatenate_datasets(list(ds_dict.values()))
    prompts = [f"问：{r['prompt']}" for r in ds]
    references = [r["response"] for r in ds]
    sources = []
    for split_name, part in ds_dict.items():
        sources.extend([split_name] * len(part))
    print(f"[eval] samples: {len(ds)} (模块: {list(ds_dict.keys())})")

    # 3. 分批生成
    predictions: list[str] = []
    for i in range(0, len(prompts), batch_size):
        preds = generate_batch(model, tokenizer, prompts[i : i + batch_size], max_new_tokens)
        predictions.extend(preds)
        print(f"[eval] {min(i + batch_size, len(prompts))}/{len(prompts)}")

    # 4. 指标（L0 链路 + L1 术语覆盖率 + 分模块）
    metrics = compute_metrics(predictions, references, sources)
    print("\n===== 评测结果 =====")
    print(f"总样本   : {metrics['total']}")
    print(f"非空率   : {metrics['non_empty']:.2%}")
    print(f"EM       : {metrics['em']:.2%}   （L0 诊断用，长文任务参考性低）")
    print(f"contain  : {metrics['contain']:.2%}   （L0 诊断用，长文任务参考性低）")
    print(f"术语覆盖率: {metrics['term_coverage']:.2%}   （L1 主指标）")
    print("\n--- 分模块 ---")
    for src, m in metrics["per_module"].items():
        print(
            f"{src:<15} n={m['total']:<4} EM={m['em']:.2%}  contain={m['contain']:.2%}  "
            f"术语覆盖率={m['term_coverage']:.2%}"
        )

    # 5. 保存产物
    out_dir = Path(cfg.get("output_dir", adapter_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = []
    if args.probe:
        tag.append("probe")
    if args.baseline:
        tag.append("baseline")
    suffix = "_" + "_".join(tag) if tag else ""

    # 5a. 全量预测 JSONL（供复核 / L2 语义相似度复用）
    rows = [
        {"prompt": p, "reference": r, "prediction": pred, "source": src}
        for p, r, pred, src in zip(prompts, references, predictions, sources)
    ]
    out_file = out_dir / f"eval_predictions{suffix}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n预测结果已保存: {out_file}")

    # 5b. L3 人工抽查清单（markdown）
    if args.review > 0:
        review_md = build_review_markdown(
            rows, per_module_n=args.review, title="PROBE" if args.probe else "EVAL"
        )
        review_file = out_dir / f"eval_review{suffix}.md"
        review_file.write_text(review_md, encoding="utf-8")
        print(f"人工抽查清单已保存: {review_file}（每模块 {args.review} 条）")


if __name__ == "__main__":
    main()
