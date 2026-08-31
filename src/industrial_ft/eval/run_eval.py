"""评测入口：加载 LoRA adapter，在官方评测集上生成并计算指标。

用法：
    # Probe（50 条）：
    python -m industrial_ft.eval.run_eval --config configs/eval/industrial_benchmark.yaml --probe
    # 正式 Eval（301 条）：
    python -m industrial_ft.eval.run_eval --config configs/eval/industrial_benchmark.yaml
    # 基座对照（不加载 adapter）：
    python -m industrial_ft.eval.run_eval --config configs/eval/industrial_benchmark.yaml --baseline
    # L2 语义指标（--semantic，需 config 的 semantic_model_path 本地 embedding 模型）
    # L3 人工抽查（--review N，默认每模块 3 条，0 关闭）

产物：eval_predictions_*.jsonl（全量预测）+ eval_review_*.md（抽查清单）
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
from industrial_ft.utils.metrics import (
    build_review_markdown,
    compute_metrics,
    compute_segment_coverage,
    compute_semantic_scores,
    summarize_semantic,
)


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
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
        help="用 Probe 集（50 条）代替完整 Eval 集",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="不加载 adapter，评测基座模型",
    )
    parser.add_argument(
        "--review",
        type=int,
        default=3,
        help="L3 人工抽查：每模块抽取条数（默认 3，0 关闭）",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="L2 语义指标（需 sentence-transformers + config 的 semantic_model_path）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model_name_or_path"]
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

    # 1. 模型（bf16 + SDPA；Blackwell 不兼容 flash_attention_2）
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
        print("[eval] baseline 模式")
    model.eval()

    # 2. 评测集：6 个模块 split（concept/forum/law/safety/specifications/synthesis），split 名即 source
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

    # 4. L0/L1
    metrics = compute_metrics(predictions, references, sources)
    print("\n===== L0/L1 =====")
    print(f"总样本   : {metrics['total']}")
    print(f"非空率   : {metrics['non_empty']:.2%}")
    print(f"EM       : {metrics['em']:.2%}   （长文任务仅诊断用）")
    print(f"contain  : {metrics['contain']:.2%}   （长文任务仅诊断用）")
    print(f"术语覆盖率: {metrics['term_coverage']:.2%}")
    print("\n--- 分模块 ---")
    for src, m in metrics["per_module"].items():
        print(f"{src:<15} n={m['total']:<4} 术语覆盖率={m['term_coverage']:.2%}")

    # 5. L2 语义指标（可选）
    if args.semantic:
        from collections import defaultdict as _dd

        from sentence_transformers import SentenceTransformer

        sem_path = cfg.get("semantic_model_path", "")
        if not sem_path:
            raise SystemExit("[eval] --semantic 需要 config 的 semantic_model_path")
        print(f"\n[L2] 加载语义模型: {sem_path} ...")
        sem_model = SentenceTransformer(sem_path, device="cuda" if torch.cuda.is_available() else "cpu")
        sims = compute_semantic_scores(predictions, references, sem_model)
        segs = compute_segment_coverage(predictions, references, sem_model)

        print("\n===== L2 语义指标 =====")
        print(f"整文相似度均值: {summarize_semantic(sims)['mean']:.3f}  （长文场景仅参考）")
        seg_summary = summarize_semantic(segs, threshold=0.5)
        print(f"分段要点覆盖率: {sum(segs)/len(segs):.2%}  命中率={seg_summary['pass_rate']:.2%}（seg>=0.5）")
        per_module_seg = _dd(list)
        for src, s in zip(sources, segs):
            per_module_seg[src].append(s)
        print("--- 分模块（分段要点覆盖率）---")
        for src in sorted(per_module_seg):
            v = per_module_seg[src]
            print(f"{src:<15} n={len(v):<4} 覆盖率={sum(v)/len(v):.2%}")

    # 6. 产物
    out_dir = Path(cfg.get("output_dir", adapter_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = []
    if args.probe:
        tag.append("probe")
    if args.baseline:
        tag.append("baseline")
    suffix = "_" + "_".join(tag) if tag else ""

    rows = [
        {"prompt": p, "reference": r, "prediction": pred, "source": src}
        for p, r, pred, src in zip(prompts, references, predictions, sources)
    ]
    out_file = out_dir / f"eval_predictions{suffix}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n预测结果已保存: {out_file}")

    if args.review > 0:
        review_file = out_dir / f"eval_review{suffix}.md"
        review_file.write_text(
            build_review_markdown(rows, per_module_n=args.review, title="PROBE" if args.probe else "EVAL"),
            encoding="utf-8",
        )
        print(f"抽查清单已保存: {review_file}")


if __name__ == "__main__":
    main()
