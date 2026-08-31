"""基于已有预测文件重新评分（无需再次模型前向生成）。

用法：
    # L0 + L1（术语覆盖率，零依赖）：
    python -m industrial_ft.eval.score_predictions \
        --predictions output/qwen2.5-7b-mining-eng-sft/eval_predictions.jsonl

    # 加 L2 语义相似度（需本地 embedding 模型，如 bge-small-zh-v1.5）：
    python -m industrial_ft.eval.score_predictions \
        --predictions output/qwen2.5-7b-mining-eng-sft/eval_predictions.jsonl \
        --semantic-model /root/autodl-tmp/modelscope/BAAI/bge-small-zh-v1.5

    # 同时生成人工抽查清单：
    python -m industrial_ft.eval.score_predictions \
        --predictions .../eval_predictions.jsonl --review 3

输入：eval_predictions*.jsonl（每行 {"prompt","reference","prediction","source"}）
输出：控制台指标报告 + 可选 eval_review_{stem}.md（与预测文件同目录）
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from industrial_ft.utils.metrics import (
    build_review_markdown,
    compute_metrics,
    compute_semantic_scores,
    summarize_semantic,
)


def load_predictions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score existing prediction files (no model forward pass)")
    parser.add_argument("--predictions", required=True, help="path to eval_predictions*.jsonl")
    parser.add_argument(
        "--semantic-model",
        default="",
        help="本地 embedding 模型路径（如 bge-small-zh-v1.5），启用 L2 语义相似度",
    )
    parser.add_argument(
        "--review",
        type=int,
        default=0,
        help="L3 人工抽查：每模块抽取条数（默认 0 关闭）",
    )
    args = parser.parse_args()

    rows = load_predictions(args.predictions)
    if not rows:
        print(f"[score] 文件为空: {args.predictions}", file=sys.stderr)
        sys.exit(1)
    preds = [r["prediction"] for r in rows]
    refs = [r["reference"] for r in rows]
    srcs = [r.get("source", "unknown") for r in rows]
    print(f"[score] 加载 {len(rows)} 条预测: {args.predictions}")

    # L0 + L1
    metrics = compute_metrics(preds, refs, srcs)
    print("\n===== L0/L1 指标 =====")
    print(f"总样本   : {metrics['total']}")
    print(f"非空率   : {metrics['non_empty']:.2%}")
    print(f"EM       : {metrics['em']:.2%}   （L0 诊断用）")
    print(f"contain  : {metrics['contain']:.2%}   （L0 诊断用）")
    print(f"术语覆盖率: {metrics['term_coverage']:.2%}   （L1）")
    print("\n--- 分模块 ---")
    for src, m in metrics["per_module"].items():
        print(f"{src:<15} n={m['total']:<4} 术语覆盖率={m['term_coverage']:.2%}")

    # L2 语义相似度（可选）
    if args.semantic_model:
        from sentence_transformers import SentenceTransformer

        print(f"\n[L2] 加载语义模型: {args.semantic_model} ...")
        model = SentenceTransformer(args.semantic_model)
        sims = compute_semantic_scores(preds, refs, model)
        summary = summarize_semantic(sims)
        per_module_sims = defaultdict(list)
        for src, s in zip(srcs, sims):
            per_module_sims[src].append(s)

        print("\n===== L2 语义相似度 =====")
        print(f"均值    : {summary['mean']:.3f}")
        print(f"达标率  : {summary['pass_rate']:.2%}   （sim >= 0.6）")
        print("\n--- 分模块 ---")
        for src in sorted(per_module_sims):
            m_sim = summarize_semantic(per_module_sims[src])
            print(f"{src:<15} n={len(per_module_sims[src]):<4} 均值={m_sim['mean']:.3f}  达标率={m_sim['pass_rate']:.2%}")

    # L3 人工抽查（可选）
    if args.review > 0:
        out_path = Path(args.predictions).parent / f"eval_review_{Path(args.predictions).stem}.md"
        md = build_review_markdown(rows, per_module_n=args.review, title=Path(args.predictions).stem)
        out_path.write_text(md, encoding="utf-8")
        print(f"\n人工抽查清单已保存: {out_path}")


if __name__ == "__main__":
    main()
