"""基于已有预测文件评分（无需模型前向）。

用法：
    python -m industrial_ft.eval.score_predictions \
        --predictions output/qwen2.5-7b-mining-eng-sft/eval_predictions.jsonl
    # 加 L2 语义指标（--semantic-model <本地 embedding 模型路径>）
    # 加 L3 抽查清单（--review N）

输入：eval_predictions*.jsonl（每行 {"prompt","reference","prediction","source"}）
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
    compute_segment_coverage,
    compute_semantic_scores,
    summarize_semantic,
)


def load_predictions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score existing prediction files (no forward pass)")
    parser.add_argument("--predictions", required=True, help="path to eval_predictions*.jsonl")
    parser.add_argument(
        "--semantic-model",
        default="",
        help="本地 embedding 模型路径（如 bge-small-zh-v1.5），启用 L2 语义指标",
    )
    parser.add_argument("--review", type=int, default=0, help="L3 抽查：每模块条数（0 关闭）")
    args = parser.parse_args()

    rows = load_predictions(args.predictions)
    if not rows:
        print(f"[score] 文件为空: {args.predictions}", file=sys.stderr)
        sys.exit(1)
    preds = [r["prediction"] for r in rows]
    refs = [r["reference"] for r in rows]
    srcs = [r.get("source", "unknown") for r in rows]
    print(f"[score] 加载 {len(rows)} 条预测: {args.predictions}")

    metrics = compute_metrics(preds, refs, srcs)
    print("\n===== L0/L1 =====")
    print(f"总样本   : {metrics['total']}")
    print(f"非空率   : {metrics['non_empty']:.2%}")
    print(f"EM       : {metrics['em']:.2%}   （长文任务仅诊断用）")
    print(f"contain  : {metrics['contain']:.2%}   （长文任务仅诊断用）")
    print(f"术语覆盖率: {metrics['term_coverage']:.2%}")
    print("\n--- 分模块 ---")
    for src, m in metrics["per_module"].items():
        print(f"{src:<15} n={m['total']:<4} 术语覆盖率={m['term_coverage']:.2%}")

    if args.semantic_model:
        from sentence_transformers import SentenceTransformer

        print(f"\n[L2] 加载语义模型: {args.semantic_model} ...")
        model = SentenceTransformer(args.semantic_model)
        sims = compute_semantic_scores(preds, refs, model)
        segs = compute_segment_coverage(preds, refs, model)

        print("\n===== L2 语义指标 =====")
        print(f"整文相似度均值: {summarize_semantic(sims)['mean']:.3f}  （长文场景仅参考）")
        seg_summary = summarize_semantic(segs, threshold=0.5)
        print(f"分段要点覆盖率: {sum(segs)/len(segs):.2%}  命中率={seg_summary['pass_rate']:.2%}（seg>=0.5）")
        per_module_seg = defaultdict(list)
        for src, s in zip(srcs, segs):
            per_module_seg[src].append(s)
        print("--- 分模块（分段要点覆盖率）---")
        for src in sorted(per_module_seg):
            v = per_module_seg[src]
            print(f"{src:<15} n={len(v):<4} 覆盖率={sum(v)/len(v):.2%}")

    if args.review > 0:
        out_path = Path(args.predictions).parent / f"eval_review_{Path(args.predictions).stem}.md"
        out_path.write_text(
            build_review_markdown(rows, per_module_n=args.review, title=Path(args.predictions).stem),
            encoding="utf-8",
        )
        print(f"\n抽查清单已保存: {out_path}")


if __name__ == "__main__":
    main()
