"""评测入口（骨架）。

用法：
    python -m industrial_ft.eval.run_eval --config configs/eval/industrial_benchmark.yaml

待办（M1 与 M2）：
- 工业指标口径由企业方确认（准确率/召回/F1/成本等），在此实现 compute_metrics
- 加载 LoRA adapter 后跑评测集，结果写 wandb 与 docs/实验记录.md
"""
from __future__ import annotations

import argparse

import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics(predictions, references) -> dict:
    """TODO(M2): 按企业确认的工业指标口径实现。"""
    raise NotImplementedError("评测指标待企业方确认口径后实现")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model")
    parser.add_argument("--config", required=True, help="path to yaml config under configs/eval/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[eval] config loaded: {cfg.get('name', 'unnamed')}")
    print(f"[eval] model: {cfg['model_name_or_path']} + adapter: {cfg['adapter_path']}")
    print("[eval] NotImplementedError: 指标口径待企业方确认（M2）")


if __name__ == "__main__":
    main()
