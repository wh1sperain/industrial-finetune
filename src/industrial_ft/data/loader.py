"""公开工业数据集下载与训练集加载。

当前以 GSM8K 作为流程占位（与工作区既有实验一致），
M1 阶段替换为实际的公开工业数据集（改 DEFAULT_DATASET 即可）。

格式约定：训练样本最终统一为对话模板化文本
（qwen 模板："Question: ...\nAnswer: ..."），见 ``format_sample``。
"""
from __future__ import annotations

import argparse
import os

from datasets import load_dataset

# TODO(M1): 替换为实际公开工业数据集的 HF 路径
DEFAULT_DATASET = "gsm8k"
DEFAULT_REVISION = "main"


def format_sample(sample: dict) -> str:
    """把原始样本格式化为模型训练文本（占位实现，按数据口径调整）。"""
    return (
        f"Question: {sample['question']}\n"
        f"Answer: {sample['answer']}"
    )


def download(dataset: str = DEFAULT_DATASET, revision: str = DEFAULT_REVISION) -> None:
    """下载数据集到 HF 缓存（不落仓库），打印各 split 规模。"""
    print(f"Downloading {dataset}@{revision} ...")
    ds = load_dataset(dataset, revision=revision)
    for split, data in ds.items():
        print(f"  {split}: {len(data)} samples")
    print("Done.")


def load_train_dataset(
    dataset: str = DEFAULT_DATASET,
    revision: str = DEFAULT_REVISION,
    max_samples: int | None = None,
):
    """加载并模板化训练集。

    企业私有数据接入后：改为从 os.environ.get("DATA_ROOT", "./data") 读取本地文件，
    并在此处统一脱敏检查。
    """
    ds = load_dataset(dataset, split="train", revision=revision)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds.map(lambda s: {"text": format_sample(s)}, remove_columns=ds.column_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a public dataset from HuggingFace")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    download(args.dataset, args.revision)


if __name__ == "__main__":
    main()
