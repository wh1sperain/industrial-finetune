"""训练数据集加载（TRL 1.x 格式：prompt/completion 字段对，loss 只对 completion 计算）。

dataset 支持两种来源：
- 本地路径（目录/文件）：直接 load_dataset 加载，用户自行准备
- HF 数据集 ID：在线加载（需网络）
"""
from __future__ import annotations

import os

from datasets import load_dataset


def format_sample(sample: dict) -> dict:
    """原始字段 prompt/response/source → {"prompt": "问：…", "completion": "答：…"}（source 列保留）。"""
    return {
        "prompt": f"问：{sample.get('prompt', '').strip()}",
        "completion": f"答：{sample.get('response', '').strip()}",
    }


def load_train_dataset(
    dataset: str,
    revision: str = "main",
    max_samples: int | None = None,
):
    """加载并拆分为 prompt/completion 训练集。dataset 为本地路径或 HF ID。"""
    if os.path.isdir(dataset) or os.path.isfile(dataset):
        ds = load_dataset(dataset, split="train")
    else:
        ds = load_dataset(dataset, split="train", revision=revision)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds.map(lambda s: format_sample(s), remove_columns=["response"])
