"""公开工业数据集下载与训练集加载。

M1 数据已定案：Mining-Engineering 矿建工程系列（中文，MIT，5,287 条）：
    https://huggingface.co/datasets/acnul/Mining-Engineering-SFT
配套评测集：acnul/Mining-Engineering-Eval（301 条）、Probe（50 条）

格式约定：字段 prompt（问题）/ response（回答）/ source（六模块分类），
统一模板化为 Qwen 训练文本，见 ``format_sample``。
"""
from __future__ import annotations

import argparse

from datasets import load_dataset

# M1 定案数据：矿建工程领域中文指令集（law/specifications/concept/safety/forum/synthesis 六模块）
DEFAULT_DATASET = "acnul/Mining-Engineering-SFT"
DEFAULT_REVISION = "main"


def format_sample(sample: dict) -> str:
    """把 Mining-Engineering 样本格式化为模型训练文本。

    字段：prompt（指令/问题）、response（标准回答）、source（知识来源分类）。
    训练时保留 source 列，用于分模块能力分析（见 docs/训练方案.md §8.1）。
    """
    prompt = sample.get("prompt", "").strip()
    response = sample.get("response", "").strip()
    return f"问：{prompt}\n答：{response}"


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
    """加载并模板化训练集（保留 source 列供分模块分析）。

    企业私有数据接入后：改为从 os.environ.get("DATA_ROOT", "./data") 读取本地文件，
    并在此处统一脱敏检查。
    """
    ds = load_dataset(dataset, split="train", revision=revision)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    keep = [c for c in ds.column_names if c != "prompt" and c != "response"]
    return ds.map(lambda s: {"text": format_sample(s)}, remove_columns=keep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a public dataset from HuggingFace")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    download(args.dataset, args.revision)


if __name__ == "__main__":
    main()
