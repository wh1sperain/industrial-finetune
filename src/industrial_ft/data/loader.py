"""公开工业数据集下载与训练集加载。

M1 数据已定案：Mining-Engineering 矿建工程系列（中文，MIT，5,287 条）：
    https://huggingface.co/datasets/acnul/Mining-Engineering-SFT
配套评测集：acnul/Mining-Engineering-Eval（301 条）、Probe（50 条）

TRL 1.x 数据约定：SFTTrainer 要求训练集含 prompt（指令）/ completion（回答）字段对，
loss 只对 completion 计算（prompt 部分不参与）。
"""
from __future__ import annotations

import argparse

from datasets import load_dataset

# M1 定案数据：矿建工程领域中文指令集（law/specifications/concept/safety/forum/synthesis 六模块）
DEFAULT_DATASET = "acnul/Mining-Engineering-SFT"
DEFAULT_REVISION = "main"


def format_sample(sample: dict) -> dict:
    """把 Mining-Engineering 样本拆分为 TRL 1.x 的 prompt/completion 格式。

    原始字段：prompt（指令/问题）、response（标准回答）、source（知识来源分类）。
    返回：{"prompt": "问：…", "completion": "答：…"}，source 列保留用于分模块分析。
    """
    return {
        "prompt": f"问：{sample.get('prompt', '').strip()}",
        "completion": f"答：{sample.get('response', '').strip()}",
    }


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
    """加载并拆分为 prompt/completion 训练集（保留 source 列供分模块分析）。

    企业私有数据接入后：改为从 os.environ.get("DATA_ROOT", "./data") 读取本地文件，
    并在此处统一脱敏检查。
    """
    ds = load_dataset(dataset, split="train", revision=revision)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    # 去掉原始 response（已并入 completion），保留 source 列
    return ds.map(lambda s: format_sample(s), remove_columns=["response"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a public dataset from HuggingFace")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    download(args.dataset, args.revision)


if __name__ == "__main__":
    main()
