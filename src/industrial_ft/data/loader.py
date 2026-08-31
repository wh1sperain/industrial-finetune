"""公开工业数据集加载（Mining-Engineering 系列）。

TRL 1.x 约定训练集含 prompt/completion 字段对（loss 只对 completion 计算）。
"""
from __future__ import annotations

import argparse

from datasets import load_dataset

DEFAULT_DATASET = "acnul/Mining-Engineering-SFT"
DEFAULT_REVISION = "main"


def format_sample(sample: dict) -> dict:
    """原始字段 prompt/response/source → {"prompt": "问：…", "completion": "答：…"}（source 列保留）。"""
    return {
        "prompt": f"问：{sample.get('prompt', '').strip()}",
        "completion": f"答：{sample.get('response', '').strip()}",
    }


def download(dataset: str = DEFAULT_DATASET, revision: str = DEFAULT_REVISION) -> None:
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
    ds = load_dataset(dataset, split="train", revision=revision)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds.map(lambda s: format_sample(s), remove_columns=["response"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a public dataset from HuggingFace")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    download(args.dataset, args.revision)


if __name__ == "__main__":
    main()
