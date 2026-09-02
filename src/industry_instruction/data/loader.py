from __future__ import annotations

import os

from datasets import load_dataset

def normalize_documents(documents):
    if isinstance(documents, str):
        documents = [documents]
    flat_docs = []
    for doc in documents:
        if isinstance(doc, list):
            flat_docs.append(" ".join(map(str, doc)))
        elif isinstance(doc, dict):
            flat_docs.append(" ".join(f"{k}: {v}" for k, v in doc.items()))
        else:
            flat_docs.append(str(doc))
    return flat_docs

def format_sample(sample: dict) -> dict:
    """原始字段 question/answer/documents → {"prompt": "", "completion": ""}（documents 列加入到prompt之中）。"""
    documents = normalize_documents(sample.get("documents", ""))
    prompt = """
        Based on relevant document answer this question.
        relevant document: {}
        question: {}
    """
    prompt = prompt.format("\n".join(documents), sample.get("question", "").strip())
    return {
        "prompt": prompt,
        "completion": sample.get('answer', '').strip()
    }

def load_train_dataset(
    dataset: str,
    revision: str = "main",
    name: str | None = None,
    max_samples: int | None = None,
):
    """加载并拆分为 prompt/completion 训练集。dataset 为本地路径或 HF ID。"""
    if os.path.isdir(dataset) or os.path.isfile(dataset):
        ds = load_dataset(dataset, split="train")
    else:
        ds = load_dataset(dataset, split="train", revision=revision, name=name)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds.map(lambda s: format_sample(s), remove_columns=["answer", "question"])