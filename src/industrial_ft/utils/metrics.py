"""评测指标（纯逻辑，无 torch 依赖，CI 可测）。

- L0：非空率 / EM / contain（长文生成任务仅诊断用）
- L1：术语覆盖率——从标准答案提取引号术语/书名号规范/英文缩写，统计输出包含比例
- L2：语义相似度——整文 cosine（bge），以及分段要点覆盖率（长文场景主用）
- L3：人工抽查清单（markdown）
"""
from __future__ import annotations

import re
from collections import defaultdict


def normalize(text: str) -> str:
    """去空白与标点（L0 用）。"""
    text = re.sub(r"\s+", "", text or "")
    text = re.sub(r"[\s，。！？、；：\"\"''（）()《》【】—-]", "", text)
    return text


def extract_key_terms(reference: str) -> list[str]:
    """从标准答案提取关键术语（L1）。

    来源：引号内术语、"《》"内规范名、英文缩写/字母数字术语。
    """
    terms: set[str] = set()
    terms.update(re.findall(r"[“\"]([^”\"\n]{1,24})[”\"]", reference))
    terms.update(re.findall(r"[《]([^》\n]{1,40})[》]", reference))
    terms.update(re.findall(r"\b[A-Za-z][A-Za-z0-9\-]{1,15}\b", reference))
    filtered = set()
    for t in terms:
        t = t.strip()
        if len(t) >= 2 and not re.fullmatch(r"[\d\s\-]+", t):
            filtered.add(t)
    return sorted(filtered)


def compute_metrics(predictions: list[str], references: list[str], sources: list[str]) -> dict:
    """L0 + L1 及分模块统计。"""
    total = len(references)
    non_empty = sum(1 for p in predictions if p.strip())
    em = contain = 0
    per_module = defaultdict(lambda: {"total": 0, "em": 0, "contain": 0, "term_hit": 0, "term_total": 0})

    for pred, ref, src in zip(predictions, references, sources):
        p_norm, r_norm = normalize(pred), normalize(ref)
        if r_norm and r_norm == p_norm:
            em += 1
        if len(r_norm) >= 4 and r_norm in p_norm:
            contain += 1
        pred_lower = pred.lower()
        terms = extract_key_terms(ref)
        hits = sum(1 for t in terms if t.lower() in pred_lower)

        m = per_module[src or "unknown"]
        m["total"] += 1
        if r_norm and r_norm == p_norm:
            m["em"] += 1
        if len(r_norm) >= 4 and r_norm in p_norm:
            m["contain"] += 1
        m["term_hit"] += hits
        m["term_total"] += len(terms)

    term_total = sum(m["term_total"] for m in per_module.values())
    term_hit = sum(m["term_hit"] for m in per_module.values())

    metrics = {
        "total": total,
        "non_empty": non_empty / total if total else 0,
        "em": em / total if total else 0,
        "contain": contain / total if total else 0,
        "term_coverage": term_hit / term_total if term_total else 0,
    }
    metrics["per_module"] = {
        src: {
            "total": m["total"],
            "em": m["em"] / m["total"] if m["total"] else 0,
            "contain": m["contain"] / m["total"] if m["total"] else 0,
            "term_coverage": m["term_hit"] / m["term_total"] if m["term_total"] else 0,
        }
        for src, m in sorted(per_module.items())
    }
    return metrics


def compute_semantic_scores(
    predictions: list[str], references: list[str], model, batch_size: int = 64
) -> list[float]:
    """L2 整文语义相似度（pred vs reference 的 embedding cosine）。

    model: SentenceTransformer 实例。
    """
    texts = []
    for p, r in zip(predictions, references):
        texts.extend([p, r])
    embs = model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    return [float(embs[i] @ embs[i + 1]) for i in range(0, len(embs), 2)]


def summarize_semantic(sims: list[float], threshold: float = 0.6) -> dict:
    n = len(sims)
    if n == 0:
        return {"mean": 0.0, "pass_rate": 0.0}
    return {"mean": sum(sims) / n, "pass_rate": sum(1 for s in sims if s >= threshold) / n}


def split_sentences(text: str, min_len: int = 8) -> list[str]:
    parts = re.split(r"[。！？；\n]", text or "")
    return [p.strip() for p in parts if len(p.strip()) >= min_len]


def compute_segment_coverage(
    predictions: list[str],
    references: list[str],
    model,
    threshold: float = 0.65,
    batch_size: int = 128,
) -> list[float]:
    """L2 分段要点覆盖率：参考答案切句，逐句与输出算相似度，sim >= threshold 记命中，
    覆盖率为命中句数/总句数。长文问答主用（整文余弦会被平均池化虚高）。"""
    all_texts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    for pred, ref in zip(predictions, references):
        sents = split_sentences(ref)
        if not sents:
            spans.append((len(all_texts), -1, -1))
            all_texts.append(pred)
            continue
        pred_idx = len(all_texts)
        all_texts.append(pred)
        sent_start = len(all_texts)
        all_texts.extend(sents)
        spans.append((pred_idx, sent_start, len(all_texts)))

    embs = model.encode(
        all_texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )

    coverages: list[float] = []
    for pred_idx, sent_start, sent_end in spans:
        if sent_start < 0:
            coverages.append(0.0)
            continue
        pred_emb = embs[pred_idx]
        hits = 0
        for s in range(sent_start, sent_end):
            if float(embs[s] @ pred_emb) >= threshold:
                hits += 1
        coverages.append(hits / (sent_end - sent_start))
    return coverages


def build_review_markdown(rows: list[dict], per_module_n: int = 3, title: str = "评测") -> str:
    """L3 人工抽查清单：按模块抽样，判定栏留空。"""
    by_module: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_module[r.get("source", "unknown")].append(r)

    lines = [
        f"# 人工抽查清单（{title}，每模块抽 {per_module_n} 条）",
        "",
        "> 判定：**答对**（要点正确且术语准确）/ **部分**（沾边但缺要点或有错）/ **跑题**（无关或空）",
        "",
        "| # | 模块 | 问题（截断） | 标准答案（截断） | 模型输出（截断） | 判定 | 备注 |",
        "|---|------|-------------|-----------------|-----------------|------|------|",
    ]
    idx = 0
    for src in sorted(by_module):
        for r in by_module[src][:per_module_n]:
            idx += 1
            q = r["prompt"].replace("\n", " ")[:80]
            ref = r["reference"].replace("\n", " ")[:120]
            pred = r["prediction"].replace("\n", " ")[:120]
            lines.append(f"| {idx} | {src} | {q} | {ref} | {pred} |  |  |")
    lines.append("")
    lines.append("> 完整文本见同目录 eval_predictions_*.jsonl。")
    return "\n".join(lines)
