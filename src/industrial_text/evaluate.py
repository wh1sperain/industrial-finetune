"""Resumable Qwen3 baseline generation and independently repeatable scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from industrial_text.utils.benchmark import load_records, score, stratified_sample, summarize
from industrial_text.utils.config import load_config


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_predictions(path):
    if not path.exists():
        return []
    # Recover only an incomplete last write; internal corruption must fail loudly.
    raw = path.read_bytes()
    parts = raw.splitlines(keepends=True)
    rows = []
    valid_bytes = 0
    for i, part in enumerate(parts):
        try:
            rows.append(json.loads(part))
        except (ValueError, UnicodeDecodeError):
            if i != len(parts)-1 or part.endswith(b"\n"):
                raise
            path.with_suffix(".partial-backup").write_bytes(raw)
            path.write_bytes(raw[:valid_bytes])
            break
        valid_bytes += len(part)
    # A complete JSON object without a newline is also recoverable.
    if rows and path.read_bytes() and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate prediction IDs; refusing ambiguous resume")
    return rows


def emit_report(out, records, predictions, audit, cfg):
    import jieba.analyse

    lookup = {r["id"]: r for r in records}
    rows = []
    for pred in predictions:
        record = lookup[pred["id"]]
        row = {**record, **pred}
        row["score"] = score(record, pred["prediction"], pred.get("error"), pred.get("truncated", False))
        if record["kind"] in {"open", "fill_blank"}:
            keywords = [w for w in jieba.analyse.extract_tags(record["reference"],
                        topK=cfg["diagnostics"]["keyword_top_k"])
                        if len(w) >= cfg["diagnostics"]["keyword_min_length"]]
            hit = sum(w.casefold() in row["score"]["answer"].casefold() for w in keywords)
            row["score"]["reference_keywords"] = keywords
            row["score"]["keyword_coverage_diagnostic"] = hit/len(keywords) if keywords else None
        rows.append(row)
    summary = summarize(rows)
    coverages = [r["score"]["keyword_coverage_diagnostic"] for r in rows
                 if r["score"].get("keyword_coverage_diagnostic") is not None]
    summary.update(expected=len(records), completed=len(rows), dataset_sha256=audit["dataset_sha256"],
                   keyword_coverage_diagnostic=sum(coverages)/len(coverages) if coverages else None,
                   gpu_peak_allocated_gb=max((r.get("gpu_peak_allocated_gb", 0) for r in rows), default=0),
                   generation_seconds=sum(r.get("generation_seconds", 0) for r in rows),
                   output_tokens=sum(r.get("output_tokens", 0) for r in rows),
                   scoring_note="Accuracy applies only to parsable-reference choice/boolean questions. Failed or truncated predictions count as incorrect. Reference answers are not independently verified. Lexical overlap is diagnostic, not correctness or semantic coverage.")
    write_json(out / "metrics.json", summary)
    with (out / "scored_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    def pct(value):
        return "N/A" if value is None else f"{value:.2%}"
    lines = ["# Qwen3-8B 工业纯文本评测", "", f"完成 {len(rows)} / {len(records)} 题。", "",
             "本结果是非 thinking、贪心解码基线。准确率仅覆盖可可靠解析参考答案的选择/判断题。",
             "参考答案尚未经过专家校验；重复/冲突数据另列。关键词、字符重合和 EM 仅供诊断，不代表语义正确率。", "",
             "| 口径 | 全部题数 | 客观题数 | 正确 | 准确率 |", "|---|---:|---:|---:|---:|"]
    for key in ("raw", "deduplicated", "deduplicated_no_conflicts"):
        item = summary[key]
        lines.append(f"| {key} | {item['total']} | {item['objective_n']} | {item['correct']} | {pct(item['accuracy'])} |")
    lines += ["", f"领域等权客观题准确率：{pct(summary['subject_macro_accuracy'])}", "",
              "| 领域 | 全部题数 | 客观题数 | 准确率 | 解析失败 |", "|---|---:|---:|---:|---:|"]
    for subject, item in summary["by_subject"].items():
        lines.append(f"| {subject} | {item['total']} | {item['objective_n']} | {pct(item['accuracy'])} | {item['parse_failures']} |")
    raw = summary["raw"]
    lines += ["", f"生成错误：{raw['errors']}；截断：{raw['truncated']}；空回答：{raw['empty']}。",
              f"开放/填空参考关键词覆盖率（诊断）：{pct(summary['keyword_coverage_diagnostic'])}。",
              f"生成总耗时：{summary['generation_seconds']:.1f} 秒；GPU 峰值分配：{summary['gpu_peak_allocated_gb']:.2f} GiB。",
              "", "## 数据审计", "", f"数据 SHA256：`{audit['dataset_sha256']}`", "",
              f"重复冗余：{audit['duplicate_excess']}；参考答案文本冲突记录：{len(audit['conflict_records'])}。",
              "", "完整分题型、认知类型和难度统计见 metrics.json，完整问答与评分见 scored_predictions.jsonl。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    review = ["# 人工复核清单", "", "待人工判定，未以大模型裁判替代。", ""]
    counts = {}
    for row in sorted(rows, key=lambda r: (not r["reference_conflict"], r["score"]["correct"] is True, r["id"])):
        subject = row["subject"]
        if not row["reference_conflict"] and counts.get(subject, 0) >= cfg["review_per_subject"]:
            continue
        counts[subject] = counts.get(subject, 0) + 1
        review += [f"## {row['id']}", "", f"题型：{row['kind']}；参考冲突：{row['reference_conflict']}", "",
                   row["question"], "", "参考答案：", row["reference"], "", "模型答案：", row["score"]["answer"],
                   "", "人工判定：待复核", ""]
    (out / "review.md").write_text("\n".join(review), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.experiment)
    records, audit = load_records(args.data_root or cfg["dataset"])
    if args.audit:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return
    records = stratified_sample(records, cfg["probe_size"] if args.probe else None, cfg["seed"])
    out = Path(cfg["output_dir"]) / ("probe" if args.probe else "full")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "audit.json", audit)
    predfile = out / "predictions.jsonl"
    predictions = read_predictions(predfile)
    code_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        code_hash.update(path.relative_to(Path(__file__).parent).as_posix().encode())
        code_hash.update(path.read_bytes())
    source_path = Path(cfg["model_name_or_path"]).parent / "model-source.json"
    model_source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.is_file() else None
    identity = {"config": cfg, "dataset_sha256": audit["dataset_sha256"],
                "ids": [r["id"] for r in records], "code_sha256": code_hash.hexdigest(),
                "model_source": model_source}
    manifest = out / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        keys = ("config", "dataset_sha256", "ids", "model_source") if args.score_only else identity.keys()
        if any(previous.get(key) != identity[key] for key in keys):
            raise ValueError("Run identity changed; use a new output directory instead of mixing predictions")
    if not manifest.exists() and predictions:
        raise ValueError("Predictions without a manifest cannot be resumed")
    if not args.score_only or not manifest.exists():
        write_json(manifest, identity)
    expected = {r["id"] for r in records}
    if any(r["id"] not in expected for r in predictions):
        raise ValueError("Unknown prediction IDs")
    if args.score_only:
        write_json(out / "scoring_version.json", {"code_sha256": code_hash.hexdigest()})
        print(json.dumps(emit_report(out, records, predictions, audit, cfg), ensure_ascii=False, indent=2))
        return
    done = {r["id"] for r in predictions}
    pending = [r for r in records if r["id"] not in done]
    if not pending:
        emit_report(out, records, predictions, audit, cfg)
        print("All predictions already present; report refreshed.")
        return
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    settings = cfg["model_settings"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], trust_remote_code=settings["trust_remote_code"], padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"],
        dtype=getattr(torch, settings["dtype"]), attn_implementation=settings["attn_implementation"],
        trust_remote_code=settings["trust_remote_code"]).to("cuda").eval()
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "transformers": transformers.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(), "model_commit": getattr(model.config, "_commit_hash", None),
                   "model_config": model.config.to_dict(),
                   "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    write_json(out / "environment.json", environment)
    g = cfg["generation"]
    size = g["batch_size"]
    with predfile.open("a", encoding="utf-8") as stream:
        while pending:
            batch = pending[:size]
            texts = [tokenizer.apply_chat_template([
                {"role": "system", "content": g["system_prompt"]},
                {"role": "user", "content": g["user_template"].format(prompt=r["question"])}],
                tokenize=False, add_generation_prompt=True, enable_thinking=g["enable_thinking"]) for r in batch]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            if inputs["input_ids"].shape[1] > g["max_input_tokens"]:
                raise ValueError("Input exceeds configured limit; refusing silent prompt truncation")
            inputs = inputs.to("cuda")
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=g["max_new_tokens"],
                        do_sample=g["do_sample"], temperature=None, top_p=None, top_k=None,
                        pad_token_id=tokenizer.pad_token_id)
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                del inputs
                torch.cuda.empty_cache()
                if size == 1:
                    raise
                size = max(1, size//2)
                print(f"OOM: reducing batch size to {size}", flush=True)
                continue
            elapsed = time.perf_counter()-started
            eos = model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, list) else [eos])
            for index, (record, output) in enumerate(zip(batch, outputs)):
                ids = output[inputs["input_ids"].shape[1]:].tolist()
                stop = next((i for i, token in enumerate(ids) if token in eos), None)
                used = ids if stop is None else ids[:stop]
                row = {"id": record["id"], "prediction": tokenizer.decode(used, skip_special_tokens=True),
                       "input_tokens": int(inputs["attention_mask"][index].sum().item()),
                       "output_tokens": len(used), "truncated": stop is None and len(ids) >= g["max_new_tokens"],
                       "generation_seconds": elapsed/len(batch), "batch_size": len(batch),
                       "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated()/1024**3, "error": None}
                stream.write(json.dumps(row, ensure_ascii=False)+"\n")
                stream.flush(); os.fsync(stream.fileno())
                predictions.append(row)
            del inputs, outputs
            pending = pending[len(batch):]
            write_json(out / "progress.json", {"completed": len(predictions), "total": len(records),
                       "last_batch_seconds": elapsed, "batch_size": size})
            print(f"{len(predictions)}/{len(records)} batch={size} seconds={elapsed:.2f}", flush=True)
    emit_report(out, records, predictions, audit, cfg)
    print(f"Finished: {out}", flush=True)


if __name__ == "__main__":
    main()
