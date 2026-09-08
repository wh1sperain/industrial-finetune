"""Audit local data; optional processor statistics load no model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics

from synselect.data import load_rows
from synselect.utils.config import load_config, local_path


def summary(values):
    values = sorted(values)
    return {"mean": statistics.mean(values), "p50": values[len(values) // 2],
            "p95": values[min(int(len(values) * .95), len(values) - 1)], "max": values[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-images", action="store_true", help="Decode-check every image")
    parser.add_argument("--tokenize-samples", type=int, default=0,
                        help="Randomly sample N actual processed lengths (no model weights)")
    args = parser.parse_args()
    if args.tokenize_samples < 0:
        parser.error("tokenize-samples cannot be negative")
    cfg = load_config(args.experiment)
    rows, selection = load_rows(cfg, return_selection=True)
    path = local_path(cfg["dataset"]["annotations"])
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    report = {"annotations": str(path), "sha256": digest.hexdigest(), "rows": len(rows),
              "unique_images": len({r["image"] for r in rows}), "missing_images": 0,
              "stage": cfg["stage"], "processor": cfg["processor"]}
    if selection:
        report["selection"] = selection
    if cfg["stage"] == "sft":
        report["completion_characters"] = summary([len(r["completion"][0]["content"]) for r in rows])
    if args.verify_images:
        from PIL import Image
        for image in sorted({r["image"] for r in rows}):
            with Image.open(image) as source:
                source.verify()
        report["image_integrity"] = "verified"
    if args.tokenize_samples:
        from PIL import Image
        from synselect.runtime import build_processor
        processor = build_processor(cfg)
        sample = random.Random(cfg["train"]["seed"]).sample(rows, min(args.tokenize_samples, len(rows)))
        lengths = []
        for row in sample:
            messages = row["prompt"] + row.get("completion", [])
            text = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=cfg["stage"] == "grpo")
            with Image.open(row["image"]) as source:
                inputs = processor(text=[text], images=[source.convert("RGB")], return_tensors="pt")
            lengths.append(inputs.input_ids.shape[1])
        report["processed_sequence_tokens"] = summary(lengths)
        report["tokenized_samples"] = len(lengths)
        limit = cfg["runtime"]["max_sequence_tokens"]
        allowance = cfg["train"].get("max_completion_length", 0)
        report["over_length_guard_in_sample"] = sum(n + allowance > limit for n in lengths)
        report["token_length_note"] = "GRPO counts prompt only; SFT counts prompt + target + image tokens"
    output = local_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    display = dict(report)
    if selection:
        display["selection"] = {k: v for k, v in selection.items() if k != "selected_ids"}
    print(json.dumps(display, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
