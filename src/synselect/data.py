"""Read local SynSelect ShareGPT and CLEVR JSON/JSONL without GPU dependencies."""
from __future__ import annotations

import json
import random
from pathlib import Path

from synselect.utils.config import local_path


def read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in handle if line.strip()]
        else:
            records = json.load(handle)
    if isinstance(records, dict) and "questions" in records:
        records = records["questions"]
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a nonempty list (or CLEVR questions object): {path}")
    return records


def make_prompt(question: str, cfg: dict) -> list[dict]:
    question = question.replace("<image>", "").strip()
    if not question:
        raise ValueError("Empty question")
    messages = []
    if cfg["prompt"].get("system"):
        messages.append({"role": "system", "content": [
            {"type": "text", "text": cfg["prompt"]["system"]}]})
    # TRL inserts the actual image into the placeholder from the image column.
    messages.append({"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": cfg["prompt"]["template"].format(prompt=question)},
    ]})
    return messages


def convert_record(record: dict, cfg: dict, spec: dict, index: int) -> dict:
    kind = spec["format"]
    if kind == "sharegpt":
        turns = record["conversations"]
        if (len(turns) != 2 or turns[0]["from"] not in {"human", "user"}
                or turns[1]["from"] not in {"assistant", "gpt"}):
            raise ValueError("Expected one user/image turn and one assistant target")
        question, target = turns[0]["value"], turns[1]["value"]
        image = record["image"]
    elif kind == "clevr":
        question, target, image = record["question"], record["answer"], record["image_filename"]
    else:
        raise ValueError(f"Unsupported format: {kind}")
    if not isinstance(image, str) or question.count("<image>") > 1:
        raise ValueError("This dataset adapter supports one image per question")
    image_path = Path(image)
    if not image_path.is_absolute():
        image_path = local_path(spec["image_root"]) / image_path
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    target = str(target).strip()
    if not target:
        raise ValueError("Empty target")
    result = {"id": str(record.get("id", record.get("question_index", index))),
              "image": str(image_path), "prompt": make_prompt(question, cfg)}
    if cfg["stage"] == "sft":
        result["completion"] = [{"role": "assistant", "content": target}]
    else:
        # Require genuine labels; never turn a synthesized reasoning trace into a gold reward.
        from synselect.utils.rewards import normalize_clevr
        answer = normalize_clevr(target)
        if answer is None:
            raise ValueError("GRPO requires a plain CLEVR gold answer, not a CoT")
        result["answer"] = answer
    return result


def load_rows(cfg: dict, spec: dict | None = None, *, return_selection: bool = False):
    spec = spec or cfg["dataset"]
    records = read_records(local_path(spec["annotations"]))
    if spec.get("count_only"):
        records = [r for r in records if r.get("program") and
                   r["program"][-1].get("function") == "count"]
    limit = spec.get("max_samples")
    report = None
    if spec.get("sampling"):
        from synselect.utils.sampling import stratified_sample
        if limit is None:
            raise ValueError("Stratified sampling requires max_samples")
        records, report = stratified_sample(records, limit, spec["sampling"],
                                            spec.get("seed", cfg["train"]["seed"]))
    elif limit is not None:
        if limit < 1:
            raise ValueError("max_samples must be positive or null")
        records = random.Random(cfg["train"]["seed"]).sample(records, min(limit, len(records)))
    if not records:
        raise ValueError("No records remain after filtering")
    rows = []
    for index, record in enumerate(records):
        try:
            rows.append(convert_record(record, cfg, spec, index))
        except (KeyError, ValueError, FileNotFoundError) as error:
            raise ValueError(f"Record {index} ({record.get('id', 'no id')}): {error}") from error
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate example IDs")
    return (rows, report) if return_selection else rows


def as_dataset(rows: list[dict]):
    from datasets import Dataset, Image
    return Dataset.from_list(rows).cast_column("image", Image())
