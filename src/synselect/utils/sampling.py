"""Stable stratified sampling, with auditable exclusions and image diversity."""
from __future__ import annotations

import bisect
import hashlib
import json
from collections import Counter, defaultdict, deque


def record_id(record: dict) -> str:
    for key in ("id", "question_index"):
        if key in record:
            return str(record[key])
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def program_depth(program: list[dict]) -> int:
    """DAG dependency depth, not simply number of operations."""
    depths = []
    for i, operation in enumerate(program):
        inputs = operation.get("inputs", [])
        if any(not isinstance(j, int) or j < 0 or j >= i for j in inputs):
            raise ValueError("CLEVR program has an invalid dependency")
        depths.append(1 + max((depths[j] for j in inputs), default=0))
    return depths[-1] if depths else 0


def clevr_family(function: str) -> str:
    if function in {"equal_integer", "less_than", "greater_than"}:
        return "compare_number"
    if function.startswith("equal_"):
        return "compare_attribute"
    if function in {"count", "exist", "query_color", "query_material", "query_size", "query_shape"}:
        return function
    raise ValueError(f"Unsupported CLEVR final operation: {function}")


def stratum(record: dict, settings: dict) -> str:
    strategy = settings["strategy"]
    if strategy == "sft_length_type":
        question = record["conversations"][0]["value"]
        completion = record["conversations"][1]["value"].strip()
        bucket = bisect.bisect_left(settings["length_bins"], len(completion))
        kind = "choice" if "choices:" in question.lower() else "open"
        # Use an existing source label only; never infer scientific domains from keywords.
        source = str(record.get("source", "unspecified"))
        return f"{source}/{kind}/length_{bucket}"
    if strategy == "clevr_task_depth":
        program = record.get("program")
        if not program:
            raise ValueError("CLEVR stratification requires official program annotations")
        bucket = bisect.bisect_left(settings["depth_bins"], program_depth(program))
        return f"{clevr_family(program[-1]['function'])}/depth_{bucket}"
    raise ValueError(f"Unknown sampling strategy: {strategy}")


def stratified_sample(records: list[dict], limit: int, settings: dict, seed: int):
    if limit < 1 or settings["allocation"] not in {"proportional", "balanced"}:
        raise ValueError("Sampling needs a positive limit and valid allocation")
    max_per_image = settings.get("max_per_image")
    if max_per_image is not None and max_per_image < 1:
        raise ValueError("max_per_image must be positive")
    grouped = defaultdict(list)
    excluded = Counter()
    seen_ids = set()
    for record in records:
        key = record_id(record)
        if key in seen_ids:
            raise ValueError(f"Duplicate sampling ID: {key}")
        seen_ids.add(key)
        cap = settings.get("max_completion_chars")
        if cap and len(record["conversations"][1]["value"].strip()) > cap:
            excluded["completion_over_character_cap"] += 1
            continue
        grouped[stratum(record, settings)].append(record)
    population = {key: len(values) for key, values in sorted(grouped.items())}
    buckets = {}
    for group, values in grouped.items():
        answers = defaultdict(list)
        for record in values:
            answer = str(record["answer"]) if settings.get("balance_answers") else "all"
            answers[answer].append(record)
        buckets[group] = {answer: deque(sorted(items, key=lambda r: stable_rank(seed, record_id(r))))
                          for answer, items in answers.items()}
    selected, counts, image_counts, answer_counts = [], Counter(), Counter(), Counter()
    while buckets and len(selected) < limit:
        group = min(buckets, key=lambda g: (
            counts[g] / (population[g] if settings["allocation"] == "proportional" else 1),
            stable_rank(seed, g)))
        answers = buckets[group]
        answer = min(answers, key=lambda a: (answer_counts[group, a], stable_rank(seed, a)))
        record = answers[answer].popleft()
        if not answers[answer]:
            del answers[answer]
        if not answers:
            del buckets[group]
        image = str(record.get("image_filename", record.get("image")))
        if max_per_image and image_counts[image] >= max_per_image:
            continue
        selected.append(record)
        counts[group] += 1
        image_counts[image] += 1
        answer_counts[group, answer] += 1
    if len(selected) != limit and settings.get("require_count", True):
        raise ValueError(f"Requested {limit} records, only {len(selected)} satisfy sampling constraints")
    ids = [record_id(record) for record in selected]
    report = {"seed": seed, "settings": settings, "requested": limit,
              "source_records": len(records), "eligible_records": sum(population.values()),
              "excluded": dict(excluded), "selected": len(selected),
              "unique_images": len(image_counts), "population_strata": population,
              "selected_strata": dict(sorted(counts.items())),
              "selected_answers": dict(Counter(str(r.get("answer", "n/a")) for r in selected)),
              "selected_ids": ids,
              "selection_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
    return selected, report
