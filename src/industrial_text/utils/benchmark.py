"""Data audit, conservative answer extraction and scoring; no torch dependency."""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from industrial_ft.utils.metrics import extract_key_terms


def normalize(text):
    # Preserve decimal points, signs and units: dropping these can create false matches.
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text))).strip("。.;；")


def options(question):
    q = unicodedata.normalize("NFKC", question)
    marked = re.findall(r"(?<![A-Za-z])([A-H])[.、:)]", q)
    spaced = re.findall(r"(?:^|\n)\s*([A-H])\s+", q)
    return sorted(set(marked + spaced))


def final_answer(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    tagged = re.findall(r"<answer>(.*?)</answer>", text, flags=re.S)
    if tagged:
        return tagged[-1].strip(), "tagged"
    if "<think>" in text or "<answer>" in text:
        return "", "incomplete_tag"
    return re.sub(r"^(?:最终答案|答案|答)\s*[:：]\s*", "", text).strip(), "untagged"


def choice_answer(text, allowed, reference=False):
    text = unicodedata.normalize("NFKC", text).strip().strip("* ")
    if reference:
        text = re.sub(r"^\d+[.、]\s*", "", text)
    text = re.sub(r"^(?:正确答案|最终答案|答案|选项|选择|选|答)\s*[:：]?\s*", "", text)
    # Accept compact labels or a leading label followed by option text; never scan explanations.
    if re.fullmatch(r"[A-Ha-h\s,，、;；/和及与]+[。.．]?", text):
        labels = set(re.findall(r"[A-H]", text.upper()))
    else:
        match = re.match(r"^([A-H])(?:[.、:：]\s*|\s+)(.+)$", text, re.S)
        if not match:
            return None
        # Explanations listing additional options are ambiguous; do not guess the first letter.
        if not reference and re.search(r"(?<![A-Za-z])[A-H](?![A-Za-z])", match[2]):
            return None
        labels = {match[1]}
    return sorted(labels) if labels and labels <= set(allowed) else None


def boolean_answer(text):
    value = normalize(text).lower().strip("()（）* ")
    if value in {"正确", "对", "是", "√", "✓", "true", "t"}:
        return True
    if value in {"错误", "错", "否", "×", "✗", "false", "f"}:
        return False
    return None


def load_records(root):
    root = Path(root)
    records = []
    digest = hashlib.sha256()
    for path in sorted(root.rglob("qa.json")):
        raw = path.read_bytes()
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode()); digest.update(b"\0"); digest.update(raw)
        values = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(values, list):
            raise ValueError(f"Expected a list: {rel}")
        for index, value in enumerate(values):
            question, answer = value.get("question"), value.get("answer")
            if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Missing question/answer: {rel}:{index}")
            labels = options(question)
            boolean = boolean_answer(answer)
            gold = choice_answer(answer, labels, reference=True) if len(labels) >= 2 else None
            if gold is None and boolean is not None and len(labels) >= 2:
                mappings = re.findall(r"([A-H])[.、:)]\s*(正确|错误|对|错)", unicodedata.normalize("NFKC", question))
                matches = [label for label, word in mappings if boolean_answer(word) == boolean]
                if len(matches) == 1:
                    gold = matches
            kind = ("multiple_choice" if len(gold) > 1 else "single_choice") if gold else (
                "choice_unresolved" if re.search(r"单选|多选|选择题", question) or re.fullmatch(r"[A-H、, ]+", answer) else "boolean" if boolean is not None else
                "fill_blank" if re.search(r"_{2,}|填空", question) else "open"
            )
            records.append({
                "id": f"{rel}:{index}", "subject": Path(rel).parts[0],
                "question": question, "reference": answer,
                "cognitive_type": value.get("question_type", "unknown"),
                "difficulty": value.get("estimated_difficulty", "unknown"),
                "kind": kind, "options": labels, "gold_choice": gold,
                "question_hash": hashlib.sha256(question.strip().encode()).hexdigest(),
            })
    if not records:
        raise ValueError(f"No qa.json records in {root}")
    grouped = defaultdict(list)
    for record in records:
        grouped[record["question_hash"]].append(record)
    for group in grouped.values():
        conflict = len({normalize(r["reference"]) for r in group}) > 1
        for i, record in enumerate(group):
            record.update(duplicate=i > 0, reference_conflict=conflict)
    audit = {
        "dataset_sha256": digest.hexdigest(), "total": len(records),
        "subjects": dict(Counter(r["subject"] for r in records)),
        "kinds": dict(Counter(r["kind"] for r in records)),
        "duplicate_excess": sum(r["duplicate"] for r in records),
        "conflict_records": [r["id"] for r in records if r["reference_conflict"]],
        "unresolved_choice_records": [r["id"] for r in records if r["kind"] == "choice_unresolved"],
    }
    return records, audit


def stratified_sample(records, limit, seed):
    if not limit or limit >= len(records):
        return records
    rng = random.Random(seed)
    groups = defaultdict(list)
    for record in records:
        groups[record["subject"]].append(record)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < limit:
        for subject in sorted(groups):
            if groups[subject] and len(selected) < limit:
                selected.append(groups[subject].pop())
    return selected


def score(record, prediction, error=None, truncated=False):
    answer, extraction = final_answer(prediction)
    objective = record["kind"] in {"single_choice", "multiple_choice", "boolean"}
    parsed = None
    correct = None
    if objective:
        if record["kind"] == "boolean":
            parsed = boolean_answer(answer)
            correct = parsed is not None and parsed == boolean_answer(record["reference"])
        else:
            parsed = choice_answer(answer, record["options"])
            correct = parsed is not None and parsed == record["gold_choice"]
        if error or truncated:
            correct = False
    terms = extract_key_terms(record["reference"])
    hits = sum(term.casefold() in answer.casefold() for term in terms)
    # Reference character-bigram recall is lexical overlap, never semantic correctness.
    def grams(text):
        text = normalize(text)
        return {text[i:i+2] for i in range(len(text)-1)}
    refgrams = grams(record["reference"])
    return {
        "answer": answer, "extraction": extraction, "objective": objective,
        "parsed": parsed, "correct": correct,
        "parse_failure": objective and parsed is None,
        "normalized_em": bool(answer) and normalize(answer) == normalize(record["reference"]),
        "term_hits": hits, "term_total": len(terms),
        "reference_bigram_recall": len(refgrams & grams(answer))/len(refgrams) if refgrams else None,
    }


def aggregate(rows):
    objective = [r for r in rows if r["score"]["objective"]]
    terms = sum(r["score"]["term_total"] for r in rows)
    recalls = [r["score"]["reference_bigram_recall"] for r in rows
               if r["kind"] in {"fill_blank", "open"} and r["score"]["reference_bigram_recall"] is not None]
    return {
        "total": len(rows), "objective_n": len(objective),
        "correct": sum(bool(r["score"]["correct"]) for r in objective),
        "accuracy": sum(bool(r["score"]["correct"]) for r in objective)/len(objective) if objective else None,
        "parse_failures": sum(r["score"]["parse_failure"] for r in objective),
        "errors": sum(bool(r.get("error")) for r in rows),
        "truncated": sum(r.get("truncated", False) for r in rows),
        "empty": sum(not r["score"]["answer"] for r in rows),
        "normalized_em_diagnostic": sum(r["score"]["normalized_em"] for r in rows)/len(rows) if rows else None,
        "term_coverage_diagnostic": sum(r["score"]["term_hits"] for r in rows)/terms if terms else None,
        "term_denominator": terms,
        "open_bigram_recall_diagnostic": sum(recalls)/len(recalls) if recalls else None,
    }


def summarize(rows):
    result = {"raw": aggregate(rows), "deduplicated": aggregate([r for r in rows if not r["duplicate"]]),
              "deduplicated_no_conflicts": aggregate([r for r in rows if not r["duplicate"] and not r["reference_conflict"]])}
    for key in ("subject", "kind", "cognitive_type", "difficulty"):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row)
        result[f"by_{key}"] = {name: aggregate(group) for name, group in sorted(groups.items())}
    accuracies = [m["accuracy"] for m in result["by_subject"].values() if m["accuracy"] is not None]
    result["subject_macro_accuracy"] = sum(accuracies)/len(accuracies) if accuracies else None
    return result
