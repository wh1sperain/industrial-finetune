"""Deterministic CLEVR accuracy + format rewards (no LLM judge)."""
from __future__ import annotations

import re

_WORDS = "zero one two three four five six seven eight nine ten".split()
_ANSWERS = set("yes no gray red blue green brown purple cyan yellow cube sphere cylinder "
                "small large rubber metal".split()) | {str(i) for i in range(11)}
_FORMAT = re.compile(r"\s*<think>([^<>]+)</think>\s*<answer>([^<>]+)</answer>\s*", re.S)


def completion_text(value) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or len(value) != 1:
        return ""
    content = value[0].get("content", "")
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if part.get("type") == "text")


def normalize_clevr(value: str) -> str | None:
    value = value.strip().lower().rstrip(".").strip()
    if value in _WORDS:
        value = str(_WORDS.index(value))
    return value if value in _ANSWERS else None


def predicted_answer(value) -> str | None:
    text = completion_text(value)
    matches = re.findall(r"<answer>([^<>]+)</answer>", text, re.S)
    if len(matches) != 1 or text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return None
    return normalize_clevr(matches[0])


def accuracy_reward(completions, answer, **kwargs) -> list[float]:
    if len(completions) != len(answer):
        raise ValueError("Completion/gold batch sizes differ")
    return [float((gold := normalize_clevr(str(a))) is not None and
                  predicted_answer(c) == gold) for c, a in zip(completions, answer)]


def format_reward(completions, **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        match = _FORMAT.fullmatch(completion_text(completion))
        rewards.append(float(bool(match and match[1].strip() and match[2].strip())))
    return rewards
