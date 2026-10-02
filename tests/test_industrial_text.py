import json

import pytest

from industrial_text.evaluate import read_predictions
from industrial_text.utils.generation import completion_status, output_budgets
from industrial_text.utils.generation import completion_status, output_budgets
from industrial_text.utils.benchmark import (
    aggregate, boolean_answer, choice_answer, final_answer, load_records, normalize,
    options, score, stratified_sample,
)


def test_choice_extraction_rejects_explanations_and_extra_options():
    assert choice_answer("A、C", "ABCD") == ["A", "C"]
    assert choice_answer("ＡＣ", "ABCD") == ["A", "C"]
    assert choice_answer("D. 稻谷", "ABCD") == ["D"]
    assert choice_answer("A错了，B才正确", "ABCD") is None
    assert choice_answer("B。A错误，C也错误", "ABCD") is None
    assert choice_answer("A 或 B", "ABCD") is None
    assert choice_answer("E", "ABCD") is None


def test_numeric_options_and_english_article():
    assert options("题目 A.2 B.3 C.4 D.6") == list("ABCD")
    assert options("题目\nA 增强\nB 不变\nC 失去") == list("ABC")
    assert len(options("A manufacturing engineer needs help.")) < 2


def test_final_answer_does_not_score_thinking():
    assert final_answer("<think>A</think><answer>B</answer>")[0] == "B"
    assert final_answer("<think>A")[0] == ""
    assert final_answer("<answer>A")[0] == ""
    assert boolean_answer("不正确") is None
    assert boolean_answer("错误") is False
    assert normalize("-1.5 mm") != normalize("15 mm")


def test_duplicates_conflicts_sampling_and_no_gold_in_question(tmp_path):
    for subject, answer in [("one", "A"), ("two", "B")]:
        directory = tmp_path / subject / "1"
        directory.mkdir(parents=True)
        (directory / "qa.json").write_text(json.dumps([{"question": "Q\nA.2 B.3", "answer": answer}]), encoding="utf-8")
    records, audit = load_records(tmp_path)
    assert audit["duplicate_excess"] == 1
    assert len(audit["conflict_records"]) == 2
    assert all(r["kind"] == "single_choice" for r in records)
    assert len(stratified_sample(records, 1, 42)) == 1
    assert stratified_sample(records, 1, 42) == stratified_sample(records, 1, 42)


def test_failed_and_truncated_predictions_stay_in_denominator():
    record = {"kind": "single_choice", "options": list("ABCD"), "gold_choice": ["B"], "reference": "B"}
    good = score(record, "<answer>B</answer>")
    invalid = score(record, "I do not know")
    truncated = score(record, "<answer>B</answer>", truncated=True)
    assert good["correct"] is True
    assert invalid["correct"] is False and invalid["parse_failure"]
    assert truncated["correct"] is False
    result = aggregate([{**record, "score": value} for value in (good, invalid, truncated)])
    assert result["objective_n"] == 3
    assert result["accuracy"] == pytest.approx(1/3)


def test_resume_repairs_partial_tail_but_rejects_internal_corruption(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_bytes(b'{"id":"one"}\n{"id":')
    assert read_predictions(path) == [{"id": "one"}]
    assert path.read_bytes() == b'{"id":"one"}\n'
    path.write_bytes(b'not-json\n{"id":"one"}\n')
    with pytest.raises(ValueError):
        read_predictions(path)


def test_resume_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"id":"one"}\n{"id":"one"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        read_predictions(path)


def test_output_budgets_respect_context_and_remove_duplicate_clamps():
    g = {"max_new_tokens": 8192, "retry_max_new_tokens": [16384, 24576], "context_limit": 32768}
    assert output_budgets(g, 1000) == [8192, 16384, 24576]
    assert output_budgets(g, 20000) == [8192, 12768]
    with pytest.raises(ValueError, match="context"):
        output_budgets(g, 32768)
    with pytest.raises(ValueError, match="increasing"):
        output_budgets({**g, "retry_max_new_tokens": [4096]}, 10)


def test_only_explicit_completion_avoids_length_truncation():
    assert completion_status([1, 2, 3], {9}, "<answer>x", 3, "</answer>")[:2] == ("length", True)
    assert completion_status([1, 2, 3], {9}, "<answer>x</answer>", 3, "</answer>")[:2] == ("answer_end", False)
    assert completion_status([1, 9, 9], {9}, "x", 3)[0] == "eos"
    assert completion_status([1], {9}, "x", 3)[0] == "unexpected_stop"


def test_output_budgets_respect_context_and_remove_duplicate_clamps():
    g = {"max_new_tokens": 8192, "retry_max_new_tokens": [16384, 24576], "context_limit": 32768}
    assert output_budgets(g, 1000) == [8192, 16384, 24576]
    assert output_budgets(g, 20000) == [8192, 12768]
    with pytest.raises(ValueError, match="context"):
        output_budgets(g, 32768)
    with pytest.raises(ValueError, match="increasing"):
        output_budgets({**g, "retry_max_new_tokens": [4096]}, 10)


def test_only_explicit_completion_avoids_length_truncation():
    assert completion_status([1, 2, 3], {9}, "<answer>x", 3, "</answer>")[:2] == ("length", True)
    assert completion_status([1, 2, 3], {9}, "<answer>x</answer>", 3, "</answer>")[:2] == ("answer_end", False)
    assert completion_status([1, 9, 9], {9}, "x", 3)[0] == "eos"
    assert completion_status([1], {9}, "x", 3)[0] == "unexpected_stop"
