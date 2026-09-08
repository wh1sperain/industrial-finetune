"""CPU tests: data integrity, reward correctness, stage handoff and time accounting."""
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synselect.data import as_dataset, convert_record, load_rows, make_prompt, read_records
from synselect.train import check_prompt_lengths, validate_adapter
from synselect.utils.config import load_config, local_path, validate_config
from synselect.utils.rewards import accuracy_reward, format_reward, predicted_answer
from synselect.utils.timing import optimizer_steps, planning_estimate


class ConfigTests(unittest.TestCase):
    def test_configs_and_handoff_match(self):
        sft, grpo = load_config("sft_5090"), load_config("grpo_5090")
        for key in ("model", "prompt", "processor"):
            self.assertEqual(sft[key], grpo[key])
        self.assertEqual(Path(sft["output_dir"]), local_path(grpo["sft_adapter"]))
        self.assertEqual(grpo["train"]["loss_type"], "grpo")

    def test_no_global_experiment_switch(self):
        from industrial_ft.utils.config import list_experiments
        before = list_experiments()
        load_config("sft_5090")
        self.assertEqual(before, list_experiments())
        self.assertIn("mining_eng_lora", before)

    def test_reject_unsafe_visual_truncation(self):
        cfg = load_config("sft_5090")
        cfg["train"]["max_length"] = 2048
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_reject_non_sdpa(self):
        cfg = load_config("sft_5090")
        cfg["model_options"]["attn_implementation"] = "flash_attention_2"
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_reject_bad_grpo_groups(self):
        cfg = load_config("grpo_5090")
        cfg["train"]["generation_batch_size"] = 3
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_reject_greedy_training(self):
        cfg = load_config("grpo_5090")
        cfg["train"]["generation_kwargs"]["do_sample"] = False
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_unresolved_environment_path(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            local_path("${SYNSELECT_TEST_UNSET}/images")

    def test_lora_does_not_target_vision(self):
        pattern = load_config("sft_5090")["lora"]["target_modules"]
        self.assertTrue(re.fullmatch(pattern, "model.language_model.layers.0.self_attn.q_proj"))
        self.assertFalse(re.fullmatch(pattern, "model.visual.blocks.0.attn.q_proj"))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # Conversion validates existence; image decoding is an independent audit step.
        (self.root / "image.png").write_bytes(b"test placeholder")
        self.cfg = load_config("sft_5090")
        self.spec = {"format": "sharegpt", "image_root": str(self.root),
                     "annotations": str(self.root / "data.json"), "max_samples": None}
        self.record = {"id": "a", "image": "image.png", "conversations": [
            {"from": "human", "value": "<image>\nHow many objects?"},
            {"from": "assistant", "value": "<think>Two visible objects.</think><answer>2</answer>"}]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_sft_target_and_image_preserved(self):
        row = convert_record(self.record, self.cfg, self.spec, 0)
        self.assertEqual(row["completion"][0]["content"], self.record["conversations"][1]["value"])
        self.assertEqual(row["prompt"][-1]["content"][0], {"type": "image"})
        self.assertEqual(row["prompt"][-1]["content"][1]["text"], "问：How many objects?")
        self.assertTrue(all(isinstance(m["content"], list) for m in row["prompt"]))
        self.assertNotIn("answer", row)

    def test_clevr_label_never_in_prompt(self):
        self.cfg["stage"] = "grpo"
        spec = dict(self.spec, format="clevr")
        r = {"question": "How many?", "answer": "two", "image_filename": "image.png"}
        row = convert_record(r, self.cfg, spec, 0)
        self.assertEqual(row["answer"], "2")
        self.assertNotIn("completion", row)
        self.assertNotIn("two", json.dumps(row["prompt"]))

    def test_refuse_cot_as_grpo_gold(self):
        self.cfg["stage"] = "grpo"
        with self.assertRaises(ValueError):
            convert_record(self.record, self.cfg, self.spec, 0)

    def test_missing_image_is_not_silently_dropped(self):
        self.record["image"] = "absent.png"
        with self.assertRaises(FileNotFoundError):
            convert_record(self.record, self.cfg, self.spec, 0)

    def test_reject_multiturn(self):
        self.record["conversations"].append({"from": "human", "value": "Again"})
        with self.assertRaises(ValueError):
            convert_record(self.record, self.cfg, self.spec, 0)

    def test_reject_duplicate_ids(self):
        Path(self.spec["annotations"]).write_text(json.dumps([self.record] * 2), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_rows(self.cfg, self.spec)

    def test_clevr_questions_object(self):
        path = self.root / "clevr.json"
        path.write_text(json.dumps({"questions": [{"question": "q"}]}), encoding="utf-8")
        self.assertEqual(read_records(path), [{"question": "q"}])

    @unittest.skipUnless(importlib.util.find_spec("datasets"), "datasets not installed locally")
    def test_arrow_schema_accepts_multimodal_messages(self):
        row = convert_record(self.record, self.cfg, self.spec, 0)
        dataset = as_dataset([row])
        self.assertEqual(len(dataset), 1)
        self.assertIn("prompt", dataset.column_names)


class RewardTests(unittest.TestCase):
    def test_exact_and_word_counts(self):
        texts = ["<think>There are two.</think><answer>two</answer>", "<answer>2</answer>"]
        self.assertEqual(accuracy_reward(texts, ["2", "2"]), [1., 1.])
        self.assertEqual(format_reward(texts), [1., 0.])

    def test_reward_hacking_and_missing_tags(self):
        texts = ["<answer>2 or 3</answer>", "<answer>2</answer><answer>3</answer>",
                 "<think>The correct count is 2</think><answer>3</answer>", "2", ""]
        self.assertEqual(accuracy_reward(texts, ["2"] * len(texts)), [0.] * len(texts))

    def test_strict_format_and_nonempty_reasoning(self):
        self.assertEqual(format_reward(["prefix <think>x</think><answer>2</answer>",
                                        "<think> </think><answer>2</answer>",
                                        "<think>x</think><answer> </answer>"]), [0., 0., 0.])

    def test_non_count_clevr_tasks(self):
        self.assertEqual(accuracy_reward(["<answer>Red.</answer>", "<answer>yes</answer>"],
                                         ["red", "no"]), [1., 0.])

    def test_conversational_completion(self):
        value = [{"role": "assistant", "content": "<answer>3</answer>"}]
        self.assertEqual(predicted_answer(value), "3")

    def test_invalid_gold_never_scores(self):
        self.assertEqual(accuracy_reward(["nonsense"], ["nonsense"]), [0.])
        with self.assertRaises(ValueError):
            accuracy_reward(["x"], [])


class HandoffAndTimingTests(unittest.TestCase):
    def test_reference_requires_sft_provenance(self):
        cfg = load_config("grpo_5090")
        with tempfile.TemporaryDirectory() as directory:
            cfg["sft_adapter"] = directory
            with self.assertRaises(FileNotFoundError):
                validate_adapter(cfg)
            path = Path(directory)
            (path / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_adapter(cfg)
            sft = load_config("sft_5090")
            (path / "experiment.json").write_text(json.dumps(sft), encoding="utf-8")
            self.assertEqual(validate_adapter(cfg), directory)
            sft["prompt"]["template"] = "different {prompt}"
            (path / "experiment.json").write_text(json.dumps(sft), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_adapter(cfg)

    def test_rollouts_count_in_step_estimate(self):
        sft, grpo = load_config("sft_5090"), load_config("grpo_5090")
        self.assertEqual(optimizer_steps(sft, 30069), 940)
        self.assertEqual(optimizer_steps(grpo, 10000), 1250)

    def test_estimate_scales_with_dataset_and_group_size(self):
        cfg = load_config("grpo_5090")
        initial = planning_estimate(cfg, 1000)
        doubled = planning_estimate(cfg, 2000)
        self.assertAlmostEqual(doubled["hours_low"], initial["hours_low"] * 2)
        cfg["train"]["num_generations"] *= 2
        self.assertAlmostEqual(planning_estimate(cfg, 1000)["hours_high"], initial["hours_high"] * 2)

    def test_explicit_steps_override_epochs(self):
        cfg = load_config("grpo_5090")
        cfg["train"]["max_steps"] = 30
        self.assertEqual(optimizer_steps(cfg, 100000), 30)
        self.assertEqual(planning_estimate(cfg, 1000)["hours_low"],
                         planning_estimate(cfg, 100000)["hours_low"])

    def test_prompt_length_guard(self):
        from types import SimpleNamespace
        cfg = load_config("grpo_5090")
        row = {"id": "long", "prompt": make_prompt("test", cfg)}
        processor = SimpleNamespace(apply_chat_template=lambda *a, **k: "text",
                                    tokenizer=SimpleNamespace(encode=lambda *a, **k: [1] * 4000))
        with self.assertRaisesRegex(ValueError, "long"):
            check_prompt_lengths([row], cfg, processor)

    def test_length_guard_preserves_native_mask(self):
        from types import SimpleNamespace
        from synselect.runtime import LengthGuardCollator
        batch = {"input_ids": SimpleNamespace(shape=(1, 100)), "labels": [-100, 3]}
        guarded = LengthGuardCollator(lambda _: batch, 200)
        self.assertIs(guarded([{"id": "ok"}]), batch)
        with self.assertRaises(ValueError):
            LengthGuardCollator(lambda _: batch, 50)([{"id": "too_long"}])


if __name__ == "__main__":
    unittest.main()
