"""Sampling and wall-clock protections, runnable without torch or server access."""
import copy
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from synselect.run_budget import compare_predictions, run_child
from synselect.utils.budget import WallBudget
from synselect.utils.config import load_config
from synselect.utils.sampling import clevr_family, program_depth, stratified_sample
from synselect.utils.timing import optimizer_steps, planning_estimate


def sft_records(n=100):
    return [{"id": str(i), "image": f"{i}.png", "conversations": [
        {"value": "Choices: A/B" if i % 2 else "Explain"},
        {"value": "x" * (100 + (i % 4) * 600)}]} for i in range(n)]


def clevr_records(n=40):
    return [{"question_index": i, "image_filename": f"{i}.png", "answer": "yes" if i < 30 else "no",
             "program": [{"function": "scene", "inputs": []},
                         {"function": "exist", "inputs": [0]}]} for i in range(n)]


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.sft = load_config("sft_8h")["dataset"]["sampling"]
        self.clevr = load_config("grpo_8h")["dataset"]["sampling"]

    def test_fixed_seed_independent_of_source_order(self):
        records = sft_records()
        a, report_a = stratified_sample(records, 20, self.sft, 42)
        b, report_b = stratified_sample(list(reversed(records)), 20, self.sft, 42)
        self.assertEqual(a, b)
        self.assertEqual(report_a, report_b)
        changed, _ = stratified_sample(records, 20, self.sft, 7)
        self.assertNotEqual(a, changed)

    def test_exact_size_and_length_exclusion(self):
        records = sft_records()
        records[0]["conversations"][1]["value"] = "x" * 3201
        selected, report = stratified_sample(records, 20, self.sft, 42)
        self.assertEqual(len(selected), 20)
        self.assertEqual(report["excluded"]["completion_over_character_cap"], 1)
        self.assertNotIn("0", report["selected_ids"])

    def test_image_uniqueness_across_strata(self):
        records = sft_records()
        for i, record in enumerate(records):
            record["image"] = f"{i // 2}.png"
        selected, report = stratified_sample(records, 40, self.sft, 42)
        self.assertEqual(len({r["image"] for r in selected}), 40)
        self.assertEqual(report["unique_images"], 40)

    def test_insufficient_pool_fails_instead_of_silently_undersampling(self):
        with self.assertRaisesRegex(ValueError, "only 3"):
            stratified_sample(sft_records(3), 5, self.sft, 42)

    def test_duplicate_source_ids_rejected(self):
        records = sft_records(4)
        records[-1]["id"] = records[0]["id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            stratified_sample(records, 2, self.sft, 42)

    def test_balanced_answers_in_clevr_stratum(self):
        selected, _ = stratified_sample(clevr_records(), 12, self.clevr, 42)
        self.assertEqual(Counter(r["answer"] for r in selected), {"yes": 6, "no": 6})

    def test_balanced_tasks_and_no_program_fallback(self):
        records = clevr_records()
        for record in records[:20]:
            record["program"][-1]["function"] = "count"
            record["answer"] = "2"
        _, report = stratified_sample(records, 12, self.clevr, 42)
        self.assertEqual(report["selected_strata"], {"count/depth_0": 6, "exist/depth_0": 6})
        del records[0]["program"]
        with self.assertRaisesRegex(ValueError, "program annotations"):
            stratified_sample(records, 12, self.clevr, 42)

    def test_proportional_allocation(self):
        records = sft_records()
        for i, record in enumerate(records):
            record["conversations"][0]["value"] = "Choices:" if i < 90 else "Explain"
            record["conversations"][1]["value"] = "short"
        _, report = stratified_sample(records, 10, self.sft, 42)
        self.assertEqual(report["selected_strata"],
                         {"unspecified/choice/length_0": 9, "unspecified/open/length_0": 1})

    def test_dag_depth_not_operation_count(self):
        program = [{"inputs": []}, {"inputs": [0]}, {"inputs": [0]},
                   {"inputs": [1, 2]}, {"inputs": [3]}]
        self.assertEqual(program_depth(program), 4)
        with self.assertRaises(ValueError):
            program_depth([{"inputs": [0]}])
        self.assertEqual(clevr_family("equal_integer"), "compare_number")
        self.assertEqual(clevr_family("equal_shape"), "compare_attribute")


class BudgetTests(unittest.TestCase):
    def test_five_hour_calibration_and_handoff(self):
        from synselect.utils.config import local_path
        sft, grpo = load_config("sft_5h"), load_config("grpo_5h")
        self.assertEqual(sft["eval"], grpo["eval"])
        self.assertEqual(local_path(grpo["sft_adapter"]), Path(sft["output_dir"]))
        self.assertEqual(optimizer_steps(sft, 8192), 512)
        self.assertEqual(optimizer_steps(grpo, 1024), 512)
        self.assertAlmostEqual(planning_estimate(sft, 8192)["hours_low"], 1)
        estimate = planning_estimate(grpo, 1024)
        self.assertAlmostEqual(estimate["hours_low"], 160 / 60)
        self.assertIn("in_progress_projection", estimate["basis"])

    def test_five_hour_stage_limits_fit_total(self):
        from industrial_ft.utils.config import load_yaml, REPO_ROOT
        plan = load_yaml(REPO_ROOT / "configs/experiments/synselect/budget_5h.yaml")
        sft = load_config(plan["sft_experiment"])
        grpo = load_config(plan["grpo_experiment"])
        scheduled = (plan["preflight_seconds"] + sft["runtime"]["max_wall_seconds"] +
                     grpo["runtime"]["max_wall_seconds"] + 3 * grpo["eval"]["max_wall_seconds"])
        self.assertEqual(scheduled, 295 * 60)
        self.assertEqual(plan["total_seconds"], 5 * 3600)
        self.assertLess(scheduled + plan["cleanup_seconds"], plan["total_seconds"])

    def test_deadline_and_reserve_choose_earliest_bound(self):
        current = [0.]
        budget = WallBudget(seconds=100, deadline=1060, reserve=10,
                            clock=lambda: current[0], wall_clock=lambda: 1000)
        self.assertEqual(budget.remaining(), 60)
        current[0] = 41
        self.assertTrue(budget.should_stop(next_work_seconds=10))
        self.assertFalse(budget.should_stop(next_work_seconds=0))
        current[0] = 51
        self.assertTrue(budget.should_stop())

    def test_unbudgeted_legacy_behavior(self):
        self.assertFalse(WallBudget().should_stop(1000000))

    def test_full_pipeline_commands_and_budget_limited_status(self):
        from synselect.run_budget import main
        sft, grpo = load_config("sft_8h"), load_config("grpo_8h")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sft["output_dir"] = str(root / "sft")
            grpo["output_dir"] = str(root / "grpo")
            grpo["sft_adapter"] = sft["output_dir"]
            grpo["eval"]["dataset"]["max_samples"] = 1
            sft["eval"] = copy.deepcopy(grpo["eval"])
            configs = {"sft_8h": sft, "grpo_8h": grpo}
            plan = {"sft_experiment": "sft_8h", "grpo_experiment": "grpo_8h",
                    "total_seconds": 28800, "preflight_seconds": 900, "cleanup_seconds": 30,
                    "eval_aggregate_tokens_per_second": [40, 100],
                    "output_dir": str(root / "reports")}
            modules = []

            def fake_child(command, timeout, log):
                module = command[command.index("-m") + 1]
                modules.append(module)
                self.assertGreater(timeout, 0)
                if module == "synselect.train":
                    self.assertIn("--deadline", command)
                    cfg = configs[command[command.index("--experiment") + 1]]
                    target = Path(cfg["output_dir"])
                    target.mkdir()
                    (target / "timing.json").write_text(json.dumps({
                        "completed_optimizer_steps": 1, "budget_stopped": True}), encoding="utf-8")
                elif module == "synselect.evaluate":
                    self.assertIn("--deadline", command)
                    target = Path(command[command.index("--output") + 1])
                    target.write_text(json.dumps({"id": "a", "answer": "2", "correct": 1}) + "\n",
                                      encoding="utf-8")

            with patch("synselect.run_budget.load_yaml", return_value=plan), \
                    patch("synselect.run_budget.load_config", side_effect=lambda n: configs[n]), \
                    patch("synselect.run_budget.run_child", side_effect=fake_child), \
                    patch("synselect.run_budget.os", SimpleNamespace(name="posix", environ={})), \
                    patch.object(sys, "argv", ["run_budget"]):
                main()
            report = json.loads((root / "reports/result.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "budget_limited_training")
            self.assertEqual(modules, ["synselect.prepare_budget", "synselect.train", "synselect.train",
                                       "synselect.evaluate", "synselect.evaluate", "synselect.evaluate"])

    def test_matching_eval_sets_and_steps(self):
        sft, grpo = load_config("sft_8h"), load_config("grpo_8h")
        self.assertEqual(sft["eval"], grpo["eval"])
        self.assertEqual(optimizer_steps(sft, 2048), 128)
        self.assertEqual(optimizer_steps(grpo, 256), 128)
        self.assertEqual(grpo["train"]["num_generations"], 4)

    def test_callback_stops_with_save_reserve(self):
        from synselect.train import timing_callback
        cfg = load_config("sft_8h")
        current = [0.]
        budget = WallBudget(seconds=100, reserve=20, clock=lambda: current[0])
        cuda = SimpleNamespace(synchronize=lambda: None)
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda),
                                      "transformers": SimpleNamespace(TrainerCallback=object)}), \
                patch("synselect.train.time.perf_counter", side_effect=lambda: current[0]):
            callback = timing_callback(cfg, {}, 2048, None, budget)
            current[0] = 60
            callback.on_step_begin(None, None, None)
            current[0] = 70
            control = SimpleNamespace(should_training_stop=False)
            callback.on_step_end(None, SimpleNamespace(global_step=1, max_steps=128), control)
            self.assertTrue(control.should_training_stop)
            self.assertTrue(callback.budget_stopped)

    def test_supervisor_kills_a_timed_out_child(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "child.log"
            with self.assertRaises(TimeoutError):
                run_child([sys.executable, "-c", "import time; time.sleep(30)"], .1, log)

    def test_supervisor_reports_child_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "exited 2"):
                run_child([sys.executable, "-c", "raise SystemExit(2)"], 5,
                          Path(directory) / "child.log")

    def test_evaluation_uses_only_common_ids_after_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for model, ids in {"base": ["a", "b"], "sft": ["a"], "grpo": ["a", "b"]}.items():
                path = Path(directory) / f"{model}.jsonl"
                path.write_text("\n".join(json.dumps({"id": key, "answer": "2", "correct": 1})
                                          for key in ids), encoding="utf-8")
                paths[model] = path
            report = compare_predictions(paths, 2)
            self.assertFalse(report["complete"])
            self.assertEqual(report["common_ids"], ["a"])
            self.assertEqual(report["completed_by_model"], {"base": 2, "sft": 1, "grpo": 2})


if __name__ == "__main__":
    unittest.main()
