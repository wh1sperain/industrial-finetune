"""One deadline for preflight, SFT, GRPO and the three matched evaluations."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from industrial_ft.utils.config import load_yaml, REPO_ROOT
from synselect.utils.config import load_config, local_path
from synselect.utils.timing import pipeline_estimate


def run_child(command, timeout, log_path):
    """Own only this job's process group; never use name-based GPU process killing."""
    if timeout <= 0:
        raise TimeoutError("No wall-clock budget remains")
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=(os.name == "posix"))
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(f"Process stopped at deadline; inspect {log_path}") from error
        finally:
            # Also clean up on Ctrl+C or an exception in the supervisor.
            if process.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        if code:
            raise RuntimeError(f"Child exited {code}; inspect {log_path}")


def compare_predictions(paths: dict, expected_count: int) -> dict:
    tables = {}
    for model, path in paths.items():
        with Path(path).open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        ids = [r["id"] for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate prediction IDs in {path}")
        tables[model] = {r["id"]: r for r in records}
    common = sorted(set.intersection(*(set(table) for table in tables.values())))
    for key in common:
        if len({table[key]["answer"] for table in tables.values()}) != 1:
            raise ValueError("Evaluation gold labels differ across models")
    return {"planned_questions": expected_count, "common_completed_questions": len(common),
            "complete": len(common) == expected_count,
            "completed_by_model": {k: len(v) for k, v in tables.items()},
            "tagged_answer_accuracy_on_common_questions": {
                model: sum(table[key]["correct"] for key in common) / len(common) if common else None
                for model, table in tables.items()},
            "common_ids": common,
            "note": "Small stratified CLEVR diagnostic; not the official distribution or paper benchmarks"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="budget_8h")
    parser.add_argument("--dry-run", action="store_true", help="Show budgets, without GPU/data loading")
    args = parser.parse_args()
    if not args.plan.replace("_", "").replace("-", "").isalnum():
        parser.error("Use a plan name, not a path")
    directory = Path(os.environ.get("INDUSTRIAL_FT_EXPERIMENTS_DIR") or
                     REPO_ROOT / "configs/experiments/synselect")
    started = time.monotonic()
    plan = load_yaml(directory / f"{args.plan}.yaml")
    sft, grpo = load_config(plan["sft_experiment"]), load_config(plan["grpo_experiment"])
    scheduled = (plan["preflight_seconds"] + sft["runtime"]["max_wall_seconds"] +
                 grpo["runtime"]["max_wall_seconds"] + 3 * grpo["eval"]["max_wall_seconds"])
    if (plan["total_seconds"] <= 0 or plan["cleanup_seconds"] < 10 or
            scheduled + plan["cleanup_seconds"] > plan["total_seconds"]):
        raise ValueError("Stage budgets plus cleanup must fit inside the total deadline")
    summary = {"total_hours": plan["total_seconds"] / 3600,
               "scheduled_hours_including_setup": scheduled / 3600,
               "sft_samples": sft["dataset"]["max_samples"],
               "grpo_questions": grpo["dataset"]["max_samples"],
               "eval_questions_per_model": grpo["eval"]["dataset"]["max_samples"],
               "eval_models": ["base", "sft", "grpo"],
               "planning_estimate": pipeline_estimate(plan, sft, grpo)}
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return
    if os.name != "posix":
        raise RuntimeError("Run this supervised training pipeline on the Linux GPU server")
    for cfg in (sft, grpo):
        path = Path(cfg["output_dir"])
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Use fresh experiment output paths for a budgeted run: {path}")
    output = local_path(plan["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    overall_end = started + plan["total_seconds"] - plan["cleanup_seconds"]
    result = {"plan": plan, "summary": summary, "stages": [], "status": "running"}

    def execute(name, module, arguments, seconds, accepts_deadline=True):
        duration = min(seconds, overall_end - time.monotonic())
        command = [sys.executable, "-u", "-m", module, *arguments]
        if accepts_deadline:
            command += ["--deadline", str(time.time() + duration)]
        print(f"Starting {name}, wall-clock limit {duration / 60:.1f} min; log: {output / (name + '.log')}", flush=True)
        beginning = time.monotonic()
        run_child(command, duration, output / f"{name}.log")
        result["stages"].append({"name": name, "seconds": time.monotonic() - beginning})

    try:
        execute("preflight", "synselect.prepare_budget",
                ["--sft-experiment", plan["sft_experiment"],
                 "--grpo-experiment", plan["grpo_experiment"], "--output", str(output)],
                plan["preflight_seconds"], False)
        for stage, cfg in (("sft", sft), ("grpo", grpo)):
            execute(stage, "synselect.train", ["--experiment", plan[f"{stage}_experiment"]],
                    cfg["runtime"]["max_wall_seconds"])
            stage_report = json.loads((Path(cfg["output_dir"]) / "timing.json").read_text(encoding="utf-8"))
            result["stages"][-1]["training"] = stage_report
            if stage_report["completed_optimizer_steps"] < 1:
                raise RuntimeError(f"{stage} did not complete an optimizer step")
        predictions = {}
        for label, experiment in (("base", plan["sft_experiment"]),
                                  ("sft", plan["sft_experiment"]),
                                  ("grpo", plan["grpo_experiment"])):
            path = output / f"eval_{label}.jsonl"
            arguments = ["--experiment", experiment, "--output", str(path)]
            if label == "base":
                arguments.append("--baseline")
            execute(f"eval_{label}", "synselect.evaluate", arguments,
                    grpo["eval"]["max_wall_seconds"])
            predictions[label] = path
        result["comparison"] = compare_predictions(predictions, grpo["eval"]["dataset"]["max_samples"])
        result["training_budget_stopped"] = any(
            stage.get("training", {}).get("budget_stopped", False) for stage in result["stages"])
        result["status"] = ("partial_evaluation" if not result["comparison"]["complete"] else
                            "budget_limited_training" if result["training_budget_stopped"] else "complete")
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Result: {output / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
