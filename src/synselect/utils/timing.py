"""Auditable planning arithmetic; no GPU imports."""
from __future__ import annotations

import math
import copy


def optimizer_steps(cfg: dict, samples: int) -> int:
    train = cfg["train"]
    if train.get("max_steps", -1) > 0:
        return train["max_steps"]
    repeats = train["num_generations"] if cfg["stage"] == "grpo" else 1
    batch = train["per_device_train_batch_size"] * train["gradient_accumulation_steps"]
    return math.ceil(math.ceil(samples * repeats / batch) * train["num_train_epochs"])


def planning_estimate(cfg: dict, samples: int) -> dict:
    t, assumptions = cfg["train"], cfg["estimate"]
    exposures = samples * t["num_train_epochs"]
    if t.get("max_steps", -1) > 0:
        exposures = (t["max_steps"] * t["per_device_train_batch_size"] *
                     t["gradient_accumulation_steps"] /
                     (t["num_generations"] if cfg["stage"] == "grpo" else 1))
    calibration = assumptions.get("calibration")
    if calibration:
        if any(calibration[key] <= 0 for key in ("seconds", "samples", "epochs")):
            raise ValueError("Calibration seconds, samples and epochs must be positive")
        low = high = exposures * calibration["seconds"] / (
            calibration["samples"] * calibration["epochs"])
    elif cfg["stage"] == "sft":
        low, high = [exposures * rate for rate in assumptions["seconds_per_sample"]]
    else:
        completions = exposures * t["num_generations"]
        tokens = completions * assumptions["average_completion_tokens"]
        slow, fast = assumptions["aggregate_generation_tokens_per_second"]
        update_low, update_high = assumptions["update_seconds_per_completion"]
        low = tokens / fast + completions * update_low
        high = tokens / slow + completions * update_high
    multiplier = (1 + assumptions["overhead_fraction"]) / 3600
    return {"samples": samples, "optimizer_steps_approx": optimizer_steps(cfg, samples),
            "hours_low": low * multiplier, "hours_high": high * multiplier,
            "basis": (f"linear scaling from {calibration['status']}; not a new measurement"
                      if calibration else "planning assumptions, not measured on a GPU"),
            "assumptions": assumptions}


def pipeline_estimate(plan: dict, sft: dict, grpo: dict) -> dict:
    sft_time = planning_estimate(sft, sft["dataset"]["max_samples"])
    grpo_time = planning_estimate(grpo, grpo["dataset"]["max_samples"])
    if grpo["estimate"].get("calibration"):
        eval_tokens = 3 * grpo["eval"]["dataset"]["max_samples"] * grpo["eval"]["max_new_tokens"]
        slow, fast = plan["eval_aggregate_tokens_per_second"]
        return {"sft_reference_scaled_hours": sft_time["hours_low"],
                "grpo_reference_scaled_hours": grpo_time["hours_low"],
                "sft_basis": sft_time["basis"], "grpo_basis": grpo_time["basis"],
                "three_eval_decode_at_cap_hours": [eval_tokens / fast / 3600,
                                                     eval_tokens / slow / 3600],
                "note": "Training estimates scale user-reported timings, assuming unchanged throughput. "
                        "GRPO reference is an in-progress projection; generation length may change. "
                        "Evaluation throughput remains unmeasured and excludes model loading/prefill."}
    capped = copy.deepcopy(grpo)
    capped["estimate"]["average_completion_tokens"] = grpo["train"]["max_completion_length"]
    grpo_cap_time = planning_estimate(capped, grpo["dataset"]["max_samples"])
    # Three serial evaluations on the same fixed set: base, SFT and GRPO.
    eval_tokens = 3 * grpo["eval"]["dataset"]["max_samples"] * grpo["eval"]["max_new_tokens"]
    slow, fast = plan["eval_aggregate_tokens_per_second"]
    return {"sft_hours": [sft_time["hours_low"], sft_time["hours_high"]],
            "grpo_at_assumed_mean_length_hours": [grpo_time["hours_low"], grpo_time["hours_high"]],
            "grpo_at_completion_cap_hours": [grpo_cap_time["hours_low"], grpo_cap_time["hours_high"]],
            "three_eval_decode_at_cap_hours": [eval_tokens / fast / 3600, eval_tokens / slow / 3600],
            "note": "Unmeasured planning assumptions. Evaluation estimate excludes model loading/prefill; wall-clock caps govern execution."}
