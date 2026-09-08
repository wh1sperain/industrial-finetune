"""Independent experiment namespace; reuse the workspace model-path resolver."""
from __future__ import annotations

import os
import re
from pathlib import Path

from industrial_ft.utils.config import REPO_ROOT, load_yaml, resolve_model_name


def local_path(value: str) -> Path:
    value = os.path.expandvars(os.path.expanduser(value))
    if re.search(r"\$\{[^}]+\}", value):
        raise ValueError(f"Set the environment variable in this path first: {value}")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(name: str) -> dict:
    if not re.fullmatch(r"[\w-]+", name):
        raise ValueError("Use an experiment name, not a path")
    directory = Path(os.environ.get("INDUSTRIAL_FT_EXPERIMENTS_DIR") or
                     REPO_ROOT / "configs/experiments/synselect")
    cfg = load_yaml(directory / f"{name}.yaml")
    cfg["model_name_or_path"] = resolve_model_name(cfg["model"])
    cfg["output_dir"] = str(local_path(cfg["output_dir"]))
    cfg.setdefault("run_name", name)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    stage, train = cfg["stage"], cfg["train"]
    for section in (cfg["runtime"], cfg["eval"]):
        cap = section.get("max_wall_seconds")
        reserve = section.get("save_reserve_seconds", 0)
        if cap is not None and (cap <= 0 or reserve < 0 or reserve >= cap):
            raise ValueError("Wall-clock cap must be positive and greater than save reserve")
    for spec in (cfg["dataset"], cfg["eval"].get("dataset", {})):
        sampling = spec.get("sampling", {})
        for key in ("length_bins", "depth_bins"):
            if key in sampling and sorted(set(sampling[key])) != sampling[key]:
                raise ValueError(f"{key} must be strictly increasing")
    if stage not in {"sft", "grpo"}:
        raise ValueError("stage must be sft or grpo")
    if cfg["model_options"]["attn_implementation"] != "sdpa":
        raise ValueError("This RTX 5090 experiment requires SDPA")
    if not train["bf16"] or train.get("fp16", False):
        raise ValueError("Use bf16 for this experiment")
    if cfg["eval"].get("do_sample") is not False:
        raise ValueError("Evaluation must use do_sample=false")
    for key in ("per_device_train_batch_size", "gradient_accumulation_steps"):
        if train[key] < 1:
            raise ValueError(f"{key} must be positive")
    if cfg["processor"]["min_pixels"] > cfg["processor"]["max_pixels"]:
        raise ValueError("min_pixels must not exceed max_pixels")
    if stage == "sft":
        if train.get("max_length") is not None or train.get("packing"):
            raise ValueError("Do not truncate/pack visual SFT sequences; use the length guard")
        if not train.get("completion_only_loss"):
            raise ValueError("SFT must mask prompt and image tokens")
    else:
        g = train["num_generations"]
        batch = train["generation_batch_size"]
        if g < 2 or batch % g or batch % train["per_device_train_batch_size"]:
            raise ValueError("generation_batch_size must contain whole GRPO groups/batches")
        steps_per_generation = batch // train["per_device_train_batch_size"]
        if train["gradient_accumulation_steps"] % steps_per_generation:
            raise ValueError("Accumulate complete generation batches per optimizer step")
        if train["loss_type"] != "grpo" or train["num_iterations"] != 1:
            raise ValueError("This reproduction uses GRPO loss and one update per rollout")
        if train["generation_kwargs"].get("do_sample") is not True:
            raise ValueError("GRPO needs stochastic group sampling")
        if not cfg.get("sft_adapter"):
            raise ValueError("GRPO requires a completed SFT adapter")
