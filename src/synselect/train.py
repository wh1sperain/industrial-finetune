"""Server-only SFT/GRPO orchestration with CPU dry-run and timed pilot support."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from synselect.data import as_dataset, load_rows
from synselect.utils.config import load_config, local_path
from synselect.utils.rewards import accuracy_reward, format_reward
from synselect.utils.timing import optimizer_steps
from synselect.utils.budget import WallBudget


def timing_callback(cfg: dict, environment: dict, samples: int, stop_steps: int | None,
                    budget: WallBudget | None = None):
    import torch
    from transformers import TrainerCallback

    class TimingCallback(TrainerCallback):
        def __init__(self):
            self.durations = []
            self.started = 0.0
            self.local_steps = 0
            self.recent_steps = []
            self.budget_stopped = False

        def on_train_begin(self, args, state, control, **kwargs):
            torch.cuda.reset_peak_memory_stats()

        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.started = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            duration = time.perf_counter() - self.started
            self.local_steps += 1
            self.recent_steps = (self.recent_steps + [duration])[-5:]
            if self.local_steps > cfg["runtime"]["benchmark_warmup_steps"]:
                self.durations.append(duration)
            if stop_steps and state.global_step >= stop_steps:
                control.should_training_stop = True
            # Finish an optimizer step (never a partial accumulation), then leave time to save.
            margin = cfg["runtime"].get("step_time_safety_factor", 1.5)
            if (budget and state.global_step < state.max_steps and
                    budget.should_stop(max(self.recent_steps) * margin)):
                self.budget_stopped = True
                control.should_training_stop = True
                print("Wall-clock budget reached: saving this stage and retaining evaluation time.")

        def on_train_end(self, args, state, control, **kwargs):
            report = {"stage": cfg["stage"], "samples": samples, "environment": environment,
                      "config": cfg, "optimizer_steps_in_schedule": state.max_steps,
                      "measured_optimizer_steps": len(self.durations),
                      "completed_optimizer_steps": state.global_step,
                      "completed_epoch_fraction": state.epoch,
                      "budget_stopped": self.budget_stopped,
                      "seconds_per_step": self.durations,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                      "note": "Excludes loading, saving and inter-step logging; GRPO lengths evolve."}
            if self.durations:
                mean = statistics.mean(self.durations)
                report.update(mean_seconds_per_optimizer_step=mean,
                              projected_train_hours=mean * state.max_steps / 3600)
            path = Path(args.output_dir) / "timing.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Timing report: {path}")

    return TimingCallback()


def validate_adapter(cfg: dict) -> str:
    adapter = local_path(cfg["sft_adapter"])
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Complete SFT first; no adapter_config.json at {adapter}")
    if adapter.resolve() == Path(cfg["output_dir"]).resolve():
        raise ValueError("GRPO output_dir must differ from the SFT adapter directory")
    adapter_cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if adapter_cfg.get("peft_type") != "LORA":
        raise ValueError("Expected a LoRA SFT adapter")
    # Workspace runs record their base model and visual/prompt settings alongside the adapter.
    provenance = adapter / "experiment.json"
    if not provenance.is_file():
        raise ValueError("Missing SFT experiment.json; use an adapter produced by this experiment")
    previous = json.loads(provenance.read_text(encoding="utf-8"))
    for key in ("model", "prompt", "processor"):
        if previous[key] != cfg[key]:
            raise ValueError(f"SFT/GRPO {key} settings differ; use the SFT settings")
    return str(adapter)


def check_prompt_lengths(rows, cfg, processor):
    """Conservative visual-token upper bound, before rollout allocation (no truncation)."""
    image_bound = cfg["processor"]["max_pixels"] // (28 * 28)
    limit = cfg["runtime"]["max_sequence_tokens"]
    for row in rows:
        text = processor.apply_chat_template(row["prompt"], tokenize=False,
                                             add_generation_prompt=True)
        length = len(processor.tokenizer.encode(text, add_special_tokens=False)) + image_bound
        if length + cfg["train"]["max_completion_length"] > limit:
            raise ValueError(f"GRPO prompt {row['id']} + completion exceeds {limit} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate local records/images only")
    parser.add_argument("--check-env", action="store_true", help="Check server APIs and bf16 SDPA")
    parser.add_argument("--resume", nargs="?", const=True, default=None,
                        help="Resume latest checkpoint or a specified checkpoint directory")
    parser.add_argument("--benchmark-steps", type=int,
                        help="Stop after N optimizer steps, keep full LR schedule, use isolated output")
    parser.add_argument("--deadline", type=float, help="Supervisor's absolute Unix deadline")
    args = parser.parse_args()
    cfg = load_config(args.experiment)
    budget = WallBudget(cfg["runtime"].get("max_wall_seconds"), args.deadline,
                        cfg["runtime"].get("save_reserve_seconds", 0))
    if args.benchmark_steps is not None and args.benchmark_steps <= 0:
        parser.error("benchmark-steps must be positive")
    if args.check_env:
        from synselect.runtime import check_environment
        check_environment(cfg, kernel_test=True)
        return
    rows, selection = load_rows(cfg, return_selection=True)
    print(json.dumps({"stage": cfg["stage"], "samples": len(rows),
                      "optimizer_steps_approx": optimizer_steps(cfg, len(rows)),
                      "model": cfg["model_name_or_path"], "output_dir": cfg["output_dir"]}, indent=2))
    if selection:
        print(json.dumps({k: v for k, v in selection.items() if k != "selected_ids"}, indent=2))
    if args.dry_run:
        if cfg["stage"] == "grpo":
            print("Data validated. SFT adapter and server dependencies are checked at training time.")
        return
    if args.benchmark_steps:
        if args.resume:
            raise ValueError("Benchmark from the stage's starting model; do not combine with resume")
        if args.benchmark_steps <= cfg["runtime"]["benchmark_warmup_steps"]:
            raise ValueError("benchmark-steps must exceed benchmark_warmup_steps")
        cfg["output_dir"] = str(Path(cfg["output_dir"]).parent / "benchmarks" /
                                 f"{args.experiment}_{datetime.now():%Y%m%d_%H%M%S_%f}")
        cfg["train"]["save_strategy"] = "no"
        cfg["train"]["report_to"] = "none"
    output = Path(cfg["output_dir"])
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is nonempty: {output}. Use --resume or a new output_dir.")
    adapter = validate_adapter(cfg) if cfg["stage"] == "grpo" else None

    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
    from transformers import set_seed
    from synselect.runtime import (LengthGuardCollator, build_model, build_processor,
                                   check_environment)

    environment = check_environment(cfg, kernel_test=True)
    set_seed(cfg["train"]["seed"])
    processor = build_processor(cfg)
    if cfg["stage"] == "grpo":
        check_prompt_lengths(rows, cfg, processor)
    model = build_model(cfg, adapter)
    dataset = as_dataset(rows)
    train_kwargs = dict(cfg["train"], output_dir=cfg["output_dir"], run_name=cfg["run_name"])
    callback = timing_callback(cfg, environment, len(rows), args.benchmark_steps, budget)
    if cfg["stage"] == "sft":
        trainer = SFTTrainer(model=model, args=SFTConfig(**train_kwargs), train_dataset=dataset,
                             processing_class=processor, peft_config=LoraConfig(**cfg["lora"]),
                             callbacks=[callback])
        trainer.data_collator = LengthGuardCollator(trainer.data_collator,
                                                    cfg["runtime"]["max_sequence_tokens"])
    else:
        # An already trainable SFT adapter is intentional. TRL >=1.12 copies it into frozen
        # "ref" instead of using the untuned base model for KL; no second 7B model is needed.
        trainer = GRPOTrainer(model=model, args=GRPOConfig(**train_kwargs), train_dataset=dataset,
                              processing_class=processor,
                              reward_funcs=[accuracy_reward, format_reward], callbacks=[callback])
        if cfg["train"]["beta"] > 0 and "ref" not in trainer.model.peft_config:
            raise RuntimeError("TRL did not create the SFT reference adapter; cannot use base-model KL")
        if any(p.requires_grad for n, p in trainer.model.named_parameters() if ".ref." in n):
            raise RuntimeError("Reference adapter must be frozen")
    trainable = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n or ".visual." in n for n in trainable):
        raise RuntimeError("Expected language-model LoRA only; inspect trainable parameters")
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    if selection:
        (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    if budget.should_stop():
        raise TimeoutError("Stage budget exhausted during model/data setup; no training started")
    trainer.train(resume_from_checkpoint=args.resume)
    if not args.benchmark_steps:
        trainer.save_model(cfg["output_dir"])
        processor.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
