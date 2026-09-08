"""Independent CLEVR accuracy smoke evaluation; paper benchmarks use VLMEvalKit."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from synselect.data import load_rows
from synselect.utils.config import load_config, local_path
from synselect.utils.rewards import accuracy_reward, format_reward, predicted_answer
from synselect.utils.budget import WallBudget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--adapter", help="Override adapter with a checkpoint path")
    parser.add_argument("--output", required=True, help="New predictions JSONL path")
    parser.add_argument("--deadline", type=float, help="Supervisor's absolute Unix deadline")
    args = parser.parse_args()
    cfg = load_config(args.experiment)
    budget = WallBudget(cfg["eval"].get("max_wall_seconds"), args.deadline,
                        cfg["eval"].get("save_reserve_seconds", 0))
    output = local_path(args.output)
    if output.exists() or output.with_suffix(".metrics.json").exists():
        raise FileExistsError(output)
    data_cfg = copy.deepcopy(cfg)
    data_cfg["stage"] = "grpo"  # Gold CLEVR labels, never the SFT completion representation
    rows, selection = load_rows(data_cfg, cfg["eval"]["dataset"], return_selection=True)

    import torch
    from PIL import Image
    from transformers import StoppingCriteria, StoppingCriteriaList
    from synselect.runtime import build_model, build_processor, check_environment
    check_environment(cfg)
    processor = build_processor(cfg)
    processor.tokenizer.padding_side = "left"
    adapter = None if args.baseline else str(local_path(args.adapter or cfg["output_dir"]))
    if adapter and not (Path(adapter) / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Adapter not found: {adapter}")
    model = build_model(cfg, adapter, trainable=False).to("cuda").eval()
    class DeadlineStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return torch.full((input_ids.shape[0],), budget.should_stop(),
                              device=input_ids.device, dtype=torch.bool)

    output.parent.mkdir(parents=True, exist_ok=True)
    correct = formatted = truncated = completed = 0
    with output.open("x", encoding="utf-8") as handle, torch.inference_mode():
        for row in rows:
            if budget.should_stop():
                break
            text = processor.apply_chat_template(row["prompt"], tokenize=False,
                                                 add_generation_prompt=True)
            with Image.open(row["image"]) as source:
                image = source.convert("RGB")
            inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
            if inputs.input_ids.shape[1] + cfg["eval"]["max_new_tokens"] > cfg["runtime"]["max_sequence_tokens"]:
                raise ValueError(f"Evaluation sequence too long: {row['id']}")
            generated = model.generate(**inputs, do_sample=cfg["eval"]["do_sample"],
                                       max_new_tokens=cfg["eval"]["max_new_tokens"], use_cache=True,
                                       stopping_criteria=StoppingCriteriaList([DeadlineStop()]))
            if budget.should_stop():
                # A deadline-interrupted answer is not a completed evaluation observation.
                break
            tokens = generated[0, inputs.input_ids.shape[1]:]
            prediction = processor.tokenizer.decode(tokens, skip_special_tokens=True)
            hit = accuracy_reward([prediction], [row["answer"]])[0]
            valid_format = format_reward([prediction])[0]
            eos_ids = model.generation_config.eos_token_id
            eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
            capped = len(tokens) == cfg["eval"]["max_new_tokens"] and tokens[-1].item() not in eos_ids
            correct += hit
            formatted += valid_format
            truncated += capped
            completed += 1
            handle.write(json.dumps({"id": row["id"], "answer": row["answer"],
                                     "prediction": prediction, "extracted": predicted_answer(prediction),
                                     "correct": hit, "truncated": capped}, ensure_ascii=False) + "\n")
            handle.flush()
    metrics = {"samples": completed, "requested_samples": len(rows),
               "budget_stopped": completed < len(rows),
               "tagged_answer_accuracy": correct / completed if completed else None,
               "format_rate": formatted / completed if completed else None,
               "truncation_rate": truncated / completed if completed else None,
               "baseline": args.baseline, "adapter": adapter,
               "note": "CLEVR diagnostic with strict answer tags, not a paper benchmark score"}
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if selection:
        output.with_suffix(".selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
