"""GPU setup and memory bounds; imported only by explicit server commands."""
from __future__ import annotations

import inspect
import json
from importlib.metadata import version


def check_environment(cfg: dict, *, kernel_test: bool = False) -> dict:
    import torch
    from packaging.version import Version
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

    if not Version("1.12.0") <= Version(version("trl")) < Version("2.0"):
        raise RuntimeError("This implementation requires trl>=1.12.0,<2.0")
    if not torch.cuda.is_available():
        raise RuntimeError("Run training on the CUDA server; no CUDA device is available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES for these configs")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support bf16")
    config_cls = SFTConfig if cfg["stage"] == "sft" else GRPOConfig
    unknown = set(cfg["train"]) - set(inspect.signature(config_cls).parameters)
    if unknown:
        raise RuntimeError(f"Installed {config_cls.__name__} does not support: {sorted(unknown)}")
    for cls in (SFTTrainer, GRPOTrainer):
        if "processing_class" not in inspect.signature(cls).parameters:
            raise RuntimeError(f"Incompatible {cls.__name__} API")
    report = {"torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
              "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
              "packages": {name: version(name) for name in
                           ("transformers", "trl", "peft", "accelerate", "datasets", "pillow")}}
    report["expected"] = {k: v for k, v in cfg["runtime"].items() if k.startswith("expected_")}
    report["matches_requested_server"] = (
        torch.__version__.split("+")[0] == cfg["runtime"]["expected_torch"] and
        torch.version.cuda == cfg["runtime"]["expected_cuda"] and
        cfg["runtime"]["expected_gpu"] in report["gpu"])
    if kernel_test:
        # A real bf16 SDPA backward detects unsupported wheel/device kernels.
        q = torch.randn(1, 4, 32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        torch.nn.functional.scaled_dot_product_attention(q, q, q).float().sum().backward()
        torch.cuda.synchronize()
        report["bf16_sdpa_backward"] = "passed"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def build_processor(cfg: dict):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        cfg["model_name_or_path"], trust_remote_code=cfg["model_options"]["trust_remote_code"],
        **cfg["processor"])
    processor.tokenizer.padding_side = "left" if cfg["stage"] == "grpo" else "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def build_model(cfg: dict, adapter: str | None = None, *, trainable: bool = True):
    import torch
    from peft import PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model_name_or_path"], dtype=torch.bfloat16, **cfg["model_options"])
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    model.config.use_cache = not trainable
    return model


class LengthGuardCollator:
    """Keep TRL's tested VLM completion mask; abort before oversized GPU forwards."""

    def __init__(self, collator, max_tokens: int):
        self.collator, self.max_tokens = collator, max_tokens

    def __call__(self, examples):
        batch = self.collator(examples)
        if batch["input_ids"].shape[-1] > self.max_tokens:
            raise ValueError(f"Visual sequence exceeds {self.max_tokens} tokens; IDs: "
                             f"{[r['id'] for r in examples]}. Inspect lengths/resolution first.")
        return batch
