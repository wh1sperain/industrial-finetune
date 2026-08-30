"""冒烟测试：CI 中运行的轻量校验（不加载模型、不真训练）。

设计原则：CI 只保证「合进来的代码不会立刻坏掉」，
完整训练在双方机器上跑（见 docs/实验规范.md）。
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_train_config_loads():
    with open(ROOT / "configs" / "train" / "lora_qwen7b_industrial.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "Qwen" in cfg["model_name_or_path"]
    assert cfg["lora_r"] > 0
    assert "output_dir" in cfg
    # M1 定案数据：Mining-Engineering 矿建工程指令集
    assert "Mining-Engineering" in cfg["dataset"]
    # 5090 加速开关（docs/训练方案.md §5.2）
    assert cfg.get("torch_compile") is True
    assert cfg.get("attn_implementation") == "sdpa"
    assert cfg.get("optim") == "adamw_torch_fused"


def test_eval_config_loads():
    with open(ROOT / "configs" / "eval" / "industrial_benchmark.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "adapter_path" in cfg


def test_seed_is_deterministic():
    import random

    from industrial_ft.utils.seed import set_seed

    set_seed(42)
    a = random.random()
    set_seed(42)
    b = random.random()
    assert a == b


def test_gitignore_blocks_secrets_and_weights():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in [".env", "*.safetensors", "output/", "wandb/"]:
        assert entry in gitignore, f".gitignore 缺少 {entry}"
