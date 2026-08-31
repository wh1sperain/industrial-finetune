"""冒烟测试：CI 中运行的轻量校验（不加载模型、不真训练）。

设计原则：CI 只保证「合进来的代码不会立刻坏掉」，
完整训练在双方机器上跑（见 docs/实验规范.md）。
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_train_config_loads():
    from industrial_ft.utils.config import load_config

    cfg = load_config(ROOT / "configs" / "train" / "lora_qwen7b_industrial.yaml")
    # 模型字段由 configs/models.yaml 单一事实源注入
    assert "Qwen" in cfg["model_name_or_path"]
    assert "output_dir" in cfg and cfg["output_dir"]
    assert "run_name" in cfg and cfg["run_name"]
    assert cfg["lora_r"] > 0
    # M1 定案数据：Mining-Engineering 矿建工程指令集
    assert "Mining-Engineering" in cfg["dataset"]
    # 5090 加速开关（docs/训练方案.md §5.2）
    assert cfg.get("torch_compile") is True
    assert cfg.get("attn_implementation") == "sdpa"
    assert cfg.get("optim") == "adamw_torch_fused"


def test_eval_config_loads():
    from industrial_ft.utils.config import load_config

    cfg = load_config(ROOT / "configs" / "eval" / "industrial_benchmark.yaml")
    # 模型/adapter 字段由 models.yaml 注入，与训练配置同源
    assert "Qwen" in cfg["model_name_or_path"]
    assert cfg.get("adapter_dir")
    assert "Mining-Engineering-Eval" in cfg["eval_dataset"]
    assert "Mining-Engineering-Probe" in cfg["probe_dataset"]


def test_models_config_is_single_source():
    from industrial_ft.utils.config import load_models_config

    models = load_models_config()
    assert "Qwen" in models["base_model"]
    assert models["adapter_dir"]
    assert models["run_name"]
    # model_path 允许为空（download_model.sh 写入）；若非空则必须是合法字段
    assert "model_path" in models


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
