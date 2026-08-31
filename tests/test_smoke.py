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


def test_metrics_term_coverage():
    """L1 术语覆盖率：metrics 模块无 torch 依赖，CI 可测。"""
    from industrial_ft.utils.metrics import compute_metrics, extract_key_terms

    # 术语提取：引号术语 + 书名号规范 + 英文缩写
    terms = extract_key_terms('"欠挖"与"超挖"的区别，参考《GBT 50562-2019 标准》和 CAPEX 指标')
    assert "欠挖" in terms and "超挖" in terms
    assert any("GBT 50562" in t for t in terms)
    assert "CAPEX" in terms

    # 覆盖率：输出含一半术语 → 0.5
    refs = ['"凿井"与"巷道掘进"的差异（CAPEX 影响）']
    preds = ["凿井和巷道掘进…… CAPEX 方面……"]  # 命中：凿井/巷道掘进/CAPEX（3/4 取决于提取）
    m = compute_metrics(preds, refs, ["concept"])
    assert 0 < m["term_coverage"] <= 1.0
    # 完全没命中 → 0
    m0 = compute_metrics(["完全无关的回答"], refs, ["concept"])
    assert m0["term_coverage"] == 0.0


def test_build_review_markdown():
    """L3 人工抽查清单：每模块抽样条数正确。"""
    from industrial_ft.utils.metrics import build_review_markdown

    rows = [
        {"prompt": "q", "reference": "r", "prediction": "p", "source": "law"},
        {"prompt": "q", "reference": "r", "prediction": "p", "source": "law"},
        {"prompt": "q", "reference": "r", "prediction": "p", "source": "safety"},
        {"prompt": "q", "reference": "r", "prediction": "p", "source": "safety"},
    ]
    md = build_review_markdown(rows, per_module_n=1, title="TEST")
    assert "TEST" in md
    assert md.count("| law |") == 1 and md.count("| safety |") == 1


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
