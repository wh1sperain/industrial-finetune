"""冒烟测试（不加载模型、不真训练）。

仅校验配置解析与纯逻辑，完整训练/评测在目标机器上跑。
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_train_config_loads():
    from industrial_ft.utils.config import load_config

    cfg = load_config(ROOT / "configs" / "train" / "lora_qwen7b_industrial.yaml")
    assert "Qwen" in cfg["model_name_or_path"]
    assert "output_dir" in cfg and cfg["output_dir"]
    assert "run_name" in cfg and cfg["run_name"]
    assert cfg["lora_r"] > 0
    assert "Mining-Engineering" in cfg["dataset"]
    assert cfg.get("torch_compile") is True
    assert cfg.get("attn_implementation") == "sdpa"
    assert cfg.get("optim") == "adamw_torch_fused"


def test_eval_config_loads():
    from industrial_ft.utils.config import load_config

    cfg = load_config(ROOT / "configs" / "eval" / "industrial_benchmark.yaml")
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
    assert "model_path" in models


def test_metrics_term_coverage():
    from industrial_ft.utils.metrics import compute_metrics, extract_key_terms

    terms = extract_key_terms("\u201c欠挖\u201d与\u201c超挖\u201d的区别，参考《GBT 50562-2019 标准》和 CAPEX 指标")
    assert "欠挖" in terms and "超挖" in terms
    assert any("GBT 50562" in t for t in terms)
    assert "CAPEX" in terms

    refs = ["\u201c凿井\u201d与\u201c巷道掘进\u201d的差异（CAPEX 影响）"]
    m = compute_metrics(["凿井和巷道掘进…… CAPEX 方面……"], refs, ["concept"])
    assert 0 < m["term_coverage"] <= 1.0
    m0 = compute_metrics(["完全无关的回答"], refs, ["concept"])
    assert m0["term_coverage"] == 0.0


def test_build_review_markdown():
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
