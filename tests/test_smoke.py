"""冒烟测试（不加载模型、不真训练）。

仅校验配置解析与纯逻辑，完整训练/评测在目标机器上跑。
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXP = "mining_eng_lora"


def test_train_config_loads():
    from industrial_ft.utils.config import load_train_config

    cfg = load_train_config(EXP)
    assert "Qwen" in cfg["model_name_or_path"]
    assert "output_dir" in cfg and cfg["output_dir"]
    assert "run_name" in cfg and cfg["run_name"]
    assert cfg["lora_r"] > 0
    assert "Mining-Engineering" in cfg["dataset"]
    assert cfg.get("torch_compile") is True
    assert cfg.get("attn_implementation") == "sdpa"
    assert cfg.get("optim") == "adamw_torch_fused"


def test_eval_config_loads():
    from industrial_ft.utils.config import load_eval_config

    cfg = load_eval_config(EXP)
    assert "Qwen" in cfg["model_name_or_path"]
    assert cfg.get("adapter_dir") == cfg["output_dir"]
    assert "Mining-Engineering-Eval" in cfg["eval_dataset"]
    assert "Mining-Engineering-Probe" in cfg["probe_dataset"]


def test_models_config_local_paths():
    from industrial_ft.utils.config import load_models_config

    models = load_models_config()
    # local_paths 默认留空（用户自填）；解析逻辑：无映射/目录不存在 → 回退模型 ID
    from industrial_ft.utils.config import resolve_model_name

    assert resolve_model_name("Qwen/Qwen2.5-7B-Instruct") == "Qwen/Qwen2.5-7B-Instruct"


def test_loader_supports_local_and_hf():
    """loader 的 dataset 参数支持本地路径与 HF ID（不触发真实加载，仅检查格式函数）。"""
    from industrial_ft.data.loader import format_sample

    out = format_sample({"prompt": "问题", "response": "答案", "source": "law"})
    assert out == {"prompt": "问：问题", "completion": "答：答案"}


def test_list_experiments():
    from industrial_ft.utils.config import list_experiments

    assert EXP in list_experiments()


def test_unknown_experiment_raises():
    from industrial_ft.utils.config import load_experiment

    try:
        load_experiment("no_such_experiment")
        assert False, "应抛出 FileNotFoundError"
    except FileNotFoundError:
        pass


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
