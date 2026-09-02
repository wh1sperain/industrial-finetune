"""实验配置解析。

- configs/experiments/<代码级实验>/<name>.yaml：实验自包含配置（model ID、数据、超参、输出目录）
- configs/models.yaml：机器相关映射（模型 ID → 本地路径）

模型路径解析：实验的 model 字段 → 查映射表 → 本地目录存在用本地，否则用 ID 在线加载。
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(os.environ.get("INDUSTRIAL_FT_ROOT") or Path(__file__).resolve().parents[3])
# 当前代码级实验名；配置按代码级实验分目录存放。
_CODE_EXPERIMENT = os.environ.get("INDUSTRIAL_FT_CODE_EXPERIMENT", "industry_instruction")
EXPERIMENTS_DIR = Path(
    os.environ.get("INDUSTRIAL_FT_EXPERIMENTS_DIR")
    or REPO_ROOT / "configs" / "experiments" / _CODE_EXPERIMENT
)
MODELS_CONFIG = Path(
    os.environ.get("INDUSTRIAL_FT_MODELS_CONFIG") or REPO_ROOT / "configs" / "models.yaml"
)

_TOP_KEYS = ("model_name_or_path", "model_path", "output_dir", "run_name", "trust_remote_code")


def load_yaml(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def list_experiments() -> list[str]:
    return sorted(p.stem for p in EXPERIMENTS_DIR.glob("*.yaml"))


def load_models_config() -> dict:
    return load_yaml(MODELS_CONFIG)


def resolve_model_name(model_id: str) -> str:
    """模型 ID → 本地路径（映射表 + 目录存在检查）；无映射/目录不存在则返回 ID。"""
    model_id = (model_id or "").strip()
    if not model_id:
        return ""
    local_path = (load_models_config().get("local_paths", {}) or {}).get(model_id, "")
    if local_path and os.path.isdir(local_path):
        return local_path
    return model_id


def load_experiment(name: str) -> dict:
    """加载实验配置：解析模型字段，返回 {"train": {...}, "eval": {...}, 顶层字段...}。"""
    path = EXPERIMENTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"实验不存在: {name}（可用: {list_experiments()}）")
    exp = load_yaml(path)
    model_id = (exp.get("model") or "").strip()
    exp["model_name_or_path"] = resolve_model_name(model_id)
    exp["model_path"] = (load_models_config().get("local_paths", {}) or {}).get(model_id, "")
    exp.setdefault("output_dir", f"output/{name}")
    exp.setdefault("run_name", name)
    exp.setdefault("trust_remote_code", False)
    exp.setdefault("train", {})
    exp.setdefault("eval", {})
    return exp


def load_train_config(name: str) -> dict:
    """训练视角：train 小节 + 顶层模型/输出字段。"""
    exp = load_experiment(name)
    cfg = dict(exp["train"])
    for k in _TOP_KEYS:
        cfg[k] = exp[k]
    return cfg


def load_eval_config(name: str) -> dict:
    """评测视角：eval 小节 + 顶层字段（adapter 目录 = 实验输出目录）。"""
    exp = load_experiment(name)
    cfg = dict(exp["eval"])
    for k in _TOP_KEYS:
        cfg[k] = exp[k]
    cfg["adapter_dir"] = exp["output_dir"]
    return cfg
