"""统一配置解析。

模型相关信息只维护在 `configs/models.yaml`：
- model_path 为本地路径（download_model.sh 写入），目录存在时优先使用，否则回退 base_model
- adapter_dir / run_name 由训练与评测共用

`load_config()` 加载 train/eval 配置并合并以上字段。
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_CONFIG = REPO_ROOT / "configs" / "models.yaml"


def load_yaml(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_models_config() -> dict:
    return load_yaml(MODELS_CONFIG)


def resolve_model_name(models: dict) -> str:
    model_path = (models.get("model_path") or "").strip()
    if model_path and os.path.isdir(model_path):
        return model_path
    return (models.get("base_model") or "").strip()


def load_config(path: str | Path) -> dict:
    """加载 train/eval 配置并注入模型字段（model_name_or_path / output_dir / run_name / trust_remote_code）。"""
    cfg = load_yaml(path)
    models = load_models_config()
    cfg["model_name_or_path"] = resolve_model_name(models)
    cfg["model_path"] = (models.get("model_path") or "").strip()
    cfg.setdefault("adapter_dir", models.get("adapter_dir", "output"))
    cfg.setdefault("output_dir", cfg["adapter_dir"])
    cfg.setdefault("run_name", models.get("run_name", "industrial-ft"))
    cfg.setdefault("trust_remote_code", models.get("trust_remote_code", False))
    return cfg
