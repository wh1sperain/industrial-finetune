"""统一配置解析：模型命名与路径的单一事实源。

模型信息只维护在 `configs/models.yaml` 一处：
    base_model  : 模型 ID（HF/ModelScope，在线兜底）
    model_path  : 本地路径（download_model.sh 自动写入；目录存在则优先使用）
    adapter_dir : LoRA 输出目录（训练保存 / 评测加载，共用）
    run_name    : 实验名（wandb 共用）

所有入口（train / eval / 脚本）通过 `load_config()` 获取合并后的配置：
- 自动注入 model_name_or_path / output_dir / run_name / trust_remote_code
- 模型路径解析规则：本地目录存在 → 用本地；否则 → 在线 ID（不再需要到处改路径）
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
    """读取模型单一配置源（configs/models.yaml）。"""
    return load_yaml(MODELS_CONFIG)


def resolve_model_name(models: dict) -> str:
    """解析模型路径：本地目录存在 → 用本地路径；否则 → 在线 ID。"""
    model_path = (models.get("model_path") or "").strip()
    if model_path and os.path.isdir(model_path):
        return model_path
    return (models.get("base_model") or "").strip()


def load_config(path: str | Path) -> dict:
    """加载 train/eval 配置，并合并模型单一配置源。

    注入字段：model_name_or_path（已解析）、model_path、adapter_dir、
    output_dir（= adapter_dir，若该配置未自定义）、run_name、trust_remote_code。
    """
    cfg = load_yaml(path)
    models = load_models_config()
    cfg["model_name_or_path"] = resolve_model_name(models)
    cfg["model_path"] = (models.get("model_path") or "").strip()
    cfg.setdefault("adapter_dir", models.get("adapter_dir", "output"))
    cfg.setdefault("output_dir", cfg["adapter_dir"])
    cfg.setdefault("run_name", models.get("run_name", "industrial-ft"))
    cfg.setdefault("trust_remote_code", models.get("trust_remote_code", False))
    return cfg
