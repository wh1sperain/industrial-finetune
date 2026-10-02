"""Configuration for the independent industrial_text experiment."""
from industrial_ft.utils.config import REPO_ROOT, load_yaml, resolve_model_name


def load_config(name):
    if not name or any(c in name for c in "/\\."):
        raise ValueError("Use an experiment name, not a path")
    cfg = load_yaml(REPO_ROOT / "configs" / "experiments" / "industrial_text" / f"{name}.yaml")
    cfg["model_name_or_path"] = resolve_model_name(cfg["model"])
    if cfg["generation"]["do_sample"] or cfg["generation"]["enable_thinking"]:
        raise ValueError("This runner implements the deterministic non-thinking baseline only")
    if cfg["model_settings"]["attn_implementation"] != "sdpa":
        raise ValueError("The RTX 5090 experiment requires SDPA")
    return cfg
