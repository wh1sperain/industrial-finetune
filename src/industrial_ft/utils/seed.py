"""全局随机种子（可复现性）。

torch / numpy 为惰性导入：CI 冒烟测试无 torch 也能运行。
"""
from __future__ import annotations


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
