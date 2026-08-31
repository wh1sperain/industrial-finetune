"""全局随机种子。torch/numpy 惰性导入（无 torch 环境也可用）。"""


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
