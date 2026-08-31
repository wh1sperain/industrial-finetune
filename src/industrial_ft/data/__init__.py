"""数据加载（loader 不依赖 torch，可独立使用）。"""
from industrial_ft.data.loader import download, load_train_dataset

__all__ = ["download", "load_train_dataset"]
