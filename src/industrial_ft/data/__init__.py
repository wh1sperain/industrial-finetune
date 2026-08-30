"""数据加载与预处理。

原则：数据本身不入库（见 docs/数据管理.md）。
- 公开数据集：脚本化下载 + 固定 revision，保证可复现。
- 企业私有数据：从 $DATA_ROOT 读取，路径由 config / .env 注入。
"""
from industrial_ft.data.loader import download, load_train_dataset

__all__ = ["download", "load_train_dataset"]
