# 数据目录

本目录**不存放数据本身**，只登记数据清单与获取方式（数据一律外部化，见 `docs/数据管理.md`）。

| 数据集 | 用途 | 来源 / revision | 获取方式 | 状态 |
|---|---|---|---|---|
| gsm8k（占位） | 流程冒烟，M1 替换为实际工业数据集 | HF `openai/gsm8k` @ main | `make download` | 待下载 |

## 本地布局约定

- 公开数据：HF 缓存（`~/.cache/huggingface`），脚本化下载
- 私有数据：挂载到 `$DATA_ROOT`（`.env` 配置），默认 `./data`，**禁止 commit**
