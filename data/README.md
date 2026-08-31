# 数据清单

数据本身不入库（外部化，HF 缓存），此处仅登记。

| 数据集 | 用途 | 规模 | 获取 |
|---|---|---|---|
| Mining-Engineering-SFT | 训练集 | 5,287 | `bash scripts/download_data.sh sft` |
| Mining-Engineering-SFT-CoT | CoT 对比 | 5,287 | `bash scripts/download_data.sh cot` |
| Mining-Engineering-Eval | 正式评测 | 301 | `bash scripts/download_data.sh eval` |
| Mining-Engineering-Probe | 快速验证 | 50 | `bash scripts/download_data.sh probe` |

字段：`prompt`（问题）/ `response`（回答）/ `source`（六模块：law/specifications/concept/safety/forum/synthesis）。
划分直接用官方（训练/Eval/Probe 独立），无需自行切分。
