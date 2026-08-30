# 数据目录

本目录**不存放数据本身**，只登记数据清单与获取方式（数据一律外部化，见 `docs/数据管理.md`）。

## 数据集清单（M1 已定案：Mining-Engineering 矿建工程系列，中文，MIT）

| 数据集 | 用途 | 来源 / revision | 获取方式 | 状态 |
|---|---|---|---|---|
| Mining-Engineering-SFT | **正式训练集**（5,287 条，六模块：法规/规范/术语/安全/实践/综合） | HF `acnul/Mining-Engineering-SFT` @ main | `make download` | 可用 |
| Mining-Engineering-SFT-CoT | **可选对比**（同 5,287 条，response 带 `<think>` 思维链） | HF `acnul/Mining-Engineering-SFT-CoT` @ main | `make download`（改 dataset 名） | 可用 |
| Mining-Engineering-Eval | **官方评测集**（301 条） | HF `acnul/Mining-Engineering-Eval` @ main | `make eval` 时加载 | 可用 |
| Mining-Engineering-Probe | 快速验证集（50 条，调参期用） | HF `acnul/Mining-Engineering-Probe` @ main | 同上 | 可用 |

## 字段与规模说明

- 字段：`prompt`（问题）/ `response`（回答）/ `source`（六模块分类，用于分模块评测）
- 来源：合肥工业大学大创项目（README 归档见 `docs/数据参考_Mining-Engineering系列.md`）
- 数据划分：**直接用官方划分**（训练 5,287 / Eval 301 / Probe 50 独立），无需自行切分

## 本地布局约定

- 公开数据：HF 缓存（`~/.cache/huggingface`），脚本化下载
- 私有数据：挂载到 `$DATA_ROOT`（`.env` 配置），默认 `./data`，**禁止 commit**
