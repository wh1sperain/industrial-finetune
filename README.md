# industrial-finetune

工业数据集模型微调（TRL + LoRA，Qwen2.5-7B）。

## 快速开始

```bash
git clone <repo-url> && cd industrial-finetune
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env          # 填写密钥（HF_TOKEN / WANDB_API_KEY 按需）
source scripts/env.sh         # 环境变量与自检

bash scripts/download_model.sh   # 下载模型（ModelScope，写入 configs/models.yaml）
bash scripts/download_data.sh    # 下载数据集（sft/cot/eval/probe）

bash scripts/train.sh configs/train/lora_qwen7b_industrial.yaml   # 训练
bash scripts/eval.sh --probe                                     # 评测（Probe）
```

## 目录结构

```
├── configs/            # 配置（models.yaml 为模型路径/实验名单一来源，train/ eval/ 为实验配置）
├── src/industrial_ft/  # 代码（data 数据加载 / train 训练 / eval 评测 / utils 工具）
├── scripts/            # 入口脚本（download_model / download_data / train / eval / env）
├── tests/              # 冒烟测试
├── data/               # 数据集清单（数据本身不入库）
├── .github/            # CI
└── Makefile            # setup / download / train / eval / test / lint
```

## 实验扩展方式

新实验 = 新增一个 `configs/train/*.yaml`（数据/超参/输出目录均可覆盖），训练与评测入口不变。
