# Industrial-Finetune

高校 × 企业联合的 **7B 开源模型工业数据集微调**仓库（私有）。

- 框架：TRL (HuggingFace) ｜ 方法：LoRA / QLoRA ｜ 基座：Qwen2.5-7B（可替换）
- 阶段：M0 骨架 → M1 用公开工业数据集跑通全流程

## 快速开始（5 分钟）

```bash
# 1. 克隆并安装
git clone <repo-url> && cd industrial-finetune
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. 配置密钥
cp .env.example .env    # 填入 HF_TOKEN / WANDB_API_KEY

# 3. 拉取公开数据集（当前为 GSM8K 占位，替换为实际工业数据集）
make download

# 4. LoRA 微调（单卡 24GB 可跑 7B QLoRA）
make train

# 5. 评测
make eval
```

> Windows 用户：`make` 命令请在 Git Bash 中执行；或直接运行 `bash scripts/train.sh`。

## 目录结构

```
├── .github/            # CI + Issue/PR 模板
├── src/industrial_ft/  # 包化代码（data 数据加载 / train 训练 / eval 评测 / utils 工具）
├── configs/            # 训练与评测配置（yaml，参数不进代码）
├── scripts/            # 统一入口脚本（train / eval / download）
├── tests/              # 冒烟测试（CI 里跑的轻量校验）
├── data/               # 只放数据清单说明，数据本身外部化
├── docs/               # 环境搭建 / 数据管理 / 实验规范 / 协作约定
└── Makefile            # setup / download / train / eval / test / lint
```

## 文档入口

| 文档 | 内容 |
|---|---|
| `docs/环境搭建.md` | 依赖、CUDA、硬件要求 |
| `docs/数据管理.md` | 数据来源、版本、私有数据接入规范 |
| `docs/实验规范.md` | 实验命名、wandb、结果沉淀 |
| `docs/协作约定.md` | 分支、PR 评审、分工 |

## 核心约定（详见 CONTRIBUTING.md）

- `main` 受保护，所有改动走 **PR + 至少 1 人交叉评审**（高校方审方法，企业方审数据口径）
- 数据、权重、日志 **不入库**；模型 LoRA adapter 上传 HF Hub / 企业网盘
- 一切训练参数走 `configs/`，代码里不硬编码
