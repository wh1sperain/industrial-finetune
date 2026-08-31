# industrial-finetune

工业数据集模型微调

## 快速开始

```bash
git clone <repo-url> && cd industrial-finetune
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env          # 填写密钥
source scripts/env.sh         # 环境变量与自检

bash scripts/experiments/industrial_ft/train.sh mining_eng_lora        # 训练
bash scripts/experiments/industrial_ft/eval.sh mining_eng_lora --probe # 评测（Probe）
```

## 模型与数据

在配置中指定位置：

- **模型**：在 `configs/models.yaml` 的 `local_paths` 填入 `模型ID: 本地路径`（未填则按模型 ID 在线加载）
- **数据集**：在配置 `configs/experiments/<实验名>/<配置名>.yaml` 的 `dataset` 字段填 HF 数据集 ID 或本地目录路径

## 目录结构

```
├── configs/
│   ├── experiments/
│   │   └── industrial_ft/  # 代码级实验 industrial_ft 的配置（一个配置一个 yaml）
│   └── models.yaml         # 模型本地路径映射
├── src/industrial_ft/      # 实验代码（data / train / eval / utils）
├── scripts/
│   ├── common.sh           # 公共环境/路径/Python 解析
│   ├── run.sh              # 兼容入口
│   └── experiments/
│       └── industrial_ft/  # 实验入口：train.sh / eval.sh
├── tests/                  # 测试
└── .github/                # CI（待定）
```

## 实验管理

代码级实验（拥有自己训练/评测代码的实验）拥有自己的入口脚本：

```bash
# industrial_ft 代码级实验
bash scripts/experiments/industrial_ft/train.sh <配置名> [--resume]
bash scripts/experiments/industrial_ft/eval.sh <配置名> [--probe|--baseline|--semantic] [--review N]
```

- **新增配置** = 在 `configs/experiments/industrial_ft/` 新增一个 yaml（模型 ID、数据集、超参、输出目录全在文件内），无需改代码。
- **新增代码级实验** = 新建 `configs/experiments/<新实验>/` 配置目录 + `scripts/experiments/<新实验>/train.sh` 和 `eval.sh`，脚本内部调用该实验自己的训练/评测模块。
