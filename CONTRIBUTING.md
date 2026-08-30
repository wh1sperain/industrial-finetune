# Contributing 规范

本仓库服务于高校 × 企业联合微调项目。规范的目的：**让每一条变更都可追溯、可复现、可评审**。

## 1. 分支模型（GitHub Flow）

```
main（受保护，永远可跑）
  └─ feature/xxx  ← 开发分支 → PR → 评审 → squash merge
```

- 每个任务/实验一个分支：`feature/sft-lora-v1`、`fix/eval-metric`、`config/qlora-4bit`
- **禁止直接 push main**（仓库已开启分支保护）。

## 2. 提交规范（Conventional Commits）

```
feat:    新功能（如新增评测指标）
fix:     修复 bug
config:  修改训练/评测配置
data:    数据脚本或数据清单变更
docs:    文档变更
refactor:重构（行为不变）
ci:      CI 相关变更
```

- 一条 commit 只做一件事；实验改配置与改代码**分开提交**。
- 示例：`config: gsm8k 换为工业数据集 demo 数据`

## 3. Pull Request 流程

1. 从 `main` 切分支 → 开发 → 本地跑 `make test` 和 `make lint`。
2. 创建 PR，按模板填写：改了什么 / 为什么 / 实验影响 / 复现方式。
3. **交叉评审**：高校方审方法正确性，企业方审数据口径与工业指标。
4. squash merge 回 `main`，删除已合并分支。

## 4. Issue 使用

- Bug 报告、需求、实验任务请用仓库的 Issue 模板（`.github/ISSUE_TEMPLATE/`）。
- 实验任务类 Issue 建议挂 milestone（M0/M1/M2），任务完成以 PR 合入为标志。

## 5. 红线（评审必查）

- [ ] 没有把数据、权重、日志、`.env` 提交进仓库
- [ ] 没有硬编码路径 / 密钥 / Token
- [ ] 日志不打印原始数据（脱敏检查）
- [ ] 训练参数在 `configs/` 里，代码无魔法数字
- [ ] 本地 `make test`、`make lint` 通过
