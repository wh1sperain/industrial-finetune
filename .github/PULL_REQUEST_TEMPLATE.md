## 改了什么

<!-- 简述本次变更内容 -->

## 为什么

<!-- 背景与动机，关联 Issue 编号（如 #12） -->

## 实验影响

<!-- 若涉及实验：
- 改了哪些配置（configs/ 下 diff）
- 跑了什么、结果如何（附 wandb 链接或结果表）
- 数据/指标口径是否有变化 -->

## 复现方式

```bash
pip install -e ".[dev]"
bash scripts/train.sh [config]
```

## 检查清单

- [ ] `make lint` 通过
- [ ] `make test` 通过
- [ ] 无数据/权重/日志/`.env` 入库（`git status` 确认）
- [ ] 无硬编码路径、密钥；日志不打印原始数据
- [ ] 实验参数在 configs/ 中，代码无魔法数字


