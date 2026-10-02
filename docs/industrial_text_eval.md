# Qwen3-8B 工业纯文本评测

本实验只运行原始 `Qwen/Qwen3-8B`，不训练、不加载 LoRA，独立于 Mining-Engineering 实验。

## 环境和入口

使用支持 RTX 5090 / sm_120 的 PyTorch CUDA 构建。验证环境为 Python 3.12、PyTorch 2.12.1+cu130；其余推理依赖见 `requirements-text-eval.txt`。
已有兼容 PyTorch 时可创建 `python -m venv --system-site-packages <环境目录>` 后安装该依赖文件。
本入口通过 PYTHONPATH 使用源码，不需要为纯推理安装 TRL 等训练依赖。

配置：`configs/experiments/industrial_text/qwen3_8b_5090.yaml`。
模型通过既有 models.yaml 映射机制解析；`INDUSTRIAL_FT_MODELS_CONFIG` 可指向仓库外机器专用映射。
模型、数据和凭据不提交。下载模型是部署步骤，不加入仓库的下载脚本。

```bash
export INDUSTRIAL_FT_PYTHON=/root/autodl-tmp/envs/qwen3-eval/bin/python
export INDUSTRIAL_FT_MODELS_CONFIG=/root/autodl-tmp/models-local.yaml
bash scripts/experiments/industrial_text/eval.sh qwen3_8b_5090 --audit
bash scripts/experiments/industrial_text/eval.sh qwen3_8b_5090 --probe
bash scripts/experiments/industrial_text/eval.sh qwen3_8b_5090
bash scripts/experiments/industrial_text/eval.sh qwen3_8b_5090 --score-only
```

本地仅审计可添加 `--data-root data/benchmark数据集整理/纯文本`。审计与评分不导入 torch。train.sh 明确返回不支持训练。

## 数据和评分

- 递归读取 qa.json，忽略 evaluation.json 的历史模型输出。
- 稳定 ID 为领域/目录/文件名加记录索引；题干哈希标记完全重复和参考答案文本冲突。
- question_type 为认知类型；单独推断作答题型。无法确定参考答案的选择题不纳入客观题准确率，并列入审计。
- 仅题目进入模型。系统提示要求 answer 标签；参考答案、参考推理、引用和参考答案推断出的选项数量不进入提示。
- Qwen3 聊天模板，用户消息为 `问：{prompt}`；关闭 thinking，do_sample=False，BF16、SDPA。
- 选择题比较选项集合；判断题比较布尔值。解析失败、空回答、生成截断保留在分母，计为错误。
- 填空和开放题仅报告规范化 EM、术语覆盖、字符二元组召回、jieba TF-IDF 关键词覆盖。这些是词面诊断，不是语义正确率、数值容差评分或人工评分。
- 精确匹配保留数字符号、小数点和单位；复杂计算题留待人工按题意评分。不使用外部付费模型裁判。
- 输出原始、去重、去重排除参考冲突三种口径，以及领域/题型/认知类型/难度分组。
- 领域宏平均仅对有客观题的领域等权平均；无客观题时准确率为 null 而非 0。

参考答案未经专家独立校验；所有“正确”仅表示符合当前参考标签。

## 运行与恢复

先运行 50 题领域分层试跑，再运行全量。输出分别在 probe/ 和 full/，不混合。
预测逐题 flush/fsync；重复执行同一命令从已有 ID 继续。
恢复时校验数据哈希、配置、模型来源和源码哈希，防止混用实验。
仅自动修复末尾未写完的 JSON 行并保留备份；中间损坏或重复 ID 明确失败。
显存不足时减半批量重试；超过输入长度时失败，不静默截断题目。
意外中断保留已完成预测，未完成部分不算成功。

产物：predictions.jsonl、scored_predictions.jsonl、metrics.json、report.md、review.md、audit.json、manifest.json、environment.json、progress.json。
机器部署另保存 pip freeze、完整日志和模型文件 SHA256 清单。
--score-only 不执行前向，可保留生成 manifest 并记录新的评分源码版本。
后台运行使用 nohup 并保存 PID，断开 SSH 不影响进程。

## 降低截断的正式配置

`qwen3_8b_5090_long` 将初始输出上限提高至 8192 token。触及长度上限的题目单独以
16384、24576 token 重试，输入与输出总长度不得超过 32768。每次尝试写入 attempts.jsonl，
最终评分只使用最后一次输出，保留首次截断、重试次数、最终截断和 finish_reason。
默认重复惩罚为 1.05，仍采用非 thinking 贪心解码；这是相对于原始试跑的显式解码设置变更。
完整输出 `</answer>` 时结束生成；没有完整最终答案的输出不会伪装成已完成答案。
此配置在独立的 qwen3-8b-long 结果目录下运行，保留原始 2048-token 试跑。
有限长度不能保证模型永不截断；耗尽重试预算的题目会如实计入最终截断率与复核清单。

`qwen3_8b_5090_long_guarded` 在上述设置上增加 `no_repeat_ngram_size: 64`，防止原始试跑中
出现的数控程序长片段循环；它约束下一 token 的选择，不会把重复输出直接当作完整答案。
该约束也可能影响合法的长重复代码，属于明确记录的解码设置，报告仍需人工复核程序题。
结果独立保存到 qwen3-8b-long-guarded；不与原始或仅增大长度上限的试跑混合。

## 验证

新增测试覆盖答案解析歧义、数字选项、重复/冲突、失败样本分母、断点文件恢复。
既有 smoke 测试中数据加载一项需要 datasets；纯推理环境不以安装训练数据栈为前提。
