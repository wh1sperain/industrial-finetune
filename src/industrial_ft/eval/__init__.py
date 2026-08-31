"""评测模块：run_eval（模型前向评测）/ score_predictions（预测文件评分）。

注意：不在 __init__.py 提前 import 子模块——
run_eval 依赖 torch，提前导入会拖慢/破坏无 torch 环境（如 CI、score_predictions）。
"""
