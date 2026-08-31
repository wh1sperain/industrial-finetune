"""训练模块：sft（TRL SFT 入口）。

注意：不在 __init__.py 提前 import 子模块——sft 依赖 torch/trl，
提前导入会产生 runpy RuntimeWarning 并拖慢无关环境。
"""
