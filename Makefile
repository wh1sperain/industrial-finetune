.PHONY: setup download train eval test lint

# 统一入口（Windows 请在 Git Bash 中执行）
setup:
	python -m pip install -e ".[dev]"

download:
	bash scripts/download_data.sh

train:
	bash scripts/train.sh

eval:
	bash scripts/eval.sh

test:
	python -m pytest tests/ -q

lint:
	ruff check src/ tests/
