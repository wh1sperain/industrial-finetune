.PHONY: setup train eval test lint

setup:
	python -m pip install -e ".[dev]"

train:
	bash scripts/experiments/industrial_ft/train.sh mining_eng_lora

eval:
	bash scripts/experiments/industrial_ft/eval.sh mining_eng_lora

test:
	python -m pytest tests/ -q

lint:
	ruff check src/ tests/
