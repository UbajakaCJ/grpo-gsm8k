CONFIG ?= configs/colab_a100.yaml
PY     ?= python
# Pull the directories out of the (possibly inherited) config so targets stay in sync.
OUTPUT_DIR  = $(shell $(PY) -c "from grpo_gsm8k.config import load_config as l; print(l('$(CONFIG)')['output_dir'])")

.PHONY: help install install-dev test lint eval-base train eval-trained report all smoke clean

help:
	@echo "make install       install training deps (GPU box / Colab)"
	@echo "make test lint     unit tests + ruff (CPU only, what CI runs)"
	@echo "make all           eval-base -> train -> eval-trained -> report  (CONFIG=$(CONFIG))"
	@echo "make smoke         whole pipeline on a tiny model / 4 steps"

install:
	pip install -e ".[train,vllm,dev]"

install-dev:
	pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

eval-base:
	$(PY) -m grpo_gsm8k.evaluate --config $(CONFIG) --tag base

train:
	$(PY) -m grpo_gsm8k.train --config $(CONFIG)

eval-trained:
	$(PY) -m grpo_gsm8k.evaluate --config $(CONFIG) --tag trained --model $(OUTPUT_DIR)/final

report:
	$(PY) -m grpo_gsm8k.report --config $(CONFIG)

# Each stage is its own process so vLLM / training memory is fully released in between.
all: eval-base train eval-trained report

smoke:
	$(MAKE) all CONFIG=configs/smoke.yaml

clean:
	rm -rf outputs results/smoke .pytest_cache .ruff_cache
