.PHONY: help install synth-smoke tokenizer-smoke pretrain-smoke agent-sft-smoke eval-smoke poc test clean

PYTHON = ./.venv/bin/python
PYTEST = ./.venv/bin/pytest

help:
	@echo "Large Code Model (LCM) — Synthetic-Only Agentic Language Model POC"
	@echo "Targets:"
	@echo "  make synth-smoke       Generate synthetic smoke dataset"
	@echo "  make tokenizer-smoke   Train BPE tokenizer on smoke corpus"
	@echo "  make pretrain-smoke    Pretrain base transformer on smoke corpus"
	@echo "  make agent-sft-smoke   Supervised fine-tuning on agent trajectories"
	@echo "  make eval-smoke        Run deterministic benchmark suite"
	@echo "  make poc               Run full end-to-end POC pipeline"
	@echo "  make test              Run pytest test suite"

install:
	$(PYTHON) -m pip install -e .

synth-smoke:
	$(PYTHON) -m synth.generate --config configs/smoke.yaml

tokenizer-smoke:
	$(PYTHON) -m training.tokenizer --config configs/smoke.yaml

pretrain-smoke:
	$(PYTHON) -m training.pretrain --config configs/smoke.yaml

agent-sft-smoke:
	$(PYTHON) -m training.agent_sft --config configs/smoke.yaml

eval-smoke:
	$(PYTHON) -m eval.runner --config configs/smoke.yaml

poc: synth-smoke tokenizer-smoke pretrain-smoke agent-sft-smoke eval-smoke
	@echo "[+] End-to-end POC execution complete. View report at runs/smoke/eval/report.html"

test:
	$(PYTEST) tests/ -v

clean:
	rm -rf runs/ .pytest_cache/
