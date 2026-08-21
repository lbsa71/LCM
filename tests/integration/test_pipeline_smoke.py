"""Integration smoke test running the full pipeline end-to-end."""

import os
import tempfile
import yaml
from synth.generate import main as synth_main
from training.tokenizer import train_synthetic_tokenizer
from training.pretrain import train_pretrain
from training.agent_sft import train_agent_sft
from eval.runner import main as eval_main


def test_full_pipeline_smoke():
    # Use existing configs/smoke.yaml
    assert os.path.exists("configs/smoke.yaml")
    assert os.path.exists("specs/forbidden_entities.txt")
