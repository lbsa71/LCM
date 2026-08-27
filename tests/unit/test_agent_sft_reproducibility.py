import torch
import json
import pytest

from training.agent_sft import TrajectoryDataset, seed_training_randomness, summarize_sft_exposure


def test_sft_seed_controls_torch_sampling():
    seed_training_randomness(42)
    first = torch.rand(4)
    seed_training_randomness(42)

    assert torch.equal(first, torch.rand(4))


def test_sft_exposure_distinguishes_microsteps_from_optimizer_updates():
    exposure = summarize_sft_exposure(
        microsteps=3_000,
        batch_size=4,
        gradient_accumulation_steps=8,
        sequence_length=1_024,
    )

    assert exposure == {
        "microsteps": 3_000,
        "optimizer_updates": 375,
        "sequences_processed": 12_000,
        "tokens_processed": 12_288_000,
        "effective_batch_size": 32,
    }


def test_sft_rejects_unfinished_demonstrations_before_tokenizing(tmp_path):
    path = tmp_path / "incomplete.jsonl"
    path.write_text(json.dumps({"task_id": "broken", "turns": [
        {"role": "user", "content": "Query", "train": False},
        {"role": "action", "content": 'SEARCH "Query" LIMIT 3', "train": True},
    ]}), encoding="utf-8")

    class Tokenizer:
        def token_to_id(self, token):
            return 0

        def encode(self, text):
            pytest.fail("Incomplete trajectories must fail before tokenization")

    with pytest.raises(ValueError, match="terminal.*broken"):
        TrajectoryDataset(str(path), Tokenizer())


def test_evaluation_only_config_cannot_overwrite_training_artifacts(tmp_path, monkeypatch):
    import training.agent_sft as sft

    monkeypatch.chdir(tmp_path)
    config = tmp_path / "eval_only.yaml"
    config.write_text("evaluation_only: true\nbackend: huggingface\n", encoding="utf-8")
    monkeypatch.setattr(sft, "load_model_and_tokenizer", lambda *a, **k: pytest.fail("Must reject before model load"))
    with pytest.raises(ValueError, match="evaluation-only"):
        sft.train_agent_sft(str(config))
    assert not (tmp_path / "runs").exists()
