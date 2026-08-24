"""Unit tests for SmolLM2 HuggingFace Model Loader & Tokenizer Adapter."""

import pytest
import torch
from training.model_loader import load_model_and_tokenizer, HuggingFaceTokenizerAdapter


def test_smollm_tokenizer_adapter():
    config = {
        "backend": "huggingface",
        "pretrained_model_name": "HuggingFaceTB/SmolLM2-135M",
        "special_tokens": ["<USER>", "<PLAN>", "<ACTION>", "<OBSERVATION>", "<FINAL>"]
    }

    # Load on CPU
    model, tokenizer = load_model_and_tokenizer(config, device="cpu")

    # Verify adapter methods
    assert tokenizer.token_to_id("<USER>") is not None
    assert tokenizer.token_to_id("<PLAN>") is not None
    assert tokenizer.token_to_id("<ACTION>") is not None
    assert tokenizer.token_to_id("<OBSERVATION>") is not None
    assert tokenizer.token_to_id("<FINAL>") is not None

    # Test encoding
    encoded = tokenizer.encode("What is 347 + 687?")
    assert hasattr(encoded, "ids")
    assert len(encoded.ids) > 0

    # Test decoding
    decoded = tokenizer.decode(encoded.ids)
    assert "What is 347 + 687?" in decoded


def test_smollm_model_cpu_forward():
    config = {
        "backend": "huggingface",
        "pretrained_model_name": "HuggingFaceTB/SmolLM2-135M",
        "special_tokens": ["<USER>", "<PLAN>", "<ACTION>", "<OBSERVATION>", "<FINAL>"]
    }

    model, tokenizer = load_model_and_tokenizer(config, device="cpu")
    model.eval()

    user_tag_id = tokenizer.token_to_id("<USER>")
    content_ids = tokenizer.encode("what is 347 + 687").ids
    input_ids = torch.tensor([[user_tag_id] + content_ids], dtype=torch.long)
    labels = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        assert outputs.loss is not None
        assert outputs.loss.item() > 0.0
