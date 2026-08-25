"""Unified Model and Tokenizer Loader supporting native SyntheticTransformer and HuggingFace base models (e.g., SmolLM2)."""

import os
import json
import torch
from typing import Tuple, Any, Dict, Optional, List


class HuggingFaceTokenizerAdapter:
    """Adapts a HuggingFace PreTrainedTokenizer to the LCM Tokenizer interface."""

    def __init__(self, hf_tokenizer):
        self.hf_tokenizer = hf_tokenizer

    def token_to_id(self, token: str) -> Optional[int]:
        t_id = self.hf_tokenizer.convert_tokens_to_ids(token)
        if t_id == self.hf_tokenizer.unk_token_id and token != self.hf_tokenizer.unk_token:
            return None
        return t_id

    def id_to_token(self, token_id: int) -> Optional[str]:
        return self.hf_tokenizer.convert_ids_to_tokens(token_id)

    def encode(self, text: str):
        class EncodedOutput:
            def __init__(self, ids):
                self.ids = ids
        
        # Don't add special tokens in raw encode to preserve manual RDL role tags
        ids = self.hf_tokenizer.encode(text, add_special_tokens=False)
        return EncodedOutput(ids)

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return self.hf_tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def get_vocab_size(self) -> int:
        return len(self.hf_tokenizer)

    def __len__(self) -> int:
        return len(self.hf_tokenizer)


def load_model_and_tokenizer(config: Dict[str, Any], device: str = "cpu", checkpoint_path: Optional[str] = None) -> Tuple[Any, Any]:
    """Loads a model and tokenizer according to the specified backend configuration."""
    backend = config.get("backend", "custom")

    if backend == "huggingface":
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = config.get("pretrained_model_name", "HuggingFaceTB/SmolLM2-135M")
        tok_source = checkpoint_path if (checkpoint_path and os.path.exists(os.path.join(checkpoint_path, "tokenizer_config.json"))) else model_name
        hf_tokenizer = AutoTokenizer.from_pretrained(tok_source)

        special_tokens = config.get("special_tokens", [
            "<PAD>", "<BOS>", "<EOS>", "<UNK>", "<USER>", "<ASSISTANT>", "<TOOL>",
            "<OBSERVATION>", "<PLAN>", "<ACTION>", "<FINAL>"
        ])

        # Add custom special tokens
        tokens_to_add = [t for t in special_tokens if t not in hf_tokenizer.get_vocab()]
        if tokens_to_add:
            hf_tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

        if hf_tokenizer.pad_token is None:
            if "<PAD>" in hf_tokenizer.get_vocab():
                hf_tokenizer.pad_token = "<PAD>"
            else:
                hf_tokenizer.pad_token = hf_tokenizer.eos_token

        tokenizer = HuggingFaceTokenizerAdapter(hf_tokenizer)

        if checkpoint_path and os.path.exists(checkpoint_path):
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)

        if model.get_input_embeddings().weight.shape[0] != len(hf_tokenizer):
            model.resize_token_embeddings(len(hf_tokenizer))

        model.to(device)
        return model, tokenizer

    else:
        from training.model import SyntheticTransformer, TransformerConfig
        from tokenizers import Tokenizer

        tok_path = config.get("tokenizer_path", "tokenizer/tokenizer.json")
        tokenizer = Tokenizer.from_file(tok_path)

        m_cfg = config.get("model", {})
        trans_config = TransformerConfig(
            vocab_size=m_cfg.get("vocab_size", 8192),
            hidden_size=m_cfg.get("hidden_size", 1024),
            num_hidden_layers=m_cfg.get("num_hidden_layers", 18),
            num_attention_heads=m_cfg.get("num_attention_heads", 16),
            intermediate_size=m_cfg.get("intermediate_size", 4096),
            max_position_embeddings=m_cfg.get("max_position_embeddings", 2048),
            rms_norm_eps=m_cfg.get("rms_norm_eps", 1.0e-5),
            rope_theta=m_cfg.get("rope_theta", 10000.0),
            tie_word_embeddings=m_cfg.get("tie_word_embeddings", True)
        )
        model = SyntheticTransformer(trans_config)
        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.save if not checkpoint_path.endswith(".pt") else torch.load
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        return model, tokenizer
