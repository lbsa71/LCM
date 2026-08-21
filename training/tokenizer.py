"""Trains a Byte-level BPE tokenizer from scratch exclusively on the synthetic corpus."""

import argparse
import json
import os
import yaml
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, processors


SPECIAL_TOKENS = [
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<UNK>",
    "<USER>",
    "<ASSISTANT>",
    "<TOOL>",
    "<OBSERVATION>",
    "<PLAN>",
    "<ACTION>",
    "<FINAL>"
]


def train_synthetic_tokenizer(
    corpus_file: str,
    output_dir: str,
    vocab_size: int = 4096,
    min_frequency: int = 2
) -> Tokenizer:
    """Trains a BPE tokenizer from scratch on a synthetic corpus file."""
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    print(f"[*] Training tokenizer on {corpus_file} (vocab_size={vocab_size})...")
    tokenizer.train([corpus_file], trainer)

    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)

    # Save special token mappings
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS},
        "pad_token_id": tokenizer.token_to_id("<PAD>"),
        "bos_token_id": tokenizer.token_to_id("<BOS>"),
        "eos_token_id": tokenizer.token_to_id("<EOS>"),
        "unk_token_id": tokenizer.token_to_id("<UNK>")
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[+] Tokenizer saved to {tokenizer_path} (actual vocab size: {tokenizer.get_vocab_size()})")
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train Synthetic Tokenizer")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml", help="Path to config yaml")
    parser.add_argument("--corpus_file", type=str, default=None, help="Corpus file path override")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory override")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    preset_name = config.get("name", "smoke")
    run_dir = f"runs/{preset_name}"
    corpus_file = args.corpus_file or os.path.join(run_dir, "data", "pretrain_train.txt")
    output_dir = args.output_dir or os.path.join(run_dir, "tokenizer")

    tok_cfg = config.get("tokenizer", {})
    vocab_size = tok_cfg.get("vocab_size", 4096)
    min_freq = tok_cfg.get("min_frequency", 2)

    train_synthetic_tokenizer(
        corpus_file=corpus_file,
        output_dir=output_dir,
        vocab_size=vocab_size,
        min_frequency=min_freq
    )


if __name__ == "__main__":
    main()
