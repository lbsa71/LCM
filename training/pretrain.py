"""Causal next-token pretraining pipeline for SyntheticTransformer."""

import argparse
import json
import math
import os
import time
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from training.model import SyntheticTransformer, TransformerConfig


class TextDataset(Dataset):
    """Tokenizes and packs lines into fixed-length chunks."""

    def __init__(self, text_file: str, tokenizer: Tokenizer, seq_len: int = 512):
        self.seq_len = seq_len
        bos_id = tokenizer.token_to_id("<BOS>")
        eos_id = tokenizer.token_to_id("<EOS>")

        all_ids = []
        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                encoded = tokenizer.encode(line).ids
                all_ids.extend([bos_id] + encoded + [eos_id])

        # Slice into chunks of seq_len
        num_chunks = len(all_ids) // seq_len
        if num_chunks == 0 and len(all_ids) > 0:
            # Pad to seq_len
            pad_id = tokenizer.token_to_id("<PAD>")
            all_ids.extend([pad_id] * (seq_len - len(all_ids)))
            num_chunks = 1

        self.samples = []
        for i in range(num_chunks):
            chunk = all_ids[i * seq_len: (i + 1) * seq_len]
            self.samples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        return chunk, chunk  # input_ids, labels


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int, lr: float, min_lr: float):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr / lr + (1.0 - min_lr / lr) * cosine_decay
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_pretrain(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    preset_name = config.get("name", "smoke")
    run_dir = f"runs/{preset_name}"
    data_dir = os.path.join(run_dir, "data")
    tok_dir = os.path.join(run_dir, "tokenizer")
    base_model_dir = os.path.join(run_dir, "base_model")
    os.makedirs(base_model_dir, exist_ok=True)

    # Load Tokenizer
    tok_path = os.path.join(tok_dir, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)
    vocab_size = tokenizer.get_vocab_size()

    # Device
    device_name = config.get("pretrain", {}).get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"[*] Pretraining on device: {device}")

    # Model Config
    m_cfg = config.get("model", {})
    trans_config = TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=m_cfg.get("hidden_size", 128),
        num_hidden_layers=m_cfg.get("num_hidden_layers", 4),
        num_attention_heads=m_cfg.get("num_attention_heads", 4),
        intermediate_size=m_cfg.get("intermediate_size", 512),
        max_position_embeddings=m_cfg.get("max_position_embeddings", 512),
        rms_norm_eps=float(m_cfg.get("rms_norm_eps", 1e-5)),
        rope_theta=float(m_cfg.get("rope_theta", 10000.0)),
        tie_word_embeddings=m_cfg.get("tie_word_embeddings", True),
        pad_token_id=tokenizer.token_to_id("<PAD>"),
        bos_token_id=tokenizer.token_to_id("<BOS>"),
        eos_token_id=tokenizer.token_to_id("<EOS>")
    )

    model = SyntheticTransformer(trans_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Model initialized with {total_params:,} parameters.")

    # Datasets
    seq_len = trans_config.max_position_embeddings
    train_ds = TextDataset(os.path.join(data_dir, "pretrain_train.txt"), tokenizer, seq_len=seq_len)
    val_ds = TextDataset(os.path.join(data_dir, "pretrain_val.txt"), tokenizer, seq_len=seq_len)

    p_cfg = config.get("pretrain", {})
    batch_size = p_cfg.get("batch_size", 8)
    grad_accum = p_cfg.get("gradient_accumulation_steps", 2)
    max_steps = p_cfg.get("max_steps", 300)
    lr = float(p_cfg.get("learning_rate", 1e-3))
    min_lr = float(p_cfg.get("min_learning_rate", 1e-4))
    warmup_steps = p_cfg.get("warmup_steps", 30)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if len(val_ds) > 0 else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=p_cfg.get("weight_decay", 0.01))
    scheduler = get_lr_scheduler(optimizer, warmup_steps, max_steps, lr, min_lr)

    step = 0
    start_time = time.time()
    model.train()
    optimizer.zero_grad()

    while step < max_steps:
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            _, loss = model(batch_x, labels=batch_y)
            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1
            if step % 50 == 0 or step == max_steps:
                elapsed = time.time() - start_time
                current_lr = scheduler.get_last_lr()[0]
                print(f"Step {step}/{max_steps} | Loss: {loss.item() * grad_accum:.4f} | LR: {current_lr:.6f} | Elapsed: {elapsed:.1f}s")

            if step >= max_steps:
                break

    # Save final model & config
    final_ckpt_path = os.path.join(base_model_dir, "base_final.pt")
    torch.save(model.state_dict(), final_ckpt_path)
    
    cfg_path = os.path.join(base_model_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(trans_config.__dict__, f, indent=2)

    print(f"[+] Pretraining completed. Model saved to {final_ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Pretrain Synthetic Transformer")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml", help="Path to config yaml")
    args = parser.parse_args()
    train_pretrain(args.config)


if __name__ == "__main__":
    main()
