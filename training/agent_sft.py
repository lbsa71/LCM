"""Supervised Fine-Tuning (SFT) pipeline for synthetic agent trajectories with loss masking."""

import argparse
import json
import os
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from training.model import SyntheticTransformer, TransformerConfig
from training.pretrain import get_lr_scheduler


class TrajectoryDataset(Dataset):
    """Encodes agent trajectories and applies target loss masking."""

    def __init__(self, jsonl_file: str, tokenizer: Tokenizer, max_len: int = 512):
        self.max_len = max_len
        self.samples = []

        pad_id = tokenizer.token_to_id("<PAD>")
        bos_id = tokenizer.token_to_id("<BOS>")
        eos_id = tokenizer.token_to_id("<EOS>")

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                turns = item.get("turns", [])
                
                input_ids = [bos_id]
                labels = [-100]  # BOS is not predicted

                for turn in turns:
                    role = turn["role"]
                    content = turn["content"]
                    is_train = turn.get("train", False)

                    # Role tags
                    tag = f"<{role.upper()}>"
                    tag_id = tokenizer.token_to_id(tag)
                    if tag_id is None:
                        tag_id = tokenizer.token_to_id("<UNK>")

                    content_ids = tokenizer.encode(content).ids
                    turn_ids = [tag_id] + content_ids

                    input_ids.extend(turn_ids)
                    if is_train:
                        # Model is trained to predict tag and content
                        labels.extend(turn_ids)
                    else:
                        # Mask out prompt and observation
                        labels.extend([-100] * len(turn_ids))

                input_ids.append(eos_id)
                labels.append(eos_id)

                # Truncate or pad to max_len
                if len(input_ids) > max_len:
                    input_ids = input_ids[:max_len]
                    labels = labels[:max_len]
                else:
                    pad_len = max_len - len(input_ids)
                    input_ids = input_ids + [pad_id] * pad_len
                    labels = labels + [-100] * pad_len

                self.samples.append((
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(labels, dtype=torch.long)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_agent_sft(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    preset_name = config.get("name", "smoke")
    run_dir = f"runs/{preset_name}"
    data_dir = os.path.join(run_dir, "data")
    tok_dir = os.path.join(run_dir, "tokenizer")
    base_model_dir = os.path.join(run_dir, "base_model")
    agent_model_dir = os.path.join(run_dir, "agent_model")
    os.makedirs(agent_model_dir, exist_ok=True)

    # Tokenizer
    tok_path = os.path.join(tok_dir, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)

    # Device & CUDA optimizations
    device_name = config.get("agent_sft", {}).get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(device)
        print(f"[*] Agent SFT on CUDA GPU: {gpu_name} (TF32 enabled)")
    else:
        print(f"[*] Agent SFT on device: {device}")

    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else torch.float32)

    # Load Model Config
    cfg_path = os.path.join(base_model_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        m_cfg_dict = json.load(f)
    trans_config = TransformerConfig(**m_cfg_dict)

    model = SyntheticTransformer(trans_config).to(device)

    # Load base pretrained weights
    base_weights_path = os.path.join(base_model_dir, "base_final.pt")
    if os.path.exists(base_weights_path):
        model.load_state_dict(torch.load(base_weights_path, map_location=device))
        print(f"[*] Loaded base model weights from {base_weights_path}")
    else:
        print("[!] Warning: base_final.pt not found, training from random initialization.")

    # SFT Dataset
    jsonl_file = os.path.join(data_dir, "agent_sft_train.jsonl")
    sft_ds = TrajectoryDataset(jsonl_file, tokenizer, max_len=trans_config.max_position_embeddings)
    print(f"[*] SFT Dataset: {len(sft_ds)} trajectory samples.")

    sft_cfg = config.get("agent_sft", {})
    batch_size = sft_cfg.get("batch_size", 4)
    grad_accum = sft_cfg.get("gradient_accumulation_steps", 2)
    max_steps = sft_cfg.get("max_steps", 250)
    lr = float(sft_cfg.get("learning_rate", 5e-4))
    min_lr = float(sft_cfg.get("min_learning_rate", 5e-5))
    warmup_steps = sft_cfg.get("warmup_steps", 25)

    train_loader = DataLoader(
        sft_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=use_cuda
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=sft_cfg.get("weight_decay", 0.01))
    scheduler = get_lr_scheduler(optimizer, warmup_steps, max_steps, lr, min_lr)

    step = 0
    start_time = time.time()
    model.train()
    optimizer.zero_grad()

    while step < max_steps:
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device, non_blocking=use_cuda), batch_y.to(device, non_blocking=use_cuda)
            
            if use_cuda:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    _, loss = model(batch_x, labels=batch_y)
            else:
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
                print(f"SFT Step {step}/{max_steps} | Loss: {loss.item() * grad_accum:.4f} | LR: {current_lr:.6f} | Elapsed: {elapsed:.1f}s")

            if step >= max_steps:
                break

    # Save agent model
    agent_ckpt_path = os.path.join(agent_model_dir, "agent_final.pt")
    torch.save(model.state_dict(), agent_ckpt_path)
    print(f"[+] Agent SFT completed. Model saved to {agent_ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for Agent Model")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml", help="Path to config yaml")
    args = parser.parse_args()
    train_agent_sft(args.config)


if __name__ == "__main__":
    main()
