"""Supervised Fine-Tuning (SFT) pipeline for synthetic agent trajectories with loss masking."""

import argparse
import json
import os
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from training.model import SyntheticTransformer, TransformerConfig
from training.model_loader import load_model_and_tokenizer
from training.pretrain import get_lr_scheduler
from utils.timer import StepTimer


class TrajectoryDataset(Dataset):
    """Encodes agent trajectories and applies target loss masking."""

    def __init__(self, jsonl_file: str, tokenizer, max_len: int = 512):
        self.max_len = max_len
        self.samples = []

        pad_id = tokenizer.token_to_id("<PAD>")
        if pad_id is None:
            pad_id = 0
        bos_id = tokenizer.token_to_id("<BOS>")
        eos_id = tokenizer.token_to_id("<EOS>")

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                turns = item.get("turns", [])
                
                input_ids = [bos_id] if bos_id is not None else []
                labels = [-100] * len(input_ids)

                for turn in turns:
                    role = turn["role"]
                    content = turn["content"]
                    is_train = turn.get("train", False)

                    # Role tags
                    tag = f"<{role.upper()}>"
                    tag_id = tokenizer.token_to_id(tag)
                    if tag_id is None:
                        tag_id = tokenizer.token_to_id("<UNK>") or pad_id

                    content_ids = tokenizer.encode(content).ids
                    closing_eos = [eos_id] if eos_id is not None else []
                    turn_ids = [tag_id] + content_ids + closing_eos

                    input_ids.extend(turn_ids)
                    if is_train:
                        # Model is trained to predict tag, content, and closing EOS
                        labels.extend(turn_ids)
                    else:
                        # Mask out prompt and observation
                        labels.extend([-100] * len(turn_ids))

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
    backend = config.get("backend", "custom")
    run_dir = f"runs/{preset_name}"
    data_dir = os.path.join(run_dir, "data")
    tok_dir = os.path.join(run_dir, "tokenizer")
    base_model_dir = os.path.join(run_dir, "base_model")
    agent_model_dir = os.path.join(run_dir, "agent_model")
    os.makedirs(agent_model_dir, exist_ok=True)

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

    # Model & Tokenizer
    if backend == "huggingface":
        model, tokenizer = load_model_and_tokenizer(config, device=device)
        max_seq_len = getattr(model.config, "max_position_embeddings", 2048)
    else:
        tok_path = os.path.join(tok_dir, "tokenizer.json")
        config["tokenizer_path"] = tok_path
        base_weights_path = os.path.join(base_model_dir, "base_final.pt")
        model, tokenizer = load_model_and_tokenizer(config, device=device, checkpoint_path=base_weights_path)
        max_seq_len = model.config.max_position_embeddings

    # SFT Dataset
    jsonl_file = os.path.join(data_dir, "agent_sft_train.jsonl")
    sft_ds = TrajectoryDataset(jsonl_file, tokenizer, max_len=min(max_seq_len, 1024))
    print(f"[*] SFT Dataset: {len(sft_ds)} trajectory samples.")

    sft_cfg = config.get("agent_sft", {})
    batch_size = sft_cfg.get("batch_size", 4)
    grad_accum = sft_cfg.get("gradient_accumulation_steps", 2)
    max_steps = sft_cfg.get("max_steps", 250)
    lr = float(sft_cfg.get("learning_rate", 5e-4))
    min_lr = float(sft_cfg.get("min_learning_rate", 5e-5))
    warmup_steps = sft_cfg.get("warmup_steps", 25)
    save_milestones = sft_cfg.get("save_milestones", [])

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
    timer = StepTimer(agent_model_dir, phase_name="agent_sft")
    milestone_timings = {}
    model.train()
    optimizer.zero_grad()

    while step < max_steps:
        for batch_x, batch_y in train_loader:
            timer.start_step()
            batch_x, batch_y = batch_x.to(device, non_blocking=use_cuda), batch_y.to(device, non_blocking=use_cuda)
            
            if use_cuda:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(batch_x, labels=batch_y)
                    loss = out.loss if hasattr(out, "loss") else out[1]
            else:
                out = model(batch_x, labels=batch_y)
                loss = out.loss if hasattr(out, "loss") else out[1]

            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1
            current_lr = scheduler.get_last_lr()[0]
            tokens_in_step = batch_size * min(max_seq_len, 1024)
            timer.end_step(step, loss.item() * grad_accum, current_lr, tokens_processed=tokens_in_step)

            if step in save_milestones:
                m_elapsed = time.time() - start_time
                if backend == "huggingface":
                    m_dir = os.path.join(agent_model_dir, f"agent_step_{step}")
                    model.save_pretrained(m_dir)
                    tokenizer.hf_tokenizer.save_pretrained(m_dir)
                else:
                    milestone_ckpt = os.path.join(agent_model_dir, f"agent_step_{step}.pt")
                    torch.save(model.state_dict(), milestone_ckpt)
                milestone_timings[step] = round(m_elapsed, 2)

            if step % 50 == 0 or step == max_steps:
                elapsed = time.time() - start_time
                print(f"SFT Step {step}/{max_steps} | Loss: {loss.item() * grad_accum:.4f} | LR: {current_lr:.6f} | Elapsed: {elapsed:.1f}s")

            if step >= max_steps:
                break

    total_time = time.time() - start_time
    seq_len = min(max_seq_len, 1024)
    total_tokens_processed = step * batch_size * seq_len
    tokens_per_sec = total_tokens_processed / max(0.001, total_time)
    ms_per_step = (total_time / max(1, step)) * 1000.0

    # Save agent model
    if backend == "huggingface":
        model.save_pretrained(agent_model_dir)
        tokenizer.hf_tokenizer.save_pretrained(agent_model_dir)
    else:
        agent_ckpt_path = os.path.join(agent_model_dir, "agent_final.pt")
        torch.save(model.state_dict(), agent_ckpt_path)

    # Export per-step CSV
    csv_metrics_path = timer.export_csv("step_metrics.csv")

    # Save training metrics log
    metrics = {
        "phase": "agent_sft",
        "preset": preset_name,
        "backend": backend,
        "total_steps": step,
        "total_wall_clock_seconds": round(total_time, 2),
        "ms_per_step": round(ms_per_step, 2),
        "tokens_per_second": round(tokens_per_sec, 2),
        "total_tokens_processed": total_tokens_processed,
        "final_loss": round(loss.item() * grad_accum, 4),
        "milestone_timings": milestone_timings
    }
    metrics_path = os.path.join(agent_model_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[+] Agent SFT completed in {total_time:.2f}s ({tokens_per_sec:.1f} tokens/s, {ms_per_step:.1f} ms/step).")
    print(f"[+] Metrics logged to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smollm2_135m_agent.yaml")
    args = parser.parse_args()
    train_agent_sft(args.config)
