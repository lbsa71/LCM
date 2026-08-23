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
from utils.timer import StepTimer


class TextDataset(Dataset):
    """Tokenizes and packs lines into fixed-length chunks."""

    def __init__(self, text_file: str, tokenizer: Tokenizer, seq_len: int = 512):
        self.seq_len = seq_len
        bos_id = tokenizer.token_to_id("<BOS>")
        eos_id = tokenizer.token_to_id("<EOS>")

        all_ids = []
        print(f"[*] Reading and batch-tokenizing {text_file}...")
        with open(text_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        batch_chunk_size = 50000
        for i in range(0, len(lines), batch_chunk_size):
            batch_lines = lines[i: i + batch_chunk_size]
            encodings = tokenizer.encode_batch(batch_lines)
            for enc in encodings:
                all_ids.extend([bos_id] + enc.ids + [eos_id])

        print(f"[+] Tokenized {len(lines)} lines into {len(all_ids)} tokens.")

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

    # Device & CUDA optimizations
    device_name = config.get("pretrain", {}).get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(device)
        print(f"[*] Pretraining on CUDA GPU: {gpu_name} (TF32 enabled)")
    else:
        print(f"[*] Pretraining on device: {device}")

    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else torch.float32)

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
    save_milestones = p_cfg.get("save_milestones", [])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=use_cuda
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=use_cuda) if len(val_ds) > 0 else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=p_cfg.get("weight_decay", 0.01))
    scheduler = get_lr_scheduler(optimizer, warmup_steps, max_steps, lr, min_lr)

    step = 0
    start_time = time.time()
    timer = StepTimer(base_model_dir, phase_name="pretrain")
    model.train()
    optimizer.zero_grad()

    while step < max_steps:
        for batch_x, batch_y in train_loader:
            timer.start_step()
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
            current_lr = scheduler.get_last_lr()[0]
            tokens_in_step = batch_size * seq_len
            timer.end_step(step, loss.item() * grad_accum, current_lr, tokens_processed=tokens_in_step)

            if step in save_milestones:
                milestone_path = os.path.join(base_model_dir, f"base_step_{step}.pt")
                torch.save(model.state_dict(), milestone_path)

            if step % 50 == 0 or step == max_steps:
                elapsed = time.time() - start_time
                print(f"Step {step}/{max_steps} | Loss: {loss.item() * grad_accum:.4f} | LR: {current_lr:.6f} | Elapsed: {elapsed:.1f}s")

            if step >= max_steps:
                break


    total_time = time.time() - start_time
    total_tokens_processed = step * batch_size * seq_len
    tokens_per_sec = total_tokens_processed / max(0.001, total_time)
    ms_per_step = (total_time / max(1, step)) * 1000.0

    # Save final model & config
    final_ckpt_path = os.path.join(base_model_dir, "base_final.pt")
    torch.save(model.state_dict(), final_ckpt_path)
    
    cfg_path = os.path.join(base_model_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(trans_config.__dict__, f, indent=2)

    # Export per-step CSV
    csv_metrics_path = timer.export_csv("step_metrics.csv")

    # Save training metrics log
    metrics = {
        "phase": "pretrain",
        "preset": preset_name,
        "total_steps": step,
        "total_params": total_params,
        "total_wall_clock_seconds": round(total_time, 2),
        "ms_per_step": round(ms_per_step, 2),
        "tokens_per_second": round(tokens_per_sec, 2),
        "total_tokens_processed": total_tokens_processed,
        "final_loss": round(float(loss.item() * grad_accum), 6),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if use_cuda else "cpu",
        "step_metrics_csv": os.path.basename(csv_metrics_path)
    }
    metrics_path = os.path.join(base_model_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[+] Pretraining completed in {total_time:.2f}s ({tokens_per_sec:.1f} tokens/s, {ms_per_step:.1f} ms/step).")
    print(f"[+] Metrics logged to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Pretrain Synthetic Transformer")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml", help="Path to config yaml")
    args = parser.parse_args()
    train_pretrain(args.config)


if __name__ == "__main__":
    main()
