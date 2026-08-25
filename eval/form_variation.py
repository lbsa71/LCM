"""Controlled semantic form-variation experiment for the LCM parser layer.

This module intentionally owns only the language-to-canonical-operation stage.
It can therefore be run independently of base pretraining, agent SFT, and the
deterministic execution shell. Future experiment families can reuse its typed
examples and split construction without coupling to a particular agent model.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import yaml
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from training.model import SyntheticTransformer, TransformerConfig
from training.pretrain import get_lr_scheduler
from training.tokenizer import train_synthetic_tokenizer
from utils.timer import StepTimer


OPERATIONS = ("ADD", "SUBTRACT", "COMPARE")


@dataclass(frozen=True)
class FormVariationExample:
    """One natural-language realization of a canonical semantic operation."""

    utterance: str
    target: str
    operation: str
    template_id: int
    left: int
    right: int


@dataclass(frozen=True)
class ConditionSplits:
    """Train/evaluation partitions for a specified amount of form variation."""

    train: tuple[FormVariationExample, ...]
    seen_form: tuple[FormVariationExample, ...]
    same_meaning_unseen_form: tuple[FormVariationExample, ...]
    unseen_operands_seen_form: tuple[FormVariationExample, ...]
    minimal_contrasts: tuple[FormVariationExample, ...]


class FormVariationGenerator:
    """Renders operation frames into deliberately controlled language variants."""

    _TEMPLATES: dict[str, tuple[str, ...]] = {
        "ADD": (
            "Calculate {left} plus {right}.",
            "What is the sum of {left} and {right}?",
            "Add {left} to {right}.",
            "What do {left} and {right} total?",
            "There are {left} zols and {right} binks. How many are there altogether?",
            "Combine {left} units with {right} units.",
            "If {left} items arrive and then {right} more arrive, how many items are present?",
            "Work out the combined count of {left} and {right}.",
            "Find the total when {left} is joined with {right}.",
            "How many objects result from grouping {left} with {right}?",
        ),
        "SUBTRACT": (
            "Calculate {left} minus {right}.",
            "What is the difference between {left} and {right}?",
            "Subtract {right} from {left}.",
            "How many more zols are there than binks when there are {left} zols and {right} binks?",
            "Start with {left} items and remove {right}. How many remain?",
            "Find the amount left after taking {right} away from {left}.",
            "Work out the gap from {left} down to {right}.",
            "What remains if {right} is deducted from {left}?",
            "Determine by how much {left} exceeds {right}.",
            "Reduce {left} by {right}.",
        ),
        "COMPARE": (
            "Are {left} greater than {right}?",
            "Does {left} exceed {right}?",
            "Which quantity is larger: {left} or {right}?",
            "Are there more zols than binks when there are {left} zols and {right} binks?",
            "Compare {left} with {right}.",
            "Is the first count, {left}, higher than the second count, {right}?",
            "Decide whether {left} is more than {right}.",
            "Tell me if {left} is larger than {right}.",
            "Which is greater, the group of {left} or the group of {right}?",
            "Check whether the count of {left} outranks {right}.",
        ),
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def render_variants(self, operation: str, left: int, right: int) -> tuple[FormVariationExample, ...]:
        """Render every available template for one operation frame."""
        if operation not in OPERATIONS:
            raise ValueError(f"Unsupported operation: {operation}")
        return tuple(
            FormVariationExample(
                utterance=template.format(left=left, right=right),
                target=f"OP={operation}",
                operation=operation,
                template_id=template_id,
                left=left,
                right=right,
            )
            for template_id, template in enumerate(self._TEMPLATES[operation])
        )

    def render_minimal_contrasts(self, left: int, right: int) -> tuple[FormVariationExample, ...]:
        """Create near-neighbour questions that require different operations."""
        return (
            FormVariationExample(
                utterance=f"There are {left} zols and {right} binks. How many are there altogether?",
                target="OP=ADD",
                operation="ADD",
                template_id=4,
                left=left,
                right=right,
            ),
            FormVariationExample(
                utterance=f"How many more zols are there than binks when there are {left} zols and {right} binks?",
                target="OP=SUBTRACT",
                operation="SUBTRACT",
                template_id=3,
                left=left,
                right=right,
            ),
            FormVariationExample(
                utterance=f"Are there more zols than binks when there are {left} zols and {right} binks?",
                target="OP=COMPARE",
                operation="COMPARE",
                template_id=3,
                left=left,
                right=right,
            ),
        )


def build_condition_splits(
    variants_per_operation: int,
    train_pairs: Sequence[tuple[int, int]],
    eval_pairs: Sequence[tuple[int, int]],
    seed: int = 42,
) -> ConditionSplits:
    """Build splits with form holdouts while retaining canonical operation balance."""
    generator = FormVariationGenerator(seed=seed)
    template_count = len(generator._TEMPLATES["ADD"])
    if not 1 <= variants_per_operation < template_count:
        raise ValueError(
            f"variants_per_operation must be in [1, {template_count - 1}], got {variants_per_operation}"
        )

    train: list[FormVariationExample] = []
    seen_form: list[FormVariationExample] = []
    unseen_form: list[FormVariationExample] = []
    unseen_operands: list[FormVariationExample] = []
    contrasts: list[FormVariationExample] = []

    for operation in OPERATIONS:
        for left, right in train_pairs:
            variants = generator.render_variants(operation, left, right)
            train.extend(variants[:variants_per_operation])
            seen_form.extend(variants[:variants_per_operation])
            unseen_form.extend(variants[variants_per_operation:])

        for left, right in eval_pairs:
            variants = generator.render_variants(operation, left, right)
            unseen_operands.extend(variants[:variants_per_operation])

    for left, right in eval_pairs:
        contrasts.extend(generator.render_minimal_contrasts(left, right))

    rng = random.Random(seed)
    for examples in (train, seen_form, unseen_form, unseen_operands, contrasts):
        rng.shuffle(examples)

    return ConditionSplits(
        train=tuple(train),
        seen_form=tuple(seen_form),
        same_meaning_unseen_form=tuple(unseen_form),
        unseen_operands_seen_form=tuple(unseen_operands),
        minimal_contrasts=tuple(contrasts),
    )


class _SemanticParseDataset(Dataset):
    """Loss-masked causal-LM examples for semantic-frame generation."""

    def __init__(self, examples: Iterable[FormVariationExample], tokenizer: Tokenizer, max_length: int):
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        pad_id = tokenizer.token_to_id("<PAD>")
        bos_id = tokenizer.token_to_id("<BOS>")
        eos_id = tokenizer.token_to_id("<EOS>")
        if None in (pad_id, bos_id, eos_id):
            raise ValueError("Tokenizer is missing required special tokens")

        for example in examples:
            prompt = f"<USER> {example.utterance}\n<ASSISTANT> "
            prompt_ids = tokenizer.encode(prompt).ids
            target_ids = tokenizer.encode(example.target).ids + [eos_id]
            input_ids = [bos_id] + prompt_ids + target_ids
            labels = [-100] * (1 + len(prompt_ids)) + target_ids
            if len(input_ids) > max_length:
                raise ValueError(f"Example exceeds max_length={max_length}: {example.utterance}")
            padding = max_length - len(input_ids)
            self.samples.append((
                torch.tensor(input_ids + [pad_id] * padding, dtype=torch.long),
                torch.tensor(labels + [-100] * padding, dtype=torch.long),
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index]


def build_training_loader(dataset: Dataset, batch_size: int, use_cuda: bool) -> DataLoader:
    """Create full batches so every ablation condition receives equal token updates."""
    if len(dataset) < batch_size:
        raise ValueError(f"Dataset has {len(dataset)} examples, fewer than batch_size={batch_size}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=use_cuda,
    )


def _write_tokenizer_corpus(examples: Iterable[FormVariationExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(f"{example.utterance}\n{example.target}\n")


def _candidate_loss(
    model: SyntheticTransformer,
    tokenizer: Tokenizer,
    utterance: str,
    target: str,
    device: torch.device,
) -> float:
    """Score a canonical target conditioned on an utterance."""
    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    prompt_ids = tokenizer.encode(f"<USER> {utterance}\n<ASSISTANT> ").ids
    target_ids = tokenizer.encode(target).ids + [eos_id]
    input_ids = [bos_id] + prompt_ids + target_ids
    labels = [-100] * (1 + len(prompt_ids)) + target_ids
    inputs = torch.tensor([input_ids], dtype=torch.long, device=device)
    target_labels = torch.tensor([labels], dtype=torch.long, device=device)
    _, loss = model(inputs, labels=target_labels)
    if loss is None:
        raise RuntimeError("Candidate scoring expected a loss")
    return float(loss.item())


@torch.no_grad()
def evaluate_operation_accuracy(
    model: SyntheticTransformer,
    tokenizer: Tokenizer,
    examples: Sequence[FormVariationExample],
    device: torch.device,
) -> float:
    """Classify each utterance by lowest canonical-target negative log likelihood."""
    model.eval()
    correct = 0
    candidates = tuple(f"OP={operation}" for operation in OPERATIONS)
    for example in examples:
        predicted = min(
            candidates,
            key=lambda candidate: _candidate_loss(model, tokenizer, example.utterance, candidate, device),
        )
        correct += int(predicted == example.target)
    return correct / max(1, len(examples))


def _device_from_name(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def run_condition(
    splits: ConditionSplits,
    output_dir: Path,
    config: dict,
    seed: int,
) -> dict:
    """Train one scratch semantic parser and evaluate all controlled partitions."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "tokenizer_corpus.txt"
    all_examples = (
        list(splits.train)
        + list(splits.seen_form)
        + list(splits.same_meaning_unseen_form)
        + list(splits.unseen_operands_seen_form)
        + list(splits.minimal_contrasts)
    )
    _write_tokenizer_corpus(all_examples, corpus_path)
    tokenizer = train_synthetic_tokenizer(
        corpus_file=str(corpus_path),
        output_dir=str(output_dir / "tokenizer"),
        vocab_size=int(config["tokenizer"]["vocab_size"]),
        min_frequency=int(config["tokenizer"]["min_frequency"]),
    )

    model_cfg = config["model"]
    tokenizer_meta = tokenizer.get_vocab_size()
    transformer_config = TransformerConfig(
        vocab_size=tokenizer_meta,
        hidden_size=int(model_cfg["hidden_size"]),
        num_hidden_layers=int(model_cfg["num_hidden_layers"]),
        num_attention_heads=int(model_cfg["num_attention_heads"]),
        intermediate_size=int(model_cfg["intermediate_size"]),
        max_position_embeddings=int(model_cfg["max_position_embeddings"]),
        rms_norm_eps=float(model_cfg.get("rms_norm_eps", 1e-5)),
        rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
        tie_word_embeddings=bool(model_cfg.get("tie_word_embeddings", True)),
        pad_token_id=tokenizer.token_to_id("<PAD>") or 0,
        bos_token_id=tokenizer.token_to_id("<BOS>") or 1,
        eos_token_id=tokenizer.token_to_id("<EOS>") or 2,
    )
    device = _device_from_name(str(config["training"].get("device", "auto")))
    model = SyntheticTransformer(transformer_config).to(device)
    dataset = _SemanticParseDataset(splits.train, tokenizer, transformer_config.max_position_embeddings)
    loader = build_training_loader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        use_cuda=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    max_steps = int(config["training"]["max_steps"])
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=int(config["training"]["warmup_steps"]),
        max_steps=max_steps,
        lr=float(config["training"]["learning_rate"]),
        min_lr=float(config["training"]["min_learning_rate"]),
    )
    timer = StepTimer(str(output_dir), phase_name="form_variation_parser")
    model.train()
    optimizer.zero_grad()
    step = 0
    started = time.time()
    grad_accumulation = int(config["training"].get("gradient_accumulation_steps", 1))
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    while step < max_steps:
        for batch_inputs, batch_labels in loader:
            timer.start_step()
            batch_inputs = batch_inputs.to(device, non_blocking=device.type == "cuda")
            batch_labels = batch_labels.to(device, non_blocking=device.type == "cuda")
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16):
                    _, loss = model(batch_inputs, labels=batch_labels)
            else:
                _, loss = model(batch_inputs, labels=batch_labels)
            if loss is None:
                raise RuntimeError("Training expected a loss")
            scaled_loss = loss / grad_accumulation
            scaled_loss.backward()
            if (step + 1) % grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            step += 1
            timer.end_step(step, float(loss.item()), scheduler.get_last_lr()[0], tokens_processed=batch_inputs.numel())
            if step >= max_steps:
                break

    metrics = {
        "seed": seed,
        "train_examples": len(splits.train),
        "steps": step,
        "wall_clock_seconds": round(time.time() - started, 2),
        "seen_form_accuracy": round(evaluate_operation_accuracy(model, tokenizer, splits.seen_form, device), 4),
        "unseen_form_accuracy": round(
            evaluate_operation_accuracy(model, tokenizer, splits.same_meaning_unseen_form, device), 4
        ),
        "unseen_operands_seen_form_accuracy": round(
            evaluate_operation_accuracy(model, tokenizer, splits.unseen_operands_seen_form, device), 4
        ),
        "minimal_contrast_accuracy": round(
            evaluate_operation_accuracy(model, tokenizer, splits.minimal_contrasts, device), 4
        ),
        "timer": timer.get_summary(),
    }
    timer.export_csv("step_metrics.csv")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    torch.save(model.state_dict(), output_dir / "parser_final.pt")
    return metrics


def run_ablation(config_path: str) -> dict:
    """Run every configured variation condition and persist a single comparison report."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    experiment = config["form_variation"]
    output_dir = Path(experiment["output_dir"])
    conditions = tuple(int(value) for value in experiment["variants_per_operation"])
    seeds = tuple(int(value) for value in experiment["seeds"])
    train_pairs = tuple(tuple(pair) for pair in experiment["train_pairs"])
    eval_pairs = tuple(tuple(pair) for pair in experiment["eval_pairs"])

    results: dict[str, list[dict]] = {}
    for variants in conditions:
        splits = build_condition_splits(variants, train_pairs, eval_pairs, seed=int(config["seed"]))
        condition_results = []
        for seed in seeds:
            run_dir = output_dir / f"variants_{variants}" / f"seed_{seed}"
            condition_results.append(run_condition(splits, run_dir, config, seed))
        results[str(variants)] = condition_results

    summary: dict[str, dict[str, float]] = {}
    metric_names = (
        "seen_form_accuracy",
        "unseen_form_accuracy",
        "unseen_operands_seen_form_accuracy",
        "minimal_contrast_accuracy",
        "wall_clock_seconds",
    )
    for variants, condition_results in results.items():
        summary[variants] = {
            metric: round(sum(result[metric] for result in condition_results) / len(condition_results), 4)
            for metric in metric_names
        }

    report = {
        "experiment": "form_variation_semantic_parser",
        "config": config_path,
        "conditions": list(conditions),
        "seeds": list(seeds),
        "summary": summary,
        "runs": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LCM semantic form-variation ablation")
    parser.add_argument("--config", default="configs/form_variation.yaml")
    args = parser.parse_args()
    report = run_ablation(args.config)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
