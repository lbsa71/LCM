"""Discriminative semantic-bottleneck experiments for the LCM parser layer."""

from __future__ import annotations

import argparse
import json
import hashlib
import random
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from eval.form_variation import (
    FormVariationExample,
    OPERATIONS,
    ConditionSplits,
    _OPERATION_TO_LABEL,
    _PRESSURE_TRACKS,
    _ROBUST_PRESSURE_TRACKS,
    _device_from_name,
    _distribution_summary,
    _transformer_config_from,
    build_clean_split_manifest,
    build_condition_splits,
    build_fixed_pressure_test,
    build_phase25_split_manifest,
    build_training_loader,
    extract_response_boundary_features,
    fit_frozen_linear_probe,
    predict_operation_labels_batched,
    run_condition,
    steps_for_exposure,
    train_fixed_byte_tokenizer,
    warmup_steps_for_training,
)
from training.model import SyntheticTransformer, TransformerConfig
from training.pretrain import get_lr_scheduler
from utils.timer import StepTimer


_B_FIRST_TEMPLATES: dict[str, tuple[str, ...]] = {
    "ADD": (
        "Alongside {right} binks are {left} zols. How many are there altogether?",
        "The amounts are {right} and then {left}; calculate their sum.",
        "Starting with the mention of {right}, add the value {left}.",
        "First consider {right}; combine it with {left} and report the total.",
    ),
    "SUBTRACT": (
        "Take {right} away from {left}; what remains?",
        "Beginning with {right} as the amount removed, calculate {left} minus it.",
        "The deduction is {right}; subtract it from the starting amount {left}.",
        "Remove the quantity {right} from the quantity {left} and report the remainder.",
    ),
    "COMPARE": (
        "Relative to {right}, is {left} larger?",
        "Use {right} as the reference; does {left} exceed it?",
        "Between {right} and {left}, is the latter quantity greater?",
        "First consider {right}; now decide whether {left} is larger.",
    ),
}

_STAGEB_SCAFFOLD_TEMPLATES: dict[str, tuple[str, str]] = {
    "ADD": (
        "Two registers list {left} zols and {right} binks. Give their combined count.",
        "Two registers list {right} binks and {left} zols. Give their combined count.",
    ),
    "SUBTRACT": (
        "Two registers list {left} zols and {right} binks. Give the zol surplus.",
        "Two registers list {right} binks and {left} zols. Give the zol surplus.",
    ),
    "COMPARE": (
        "Two registers list {left} zols and {right} binks. Do zols outnumber binks?",
        "Two registers list {right} binks and {left} zols. Do zols outnumber binks?",
    ),
}

_STAGEB_DEVELOPMENT_TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    "seen_form_unseen_operands": {
        "ADD": ("Calculate {left} plus {right}.", "Add {right} to {left}."),
        "SUBTRACT": ("Calculate {left} minus {right}.", "Subtract {right} from {left}."),
        "COMPARE": ("Does {left} exceed {right}?", "Relative to {right}, does {left} exceed it?"),
    },
    "held_out_templates": {
        "ADD": ("State the sum formed from {left} and {right}.", "With {right} first, total it with {left}."),
        "SUBTRACT": (
            "State the difference of {left} relative to {right}.",
            "Deduct {right} from {left} and state the remainder.",
        ),
        "COMPARE": (
            "Assess whether {left} is larger than {right}.",
            "Using {right} as reference, is {left} larger?",
        ),
    },
    "lexical_shift": {
        "ADD": ("Aggregate {left} with {right}.", "Unite {right} with {left}."),
        "SUBTRACT": (
            "What surplus does {left} retain over {right}?",
            "Against {right}, what surplus has {left}?",
        ),
        "COMPARE": ("Does {left} eclipse {right}?", "Compared with {right}, does {left} eclipse it?"),
    },
    "syntax_order_reversal": {
        "ADD": (
            "When {left} joins {right}, what total results?",
            "When {right} is joined by {left}, what total results?",
        ),
        "SUBTRACT": ("From {left}, remove {right}.", "Remove {right} from {left}."),
        "COMPARE": (
            "Is the first value {left} above the second {right}?",
            "Against {right}, is {left} the larger value?",
        ),
    },
    "discourse_distractor": {
        "ADD": (
            "Though {left} may exceed {right}, report their total.",
            "Though {right} comes first, combine it with {left}.",
        ),
        "SUBTRACT": (
            "A total is irrelevant; calculate {left} less {right}.",
            "The removed amount is {right}; deduct it from {left}.",
        ),
        "COMPARE": (
            "A sum can wait; decide whether {left} exceeds {right}.",
            "Using {right} as reference, decide whether {left} exceeds it.",
        ),
    },
    "minimal_contrast": {
        "ADD": (
            "Two inventories hold {left} zols and {right} binks. Report the combined count.",
            "Two inventories hold {right} binks and {left} zols. Report the combined count.",
        ),
        "SUBTRACT": (
            "Two inventories hold {left} zols and {right} binks. Report the zol surplus.",
            "Two inventories hold {right} binks and {left} zols. Report the zol surplus.",
        ),
        "COMPARE": (
            "Two inventories hold {left} zols and {right} binks. Do zols outnumber binks?",
            "Two inventories hold {right} binks and {left} zols. Do zols outnumber binks?",
        ),
    },
}

_STAGEC_DEVELOPMENT_TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    "seen_form_unseen_operands": {
        "ADD": ("What is the sum of {left} and {right}?", "Add {right} to {left}."),
        "SUBTRACT": ("Find {left} minus {right}.", "Take {right} away from {left}."),
        "COMPARE": ("Is {left} greater than {right}?", "Relative to {right}, is {left} greater?"),
    },
    "held_out_templates": {
        "ADD": ("Return the total yielded by {left} with {right}.", "Given {right} first, total it with {left}."),
        "SUBTRACT": ("From {left}, return what remains after {right} leaves.", "With {right} removed, what remains of {left}?"),
        "COMPARE": ("Judge if {left} is the larger of {left} and {right}.", "Taking {right} as reference, judge if {left} is larger."),
    },
    "lexical_shift": {
        "ADD": ("Amass {left} together with {right}.", "Pool {right} with {left}."),
        "SUBTRACT": ("What residue has {left} beyond {right}?", "Beyond {right}, what residue has {left}?"),
        "COMPARE": ("Does {left} surpass {right}?", "Against {right}, does {left} surpass it?"),
    },
    "syntax_order_reversal": {
        "ADD": ("Joining {left} to {right} yields what total?", "Joined to {right}, what total does {left} yield?"),
        "SUBTRACT": ("From {left}, taking {right} leaves what?", "Take away {right}; from {left}, what stays?"),
        "COMPARE": ("The first value is {left}; is it above {right}?", "The reference is {right}; above it is {left}?"),
    },
    "discourse_distractor": {
        "ADD": ("Their order is irrelevant; total {left} and {right}.", "Although {right} is smaller, total it with {left}."),
        "SUBTRACT": ("With {left} available, ignore comparison and subtract {right}.", "The possible total is irrelevant; remove {right} from {left}."),
        "COMPARE": ("Do not total them; decide if {left} exceeds {right}.", "Ignore any difference from {right}; decide if {left} exceeds it."),
    },
    "minimal_contrast": {
        "ADD": ("Two manifests show {left} zols and {right} binks. Give the combined amount.", "Two manifests show {right} binks and {left} zols. Give the combined amount."),
        "SUBTRACT": ("Two manifests show {left} zols and {right} binks. Give the zol excess.", "Two manifests show {right} binks and {left} zols. Give the zol excess."),
        "COMPARE": ("Two manifests show {left} zols and {right} binks. Are zols more numerous?", "Two manifests show {right} binks and {left} zols. Are zols more numerous?"),
    },
}

_STAGEC_CONFIRMATION_TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    "seen_form_unseen_operands": {
        "ADD": ("Combine the values {left} and {right}.", "Combine {right} with {left}."),
        "SUBTRACT": ("Starting from {left}, remove {right}.", "Remove {right} from the starting value {left}."),
        "COMPARE": ("Determine whether {left} is above {right}.", "Taking {right} as the benchmark, is {left} above it?"),
    },
    "held_out_templates": {
        "ADD": ("What quantity emerges by merging {left} with {right}?", "The first part is {right}; merge it with {left}."),
        "SUBTRACT": ("How much of {left} survives a reduction by {right}?", "The reduction is {right}; apply it to {left} and report what survives."),
        "COMPARE": ("Choose whether {left} stands higher than {right}.", "With {right} as the reference point, does {left} stand higher?"),
    },
    "lexical_shift": {
        "ADD": ("Accumulate {left} together with {right}.", "Accrete {right} onto {left}."),
        "SUBTRACT": ("What margin does {left} retain over {right}?", "Relative to {right}, what margin does {left} retain?"),
        "COMPARE": ("Does {left} outrank {right} numerically?", "Against {right}, does {left} outrank it numerically?"),
    },
    "syntax_order_reversal": {
        "ADD": ("The total after {left} receives {right} is what?", "Added to {right}, what total does {left} produce?"),
        "SUBTRACT": ("Starting at {left}, taking out {right} leaves what?", "Taken out is {right}; the source amount is {left}. What remains?"),
        "COMPARE": ("The candidate is {left}; above {right} is it?", "The benchmark being {right}, above it is {left}?"),
    },
    "discourse_distractor": {
        "ADD": ("Whether {left} exceeds {right} is beside the point; supply their total.", "Do not subtract {right} from {left}; supply their total."),
        "SUBTRACT": ("Do not report the total of {left} and {right}; report the former minus the latter.", "Although {right} could be compared with {left}, remove it from {left}."),
        "COMPARE": ("A sum of {left} and {right} is not requested; say whether the former is larger.", "Subtracting {right} from {left} is unnecessary; say whether {left} is larger."),
    },
    "minimal_contrast": {
        "ADD": ("Two ledgers record {left} taves and {right} nims. Give the combined quantity.", "Two ledgers record {right} nims and {left} taves. Give the combined quantity."),
        "SUBTRACT": ("Two ledgers record {left} taves and {right} nims. Give the tave remainder.", "Two ledgers record {right} nims and {left} taves. Give the tave remainder."),
        "COMPARE": ("Two ledgers record {left} taves and {right} nims. Are taves more numerous?", "Two ledgers record {right} nims and {left} taves. Are taves more numerous?"),
    },
}


def semantic_binding_label(example: FormVariationExample) -> int:
    """Return 0 when canonical A is mentioned first and 1 when B is first."""
    if example.left == example.right:
        raise ValueError("Binding order is undefined when canonical arguments are equal")
    mentions = [(match.start(), int(match.group())) for match in re.finditer(r"-?\d+", example.utterance)]
    left_positions = [position for position, value in mentions if value == example.left]
    right_positions = [position for position, value in mentions if value == example.right]
    if not left_positions or not right_positions:
        raise ValueError(f"Utterance does not mention both canonical arguments: {example.utterance!r}")
    return int(right_positions[0] < left_positions[0])


def build_binding_counterbalanced_examples(
    examples: Sequence[FormVariationExample],
) -> tuple[FormVariationExample, ...]:
    """Replace forms in place so mention order is 50/50 within every frame.

    The operation and operands are unchanged and no examples are added. This
    prevents the auxiliary binding label from becoming either a majority-class
    shortcut or a proxy for one operation.
    """
    grouped: dict[tuple[str, int, int], list[int]] = {}
    for index, example in enumerate(examples):
        if example.operation not in OPERATIONS:
            raise ValueError(f"Unsupported operation: {example.operation}")
        grouped.setdefault((example.operation, example.left, example.right), []).append(index)
    result = list(examples)
    for group_key in sorted(grouped):
        indices = sorted(grouped[group_key], key=lambda index: examples[index].template_id)
        if len(indices) % 2:
            raise ValueError(f"Binding counterbalance requires an even cell size: {group_key}")
        target_b_first = len(indices) // 2
        current_b_first = [index for index in indices if semantic_binding_label(examples[index]) == 1]
        if len(current_b_first) > target_b_first:
            raise ValueError(f"Binding cell already has too many B-first examples: {group_key}")
        candidates = [index for index in reversed(indices) if semantic_binding_label(examples[index]) == 0]
        replacement_count = target_b_first - len(current_b_first)
        operation, left, right = group_key
        for rank, index in enumerate(candidates[:replacement_count]):
            template = _B_FIRST_TEMPLATES[operation][rank % len(_B_FIRST_TEMPLATES[operation])]
            result[index] = FormVariationExample(
                utterance=template.format(left=left, right=right),
                target=f"OP={operation}",
                operation=operation,
                template_id=4_000 + OPERATIONS.index(operation) * 10 + rank,
                left=left,
                right=right,
            )
    return tuple(result)


def build_scaffold_counterbalanced_examples(
    examples: Sequence[FormVariationExample],
    *,
    replacement_fraction: float,
    seed: int,
) -> tuple[FormVariationExample, ...]:
    """Replace equal A-first/B-first counts with shared training scaffolds."""
    if not 0 < replacement_fraction <= 1:
        raise ValueError("replacement_fraction must be in (0, 1]")
    grouped: dict[tuple[str, int, int], list[int]] = {}
    for index, example in enumerate(examples):
        grouped.setdefault((example.operation, example.left, example.right), []).append(index)
    result = list(examples)
    rng = random.Random(seed)
    for group_key in sorted(grouped):
        indices = grouped[group_key]
        replacement_count = round(len(indices) * replacement_fraction)
        if replacement_count < 2 or replacement_count % 2:
            raise ValueError("Scaffold replacement must select an even positive count per cell")
        per_binding = replacement_count // 2
        candidates = {
            binding: [index for index in indices if semantic_binding_label(examples[index]) == binding]
            for binding in (0, 1)
        }
        if len(candidates[0]) != len(candidates[1]) or any(
            len(candidates[binding]) < per_binding for binding in (0, 1)
        ):
            raise ValueError(f"Scaffold source cell is not binding-balanced: {group_key}")
        operation, left, right = group_key
        for binding in (0, 1):
            rng.shuffle(candidates[binding])
            template = _STAGEB_SCAFFOLD_TEMPLATES[operation][binding]
            for index in candidates[binding][:per_binding]:
                result[index] = FormVariationExample(
                    utterance=template.format(left=left, right=right),
                    target=f"OP={operation}",
                    operation=operation,
                    template_id=5_000 + OPERATIONS.index(operation) * 10 + binding,
                    left=left,
                    right=right,
                )
    return tuple(result)


def build_stagec_invariance_pairs(
    examples: Sequence[FormVariationExample], *, seed: int
) -> tuple[tuple[FormVariationExample, FormVariationExample], ...]:
    """Pair every form once with an opposite-order form from the same semantic frame."""
    grouped: dict[tuple[str, int, int], list[FormVariationExample]] = {}
    for example in examples:
        grouped.setdefault((example.operation, example.left, example.right), []).append(example)
    rng = random.Random(seed)
    pairs: list[tuple[FormVariationExample, FormVariationExample]] = []
    for frame in sorted(grouped):
        by_binding = {
            binding: sorted(
                (example for example in grouped[frame] if semantic_binding_label(example) == binding),
                key=lambda example: (example.template_id, example.utterance),
            )
            for binding in (0, 1)
        }
        if len(by_binding[0]) != len(by_binding[1]) or not by_binding[0]:
            raise ValueError(f"Stage-C invariance frame is not binding-balanced: {frame}")
        rng.shuffle(by_binding[1])
        pairs.extend(zip(by_binding[0], by_binding[1]))
    return tuple(pairs)


def build_phase4_stageb_development(
    pairs: Sequence[tuple[int, int]],
) -> dict[str, tuple[FormVariationExample, ...]]:
    """Build fresh Stage-B text with balanced mention order in every track/operation."""
    if len(pairs) != 6:
        raise ValueError("Phase 4 Stage B requires exactly six development operand pairs")
    groups: dict[str, list[FormVariationExample]] = {
        track: [] for track in _STAGEB_DEVELOPMENT_TEMPLATES
    }
    for track_index, (track, operation_templates) in enumerate(
        _STAGEB_DEVELOPMENT_TEMPLATES.items()
    ):
        for operation in OPERATIONS:
            templates = operation_templates[operation]
            for pair_index, (left, right) in enumerate(pairs):
                binding = pair_index % 2
                groups[track].append(
                    FormVariationExample(
                        utterance=templates[binding].format(left=left, right=right),
                        target=f"OP={operation}",
                        operation=operation,
                        template_id=(
                            6_000
                            + track_index * 100
                            + OPERATIONS.index(operation) * 10
                            + binding
                        ),
                        left=left,
                        right=right,
                    )
                )
    return {track: tuple(examples) for track, examples in groups.items()}


def build_phase4_stagec_development(
    pairs: Sequence[tuple[int, int]],
) -> dict[str, tuple[FormVariationExample, ...]]:
    """Build the third fresh, binding-balanced development suite."""
    if len(pairs) != 6:
        raise ValueError("Phase 4 Stage C requires exactly six development operand pairs")
    groups: dict[str, list[FormVariationExample]] = {
        track: [] for track in _STAGEC_DEVELOPMENT_TEMPLATES
    }
    for track_index, (track, operation_templates) in enumerate(
        _STAGEC_DEVELOPMENT_TEMPLATES.items()
    ):
        for operation in OPERATIONS:
            templates = operation_templates[operation]
            for pair_index, (left, right) in enumerate(pairs):
                binding = pair_index % 2
                groups[track].append(
                    FormVariationExample(
                        utterance=templates[binding].format(left=left, right=right),
                        target=f"OP={operation}",
                        operation=operation,
                        template_id=(
                            7_000
                            + track_index * 100
                            + OPERATIONS.index(operation) * 10
                            + binding
                        ),
                        left=left,
                        right=right,
                    )
                )
    return {track: tuple(examples) for track, examples in groups.items()}


def build_phase4_stagec_confirmation(
    pairs: Sequence[tuple[int, int]],
) -> dict[str, tuple[FormVariationExample, ...]]:
    """Build a sealed suite containing both mention orders for every frame."""
    if len(pairs) != 6:
        raise ValueError("Phase 4 Stage C confirmation requires exactly six operand pairs")
    groups: dict[str, list[FormVariationExample]] = {
        track: [] for track in _STAGEC_CONFIRMATION_TEMPLATES
    }
    for track_index, (track, operation_templates) in enumerate(
        _STAGEC_CONFIRMATION_TEMPLATES.items()
    ):
        for operation in OPERATIONS:
            templates = operation_templates[operation]
            for pair_index, (left, right) in enumerate(pairs):
                for binding, template in enumerate(templates):
                    groups[track].append(
                        FormVariationExample(
                            utterance=template.format(left=left, right=right),
                            target=f"OP={operation}",
                            operation=operation,
                            template_id=(
                                8_000
                                + track_index * 100
                                + OPERATIONS.index(operation) * 10
                                + binding
                            ),
                            left=left,
                            right=right,
                        )
                    )
    return {track: tuple(examples) for track, examples in groups.items()}


class SemanticBottleneckDataset(Dataset):
    """Prompt-only classifier examples with no serialized target leakage."""

    def __init__(self, examples: Iterable[FormVariationExample], tokenizer: Tokenizer, max_length: int):
        self.samples: list[dict[str, torch.Tensor]] = []
        pad_id = tokenizer.token_to_id("<PAD>")
        bos_id = tokenizer.token_to_id("<BOS>")
        if pad_id is None or bos_id is None:
            raise ValueError("Tokenizer is missing required PAD or BOS tokens")
        for example in examples:
            if example.operation not in _OPERATION_TO_LABEL:
                raise ValueError(f"Unsupported operation: {example.operation}")
            prompt = f"<USER> {example.utterance}\n<ASSISTANT> "
            input_ids = [bos_id, *tokenizer.encode(prompt).ids]
            if len(input_ids) > max_length:
                raise ValueError(f"Example exceeds max_length={max_length}: {example.utterance}")
            boundary_index = len(input_ids) - 1
            padded = input_ids + [pad_id] * (max_length - len(input_ids))
            self.samples.append(
                {
                    "input_ids": torch.tensor(padded, dtype=torch.long),
                    "boundary_index": torch.tensor(boundary_index, dtype=torch.long),
                    "operation_label": torch.tensor(_OPERATION_TO_LABEL[example.operation], dtype=torch.long),
                    "binding_label": torch.tensor(semantic_binding_label(example), dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


class SemanticInvarianceDataset(Dataset):
    """Two opposite-order forms per item for fixed-budget consistency training."""

    def __init__(
        self,
        pairs: Sequence[tuple[FormVariationExample, FormVariationExample]],
        tokenizer: Tokenizer,
        max_length: int,
    ):
        first = SemanticBottleneckDataset(
            (pair[0] for pair in pairs), tokenizer, max_length
        )
        second = SemanticBottleneckDataset(
            (pair[1] for pair in pairs), tokenizer, max_length
        )
        self.samples: list[dict[str, torch.Tensor]] = []
        for first_item, second_item in zip(first.samples, second.samples):
            self.samples.append(
                {
                    **{f"first_{key}": value for key, value in first_item.items()},
                    **{f"second_{key}": value for key, value in second_item.items()},
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


class SemanticBottleneckParser(nn.Module):
    """Shared causal encoder with explicit operation and role-order heads."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.encoder = SyntheticTransformer(config)
        self.operation_head = nn.Linear(config.hidden_size, len(_OPERATION_TO_LABEL))
        self.binding_head = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.operation_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.operation_head.bias)
        nn.init.normal_(self.binding_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.binding_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        boundary_indices: torch.Tensor,
        *,
        operation_labels: torch.Tensor | None = None,
        binding_labels: torch.Tensor | None = None,
        binding_loss_weight: float = 0.0,
    ) -> dict[str, torch.Tensor | None]:
        if binding_loss_weight < 0:
            raise ValueError("binding_loss_weight must be non-negative")
        hidden = self.encoder.forward_hidden_states(input_ids)
        if boundary_indices.ndim != 1 or boundary_indices.shape[0] != input_ids.shape[0]:
            raise ValueError("boundary_indices must contain one index per batch item")
        if torch.any(boundary_indices < 0) or torch.any(boundary_indices >= input_ids.shape[1]):
            raise ValueError("boundary index is outside the input sequence")
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        boundary_states = hidden[batch_indices, boundary_indices]
        operation_logits = self.operation_head(boundary_states)
        binding_logits = self.binding_head(boundary_states)

        operation_loss = None
        binding_loss = None
        loss = None
        if operation_labels is not None:
            operation_loss = F.cross_entropy(operation_logits, operation_labels)
            loss = operation_loss
        if binding_labels is not None:
            binding_loss = F.cross_entropy(binding_logits, binding_labels)
            if loss is not None:
                loss = loss + binding_loss_weight * binding_loss
        return {
            "operation_logits": operation_logits,
            "binding_logits": binding_logits,
            "operation_loss": operation_loss,
            "binding_loss": binding_loss,
            "loss": loss,
        }


def score_bottleneck_predictions(
    examples: Sequence[FormVariationExample],
    operation_predictions: Sequence[str],
    binding_predictions: Sequence[int],
) -> dict[str, float]:
    """Score intent, canonical role order, and their conjunction separately."""
    if not (len(examples) == len(operation_predictions) == len(binding_predictions)):
        raise ValueError("Expected examples and prediction counts to match")
    if not examples:
        raise ValueError("Cannot score an empty bottleneck evaluation set")
    operation_correct = 0
    binding_correct = 0
    joint_correct = 0
    for example, operation, binding in zip(examples, operation_predictions, binding_predictions):
        if operation not in OPERATIONS:
            raise ValueError(f"Unsupported predicted operation: {operation}")
        if binding not in (0, 1):
            raise ValueError(f"Unsupported predicted binding label: {binding}")
        operation_match = operation == example.operation
        binding_match = binding == semantic_binding_label(example)
        operation_correct += int(operation_match)
        binding_correct += int(binding_match)
        joint_correct += int(operation_match and binding_match)
    denominator = len(examples)
    return {
        "operation_accuracy": operation_correct / denominator,
        "binding_accuracy": binding_correct / denominator,
        "joint_accuracy": joint_correct / denominator,
    }


@torch.no_grad()
def predict_bottleneck_labels(
    model: SemanticBottleneckParser,
    tokenizer: Tokenizer,
    examples: Sequence[FormVariationExample],
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Return explicit operation and binding predictions for case-level audits."""
    dataset = SemanticBottleneckDataset(examples, tokenizer, model.config.max_position_embeddings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    operation_predictions: list[str] = []
    binding_predictions: list[int] = []
    was_training = model.training
    model.eval()
    for batch in loader:
        output = model(
            batch["input_ids"].to(device),
            batch["boundary_index"].to(device),
        )
        operation_logits = output["operation_logits"]
        binding_logits = output["binding_logits"]
        if operation_logits is None or binding_logits is None:
            raise RuntimeError("Bottleneck evaluation expected both logits")
        operation_predictions.extend(OPERATIONS[index] for index in operation_logits.argmax(dim=-1).cpu().tolist())
        binding_predictions.extend(int(index) for index in binding_logits.argmax(dim=-1).cpu().tolist())
    if was_training:
        model.train()
    return tuple(operation_predictions), tuple(binding_predictions)


@torch.no_grad()
def evaluate_bottleneck(
    model: SemanticBottleneckParser,
    tokenizer: Tokenizer,
    examples: Sequence[FormVariationExample],
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate explicit semantic heads without target candidate scoring."""
    operation_predictions, binding_predictions = predict_bottleneck_labels(
        model, tokenizer, examples, device, batch_size=batch_size
    )
    return score_bottleneck_predictions(examples, operation_predictions, binding_predictions)


def paired_operation_consistency_loss(
    first_logits: torch.Tensor, second_logits: torch.Tensor
) -> torch.Tensor:
    """Match operation distributions for same-intent forms without aligning binding state."""
    if first_logits.shape != second_logits.shape or first_logits.ndim != 2:
        raise ValueError("Paired operation logits must have identical rank-two shapes")
    first_log_probabilities = F.log_softmax(first_logits, dim=-1)
    second_log_probabilities = F.log_softmax(second_logits, dim=-1)
    return F.mse_loss(first_log_probabilities, second_log_probabilities)


def analyze_bottleneck_case_records(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize case-level layer failures without training or model selection."""
    if not records:
        raise ValueError("Bottleneck case audit requires records")

    def accuracy(cases: Sequence[Mapping[str, object]], prediction: str, expected: str) -> float:
        return sum(case[prediction] == case[expected] for case in cases) / len(cases)

    tracks: dict[str, object] = {}
    for track in sorted({str(record["track"]) for record in records}):
        cases = [record for record in records if record["track"] == track]
        tracks[track] = {
            "n": len(cases),
            "baseline_operation_accuracy": accuracy(cases, "baseline_operation", "expected_operation"),
            "operation_only_operation_accuracy": accuracy(
                cases, "operation_only_operation", "expected_operation"
            ),
            "operation_only_binding_accuracy": accuracy(cases, "operation_only_binding", "expected_binding"),
            "multitask_operation_accuracy": accuracy(cases, "multitask_operation", "expected_operation"),
            "multitask_binding_accuracy": accuracy(cases, "multitask_binding", "expected_binding"),
            "multitask_frozen_probe_binding_accuracy": accuracy(
                cases, "multitask_frozen_probe_binding", "expected_binding"
            ),
        }

    confusion = {expected: {predicted: 0 for predicted in OPERATIONS} for expected in OPERATIONS}
    for record in records:
        expected = str(record["expected_operation"])
        predicted = str(record["multitask_operation"])
        if expected not in confusion or predicted not in confusion[expected]:
            raise ValueError(f"Unsupported operation in audit record: {expected}/{predicted}")
        confusion[expected][predicted] += 1
    baseline_regressions = [
        dict(record)
        for record in records
        if record["baseline_operation"] == record["expected_operation"]
        and record["multitask_operation"] != record["expected_operation"]
    ]
    operation_regressions = [
        dict(record)
        for record in records
        if record["operation_only_operation"] == record["expected_operation"]
        and record["multitask_operation"] != record["expected_operation"]
    ]
    binding_errors = [
        dict(record) for record in records if record["multitask_binding"] != record["expected_binding"]
    ]
    return {
        "record_count": len(records),
        "tracks": tracks,
        "multitask_operation_confusion": confusion,
        "baseline_correct_multitask_wrong": baseline_regressions,
        "operation_only_correct_multitask_wrong": operation_regressions,
        "multitask_binding_errors": binding_errors,
    }


def run_bottleneck_condition(
    splits: ConditionSplits,
    output_dir: Path,
    config: Mapping[str, object],
    *,
    seed: int,
    pressure_groups: Mapping[str, Sequence[FormVariationExample]],
    binding_loss_weight: float,
    arm_name: str,
    run_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Train and persist one paired discriminative-bottleneck condition."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = train_fixed_byte_tokenizer(output_dir / "tokenizer")
    transformer_config = _transformer_config_from(tokenizer, config)
    training = config["training"]
    if not isinstance(training, Mapping):
        raise ValueError("Training configuration must be a mapping")
    device = _device_from_name(str(training.get("device", "auto")))
    model = SemanticBottleneckParser(transformer_config).to(device)
    dataset = SemanticBottleneckDataset(splits.train, tokenizer, transformer_config.max_position_embeddings)
    loader = build_training_loader(
        dataset,
        batch_size=int(training["batch_size"]),
        use_cuda=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    max_steps = int(training["max_steps"])
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=int(training["warmup_steps"]),
        max_steps=max_steps,
        lr=float(training["learning_rate"]),
        min_lr=float(training["min_learning_rate"]),
    )
    timer = StepTimer(str(output_dir), phase_name="semantic_bottleneck")
    model.train()
    optimizer.zero_grad()
    step = 0
    started = time.time()
    gradient_accumulation = int(training.get("gradient_accumulation_steps", 1))
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    while step < max_steps:
        for batch in loader:
            timer.start_step()
            input_ids = batch["input_ids"].to(device, non_blocking=device.type == "cuda")
            boundaries = batch["boundary_index"].to(device, non_blocking=device.type == "cuda")
            operation_labels = batch["operation_label"].to(device, non_blocking=device.type == "cuda")
            binding_labels = batch["binding_label"].to(device, non_blocking=device.type == "cuda")
            if device.type == "cuda":
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16
                ):
                    output = model(
                        input_ids,
                        boundaries,
                        operation_labels=operation_labels,
                        binding_labels=binding_labels,
                        binding_loss_weight=binding_loss_weight,
                    )
            else:
                output = model(
                    input_ids,
                    boundaries,
                    operation_labels=operation_labels,
                    binding_labels=binding_labels,
                    binding_loss_weight=binding_loss_weight,
                )
            loss = output["loss"]
            if loss is None:
                raise RuntimeError("Bottleneck training expected a loss")
            (loss / gradient_accumulation).backward()
            if (step + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            step += 1
            timer.end_step(
                step,
                float(loss.item()),
                scheduler.get_last_lr()[0],
                tokens_processed=input_ids.numel(),
            )
            if step >= max_steps:
                break

    evaluation_batch_size = int(config.get("evaluation", {}).get("batch_size", 64))

    def score(examples: Sequence[FormVariationExample]) -> dict[str, float]:
        return evaluate_bottleneck(
            model,
            tokenizer,
            examples,
            device,
            batch_size=evaluation_batch_size,
        )

    split_details = {
        "seen_form": score(splits.seen_form),
        "unseen_form": score(splits.same_meaning_unseen_form),
        "unseen_operands_seen_form": score(splits.unseen_operands_seen_form),
        "minimal_contrast": score(splits.minimal_contrasts),
    }
    unknown_groups = set(pressure_groups).difference(_PRESSURE_TRACKS)
    if unknown_groups:
        raise ValueError(f"Unknown pressure-test tracks: {sorted(unknown_groups)}")
    pressure_details = {track: score(examples) for track, examples in pressure_groups.items()}
    pressure_operation = {
        track: round(details["operation_accuracy"], 4) for track, details in pressure_details.items()
    }
    metrics: dict[str, object] = {
        "seed": seed,
        "arm": arm_name,
        "binding_loss_weight": binding_loss_weight,
        "train_examples": len(splits.train),
        "steps": step,
        "wall_clock_seconds": round(time.time() - started, 2),
        "split_details": {
            split_name: {metric: round(value, 4) for metric, value in details.items()}
            for split_name, details in split_details.items()
        },
        "pressure_groups": pressure_operation,
        "pressure_group_details": {
            track: {metric: round(value, 4) for metric, value in details.items()}
            for track, details in pressure_details.items()
        },
        "worst_robust_accuracy": round(
            min(pressure_operation[track] for track in _ROBUST_PRESSURE_TRACKS), 4
        ),
        "timer": timer.get_summary(),
    }
    if run_metadata:
        metrics.update(run_metadata)
    timer.export_csv("step_metrics.csv")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    torch.save(
        {"transformer_config": asdict(transformer_config), "model_state_dict": model.state_dict()},
        output_dir / "bottleneck_final.pt",
    )
    return metrics


def run_invariance_condition(
    splits: ConditionSplits,
    pairs: Sequence[tuple[FormVariationExample, FormVariationExample]],
    output_dir: Path,
    config: Mapping[str, object],
    *,
    seed: int,
    pressure_groups: Mapping[str, Sequence[FormVariationExample]],
    binding_loss_weight: float,
    consistency_weight: float,
    pair_batch_size: int,
    arm_name: str,
    run_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Train one fixed-sequence-budget paired consistency condition."""
    if consistency_weight <= 0:
        raise ValueError("Invariance training requires a positive consistency weight")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = train_fixed_byte_tokenizer(output_dir / "tokenizer")
    transformer_config = _transformer_config_from(tokenizer, config)
    training = config["training"]
    if not isinstance(training, Mapping):
        raise ValueError("Training configuration must be a mapping")
    device = _device_from_name(str(training.get("device", "auto")))
    model = SemanticBottleneckParser(transformer_config).to(device)
    dataset = SemanticInvarianceDataset(pairs, tokenizer, transformer_config.max_position_embeddings)
    loader = build_training_loader(dataset, batch_size=pair_batch_size, use_cuda=device.type == "cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    max_steps = int(training["max_steps"])
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=int(training["warmup_steps"]),
        max_steps=max_steps,
        lr=float(training["learning_rate"]),
        min_lr=float(training["min_learning_rate"]),
    )
    timer = StepTimer(str(output_dir), phase_name="semantic_invariance")
    model.train()
    optimizer.zero_grad()
    step = 0
    started = time.time()
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    latest_consistency = 0.0
    while step < max_steps:
        for batch in loader:
            timer.start_step()

            def forward(prefix: str) -> dict[str, torch.Tensor | None]:
                return model(
                    batch[f"{prefix}_input_ids"].to(device, non_blocking=device.type == "cuda"),
                    batch[f"{prefix}_boundary_index"].to(
                        device, non_blocking=device.type == "cuda"
                    ),
                    operation_labels=batch[f"{prefix}_operation_label"].to(
                        device, non_blocking=device.type == "cuda"
                    ),
                    binding_labels=batch[f"{prefix}_binding_label"].to(
                        device, non_blocking=device.type == "cuda"
                    ),
                    binding_loss_weight=binding_loss_weight,
                )

            if device.type == "cuda":
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16
                ):
                    first_output = forward("first")
                    second_output = forward("second")
                    first_loss = first_output["loss"]
                    second_loss = second_output["loss"]
                    if first_loss is None or second_loss is None:
                        raise RuntimeError("Invariance training expected supervised losses")
                    consistency = paired_operation_consistency_loss(
                        first_output["operation_logits"], second_output["operation_logits"]
                    )
                    loss = 0.5 * (first_loss + second_loss) + consistency_weight * consistency
            else:
                first_output = forward("first")
                second_output = forward("second")
                first_loss = first_output["loss"]
                second_loss = second_output["loss"]
                if first_loss is None or second_loss is None:
                    raise RuntimeError("Invariance training expected supervised losses")
                consistency = paired_operation_consistency_loss(
                    first_output["operation_logits"], second_output["operation_logits"]
                )
                loss = 0.5 * (first_loss + second_loss) + consistency_weight * consistency
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            latest_consistency = float(consistency.item())
            sequence_tokens = batch["first_input_ids"].numel() + batch["second_input_ids"].numel()
            timer.end_step(
                step,
                float(loss.item()),
                scheduler.get_last_lr()[0],
                tokens_processed=sequence_tokens,
            )
            if step >= max_steps:
                break

    evaluation_batch_size = int(config.get("evaluation", {}).get("batch_size", 64))

    def score(examples: Sequence[FormVariationExample]) -> dict[str, float]:
        return evaluate_bottleneck(
            model, tokenizer, examples, device, batch_size=evaluation_batch_size
        )

    split_details = {
        "seen_form": score(splits.seen_form),
        "unseen_form": score(splits.same_meaning_unseen_form),
        "unseen_operands_seen_form": score(splits.unseen_operands_seen_form),
        "minimal_contrast": score(splits.minimal_contrasts),
    }
    pressure_details = {track: score(examples) for track, examples in pressure_groups.items()}
    pressure_operation = {
        track: round(details["operation_accuracy"], 4) for track, details in pressure_details.items()
    }
    metrics: dict[str, object] = {
        "seed": seed,
        "arm": arm_name,
        "binding_loss_weight": binding_loss_weight,
        "consistency_weight": consistency_weight,
        "pair_count": len(pairs),
        "pair_batch_size": pair_batch_size,
        "sequences_per_step": pair_batch_size * 2,
        "train_examples": len(splits.train),
        "steps": step,
        "wall_clock_seconds": round(time.time() - started, 2),
        "final_consistency_loss": round(latest_consistency, 6),
        "split_details": {
            name: {metric: round(value, 4) for metric, value in details.items()}
            for name, details in split_details.items()
        },
        "pressure_groups": pressure_operation,
        "pressure_group_details": {
            track: {metric: round(value, 4) for metric, value in details.items()}
            for track, details in pressure_details.items()
        },
        "worst_robust_accuracy": round(
            min(pressure_operation[track] for track in _ROBUST_PRESSURE_TRACKS), 4
        ),
        "timer": timer.get_summary(),
    }
    if run_metadata:
        metrics.update(run_metadata)
    timer.export_csv("step_metrics.csv")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    torch.save(
        {"transformer_config": asdict(transformer_config), "model_state_dict": model.state_dict()},
        output_dir / "bottleneck_final.pt",
    )
    return metrics


def _bottleneck_training_fingerprint(
    examples: Sequence[FormVariationExample],
    config: Mapping[str, object],
    schedule: Mapping[str, int],
    *,
    binding_loss_weight: float,
    objective: str = "discriminative",
) -> str:
    payload = {
        "examples": [
            {
                "utterance": example.utterance,
                "operation": example.operation,
                "left": example.left,
                "right": example.right,
                "template_id": example.template_id,
            }
            for example in examples
        ],
        "tokenizer": config["tokenizer"],
        "model": config["model"],
        "schedule": dict(schedule),
        "objective": {
            "kind": objective,
            "operation_classes": list(OPERATIONS),
            "binding_classes": ["A_FIRST", "B_FIRST"],
            "binding_loss_weight": binding_loss_weight,
            "target_serialization": objective == "generative_operation",
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stagec_training_fingerprint(
    pairs: Sequence[tuple[FormVariationExample, FormVariationExample]],
    config: Mapping[str, object],
    schedule: Mapping[str, int],
    *,
    binding_loss_weight: float,
    consistency_weight: float,
) -> str:
    payload = {
        "pairs": [
            [
                {
                    "utterance": example.utterance,
                    "operation": example.operation,
                    "left": example.left,
                    "right": example.right,
                    "template_id": example.template_id,
                }
                for example in pair
            ]
            for pair in pairs
        ],
        "tokenizer": config["tokenizer"],
        "model": config["model"],
        "schedule": dict(schedule),
        "objective": {
            "kind": "paired_operation_log_probability_mse",
            "operation_classes": list(OPERATIONS),
            "binding_classes": ["A_FIRST", "B_FIRST"],
            "binding_loss_weight": binding_loss_weight,
            "consistency_weight": consistency_weight,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prepare_bottleneck_screen(config_path: str) -> dict[str, object]:
    """Validate and register Phase 4 without creating any training artifacts."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "phase4_bottleneck" not in config:
        raise ValueError("Configuration is missing phase4_bottleneck")
    experiment = config["phase4_bottleneck"]
    if not isinstance(experiment, Mapping):
        raise ValueError("phase4_bottleneck must be a mapping")
    if config["tokenizer"].get("mode") != "fixed_byte":
        raise ValueError("Phase 4 requires the registered fixed-byte tokenizer")
    if experiment.get("binding_counterbalance") is not True:
        raise ValueError("Phase 4 requires the registered binding counterbalance")

    train_pairs = tuple(tuple(pair) for pair in experiment["train_pairs"])
    eval_pairs = tuple(tuple(pair) for pair in experiment["eval_pairs"])
    pressure_pairs = tuple(tuple(pair) for pair in experiment["pressure_pairs"])
    variants = int(experiment["variants_per_operation"])
    original_splits = build_condition_splits(variants, train_pairs, eval_pairs, seed=int(config["seed"]))
    splits = ConditionSplits(
        train=build_binding_counterbalanced_examples(original_splits.train),
        seen_form=original_splits.seen_form,
        same_meaning_unseen_form=original_splits.same_meaning_unseen_form,
        unseen_operands_seen_form=original_splits.unseen_operands_seen_form,
        minimal_contrasts=original_splits.minimal_contrasts,
    )
    pressure_groups = build_fixed_pressure_test(pressure_pairs)
    split_manifest = build_clean_split_manifest(splits.train, pressure_groups)
    if split_manifest["validation"] != "PASS":
        raise ValueError("Phase 4 split manifest failed validation")

    training = config["training"]
    batch_size = int(training["batch_size"])
    max_steps = steps_for_exposure(len(splits.train), batch_size, float(experiment["exposure"]))
    warmup_steps = warmup_steps_for_training(max_steps)
    schedule = {"batch_size": batch_size, "max_steps": max_steps, "warmup_steps": warmup_steps}
    if int(training["max_steps"]) != max_steps or int(training["warmup_steps"]) != warmup_steps:
        raise ValueError(f"Configured Phase 4 schedule does not match derived schedule {schedule}")

    source_path = Path(str(experiment["source_results"]))
    with source_path.open("r", encoding="utf-8") as handle:
        source_report = json.load(handle)
    source_variant = str(int(experiment["source_variants"]))
    source_exposure = str(experiment["source_exposure"])
    source_manifest = source_report["split_manifests"][source_variant]
    historical_manifest = build_clean_split_manifest(original_splits.train, pressure_groups)
    if source_manifest != historical_manifest:
        raise ValueError("Historical source split manifest does not match the registered reference cell")
    historical_runs = source_report["runs"][source_variant][source_exposure]
    expected_seeds = tuple(int(seed) for seed in experiment["seeds"])
    actual_seeds = tuple(sorted(int(run["seed"]) for run in historical_runs))
    if actual_seeds != tuple(sorted(expected_seeds)):
        raise ValueError("Reused source seeds do not match Phase 4")
    for run in historical_runs:
        if (
            int(run["steps"]) != max_steps
            or int(run["train_examples"]) != len(splits.train)
            or run.get("tokenizer_mode") != "fixed_byte"
        ):
            raise ValueError(f"Reused source run {run['seed']} has an incompatible schedule or tokenizer")

    binding_weight = float(experiment["binding_loss_weight"])
    arm_fingerprints = {
        "matched_generative_baseline": _bottleneck_training_fingerprint(
            splits.train,
            config,
            schedule,
            binding_loss_weight=0.0,
            objective="generative_operation",
        ),
        "discriminative_operation": _bottleneck_training_fingerprint(
            splits.train, config, schedule, binding_loss_weight=0.0
        ),
        "discriminative_operation_binding": _bottleneck_training_fingerprint(
            splits.train, config, schedule, binding_loss_weight=binding_weight
        ),
    }
    binding_counts = {"A_FIRST": 0, "B_FIRST": 0}
    for example in splits.train:
        label = semantic_binding_label(example)
        binding_counts["B_FIRST" if label else "A_FIRST"] += 1
    registration = {
        "experiment": str(config.get("name", "phase4_bottleneck")),
        "config": config_path,
        "source_results": str(source_path),
        "source_cell": f"K{source_variant}/{source_exposure}",
        "historical_reference_validation": "PASS",
        "seeds": list(expected_seeds),
        "schedule": schedule,
        "train_examples": len(splits.train),
        "binding_label_counts": binding_counts,
        "arm_fingerprints": arm_fingerprints,
        "gate": dict(experiment["gate"]),
        "training_started": False,
    }
    output_dir = Path(str(experiment["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "registration.json").open("w", encoding="utf-8") as handle:
        json.dump(registration, handle, indent=2)
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2)
    return {
        "config": config,
        "experiment": experiment,
        "splits": splits,
        "pressure_groups": pressure_groups,
        "historical_runs": historical_runs,
        "registration": registration,
    }


def run_bottleneck_screen(config_path: str) -> dict[str, object]:
    """Train both registered discriminative arms and apply the screen gate."""
    preparation = prepare_bottleneck_screen(config_path)
    config = preparation["config"]
    experiment = preparation["experiment"]
    splits = preparation["splits"]
    pressure_groups = preparation["pressure_groups"]
    registration = preparation["registration"]
    if not isinstance(config, Mapping) or not isinstance(experiment, Mapping):
        raise ValueError("Invalid Phase 4 preparation payload")
    if not isinstance(splits, ConditionSplits) or not isinstance(pressure_groups, Mapping):
        raise ValueError("Invalid Phase 4 split payload")
    if not isinstance(registration, Mapping):
        raise ValueError("Invalid Phase 4 registration payload")

    output_dir = Path(str(experiment["output_dir"]))
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    weights = {
        "discriminative_operation": 0.0,
        "discriminative_operation_binding": float(experiment["binding_loss_weight"]),
    }
    arms: dict[str, Sequence[Mapping[str, object]]] = {
        "historical_generative_reference": preparation["historical_runs"],
    }
    fingerprints = registration["arm_fingerprints"]
    if not isinstance(fingerprints, Mapping):
        raise ValueError("Phase 4 registration is missing arm fingerprints")
    generative_name = "matched_generative_baseline"
    generative_fingerprint = str(fingerprints[generative_name])
    generative_runs: list[Mapping[str, object]] = []
    for seed in seeds:
        run_dir = output_dir / generative_name / f"seed_{seed}"
        metrics_path = run_dir / "metrics.json"
        checkpoint_path = run_dir / "parser_final.pt"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            if metrics.get("phase4_training_fingerprint") != generative_fingerprint or not checkpoint_path.exists():
                raise ValueError(f"Existing Phase 4 artifact is incompatible: {run_dir}")
        else:
            metrics = run_condition(
                splits,
                run_dir,
                dict(config),
                seed,
                pressure_groups=pressure_groups,
                run_metadata={
                    "arm": generative_name,
                    "phase4_training_fingerprint": generative_fingerprint,
                    "variants_per_operation": int(experiment["variants_per_operation"]),
                    "exposure_target": float(experiment["exposure"]),
                    "tokenizer_mode": "fixed_byte",
                },
            )
        generative_runs.append(metrics)
    arms[generative_name] = generative_runs

    for arm_name, binding_weight in weights.items():
        arm_runs: list[Mapping[str, object]] = []
        fingerprint = str(fingerprints[arm_name])
        for seed in seeds:
            run_dir = output_dir / arm_name / f"seed_{seed}"
            metrics_path = run_dir / "metrics.json"
            checkpoint_path = run_dir / "bottleneck_final.pt"
            if metrics_path.exists():
                with metrics_path.open("r", encoding="utf-8") as handle:
                    metrics = json.load(handle)
                if metrics.get("phase4_training_fingerprint") != fingerprint or not checkpoint_path.exists():
                    raise ValueError(f"Existing Phase 4 artifact is incompatible: {run_dir}")
            else:
                metrics = run_bottleneck_condition(
                    splits,
                    run_dir,
                    config,
                    seed=seed,
                    pressure_groups=pressure_groups,
                    binding_loss_weight=binding_weight,
                    arm_name=arm_name,
                    run_metadata={
                        "phase4_training_fingerprint": fingerprint,
                        "variants_per_operation": int(experiment["variants_per_operation"]),
                        "exposure_target": float(experiment["exposure"]),
                        "tokenizer_mode": "fixed_byte",
                    },
                )
            arm_runs.append(metrics)
        arms[arm_name] = arm_runs

    report = {
        "experiment": str(config.get("name", "phase4_bottleneck")),
        "config": config_path,
        "registration": registration,
        "baseline_arm": generative_name,
        "gate": dict(experiment["gate"]),
        "arms": arms,
    }
    analysis = analyze_bottleneck_screen(report)
    with (output_dir / "screen_results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "screen_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    return {"report": report, "analysis": analysis}


def _splits_with_training(
    splits: ConditionSplits, training_examples: Sequence[FormVariationExample]
) -> ConditionSplits:
    return ConditionSplits(
        train=tuple(training_examples),
        seen_form=splits.seen_form,
        same_meaning_unseen_form=splits.same_meaning_unseen_form,
        unseen_operands_seen_form=splits.unseen_operands_seen_form,
        minimal_contrasts=splits.minimal_contrasts,
    )


def _binding_counts(examples: Sequence[FormVariationExample]) -> dict[str, int]:
    counts = {"A_FIRST": 0, "B_FIRST": 0}
    for example in examples:
        counts["B_FIRST" if semantic_binding_label(example) else "A_FIRST"] += 1
    return counts


def prepare_phase4_stageb(config_path: str) -> dict[str, object]:
    """Register the Stage-B factorial and validate every reused source artifact."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "phase4_stageb" not in config:
        raise ValueError("Configuration is missing phase4_stageb")
    experiment = config["phase4_stageb"]
    if not isinstance(experiment, Mapping):
        raise ValueError("phase4_stageb must be a mapping")
    if config["tokenizer"].get("mode") != "fixed_byte":
        raise ValueError("Phase 4 Stage B requires the fixed-byte tokenizer")

    train_pairs = tuple(tuple(int(value) for value in pair) for pair in experiment["train_pairs"])
    eval_pairs = tuple(tuple(int(value) for value in pair) for pair in experiment["eval_pairs"])
    development_pairs = tuple(
        tuple(int(value) for value in pair) for pair in experiment["development_pairs"]
    )
    if set(train_pairs + eval_pairs).intersection(development_pairs):
        raise ValueError("Stage-B development operands must be disjoint from training/evaluation operands")
    variants = int(experiment["variants_per_operation"])
    base_splits = build_condition_splits(variants, train_pairs, eval_pairs, seed=int(config["seed"]))
    ordinary_training = build_binding_counterbalanced_examples(base_splits.train)
    scaffold_training = build_scaffold_counterbalanced_examples(
        ordinary_training,
        replacement_fraction=float(experiment["scaffold_replacement_fraction"]),
        seed=int(config["seed"]),
    )
    ordinary_splits = _splits_with_training(base_splits, ordinary_training)
    scaffold_splits = _splits_with_training(base_splits, scaffold_training)
    development = build_phase4_stageb_development(development_pairs)
    manifests = {
        "ordinary": build_phase25_split_manifest(ordinary_training, development),
        "scaffold": build_phase25_split_manifest(scaffold_training, development),
    }
    if any(manifest["validation"] != "PASS" for manifest in manifests.values()):
        raise ValueError("Phase 4 Stage-B train/development isolation failed")
    if len(ordinary_training) != len(scaffold_training):
        raise ValueError("Stage-B scaffold factor changed the training budget")
    if _binding_counts(ordinary_training) != _binding_counts(scaffold_training):
        raise ValueError("Stage-B scaffold factor changed global binding balance")

    training = config["training"]
    batch_size = int(training["batch_size"])
    max_steps = steps_for_exposure(
        len(ordinary_training), batch_size, float(experiment["exposure"])
    )
    warmup_steps = warmup_steps_for_training(max_steps)
    schedule = {"batch_size": batch_size, "max_steps": max_steps, "warmup_steps": warmup_steps}
    if int(training["max_steps"]) != max_steps or int(training["warmup_steps"]) != warmup_steps:
        raise ValueError(f"Configured Stage-B schedule does not match derived schedule {schedule}")

    full_weight = float(experiment["full_binding_loss_weight"])
    reduced_weight = float(experiment["reduced_binding_loss_weight"])
    if not 0 <= reduced_weight < full_weight:
        raise ValueError("Stage-B requires 0 <= reduced binding weight < full binding weight")
    fingerprints = {
        "matched_generative_baseline": _bottleneck_training_fingerprint(
            ordinary_training,
            config,
            schedule,
            binding_loss_weight=0.0,
            objective="generative_operation",
        ),
        "weight_1_standard": _bottleneck_training_fingerprint(
            ordinary_training, config, schedule, binding_loss_weight=full_weight
        ),
        "weight_1_scaffold": _bottleneck_training_fingerprint(
            scaffold_training, config, schedule, binding_loss_weight=full_weight
        ),
        "weight_025_standard": _bottleneck_training_fingerprint(
            ordinary_training, config, schedule, binding_loss_weight=reduced_weight
        ),
        "weight_025_scaffold": _bottleneck_training_fingerprint(
            scaffold_training, config, schedule, binding_loss_weight=reduced_weight
        ),
    }

    stagea_dir = Path(str(experiment["stagea_output_dir"]))
    with (stagea_dir / "registration.json").open("r", encoding="utf-8") as handle:
        stagea_registration = json.load(handle)
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    if tuple(sorted(int(seed) for seed in stagea_registration["seeds"])) != tuple(sorted(seeds)):
        raise ValueError("Stage-A and Stage-B seeds differ")
    if stagea_registration["schedule"] != schedule:
        raise ValueError("Stage-A and Stage-B schedules differ")
    source_fingerprints = stagea_registration["arm_fingerprints"]
    expected_source = {
        "matched_generative_baseline": fingerprints["matched_generative_baseline"],
        "discriminative_operation_binding": fingerprints["weight_1_standard"],
    }
    for source_name, fingerprint in expected_source.items():
        if source_fingerprints.get(source_name) != fingerprint:
            raise ValueError(f"Stage-A fingerprint mismatch for {source_name}")
        checkpoint_name = (
            "parser_final.pt" if source_name == "matched_generative_baseline" else "bottleneck_final.pt"
        )
        for seed in seeds:
            run_dir = stagea_dir / source_name / f"seed_{seed}"
            with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            if metrics.get("phase4_training_fingerprint") != fingerprint:
                raise ValueError(f"Stage-A metrics fingerprint mismatch: {run_dir}")
            if not (run_dir / checkpoint_name).exists():
                raise ValueError(f"Stage-A checkpoint missing: {run_dir / checkpoint_name}")

    arm_registration = {
        "matched_generative_baseline": {"requires_training": False, "source_arm": "matched_generative_baseline"},
        "weight_1_standard": {"requires_training": False, "source_arm": "discriminative_operation_binding"},
        "weight_1_scaffold": {"requires_training": True, "binding_loss_weight": full_weight, "corpus": "scaffold"},
        "weight_025_standard": {"requires_training": True, "binding_loss_weight": reduced_weight, "corpus": "ordinary"},
        "weight_025_scaffold": {"requires_training": True, "binding_loss_weight": reduced_weight, "corpus": "scaffold"},
    }
    registration = {
        "experiment": str(config.get("name", "phase4_stageb")),
        "config": config_path,
        "stagea_output_dir": str(stagea_dir),
        "source_validation": "PASS",
        "seeds": list(seeds),
        "schedule": schedule,
        "train_examples": len(ordinary_training),
        "development_examples": sum(len(examples) for examples in development.values()),
        "binding_label_counts": {
            "ordinary": _binding_counts(ordinary_training),
            "scaffold": _binding_counts(scaffold_training),
        },
        "arm_fingerprints": fingerprints,
        "arms": arm_registration,
        "gate": dict(experiment["gate"]),
        "sealed_suite_created": False,
    }
    output_dir = Path(str(experiment["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "registration.json").open("w", encoding="utf-8") as handle:
        json.dump(registration, handle, indent=2)
    with (output_dir / "split_manifests.json").open("w", encoding="utf-8") as handle:
        json.dump(manifests, handle, indent=2)
    with (output_dir / "development_suite.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {track: [asdict(example) for example in examples] for track, examples in development.items()},
            handle,
            indent=2,
        )
    return {
        "config": config,
        "experiment": experiment,
        "ordinary_splits": ordinary_splits,
        "scaffold_splits": scaffold_splits,
        "development": development,
        "registration": registration,
    }


def prepare_phase4_stagec(config_path: str) -> dict[str, object]:
    """Register Stage C and prove source, pair-budget, and suite isolation invariants."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "phase4_stagec" not in config:
        raise ValueError("Configuration is missing phase4_stagec")
    experiment = config["phase4_stagec"]
    if not isinstance(experiment, Mapping):
        raise ValueError("phase4_stagec must be a mapping")
    if config["tokenizer"].get("mode") != "fixed_byte":
        raise ValueError("Phase 4 Stage C requires the fixed-byte tokenizer")
    train_pairs = tuple(tuple(int(value) for value in pair) for pair in experiment["train_pairs"])
    eval_pairs = tuple(tuple(int(value) for value in pair) for pair in experiment["eval_pairs"])
    development_pairs = tuple(
        tuple(int(value) for value in pair) for pair in experiment["development_pairs"]
    )
    if set(train_pairs + eval_pairs).intersection(development_pairs):
        raise ValueError("Stage-C development operands must be disjoint")
    variants = int(experiment["variants_per_operation"])
    base_splits = build_condition_splits(variants, train_pairs, eval_pairs, seed=int(config["seed"]))
    ordinary_training = build_binding_counterbalanced_examples(base_splits.train)
    scaffold_training = build_scaffold_counterbalanced_examples(
        ordinary_training,
        replacement_fraction=float(experiment["scaffold_replacement_fraction"]),
        seed=int(config["seed"]),
    )
    scaffold_splits = _splits_with_training(base_splits, scaffold_training)
    invariance_pairs = build_stagec_invariance_pairs(scaffold_training, seed=int(config["seed"]))
    if len(invariance_pairs) * 2 != len(scaffold_training):
        raise ValueError("Stage-C pairing changed the sequence budget")
    development = build_phase4_stagec_development(development_pairs)
    split_manifest = build_phase25_split_manifest(scaffold_training, development)
    if split_manifest["validation"] != "PASS":
        raise ValueError("Stage-C train/development isolation failed")
    stageb_dir = Path(str(experiment["stageb_output_dir"]))
    with (stageb_dir / "development_suite.json").open("r", encoding="utf-8") as handle:
        stageb_development = json.load(handle)
    old_utterances = {
        str(example["utterance"])
        for examples in stageb_development.values()
        for example in examples
    }
    new_utterances = {
        example.utterance for examples in development.values() for example in examples
    }
    if old_utterances.intersection(new_utterances):
        raise ValueError("Stage-C development text overlaps Stage B")

    training = config["training"]
    sequence_batch_size = int(training["batch_size"])
    pair_batch_size = int(experiment["pair_batch_size"])
    if pair_batch_size * 2 != sequence_batch_size:
        raise ValueError("Stage-C pair batch must preserve sequences per update")
    max_steps = steps_for_exposure(
        len(invariance_pairs), pair_batch_size, float(experiment["exposure"])
    )
    warmup_steps = warmup_steps_for_training(max_steps)
    base_schedule = {
        "batch_size": sequence_batch_size,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
    }
    pair_schedule = {
        "pair_batch_size": pair_batch_size,
        "sequences_per_step": pair_batch_size * 2,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
    }
    if int(training["max_steps"]) != max_steps or int(training["warmup_steps"]) != warmup_steps:
        raise ValueError("Configured Stage-C schedule does not match derived schedule")
    binding_weight = float(experiment["binding_loss_weight"])
    consistency_weights = tuple(float(weight) for weight in experiment["consistency_weights"])
    if consistency_weights != (0.25, 1.0):
        raise ValueError("Stage-C consistency weights differ from the registration")

    source_fingerprints = {
        "matched_generative_baseline": _bottleneck_training_fingerprint(
            ordinary_training,
            config,
            base_schedule,
            binding_loss_weight=0.0,
            objective="generative_operation",
        ),
        "lead_no_consistency": _bottleneck_training_fingerprint(
            scaffold_training, config, base_schedule, binding_loss_weight=binding_weight
        ),
    }
    stagea_dir = Path(str(experiment["stagea_output_dir"]))
    with (stagea_dir / "registration.json").open("r", encoding="utf-8") as handle:
        stagea_registration = json.load(handle)
    with (stageb_dir / "registration.json").open("r", encoding="utf-8") as handle:
        stageb_registration = json.load(handle)
    if (
        stagea_registration["arm_fingerprints"].get("matched_generative_baseline")
        != source_fingerprints["matched_generative_baseline"]
    ):
        raise ValueError("Stage-C generative source fingerprint mismatch")
    if (
        stageb_registration["arm_fingerprints"].get("weight_025_scaffold")
        != source_fingerprints["lead_no_consistency"]
    ):
        raise ValueError("Stage-C lead source fingerprint mismatch")
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    for seed in seeds:
        source_checks = (
            (
                stagea_dir / "matched_generative_baseline" / f"seed_{seed}",
                "parser_final.pt",
                "phase4_training_fingerprint",
                source_fingerprints["matched_generative_baseline"],
            ),
            (
                stageb_dir / "weight_025_scaffold" / f"seed_{seed}",
                "bottleneck_final.pt",
                "phase4_stageb_training_fingerprint",
                source_fingerprints["lead_no_consistency"],
            ),
        )
        for run_dir, checkpoint_name, fingerprint_key, expected_fingerprint in source_checks:
            with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            if metrics.get(fingerprint_key) != expected_fingerprint:
                raise ValueError(f"Stage-C source metrics mismatch: {run_dir}")
            if not (run_dir / checkpoint_name).exists():
                raise ValueError(f"Stage-C source checkpoint missing: {run_dir / checkpoint_name}")

    arm_fingerprints = {
        **source_fingerprints,
        "consistency_025": _stagec_training_fingerprint(
            invariance_pairs,
            config,
            pair_schedule,
            binding_loss_weight=binding_weight,
            consistency_weight=consistency_weights[0],
        ),
        "consistency_1": _stagec_training_fingerprint(
            invariance_pairs,
            config,
            pair_schedule,
            binding_loss_weight=binding_weight,
            consistency_weight=consistency_weights[1],
        ),
    }
    pairing_payload = [
        [
            {
                "operation": example.operation,
                "left": example.left,
                "right": example.right,
                "template_id": example.template_id,
                "binding": semantic_binding_label(example),
                "utterance": example.utterance,
            }
            for example in pair
        ]
        for pair in invariance_pairs
    ]
    pairing_hash = hashlib.sha256(
        json.dumps(pairing_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    registration = {
        "experiment": str(config.get("name", "phase4_stagec")),
        "config": config_path,
        "source_validation": "PASS",
        "seeds": list(seeds),
        "base_schedule": base_schedule,
        "pair_schedule": pair_schedule,
        "train_examples": len(scaffold_training),
        "pair_count": len(invariance_pairs),
        "pairing_sha256": pairing_hash,
        "development_examples": sum(len(examples) for examples in development.values()),
        "arm_fingerprints": arm_fingerprints,
        "gate": dict(experiment["gate"]),
        "sealed_suite_created": False,
    }
    output_dir = Path(str(experiment["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "registration.json").open("w", encoding="utf-8") as handle:
        json.dump(registration, handle, indent=2)
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2)
    with (output_dir / "pair_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"sha256": pairing_hash, "pairs": pairing_payload}, handle, indent=2)
    with (output_dir / "development_suite.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {track: [asdict(example) for example in examples] for track, examples in development.items()},
            handle,
            indent=2,
        )
    return {
        "config": config,
        "experiment": experiment,
        "splits": scaffold_splits,
        "pairs": invariance_pairs,
        "development": development,
        "registration": registration,
    }


def prepare_phase4_stagec_confirmation(config_path: str) -> dict[str, object]:
    """Create the sealed suite only after Stage C selects a passing arm."""
    preparation = prepare_phase4_stagec(config_path)
    config = preparation["config"]
    experiment = preparation["experiment"]
    splits = preparation["splits"]
    development = preparation["development"]
    stagec_registration = preparation["registration"]
    if not isinstance(config, Mapping) or not isinstance(experiment, Mapping):
        raise ValueError("Invalid Stage-C confirmation configuration")
    if not isinstance(splits, ConditionSplits) or not isinstance(development, Mapping):
        raise ValueError("Invalid Stage-C confirmation split payload")
    if not isinstance(stagec_registration, Mapping):
        raise ValueError("Invalid Stage-C confirmation registration")

    output_dir = Path(str(experiment["output_dir"]))
    with (output_dir / "screen_analysis.json").open("r", encoding="utf-8") as handle:
        screen_analysis = json.load(handle)
    selected_arm = screen_analysis.get("selected_arm")
    if selected_arm != "consistency_025" or not screen_analysis.get(
        "sealed_confirmation_required"
    ):
        raise ValueError("Stage C did not select the registered confirmation candidate")

    confirmation_pairs = tuple(
        tuple(int(value) for value in pair) for pair in experiment["confirmation_pairs"]
    )
    earlier_pairs = tuple(
        tuple(int(value) for value in pair)
        for key in ("train_pairs", "eval_pairs", "development_pairs")
        for pair in experiment[key]
    )
    if set(confirmation_pairs).intersection(earlier_pairs):
        raise ValueError("Stage-C confirmation operands overlap an earlier split")
    confirmation = build_phase4_stagec_confirmation(confirmation_pairs)
    split_manifest = build_phase25_split_manifest(
        splits.train,
        development,
        sealed_groups=confirmation,
    )
    if split_manifest["validation"] != "PASS":
        raise ValueError("Stage-C sealed split isolation failed")

    prior_utterances = {
        example.utterance for examples in development.values() for example in examples
    }
    stageb_suite_path = Path(str(experiment["stageb_output_dir"])) / "development_suite.json"
    if stageb_suite_path.exists():
        with stageb_suite_path.open("r", encoding="utf-8") as handle:
            stageb_suite = json.load(handle)
        prior_utterances.update(
            str(example["utterance"])
            for examples in stageb_suite.values()
            for example in examples
        )
    sealed_utterances = {
        example.utterance for examples in confirmation.values() for example in examples
    }
    if prior_utterances.intersection(sealed_utterances):
        raise ValueError("Stage-C confirmation text overlaps an earlier development suite")

    fingerprints = stagec_registration["arm_fingerprints"]
    if not isinstance(fingerprints, Mapping):
        raise ValueError("Stage-C confirmation source fingerprints are missing")
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    stagea_dir = Path(str(experiment["stagea_output_dir"]))
    stageb_dir = Path(str(experiment["stageb_output_dir"]))
    source_specs = {
        "matched_generative_baseline": (
            stagea_dir / "matched_generative_baseline",
            "parser_final.pt",
            "phase4_training_fingerprint",
        ),
        "lead_no_consistency": (
            stageb_dir / "weight_025_scaffold",
            "bottleneck_final.pt",
            "phase4_stageb_training_fingerprint",
        ),
        "consistency_025": (
            output_dir / "consistency_025",
            "bottleneck_final.pt",
            "phase4_stagec_training_fingerprint",
        ),
    }
    for arm_name, (arm_dir, checkpoint_name, fingerprint_key) in source_specs.items():
        expected_fingerprint = fingerprints[arm_name]
        for seed in seeds:
            run_dir = arm_dir / f"seed_{seed}"
            with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            if metrics.get(fingerprint_key) != expected_fingerprint:
                raise ValueError(f"Stage-C confirmation source mismatch: {run_dir}")
            if not (run_dir / checkpoint_name).exists():
                raise ValueError(f"Stage-C confirmation checkpoint missing: {run_dir}")

    suite_payload = {
        track: [asdict(example) for example in examples]
        for track, examples in confirmation.items()
    }
    suite_hash = hashlib.sha256(
        json.dumps(suite_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    registration = {
        "experiment": f"{config.get('name', 'phase4_stagec')}_sealed_confirmation",
        "config": config_path,
        "selected_arm": selected_arm,
        "baseline_arm": "matched_generative_baseline",
        "lead_arm": "lead_no_consistency",
        "seeds": list(seeds),
        "confirmation_pairs": [list(pair) for pair in confirmation_pairs],
        "sealed_examples": sum(len(examples) for examples in confirmation.values()),
        "cases_per_track": {
            track: len(examples) for track, examples in confirmation.items()
        },
        "both_orders_per_frame": True,
        "suite_sha256": suite_hash,
        "source_fingerprints": {
            arm_name: fingerprints[arm_name] for arm_name in source_specs
        },
        "gate": dict(experiment["gate"]),
        "sealed_suite_created": True,
        "evaluation_started": False,
    }
    confirmation_dir = output_dir / "sealed_confirmation"
    confirmation_dir.mkdir(parents=True, exist_ok=True)
    with (confirmation_dir / "registration.json").open("w", encoding="utf-8") as handle:
        json.dump(registration, handle, indent=2)
    with (confirmation_dir / "suite.json").open("w", encoding="utf-8") as handle:
        json.dump(suite_payload, handle, indent=2)
    with (confirmation_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2)
    return {
        "config": config,
        "experiment": experiment,
        "confirmation": confirmation,
        "registration": registration,
    }


def _load_bottleneck_run(run_dir: Path, device: torch.device) -> tuple[SemanticBottleneckParser, Tokenizer]:
    tokenizer = Tokenizer.from_file(str(run_dir / "tokenizer" / "tokenizer.json"))
    payload = torch.load(run_dir / "bottleneck_final.pt", map_location=device, weights_only=True)
    transformer_config = TransformerConfig(**payload["transformer_config"])
    model = SemanticBottleneckParser(transformer_config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, tokenizer


def _load_generative_run(
    run_dir: Path, config: Mapping[str, object], device: torch.device
) -> tuple[SyntheticTransformer, Tokenizer]:
    tokenizer = Tokenizer.from_file(str(run_dir / "tokenizer" / "tokenizer.json"))
    transformer_config = _transformer_config_from(tokenizer, config)
    model = SyntheticTransformer(transformer_config).to(device)
    state_dict = torch.load(run_dir / "parser_final.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer


def _evaluate_stageb_generative_source(
    run_dir: Path,
    config: Mapping[str, object],
    development: Mapping[str, Sequence[FormVariationExample]],
    *,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model, tokenizer = _load_generative_run(run_dir, config, device)
    group_scores: dict[str, float] = {}
    records: list[dict[str, object]] = []
    for track, raw_examples in development.items():
        examples = tuple(raw_examples)
        raw_predictions = predict_operation_labels_batched(
            model, tokenizer, examples, device, batch_size=batch_size
        )
        predictions = tuple(prediction.removeprefix("OP=") for prediction in raw_predictions)
        score = sum(
            prediction == example.operation for prediction, example in zip(predictions, examples)
        ) / len(examples)
        group_scores[track] = round(score, 4)
        records.extend(
            {
                "seed": seed,
                "arm": "matched_generative_baseline",
                "track": track,
                "utterance": example.utterance,
                "template_id": example.template_id,
                "left": example.left,
                "right": example.right,
                "expected_operation": example.operation,
                "expected_binding": semantic_binding_label(example),
                "predicted_operation": prediction,
                "predicted_binding": None,
            }
            for example, prediction in zip(examples, predictions)
        )
    del model
    metrics: dict[str, object] = {
        "seed": seed,
        "arm": "matched_generative_baseline",
        "evaluation_only": True,
        "source_checkpoint": str(run_dir / "parser_final.pt"),
        "pressure_groups": group_scores,
        "worst_robust_accuracy": round(
            min(group_scores[track] for track in _ROBUST_PRESSURE_TRACKS), 4
        ),
    }
    return metrics, records


def _evaluate_stageb_bottleneck_source(
    run_dir: Path,
    development: Mapping[str, Sequence[FormVariationExample]],
    *,
    seed: int,
    arm_name: str,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model, tokenizer = _load_bottleneck_run(run_dir, device)
    details: dict[str, dict[str, float]] = {}
    records: list[dict[str, object]] = []
    for track, raw_examples in development.items():
        examples = tuple(raw_examples)
        operation_predictions, binding_predictions = predict_bottleneck_labels(
            model, tokenizer, examples, device, batch_size=batch_size
        )
        details[track] = {
            metric: round(value, 4)
            for metric, value in score_bottleneck_predictions(
                examples, operation_predictions, binding_predictions
            ).items()
        }
        records.extend(
            {
                "seed": seed,
                "arm": arm_name,
                "track": track,
                "utterance": example.utterance,
                "template_id": example.template_id,
                "left": example.left,
                "right": example.right,
                "expected_operation": example.operation,
                "expected_binding": semantic_binding_label(example),
                "predicted_operation": operation,
                "predicted_binding": binding,
            }
            for example, operation, binding in zip(
                examples, operation_predictions, binding_predictions
            )
        )
    del model
    groups = {track: track_details["operation_accuracy"] for track, track_details in details.items()}
    metrics: dict[str, object] = {
        "seed": seed,
        "arm": arm_name,
        "evaluation_only": True,
        "source_checkpoint": str(run_dir / "bottleneck_final.pt"),
        "pressure_groups": groups,
        "pressure_group_details": details,
        "worst_robust_accuracy": round(
            min(groups[track] for track in _ROBUST_PRESSURE_TRACKS), 4
        ),
    }
    return metrics, records


def run_phase4_stageb(config_path: str) -> dict[str, object]:
    """Run the three new Stage-B cells, re-evaluate reused sources, and apply the gate."""
    preparation = prepare_phase4_stageb(config_path)
    config = preparation["config"]
    experiment = preparation["experiment"]
    registration = preparation["registration"]
    development = preparation["development"]
    ordinary_splits = preparation["ordinary_splits"]
    scaffold_splits = preparation["scaffold_splits"]
    if not isinstance(config, Mapping) or not isinstance(experiment, Mapping):
        raise ValueError("Invalid Stage-B preparation payload")
    if not isinstance(registration, Mapping) or not isinstance(development, Mapping):
        raise ValueError("Invalid Stage-B registration or development suite")
    if not isinstance(ordinary_splits, ConditionSplits) or not isinstance(
        scaffold_splits, ConditionSplits
    ):
        raise ValueError("Invalid Stage-B training splits")

    output_dir = Path(str(experiment["output_dir"]))
    stagea_dir = Path(str(experiment["stagea_output_dir"]))
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    training = config["training"]
    if not isinstance(training, Mapping):
        raise ValueError("Stage-B training configuration must be a mapping")
    device = _device_from_name(str(training.get("device", "auto")))
    batch_size = int(config.get("evaluation", {}).get("batch_size", 64))
    fingerprints = registration["arm_fingerprints"]
    if not isinstance(fingerprints, Mapping):
        raise ValueError("Stage-B registration is missing fingerprints")

    arms: dict[str, list[Mapping[str, object]]] = {
        "matched_generative_baseline": [],
        "weight_1_standard": [],
    }
    case_records: list[dict[str, object]] = []
    for seed in seeds:
        baseline_metrics, baseline_records = _evaluate_stageb_generative_source(
            stagea_dir / "matched_generative_baseline" / f"seed_{seed}",
            config,
            development,
            seed=seed,
            device=device,
            batch_size=batch_size,
        )
        standard_metrics, standard_records = _evaluate_stageb_bottleneck_source(
            stagea_dir / "discriminative_operation_binding" / f"seed_{seed}",
            development,
            seed=seed,
            arm_name="weight_1_standard",
            device=device,
            batch_size=batch_size,
        )
        arms["matched_generative_baseline"].append(baseline_metrics)
        arms["weight_1_standard"].append(standard_metrics)
        case_records.extend(baseline_records)
        case_records.extend(standard_records)

    new_arms = {
        "weight_1_scaffold": (
            scaffold_splits,
            float(experiment["full_binding_loss_weight"]),
        ),
        "weight_025_standard": (
            ordinary_splits,
            float(experiment["reduced_binding_loss_weight"]),
        ),
        "weight_025_scaffold": (
            scaffold_splits,
            float(experiment["reduced_binding_loss_weight"]),
        ),
    }
    for arm_name, (splits, binding_weight) in new_arms.items():
        arm_runs: list[Mapping[str, object]] = []
        fingerprint = str(fingerprints[arm_name])
        for seed in seeds:
            run_dir = output_dir / arm_name / f"seed_{seed}"
            metrics_path = run_dir / "metrics.json"
            checkpoint_path = run_dir / "bottleneck_final.pt"
            if metrics_path.exists():
                with metrics_path.open("r", encoding="utf-8") as handle:
                    metrics = json.load(handle)
                if (
                    metrics.get("phase4_stageb_training_fingerprint") != fingerprint
                    or not checkpoint_path.exists()
                ):
                    raise ValueError(f"Existing Stage-B artifact is incompatible: {run_dir}")
            else:
                metrics = run_bottleneck_condition(
                    splits,
                    run_dir,
                    config,
                    seed=seed,
                    pressure_groups=development,
                    binding_loss_weight=binding_weight,
                    arm_name=arm_name,
                    run_metadata={
                        "phase4_stageb_training_fingerprint": fingerprint,
                        "variants_per_operation": int(experiment["variants_per_operation"]),
                        "exposure_target": float(experiment["exposure"]),
                        "tokenizer_mode": "fixed_byte",
                    },
                )
            arm_runs.append(metrics)
            _, records = _evaluate_stageb_bottleneck_source(
                run_dir,
                development,
                seed=seed,
                arm_name=arm_name,
                device=device,
                batch_size=batch_size,
            )
            case_records.extend(records)
        arms[arm_name] = arm_runs

    report = {
        "experiment": str(config.get("name", "phase4_stageb")),
        "config": config_path,
        "registration": registration,
        "baseline_arm": "matched_generative_baseline",
        "gate": dict(experiment["gate"]),
        "arms": arms,
    }
    analysis = analyze_phase4_stageb(report)
    with (output_dir / "screen_results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "screen_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    with (output_dir / "case_predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(case_records, handle, indent=2)
    return {"report": report, "analysis": analysis, "case_records": case_records}


def run_phase4_stagec(config_path: str) -> dict[str, object]:
    """Run two paired-consistency weights and apply the unchanged robust gate."""
    preparation = prepare_phase4_stagec(config_path)
    config = preparation["config"]
    experiment = preparation["experiment"]
    splits = preparation["splits"]
    pairs = preparation["pairs"]
    development = preparation["development"]
    registration = preparation["registration"]
    if not isinstance(config, Mapping) or not isinstance(experiment, Mapping):
        raise ValueError("Invalid Stage-C preparation payload")
    if not isinstance(splits, ConditionSplits) or not isinstance(development, Mapping):
        raise ValueError("Invalid Stage-C splits or development suite")
    if not isinstance(pairs, Sequence) or not isinstance(registration, Mapping):
        raise ValueError("Invalid Stage-C pairs or registration")
    output_dir = Path(str(experiment["output_dir"]))
    stagea_dir = Path(str(experiment["stagea_output_dir"]))
    stageb_dir = Path(str(experiment["stageb_output_dir"]))
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    device = _device_from_name(str(config["training"].get("device", "auto")))
    batch_size = int(config.get("evaluation", {}).get("batch_size", 64))
    fingerprints = registration["arm_fingerprints"]
    if not isinstance(fingerprints, Mapping):
        raise ValueError("Stage-C registration is missing fingerprints")
    arms: dict[str, list[Mapping[str, object]]] = {
        "matched_generative_baseline": [],
        "lead_no_consistency": [],
    }
    case_records: list[dict[str, object]] = []
    for seed in seeds:
        baseline_metrics, baseline_records = _evaluate_stageb_generative_source(
            stagea_dir / "matched_generative_baseline" / f"seed_{seed}",
            config,
            development,
            seed=seed,
            device=device,
            batch_size=batch_size,
        )
        lead_metrics, lead_records = _evaluate_stageb_bottleneck_source(
            stageb_dir / "weight_025_scaffold" / f"seed_{seed}",
            development,
            seed=seed,
            arm_name="lead_no_consistency",
            device=device,
            batch_size=batch_size,
        )
        arms["matched_generative_baseline"].append(baseline_metrics)
        arms["lead_no_consistency"].append(lead_metrics)
        case_records.extend(baseline_records)
        case_records.extend(lead_records)
    weights = tuple(float(weight) for weight in experiment["consistency_weights"])
    new_arms = {"consistency_025": weights[0], "consistency_1": weights[1]}
    for arm_name, consistency_weight in new_arms.items():
        fingerprint = str(fingerprints[arm_name])
        arm_runs: list[Mapping[str, object]] = []
        for seed in seeds:
            run_dir = output_dir / arm_name / f"seed_{seed}"
            metrics_path = run_dir / "metrics.json"
            checkpoint_path = run_dir / "bottleneck_final.pt"
            if metrics_path.exists():
                with metrics_path.open("r", encoding="utf-8") as handle:
                    metrics = json.load(handle)
                if (
                    metrics.get("phase4_stagec_training_fingerprint") != fingerprint
                    or not checkpoint_path.exists()
                ):
                    raise ValueError(f"Existing Stage-C artifact is incompatible: {run_dir}")
            else:
                metrics = run_invariance_condition(
                    splits,
                    pairs,
                    run_dir,
                    config,
                    seed=seed,
                    pressure_groups=development,
                    binding_loss_weight=float(experiment["binding_loss_weight"]),
                    consistency_weight=consistency_weight,
                    pair_batch_size=int(experiment["pair_batch_size"]),
                    arm_name=arm_name,
                    run_metadata={
                        "phase4_stagec_training_fingerprint": fingerprint,
                        "variants_per_operation": int(experiment["variants_per_operation"]),
                        "exposure_target": float(experiment["exposure"]),
                        "tokenizer_mode": "fixed_byte",
                    },
                )
            arm_runs.append(metrics)
            _, records = _evaluate_stageb_bottleneck_source(
                run_dir,
                development,
                seed=seed,
                arm_name=arm_name,
                device=device,
                batch_size=batch_size,
            )
            case_records.extend(records)
        arms[arm_name] = arm_runs
    report = {
        "experiment": str(config.get("name", "phase4_stagec")),
        "config": config_path,
        "registration": registration,
        "baseline_arm": "matched_generative_baseline",
        "lead_arm": "lead_no_consistency",
        "gate": dict(experiment["gate"]),
        "arms": arms,
    }
    analysis = analyze_phase4_stagec(report)
    with (output_dir / "screen_results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "screen_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    with (output_dir / "case_predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(case_records, handle, indent=2)
    return {"report": report, "analysis": analysis, "case_records": case_records}


def run_phase4_stagec_confirmation(config_path: str) -> dict[str, object]:
    """Evaluate the selected Stage-C arm once on the paired sealed suite."""
    preparation = prepare_phase4_stagec_confirmation(config_path)
    config = preparation["config"]
    experiment = preparation["experiment"]
    confirmation = preparation["confirmation"]
    registration = preparation["registration"]
    if not isinstance(config, Mapping) or not isinstance(experiment, Mapping):
        raise ValueError("Invalid Stage-C confirmation configuration")
    if not isinstance(confirmation, Mapping) or not isinstance(registration, Mapping):
        raise ValueError("Invalid Stage-C confirmation payload")

    output_dir = Path(str(experiment["output_dir"]))
    confirmation_dir = output_dir / "sealed_confirmation"
    confirmation_dir.mkdir(parents=True, exist_ok=True)
    if (confirmation_dir / "results.json").exists():
        raise ValueError("Stage-C sealed confirmation is already completed")
    stagea_dir = Path(str(experiment["stagea_output_dir"]))
    stageb_dir = Path(str(experiment["stageb_output_dir"]))
    seeds = tuple(int(seed) for seed in registration["seeds"])
    device = _device_from_name(str(config["training"].get("device", "auto")))
    batch_size = int(config.get("evaluation", {}).get("batch_size", 64))
    arms: dict[str, list[Mapping[str, object]]] = {
        "matched_generative_baseline": [],
        "lead_no_consistency": [],
        "consistency_025": [],
    }
    case_records: list[dict[str, object]] = []
    evaluation_metrics: list[dict[str, object]] = []
    total_started = time.perf_counter()
    case_count = sum(len(examples) for examples in confirmation.values())
    for seed in seeds:
        evaluations = (
            (
                "matched_generative_baseline",
                stagea_dir / "matched_generative_baseline" / f"seed_{seed}",
            ),
            (
                "lead_no_consistency",
                stageb_dir / "weight_025_scaffold" / f"seed_{seed}",
            ),
            (
                "consistency_025",
                output_dir / "consistency_025" / f"seed_{seed}",
            ),
        )
        for arm_name, run_dir in evaluations:
            started = time.perf_counter()
            if arm_name == "matched_generative_baseline":
                metrics, records = _evaluate_stageb_generative_source(
                    run_dir,
                    config,
                    confirmation,
                    seed=seed,
                    device=device,
                    batch_size=batch_size,
                )
            else:
                metrics, records = _evaluate_stageb_bottleneck_source(
                    run_dir,
                    confirmation,
                    seed=seed,
                    arm_name=arm_name,
                    device=device,
                    batch_size=batch_size,
                )
            elapsed = time.perf_counter() - started
            arms[arm_name].append(metrics)
            case_records.extend(records)
            evaluation_metrics.append(
                {
                    "seed": seed,
                    "arm": arm_name,
                    "cases": case_count,
                    "wall_clock_seconds": round(elapsed, 6),
                    "milliseconds_per_case": round(elapsed * 1_000 / case_count, 6),
                }
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    report = {
        "experiment": str(registration["experiment"]),
        "config": config_path,
        "registration": registration,
        "baseline_arm": "matched_generative_baseline",
        "lead_arm": "lead_no_consistency",
        "candidate_arm": "consistency_025",
        "gate": dict(registration["gate"]),
        "arms": arms,
    }
    analysis = analyze_phase4_stagec_confirmation(report, case_records)
    eval_metrics = {
        "phase": "phase4_stagec_sealed_confirmation",
        "device": str(device),
        "cases_per_evaluation": case_count,
        "evaluation_count": len(evaluation_metrics),
        "total_wall_clock_seconds": round(time.perf_counter() - total_started, 6),
        "evaluations": evaluation_metrics,
    }
    completed_registration = {
        **dict(registration),
        "evaluation_started": True,
        "evaluation_completed": True,
        "confirmation_passed": bool(analysis["confirmation_passed"]),
    }
    report["registration"] = completed_registration
    with (confirmation_dir / "registration.json").open("w", encoding="utf-8") as handle:
        json.dump(completed_registration, handle, indent=2)
    with (confirmation_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (confirmation_dir / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    with (confirmation_dir / "case_predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(case_records, handle, indent=2)
    with (confirmation_dir / "eval_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(eval_metrics, handle, indent=2)
    return {
        "report": report,
        "analysis": analysis,
        "case_records": case_records,
        "eval_metrics": eval_metrics,
    }


def audit_bottleneck_screen(config_path: str) -> dict[str, object]:
    """Persist a read-only case audit of the completed Phase 4 development screen."""
    preparation = prepare_bottleneck_screen(config_path)
    config = preparation["config"]
    experiment = preparation["experiment"]
    pressure_groups = preparation["pressure_groups"]
    splits = preparation["splits"]
    if not isinstance(config, Mapping) or not isinstance(experiment, Mapping):
        raise ValueError("Invalid Phase 4 preparation payload")
    if not isinstance(pressure_groups, Mapping) or not isinstance(splits, ConditionSplits):
        raise ValueError("Invalid Phase 4 pressure suite")
    output_dir = Path(str(experiment["output_dir"]))
    if not (output_dir / "screen_results.json").exists():
        raise ValueError("Phase 4 screen must complete before its read-only audit")
    device = torch.device("cpu")
    batch_size = int(config.get("evaluation", {}).get("batch_size", 64))
    records: list[dict[str, object]] = []
    for seed in (int(value) for value in experiment["seeds"]):
        baseline_model, baseline_tokenizer = _load_generative_run(
            output_dir / "matched_generative_baseline" / f"seed_{seed}", config, device
        )
        operation_model, operation_tokenizer = _load_bottleneck_run(
            output_dir / "discriminative_operation" / f"seed_{seed}", device
        )
        multitask_model, multitask_tokenizer = _load_bottleneck_run(
            output_dir / "discriminative_operation_binding" / f"seed_{seed}", device
        )
        train_features, _ = extract_response_boundary_features(
            multitask_model.encoder, multitask_tokenizer, splits.train, device
        )
        train_binding_labels = torch.tensor(
            [semantic_binding_label(example) for example in splits.train], dtype=torch.long
        )
        for track, raw_examples in pressure_groups.items():
            examples = tuple(raw_examples)
            baseline_predictions = predict_operation_labels_batched(
                baseline_model, baseline_tokenizer, examples, device, batch_size=batch_size
            )
            operation_predictions, operation_bindings = predict_bottleneck_labels(
                operation_model, operation_tokenizer, examples, device, batch_size=batch_size
            )
            multitask_predictions, multitask_bindings = predict_bottleneck_labels(
                multitask_model, multitask_tokenizer, examples, device, batch_size=batch_size
            )
            evaluation_features, _ = extract_response_boundary_features(
                multitask_model.encoder, multitask_tokenizer, examples, device
            )
            frozen_probe_bindings = fit_frozen_linear_probe(
                train_features,
                train_binding_labels,
                evaluation_features,
                num_classes=2,
                seed=seed,
            ).tolist()
            for (
                example,
                baseline,
                operation,
                operation_binding,
                multitask,
                multitask_binding,
                frozen_probe_binding,
            ) in zip(
                examples,
                baseline_predictions,
                operation_predictions,
                operation_bindings,
                multitask_predictions,
                multitask_bindings,
                frozen_probe_bindings,
            ):
                records.append(
                    {
                        "seed": seed,
                        "track": str(track),
                        "utterance": example.utterance,
                        "template_id": example.template_id,
                        "left": example.left,
                        "right": example.right,
                        "expected_operation": example.operation,
                        "expected_binding": semantic_binding_label(example),
                        "baseline_operation": baseline.removeprefix("OP="),
                        "operation_only_operation": operation,
                        "operation_only_binding": operation_binding,
                        "multitask_operation": multitask,
                        "multitask_binding": multitask_binding,
                        "multitask_frozen_probe_binding": int(frozen_probe_binding),
                    }
                )
        del baseline_model, operation_model, multitask_model
    analysis = analyze_bottleneck_case_records(records)
    with (output_dir / "failure_audit_records.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    with (output_dir / "failure_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    return {"records": records, "analysis": analysis}


def analyze_phase4_stageb_probe_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Separate head failures from information recoverable by frozen linear probes."""
    if not records:
        raise ValueError("Stage-B frozen-probe audit requires records")

    def summarize(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
        n = len(cases)
        operation_head = sum(
            case["head_operation"] == case["expected_operation"] for case in cases
        )
        operation_probe = sum(
            case["probe_operation"] == case["expected_operation"] for case in cases
        )
        binding_head = sum(case["head_binding"] == case["expected_binding"] for case in cases)
        binding_probe = sum(case["probe_binding"] == case["expected_binding"] for case in cases)
        return {
            "n": n,
            "operation_head_accuracy": operation_head / n,
            "operation_probe_accuracy": operation_probe / n,
            "operation_probe_delta": (operation_probe - operation_head) / n,
            "operation_probe_repairs": sum(
                case["head_operation"] != case["expected_operation"]
                and case["probe_operation"] == case["expected_operation"]
                for case in cases
            ),
            "binding_head_accuracy": binding_head / n,
            "binding_probe_accuracy": binding_probe / n,
            "binding_probe_delta": (binding_probe - binding_head) / n,
            "joint_head_accuracy": sum(
                case["head_operation"] == case["expected_operation"]
                and case["head_binding"] == case["expected_binding"]
                for case in cases
            )
            / n,
            "joint_probe_accuracy": sum(
                case["probe_operation"] == case["expected_operation"]
                and case["probe_binding"] == case["expected_binding"]
                for case in cases
            )
            / n,
        }

    arms: dict[str, object] = {}
    for arm in sorted({str(record["arm"]) for record in records}):
        arm_cases = [record for record in records if record["arm"] == arm]
        present_tracks = sorted({str(record["track"]) for record in arm_cases})
        tracks = {
            track: summarize([record for record in arm_cases if record["track"] == track])
            for track in present_tracks
        }
        robust_cases = [
            record for record in arm_cases if record["track"] in _ROBUST_PRESSURE_TRACKS
        ]
        if not robust_cases:
            robust_cases = arm_cases
        arms[arm] = {"overall_robust": summarize(robust_cases), "tracks": tracks}
    return {"record_count": len(records), "arms": arms}


def audit_phase4_stageb(config_path: str) -> dict[str, object]:
    """Fit frozen operation/binding probes on each completed Stage-B encoder."""
    preparation = prepare_phase4_stageb(config_path)
    config = preparation["config"]
    experiment = preparation["experiment"]
    development = preparation["development"]
    ordinary_splits = preparation["ordinary_splits"]
    scaffold_splits = preparation["scaffold_splits"]
    if not isinstance(config, Mapping) or not isinstance(experiment, Mapping):
        raise ValueError("Invalid Stage-B audit preparation payload")
    if not isinstance(development, Mapping):
        raise ValueError("Invalid Stage-B audit development suite")
    if not isinstance(ordinary_splits, ConditionSplits) or not isinstance(
        scaffold_splits, ConditionSplits
    ):
        raise ValueError("Invalid Stage-B audit splits")
    output_dir = Path(str(experiment["output_dir"]))
    if not (output_dir / "screen_analysis.json").exists():
        raise ValueError("Stage-B screen must complete before the frozen-probe audit")
    stagea_dir = Path(str(experiment["stagea_output_dir"]))
    device = _device_from_name(str(config["training"].get("device", "auto")))
    batch_size = int(config.get("evaluation", {}).get("batch_size", 64))
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    arm_sources = {
        "weight_1_standard": (
            stagea_dir / "discriminative_operation_binding",
            ordinary_splits,
        ),
        "weight_1_scaffold": (output_dir / "weight_1_scaffold", scaffold_splits),
        "weight_025_standard": (output_dir / "weight_025_standard", ordinary_splits),
        "weight_025_scaffold": (output_dir / "weight_025_scaffold", scaffold_splits),
    }
    ordered_tracks = tuple(development)
    evaluation_examples = tuple(
        example for track in ordered_tracks for example in development[track]
    )
    records: list[dict[str, object]] = []
    for arm_name, (arm_dir, splits) in arm_sources.items():
        for seed in seeds:
            model, tokenizer = _load_bottleneck_run(arm_dir / f"seed_{seed}", device)
            train_features, train_operation_labels = extract_response_boundary_features(
                model.encoder, tokenizer, splits.train, device
            )
            train_binding_labels = torch.tensor(
                [semantic_binding_label(example) for example in splits.train], dtype=torch.long
            )
            evaluation_features, _ = extract_response_boundary_features(
                model.encoder, tokenizer, evaluation_examples, device
            )
            operation_probe = fit_frozen_linear_probe(
                train_features,
                train_operation_labels,
                evaluation_features,
                num_classes=len(OPERATIONS),
                seed=seed,
            )
            binding_probe = fit_frozen_linear_probe(
                train_features,
                train_binding_labels,
                evaluation_features,
                num_classes=2,
                seed=seed,
            )
            head_operations, head_bindings = predict_bottleneck_labels(
                model,
                tokenizer,
                evaluation_examples,
                device,
                batch_size=batch_size,
            )
            offset = 0
            for track in ordered_tracks:
                track_examples = development[track]
                for local_index, example in enumerate(track_examples):
                    index = offset + local_index
                    records.append(
                        {
                            "arm": arm_name,
                            "seed": seed,
                            "track": track,
                            "utterance": example.utterance,
                            "template_id": example.template_id,
                            "left": example.left,
                            "right": example.right,
                            "expected_operation": example.operation,
                            "expected_binding": semantic_binding_label(example),
                            "head_operation": head_operations[index],
                            "head_binding": head_bindings[index],
                            "probe_operation": OPERATIONS[int(operation_probe[index])],
                            "probe_binding": int(binding_probe[index]),
                        }
                    )
                offset += len(track_examples)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    analysis = analyze_phase4_stageb_probe_records(records)
    with (output_dir / "frozen_probe_records.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    with (output_dir / "frozen_probe_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    return {"records": records, "analysis": analysis}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LCM discriminative semantic-bottleneck screen")
    parser.add_argument("--config", default="configs/phase4_bottleneck_screen.yaml")
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "screen",
            "audit",
            "prepare-stageb",
            "screen-stageb",
            "audit-stageb",
            "prepare-stagec",
            "screen-stagec",
            "prepare-confirm-stagec",
            "confirm-stagec",
        ),
        required=True,
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare_bottleneck_screen(args.config)
        print(json.dumps(result["registration"], indent=2))
    elif args.stage == "screen":
        result = run_bottleneck_screen(args.config)
        print(json.dumps(result["analysis"], indent=2))
    elif args.stage == "audit":
        result = audit_bottleneck_screen(args.config)
        print(json.dumps(result["analysis"], indent=2))
    elif args.stage == "prepare-stageb":
        result = prepare_phase4_stageb(args.config)
        print(json.dumps(result["registration"], indent=2))
    elif args.stage == "screen-stageb":
        result = run_phase4_stageb(args.config)
        print(json.dumps(result["analysis"], indent=2))
    elif args.stage == "audit-stageb":
        result = audit_phase4_stageb(args.config)
        print(json.dumps(result["analysis"], indent=2))
    elif args.stage == "prepare-stagec":
        result = prepare_phase4_stagec(args.config)
        print(json.dumps(result["registration"], indent=2))
    elif args.stage == "screen-stagec":
        result = run_phase4_stagec(args.config)
        print(json.dumps(result["analysis"], indent=2))
    elif args.stage == "prepare-confirm-stagec":
        result = prepare_phase4_stagec_confirmation(args.config)
        print(json.dumps(result["registration"], indent=2))
    else:
        result = run_phase4_stagec_confirmation(args.config)
        print(json.dumps(result["analysis"], indent=2))


def _macro_metric(run: Mapping[str, object], metric: str) -> float:
    if metric == "operation_accuracy":
        groups = run["pressure_groups"]
        if not isinstance(groups, Mapping):
            raise ValueError("Run pressure_groups must be a mapping")
        return sum(float(groups[track]) for track in _ROBUST_PRESSURE_TRACKS) / len(_ROBUST_PRESSURE_TRACKS)
    details = run["pressure_group_details"]
    if not isinstance(details, Mapping):
        raise ValueError("Run pressure_group_details must be a mapping")
    return sum(float(details[track][metric]) for track in _ROBUST_PRESSURE_TRACKS) / len(_ROBUST_PRESSURE_TRACKS)


def analyze_bottleneck_screen(report: Mapping[str, object]) -> dict[str, object]:
    """Apply the paired screen gate and the registered binding-head preference."""
    arms = report["arms"]
    gate = report["gate"]
    if not isinstance(arms, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("Bottleneck report is missing arms or gate")
    baseline_arm = str(report.get("baseline_arm", "generative_baseline"))
    baseline_runs = arms[baseline_arm]
    if not isinstance(baseline_runs, Sequence):
        raise ValueError("Generative baseline runs must be a sequence")
    baseline = {int(run["seed"]): run for run in baseline_runs}
    minimum_gain = float(gate["minimum_worst_group_gain"])
    maximum_regression = float(gate["maximum_track_regression"])

    effects: dict[str, object] = {}
    for arm_name in ("discriminative_operation", "discriminative_operation_binding"):
        raw_runs = arms[arm_name]
        if not isinstance(raw_runs, Sequence):
            raise ValueError(f"{arm_name} runs must be a sequence")
        arm = {int(run["seed"]): run for run in raw_runs}
        seeds = sorted(set(baseline).intersection(arm))
        if not seeds:
            raise ValueError(f"No paired seeds for {arm_name}")
        worst_delta = _distribution_summary(
            [float(arm[seed]["worst_robust_accuracy"]) - float(baseline[seed]["worst_robust_accuracy"]) for seed in seeds]
        )
        macro_delta = _distribution_summary(
            [_macro_metric(arm[seed], "operation_accuracy") - _macro_metric(baseline[seed], "operation_accuracy") for seed in seeds]
        )
        track_deltas = {
            track: _distribution_summary(
                [
                    float(arm[seed]["pressure_groups"][track])
                    - float(baseline[seed]["pressure_groups"][track])
                    for seed in seeds
                ]
            )
            for track in _ROBUST_PRESSURE_TRACKS
        }
        lower_bound = float(worst_delta["bootstrap_95_ci"][0])
        gate_passed = (
            float(worst_delta["mean"]) >= minimum_gain
            and lower_bound > 0
            and all(float(summary["mean"]) >= -maximum_regression for summary in track_deltas.values())
        )
        effects[arm_name] = {
            "paired_seeds": seeds,
            "worst_group_delta": worst_delta,
            "macro_operation_delta": macro_delta,
            "track_operation_deltas": track_deltas,
            "gate_passed": gate_passed,
        }

    operation_runs = {int(run["seed"]): run for run in arms["discriminative_operation"]}
    binding_runs = {int(run["seed"]): run for run in arms["discriminative_operation_binding"]}
    comparison_seeds = sorted(set(operation_runs).intersection(binding_runs))
    operation_track_deltas = {
        track: _distribution_summary(
            [
                float(binding_runs[seed]["pressure_groups"][track])
                - float(operation_runs[seed]["pressure_groups"][track])
                for seed in comparison_seeds
            ]
        )
        for track in _ROBUST_PRESSURE_TRACKS
    }
    joint_macro_delta = _distribution_summary(
        [
            _macro_metric(binding_runs[seed], "joint_accuracy")
            - _macro_metric(operation_runs[seed], "joint_accuracy")
            for seed in comparison_seeds
        ]
    )
    operation_noninferior = all(
        float(summary["mean"]) >= -maximum_regression for summary in operation_track_deltas.values()
    )
    binding_comparison = {
        "paired_seeds": comparison_seeds,
        "operation_track_deltas": operation_track_deltas,
        "operation_noninferior": operation_noninferior,
        "joint_macro_delta": joint_macro_delta,
    }

    operation_passed = bool(effects["discriminative_operation"]["gate_passed"])
    binding_passed = bool(effects["discriminative_operation_binding"]["gate_passed"])
    if binding_passed and operation_noninferior and float(joint_macro_delta["mean"]) > 0:
        selected_arm: str | None = "discriminative_operation_binding"
    elif operation_passed:
        selected_arm = "discriminative_operation"
    elif binding_passed:
        selected_arm = "discriminative_operation_binding"
    else:
        selected_arm = None
    return {
        "arm_effects": effects,
        "binding_head_comparison": binding_comparison,
        "selected_arm": selected_arm,
        "sealed_confirmation_required": selected_arm is not None,
    }


def analyze_phase4_stageb(report: Mapping[str, object]) -> dict[str, object]:
    """Analyze the registered binding-weight × scaffold factorial."""
    arms = report["arms"]
    gate = report["gate"]
    if not isinstance(arms, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("Stage-B report is missing arms or gate")
    baseline_name = str(report.get("baseline_arm", "matched_generative_baseline"))
    baseline_runs = arms[baseline_name]
    if not isinstance(baseline_runs, Sequence):
        raise ValueError("Stage-B baseline runs must be a sequence")
    baseline = {int(run["seed"]): run for run in baseline_runs}
    arm_names = (
        "weight_1_standard",
        "weight_1_scaffold",
        "weight_025_standard",
        "weight_025_scaffold",
    )
    minimum_gain = float(gate["minimum_worst_group_gain"])
    maximum_regression = float(gate["maximum_track_regression"])
    effects: dict[str, object] = {}
    indexed: dict[str, dict[int, Mapping[str, object]]] = {}
    for arm_name in arm_names:
        raw_runs = arms[arm_name]
        if not isinstance(raw_runs, Sequence):
            raise ValueError(f"{arm_name} runs must be a sequence")
        indexed[arm_name] = {int(run["seed"]): run for run in raw_runs}
        seeds = sorted(set(baseline).intersection(indexed[arm_name]))
        if not seeds:
            raise ValueError(f"No paired Stage-B seeds for {arm_name}")
        worst_delta = _distribution_summary(
            [
                float(indexed[arm_name][seed]["worst_robust_accuracy"])
                - float(baseline[seed]["worst_robust_accuracy"])
                for seed in seeds
            ]
        )
        macro_delta = _distribution_summary(
            [
                _macro_metric(indexed[arm_name][seed], "operation_accuracy")
                - _macro_metric(baseline[seed], "operation_accuracy")
                for seed in seeds
            ]
        )
        track_deltas = {
            track: _distribution_summary(
                [
                    float(indexed[arm_name][seed]["pressure_groups"][track])
                    - float(baseline[seed]["pressure_groups"][track])
                    for seed in seeds
                ]
            )
            for track in _ROBUST_PRESSURE_TRACKS
        }
        effects[arm_name] = {
            "paired_seeds": seeds,
            "worst_group_delta": worst_delta,
            "macro_operation_delta": macro_delta,
            "track_operation_deltas": track_deltas,
            "gate_passed": (
                float(worst_delta["mean"]) >= minimum_gain
                and float(worst_delta["bootstrap_95_ci"][0]) > 0
                and all(
                    float(summary["mean"]) >= -maximum_regression
                    for summary in track_deltas.values()
                )
            ),
        }

    metric_functions = {
        "operation_worst": lambda run: float(run["worst_robust_accuracy"]),
        "operation_macro": lambda run: _macro_metric(run, "operation_accuracy"),
        "binding_macro": lambda run: _macro_metric(run, "binding_accuracy"),
        "joint_macro": lambda run: _macro_metric(run, "joint_accuracy"),
    }
    contrasts = {
        "scaffold_at_weight_1": ("weight_1_scaffold", "weight_1_standard"),
        "scaffold_at_weight_025": ("weight_025_scaffold", "weight_025_standard"),
        "reduced_weight_standard": ("weight_025_standard", "weight_1_standard"),
        "reduced_weight_scaffold": ("weight_025_scaffold", "weight_1_scaffold"),
    }
    factorial_effects: dict[str, object] = {}
    for contrast_name, (positive_name, negative_name) in contrasts.items():
        seeds = sorted(set(indexed[positive_name]).intersection(indexed[negative_name]))
        factorial_effects[contrast_name] = {
            metric_name: _distribution_summary(
                [
                    metric(indexed[positive_name][seed])
                    - metric(indexed[negative_name][seed])
                    for seed in seeds
                ]
            )
            for metric_name, metric in metric_functions.items()
        }
    interaction_seeds = sorted(set.intersection(*(set(indexed[name]) for name in arm_names)))
    factorial_effects["interaction"] = {
        metric_name: _distribution_summary(
            [
                (
                    metric(indexed["weight_025_scaffold"][seed])
                    - metric(indexed["weight_025_standard"][seed])
                )
                - (
                    metric(indexed["weight_1_scaffold"][seed])
                    - metric(indexed["weight_1_standard"][seed])
                )
                for seed in interaction_seeds
            ]
        )
        for metric_name, metric in metric_functions.items()
    }
    passing = [name for name in arm_names if effects[name]["gate_passed"]]
    selected_arm = max(
        passing,
        key=lambda name: (
            float(effects[name]["worst_group_delta"]["mean"]),
            float(effects[name]["macro_operation_delta"]["mean"]),
        ),
        default=None,
    )
    return {
        "arm_effects": effects,
        "factorial_effects": factorial_effects,
        "selected_arm": selected_arm,
        "sealed_confirmation_required": selected_arm is not None,
    }


def analyze_phase4_stagec(report: Mapping[str, object]) -> dict[str, object]:
    """Apply the robust gate and estimate paired consistency-weight effects."""
    arms = report["arms"]
    gate = report["gate"]
    if not isinstance(arms, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("Stage-C report is missing arms or gate")
    baseline_name = str(report.get("baseline_arm", "matched_generative_baseline"))
    lead_name = str(report.get("lead_arm", "lead_no_consistency"))
    candidate_names = (lead_name, "consistency_025", "consistency_1")
    baseline = {int(run["seed"]): run for run in arms[baseline_name]}
    indexed = {
        arm_name: {int(run["seed"]): run for run in arms[arm_name]}
        for arm_name in candidate_names
    }
    minimum_gain = float(gate["minimum_worst_group_gain"])
    maximum_regression = float(gate["maximum_track_regression"])
    effects: dict[str, object] = {}
    for arm_name in candidate_names:
        seeds = sorted(set(baseline).intersection(indexed[arm_name]))
        if not seeds:
            raise ValueError(f"No paired Stage-C seeds for {arm_name}")
        worst = _distribution_summary(
            [
                float(indexed[arm_name][seed]["worst_robust_accuracy"])
                - float(baseline[seed]["worst_robust_accuracy"])
                for seed in seeds
            ]
        )
        macro = _distribution_summary(
            [
                _macro_metric(indexed[arm_name][seed], "operation_accuracy")
                - _macro_metric(baseline[seed], "operation_accuracy")
                for seed in seeds
            ]
        )
        tracks = {
            track: _distribution_summary(
                [
                    float(indexed[arm_name][seed]["pressure_groups"][track])
                    - float(baseline[seed]["pressure_groups"][track])
                    for seed in seeds
                ]
            )
            for track in _ROBUST_PRESSURE_TRACKS
        }
        effects[arm_name] = {
            "paired_seeds": seeds,
            "worst_group_delta": worst,
            "macro_operation_delta": macro,
            "track_operation_deltas": tracks,
            "gate_passed": (
                float(worst["mean"]) >= minimum_gain
                and float(worst["bootstrap_95_ci"][0]) > 0
                and all(
                    float(summary["mean"]) >= -maximum_regression
                    for summary in tracks.values()
                )
            ),
        }

    metric_functions = {
        "worst_group": lambda run: float(run["worst_robust_accuracy"]),
        "macro_operation": lambda run: _macro_metric(run, "operation_accuracy"),
        "macro_binding": lambda run: _macro_metric(run, "binding_accuracy"),
        "macro_joint": lambda run: _macro_metric(run, "joint_accuracy"),
    }
    contrasts = {
        "consistency_025_vs_lead": ("consistency_025", lead_name),
        "consistency_1_vs_lead": ("consistency_1", lead_name),
        "consistency_1_vs_025": ("consistency_1", "consistency_025"),
    }
    consistency_effects: dict[str, object] = {}
    for contrast_name, (positive, negative) in contrasts.items():
        seeds = sorted(set(indexed[positive]).intersection(indexed[negative]))
        consistency_effects[contrast_name] = {
            metric_name: _distribution_summary(
                [
                    metric(indexed[positive][seed]) - metric(indexed[negative][seed])
                    for seed in seeds
                ]
            )
            for metric_name, metric in metric_functions.items()
        }
    passing = [name for name in candidate_names if effects[name]["gate_passed"]]
    selected = max(
        passing,
        key=lambda name: (
            float(effects[name]["worst_group_delta"]["mean"]),
            float(effects[name]["macro_operation_delta"]["mean"]),
        ),
        default=None,
    )
    return {
        "arm_effects": effects,
        "consistency_effects": consistency_effects,
        "selected_arm": selected,
        "sealed_confirmation_required": selected is not None,
    }


def _paired_invariance_by_seed(
    records: Sequence[Mapping[str, object]],
    arm_name: str,
    seeds: Sequence[int],
) -> dict[int, dict[str, object]]:
    """Score operation agreement and pair correctness for sealed A/B forms."""
    result: dict[int, dict[str, object]] = {}
    for seed in seeds:
        grouped: dict[tuple[str, str, int, int], list[Mapping[str, object]]] = {}
        for record in records:
            if str(record["arm"]) != arm_name or int(record["seed"]) != seed:
                continue
            key = (
                str(record["track"]),
                str(record["expected_operation"]),
                int(record["left"]),
                int(record["right"]),
            )
            grouped.setdefault(key, []).append(record)
        if not grouped:
            raise ValueError(f"No sealed pair records for {arm_name} seed {seed}")
        track_values: dict[str, dict[str, list[float]]] = {}
        for (track, expected, _left, _right), pair in grouped.items():
            if len(pair) != 2 or {int(case["expected_binding"]) for case in pair} != {0, 1}:
                raise ValueError(f"Invalid sealed pair for {arm_name} seed {seed}: {track}")
            predictions = [str(case["predicted_operation"]) for case in pair]
            values = track_values.setdefault(track, {"agreement": [], "pair_correct": []})
            values["agreement"].append(float(predictions[0] == predictions[1]))
            values["pair_correct"].append(float(all(prediction == expected for prediction in predictions)))
        track_scores = {
            track: {
                metric: sum(values) / len(values)
                for metric, values in metrics.items()
            }
            for track, metrics in track_values.items()
        }
        missing = set(_ROBUST_PRESSURE_TRACKS).difference(track_scores)
        if missing:
            raise ValueError(f"Sealed pair records are missing tracks: {sorted(missing)}")
        result[seed] = {
            "track_scores": track_scores,
            "macro_agreement": sum(
                track_scores[track]["agreement"] for track in _ROBUST_PRESSURE_TRACKS
            )
            / len(_ROBUST_PRESSURE_TRACKS),
            "macro_pair_correct": sum(
                track_scores[track]["pair_correct"] for track in _ROBUST_PRESSURE_TRACKS
            )
            / len(_ROBUST_PRESSURE_TRACKS),
        }
    return result


def analyze_phase4_stagec_confirmation(
    report: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Apply the sealed gate and report paired-form invariance diagnostics."""
    arms = report["arms"]
    gate = report["gate"]
    if not isinstance(arms, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("Stage-C confirmation report is missing arms or gate")
    baseline_name = str(report.get("baseline_arm", "matched_generative_baseline"))
    lead_name = str(report.get("lead_arm", "lead_no_consistency"))
    candidate_name = str(report.get("candidate_arm", "consistency_025"))
    indexed = {
        arm_name: {int(run["seed"]): run for run in arms[arm_name]}
        for arm_name in (baseline_name, lead_name, candidate_name)
    }

    def arm_effect(positive: str, negative: str) -> dict[str, object]:
        seeds = sorted(set(indexed[positive]).intersection(indexed[negative]))
        if not seeds:
            raise ValueError(f"No paired confirmation seeds for {positive} versus {negative}")
        return {
            "paired_seeds": seeds,
            "worst_group_delta": _distribution_summary(
                [
                    float(indexed[positive][seed]["worst_robust_accuracy"])
                    - float(indexed[negative][seed]["worst_robust_accuracy"])
                    for seed in seeds
                ]
            ),
            "macro_operation_delta": _distribution_summary(
                [
                    _macro_metric(indexed[positive][seed], "operation_accuracy")
                    - _macro_metric(indexed[negative][seed], "operation_accuracy")
                    for seed in seeds
                ]
            ),
            "track_operation_deltas": {
                track: _distribution_summary(
                    [
                        float(indexed[positive][seed]["pressure_groups"][track])
                        - float(indexed[negative][seed]["pressure_groups"][track])
                        for seed in seeds
                    ]
                )
                for track in _ROBUST_PRESSURE_TRACKS
            },
        }

    candidate_effect = arm_effect(candidate_name, baseline_name)
    candidate_vs_lead = arm_effect(candidate_name, lead_name)
    seeds = candidate_effect["paired_seeds"]
    pair_by_arm = {
        arm_name: _paired_invariance_by_seed(records, arm_name, seeds)
        for arm_name in (baseline_name, lead_name, candidate_name)
    }
    pair_invariance = {
        arm_name: {
            metric: _distribution_summary(
                [float(pair_by_arm[arm_name][seed][metric]) for seed in seeds]
            )
            for metric in ("macro_agreement", "macro_pair_correct")
        }
        for arm_name in pair_by_arm
    }

    def pair_effect(positive: str, negative: str) -> dict[str, object]:
        return {
            metric: _distribution_summary(
                [
                    float(pair_by_arm[positive][seed][metric])
                    - float(pair_by_arm[negative][seed][metric])
                    for seed in seeds
                ]
            )
            for metric in ("macro_agreement", "macro_pair_correct")
        }

    worst = candidate_effect["worst_group_delta"]
    tracks = candidate_effect["track_operation_deltas"]
    minimum_gain = float(gate["minimum_worst_group_gain"])
    maximum_regression = float(gate["maximum_track_regression"])
    confirmation_passed = (
        float(worst["mean"]) >= minimum_gain
        and float(worst["bootstrap_95_ci"][0]) > 0
        and all(float(summary["mean"]) >= -maximum_regression for summary in tracks.values())
    )
    return {
        "candidate_arm": candidate_name,
        "candidate_effect": candidate_effect,
        "candidate_vs_lead": candidate_vs_lead,
        "pair_invariance": pair_invariance,
        "pair_invariance_effects": {
            "candidate_vs_baseline": pair_effect(candidate_name, baseline_name),
            "candidate_vs_lead": pair_effect(candidate_name, lead_name),
        },
        "confirmation_passed": confirmation_passed,
    }


if __name__ == "__main__":
    main()
