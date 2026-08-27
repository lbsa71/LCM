"""Controlled semantic form-variation experiment for the LCM parser layer.

This module intentionally owns only the language-to-canonical-operation stage.
It can therefore be run independently of base pretraining, agent SFT, and the
deterministic execution shell. Future experiment families can reuse its typed
examples and split construction without coupling to a particular agent model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import yaml
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from torch.utils.data import DataLoader, Dataset

from training.model import SyntheticTransformer, TransformerConfig
from training.pretrain import get_lr_scheduler
from training.tokenizer import train_synthetic_tokenizer
from utils.timer import StepTimer


OPERATIONS = ("ADD", "SUBTRACT", "COMPARE")
_OPERATION_TO_LABEL = {operation: index for index, operation in enumerate(OPERATIONS)}
_TYPED_FRAME_PATTERN = re.compile(r"^OP=(ADD|SUBTRACT|COMPARE);A=(-?\d+);B=(-?\d+)$")


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


def serialize_typed_frame(example: FormVariationExample) -> str:
    """Serialize the canonical parser IR with semantic, not mention-order, roles."""
    if example.operation not in OPERATIONS:
        raise ValueError(f"Unsupported operation: {example.operation}")
    return f"OP={example.operation};A={example.left};B={example.right}"


def parse_typed_frame(value: str) -> dict[str, object]:
    """Parse the one accepted deterministic typed-frame wire format."""
    match = _TYPED_FRAME_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid typed frame: {value!r}")
    operation, left, right = match.groups()
    return {"operation": operation, "A": int(left), "B": int(right)}


def _with_target_mode(example: FormVariationExample, target_mode: str) -> FormVariationExample:
    if target_mode == "operation":
        target = f"OP={example.operation}"
    elif target_mode == "typed_frame":
        target = serialize_typed_frame(example)
    else:
        raise ValueError(f"Unsupported parser target mode: {target_mode}")
    return FormVariationExample(
        utterance=example.utterance,
        target=target,
        operation=example.operation,
        template_id=example.template_id,
        left=example.left,
        right=example.right,
    )


def condition_splits_with_target_mode(splits: ConditionSplits, target_mode: str) -> ConditionSplits:
    """Copy every split with a registered operation-only or typed-frame target."""
    return ConditionSplits(
        train=tuple(_with_target_mode(example, target_mode) for example in splits.train),
        seen_form=tuple(_with_target_mode(example, target_mode) for example in splits.seen_form),
        same_meaning_unseen_form=tuple(
            _with_target_mode(example, target_mode) for example in splits.same_meaning_unseen_form
        ),
        unseen_operands_seen_form=tuple(
            _with_target_mode(example, target_mode) for example in splits.unseen_operands_seen_form
        ),
        minimal_contrasts=tuple(_with_target_mode(example, target_mode) for example in splits.minimal_contrasts),
    )


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


_TRAINING_CONTRAST_TEMPLATES: dict[str, tuple[str, ...]] = {
    "ADD": (
        "A dispatch records {left} zols and {right} binks. Give their combined count.",
        "A slate shows {left} zols beside {right} binks. How many are listed in total?",
        "A tally contains {left} zols and {right} binks. Find the amount altogether.",
        "A register pairs {left} zols with {right} binks. Return their sum.",
    ),
    "SUBTRACT": (
        "A dispatch records {left} zols and {right} binks. Give how many more zols are recorded.",
        "A slate shows {left} zols beside {right} binks. Find the excess of zols over binks.",
        "A tally contains {left} zols and {right} binks. Return the difference in their counts.",
        "A register pairs {left} zols with {right} binks. Subtract the bink count from the zol count.",
    ),
    "COMPARE": (
        "A dispatch records {left} zols and {right} binks. Decide whether zols outnumber binks.",
        "A slate shows {left} zols beside {right} binks. Say whether the zol count is larger.",
        "A tally contains {left} zols and {right} binks. Check whether there are more zols.",
        "A register pairs {left} zols with {right} binks. Determine whether zols exceed binks.",
    ),
}

_TRAINING_LEXICAL_TEMPLATES: dict[str, tuple[str, ...]] = {
    "ADD": (
        "Compute the aggregate produced by {left} alongside {right}.",
        "Accumulate {left} together with {right} and report the result.",
        "What combined magnitude emerges from {left} paired with {right}?",
        "Return the aggregate of the quantities {left} and {right}.",
    ),
    "SUBTRACT": (
        "Quantify the shortfall from {right} up to {left}.",
        "What excess remains when {right} is removed from {left}?",
        "Return the decrement between {left} and {right}.",
        "Measure how far {left} stands above {right}.",
    ),
    "COMPARE": (
        "Does {left} surpass {right}?",
        "Determine whether {left} ranks above {right}.",
        "Is {left} the superior magnitude relative to {right}?",
        "Judge whether {left} is beyond {right}.",
    ),
}

_TRAINING_DISCOURSE_CUES = (
    "Although a comparison might be considered, ",
    "A total may be useful elsewhere, but here ",
    "Someone suggested taking one amount away; instead, ",
    "Ignore the order in which the numbers were mentioned and ",
)

_TRAINING_DISCOURSE_REQUESTS = {
    "ADD": "combine {left} with {right} and report the total.",
    "SUBTRACT": "remove {right} from {left} and report what remains.",
    "COMPARE": "decide whether {left} is greater than {right}.",
}


def build_contrast_balanced_examples(
    examples: Sequence[FormVariationExample],
    replacement_fraction: float,
    seed: int,
) -> tuple[FormVariationExample, ...]:
    """Replace a matched fraction with shared-scaffold minimal contrasts.

    Replacement happens within each operand-pair/operation cell.  It therefore
    preserves the number of examples, operand coverage, operation balance, and
    update-token budget instead of granting the intervention extra examples.
    """
    if not examples:
        raise ValueError("Contrast balancing requires at least one example")
    if not 0 < replacement_fraction <= 1:
        raise ValueError("replacement_fraction must be in (0, 1]")

    grouped_indices: dict[tuple[int, int, str], list[int]] = {}
    for index, example in enumerate(examples):
        if example.operation not in OPERATIONS:
            raise ValueError(f"Unsupported operation: {example.operation}")
        grouped_indices.setdefault((example.left, example.right, example.operation), []).append(index)

    result = list(examples)
    rng = random.Random(seed)
    for group_key in sorted(grouped_indices):
        indices = grouped_indices[group_key]
        replacement_count = min(len(indices), max(1, int(round(len(indices) * replacement_fraction))))
        selected = sorted(rng.sample(indices, replacement_count))
        left, right, operation = group_key
        templates = _TRAINING_CONTRAST_TEMPLATES[operation]
        for contrast_index, example_index in enumerate(selected):
            template = templates[contrast_index % len(templates)]
            result[example_index] = FormVariationExample(
                utterance=template.format(left=left, right=right),
                target=f"OP={operation}",
                operation=operation,
                template_id=1_000 + contrast_index,
                left=left,
                right=right,
            )
    return tuple(result)


def build_lexical_contrast_examples(
    examples: Sequence[FormVariationExample],
    replacement_fraction: float,
    seed: int,
) -> tuple[FormVariationExample, ...]:
    """Replace standard forms with training-only lexical contrast variants.

    Existing minimal-contrast replacements use template IDs 1000–1999 and are
    deliberately ineligible. This makes the two factors disjoint in the
    combined Phase-2.6 arm while preserving the total example budget.
    """
    if not examples:
        raise ValueError("Lexical contrast balancing requires at least one example")
    if not 0 < replacement_fraction <= 1:
        raise ValueError("replacement_fraction must be in (0, 1]")

    grouped_indices: dict[tuple[int, int, str], list[int]] = {}
    for index, example in enumerate(examples):
        if example.operation not in OPERATIONS:
            raise ValueError(f"Unsupported operation: {example.operation}")
        if example.template_id < 1_000:
            grouped_indices.setdefault((example.left, example.right, example.operation), []).append(index)

    result = list(examples)
    rng = random.Random(seed)
    for group_key in sorted(grouped_indices):
        indices = grouped_indices[group_key]
        replacement_count = min(len(indices), max(1, int(round(len(indices) * replacement_fraction))))
        selected = sorted(rng.sample(indices, replacement_count))
        left, right, operation = group_key
        templates = _TRAINING_LEXICAL_TEMPLATES[operation]
        for contrast_index, example_index in enumerate(selected):
            result[example_index] = FormVariationExample(
                utterance=templates[contrast_index % len(templates)].format(left=left, right=right),
                target=f"OP={operation}",
                operation=operation,
                template_id=2_000 + contrast_index,
                left=left,
                right=right,
            )
    return tuple(result)


def discourse_cue_id(template_id: int) -> int:
    """Decode the counterbalanced cue family from a Phase-2.8 template ID."""
    if not 3_000 <= template_id < 4_000:
        raise ValueError(f"Not a Phase-2.8 discourse template ID: {template_id}")
    return (template_id - 3_000) % len(_TRAINING_DISCOURSE_CUES)


def build_counterfactual_discourse_examples(
    examples: Sequence[FormVariationExample],
    replacement_fraction: float,
    seed: int,
) -> tuple[FormVariationExample, ...]:
    """Replace a matched fraction with cue-balanced distractor utterances.

    Every operand pair receives the same cue IDs for all three operations, so
    the distractor clause cannot identify the requested operation by itself.
    """
    if not examples:
        raise ValueError("Counterfactual discourse balancing requires examples")
    if not 0 < replacement_fraction <= 1:
        raise ValueError("replacement_fraction must be in (0, 1]")
    grouped: dict[tuple[int, int, str], list[int]] = {}
    for index, example in enumerate(examples):
        if example.operation not in OPERATIONS:
            raise ValueError(f"Unsupported operation: {example.operation}")
        if example.template_id < 1_000:
            grouped.setdefault((example.left, example.right, example.operation), []).append(index)
    pair_rank = {pair: rank for rank, pair in enumerate(sorted({key[:2] for key in grouped}))}
    result = list(examples)
    rng = random.Random(seed)
    for left, right, operation in sorted(grouped):
        indices = grouped[(left, right, operation)]
        replacement_count = min(len(indices), max(1, int(round(len(indices) * replacement_fraction))))
        selected = sorted(rng.sample(indices, replacement_count))
        for offset, example_index in enumerate(selected):
            cue = (pair_rank[(left, right)] * replacement_count + offset) % len(_TRAINING_DISCOURSE_CUES)
            utterance = (_TRAINING_DISCOURSE_CUES[cue] + _TRAINING_DISCOURSE_REQUESTS[operation]).format(
                left=left, right=right
            )
            result[example_index] = FormVariationExample(
                utterance=utterance,
                target=f"OP={operation}",
                operation=operation,
                template_id=3_000 + cue,
                left=left,
                right=right,
            )
    return tuple(result)


def build_counterfactual_discourse_augmentation(
    examples: Sequence[FormVariationExample],
    augmentation_fraction: float,
    seed: int,
) -> tuple[FormVariationExample, ...]:
    """Append the counterbalanced discourse forms without deleting standards."""
    replaced = build_counterfactual_discourse_examples(
        examples, replacement_fraction=augmentation_fraction, seed=seed
    )
    additions = tuple(example for example in replaced if 3_000 <= example.template_id < 4_000)
    return tuple(examples) + additions


_PRESSURE_TRACKS = (
    "seen_form_unseen_operands",
    "held_out_templates",
    "lexical_shift",
    "syntax_order_reversal",
    "discourse_distractor",
    "minimal_contrast",
)
_ROBUST_PRESSURE_TRACKS = tuple(track for track in _PRESSURE_TRACKS if track != "seen_form_unseen_operands")


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


def build_phase25_stage_b_arms(
    splits: ConditionSplits,
    *,
    replacement_fraction: float,
    contrast_seed: int,
) -> dict[str, dict[str, object]]:
    """Register the Stage-B factorial while marking the immutable baseline reusable."""
    contrast_splits = ConditionSplits(
        train=build_contrast_balanced_examples(
            splits.train,
            replacement_fraction=replacement_fraction,
            seed=contrast_seed,
        ),
        seen_form=splits.seen_form,
        same_meaning_unseen_form=splits.same_meaning_unseen_form,
        unseen_operands_seen_form=splits.unseen_operands_seen_form,
        minimal_contrasts=splits.minimal_contrasts,
    )
    return {
        "baseline": {"splits": splits, "target_mode": "operation", "requires_training": False},
        "minimal_contrast": {
            "splits": contrast_splits,
            "target_mode": "operation",
            "requires_training": True,
        },
        "typed_frame": {"splits": splits, "target_mode": "typed_frame", "requires_training": True},
        "typed_frame_contrast": {
            "splits": contrast_splits,
            "target_mode": "typed_frame",
            "requires_training": True,
        },
    }


def build_phase26_contrast_arms(
    splits: ConditionSplits,
    *,
    replacement_fraction: float,
    minimal_seed: int,
    lexical_seed: int,
) -> dict[str, dict[str, object]]:
    """Register the minimal × lexical coverage factorial for Phase 2.6."""

    def replace_training(training: Sequence[FormVariationExample]) -> ConditionSplits:
        return ConditionSplits(
            train=tuple(training),
            seen_form=splits.seen_form,
            same_meaning_unseen_form=splits.same_meaning_unseen_form,
            unseen_operands_seen_form=splits.unseen_operands_seen_form,
            minimal_contrasts=splits.minimal_contrasts,
        )

    minimal_training = build_contrast_balanced_examples(
        splits.train,
        replacement_fraction=replacement_fraction,
        seed=minimal_seed,
    )
    lexical_training = build_lexical_contrast_examples(
        splits.train,
        replacement_fraction=replacement_fraction,
        seed=lexical_seed,
    )
    combined_training = build_lexical_contrast_examples(
        minimal_training,
        replacement_fraction=replacement_fraction,
        seed=lexical_seed,
    )
    return {
        "baseline": {"splits": splits, "target_mode": "operation", "requires_training": False},
        "minimal_contrast": {
            "splits": replace_training(minimal_training),
            "target_mode": "operation",
            "requires_training": False,
        },
        "lexical_contrast": {
            "splits": replace_training(lexical_training),
            "target_mode": "operation",
            "requires_training": True,
        },
        "minimal_lexical_contrast": {
            "splits": replace_training(combined_training),
            "target_mode": "operation",
            "requires_training": True,
        },
    }


def build_phase26_confirmation_arms(
    splits: ConditionSplits,
    *,
    replacement_fraction: float,
    minimal_seed: int,
    lexical_seed: int,
) -> dict[str, dict[str, object]]:
    """Build independent baseline and selected-arm data for sealed confirmation."""
    screen_arms = build_phase26_contrast_arms(
        splits,
        replacement_fraction=replacement_fraction,
        minimal_seed=minimal_seed,
        lexical_seed=lexical_seed,
    )
    return {
        arm_name: {
            "splits": screen_arms[arm_name]["splits"],
            "target_mode": "operation",
            "requires_training": True,
        }
        for arm_name in ("baseline", "minimal_lexical_contrast")
    }


def build_phase28_discourse_arms(
    splits: ConditionSplits,
    *,
    replacement_fraction: float,
    discourse_seed: int,
) -> dict[str, dict[str, object]]:
    """Build fresh baseline and counterfactual-discourse screen arms."""
    discourse_splits = ConditionSplits(
        train=build_counterfactual_discourse_examples(
            splits.train, replacement_fraction=replacement_fraction, seed=discourse_seed
        ),
        seen_form=splits.seen_form,
        same_meaning_unseen_form=splits.same_meaning_unseen_form,
        unseen_operands_seen_form=splits.unseen_operands_seen_form,
        minimal_contrasts=splits.minimal_contrasts,
    )
    return {
        "baseline": {"splits": splits, "target_mode": "operation", "requires_training": True},
        "counterfactual_discourse": {
            "splits": discourse_splits,
            "target_mode": "operation",
            "requires_training": True,
        },
    }


def build_phase29_augmentation_arms(
    splits: ConditionSplits,
    *,
    augmentation_fraction: float,
    discourse_seed: int,
) -> dict[str, dict[str, object]]:
    """Build Phase 2.9's reused and newly trained augmentation cells."""
    replacement = build_phase28_discourse_arms(
        splits, replacement_fraction=augmentation_fraction, discourse_seed=discourse_seed
    )["counterfactual_discourse"]["splits"]
    if not isinstance(replacement, ConditionSplits):
        raise ValueError("Invalid Phase-2.9 replacement splits")
    augmented = ConditionSplits(
        train=build_counterfactual_discourse_augmentation(
            splits.train, augmentation_fraction=augmentation_fraction, seed=discourse_seed
        ),
        seen_form=splits.seen_form,
        same_meaning_unseen_form=splits.same_meaning_unseen_form,
        unseen_operands_seen_form=splits.unseen_operands_seen_form,
        minimal_contrasts=splits.minimal_contrasts,
    )
    return {
        "baseline": {"splits": splits, "target_mode": "operation", "requires_training": False},
        "counterfactual_replacement": {
            "splits": replacement,
            "target_mode": "operation",
            "requires_training": False,
        },
        "augmentation_fixed_updates": {
            "splits": augmented,
            "target_mode": "operation",
            "requires_training": True,
        },
        "augmentation_matched_exposure": {
            "splits": augmented,
            "target_mode": "operation",
            "requires_training": True,
        },
    }


def _pressure_example(
    utterance: str,
    operation: str,
    template_id: int,
    left: int,
    right: int,
) -> FormVariationExample:
    """Build a fixed-pressure example without coupling it to a train template."""
    return FormVariationExample(
        utterance=utterance.format(left=left, right=right),
        target=f"OP={operation}",
        operation=operation,
        template_id=template_id,
        left=left,
        right=right,
    )


def build_fixed_pressure_test(
    pairs: Sequence[tuple[int, int]],
) -> dict[str, tuple[FormVariationExample, ...]]:
    """Build the fixed, tokenizer-excluded robustness suite for every condition.

    The suite deliberately stays constant as training diversity K changes. This
    makes adjacent accuracy changes attributable to the training condition,
    rather than a changing held-out set.
    """
    generator = FormVariationGenerator(seed=0)
    groups: dict[str, list[FormVariationExample]] = {track: [] for track in _PRESSURE_TRACKS}
    fresh_templates = {
        "lexical_shift": {
            "ADD": "How much do {left} and {right} make in all?",
            "SUBTRACT": "By what amount does {left} outrun {right}?",
            "COMPARE": "Is {left} above {right}?",
        },
        "syntax_order_reversal": {
            "ADD": "When {left} is added to {right}, what number results?",
            "SUBTRACT": "From {left}, take {right}; what is the result?",
            "COMPARE": "Does the first number, {left}, exceed the second, {right}?",
        },
        "discourse_distractor": {
            "ADD": "Although {left} exceeds {right}, give the total of {left} zols and {right} binks.",
            "SUBTRACT": "The groups belong together, but remove {right} binks from {left} zols. How many remain?",
            "COMPARE": "A total may be useful later. First decide whether {left} zols outnumber {right} binks.",
        },
        "minimal_contrast": {
            "ADD": "Two ledgers list {left} zols and {right} binks. Give their total.",
            "SUBTRACT": "Two ledgers list {left} zols and {right} binks. Give how many more zols there are.",
            "COMPARE": "Two ledgers list {left} zols and {right} binks. Decide whether zols outnumber binks.",
        },
    }
    for left, right in pairs:
        for operation in OPERATIONS:
            groups["seen_form_unseen_operands"].append(generator.render_variants(operation, left, right)[0])
            groups["held_out_templates"].extend(generator.render_variants(operation, left, right)[8:])
            for track, templates in fresh_templates.items():
                groups[track].append(
                    _pressure_example(templates[operation], operation, 100 + len(groups[track]), left, right)
                )
    return {track: tuple(groups[track]) for track in _PRESSURE_TRACKS}


def build_phase26_sealed_pressure_test(
    pairs: Sequence[tuple[int, int]],
) -> dict[str, tuple[FormVariationExample, ...]]:
    """Build a fresh confirmation-only suite with distinct wording and operands."""
    sealed_templates = {
        "held_out_templates": {
            "ADD": "State the numerical total formed from {left} and {right}.",
            "SUBTRACT": "State the numerical difference of {left} relative to {right}.",
            "COMPARE": "Assess whether the quantity {left} is greater than {right}.",
        },
        "lexical_shift": {
            "ADD": "What aggregate is obtained by uniting {left} with {right}?",
            "SUBTRACT": "What surplus does {left} retain over {right}?",
            "COMPARE": "Does {left} eclipse {right}?",
        },
        "syntax_order_reversal": {
            "ADD": "With {right} joined to {left}, determine the total.",
            "SUBTRACT": "After deducting {right} from {left}, determine what remains.",
            "COMPARE": "Relative to {right}, is {left} the larger value?",
        },
        "discourse_distractor": {
            "ADD": "The values may differ, yet calculate the total of {left} and {right}.",
            "SUBTRACT": "An aggregate is irrelevant; calculate {left} less {right}.",
            "COMPARE": "A difference could be computed later; first assess whether {left} exceeds {right}.",
        },
        "minimal_contrast": {
            "ADD": "A record contains {left} zols plus {right} binks. Report the combined count.",
            "SUBTRACT": "A record contains {left} zols and {right} binks. Report the zol excess.",
            "COMPARE": "A record contains {left} zols and {right} binks. Are zols more numerous?",
        },
    }
    groups: dict[str, list[FormVariationExample]] = {track: [] for track in _ROBUST_PRESSURE_TRACKS}
    for left, right in pairs:
        for operation in OPERATIONS:
            for track, templates in sealed_templates.items():
                groups[track].append(
                    _pressure_example(templates[operation], operation, 500 + len(groups[track]), left, right)
                )
    return {track: tuple(examples) for track, examples in groups.items()}


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


def write_training_tokenizer_corpus(examples: Iterable[FormVariationExample], path: Path) -> None:
    """Write only training examples for tokenizer fitting; never include evaluation text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(f"{example.utterance}\n{example.target}\n")


def train_fixed_byte_tokenizer(output_dir: Path) -> Tokenizer:
    """Create the same byte vocabulary for every factorial cell.

    No training text is consumed. This removes learned-tokenizer differences
    from the breadth/reinforcement comparison while retaining the model's
    ability to represent arbitrary held-out wording.
    """
    special_tokens = (
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
        "<FINAL>",
    )
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: index for index, token in enumerate(special_tokens)}
    vocab.update({token: index + len(vocab) for index, token in enumerate(alphabet) if token not in vocab})
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<UNK>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(list(special_tokens))
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    return tokenizer


def build_clean_split_manifest(
    training_examples: Sequence[FormVariationExample],
    pressure_groups: Mapping[str, Sequence[FormVariationExample]],
) -> dict[str, object]:
    """Produce a deterministic proof that tokenizer fitting excludes pressure text."""
    training_utterances = [example.utterance for example in training_examples]
    pressure_utterances = [example.utterance for examples in pressure_groups.values() for example in examples]
    training_corpus = "".join(f"{example.utterance}\n{example.target}\n" for example in training_examples)
    overlap = sorted(set(training_utterances).intersection(pressure_utterances))
    tokenizer_hash = hashlib.sha256(training_corpus.encode("utf-8")).hexdigest()
    return {
        "tokenizer_policy": "training_examples_only",
        "training_example_count": len(training_examples),
        "pressure_example_count": len(pressure_utterances),
        "training_unique_utterance_count": len(set(training_utterances)),
        "pressure_unique_utterance_count": len(set(pressure_utterances)),
        "exact_utterance_overlap_count": len(overlap),
        "training_corpus_sha256": tokenizer_hash,
        "tokenizer_corpus_sha256": tokenizer_hash,
        "validation": "PASS" if not overlap else "FAIL",
    }


def _utterance_set(groups: Mapping[str, Sequence[FormVariationExample]]) -> set[str]:
    return {example.utterance for examples in groups.values() for example in examples}


def build_phase25_split_manifest(
    training_examples: Sequence[FormVariationExample],
    development_groups: Mapping[str, Sequence[FormVariationExample]],
    *,
    sealed_groups: Mapping[str, Sequence[FormVariationExample]] | None = None,
) -> dict[str, object]:
    """Validate tokenizer/train, development, and optional sealed text isolation."""
    training_utterances = {example.utterance for example in training_examples}
    development_utterances = _utterance_set(development_groups)
    sealed_utterances = _utterance_set(sealed_groups or {})
    train_development = training_utterances.intersection(development_utterances)
    train_sealed = training_utterances.intersection(sealed_utterances)
    development_sealed = development_utterances.intersection(sealed_utterances)
    training_corpus = "".join(f"{example.utterance}\n{example.target}\n" for example in training_examples)
    training_hash = hashlib.sha256(training_corpus.encode("utf-8")).hexdigest()
    valid = not (train_development or train_sealed or development_sealed)
    return {
        "tokenizer_policy": "training_examples_only",
        "training_example_count": len(training_examples),
        "development_example_count": sum(len(examples) for examples in development_groups.values()),
        "sealed_example_count": sum(len(examples) for examples in (sealed_groups or {}).values()),
        "train_development_exact_overlap_count": len(train_development),
        "train_sealed_exact_overlap_count": len(train_sealed),
        "development_sealed_exact_overlap_count": len(development_sealed),
        "training_corpus_sha256": training_hash,
        "tokenizer_corpus_sha256": training_hash,
        "validation": "PASS" if valid else "FAIL",
    }


def compute_phase25_training_fingerprint(
    examples: Sequence[FormVariationExample],
    schedule: Mapping[str, object],
    *,
    target_mode: str,
) -> str:
    """Hash all inputs that must match before a Phase-2 baseline is reused."""
    payload = {
        "examples": [
            {
                "utterance": example.utterance,
                "target": _with_target_mode(example, target_mode).target,
                "operation": example.operation,
                "left": example.left,
                "right": example.right,
                "template_id": example.template_id,
            }
            for example in examples
        ],
        "schedule": dict(schedule),
        "target_mode": target_mode,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_clean_split_manifests(config_path: str) -> dict[str, object]:
    """Persist per-K split manifests without running or rerunning training."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    experiment = config["form_variation_v2"]
    train_pairs = tuple(tuple(pair) for pair in experiment["train_pairs"])
    eval_pairs = tuple(tuple(pair) for pair in experiment["eval_pairs"])
    pressure_groups = build_fixed_pressure_test(tuple(tuple(pair) for pair in experiment["pressure_pairs"]))
    manifests = {
        str(variants): build_clean_split_manifest(
            build_condition_splits(variants, train_pairs, eval_pairs, seed=int(config["seed"])).train,
            pressure_groups,
        )
        for variants in (int(value) for value in experiment["variants_per_operation"])
    }
    report = {"experiment": "form_variation_clean_v2", "manifests": manifests}
    output_dir = Path(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "split_manifests.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


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
def extract_response_boundary_features(
    model: SyntheticTransformer,
    tokenizer: Tokenizer,
    examples: Sequence[FormVariationExample],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract one frozen final-layer vector at each assistant boundary."""
    bos_id = tokenizer.token_to_id("<BOS>")
    if bos_id is None:
        raise ValueError("Tokenizer is missing the BOS token")
    was_training = model.training
    model.eval()
    features: list[torch.Tensor] = []
    labels: list[int] = []
    for example in examples:
        prompt_ids = tokenizer.encode(f"<USER> {example.utterance}\n<ASSISTANT> ").ids
        input_ids = torch.tensor([[bos_id, *prompt_ids]], dtype=torch.long, device=device)
        if input_ids.shape[1] > model.config.max_position_embeddings:
            raise ValueError(
                f"Probe prompt exceeds max_position_embeddings={model.config.max_position_embeddings}: "
                f"{example.utterance}"
            )
        hidden = model.forward_hidden_states(input_ids)
        features.append(hidden[0, -1].detach().cpu())
        labels.append(_OPERATION_TO_LABEL[example.operation])
    if was_training:
        model.train()
    if not features:
        return (
            torch.empty((0, model.config.hidden_size), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
        )
    return torch.stack(features), torch.tensor(labels, dtype=torch.long)


def fit_frozen_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    evaluation_features: torch.Tensor,
    *,
    num_classes: int,
    seed: int,
) -> torch.Tensor:
    """Fit a deterministic least-squares linear probe on detached features."""
    del seed  # The closed-form fit is deterministic; retained for a stable runner API.
    if train_features.ndim != 2 or evaluation_features.ndim != 2:
        raise ValueError("Probe features must be rank-two tensors")
    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError("Probe feature and label counts differ")
    if train_features.shape[1] != evaluation_features.shape[1]:
        raise ValueError("Train and evaluation feature widths differ")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    x_train = train_features.detach().to(dtype=torch.float64, device="cpu")
    x_eval = evaluation_features.detach().to(dtype=torch.float64, device="cpu")
    labels = train_labels.detach().to(dtype=torch.long, device="cpu")
    if labels.numel() == 0 or int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise ValueError("Probe labels fall outside the registered class inventory")
    targets = torch.nn.functional.one_hot(labels, num_classes=num_classes).to(torch.float64)
    weights = torch.linalg.pinv(x_train) @ targets
    return torch.argmax(x_eval @ weights, dim=1).to(torch.long)


def fit_frozen_linear_regression(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    evaluation_features: torch.Tensor,
) -> torch.Tensor:
    """Fit a deterministic multi-output linear probe without updating the parser."""
    if train_features.ndim != 2 or train_targets.ndim != 2 or evaluation_features.ndim != 2:
        raise ValueError("Regression features and targets must be rank-two tensors")
    if train_features.shape[0] != train_targets.shape[0]:
        raise ValueError("Regression feature and target counts differ")
    if train_features.shape[1] != evaluation_features.shape[1]:
        raise ValueError("Train and evaluation feature widths differ")
    x_train = train_features.detach().to(dtype=torch.float64, device="cpu")
    y_train = train_targets.detach().to(dtype=torch.float64, device="cpu")
    x_eval = evaluation_features.detach().to(dtype=torch.float64, device="cpu")
    weights = torch.linalg.pinv(x_train) @ y_train
    return (x_eval @ weights).to(torch.float32)


def score_argument_role_probe(
    examples: Sequence[FormVariationExample],
    predicted_values: torch.Tensor,
) -> dict[str, float]:
    """Score numeric value recovery and whether canonical A/B roles are swapped."""
    if predicted_values.shape != (len(examples), 2):
        raise ValueError("Argument predictions must have shape [examples, 2]")
    expected = torch.tensor([[example.left, example.right] for example in examples], dtype=torch.float32)
    predictions = predicted_values.detach().to(dtype=torch.float32, device="cpu")
    correct_distance = torch.abs(predictions - expected).sum(dim=1)
    swapped_distance = torch.abs(predictions - expected.flip(dims=(1,))).sum(dim=1)
    role_accuracy = float((correct_distance <= swapped_distance).to(torch.float32).mean().item()) if examples else 0.0
    mean_absolute_error = float(torch.abs(predictions - expected).mean().item()) if examples else 0.0
    return {
        "canonical_role_accuracy": round(role_accuracy, 4),
        "mean_absolute_value_error": round(mean_absolute_error, 4),
    }


def score_typed_frame_predictions(
    expected_examples: Sequence[FormVariationExample],
    predicted_targets: Sequence[str],
) -> dict[str, float]:
    """Keep protocol, intent, argument binding, and exact-frame scores separate."""
    if len(expected_examples) != len(predicted_targets):
        raise ValueError("Expected and predicted typed-frame counts differ")
    valid = operation_correct = binding_correct = exact = 0
    for example, prediction in zip(expected_examples, predicted_targets):
        try:
            parsed = parse_typed_frame(prediction)
        except ValueError:
            continue
        valid += 1
        operation_match = parsed["operation"] == example.operation
        binding_match = parsed["A"] == example.left and parsed["B"] == example.right
        operation_correct += int(operation_match)
        binding_correct += int(binding_match)
        exact += int(operation_match and binding_match)
    denominator = max(1, len(expected_examples))
    return {
        "operation_accuracy": operation_correct / denominator,
        "argument_binding_accuracy": binding_correct / denominator,
        "full_frame_exact_match": exact / denominator,
        "protocol_validity": valid / denominator,
    }


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


@torch.no_grad()
def predict_operation_labels_batched(
    model: SyntheticTransformer,
    tokenizer: Tokenizer,
    examples: Sequence[FormVariationExample],
    device: torch.device,
    batch_size: int = 64,
) -> tuple[str, ...]:
    """Return the lowest-loss canonical operation for every input in batches."""
    model.eval()
    candidates = tuple(f"OP={operation}" for operation in OPERATIONS)
    pad_id = tokenizer.token_to_id("<PAD>") or 0
    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer is missing BOS/EOS tokens")

    rows: list[tuple[int, list[int], list[int]]] = []
    for example_index, example in enumerate(examples):
        prompt_ids = tokenizer.encode(f"<USER> {example.utterance}\n<ASSISTANT> ").ids
        for candidate_index, candidate in enumerate(candidates):
            target_ids = tokenizer.encode(candidate).ids + [eos_id]
            input_ids = [bos_id] + prompt_ids + target_ids
            labels = [-100] * (1 + len(prompt_ids)) + target_ids
            rows.append((example_index * len(candidates) + candidate_index, input_ids, labels))

    losses: list[float] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        max_length = max(len(row[1]) for row in chunk)
        inputs = torch.full((len(chunk), max_length), pad_id, dtype=torch.long, device=device)
        labels = torch.full((len(chunk), max_length), -100, dtype=torch.long, device=device)
        for row_index, (_, input_ids, row_labels) in enumerate(chunk):
            inputs[row_index, : len(input_ids)] = torch.tensor(input_ids, dtype=torch.long, device=device)
            labels[row_index, : len(row_labels)] = torch.tensor(row_labels, dtype=torch.long, device=device)
        logits, _ = model(inputs)
        token_losses = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape(labels[:, 1:].shape)
        target_mask = labels[:, 1:] != -100
        losses.extend(
            ((token_losses * target_mask).sum(dim=1) / target_mask.sum(dim=1).clamp_min(1)).detach().cpu().tolist()
        )

    predictions: list[str] = []
    for example_index, example in enumerate(examples):
        candidate_losses = losses[example_index * len(candidates) : (example_index + 1) * len(candidates)]
        predicted = candidates[min(range(len(candidates)), key=lambda index: candidate_losses[index])]
        predictions.append(predicted)
    return tuple(predictions)


@torch.no_grad()
def evaluate_operation_accuracy_batched(
    model: SyntheticTransformer,
    tokenizer: Tokenizer,
    examples: Sequence[FormVariationExample],
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Evaluate canonical operation accuracy using batched candidate scoring."""
    predictions = predict_operation_labels_batched(model, tokenizer, examples, device, batch_size=batch_size)
    return sum(prediction == example.target for example, prediction in zip(examples, predictions)) / max(1, len(examples))


def operation_confusion_matrix(
    examples: Sequence[FormVariationExample], predictions: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Count canonical operation confusions while retaining every evaluated case."""
    if len(examples) != len(predictions):
        raise ValueError("Examples and predictions must have equal length")
    matrix = {actual: {predicted: 0 for predicted in OPERATIONS} for actual in OPERATIONS}
    for example, prediction in zip(examples, predictions):
        predicted_operation = prediction.removeprefix("OP=")
        if predicted_operation not in OPERATIONS:
            raise ValueError(f"Invalid operation prediction: {prediction}")
        matrix[example.operation][predicted_operation] += 1
    return matrix


@torch.no_grad()
def predict_typed_frames_batched(
    model: SyntheticTransformer,
    tokenizer: Tokenizer,
    examples: Sequence[FormVariationExample],
    device: torch.device,
    batch_size: int = 64,
) -> tuple[str, ...]:
    """Choose among canonical operation and A/B-binding frame candidates."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    pad_id = tokenizer.token_to_id("<PAD>")
    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    if pad_id is None or bos_id is None or eos_id is None:
        raise ValueError("Tokenizer is missing PAD/BOS/EOS tokens")

    candidate_sets: list[tuple[str, ...]] = []
    rows: list[tuple[list[int], list[int]]] = []
    for example in examples:
        bindings = ((example.left, example.right),)
        if example.left != example.right:
            bindings = (*bindings, (example.right, example.left))
        candidates = tuple(
            f"OP={operation};A={left};B={right}"
            for operation in OPERATIONS
            for left, right in bindings
        )
        candidate_sets.append(candidates)
        prompt_ids = tokenizer.encode(f"<USER> {example.utterance}\n<ASSISTANT> ").ids
        for candidate in candidates:
            target_ids = tokenizer.encode(candidate).ids + [eos_id]
            input_ids = [bos_id] + prompt_ids + target_ids
            if len(input_ids) > model.config.max_position_embeddings:
                raise ValueError(
                    f"Typed candidate exceeds max_position_embeddings={model.config.max_position_embeddings}: "
                    f"{example.utterance}"
                )
            rows.append((input_ids, [-100] * (1 + len(prompt_ids)) + target_ids))

    losses: list[float] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        max_length = max(len(input_ids) for input_ids, _ in chunk)
        inputs = torch.full((len(chunk), max_length), pad_id, dtype=torch.long, device=device)
        labels = torch.full((len(chunk), max_length), -100, dtype=torch.long, device=device)
        for row_index, (input_ids, row_labels) in enumerate(chunk):
            inputs[row_index, : len(input_ids)] = torch.tensor(input_ids, dtype=torch.long, device=device)
            labels[row_index, : len(row_labels)] = torch.tensor(row_labels, dtype=torch.long, device=device)
        logits, _ = model(inputs)
        token_losses = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape(labels[:, 1:].shape)
        target_mask = labels[:, 1:] != -100
        losses.extend(
            ((token_losses * target_mask).sum(dim=1) / target_mask.sum(dim=1).clamp_min(1)).cpu().tolist()
        )

    predictions: list[str] = []
    cursor = 0
    for candidates in candidate_sets:
        candidate_losses = losses[cursor : cursor + len(candidates)]
        predictions.append(candidates[min(range(len(candidates)), key=lambda index: candidate_losses[index])])
        cursor += len(candidates)
    return tuple(predictions)


def evaluate_typed_frame_metrics_batched(
    model: SyntheticTransformer,
    tokenizer: Tokenizer,
    examples: Sequence[FormVariationExample],
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, float]:
    predictions = predict_typed_frames_batched(model, tokenizer, examples, device, batch_size=batch_size)
    return score_typed_frame_predictions(examples, predictions)


def steps_for_exposure(dataset_size: int, batch_size: int, exposure: float) -> int:
    """Convert target presentations per unique example into full-batch updates."""
    if dataset_size <= 0 or batch_size <= 0 or exposure <= 0:
        raise ValueError("dataset_size, batch_size, and exposure must be positive")
    return max(1, int(round(dataset_size * exposure / batch_size)))


def warmup_steps_for_training(max_steps: int, fraction: float = 0.1) -> int:
    """Scale warmup with each cell's budget so schedules remain comparable."""
    if max_steps <= 0 or not 0 < fraction < 1:
        raise ValueError("max_steps must be positive and fraction must be in (0, 1)")
    return max(1, int(round(max_steps * fraction)))


def _device_from_name(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _transformer_config_from(tokenizer: Tokenizer, config: Mapping[str, object]) -> TransformerConfig:
    model_cfg = config["model"]
    if not isinstance(model_cfg, Mapping):
        raise ValueError("Model configuration must be a mapping")
    return TransformerConfig(
        vocab_size=tokenizer.get_vocab_size(),
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


def run_condition(
    splits: ConditionSplits,
    output_dir: Path,
    config: dict,
    seed: int,
    pressure_groups: Mapping[str, Sequence[FormVariationExample]] | None = None,
    run_metadata: Mapping[str, object] | None = None,
    target_mode: str = "operation",
) -> dict:
    """Train one scratch semantic parser and evaluate all controlled partitions."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    effective_splits = condition_splits_with_target_mode(splits, target_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "tokenizer_corpus.txt"
    write_training_tokenizer_corpus(effective_splits.train, corpus_path)
    if config["tokenizer"].get("mode") == "fixed_byte":
        tokenizer = train_fixed_byte_tokenizer(output_dir / "tokenizer")
    else:
        tokenizer = train_synthetic_tokenizer(
            corpus_file=str(corpus_path),
            output_dir=str(output_dir / "tokenizer"),
            vocab_size=int(config["tokenizer"]["vocab_size"]),
            min_frequency=int(config["tokenizer"]["min_frequency"]),
        )

    transformer_config = _transformer_config_from(tokenizer, config)
    device = _device_from_name(str(config["training"].get("device", "auto")))
    model = SyntheticTransformer(transformer_config).to(device)
    dataset = _SemanticParseDataset(effective_splits.train, tokenizer, transformer_config.max_position_embeddings)
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

    evaluation_batch_size = int(config.get("evaluation", {}).get("batch_size", 1))
    if evaluation_batch_size > 1:
        score_operation_accuracy = lambda examples: evaluate_operation_accuracy_batched(
            model, tokenizer, examples, device, batch_size=evaluation_batch_size
        )
    else:
        score_operation_accuracy = lambda examples: evaluate_operation_accuracy(model, tokenizer, examples, device)

    def score_examples(examples: Sequence[FormVariationExample]) -> dict[str, float]:
        if target_mode == "operation":
            return {"operation_accuracy": score_operation_accuracy(examples)}
        if target_mode == "typed_frame":
            return evaluate_typed_frame_metrics_batched(
                model,
                tokenizer,
                examples,
                device,
                batch_size=max(1, evaluation_batch_size),
            )
        raise ValueError(f"Unsupported parser target mode: {target_mode}")

    seen_scores = score_examples(effective_splits.seen_form)
    unseen_scores = score_examples(effective_splits.same_meaning_unseen_form)
    operand_scores = score_examples(effective_splits.unseen_operands_seen_form)
    contrast_scores = score_examples(effective_splits.minimal_contrasts)
    metrics: dict[str, object] = {
        "seed": seed,
        "train_examples": len(effective_splits.train),
        "steps": step,
        "wall_clock_seconds": round(time.time() - started, 2),
        "parser_target_mode": target_mode,
        "seen_form_accuracy": round(seen_scores["operation_accuracy"], 4),
        "unseen_form_accuracy": round(unseen_scores["operation_accuracy"], 4),
        "unseen_operands_seen_form_accuracy": round(operand_scores["operation_accuracy"], 4),
        "minimal_contrast_accuracy": round(contrast_scores["operation_accuracy"], 4),
        "timer": timer.get_summary(),
    }
    if target_mode == "typed_frame":
        for prefix, scores in (
            ("seen_form", seen_scores),
            ("unseen_form", unseen_scores),
            ("unseen_operands_seen_form", operand_scores),
            ("minimal_contrast", contrast_scores),
        ):
            metrics[f"{prefix}_argument_binding_accuracy"] = round(scores["argument_binding_accuracy"], 4)
            metrics[f"{prefix}_full_frame_exact_match"] = round(scores["full_frame_exact_match"], 4)
            metrics[f"{prefix}_protocol_validity"] = round(scores["protocol_validity"], 4)
    if pressure_groups is not None:
        unknown_groups = set(pressure_groups).difference(_PRESSURE_TRACKS)
        if unknown_groups:
            raise ValueError(f"Unknown pressure-test tracks: {sorted(unknown_groups)}")
        group_details = {track: score_examples(examples) for track, examples in pressure_groups.items()}
        group_metrics = {
            track: round(group_details[track]["operation_accuracy"], 4) for track in pressure_groups
        }
        metrics["pressure_groups"] = group_metrics
        if target_mode == "typed_frame":
            metrics["pressure_group_details"] = {
                track: {metric: round(value, 4) for metric, value in scores.items()}
                for track, scores in group_details.items()
            }
        metrics["worst_robust_accuracy"] = round(
            min(group_metrics[track] for track in _ROBUST_PRESSURE_TRACKS), 4
        )
    if run_metadata:
        metrics.update(run_metadata)
    timer.export_csv("step_metrics.csv")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    torch.save(model.state_dict(), output_dir / "parser_final.pt")
    return metrics


def run_clean_ablation(config_path: str) -> dict:
    """Run the Phase-0/1 clean form-diversity curve on one fixed pressure suite."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    experiment = config["form_variation_v2"]
    output_dir = Path(experiment["output_dir"])
    conditions = tuple(int(value) for value in experiment["variants_per_operation"])
    seeds = tuple(int(value) for value in experiment["seeds"])
    train_pairs = tuple(tuple(pair) for pair in experiment["train_pairs"])
    eval_pairs = tuple(tuple(pair) for pair in experiment["eval_pairs"])
    pressure_pairs = tuple(tuple(pair) for pair in experiment["pressure_pairs"])
    pressure_groups = build_fixed_pressure_test(pressure_pairs)
    write_clean_split_manifests(config_path)

    results: dict[str, list[dict]] = {}
    for variants in conditions:
        splits = build_condition_splits(variants, train_pairs, eval_pairs, seed=int(config["seed"]))
        condition_results = []
        for seed in seeds:
            run_dir = output_dir / f"variants_{variants}" / f"seed_{seed}"
            condition_results.append(run_condition(splits, run_dir, config, seed, pressure_groups=pressure_groups))
        results[str(variants)] = condition_results

    summary: dict[str, dict[str, object]] = {}
    for variants, condition_results in results.items():
        pressure_summary = {
            track: round(
                sum(float(result["pressure_groups"][track]) for result in condition_results) / len(condition_results), 4
            )
            for track in _PRESSURE_TRACKS
        }
        summary[variants] = {
            "worst_robust_accuracy": round(
                sum(float(result["worst_robust_accuracy"]) for result in condition_results) / len(condition_results), 4
            ),
            "pressure_groups": pressure_summary,
            "seen_form_accuracy": round(
                sum(float(result["seen_form_accuracy"]) for result in condition_results) / len(condition_results), 4
            ),
        }

    report = {
        "experiment": "form_variation_clean_v2",
        "config": config_path,
        "tokenizer_policy": "training_examples_only",
        "conditions": list(conditions),
        "seeds": list(seeds),
        "pressure_tracks": list(_PRESSURE_TRACKS),
        "robust_pressure_tracks": list(_ROBUST_PRESSURE_TRACKS),
        "summary": summary,
        "runs": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    _write_clean_analysis(report, output_dir / "analysis.json")
    return report


def run_breadth_reinforcement(config_path: str) -> dict:
    """Run the registered K-by-exposure factorial with a frozen byte tokenizer."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    experiment = config["breadth_reinforcement"]
    output_dir = Path(experiment["output_dir"])
    conditions = tuple(int(value) for value in experiment["variants_per_operation"])
    exposures = tuple(float(value) for value in experiment["exposures"])
    seeds = tuple(int(value) for value in experiment["seeds"])
    train_pairs = tuple(tuple(pair) for pair in experiment["train_pairs"])
    eval_pairs = tuple(tuple(pair) for pair in experiment["eval_pairs"])
    pressure_groups = build_fixed_pressure_test(tuple(tuple(pair) for pair in experiment["pressure_pairs"]))
    batch_size = int(config["training"]["batch_size"])
    evaluation_batch_size = int(config.get("evaluation", {}).get("batch_size", 64))

    manifests: dict[str, object] = {}
    results: dict[str, dict[str, list[dict]]] = {}
    for variants in conditions:
        splits = build_condition_splits(variants, train_pairs, eval_pairs, seed=int(config["seed"]))
        manifests[str(variants)] = build_clean_split_manifest(splits.train, pressure_groups)
        results[str(variants)] = {}
        for exposure in exposures:
            exposure_key = f"R{round(exposure):g}"
            steps = steps_for_exposure(len(splits.train), batch_size, exposure)
            condition_config = copy.deepcopy(config)
            condition_config["training"]["max_steps"] = steps
            condition_config["training"]["warmup_steps"] = warmup_steps_for_training(steps)
            condition_config["evaluation"]["batch_size"] = evaluation_batch_size
            results[str(variants)][exposure_key] = []
            for seed in seeds:
                run_dir = output_dir / f"variants_{variants}" / exposure_key / f"seed_{seed}"
                run_metadata = {
                    "variants_per_operation": variants,
                    "exposure_target": exposure,
                    "exposure_actual": round(steps * batch_size / len(splits.train), 6),
                    "evaluation_batch_size": evaluation_batch_size,
                    "tokenizer_mode": "fixed_byte",
                }
                result = run_condition(
                    splits,
                    run_dir,
                    condition_config,
                    seed,
                    pressure_groups=pressure_groups,
                    run_metadata=run_metadata,
                )
                results[str(variants)][exposure_key].append(result)

    report = {
        "experiment": str(config.get("name", "breadth_reinforcement_factorial")),
        "config": config_path,
        "tokenizer_policy": "fixed_byte_vocabulary",
        "conditions": list(conditions),
        "exposures": {f"R{round(exposure):g}": exposure for exposure in exposures},
        "seeds": list(seeds),
        "pressure_tracks": list(_PRESSURE_TRACKS),
        "robust_pressure_tracks": list(_ROBUST_PRESSURE_TRACKS),
        "split_manifests": manifests,
        "runs": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "split_manifests.json").open("w", encoding="utf-8") as handle:
        json.dump({"experiment": report["experiment"], "manifests": manifests}, handle, indent=2)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    _write_breadth_analysis(report, output_dir / "analysis.json")
    return report


def _load_phase25_config(config_path: str) -> tuple[dict[str, object], dict[str, object]]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "phase25_representation" not in config:
        raise ValueError("Configuration is missing phase25_representation")
    experiment = config["phase25_representation"]
    if not isinstance(experiment, dict):
        raise ValueError("phase25_representation must be a mapping")
    return config, experiment


def _phase25_splits(config: Mapping[str, object], experiment: Mapping[str, object], variants: int) -> ConditionSplits:
    return build_condition_splits(
        variants,
        tuple(tuple(pair) for pair in experiment["train_pairs"]),
        tuple(tuple(pair) for pair in experiment["eval_pairs"]),
        seed=int(config["seed"]),
    )


def _phase25_schedule(config: Mapping[str, object], experiment: Mapping[str, object], dataset_size: int) -> dict[str, int]:
    training = config["training"]
    if not isinstance(training, Mapping):
        raise ValueError("training must be a mapping")
    batch_size = int(training["batch_size"])
    max_steps = steps_for_exposure(dataset_size, batch_size, float(experiment["screen_exposure"]))
    return {
        "batch_size": batch_size,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps_for_training(max_steps),
    }


def prepare_phase25(config_path: str) -> dict[str, object]:
    """Write the Stage-A/B registration and manifests without loading or training a model."""
    config, experiment = _load_phase25_config(config_path)
    variants = int(experiment["screen_variants"])
    splits = _phase25_splits(config, experiment, variants)
    arms = build_phase25_stage_b_arms(
        splits,
        replacement_fraction=float(experiment["contrast_replacement_fraction"]),
        contrast_seed=int(experiment["contrast_seed"]),
    )
    pressure_groups = build_fixed_pressure_test(tuple(tuple(pair) for pair in experiment["pressure_pairs"]))
    schedule = _phase25_schedule(config, experiment, len(splits.train))
    fingerprint_controls: dict[str, object] = {
        "training": schedule,
        "model": config["model"],
        "tokenizer": config["tokenizer"],
    }
    manifests: dict[str, object] = {}
    registration_arms: dict[str, object] = {}
    for arm_name, arm in arms.items():
        arm_splits = arm["splits"]
        if not isinstance(arm_splits, ConditionSplits):
            raise ValueError(f"Arm {arm_name} has invalid splits")
        target_mode = str(arm["target_mode"])
        effective_training = condition_splits_with_target_mode(arm_splits, target_mode).train
        manifests[arm_name] = build_phase25_split_manifest(effective_training, pressure_groups)
        registration_arms[arm_name] = {
            "requires_training": bool(arm["requires_training"]),
            "target_mode": target_mode,
            "training_examples": len(effective_training),
            "training_fingerprint": compute_phase25_training_fingerprint(
                arm_splits.train,
                fingerprint_controls,
                target_mode=target_mode,
            ),
        }
    if any(manifest["validation"] != "PASS" for manifest in manifests.values()):
        raise ValueError("A Phase-2.5 train/development split manifest failed")
    registration = {
        "experiment": str(config.get("name", "phase25_representation")),
        "config": config_path,
        "status": "registered_not_run",
        "screen_variants": variants,
        "screen_exposure": float(experiment["screen_exposure"]),
        "seeds": [int(seed) for seed in experiment["seeds"]],
        "schedule": schedule,
        "sealed_suite_status": "not_created_not_opened",
        "arms": registration_arms,
    }
    baseline_runs = _phase25_baseline_runs(config, experiment, splits, schedule)
    registration["baseline_reuse_validation"] = {
        "validation": "PASS",
        "validated_seeds": [int(run["seed"]) for run in baseline_runs],
        "source_results": str(experiment["source_results"]),
        "source_run_dir": str(experiment["source_run_dir"]),
    }
    output_dir = Path(str(experiment["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "registration.json").open("w", encoding="utf-8") as handle:
        json.dump(registration, handle, indent=2)
    with (output_dir / "split_manifests.json").open("w", encoding="utf-8") as handle:
        json.dump({"experiment": registration["experiment"], "manifests": manifests}, handle, indent=2)
    return {"registration": registration, "manifests": manifests}


def _load_phase25_checkpoint(
    checkpoint_dir: Path,
    config: Mapping[str, object],
    device: torch.device,
) -> tuple[SyntheticTransformer, Tokenizer]:
    tokenizer_path = checkpoint_dir / "tokenizer" / "tokenizer.json"
    checkpoint_path = checkpoint_dir / "parser_final.pt"
    if not tokenizer_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"Incomplete Phase-2 source checkpoint: {checkpoint_dir}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = SyntheticTransformer(_transformer_config_from(tokenizer, config)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model, tokenizer


def _triplet_consistency(
    examples: Sequence[FormVariationExample],
    predictions: Sequence[int],
) -> float:
    grouped: dict[tuple[int, int], list[bool]] = {}
    for example, prediction in zip(examples, predictions):
        grouped.setdefault((example.left, example.right), []).append(
            prediction == _OPERATION_TO_LABEL[example.operation]
        )
    return sum(all(values) for values in grouped.values()) / max(1, len(grouped))


def run_phase25_probe(config_path: str) -> dict[str, object]:
    """Run Stage A only: fit closed-form probes on frozen Phase-2 checkpoints."""
    config, experiment = _load_phase25_config(config_path)
    source_root = Path(str(experiment["source_run_dir"]))
    exposure_key = str(experiment["source_exposure_key"])
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    variants_values = tuple(int(value) for value in experiment["probe_variants"])
    pressure_groups = build_fixed_pressure_test(tuple(tuple(pair) for pair in experiment["pressure_pairs"]))
    device = _device_from_name(str(config["training"].get("device", "auto")))
    all_runs: dict[str, list[dict[str, object]]] = {}

    for variants in variants_values:
        splits = _phase25_splits(config, experiment, variants)
        all_runs[str(variants)] = []
        for seed in seeds:
            checkpoint_dir = source_root / f"variants_{variants}" / exposure_key / f"seed_{seed}"
            model, tokenizer = _load_phase25_checkpoint(checkpoint_dir, config, device)
            train_features, train_labels = extract_response_boundary_features(
                model, tokenizer, splits.train, device
            )
            argument_targets = torch.tensor(
                [[example.left, example.right] for example in splits.train],
                dtype=torch.float32,
            )
            source_metrics_path = checkpoint_dir / "metrics.json"
            with source_metrics_path.open("r", encoding="utf-8") as handle:
                source_metrics = json.load(handle)
            groups: dict[str, object] = {}
            for track, examples in pressure_groups.items():
                features, labels = extract_response_boundary_features(model, tokenizer, examples, device)
                operation_predictions = fit_frozen_linear_probe(
                    train_features,
                    train_labels,
                    features,
                    num_classes=len(OPERATIONS),
                    seed=seed,
                )
                argument_predictions = fit_frozen_linear_regression(
                    train_features,
                    argument_targets,
                    features,
                )
                operation_accuracy = float((operation_predictions == labels).to(torch.float32).mean().item())
                group_result: dict[str, object] = {
                    "decoder_operation_accuracy": float(source_metrics["pressure_groups"][track]),
                    "probe_operation_accuracy": round(operation_accuracy, 4),
                    **score_argument_role_probe(examples, argument_predictions),
                }
                if track == "minimal_contrast":
                    group_result["minimal_contrast_triplet_consistency"] = round(
                        _triplet_consistency(examples, operation_predictions.tolist()), 4
                    )
                groups[track] = group_result
            robust_probe = [float(groups[track]["probe_operation_accuracy"]) for track in _ROBUST_PRESSURE_TRACKS]
            all_runs[str(variants)].append(
                {
                    "seed": seed,
                    "checkpoint": str(checkpoint_dir / "parser_final.pt"),
                    "groups": groups,
                    "worst_probe_operation_accuracy": min(robust_probe),
                    "macro_probe_operation_accuracy": sum(robust_probe) / len(robust_probe),
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = {
        variants: {
            "worst_probe_operation_accuracy": _distribution_summary(
                [float(run["worst_probe_operation_accuracy"]) for run in runs]
            ),
            "macro_probe_operation_accuracy": _distribution_summary(
                [float(run["macro_probe_operation_accuracy"]) for run in runs]
            ),
        }
        for variants, runs in all_runs.items()
    }
    report = {
        "experiment": str(config.get("name", "phase25_representation")),
        "stage": "A_frozen_checkpoint_probe",
        "source_exposure_key": exposure_key,
        "variants": list(variants_values),
        "seeds": list(seeds),
        "development_suite_only": True,
        "runs": all_runs,
        "summary": summary,
    }
    output_dir = Path(str(experiment["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "stage_a_probe_results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def _phase25_baseline_runs(
    config: Mapping[str, object],
    experiment: Mapping[str, object],
    splits: ConditionSplits,
    schedule: Mapping[str, object],
) -> list[dict[str, object]]:
    source_results_path = Path(str(experiment["source_results"]))
    with source_results_path.open("r", encoding="utf-8") as handle:
        source_report = json.load(handle)
    variants_key = str(int(experiment["screen_variants"]))
    exposure_key = str(experiment["source_exposure_key"])
    source_runs = {int(run["seed"]): run for run in source_report["runs"][variants_key][exposure_key]}
    expected_examples = condition_splits_with_target_mode(splits, "operation").train
    expected_corpus = "".join(f"{example.utterance}\n{example.target}\n" for example in expected_examples)
    expected_hash = hashlib.sha256(expected_corpus.encode("utf-8")).hexdigest()
    fingerprint_controls = {
        "training": dict(schedule),
        "model": config["model"],
        "tokenizer": config["tokenizer"],
    }
    fingerprint = compute_phase25_training_fingerprint(
        splits.train,
        fingerprint_controls,
        target_mode="operation",
    )
    source_root = Path(str(experiment["source_run_dir"]))
    result: list[dict[str, object]] = []
    for seed_value in experiment["seeds"]:
        seed = int(seed_value)
        if seed not in source_runs:
            raise ValueError(f"Missing reusable baseline seed {seed}")
        run_dir = source_root / f"variants_{variants_key}" / exposure_key / f"seed_{seed}"
        corpus_path = run_dir / "tokenizer_corpus.txt"
        checkpoint_path = run_dir / "parser_final.pt"
        if not corpus_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError(f"Incomplete reusable baseline: {run_dir}")
        normalized_corpus = corpus_path.read_text(encoding="utf-8")
        actual_hash = hashlib.sha256(normalized_corpus.encode("utf-8")).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Baseline training corpus mismatch for seed {seed}")
        run = dict(source_runs[seed])
        if int(run["steps"]) != int(schedule["max_steps"]):
            raise ValueError(f"Baseline schedule mismatch for seed {seed}")
        run["phase25_training_fingerprint"] = fingerprint
        run["reused_from"] = str(run_dir)
        result.append(run)
    return result


def run_phase25_screen(config_path: str) -> dict[str, object]:
    """Run Stage B's three new arms; reuse and validate the five-run baseline."""
    preparation = prepare_phase25(config_path)
    config, experiment = _load_phase25_config(config_path)
    variants = int(experiment["screen_variants"])
    splits = _phase25_splits(config, experiment, variants)
    arms = build_phase25_stage_b_arms(
        splits,
        replacement_fraction=float(experiment["contrast_replacement_fraction"]),
        contrast_seed=int(experiment["contrast_seed"]),
    )
    pressure_groups = build_fixed_pressure_test(tuple(tuple(pair) for pair in experiment["pressure_pairs"]))
    schedule = _phase25_schedule(config, experiment, len(splits.train))
    condition_config = copy.deepcopy(config)
    condition_config["training"]["max_steps"] = schedule["max_steps"]
    condition_config["training"]["warmup_steps"] = schedule["warmup_steps"]
    output_dir = Path(str(experiment["output_dir"]))
    results: dict[str, list[dict[str, object]]] = {
        "baseline": _phase25_baseline_runs(config, experiment, splits, schedule)
    }

    registration_arms = preparation["registration"]["arms"]
    for arm_name, arm in arms.items():
        if not bool(arm["requires_training"]):
            continue
        arm_splits = arm["splits"]
        if not isinstance(arm_splits, ConditionSplits):
            raise ValueError(f"Arm {arm_name} has invalid splits")
        target_mode = str(arm["target_mode"])
        fingerprint = str(registration_arms[arm_name]["training_fingerprint"])
        results[arm_name] = []
        for seed_value in experiment["seeds"]:
            seed = int(seed_value)
            run_dir = output_dir / "stage_b" / arm_name / f"seed_{seed}"
            metrics_path = run_dir / "metrics.json"
            checkpoint_path = run_dir / "parser_final.pt"
            if metrics_path.exists() or checkpoint_path.exists():
                if not metrics_path.exists() or not checkpoint_path.exists():
                    raise RuntimeError(f"Partial Phase-2.5 run requires manual inspection: {run_dir}")
                with metrics_path.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                if existing.get("phase25_training_fingerprint") != fingerprint:
                    raise RuntimeError(f"Refusing to reuse mismatched Phase-2.5 run: {run_dir}")
                results[arm_name].append(existing)
                continue
            results[arm_name].append(
                run_condition(
                    arm_splits,
                    run_dir,
                    condition_config,
                    seed,
                    pressure_groups=pressure_groups,
                    run_metadata={
                        "phase25_arm": arm_name,
                        "phase25_training_fingerprint": fingerprint,
                        "variants_per_operation": variants,
                        "exposure_target": float(experiment["screen_exposure"]),
                    },
                    target_mode=target_mode,
                )
            )

    report: dict[str, object] = {
        "experiment": str(config.get("name", "phase25_representation")),
        "stage": "B_K4_representation_screen",
        "config": config_path,
        "seeds": [int(seed) for seed in experiment["seeds"]],
        "schedule": schedule,
        "sealed_suite_status": "not_created_not_opened",
        "arms": results,
    }
    analysis = analyze_phase25_screen(report)
    with (output_dir / "stage_b_results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "stage_b_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    return {"report": report, "analysis": analysis}


def _bootstrap_interval(values: Sequence[float], samples: int = 10_000) -> tuple[float, float]:
    """Return a deterministic non-parametric 95% bootstrap interval for a mean."""
    if not values:
        raise ValueError("Bootstrap requires at least one value")
    if len(values) == 1:
        return (round(values[0], 4), round(values[0], 4))
    rng = random.Random(0)
    size = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(size)) / size for _ in range(samples))
    return (round(means[int(0.025 * samples)], 4), round(means[int(0.975 * samples) - 1], 4))


def _distribution_summary(values: Sequence[float]) -> dict[str, object]:
    """Summarize seed-level measurements without concealing seed variance."""
    mean = sum(values) / len(values)
    sample_std = math.sqrt(sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1))
    return {
        "n": len(values),
        "mean": round(mean, 4),
        "sample_std": round(sample_std, 4),
        "bootstrap_95_ci": list(_bootstrap_interval(values)),
    }


def analyze_phase25_screen(report: Mapping[str, object]) -> dict[str, object]:
    """Apply the registered paired-seed continuation gate to Stage-B arms."""
    raw_arms = report.get("arms")
    if not isinstance(raw_arms, Mapping) or "baseline" not in raw_arms:
        raise ValueError("Phase-2.5 report requires a baseline arm")

    def by_seed(raw_runs: object) -> dict[int, Mapping[str, object]]:
        if not isinstance(raw_runs, Sequence):
            raise ValueError("Phase-2.5 arm runs must be a sequence")
        result: dict[int, Mapping[str, object]] = {}
        for run in raw_runs:
            if not isinstance(run, Mapping):
                raise ValueError("Phase-2.5 arm contains an invalid run")
            result[int(run["seed"])] = run
        return result

    baseline = by_seed(raw_arms["baseline"])
    analysis: dict[str, object] = {
        "gate": {
            "minimum_worst_group_gain": 0.10,
            "paired_bootstrap_lower_bound_must_exceed": 0.0,
            "maximum_individual_track_regression": 0.05,
        },
        "arms": {},
        "selected_arm": None,
    }
    passing: list[tuple[float, str]] = []
    for arm_name, raw_runs in raw_arms.items():
        if arm_name == "baseline":
            continue
        arm = by_seed(raw_runs)
        seeds = sorted(set(baseline).intersection(arm))
        if not seeds:
            raise ValueError(f"Arm {arm_name} has no seeds paired with baseline")
        worst_deltas = [
            float(arm[seed]["worst_robust_accuracy"]) - float(baseline[seed]["worst_robust_accuracy"])
            for seed in seeds
        ]
        worst_summary = _distribution_summary(worst_deltas)
        track_deltas: dict[str, dict[str, object]] = {}
        for track in _ROBUST_PRESSURE_TRACKS:
            values = [
                float(arm[seed]["pressure_groups"][track])
                - float(baseline[seed]["pressure_groups"][track])
                for seed in seeds
            ]
            track_deltas[track] = _distribution_summary(values)
        no_track_regression = all(float(summary["mean"]) >= -0.05 for summary in track_deltas.values())
        passes = (
            float(worst_summary["mean"]) >= 0.10
            and float(worst_summary["bootstrap_95_ci"][0]) > 0.0
            and no_track_regression
        )
        analysis["arms"][arm_name] = {
            "paired_seeds": seeds,
            "worst_group_delta": worst_summary,
            "track_deltas": track_deltas,
            "no_track_regression_over_5pp": no_track_regression,
            "passes_continuation_gate": passes,
        }
        if passes:
            passing.append((float(worst_summary["mean"]), str(arm_name)))
    if passing:
        analysis["selected_arm"] = max(passing)[1]
    return analysis


def validate_phase26_reused_metrics(
    source_report: Mapping[str, object],
    expected_fingerprints: Mapping[str, str],
    *,
    seeds: Sequence[int],
    max_steps: int,
) -> dict[str, list[dict[str, object]]]:
    """Validate Phase-2.6's two reused arms before any new training starts."""
    raw_arms = source_report.get("arms")
    if not isinstance(raw_arms, Mapping):
        raise ValueError("Phase-2.6 source report is missing arms")
    reused: dict[str, list[dict[str, object]]] = {}
    for arm_name in ("baseline", "minimal_contrast"):
        raw_runs = raw_arms.get(arm_name)
        if not isinstance(raw_runs, Sequence):
            raise ValueError(f"Phase-2.6 source report is missing {arm_name}")
        by_seed: dict[int, dict[str, object]] = {}
        for raw_run in raw_runs:
            if not isinstance(raw_run, Mapping):
                raise ValueError(f"Invalid {arm_name} source run")
            run = dict(raw_run)
            seed = int(run["seed"])
            if seed in by_seed:
                raise ValueError(f"Duplicate {arm_name} source seed {seed}")
            by_seed[seed] = run
        reused[arm_name] = []
        for seed_value in seeds:
            seed = int(seed_value)
            if seed not in by_seed:
                raise ValueError(f"Missing {arm_name} source seed {seed}")
            run = by_seed[seed]
            if int(run["steps"]) != max_steps:
                raise ValueError(f"{arm_name} seed {seed} schedule mismatch")
            if run.get("phase25_training_fingerprint") != expected_fingerprints[arm_name]:
                raise ValueError(f"{arm_name} seed {seed} fingerprint mismatch")
            reused[arm_name].append(run)
    return reused


def analyze_phase26_screen(report: Mapping[str, object]) -> dict[str, object]:
    """Apply the continuation gate and estimate paired 2×2 factorial effects."""
    gate_analysis = analyze_phase25_screen(report)
    raw_arms = report.get("arms")
    if not isinstance(raw_arms, Mapping):
        raise ValueError("Phase-2.6 report is missing arms")
    required = ("baseline", "minimal_contrast", "lexical_contrast", "minimal_lexical_contrast")
    if any(arm not in raw_arms for arm in required):
        raise ValueError("Phase-2.6 report is missing a factorial arm")

    by_arm: dict[str, dict[int, Mapping[str, object]]] = {}
    for arm_name in required:
        runs = raw_arms[arm_name]
        if not isinstance(runs, Sequence):
            raise ValueError(f"Phase-2.6 arm {arm_name} is invalid")
        by_arm[arm_name] = {int(run["seed"]): run for run in runs if isinstance(run, Mapping)}
    seeds = sorted(set.intersection(*(set(runs) for runs in by_arm.values())))
    if not seeds:
        raise ValueError("Phase-2.6 factorial has no fully paired seeds")

    def metric(run: Mapping[str, object], metric_name: str) -> float:
        if metric_name == "worst_robust_accuracy":
            return float(run[metric_name])
        groups = run["pressure_groups"]
        return sum(float(groups[track]) for track in _ROBUST_PRESSURE_TRACKS) / len(_ROBUST_PRESSURE_TRACKS)

    factorial_effects: dict[str, object] = {}
    for metric_name in ("worst_robust_accuracy", "macro_robust_accuracy"):
        cells = {
            arm_name: {seed: metric(by_arm[arm_name][seed], metric_name) for seed in seeds}
            for arm_name in required
        }
        deltas = {
            "minimal_without_lexical": [
                cells["minimal_contrast"][seed] - cells["baseline"][seed] for seed in seeds
            ],
            "lexical_without_minimal": [
                cells["lexical_contrast"][seed] - cells["baseline"][seed] for seed in seeds
            ],
            "minimal_with_lexical": [
                cells["minimal_lexical_contrast"][seed] - cells["lexical_contrast"][seed]
                for seed in seeds
            ],
            "lexical_with_minimal": [
                cells["minimal_lexical_contrast"][seed] - cells["minimal_contrast"][seed]
                for seed in seeds
            ],
            "interaction": [
                cells["minimal_lexical_contrast"][seed]
                - cells["minimal_contrast"][seed]
                - cells["lexical_contrast"][seed]
                + cells["baseline"][seed]
                for seed in seeds
            ],
        }
        factorial_effects[metric_name] = {
            name: {**_distribution_summary(values), "paired_seeds": seeds} for name, values in deltas.items()
        }

    eligible = []
    for arm_name in ("lexical_contrast", "minimal_lexical_contrast"):
        arm_analysis = gate_analysis["arms"][arm_name]
        if arm_analysis["passes_continuation_gate"]:
            eligible.append((float(arm_analysis["worst_group_delta"]["mean"]), arm_name))
    gate_analysis["selected_arm"] = max(eligible)[1] if eligible else None
    gate_analysis["factorial_effects"] = factorial_effects
    return gate_analysis


def analyze_phase26_confirmation(report: Mapping[str, object]) -> dict[str, object]:
    """Apply Phase 2.6's preregistered gate independently at K=4 and K=8."""
    raw_runs = report.get("runs")
    if not isinstance(raw_runs, Mapping):
        raise ValueError("Phase-2.6 confirmation report is missing runs")
    by_variants: dict[str, object] = {}
    all_pass = True
    for variant, cells in raw_runs.items():
        if not isinstance(cells, Mapping):
            raise ValueError(f"Phase-2.6 confirmation K={variant} cells are invalid")
        gate = analyze_phase25_screen({"arms": cells})
        selected = gate["arms"].get("minimal_lexical_contrast")
        if not isinstance(selected, Mapping):
            raise ValueError(f"Phase-2.6 confirmation K={variant} is missing selected arm")
        passes = bool(selected["passes_continuation_gate"])
        by_variants[str(variant)] = {
            "selected_arm": "minimal_lexical_contrast",
            "worst_group_delta": selected["worst_group_delta"],
            "track_deltas": selected["track_deltas"],
            "no_track_regression_over_5pp": selected["no_track_regression_over_5pp"],
            "passes_confirmation_gate": passes,
        }
        all_pass = all_pass and passes
    if set(by_variants) != {"4", "8"}:
        raise ValueError("Phase-2.6 confirmation requires both K=4 and K=8")
    return {
        "gate": {
            "minimum_worst_group_gain": 0.10,
            "paired_bootstrap_lower_bound_must_exceed": 0.0,
            "maximum_individual_track_regression": 0.05,
            "required_breadths": [4, 8],
        },
        "by_variants": by_variants,
        "confirmation_passed": all_pass,
    }


def analyze_phase28_screen(report: Mapping[str, object]) -> dict[str, object]:
    """Apply the existing paired continuation gate to the Phase-2.8 screen."""
    analysis = analyze_phase25_screen(report)
    arm = analysis.get("arms", {}).get("counterfactual_discourse")
    analysis["selected_arm"] = (
        "counterfactual_discourse"
        if isinstance(arm, Mapping) and bool(arm.get("passes_continuation_gate"))
        else None
    )
    return analysis


def analyze_phase29_screen(report: Mapping[str, object]) -> dict[str, object]:
    """Gate augmentation arms and estimate the paired 25%-compute effect."""
    analysis = analyze_phase25_screen(report)
    raw_arms = report.get("arms")
    if not isinstance(raw_arms, Mapping):
        raise ValueError("Phase-2.9 report is missing arms")

    def by_seed(name: str) -> dict[int, Mapping[str, object]]:
        runs = raw_arms.get(name)
        if not isinstance(runs, Sequence):
            raise ValueError(f"Phase-2.9 report is missing {name}")
        return {int(run["seed"]): run for run in runs if isinstance(run, Mapping)}

    fixed = by_seed("augmentation_fixed_updates")
    matched = by_seed("augmentation_matched_exposure")
    seeds = sorted(set(fixed).intersection(matched))
    if not seeds:
        raise ValueError("Phase-2.9 augmentation cells have no paired seeds")

    def metric(run: Mapping[str, object], name: str) -> float:
        if name == "worst_robust_accuracy":
            return float(run[name])
        groups = run["pressure_groups"]
        return sum(float(groups[track]) for track in _ROBUST_PRESSURE_TRACKS) / len(_ROBUST_PRESSURE_TRACKS)

    analysis["compute_effect"] = {
        name: {
            **_distribution_summary([metric(matched[seed], name) - metric(fixed[seed], name) for seed in seeds]),
            "paired_seeds": seeds,
        }
        for name in ("worst_robust_accuracy", "macro_robust_accuracy")
    }
    passing = []
    for arm_name in ("augmentation_fixed_updates", "augmentation_matched_exposure"):
        arm = analysis["arms"][arm_name]
        if arm["passes_continuation_gate"]:
            passing.append((float(arm["worst_group_delta"]["mean"]), arm_name))
    analysis["selected_arm"] = max(passing)[1] if passing else None
    return analysis


def phase210_wide_model_config(
    model: Mapping[str, object], *, hidden_size: int, intermediate_size: int
) -> dict[str, object]:
    """Copy a model configuration with only the registered capacity fields changed."""
    if hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError("Phase-2.10 capacity fields must be positive")
    result = dict(model)
    result["hidden_size"] = hidden_size
    result["intermediate_size"] = intermediate_size
    return result


def analyze_phase210_screen(report: Mapping[str, object]) -> dict[str, object]:
    """Estimate width effects and gate wide augmentation against its wide baseline."""
    raw_arms = report.get("arms")
    if not isinstance(raw_arms, Mapping):
        raise ValueError("Phase-2.10 report is missing arms")
    required = ("narrow_baseline", "narrow_augmentation", "wide_baseline", "wide_augmentation")

    def by_seed(name: str) -> dict[int, Mapping[str, object]]:
        runs = raw_arms.get(name)
        if not isinstance(runs, Sequence):
            raise ValueError(f"Phase-2.10 report is missing {name}")
        return {int(run["seed"]): run for run in runs if isinstance(run, Mapping)}

    cells = {name: by_seed(name) for name in required}
    seeds = sorted(set.intersection(*(set(cell) for cell in cells.values())))
    if not seeds:
        raise ValueError("Phase-2.10 has no fully paired seeds")

    def metric(run: Mapping[str, object], name: str) -> float:
        if name == "worst_robust_accuracy":
            return float(run[name])
        groups = run["pressure_groups"]
        return sum(float(groups[track]) for track in _ROBUST_PRESSURE_TRACKS) / len(_ROBUST_PRESSURE_TRACKS)

    effects: dict[str, object] = {}
    for metric_name in ("worst_robust_accuracy", "macro_robust_accuracy"):
        values = {
            name: {seed: metric(cells[name][seed], metric_name) for seed in seeds} for name in required
        }
        deltas = {
            "capacity_without_augmentation": [
                values["wide_baseline"][seed] - values["narrow_baseline"][seed] for seed in seeds
            ],
            "capacity_with_augmentation": [
                values["wide_augmentation"][seed] - values["narrow_augmentation"][seed] for seed in seeds
            ],
            "interaction": [
                values["wide_augmentation"][seed]
                - values["wide_baseline"][seed]
                - values["narrow_augmentation"][seed]
                + values["narrow_baseline"][seed]
                for seed in seeds
            ],
        }
        effects[metric_name] = {
            name: {**_distribution_summary(delta), "paired_seeds": seeds} for name, delta in deltas.items()
        }
    wide_gate = analyze_phase25_screen({
        "arms": {
            "baseline": list(raw_arms["wide_baseline"]),
            "wide_augmentation": list(raw_arms["wide_augmentation"]),
        }
    })
    selected = (
        "wide_augmentation"
        if wide_gate["arms"]["wide_augmentation"]["passes_continuation_gate"]
        else None
    )
    return {
        "capacity_effects": effects,
        "wide_gate": wide_gate,
        "selected_arm": selected,
    }


def _phase26_confirmation_schedule(
    *, dataset_size: int,
    batch_size: int,
    exposure: float,
) -> dict[str, int]:
    """Maintain matched effective exposure when confirmation changes breadth."""
    max_steps = steps_for_exposure(dataset_size, batch_size, exposure)
    return {
        "batch_size": batch_size,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps_for_training(max_steps),
    }


def _load_phase26_config(config_path: str) -> tuple[dict[str, object], dict[str, object]]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "phase26_contrast_coverage" not in config:
        raise ValueError("Configuration is missing phase26_contrast_coverage")
    experiment = config["phase26_contrast_coverage"]
    if not isinstance(experiment, dict):
        raise ValueError("phase26_contrast_coverage must be a mapping")
    return config, experiment


def _phase26_arms(
    config: Mapping[str, object],
    experiment: Mapping[str, object],
) -> tuple[ConditionSplits, dict[str, dict[str, object]]]:
    splits = _phase25_splits(config, experiment, int(experiment["screen_variants"]))
    arms = build_phase26_contrast_arms(
        splits,
        replacement_fraction=float(experiment["replacement_fraction"]),
        minimal_seed=int(experiment["minimal_seed"]),
        lexical_seed=int(experiment["lexical_seed"]),
    )
    return splits, arms


def _phase26_fingerprints(
    config: Mapping[str, object],
    schedule: Mapping[str, object],
    arms: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    controls = {
        "training": dict(schedule),
        "model": config["model"],
        "tokenizer": config["tokenizer"],
    }
    result: dict[str, str] = {}
    for arm_name, arm in arms.items():
        splits = arm["splits"]
        if not isinstance(splits, ConditionSplits):
            raise ValueError(f"Arm {arm_name} has invalid splits")
        result[arm_name] = compute_phase25_training_fingerprint(
            splits.train,
            controls,
            target_mode=str(arm["target_mode"]),
        )
    return result


def _phase26_source_run_dir(
    experiment: Mapping[str, object],
    arm_name: str,
    run: Mapping[str, object],
) -> Path:
    if arm_name == "baseline":
        reused_from = run.get("reused_from")
        if not reused_from:
            raise ValueError(f"Baseline seed {run.get('seed')} is missing reused_from")
        return Path(str(reused_from))
    return Path(str(experiment["source_run_dir"])) / arm_name / f"seed_{int(run['seed'])}"


def prepare_phase26(config_path: str) -> dict[str, object]:
    """Register Phase 2.6 and validate both reused arms without training."""
    config, experiment = _load_phase26_config(config_path)
    splits, arms = _phase26_arms(config, experiment)
    schedule = _phase25_schedule(config, experiment, len(splits.train))
    fingerprints = _phase26_fingerprints(config, schedule, arms)
    pressure_groups = build_fixed_pressure_test(tuple(tuple(pair) for pair in experiment["pressure_pairs"]))

    manifests: dict[str, object] = {}
    registration_arms: dict[str, object] = {}
    for arm_name, arm in arms.items():
        arm_splits = arm["splits"]
        if not isinstance(arm_splits, ConditionSplits):
            raise ValueError(f"Arm {arm_name} has invalid splits")
        target_mode = str(arm["target_mode"])
        effective_training = condition_splits_with_target_mode(arm_splits, target_mode).train
        manifests[arm_name] = build_phase25_split_manifest(effective_training, pressure_groups)
        registration_arms[arm_name] = {
            "requires_training": bool(arm["requires_training"]),
            "target_mode": target_mode,
            "training_examples": len(effective_training),
            "training_fingerprint": fingerprints[arm_name],
        }
    if any(manifest["validation"] != "PASS" for manifest in manifests.values()):
        raise ValueError("A Phase-2.6 train/development split manifest failed")

    source_results_path = Path(str(experiment["source_results"]))
    with source_results_path.open("r", encoding="utf-8") as handle:
        source_report = json.load(handle)
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    reused = validate_phase26_reused_metrics(
        source_report,
        {arm: fingerprints[arm] for arm in ("baseline", "minimal_contrast")},
        seeds=seeds,
        max_steps=int(schedule["max_steps"]),
    )
    for arm_name, runs in reused.items():
        arm_splits = arms[arm_name]["splits"]
        if not isinstance(arm_splits, ConditionSplits):
            raise ValueError(f"Arm {arm_name} has invalid splits")
        expected_training = condition_splits_with_target_mode(arm_splits, "operation").train
        expected_corpus = "".join(f"{example.utterance}\n{example.target}\n" for example in expected_training)
        expected_hash = hashlib.sha256(expected_corpus.encode("utf-8")).hexdigest()
        for run in runs:
            run_dir = _phase26_source_run_dir(experiment, arm_name, run)
            checkpoint_path = run_dir / "parser_final.pt"
            corpus_path = run_dir / "tokenizer_corpus.txt"
            if not checkpoint_path.exists() or not corpus_path.exists():
                raise FileNotFoundError(f"Incomplete Phase-2.6 reused run: {run_dir}")
            normalized_corpus = corpus_path.read_text(encoding="utf-8")
            actual_hash = hashlib.sha256(normalized_corpus.encode("utf-8")).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"Phase-2.6 {arm_name} corpus mismatch for seed {run['seed']}")

    registration = {
        "experiment": str(config.get("name", "phase26_contrast_coverage")),
        "config": config_path,
        "status": "registered_not_run",
        "screen_variants": int(experiment["screen_variants"]),
        "screen_exposure": float(experiment["screen_exposure"]),
        "seeds": list(seeds),
        "schedule": schedule,
        "new_training_runs": sum(bool(arm["requires_training"]) for arm in arms.values()) * len(seeds),
        "sealed_suite_status": "not_created_not_opened",
        "arms": registration_arms,
        "source_reuse_validation": {
            "validation": "PASS",
            "arms": ["baseline", "minimal_contrast"],
            "validated_seeds": list(seeds),
            "source_results": str(source_results_path),
        },
    }
    output_dir = Path(str(experiment["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "registration.json").open("w", encoding="utf-8") as handle:
        json.dump(registration, handle, indent=2)
    with (output_dir / "split_manifests.json").open("w", encoding="utf-8") as handle:
        json.dump({"experiment": registration["experiment"], "manifests": manifests}, handle, indent=2)
    return {"registration": registration, "manifests": manifests}


def run_phase26_screen(config_path: str) -> dict[str, object]:
    """Run only Phase 2.6's lexical and minimal-plus-lexical arms."""
    preparation = prepare_phase26(config_path)
    config, experiment = _load_phase26_config(config_path)
    splits, arms = _phase26_arms(config, experiment)
    schedule = _phase25_schedule(config, experiment, len(splits.train))
    fingerprints = _phase26_fingerprints(config, schedule, arms)
    pressure_groups = build_fixed_pressure_test(tuple(tuple(pair) for pair in experiment["pressure_pairs"]))
    with Path(str(experiment["source_results"])).open("r", encoding="utf-8") as handle:
        source_report = json.load(handle)
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    results = validate_phase26_reused_metrics(
        source_report,
        {arm: fingerprints[arm] for arm in ("baseline", "minimal_contrast")},
        seeds=seeds,
        max_steps=int(schedule["max_steps"]),
    )
    condition_config = copy.deepcopy(config)
    condition_config["training"]["max_steps"] = schedule["max_steps"]
    condition_config["training"]["warmup_steps"] = schedule["warmup_steps"]
    output_dir = Path(str(experiment["output_dir"]))

    for arm_name, arm in arms.items():
        if not bool(arm["requires_training"]):
            continue
        arm_splits = arm["splits"]
        if not isinstance(arm_splits, ConditionSplits):
            raise ValueError(f"Arm {arm_name} has invalid splits")
        fingerprint = fingerprints[arm_name]
        results[arm_name] = []
        for seed in seeds:
            run_dir = output_dir / "screen" / arm_name / f"seed_{seed}"
            metrics_path = run_dir / "metrics.json"
            checkpoint_path = run_dir / "parser_final.pt"
            if metrics_path.exists() or checkpoint_path.exists():
                if not metrics_path.exists() or not checkpoint_path.exists():
                    raise RuntimeError(f"Partial Phase-2.6 run requires manual inspection: {run_dir}")
                with metrics_path.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                if existing.get("phase26_training_fingerprint") != fingerprint:
                    raise RuntimeError(f"Refusing to reuse mismatched Phase-2.6 run: {run_dir}")
                results[arm_name].append(existing)
                continue
            results[arm_name].append(
                run_condition(
                    arm_splits,
                    run_dir,
                    condition_config,
                    seed,
                    pressure_groups=pressure_groups,
                    run_metadata={
                        "phase26_arm": arm_name,
                        "phase26_training_fingerprint": fingerprint,
                        "variants_per_operation": int(experiment["screen_variants"]),
                        "exposure_target": float(experiment["screen_exposure"]),
                    },
                    target_mode="operation",
                )
            )

    report: dict[str, object] = {
        "experiment": str(config.get("name", "phase26_contrast_coverage")),
        "stage": "phase26_minimal_by_lexical_screen",
        "config": config_path,
        "seeds": list(seeds),
        "schedule": schedule,
        "sealed_suite_status": "not_created_not_opened",
        "arms": results,
    }
    analysis = analyze_phase26_screen(report)
    with (output_dir / "screen_results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "screen_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    return {"report": report, "analysis": analysis, "registration": preparation["registration"]}


def _load_phase26_confirmation_config(config_path: str) -> tuple[dict[str, object], dict[str, object]]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "phase26_confirmation" not in config:
        raise ValueError("Configuration is missing phase26_confirmation")
    experiment = config["phase26_confirmation"]
    if not isinstance(experiment, dict):
        raise ValueError("phase26_confirmation must be a mapping")
    return config, experiment


def _phase26_confirmation_conditions(
    config: Mapping[str, object], experiment: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    conditions: dict[str, dict[str, object]] = {}
    for variants in (int(value) for value in experiment["variants_per_operation"]):
        splits = build_condition_splits(
            variants,
            tuple(tuple(pair) for pair in experiment["train_pairs"]),
            tuple(tuple(pair) for pair in experiment["eval_pairs"]),
            seed=int(config["seed"]),
        )
        arms = build_phase26_confirmation_arms(
            splits,
            replacement_fraction=float(experiment["replacement_fraction"]),
            minimal_seed=int(experiment["minimal_seed"]),
            lexical_seed=int(experiment["lexical_seed"]),
        )
        schedule = _phase26_confirmation_schedule(
            dataset_size=len(splits.train),
            batch_size=int(config["training"]["batch_size"]),
            exposure=float(experiment["exposure"]),
        )
        conditions[str(variants)] = {"splits": splits, "arms": arms, "schedule": schedule}
    return conditions


def prepare_phase26_confirmation(config_path: str) -> dict[str, object]:
    """Register and validate the fresh-seed sealed confirmation without training."""
    config, experiment = _load_phase26_confirmation_config(config_path)
    with Path(str(experiment["source_screen_analysis"])).open("r", encoding="utf-8") as handle:
        screen_analysis = json.load(handle)
    if screen_analysis.get("selected_arm") != experiment["selected_arm"]:
        raise ValueError("Confirmation selected arm does not match the completed screen")
    if not screen_analysis.get("arms", {}).get(experiment["selected_arm"], {}).get("passes_continuation_gate"):
        raise ValueError("Confirmation may only follow a screen arm that passed its gate")

    development = build_fixed_pressure_test(tuple(tuple(pair) for pair in experiment["development_pressure_pairs"]))
    sealed = build_phase26_sealed_pressure_test(tuple(tuple(pair) for pair in experiment["sealed_pressure_pairs"]))
    conditions = _phase26_confirmation_conditions(config, experiment)
    manifests: dict[str, object] = {}
    fingerprints: dict[str, object] = {}
    for variants, condition in conditions.items():
        arms = condition["arms"]
        schedule = condition["schedule"]
        if not isinstance(arms, Mapping) or not isinstance(schedule, Mapping):
            raise ValueError("Invalid Phase-2.6 confirmation condition")
        manifests[variants] = {}
        fingerprints[variants] = {}
        for arm_name, arm in arms.items():
            splits = arm["splits"]
            if not isinstance(splits, ConditionSplits):
                raise ValueError(f"Invalid confirmation {arm_name} splits")
            effective_training = condition_splits_with_target_mode(splits, "operation").train
            manifests[variants][arm_name] = build_phase25_split_manifest(
                effective_training, development, sealed_groups=sealed
            )
            fingerprints[variants][arm_name] = compute_phase25_training_fingerprint(
                splits.train, {"training": dict(schedule), "model": config["model"], "tokenizer": config["tokenizer"]}, target_mode="operation"
            )
    if any(manifest["validation"] != "PASS" for cells in manifests.values() for manifest in cells.values()):
        raise ValueError("A Phase-2.6 confirmation isolation manifest failed")
    registration = {
        "experiment": str(config.get("name", "phase26_contrast_coverage_confirmation")),
        "config": config_path,
        "status": "registered_not_run",
        "selected_arm": str(experiment["selected_arm"]),
        "variants_per_operation": [int(value) for value in experiment["variants_per_operation"]],
        "seeds": [int(seed) for seed in experiment["seeds"]],
        "new_training_runs": 2 * len(experiment["variants_per_operation"]) * len(experiment["seeds"]),
        "sealed_suite_status": "created_but_not_evaluated",
        "schedules": {variants: condition["schedule"] for variants, condition in conditions.items()},
        "training_fingerprints": fingerprints,
    }
    output_dir = Path(str(experiment["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "registration.json").open("w", encoding="utf-8") as handle:
        json.dump(registration, handle, indent=2)
    with (output_dir / "split_manifests.json").open("w", encoding="utf-8") as handle:
        json.dump({"experiment": registration["experiment"], "manifests": manifests}, handle, indent=2)
    return {"registration": registration, "conditions": conditions, "sealed": sealed}


def run_phase26_confirmation(config_path: str) -> dict[str, object]:
    """Run the registered K=4/K=8 fresh-seed confirmation and open its sealed suite."""
    preparation = prepare_phase26_confirmation(config_path)
    config, experiment = _load_phase26_confirmation_config(config_path)
    condition_config = copy.deepcopy(config)
    sealed = preparation["sealed"]
    if not isinstance(sealed, Mapping):
        raise ValueError("Invalid sealed pressure suite")
    results: dict[str, object] = {}
    output_dir = Path(str(experiment["output_dir"]))
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    for variants, condition in preparation["conditions"].items():
        arms = condition["arms"]
        schedule = condition["schedule"]
        if not isinstance(arms, Mapping) or not isinstance(schedule, Mapping):
            raise ValueError("Invalid confirmation condition")
        condition_config["training"]["max_steps"] = schedule["max_steps"]
        condition_config["training"]["warmup_steps"] = schedule["warmup_steps"]
        results[variants] = {}
        for arm_name, arm in arms.items():
            splits = arm["splits"]
            if not isinstance(splits, ConditionSplits):
                raise ValueError(f"Invalid confirmation {arm_name} splits")
            fingerprint = preparation["registration"]["training_fingerprints"][variants][arm_name]
            arm_results: list[dict[str, object]] = []
            for seed in seeds:
                run_dir = output_dir / f"variants_{variants}" / arm_name / f"seed_{seed}"
                metrics_path = run_dir / "metrics.json"
                checkpoint_path = run_dir / "parser_final.pt"
                if metrics_path.exists() or checkpoint_path.exists():
                    if not metrics_path.exists() or not checkpoint_path.exists():
                        raise RuntimeError(f"Partial confirmation run requires manual inspection: {run_dir}")
                    with metrics_path.open("r", encoding="utf-8") as handle:
                        existing = json.load(handle)
                    if existing.get("phase26_confirmation_training_fingerprint") != fingerprint:
                        raise RuntimeError(f"Refusing mismatched confirmation reuse: {run_dir}")
                    arm_results.append(existing)
                    continue
                arm_results.append(run_condition(
                    splits, run_dir, condition_config, seed, pressure_groups=sealed,
                    run_metadata={
                        "phase26_confirmation_arm": arm_name,
                        "phase26_confirmation_training_fingerprint": fingerprint,
                        "variants_per_operation": int(variants),
                        "exposure_target": float(experiment["exposure"]),
                    }, target_mode="operation",
                ))
            results[variants][arm_name] = arm_results
    report = {
        "experiment": str(config.get("name", "phase26_contrast_coverage_confirmation")),
        "stage": "phase26_sealed_confirmation",
        "config": config_path,
        "selected_arm": str(experiment["selected_arm"]),
        "seeds": list(seeds),
        "sealed_suite_status": "evaluated_after_registration",
        "runs": results,
    }
    analysis = analyze_phase26_confirmation(report)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    return {"report": report, "analysis": analysis, "registration": preparation["registration"]}


def _load_phase28_config(config_path: str) -> tuple[dict[str, object], dict[str, object]]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "phase28_discourse_coverage" not in config:
        raise ValueError("Configuration is missing phase28_discourse_coverage")
    experiment = config["phase28_discourse_coverage"]
    if not isinstance(experiment, dict):
        raise ValueError("phase28_discourse_coverage must be a mapping")
    return config, experiment


def _phase28_condition(
    config: Mapping[str, object], experiment: Mapping[str, object]
) -> tuple[ConditionSplits, dict[str, dict[str, object]], dict[str, int]]:
    splits = build_condition_splits(
        int(experiment["variants_per_operation"]),
        tuple(tuple(pair) for pair in experiment["train_pairs"]),
        tuple(tuple(pair) for pair in experiment["eval_pairs"]),
        seed=int(config["seed"]),
    )
    arms = build_phase28_discourse_arms(
        splits,
        replacement_fraction=float(experiment["replacement_fraction"]),
        discourse_seed=int(experiment["discourse_seed"]),
    )
    schedule = _phase26_confirmation_schedule(
        dataset_size=len(splits.train),
        batch_size=int(config["training"]["batch_size"]),
        exposure=float(experiment["exposure"]),
    )
    return splits, arms, schedule


def prepare_phase28(config_path: str) -> dict[str, object]:
    """Register Phase 2.8 and validate its development isolation without training."""
    config, experiment = _load_phase28_config(config_path)
    _, arms, schedule = _phase28_condition(config, experiment)
    development = build_fixed_pressure_test(
        tuple(tuple(pair) for pair in experiment["development_pressure_pairs"])
    )
    manifests: dict[str, object] = {}
    fingerprints: dict[str, str] = {}
    for arm_name, arm in arms.items():
        splits = arm["splits"]
        if not isinstance(splits, ConditionSplits):
            raise ValueError(f"Invalid Phase-2.8 {arm_name} splits")
        manifests[arm_name] = build_phase25_split_manifest(splits.train, development)
        fingerprints[arm_name] = compute_phase25_training_fingerprint(
            splits.train,
            {"training": schedule, "model": config["model"], "tokenizer": config["tokenizer"]},
            target_mode="operation",
        )
    if any(manifest["validation"] != "PASS" for manifest in manifests.values()):
        raise ValueError("A Phase-2.8 train/development split manifest failed")
    seeds = [int(seed) for seed in experiment["seeds"]]
    registration = {
        "experiment": str(config.get("name", "phase28_counterfactual_discourse")),
        "config": config_path,
        "status": "registered_not_run",
        "variants_per_operation": int(experiment["variants_per_operation"]),
        "exposure": float(experiment["exposure"]),
        "seeds": seeds,
        "schedule": schedule,
        "new_training_runs": len(arms) * len(seeds),
        "sealed_suite_status": "not_created_not_opened",
        "training_fingerprints": fingerprints,
    }
    output_dir = Path(str(experiment["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "registration.json").open("w", encoding="utf-8") as handle:
        json.dump(registration, handle, indent=2)
    with (output_dir / "split_manifests.json").open("w", encoding="utf-8") as handle:
        json.dump({"experiment": registration["experiment"], "manifests": manifests}, handle, indent=2)
    return {"registration": registration, "arms": arms, "development": development}


def run_phase28_screen(config_path: str) -> dict[str, object]:
    """Run the registered fresh-baseline counterfactual-discourse screen."""
    preparation = prepare_phase28(config_path)
    config, experiment = _load_phase28_config(config_path)
    _, arms, schedule = _phase28_condition(config, experiment)
    development = preparation["development"]
    if not isinstance(development, Mapping):
        raise ValueError("Invalid Phase-2.8 development suite")
    run_config = copy.deepcopy(config)
    run_config["training"]["max_steps"] = schedule["max_steps"]
    run_config["training"]["warmup_steps"] = schedule["warmup_steps"]
    output_dir = Path(str(experiment["output_dir"]))
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    results: dict[str, list[dict[str, object]]] = {}
    for arm_name, arm in arms.items():
        splits = arm["splits"]
        if not isinstance(splits, ConditionSplits):
            raise ValueError(f"Invalid Phase-2.8 {arm_name} splits")
        fingerprint = preparation["registration"]["training_fingerprints"][arm_name]
        results[arm_name] = []
        for seed in seeds:
            run_dir = output_dir / "screen" / arm_name / f"seed_{seed}"
            metrics_path = run_dir / "metrics.json"
            checkpoint_path = run_dir / "parser_final.pt"
            if metrics_path.exists() or checkpoint_path.exists():
                if not metrics_path.exists() or not checkpoint_path.exists():
                    raise RuntimeError(f"Partial Phase-2.8 run requires manual inspection: {run_dir}")
                with metrics_path.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                if existing.get("phase28_training_fingerprint") != fingerprint:
                    raise RuntimeError(f"Refusing mismatched Phase-2.8 reuse: {run_dir}")
                results[arm_name].append(existing)
                continue
            results[arm_name].append(run_condition(
                splits,
                run_dir,
                run_config,
                seed,
                pressure_groups=development,
                run_metadata={
                    "phase28_arm": arm_name,
                    "phase28_training_fingerprint": fingerprint,
                    "variants_per_operation": int(experiment["variants_per_operation"]),
                    "exposure_target": float(experiment["exposure"]),
                },
                target_mode="operation",
            ))
    report = {
        "experiment": str(config.get("name", "phase28_counterfactual_discourse")),
        "stage": "phase28_counterfactual_discourse_screen",
        "config": config_path,
        "seeds": list(seeds),
        "schedule": schedule,
        "sealed_suite_status": "not_created_not_opened",
        "arms": results,
    }
    analysis = analyze_phase28_screen(report)
    with (output_dir / "screen_results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "screen_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    return {"report": report, "analysis": analysis, "registration": preparation["registration"]}


def _load_phase29_config(config_path: str) -> tuple[dict[str, object], dict[str, object]]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "phase29_discourse_augmentation" not in config:
        raise ValueError("Configuration is missing phase29_discourse_augmentation")
    experiment = config["phase29_discourse_augmentation"]
    if not isinstance(experiment, dict):
        raise ValueError("phase29_discourse_augmentation must be a mapping")
    return config, experiment


def _phase29_condition(
    config: Mapping[str, object], experiment: Mapping[str, object]
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, int]]]:
    splits = build_condition_splits(
        int(experiment["variants_per_operation"]),
        tuple(tuple(pair) for pair in experiment["train_pairs"]),
        tuple(tuple(pair) for pair in experiment["eval_pairs"]),
        seed=int(config["seed"]),
    )
    arms = build_phase29_augmentation_arms(
        splits,
        augmentation_fraction=float(experiment["augmentation_fraction"]),
        discourse_seed=int(experiment["discourse_seed"]),
    )
    batch_size = int(config["training"]["batch_size"])
    exposure = float(experiment["exposure"])
    base_schedule = _phase26_confirmation_schedule(
        dataset_size=len(splits.train), batch_size=batch_size, exposure=exposure
    )
    augmented_splits = arms["augmentation_fixed_updates"]["splits"]
    if not isinstance(augmented_splits, ConditionSplits):
        raise ValueError("Invalid Phase-2.9 augmented splits")
    matched_schedule = _phase26_confirmation_schedule(
        dataset_size=len(augmented_splits.train), batch_size=batch_size, exposure=exposure
    )
    schedules = {
        "baseline": base_schedule,
        "counterfactual_replacement": base_schedule,
        "augmentation_fixed_updates": base_schedule,
        "augmentation_matched_exposure": matched_schedule,
    }
    return arms, schedules


def _phase29_fingerprints(
    config: Mapping[str, object],
    arms: Mapping[str, Mapping[str, object]],
    schedules: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for arm_name, arm in arms.items():
        splits = arm["splits"]
        if not isinstance(splits, ConditionSplits):
            raise ValueError(f"Invalid Phase-2.9 {arm_name} splits")
        fingerprints[arm_name] = compute_phase25_training_fingerprint(
            splits.train,
            {"training": dict(schedules[arm_name]), "model": config["model"], "tokenizer": config["tokenizer"]},
            target_mode="operation",
        )
    return fingerprints


def _validate_phase29_source(
    source: Mapping[str, object],
    fingerprints: Mapping[str, str],
    seeds: Sequence[int],
    max_steps: int,
) -> dict[str, list[dict[str, object]]]:
    raw_arms = source.get("arms")
    if not isinstance(raw_arms, Mapping):
        raise ValueError("Phase-2.9 source report is missing arms")
    source_names = {"baseline": "baseline", "counterfactual_replacement": "counterfactual_discourse"}
    reused: dict[str, list[dict[str, object]]] = {}
    for target_name, source_name in source_names.items():
        runs = raw_arms.get(source_name)
        if not isinstance(runs, Sequence):
            raise ValueError(f"Phase-2.9 source report is missing {source_name}")
        by_seed = {int(run["seed"]): dict(run) for run in runs if isinstance(run, Mapping)}
        reused[target_name] = []
        for seed_value in seeds:
            seed = int(seed_value)
            run = by_seed.get(seed)
            if run is None:
                raise ValueError(f"Missing Phase-2.9 source seed {seed}")
            if int(run["steps"]) != max_steps:
                raise ValueError(f"Phase-2.9 source schedule mismatch for {source_name} seed {seed}")
            if run.get("phase28_training_fingerprint") != fingerprints[target_name]:
                raise ValueError(f"Phase-2.9 source fingerprint mismatch for {source_name} seed {seed}")
            reused[target_name].append(run)
    return reused


def prepare_phase29(config_path: str) -> dict[str, object]:
    """Register Phase 2.9 and validate both Phase-2.8 source cells."""
    config, experiment = _load_phase29_config(config_path)
    arms, schedules = _phase29_condition(config, experiment)
    fingerprints = _phase29_fingerprints(config, arms, schedules)
    development = build_fixed_pressure_test(
        tuple(tuple(pair) for pair in experiment["development_pressure_pairs"])
    )
    manifests: dict[str, object] = {}
    for arm_name, arm in arms.items():
        splits = arm["splits"]
        if not isinstance(splits, ConditionSplits):
            raise ValueError(f"Invalid Phase-2.9 {arm_name} splits")
        manifests[arm_name] = build_phase25_split_manifest(splits.train, development)
    if any(manifest["validation"] != "PASS" for manifest in manifests.values()):
        raise ValueError("A Phase-2.9 train/development split manifest failed")
    with Path(str(experiment["source_results"])).open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    reused = _validate_phase29_source(
        source, fingerprints, seeds, int(schedules["baseline"]["max_steps"])
    )
    source_names = {"baseline": "baseline", "counterfactual_replacement": "counterfactual_discourse"}
    source_root = Path(str(experiment["source_run_dir"]))
    for arm_name, runs in reused.items():
        splits = arms[arm_name]["splits"]
        if not isinstance(splits, ConditionSplits):
            raise ValueError(f"Invalid Phase-2.9 reused {arm_name} splits")
        expected = "".join(f"{example.utterance}\n{example.target}\n" for example in splits.train)
        expected_hash = hashlib.sha256(expected.encode("utf-8")).hexdigest()
        for run in runs:
            run_dir = source_root / source_names[arm_name] / f"seed_{int(run['seed'])}"
            if not (run_dir / "parser_final.pt").exists() or not (run_dir / "tokenizer_corpus.txt").exists():
                raise FileNotFoundError(f"Incomplete Phase-2.9 source run: {run_dir}")
            normalized_corpus = (run_dir / "tokenizer_corpus.txt").read_text(encoding="utf-8")
            actual_hash = hashlib.sha256(normalized_corpus.encode("utf-8")).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"Phase-2.9 source corpus mismatch: {run_dir}")
    registration = {
        "experiment": str(config.get("name", "phase29_discourse_augmentation")),
        "config": config_path,
        "status": "registered_not_run",
        "seeds": list(seeds),
        "schedules": schedules,
        "new_training_runs": 2 * len(seeds),
        "sealed_suite_status": "not_created_not_opened",
        "training_fingerprints": fingerprints,
        "source_reuse_validation": {"validation": "PASS", "arms": list(reused), "seeds": list(seeds)},
    }
    output_dir = Path(str(experiment["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "registration.json").open("w", encoding="utf-8") as handle:
        json.dump(registration, handle, indent=2)
    with (output_dir / "split_manifests.json").open("w", encoding="utf-8") as handle:
        json.dump({"experiment": registration["experiment"], "manifests": manifests}, handle, indent=2)
    return {"registration": registration, "arms": arms, "development": development, "reused": reused}


def run_phase29_screen(config_path: str) -> dict[str, object]:
    """Train only Phase 2.9's two augmentation cells and apply its gate."""
    preparation = prepare_phase29(config_path)
    config, experiment = _load_phase29_config(config_path)
    arms, schedules = _phase29_condition(config, experiment)
    fingerprints = preparation["registration"]["training_fingerprints"]
    development = preparation["development"]
    results = copy.deepcopy(preparation["reused"])
    output_dir = Path(str(experiment["output_dir"]))
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    for arm_name in ("augmentation_fixed_updates", "augmentation_matched_exposure"):
        splits = arms[arm_name]["splits"]
        if not isinstance(splits, ConditionSplits):
            raise ValueError(f"Invalid Phase-2.9 {arm_name} splits")
        run_config = copy.deepcopy(config)
        run_config["training"]["max_steps"] = schedules[arm_name]["max_steps"]
        run_config["training"]["warmup_steps"] = schedules[arm_name]["warmup_steps"]
        fingerprint = fingerprints[arm_name]
        results[arm_name] = []
        for seed in seeds:
            run_dir = output_dir / "screen" / arm_name / f"seed_{seed}"
            metrics_path = run_dir / "metrics.json"
            checkpoint_path = run_dir / "parser_final.pt"
            if metrics_path.exists() or checkpoint_path.exists():
                if not metrics_path.exists() or not checkpoint_path.exists():
                    raise RuntimeError(f"Partial Phase-2.9 run requires manual inspection: {run_dir}")
                with metrics_path.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                if existing.get("phase29_training_fingerprint") != fingerprint:
                    raise RuntimeError(f"Refusing mismatched Phase-2.9 reuse: {run_dir}")
                results[arm_name].append(existing)
                continue
            results[arm_name].append(run_condition(
                splits, run_dir, run_config, seed, pressure_groups=development,
                run_metadata={
                    "phase29_arm": arm_name,
                    "phase29_training_fingerprint": fingerprint,
                    "variants_per_operation": int(experiment["variants_per_operation"]),
                    "exposure_target": (
                        float(experiment["exposure"])
                        if arm_name == "augmentation_matched_exposure"
                        else round(schedules[arm_name]["max_steps"] * int(config["training"]["batch_size"]) / len(splits.train), 6)
                    ),
                }, target_mode="operation",
            ))
    report = {
        "experiment": str(config.get("name", "phase29_discourse_augmentation")),
        "stage": "phase29_replacement_vs_augmentation_screen",
        "config": config_path,
        "seeds": list(seeds),
        "schedules": schedules,
        "sealed_suite_status": "not_created_not_opened",
        "arms": results,
    }
    analysis = analyze_phase29_screen(report)
    with (output_dir / "screen_results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "screen_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    return {"report": report, "analysis": analysis, "registration": preparation["registration"]}


def _load_phase210_config(config_path: str) -> tuple[dict[str, object], dict[str, object]]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "phase210_capacity_interaction" not in config:
        raise ValueError("Configuration is missing phase210_capacity_interaction")
    experiment = config["phase210_capacity_interaction"]
    if not isinstance(experiment, dict):
        raise ValueError("phase210_capacity_interaction must be a mapping")
    return config, experiment


def _phase210_design(
    config: Mapping[str, object], experiment: Mapping[str, object]
) -> tuple[dict[str, ConditionSplits], dict[str, dict[str, object]], dict[str, int]]:
    base = build_condition_splits(
        int(experiment["variants_per_operation"]),
        tuple(tuple(pair) for pair in experiment["train_pairs"]),
        tuple(tuple(pair) for pair in experiment["eval_pairs"]),
        seed=int(config["seed"]),
    )
    augmented_train = build_counterfactual_discourse_augmentation(
        base.train,
        augmentation_fraction=float(experiment["augmentation_fraction"]),
        seed=int(experiment["discourse_seed"]),
    )
    augmented = ConditionSplits(
        train=augmented_train,
        seen_form=base.seen_form,
        same_meaning_unseen_form=base.same_meaning_unseen_form,
        unseen_operands_seen_form=base.unseen_operands_seen_form,
        minimal_contrasts=base.minimal_contrasts,
    )
    splits = {
        "narrow_baseline": base,
        "narrow_augmentation": augmented,
        "wide_baseline": base,
        "wide_augmentation": augmented,
    }
    narrow_model = dict(config["model"])
    wide_model = phase210_wide_model_config(
        narrow_model,
        hidden_size=int(experiment["wide_hidden_size"]),
        intermediate_size=int(experiment["wide_intermediate_size"]),
    )
    models_by_arm = {
        "narrow_baseline": narrow_model,
        "narrow_augmentation": narrow_model,
        "wide_baseline": wide_model,
        "wide_augmentation": wide_model,
    }
    schedule = _phase26_confirmation_schedule(
        dataset_size=len(base.train),
        batch_size=int(config["training"]["batch_size"]),
        exposure=float(experiment["exposure"]),
    )
    return splits, models_by_arm, schedule


def _phase210_fingerprints(
    config: Mapping[str, object],
    splits: Mapping[str, ConditionSplits],
    models_by_arm: Mapping[str, Mapping[str, object]],
    schedule: Mapping[str, object],
) -> dict[str, str]:
    return {
        arm_name: compute_phase25_training_fingerprint(
            arm_splits.train,
            {"training": dict(schedule), "model": models_by_arm[arm_name], "tokenizer": config["tokenizer"]},
            target_mode="operation",
        )
        for arm_name, arm_splits in splits.items()
    }


def prepare_phase210(config_path: str) -> dict[str, object]:
    """Register the capacity interaction and validate narrow source cells."""
    config, experiment = _load_phase210_config(config_path)
    splits, models_by_arm, schedule = _phase210_design(config, experiment)
    fingerprints = _phase210_fingerprints(config, splits, models_by_arm, schedule)
    development = build_fixed_pressure_test(
        tuple(tuple(pair) for pair in experiment["development_pressure_pairs"])
    )
    manifests = {
        arm_name: build_phase25_split_manifest(arm_splits.train, development)
        for arm_name, arm_splits in splits.items()
    }
    if any(manifest["validation"] != "PASS" for manifest in manifests.values()):
        raise ValueError("A Phase-2.10 train/development manifest failed")
    with Path(str(experiment["source_results"])).open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    raw_arms = source.get("arms")
    if not isinstance(raw_arms, Mapping):
        raise ValueError("Phase-2.10 source report is missing arms")
    source_specs = {
        "narrow_baseline": ("baseline", "phase28_training_fingerprint", Path(str(experiment["narrow_baseline_run_dir"]))),
        "narrow_augmentation": (
            "augmentation_fixed_updates",
            "phase29_training_fingerprint",
            Path(str(experiment["narrow_augmentation_run_dir"])),
        ),
    }
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    reused: dict[str, list[dict[str, object]]] = {}
    for target_name, (source_name, fingerprint_field, run_root) in source_specs.items():
        raw_runs = raw_arms.get(source_name)
        if not isinstance(raw_runs, Sequence):
            raise ValueError(f"Phase-2.10 source report is missing {source_name}")
        by_seed = {int(run["seed"]): dict(run) for run in raw_runs if isinstance(run, Mapping)}
        reused[target_name] = []
        expected_corpus = "".join(
            f"{example.utterance}\n{example.target}\n" for example in splits[target_name].train
        )
        expected_hash = hashlib.sha256(expected_corpus.encode("utf-8")).hexdigest()
        for seed in seeds:
            run = by_seed.get(seed)
            if run is None:
                raise ValueError(f"Missing Phase-2.10 source {source_name} seed {seed}")
            if int(run["steps"]) != int(schedule["max_steps"]):
                raise ValueError(f"Phase-2.10 source schedule mismatch for {source_name} seed {seed}")
            if run.get(fingerprint_field) != fingerprints[target_name]:
                raise ValueError(f"Phase-2.10 source fingerprint mismatch for {source_name} seed {seed}")
            run_dir = run_root / f"seed_{seed}"
            corpus_path = run_dir / "tokenizer_corpus.txt"
            if not (run_dir / "parser_final.pt").exists() or not corpus_path.exists():
                raise FileNotFoundError(f"Incomplete Phase-2.10 source run: {run_dir}")
            actual_hash = hashlib.sha256(corpus_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"Phase-2.10 source corpus mismatch: {run_dir}")
            reused[target_name].append(run)
    registration = {
        "experiment": str(config.get("name", "phase210_capacity_interaction")),
        "config": config_path,
        "status": "registered_not_run",
        "seeds": list(seeds),
        "schedule": schedule,
        "narrow_hidden_size": int(config["model"]["hidden_size"]),
        "wide_hidden_size": int(experiment["wide_hidden_size"]),
        "new_training_runs": 2 * len(seeds),
        "sealed_suite_status": "not_created_not_opened",
        "training_fingerprints": fingerprints,
        "source_reuse_validation": {"validation": "PASS", "arms": list(reused), "seeds": list(seeds)},
    }
    output_dir = Path(str(experiment["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "registration.json").open("w", encoding="utf-8") as handle:
        json.dump(registration, handle, indent=2)
    with (output_dir / "split_manifests.json").open("w", encoding="utf-8") as handle:
        json.dump({"experiment": registration["experiment"], "manifests": manifests}, handle, indent=2)
    return {
        "registration": registration,
        "splits": splits,
        "models_by_arm": models_by_arm,
        "development": development,
        "reused": reused,
    }


def run_phase210_screen(config_path: str) -> dict[str, object]:
    """Train only the two wide Phase-2.10 cells and estimate the interaction."""
    preparation = prepare_phase210(config_path)
    config, experiment = _load_phase210_config(config_path)
    _, _, schedule = _phase210_design(config, experiment)
    results = copy.deepcopy(preparation["reused"])
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    output_dir = Path(str(experiment["output_dir"]))
    for arm_name in ("wide_baseline", "wide_augmentation"):
        arm_splits = preparation["splits"][arm_name]
        run_config = copy.deepcopy(config)
        run_config["model"] = copy.deepcopy(preparation["models_by_arm"][arm_name])
        run_config["training"]["max_steps"] = int(schedule["max_steps"])
        run_config["training"]["warmup_steps"] = int(schedule["warmup_steps"])
        fingerprint = preparation["registration"]["training_fingerprints"][arm_name]
        results[arm_name] = []
        for seed in seeds:
            run_dir = output_dir / "screen" / arm_name / f"seed_{seed}"
            metrics_path = run_dir / "metrics.json"
            checkpoint_path = run_dir / "parser_final.pt"
            if metrics_path.exists() or checkpoint_path.exists():
                if not metrics_path.exists() or not checkpoint_path.exists():
                    raise RuntimeError(f"Partial Phase-2.10 run requires manual inspection: {run_dir}")
                with metrics_path.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                if existing.get("phase210_training_fingerprint") != fingerprint:
                    raise RuntimeError(f"Refusing mismatched Phase-2.10 reuse: {run_dir}")
                results[arm_name].append(existing)
                continue
            results[arm_name].append(run_condition(
                arm_splits,
                run_dir,
                run_config,
                seed,
                pressure_groups=preparation["development"],
                run_metadata={
                    "phase210_arm": arm_name,
                    "phase210_training_fingerprint": fingerprint,
                    "variants_per_operation": int(experiment["variants_per_operation"]),
                    "exposure_target": round(
                        int(schedule["max_steps"])
                        * int(config["training"]["batch_size"])
                        / len(arm_splits.train),
                        6,
                    ),
                    "hidden_size": int(experiment["wide_hidden_size"]),
                },
                target_mode="operation",
            ))
    report = {
        "experiment": str(config.get("name", "phase210_capacity_interaction")),
        "stage": "phase210_width_by_augmentation_screen",
        "config": config_path,
        "seeds": list(seeds),
        "schedule": schedule,
        "sealed_suite_status": "not_created_not_opened",
        "arms": results,
    }
    analysis = analyze_phase210_screen(report)
    with (output_dir / "screen_results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "screen_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    return {"report": report, "analysis": analysis, "registration": preparation["registration"]}


def analyze_clean_report(report: Mapping[str, object]) -> dict[str, object]:
    """Compute paired per-doubling slopes from a completed clean-ablation report."""
    raw_runs = report["runs"]
    if not isinstance(raw_runs, Mapping):
        raise ValueError("Clean report is missing run records")
    variants = sorted(raw_runs, key=int)
    metric_values: dict[str, dict[str, dict[int, float]]] = {
        "worst_robust_accuracy": {},
        "macro_robust_accuracy": {},
    }
    for variant in variants:
        condition_runs = raw_runs[variant]
        if not isinstance(condition_runs, Sequence):
            raise ValueError(f"Condition {variant} has invalid run records")
        for run in condition_runs:
            if not isinstance(run, Mapping):
                raise ValueError(f"Condition {variant} contains an invalid run")
            seed = int(run["seed"])
            groups = run["pressure_groups"]
            if not isinstance(groups, Mapping):
                raise ValueError(f"Condition {variant}, seed {seed} is missing pressure groups")
            robust_values = [float(groups[track]) for track in _ROBUST_PRESSURE_TRACKS]
            metric_values["worst_robust_accuracy"].setdefault(variant, {})[seed] = float(run["worst_robust_accuracy"])
            metric_values["macro_robust_accuracy"].setdefault(variant, {})[seed] = sum(robust_values) / len(robust_values)

    analysis: dict[str, object] = {}
    for metric, by_variant in metric_values.items():
        summaries = {
            variant: _distribution_summary(list(by_variant[variant].values())) for variant in variants
        }
        slopes: dict[str, dict[str, object]] = {}
        for left, right in zip(variants, variants[1:]):
            common_seeds = sorted(set(by_variant[left]).intersection(by_variant[right]))
            if not common_seeds:
                raise ValueError(f"No paired seeds for K={left} and K={right}")
            deltas = [by_variant[right][seed] - by_variant[left][seed] for seed in common_seeds]
            slopes[f"{left}_to_{right}"] = {
                **_distribution_summary(deltas),
                "paired_seeds": common_seeds,
                "unit": "accuracy_points_per_doubling",
            }
        analysis[metric] = {"by_variants": summaries, "adjacent_doubling_slopes": slopes}
    return analysis


def _write_clean_analysis(report: Mapping[str, object], output_path: Path) -> dict[str, object]:
    """Persist a read-only analysis companion for a completed training report."""
    analysis = analyze_clean_report(report)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    return analysis


def write_clean_analysis(results_path: str) -> dict[str, object]:
    """Analyze an existing result artifact without rerunning any training."""
    path = Path(results_path)
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    return _write_clean_analysis(report, path.with_name("analysis.json"))


def analyze_breadth_report(report: Mapping[str, object]) -> dict[str, object]:
    """Compute breadth and reinforcement slopes from the factorial report."""
    raw_runs = report["runs"]
    raw_exposures = report["exposures"]
    if not isinstance(raw_runs, Mapping) or not isinstance(raw_exposures, Mapping):
        raise ValueError("Factorial report is missing runs or exposures")
    variants = sorted(raw_runs, key=int)
    exposure_keys = sorted(raw_exposures, key=lambda key: float(raw_exposures[key]))
    cells: dict[str, dict[str, dict[str, dict[int, float]]]] = {
        "worst_robust_accuracy": {},
        "macro_robust_accuracy": {},
    }
    for metric in cells:
        for variant in variants:
            cells[metric][variant] = {}
            for exposure_key in exposure_keys:
                cell_runs = raw_runs[variant][exposure_key]
                values: dict[int, float] = {}
                for run in cell_runs:
                    seed = int(run["seed"])
                    if metric == "worst_robust_accuracy":
                        values[seed] = float(run[metric])
                    else:
                        groups = run["pressure_groups"]
                        values[seed] = sum(float(groups[track]) for track in _ROBUST_PRESSURE_TRACKS) / len(
                            _ROBUST_PRESSURE_TRACKS
                        )
                cells[metric][variant][exposure_key] = values

    def paired_slopes(left_values: Mapping[int, float], right_values: Mapping[int, float], denominator: float) -> dict:
        seeds = sorted(set(left_values).intersection(right_values))
        if not seeds:
            raise ValueError("No paired seeds in factorial report")
        deltas = [(right_values[seed] - left_values[seed]) / denominator for seed in seeds]
        return {**_distribution_summary(deltas), "paired_seeds": seeds, "unit": "accuracy_points_per_log2_resource"}

    analysis: dict[str, object] = {"cells": {}, "breadth_slopes": {}, "reinforcement_slopes": {}}
    for metric, by_variant in cells.items():
        analysis["cells"][metric] = {
            variant: {exposure: _distribution_summary(list(values.values())) for exposure, values in by_variant[variant].items()}
            for variant in variants
        }
        analysis["breadth_slopes"][metric] = {}
        for exposure_key in exposure_keys:
            analysis["breadth_slopes"][metric][exposure_key] = {}
            for left, right in zip(variants, variants[1:]):
                denominator = math.log2(int(right) / int(left))
                analysis["breadth_slopes"][metric][exposure_key][f"{left}_to_{right}"] = paired_slopes(
                    by_variant[left][exposure_key], by_variant[right][exposure_key], denominator
                )
        analysis["reinforcement_slopes"][metric] = {}
        for variant in variants:
            analysis["reinforcement_slopes"][metric][variant] = {}
            for left, right in zip(exposure_keys, exposure_keys[1:]):
                denominator = math.log2(float(raw_exposures[right]) / float(raw_exposures[left]))
                analysis["reinforcement_slopes"][metric][variant][f"{left}_to_{right}"] = paired_slopes(
                    by_variant[variant][left], by_variant[variant][right], denominator
                )
    return analysis


def analyze_scaling_reports(reports: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Merge compatible sequential scaling reports and analyze their full curve.

    Sequential stages intentionally live in separate output directories. This
    validator prevents a curve from silently combining different seeds,
    tokenizers, pressure tracks, or training splits before delegating the
    slope calculation to :func:`analyze_breadth_report`.
    """
    if not reports:
        raise ValueError("At least one scaling report is required")

    first = reports[0]
    expected_seeds = tuple(int(seed) for seed in first["seeds"])
    expected_tokenizer = first.get("tokenizer_policy")
    expected_tracks = tuple(str(track) for track in first["robust_pressure_tracks"])
    run_mappings: list[Mapping[str, object]] = []
    common_variants: set[str] | None = None
    for report in reports:
        if tuple(int(seed) for seed in report["seeds"]) != expected_seeds:
            raise ValueError("Scaling reports use different seeds")
        if report.get("tokenizer_policy") != expected_tokenizer:
            raise ValueError("Scaling reports use different tokenizer policies")
        if tuple(str(track) for track in report["robust_pressure_tracks"]) != expected_tracks:
            raise ValueError("Scaling reports use different robust pressure tracks")
        raw_runs = report["runs"]
        if not isinstance(raw_runs, Mapping):
            raise ValueError("Scaling report runs must be a mapping")
        run_mappings.append(raw_runs)
        variants = {str(variant) for variant in raw_runs}
        common_variants = variants if common_variants is None else common_variants.intersection(variants)

    if not common_variants:
        raise ValueError("Scaling reports have no common breadth variants")
    variants = sorted(common_variants, key=int)

    expected_manifests = first["split_manifests"]
    if not isinstance(expected_manifests, Mapping):
        raise ValueError("Scaling report split manifests must be a mapping")
    for report in reports[1:]:
        manifests = report["split_manifests"]
        if not isinstance(manifests, Mapping):
            raise ValueError("Scaling report split manifests must be a mapping")
        for variant in variants:
            if manifests.get(variant) != expected_manifests.get(variant):
                raise ValueError(f"Scaling split manifest drift for breadth {variant}")

    merged_runs: dict[str, dict[str, object]] = {variant: {} for variant in variants}
    merged_exposures: dict[str, float] = {}
    for report, raw_runs in zip(reports, run_mappings):
        exposures = report["exposures"]
        if not isinstance(exposures, Mapping):
            raise ValueError("Scaling report exposures must be a mapping")
        for exposure_key, exposure_value in exposures.items():
            key = str(exposure_key)
            if key in merged_exposures:
                raise ValueError(f"Duplicate exposure {key} across scaling reports")
            merged_exposures[key] = float(exposure_value)
            for variant in variants:
                by_exposure = raw_runs[variant]
                if not isinstance(by_exposure, Mapping) or key not in by_exposure:
                    raise ValueError(f"Scaling report is missing {variant}/{key}")
                merged_runs[variant][key] = by_exposure[key]

    merged_report = {"runs": merged_runs, "exposures": merged_exposures}
    analysis = analyze_breadth_report(merged_report)
    exposure_keys = sorted(merged_exposures, key=lambda key: merged_exposures[key])
    return {
        "sources": [str(report.get("experiment", "unnamed")) for report in reports],
        "variants": variants,
        "exposures": exposure_keys,
        **analysis,
    }


def _write_breadth_analysis(report: Mapping[str, object], output_path: Path) -> dict[str, object]:
    analysis = analyze_breadth_report(report)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    return analysis


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
    parser.add_argument("--analyze-existing", default=None, help="Analyze a completed clean result JSON without training")
    parser.add_argument("--write-manifests", action="store_true", help="Write clean split manifests without training")
    parser.add_argument(
        "--phase25-stage",
        choices=("prepare", "probe", "screen"),
        default=None,
        help="Explicitly select Phase 2.5 preparation, frozen probing, or model training",
    )
    parser.add_argument(
        "--phase26-stage",
        choices=("prepare", "screen"),
        default=None,
        help="Explicitly select Phase 2.6 preparation or the ten-run training screen",
    )
    parser.add_argument(
        "--phase26-confirmation-stage",
        choices=("prepare", "run"),
        default=None,
        help="Explicitly select Phase 2.6 sealed-confirmation preparation or training",
    )
    parser.add_argument(
        "--phase28-stage",
        choices=("prepare", "screen"),
        default=None,
        help="Explicitly select Phase 2.8 preparation or its ten-run training screen",
    )
    parser.add_argument(
        "--phase29-stage",
        choices=("prepare", "screen"),
        default=None,
        help="Explicitly select Phase 2.9 preparation or its ten-run augmentation screen",
    )
    parser.add_argument(
        "--phase210-stage",
        choices=("prepare", "screen"),
        default=None,
        help="Explicitly select Phase 2.10 preparation or its ten-run capacity screen",
    )
    args = parser.parse_args()
    if args.analyze_existing:
        print(json.dumps(write_clean_analysis(args.analyze_existing), indent=2))
        return
    if args.write_manifests:
        print(json.dumps(write_clean_split_manifests(args.config), indent=2))
        return
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if "phase210_capacity_interaction" in config:
        if args.phase210_stage is None:
            parser.error("Phase 2.10 configs require --phase210-stage; no experiment is started implicitly")
        if args.phase210_stage == "prepare":
            result = prepare_phase210(args.config)
            print(json.dumps(result["registration"], indent=2))
        else:
            result = run_phase210_screen(args.config)
            print(json.dumps(result["analysis"], indent=2))
    elif "phase29_discourse_augmentation" in config:
        if args.phase29_stage is None:
            parser.error("Phase 2.9 configs require --phase29-stage; no experiment is started implicitly")
        if args.phase29_stage == "prepare":
            result = prepare_phase29(args.config)
            print(json.dumps(result["registration"], indent=2))
        else:
            result = run_phase29_screen(args.config)
            print(json.dumps(result["analysis"], indent=2))
    elif "phase28_discourse_coverage" in config:
        if args.phase28_stage is None:
            parser.error("Phase 2.8 configs require --phase28-stage; no experiment is started implicitly")
        if args.phase28_stage == "prepare":
            result = prepare_phase28(args.config)
            print(json.dumps(result["registration"], indent=2))
        else:
            result = run_phase28_screen(args.config)
            print(json.dumps(result["analysis"], indent=2))
    elif "phase26_confirmation" in config:
        if args.phase26_confirmation_stage is None:
            parser.error("Phase 2.6 confirmation requires --phase26-confirmation-stage; no experiment is started implicitly")
        if args.phase26_confirmation_stage == "prepare":
            result = prepare_phase26_confirmation(args.config)
            print(json.dumps(result["registration"], indent=2))
        else:
            result = run_phase26_confirmation(args.config)
            print(json.dumps(result["analysis"], indent=2))
    elif "phase26_contrast_coverage" in config:
        if args.phase26_stage is None:
            parser.error("Phase 2.6 configs require --phase26-stage; no experiment is started implicitly")
        if args.phase26_stage == "prepare":
            result = prepare_phase26(args.config)
            print(json.dumps(result["registration"], indent=2))
        else:
            result = run_phase26_screen(args.config)
            print(json.dumps(result["analysis"], indent=2))
    elif "phase25_representation" in config:
        if args.phase25_stage is None:
            parser.error("Phase 2.5 configs require --phase25-stage; no experiment is started implicitly")
        if args.phase25_stage == "prepare":
            result = prepare_phase25(args.config)
            print(json.dumps(result["registration"], indent=2))
        elif args.phase25_stage == "probe":
            report = run_phase25_probe(args.config)
            print(json.dumps(report["summary"], indent=2))
        else:
            result = run_phase25_screen(args.config)
            print(json.dumps(result["analysis"], indent=2))
    elif "breadth_reinforcement" in config:
        report = run_breadth_reinforcement(args.config)
        print(json.dumps({"experiment": report["experiment"], "cells": len(report["conditions"]) * len(report["exposures"])}, indent=2))
    elif "form_variation_v2" in config:
        report = run_clean_ablation(args.config)
        print(json.dumps(report["summary"], indent=2))
    else:
        report = run_ablation(args.config)
        print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
