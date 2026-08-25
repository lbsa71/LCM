"""Tests for controlled semantic form-variation experiment construction."""

from eval.form_variation import (
    FormVariationGenerator,
    build_training_loader,
    build_condition_splits,
)
import torch
from torch.utils.data import TensorDataset


def test_variants_preserve_their_canonical_operation():
    generator = FormVariationGenerator(seed=7)

    variants = generator.render_variants("ADD", left=345, right=456)

    assert len(variants) >= 4
    assert {variant.target for variant in variants} == {"OP=ADD"}
    assert {variant.template_id for variant in variants} == set(range(len(variants)))


def test_form_holdout_never_leaks_a_template_into_training():
    splits = build_condition_splits(
        variants_per_operation=2,
        train_pairs=[(12, 19), (23, 41)],
        eval_pairs=[(12, 19), (23, 41)],
        seed=11,
    )

    train_templates = {example.template_id for example in splits.train}
    held_out_templates = {example.template_id for example in splits.same_meaning_unseen_form}

    assert train_templates.isdisjoint(held_out_templates)
    assert {example.target for example in splits.train} == {"OP=ADD", "OP=SUBTRACT", "OP=COMPARE"}


def test_form_and_operand_holdouts_are_independent():
    train_pairs = [(12, 19), (23, 41)]
    eval_pairs = [(53, 47)]
    splits = build_condition_splits(
        variants_per_operation=2,
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
        seed=11,
    )

    assert {(example.left, example.right) for example in splits.same_meaning_unseen_form} == set(train_pairs)
    assert {(example.left, example.right) for example in splits.unseen_operands_seen_form} == set(eval_pairs)


def test_minimal_contrasts_have_distinct_targets():
    generator = FormVariationGenerator(seed=3)

    contrasts = generator.render_minimal_contrasts(left=345, right=456)

    assert {example.target for example in contrasts} == {"OP=ADD", "OP=SUBTRACT", "OP=COMPARE"}
    assert len({example.utterance for example in contrasts}) == 3


def test_training_loader_uses_only_full_batches_for_equal_update_budgets():
    dataset = TensorDataset(torch.arange(36))

    loader = build_training_loader(dataset, batch_size=16, use_cuda=False)

    assert loader.drop_last is True
    assert len(loader) == 2
