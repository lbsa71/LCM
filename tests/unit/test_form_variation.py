"""Tests for controlled semantic form-variation experiment construction."""

from eval.form_variation import (
    FormVariationGenerator,
    build_training_loader,
    build_condition_splits,
    build_clean_split_manifest,
    build_fixed_pressure_test,
    analyze_clean_report,
    analyze_breadth_report,
    evaluate_operation_accuracy,
    evaluate_operation_accuracy_batched,
    steps_for_exposure,
    warmup_steps_for_training,
    train_fixed_byte_tokenizer,
    write_training_tokenizer_corpus,
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


def test_fixed_pressure_suite_is_identical_across_diversity_conditions_and_uses_all_operations():
    pressure_pairs = [(101, 44), (207, 86)]
    first = build_fixed_pressure_test(pressure_pairs)
    second = build_fixed_pressure_test(pressure_pairs)

    assert first == second
    assert set(first) == {
        "seen_form_unseen_operands",
        "held_out_templates",
        "lexical_shift",
        "syntax_order_reversal",
        "discourse_distractor",
        "minimal_contrast",
    }
    for examples in first.values():
        assert {example.operation for example in examples} == {"ADD", "SUBTRACT", "COMPARE"}


def test_clean_tokenizer_corpus_contains_training_examples_only(tmp_path):
    splits = build_condition_splits(
        variants_per_operation=2,
        train_pairs=[(12, 19), (23, 41)],
        eval_pairs=[(53, 47)],
        seed=11,
    )
    pressure = build_fixed_pressure_test([(101, 44)])
    corpus_path = tmp_path / "tokenizer_corpus.txt"

    write_training_tokenizer_corpus(splits.train, corpus_path)

    corpus = corpus_path.read_text(encoding="utf-8")
    assert all(example.utterance in corpus for example in splits.train)
    assert all(example.utterance not in corpus for examples in pressure.values() for example in examples)


def test_clean_split_manifest_proves_exact_text_holdout():
    splits = build_condition_splits(
        variants_per_operation=2,
        train_pairs=[(12, 19), (23, 41)],
        eval_pairs=[(53, 47)],
        seed=11,
    )

    manifest = build_clean_split_manifest(splits.train, build_fixed_pressure_test([(101, 44)]))

    assert manifest["validation"] == "PASS"
    assert manifest["exact_utterance_overlap_count"] == 0
    assert manifest["tokenizer_corpus_sha256"] == manifest["training_corpus_sha256"]


def test_clean_analysis_reports_paired_adjacent_doubling_slopes():
    def run(seed: int, score: float) -> dict:
        return {
            "seed": seed,
            "worst_robust_accuracy": score,
            "pressure_groups": {
                "seen_form_unseen_operands": 1.0,
                "held_out_templates": score,
                "lexical_shift": score,
                "syntax_order_reversal": score,
                "discourse_distractor": score,
                "minimal_contrast": score,
            },
        }

    analysis = analyze_clean_report({"runs": {"1": [run(1, 0.2), run(2, 0.4)], "2": [run(1, 0.4), run(2, 0.6)]}})

    assert analysis["macro_robust_accuracy"]["by_variants"]["1"]["mean"] == 0.3
    assert analysis["worst_robust_accuracy"]["adjacent_doubling_slopes"]["1_to_2"]["mean"] == 0.2


def test_fixed_byte_tokenizer_is_independent_of_training_corpus(tmp_path):
    first = train_fixed_byte_tokenizer(tmp_path / "first")
    second = train_fixed_byte_tokenizer(tmp_path / "second")

    assert first.get_vocab_size() == second.get_vocab_size()
    for token in ("<PAD>", "<BOS>", "<EOS>", "<USER>", "<ASSISTANT>"):
        assert first.token_to_id(token) == second.token_to_id(token)
    assert first.encode("<USER> Calculate 345 plus 456.").ids


def test_batched_candidate_scoring_matches_sequential_scoring(tmp_path):
    from training.model import SyntheticTransformer, TransformerConfig
    import torch

    tokenizer = train_fixed_byte_tokenizer(tmp_path / "tokenizer")
    generator = FormVariationGenerator(seed=4)
    examples = generator.render_variants("ADD", 345, 456)[:2]
    model = SyntheticTransformer(
        TransformerConfig(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=128,
            pad_token_id=tokenizer.token_to_id("<PAD>") or 0,
            bos_token_id=tokenizer.token_to_id("<BOS>") or 1,
            eos_token_id=tokenizer.token_to_id("<EOS>") or 2,
        )
    )
    model.eval()

    sequential = evaluate_operation_accuracy(model, tokenizer, examples, torch.device("cpu"))
    batched = evaluate_operation_accuracy_batched(model, tokenizer, examples, torch.device("cpu"), batch_size=8)

    assert batched == sequential


def test_exposure_steps_round_to_full_batches():
    assert steps_for_exposure(dataset_size=36, batch_size=16, exposure=22.222222) == 50
    assert steps_for_exposure(dataset_size=288, batch_size=16, exposure=177.777778) == 3200
    assert warmup_steps_for_training(50) == 5
    assert warmup_steps_for_training(400) == 40
    assert warmup_steps_for_training(3200) == 320


def test_breadth_analysis_normalizes_non_doubling_exposure_slopes():
    def run(seed: int, score: float) -> dict:
        return {
            "seed": seed,
            "worst_robust_accuracy": score,
            "pressure_groups": {
                "seen_form_unseen_operands": 1.0,
                "held_out_templates": score,
                "lexical_shift": score,
                "syntax_order_reversal": score,
                "discourse_distractor": score,
                "minimal_contrast": score,
            },
        }

    report = {
        "runs": {
            "1": {"R22": [run(1, 0.2)], "R89": [run(1, 0.6)], "R178": [run(1, 0.8)]},
            "2": {"R22": [run(1, 0.4)], "R89": [run(1, 0.8)], "R178": [run(1, 1.0)]},
        },
        "exposures": {"R22": 22.222222, "R89": 88.888889, "R178": 177.777778},
    }

    analysis = analyze_breadth_report(report)

    assert analysis["breadth_slopes"]["worst_robust_accuracy"]["R22"]["1_to_2"]["mean"] == 0.2
    assert analysis["reinforcement_slopes"]["worst_robust_accuracy"]["1"]["R22_to_R89"]["mean"] == 0.2
