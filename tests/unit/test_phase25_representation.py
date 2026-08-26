"""Phase 2.5 specifications for representation diagnostics and ablations.

These tests intentionally define the experiment contract before its runner or
training interventions are implemented.  They protect both the parser
semantics and the controls needed for a scientific comparison.
"""

from __future__ import annotations

import copy

import pytest
import torch

from eval.form_variation import (
    FormVariationExample,
    FormVariationGenerator,
    build_condition_splits,
    build_fixed_pressure_test,
    train_fixed_byte_tokenizer,
)
from training.model import SyntheticTransformer, TransformerConfig


def _example(
    utterance: str,
    operation: str,
    *,
    left: int = 345,
    right: int = 456,
    template_id: int = 0,
) -> FormVariationExample:
    return FormVariationExample(
        utterance=utterance,
        target=f"OP={operation}",
        operation=operation,
        template_id=template_id,
        left=left,
        right=right,
    )


def _tiny_model(tokenizer) -> SyntheticTransformer:
    return SyntheticTransformer(
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


def test_typed_frame_serializes_canonical_roles_not_surface_mention_order():
    from eval.form_variation import serialize_typed_frame

    example = _example(
        "Subtract 456 from 345.",
        "SUBTRACT",
        left=345,
        right=456,
        template_id=2,
    )

    assert serialize_typed_frame(example) == "OP=SUBTRACT;A=345;B=456"


def test_typed_frame_parser_rejects_ambiguous_or_noncanonical_protocol():
    from eval.form_variation import parse_typed_frame

    assert parse_typed_frame("OP=ADD;A=345;B=456") == {
        "operation": "ADD",
        "A": 345,
        "B": 456,
    }
    with pytest.raises(ValueError):
        parse_typed_frame("A=345;OP=ADD;B=456")
    with pytest.raises(ValueError):
        parse_typed_frame("OP=ADD;A=345;A=456")
    with pytest.raises(ValueError):
        parse_typed_frame("OP=UNKNOWN;A=345;B=456")


def test_contrast_balancing_is_deterministic_and_preserves_the_training_budget():
    from eval.form_variation import build_contrast_balanced_examples

    splits = build_condition_splits(
        variants_per_operation=4,
        train_pairs=[(11, 7), (13, 8), (17, 9)],
        eval_pairs=[(59, 31)],
        seed=42,
    )

    first = build_contrast_balanced_examples(splits.train, replacement_fraction=0.25, seed=17)
    second = build_contrast_balanced_examples(splits.train, replacement_fraction=0.25, seed=17)

    assert first == second
    assert len(first) == len(splits.train)
    assert {operation: sum(example.operation == operation for example in first) for operation in ("ADD", "SUBTRACT", "COMPARE")} == {
        operation: sum(example.operation == operation for example in splits.train)
        for operation in ("ADD", "SUBTRACT", "COMPARE")
    }
    assert {(example.left, example.right) for example in first} == {
        (example.left, example.right) for example in splits.train
    }
    assert {example.utterance for example in first} != {
        example.utterance for example in splits.train
    }


def test_contrast_training_text_remains_disjoint_from_the_development_suite():
    from eval.form_variation import build_contrast_balanced_examples, build_phase25_split_manifest

    splits = build_condition_splits(
        variants_per_operation=4,
        train_pairs=[(11, 7), (13, 8), (17, 9)],
        eval_pairs=[(59, 31)],
        seed=42,
    )
    contrast_examples = build_contrast_balanced_examples(
        splits.train,
        replacement_fraction=0.25,
        seed=17,
    )
    development_groups = build_fixed_pressure_test([(101, 44), (207, 86)])

    manifest = build_phase25_split_manifest(contrast_examples, development_groups)

    assert manifest["validation"] == "PASS"
    assert manifest["train_development_exact_overlap_count"] == 0
    assert manifest["tokenizer_corpus_sha256"] == manifest["training_corpus_sha256"]


def test_phase25_manifest_detects_any_future_sealed_suite_leakage():
    from eval.form_variation import build_phase25_split_manifest

    training = (_example("Training wording.", "ADD"),)
    development = {"held_out_templates": (_example("Development wording.", "ADD"),)}
    sealed = {"held_out_templates": (_example("Training wording.", "ADD"),)}

    manifest = build_phase25_split_manifest(training, development, sealed_groups=sealed)

    assert manifest["validation"] == "FAIL"
    assert manifest["train_sealed_exact_overlap_count"] == 1
    assert manifest["development_sealed_exact_overlap_count"] == 0


def test_model_exposes_final_hidden_states_without_changing_logits_or_weights(tmp_path):
    tokenizer = train_fixed_byte_tokenizer(tmp_path / "tokenizer")
    model = _tiny_model(tokenizer).eval()
    input_ids = torch.tensor(
        [[tokenizer.token_to_id("<BOS>"), *tokenizer.encode("probe").ids]],
        dtype=torch.long,
    )
    before = copy.deepcopy(model.state_dict())

    hidden = model.forward_hidden_states(input_ids)
    logits, _ = model(input_ids)

    assert hidden.shape == (1, input_ids.shape[1], model.config.hidden_size)
    assert torch.allclose(model.lm_head(hidden), logits)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_response_boundary_extraction_is_frozen_and_has_one_row_per_example(tmp_path):
    from eval.form_variation import extract_response_boundary_features

    tokenizer = train_fixed_byte_tokenizer(tmp_path / "tokenizer")
    model = _tiny_model(tokenizer).eval()
    examples = FormVariationGenerator(seed=4).render_minimal_contrasts(345, 456)

    features, labels = extract_response_boundary_features(
        model,
        tokenizer,
        examples,
        device=torch.device("cpu"),
    )

    assert features.shape == (3, model.config.hidden_size)
    assert features.requires_grad is False
    assert labels.tolist() == [0, 1, 2]
    assert all(parameter.grad is None for parameter in model.parameters())


def test_frozen_linear_probe_generalizes_a_linearly_separable_boundary():
    from eval.form_variation import fit_frozen_linear_probe

    train_features = torch.tensor(
        [[3.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 2.0], [-3.0, -3.0], [-2.0, -2.0]]
    )
    train_labels = torch.tensor([0, 0, 1, 1, 2, 2])
    evaluation_features = torch.tensor([[4.0, 0.0], [0.0, 4.0], [-4.0, -4.0]])

    predictions = fit_frozen_linear_probe(
        train_features,
        train_labels,
        evaluation_features,
        num_classes=3,
        seed=17,
    )

    assert predictions.tolist() == [0, 1, 2]
    assert train_features.grad is None
    assert evaluation_features.grad is None


def test_frozen_argument_probe_scores_canonical_roles_independently_of_value_error():
    from eval.form_variation import fit_frozen_linear_regression, score_argument_role_probe

    train_features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]],
        dtype=torch.float32,
    )
    train_targets = torch.tensor(
        [[10.0, 5.0], [20.0, 8.0], [20.0, 10.0], [40.0, 16.0]],
        dtype=torch.float32,
    )
    evaluation_features = torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
    expected = (
        _example("Subtract 15 from 30.", "SUBTRACT", left=30, right=15),
        _example("Subtract 24 from 60.", "SUBTRACT", left=60, right=24),
    )

    predictions = fit_frozen_linear_regression(train_features, train_targets, evaluation_features)
    scores = score_argument_role_probe(expected, predictions)

    assert torch.allclose(predictions, torch.tensor([[30.0, 15.0], [60.0, 24.0]]), atol=1e-3)
    assert scores["canonical_role_accuracy"] == 1.0
    assert scores["mean_absolute_value_error"] == pytest.approx(0.0, abs=1e-3)


def test_typed_frame_scoring_keeps_operation_binding_and_exact_match_separate():
    from eval.form_variation import score_typed_frame_predictions

    expected = (
        _example("Add the values.", "ADD", left=345, right=456),
        _example("Subtract the values.", "SUBTRACT", left=345, right=456),
        _example("Compare the values.", "COMPARE", left=345, right=456),
    )
    predictions = (
        "OP=ADD;A=345;B=456",       # fully correct
        "OP=SUBTRACT;A=456;B=345",  # correct operation, wrong binding
        "OP=ADD;A=345;B=456",       # wrong operation, correct binding
    )

    scores = score_typed_frame_predictions(expected, predictions)

    assert scores == {
        "operation_accuracy": pytest.approx(2 / 3),
        "argument_binding_accuracy": pytest.approx(2 / 3),
        "full_frame_exact_match": pytest.approx(1 / 3),
        "protocol_validity": 1.0,
    }


def test_training_fingerprint_changes_for_data_target_or_schedule_changes():
    from eval.form_variation import compute_phase25_training_fingerprint

    examples = (
        _example("Calculate 345 plus 456.", "ADD"),
        _example("Calculate 345 minus 456.", "SUBTRACT"),
    )
    schedule = {"max_steps": 400, "warmup_steps": 40, "batch_size": 16}

    baseline = compute_phase25_training_fingerprint(examples, schedule, target_mode="operation")
    identical = compute_phase25_training_fingerprint(examples, dict(schedule), target_mode="operation")
    changed_data = compute_phase25_training_fingerprint(examples[::-1], schedule, target_mode="operation")
    changed_target = compute_phase25_training_fingerprint(examples, schedule, target_mode="typed_frame")
    changed_schedule = compute_phase25_training_fingerprint(
        examples,
        {**schedule, "max_steps": 401},
        target_mode="operation",
    )

    assert baseline == identical
    assert len(baseline) == 64
    assert len({baseline, changed_data, changed_target, changed_schedule}) == 4


def test_stage_b_plan_reuses_baseline_and_registers_only_three_training_arms():
    from eval.form_variation import build_phase25_stage_b_arms

    splits = build_condition_splits(
        variants_per_operation=4,
        train_pairs=[(11, 7), (13, 8), (17, 9)],
        eval_pairs=[(59, 31)],
        seed=42,
    )

    arms = build_phase25_stage_b_arms(splits, replacement_fraction=0.25, contrast_seed=17)

    assert set(arms) == {"baseline", "minimal_contrast", "typed_frame", "typed_frame_contrast"}
    assert arms["baseline"]["requires_training"] is False
    assert sum(bool(arm["requires_training"]) for arm in arms.values()) == 3
    assert arms["baseline"]["target_mode"] == "operation"
    assert arms["minimal_contrast"]["target_mode"] == "operation"
    assert arms["typed_frame"]["target_mode"] == "typed_frame"
    assert arms["typed_frame_contrast"]["target_mode"] == "typed_frame"
    assert {len(arm["splits"].train) for arm in arms.values()} == {len(splits.train)}


def test_typed_candidate_prediction_returns_only_canonical_frames(tmp_path):
    from eval.form_variation import predict_typed_frames_batched

    tokenizer = train_fixed_byte_tokenizer(tmp_path / "tokenizer")
    model = _tiny_model(tokenizer).eval()
    examples = FormVariationGenerator(seed=4).render_minimal_contrasts(34, 21)

    predictions = predict_typed_frames_batched(
        model,
        tokenizer,
        examples,
        torch.device("cpu"),
        batch_size=8,
    )

    assert len(predictions) == len(examples)
    for example, prediction in zip(examples, predictions):
        parsed = __import__("eval.form_variation", fromlist=["parse_typed_frame"]).parse_typed_frame(prediction)
        assert parsed["operation"] in {"ADD", "SUBTRACT", "COMPARE"}
        assert {parsed["A"], parsed["B"]} == {example.left, example.right}


def test_stage_b_analysis_enforces_the_preregistered_worst_group_gate():
    from eval.form_variation import analyze_phase25_screen

    tracks = (
        "held_out_templates",
        "lexical_shift",
        "syntax_order_reversal",
        "discourse_distractor",
        "minimal_contrast",
    )

    def run(seed: int, score: float) -> dict:
        return {
            "seed": seed,
            "worst_robust_accuracy": score,
            "pressure_groups": {track: score for track in tracks},
        }

    report = {
        "arms": {
            "baseline": [run(seed, 0.2) for seed in (1, 2, 3, 4, 5)],
            "typed_frame": [run(seed, 0.35) for seed in (1, 2, 3, 4, 5)],
            "minimal_contrast": [run(seed, 0.25) for seed in (1, 2, 3, 4, 5)],
        }
    }

    analysis = analyze_phase25_screen(report)

    assert analysis["arms"]["typed_frame"]["passes_continuation_gate"] is True
    assert analysis["arms"]["minimal_contrast"]["passes_continuation_gate"] is False
    assert analysis["selected_arm"] == "typed_frame"
