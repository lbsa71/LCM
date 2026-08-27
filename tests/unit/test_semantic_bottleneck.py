import json

import pytest
import torch
import yaml

from eval.form_variation import (
    ConditionSplits,
    FormVariationExample,
    build_condition_splits,
    build_fixed_pressure_test,
    train_fixed_byte_tokenizer,
)
from eval.semantic_bottleneck import (
    SemanticInvarianceDataset,
    SemanticBottleneckDataset,
    SemanticBottleneckParser,
    analyze_phase4_stageb,
    analyze_phase4_stageb_probe_records,
    analyze_phase4_stagec,
    analyze_phase4_stagec_confirmation,
    analyze_bottleneck_screen,
    analyze_bottleneck_case_records,
    build_binding_counterbalanced_examples,
    build_phase4_stageb_development,
    build_phase4_stagec_confirmation,
    build_phase4_stagec_development,
    build_scaffold_counterbalanced_examples,
    build_stagec_invariance_pairs,
    paired_operation_consistency_loss,
    prepare_phase4_stagec_confirmation,
    prepare_bottleneck_screen,
    predict_bottleneck_labels,
    run_bottleneck_condition,
    run_phase4_stagec_confirmation,
    run_invariance_condition,
    score_bottleneck_predictions,
    semantic_binding_label,
)
from training.model import TransformerConfig


def _example(utterance: str, operation: str = "SUBTRACT") -> FormVariationExample:
    return FormVariationExample(
        utterance=utterance,
        target=f"OP={operation}",
        operation=operation,
        template_id=0,
        left=345,
        right=456,
    )


def test_semantic_binding_label_tracks_canonical_argument_mention_order():
    assert semantic_binding_label(_example("Calculate 345 minus 456.")) == 0
    assert semantic_binding_label(_example("Subtract 456 from 345.")) == 1


def test_binding_counterbalance_is_fixed_budget_and_independent_of_operation():
    splits = build_condition_splits(
        8,
        train_pairs=((11, 7), (13, 8)),
        eval_pairs=((17, 9),),
        seed=42,
    )

    balanced = build_binding_counterbalanced_examples(splits.train)

    assert len(balanced) == len(splits.train)
    original_frames = sorted((x.operation, x.left, x.right) for x in splits.train)
    balanced_frames = sorted((x.operation, x.left, x.right) for x in balanced)
    assert balanced_frames == original_frames
    for operation in ("ADD", "SUBTRACT", "COMPARE"):
        for left, right in ((11, 7), (13, 8)):
            labels = [
                semantic_binding_label(example)
                for example in balanced
                if (example.operation, example.left, example.right) == (operation, left, right)
            ]
            assert labels.count(0) == labels.count(1) == 4


def test_scaffold_counterbalance_replaces_without_changing_frames_or_binding_balance():
    splits = build_condition_splits(
        8,
        train_pairs=((11, 7), (13, 8)),
        eval_pairs=((17, 9),),
        seed=42,
    )
    balanced = build_binding_counterbalanced_examples(splits.train)

    scaffolded = build_scaffold_counterbalanced_examples(
        balanced,
        replacement_fraction=0.25,
        seed=42,
    )

    assert len(scaffolded) == len(balanced)
    assert sorted((x.operation, x.left, x.right) for x in scaffolded) == sorted(
        (x.operation, x.left, x.right) for x in balanced
    )
    for operation in ("ADD", "SUBTRACT", "COMPARE"):
        for left, right in ((11, 7), (13, 8)):
            cell = [
                example
                for example in scaffolded
                if (example.operation, example.left, example.right) == (operation, left, right)
            ]
            labels = [semantic_binding_label(example) for example in cell]
            scaffold = [example for example in cell if 5_000 <= example.template_id < 6_000]
            assert labels.count(0) == labels.count(1) == 4
            assert len(scaffold) == 2
            assert sorted(semantic_binding_label(example) for example in scaffold) == [0, 1]


def test_stageb_development_is_fresh_and_binding_balanced_by_track_and_operation():
    train_splits = build_condition_splits(
        8,
        train_pairs=((11, 7), (13, 8)),
        eval_pairs=((17, 9),),
        seed=42,
    )
    ordinary = build_binding_counterbalanced_examples(train_splits.train)
    scaffolded = build_scaffold_counterbalanced_examples(
        ordinary,
        replacement_fraction=0.25,
        seed=42,
    )

    development = build_phase4_stageb_development(
        ((751, 363), (863, 421), (977, 488), (1091, 537), (1217, 608), (1321, 659))
    )

    assert set(development) == {
        "seen_form_unseen_operands",
        "held_out_templates",
        "lexical_shift",
        "syntax_order_reversal",
        "discourse_distractor",
        "minimal_contrast",
    }
    train_utterances = {example.utterance for example in ordinary + scaffolded}
    development_utterances = {
        example.utterance for examples in development.values() for example in examples
    }
    assert train_utterances.isdisjoint(development_utterances)
    for examples in development.values():
        assert len(examples) == 18
        for operation in ("ADD", "SUBTRACT", "COMPARE"):
            labels = [
                semantic_binding_label(example)
                for example in examples
                if example.operation == operation
            ]
            assert labels.count(0) == labels.count(1) == 3


def test_analyze_phase4_stageb_estimates_factorial_effects_and_applies_gate():
    tracks = (
        "held_out_templates",
        "lexical_shift",
        "syntax_order_reversal",
        "discourse_distractor",
        "minimal_contrast",
    )

    def run(seed: int, operation: float, binding: float = 0.5) -> dict:
        return {
            "seed": seed,
            "worst_robust_accuracy": operation,
            "pressure_groups": {track: operation for track in tracks},
            "pressure_group_details": {
                track: {
                    "operation_accuracy": operation,
                    "binding_accuracy": binding,
                    "joint_accuracy": min(operation, binding),
                }
                for track in tracks
            },
        }

    report = {
        "baseline_arm": "matched_generative_baseline",
        "gate": {"minimum_worst_group_gain": 0.10, "maximum_track_regression": 0.05},
        "arms": {
            "matched_generative_baseline": [run(17, 0.20), run(29, 0.20)],
            "weight_1_standard": [run(17, 0.28, 0.60), run(29, 0.28, 0.60)],
            "weight_1_scaffold": [run(17, 0.32, 0.70), run(29, 0.32, 0.70)],
            "weight_025_standard": [run(17, 0.35, 0.65), run(29, 0.35, 0.65)],
            "weight_025_scaffold": [run(17, 0.40, 0.80), run(29, 0.40, 0.80)],
        },
    }

    analysis = analyze_phase4_stageb(report)

    assert analysis["arm_effects"]["weight_025_scaffold"]["gate_passed"] is True
    assert analysis["factorial_effects"]["scaffold_at_weight_1"]["operation_worst"]["mean"] == pytest.approx(0.04)
    assert analysis["factorial_effects"]["reduced_weight_standard"]["operation_macro"]["mean"] == pytest.approx(0.07)
    assert analysis["factorial_effects"]["interaction"]["operation_worst"]["mean"] == pytest.approx(0.01)
    assert analysis["selected_arm"] == "weight_025_scaffold"


def test_analyze_stageb_probe_records_separates_head_and_recoverable_representation():
    records = [
        {
            "arm": "weight_025_scaffold",
            "seed": 17,
            "track": "lexical_shift",
            "expected_operation": "SUBTRACT",
            "expected_binding": 0,
            "head_operation": "ADD",
            "head_binding": 0,
            "probe_operation": "SUBTRACT",
            "probe_binding": 0,
        },
        {
            "arm": "weight_025_scaffold",
            "seed": 17,
            "track": "lexical_shift",
            "expected_operation": "COMPARE",
            "expected_binding": 1,
            "head_operation": "COMPARE",
            "head_binding": 0,
            "probe_operation": "COMPARE",
            "probe_binding": 1,
        },
    ]

    analysis = analyze_phase4_stageb_probe_records(records)
    track = analysis["arms"]["weight_025_scaffold"]["tracks"]["lexical_shift"]

    assert track["operation_head_accuracy"] == 0.5
    assert track["operation_probe_accuracy"] == 1.0
    assert track["binding_head_accuracy"] == 0.5
    assert track["binding_probe_accuracy"] == 1.0
    assert track["operation_probe_repairs"] == 1


def test_stagec_pairs_cover_each_example_once_with_opposite_binding_order():
    splits = build_condition_splits(
        8,
        train_pairs=((11, 7), (13, 8)),
        eval_pairs=((17, 9),),
        seed=42,
    )
    ordinary = build_binding_counterbalanced_examples(splits.train)
    scaffolded = build_scaffold_counterbalanced_examples(
        ordinary, replacement_fraction=0.25, seed=42
    )

    pairs = build_stagec_invariance_pairs(scaffolded, seed=42)

    assert len(pairs) * 2 == len(scaffolded)
    paired_examples = [example for pair in pairs for example in pair]
    assert sorted(
        (x.operation, x.left, x.right, x.template_id, x.utterance) for x in paired_examples
    ) == sorted((x.operation, x.left, x.right, x.template_id, x.utterance) for x in scaffolded)
    for first, second in pairs:
        assert (first.operation, first.left, first.right) == (
            second.operation,
            second.left,
            second.right,
        )
        assert semantic_binding_label(first) != semantic_binding_label(second)


def test_invariance_dataset_and_consistency_loss_preserve_two_sequences_per_pair(tmp_path):
    tokenizer = train_fixed_byte_tokenizer(tmp_path / "tokenizer")
    pairs = (
        (
            _example("Calculate 345 minus 456."),
            _example("Subtract 456 from 345."),
        ),
    )

    dataset = SemanticInvarianceDataset(pairs, tokenizer, max_length=64)
    item = dataset[0]
    same = torch.tensor([[2.0, -1.0, 0.5]])
    different = torch.tensor([[-1.0, 2.0, 0.5]])

    assert set(item) == {
        "first_input_ids",
        "first_boundary_index",
        "first_operation_label",
        "first_binding_label",
        "second_input_ids",
        "second_boundary_index",
        "second_operation_label",
        "second_binding_label",
    }
    assert item["first_operation_label"].item() == item["second_operation_label"].item()
    assert item["first_binding_label"].item() != item["second_binding_label"].item()
    assert paired_operation_consistency_loss(same, same).item() == pytest.approx(0.0)
    assert paired_operation_consistency_loss(same, different).item() > 0


def test_stagec_development_is_fresh_balanced_and_distinct_from_stageb():
    stageb = build_phase4_stageb_development(
        ((751, 363), (863, 421), (977, 488), (1091, 537), (1217, 608), (1321, 659))
    )
    stagec = build_phase4_stagec_development(
        ((2129, 1061), (2267, 1129), (2399, 1193), (2531, 1261), (2671, 1331), (2801, 1399))
    )

    assert set(stagec) == set(stageb)
    assert {
        example.utterance for examples in stagec.values() for example in examples
    }.isdisjoint({example.utterance for examples in stageb.values() for example in examples})
    for examples in stagec.values():
        assert len(examples) == 18
        for operation in ("ADD", "SUBTRACT", "COMPARE"):
            labels = [
                semantic_binding_label(example)
                for example in examples
                if example.operation == operation
            ]
            assert labels.count(0) == labels.count(1) == 3


def test_stagec_confirmation_is_fresh_and_pairs_both_orders_for_every_frame():
    pairs = ((4001, 1999), (4127, 2063), (4259, 2129), (4391, 2197), (4523, 2267), (4657, 2333))
    stageb = build_phase4_stageb_development(pairs)
    stagec = build_phase4_stagec_development(pairs)
    confirmation = build_phase4_stagec_confirmation(pairs)

    confirmation_text = {
        example.utterance for examples in confirmation.values() for example in examples
    }
    prior_text = {
        example.utterance
        for suite in (stageb, stagec)
        for examples in suite.values()
        for example in examples
    }
    assert confirmation_text.isdisjoint(prior_text)
    assert set(confirmation) == set(stagec)
    for examples in confirmation.values():
        assert len(examples) == 36
        for operation in ("ADD", "SUBTRACT", "COMPARE"):
            operation_examples = [example for example in examples if example.operation == operation]
            assert len(operation_examples) == 12
            by_frame = {}
            for example in operation_examples:
                by_frame.setdefault((example.left, example.right), []).append(
                    semantic_binding_label(example)
                )
            assert all(sorted(labels) == [0, 1] for labels in by_frame.values())


def test_analyze_stagec_confirmation_applies_gate_and_reports_pair_agreement():
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

    arms = {
        "matched_generative_baseline": [run(17, 0.20), run(29, 0.20)],
        "lead_no_consistency": [run(17, 0.30), run(29, 0.30)],
        "consistency_025": [run(17, 0.40), run(29, 0.40)],
    }
    records = []
    for seed in (17, 29):
        for arm in arms:
            for track in tracks:
                for binding in (0, 1):
                    records.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "track": track,
                            "left": 4001,
                            "right": 1999,
                            "expected_operation": "ADD",
                            "expected_binding": binding,
                            "predicted_operation": (
                                "ADD"
                                if arm != "matched_generative_baseline" or binding == 0
                                else "SUBTRACT"
                            ),
                        }
                    )
    report = {
        "baseline_arm": "matched_generative_baseline",
        "lead_arm": "lead_no_consistency",
        "candidate_arm": "consistency_025",
        "gate": {"minimum_worst_group_gain": 0.10, "maximum_track_regression": 0.05},
        "arms": arms,
    }

    analysis = analyze_phase4_stagec_confirmation(report, records)

    assert analysis["confirmation_passed"] is True
    assert analysis["candidate_effect"]["worst_group_delta"]["mean"] == pytest.approx(0.20)
    assert analysis["candidate_vs_lead"]["worst_group_delta"]["mean"] == pytest.approx(0.10)
    assert analysis["pair_invariance"]["consistency_025"]["macro_agreement"]["mean"] == 1.0
    assert (
        analysis["pair_invariance_effects"]["candidate_vs_baseline"]["macro_agreement"]["mean"]
        == pytest.approx(1.0)
    )


def test_prepare_stagec_confirmation_requires_selected_sources_and_seals_suite(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "stagec"
    stagea_dir = tmp_path / "stagea"
    stageb_dir = tmp_path / "stageb"
    confirmation_pairs = (
        (4001, 1999),
        (4127, 2063),
        (4259, 2129),
        (4391, 2197),
        (4523, 2267),
        (4657, 2333),
    )
    development = build_phase4_stagec_development(
        ((2129, 1061), (2267, 1129), (2399, 1193), (2531, 1261), (2671, 1331), (2801, 1399))
    )
    training_example = _example("Calculate 345 minus 456.")
    splits = ConditionSplits(
        train=(training_example,),
        seen_form=(),
        same_meaning_unseen_form=(),
        unseen_operands_seen_form=(),
        minimal_contrasts=(),
    )
    config = {
        "name": "confirmation-test",
        "phase4_stagec": {
            "output_dir": str(output_dir),
            "stagea_output_dir": str(stagea_dir),
            "stageb_output_dir": str(stageb_dir),
            "seeds": [17],
            "train_pairs": [[11, 7]],
            "eval_pairs": [[59, 31]],
            "development_pairs": [[2129, 1061]],
            "confirmation_pairs": [list(pair) for pair in confirmation_pairs],
            "gate": {"minimum_worst_group_gain": 0.10, "maximum_track_regression": 0.05},
        },
        "training": {"device": "cpu"},
        "evaluation": {"batch_size": 64},
    }
    registration = {
        "seeds": [17],
        "arm_fingerprints": {
            "matched_generative_baseline": "baseline-fp",
            "lead_no_consistency": "lead-fp",
            "consistency_025": "candidate-fp",
        },
        "gate": config["phase4_stagec"]["gate"],
    }
    output_dir.mkdir(parents=True)
    (output_dir / "screen_analysis.json").write_text(
        json.dumps({"selected_arm": "consistency_025", "sealed_confirmation_required": True}),
        encoding="utf-8",
    )
    for directory, checkpoint, fingerprint_key, fingerprint in (
        (stagea_dir / "matched_generative_baseline" / "seed_17", "parser_final.pt", "phase4_training_fingerprint", "baseline-fp"),
        (stageb_dir / "weight_025_scaffold" / "seed_17", "bottleneck_final.pt", "phase4_stageb_training_fingerprint", "lead-fp"),
        (output_dir / "consistency_025" / "seed_17", "bottleneck_final.pt", "phase4_stagec_training_fingerprint", "candidate-fp"),
    ):
        directory.mkdir(parents=True)
        (directory / checkpoint).write_bytes(b"checkpoint")
        (directory / "metrics.json").write_text(
            json.dumps({"seed": 17, fingerprint_key: fingerprint}), encoding="utf-8"
        )
    monkeypatch.setattr(
        "eval.semantic_bottleneck.prepare_phase4_stagec",
        lambda _config_path: {
            "config": config,
            "experiment": config["phase4_stagec"],
            "splits": splits,
            "development": development,
            "registration": registration,
        },
    )

    prepared = prepare_phase4_stagec_confirmation("unused.yaml")

    assert prepared["registration"]["sealed_suite_created"] is True
    assert prepared["registration"]["selected_arm"] == "consistency_025"
    assert prepared["registration"]["sealed_examples"] == 216
    confirmation_dir = output_dir / "sealed_confirmation"
    assert (confirmation_dir / "registration.json").exists()
    assert (confirmation_dir / "suite.json").exists()
    assert (confirmation_dir / "split_manifest.json").exists()


def test_run_stagec_confirmation_writes_results_and_eval_metrics(tmp_path, monkeypatch):
    output_dir = tmp_path / "stagec"
    suite = build_phase4_stagec_confirmation(
        ((4001, 1999), (4127, 2063), (4259, 2129), (4391, 2197), (4523, 2267), (4657, 2333))
    )
    experiment = {
        "output_dir": str(output_dir),
        "stagea_output_dir": str(tmp_path / "stagea"),
        "stageb_output_dir": str(tmp_path / "stageb"),
        "seeds": [17, 29],
        "gate": {"minimum_worst_group_gain": 0.10, "maximum_track_regression": 0.05},
    }
    config = {
        "name": "confirmation-run-test",
        "training": {"device": "cpu"},
        "evaluation": {"batch_size": 64},
    }
    registration = {
        "experiment": "confirmation-run-test_sealed_confirmation",
        "selected_arm": "consistency_025",
        "seeds": [17, 29],
        "gate": experiment["gate"],
        "sealed_suite_created": True,
    }
    monkeypatch.setattr(
        "eval.semantic_bottleneck.prepare_phase4_stagec_confirmation",
        lambda _config_path: {
            "config": config,
            "experiment": experiment,
            "confirmation": suite,
            "registration": registration,
        },
    )

    def fake_evaluation(_run_dir, _suite, *, seed, arm_name, **_kwargs):
        score = {"lead_no_consistency": 0.30, "consistency_025": 0.40}[arm_name]
        metrics = {
            "seed": seed,
            "arm": arm_name,
            "pressure_groups": {track: score for track in suite},
            "worst_robust_accuracy": score,
        }
        records = [
            {
                "seed": seed,
                "arm": arm_name,
                "track": track,
                "left": example.left,
                "right": example.right,
                "expected_operation": example.operation,
                "expected_binding": semantic_binding_label(example),
                "predicted_operation": example.operation,
            }
            for track, examples in suite.items()
            for example in examples
        ]
        return metrics, records

    def fake_generative(_run_dir, _config, _suite, *, seed, **_kwargs):
        metrics, records = fake_evaluation(
            _run_dir,
            _suite,
            seed=seed,
            arm_name="lead_no_consistency",
        )
        metrics.update(arm="matched_generative_baseline", worst_robust_accuracy=0.20)
        metrics["pressure_groups"] = {track: 0.20 for track in suite}
        for record in records:
            record["arm"] = "matched_generative_baseline"
        return metrics, records

    monkeypatch.setattr(
        "eval.semantic_bottleneck._evaluate_stageb_generative_source", fake_generative
    )
    monkeypatch.setattr(
        "eval.semantic_bottleneck._evaluate_stageb_bottleneck_source", fake_evaluation
    )

    result = run_phase4_stagec_confirmation("unused.yaml")

    assert result["analysis"]["confirmation_passed"] is True
    confirmation_dir = output_dir / "sealed_confirmation"
    assert (confirmation_dir / "results.json").exists()
    assert (confirmation_dir / "analysis.json").exists()
    completed_registration = json.loads(
        (confirmation_dir / "registration.json").read_text(encoding="utf-8")
    )
    assert completed_registration["evaluation_completed"] is True
    assert completed_registration["confirmation_passed"] is True
    eval_metrics = json.loads((confirmation_dir / "eval_metrics.json").read_text(encoding="utf-8"))
    assert len(eval_metrics["evaluations"]) == 6
    with pytest.raises(ValueError, match="already completed"):
        run_phase4_stagec_confirmation("unused.yaml")


def test_analyze_phase4_stagec_applies_gate_and_reports_consistency_weight_effects():
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
            "pressure_group_details": {
                track: {
                    "operation_accuracy": score,
                    "binding_accuracy": 0.8,
                    "joint_accuracy": score,
                }
                for track in tracks
            },
        }

    report = {
        "baseline_arm": "matched_generative_baseline",
        "lead_arm": "lead_no_consistency",
        "gate": {"minimum_worst_group_gain": 0.10, "maximum_track_regression": 0.05},
        "arms": {
            "matched_generative_baseline": [run(17, 0.20), run(29, 0.20)],
            "lead_no_consistency": [run(17, 0.28), run(29, 0.28)],
            "consistency_025": [run(17, 0.35), run(29, 0.35)],
            "consistency_1": [run(17, 0.42), run(29, 0.42)],
        },
    }

    analysis = analyze_phase4_stagec(report)

    assert analysis["arm_effects"]["consistency_1"]["gate_passed"] is True
    assert analysis["consistency_effects"]["consistency_025_vs_lead"]["worst_group"]["mean"] == pytest.approx(0.07)
    assert analysis["consistency_effects"]["consistency_1_vs_025"]["macro_operation"]["mean"] == pytest.approx(0.07)
    assert analysis["selected_arm"] == "consistency_1"


def test_semantic_bottleneck_dataset_ends_at_response_boundary(tmp_path):
    tokenizer = train_fixed_byte_tokenizer(tmp_path / "tokenizer")
    dataset = SemanticBottleneckDataset(
        (_example("Subtract 456 from 345."),), tokenizer, max_length=64
    )

    item = dataset[0]

    assert set(item) == {"input_ids", "boundary_index", "operation_label", "binding_label"}
    assert item["input_ids"].shape == (64,)
    assert item["operation_label"].item() == 1
    assert item["binding_label"].item() == 1
    boundary = item["boundary_index"].item()
    decoded = tokenizer.decode(item["input_ids"][: boundary + 1].tolist(), skip_special_tokens=False)
    assert decoded.endswith("<ASSISTANT> ")
    assert "OP=SUBTRACT" not in decoded


def test_semantic_bottleneck_parser_separates_operation_and_binding_losses():
    config = TransformerConfig(
        vocab_size=267,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    model = SemanticBottleneckParser(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    boundaries = torch.tensor([7, 9])
    operations = torch.tensor([0, 2])
    bindings = torch.tensor([0, 1])

    operation_only = model(
        input_ids,
        boundaries,
        operation_labels=operations,
        binding_labels=bindings,
        binding_loss_weight=0.0,
    )
    multitask = model(
        input_ids,
        boundaries,
        operation_labels=operations,
        binding_labels=bindings,
        binding_loss_weight=1.0,
    )

    assert operation_only["operation_logits"].shape == (2, 3)
    assert operation_only["binding_logits"].shape == (2, 2)
    assert operation_only["loss"].item() == operation_only["operation_loss"].item()
    assert multitask["loss"].item() == pytest.approx(
        multitask["operation_loss"].item() + multitask["binding_loss"].item()
    )


def test_predict_bottleneck_labels_returns_explicit_layer_outputs(tmp_path):
    tokenizer = train_fixed_byte_tokenizer(tmp_path / "tokenizer")
    config = TransformerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    model = SemanticBottleneckParser(config)
    examples = (
        _example("Calculate 345 minus 456."),
        _example("Subtract 456 from 345."),
    )

    operations, bindings = predict_bottleneck_labels(
        model, tokenizer, examples, torch.device("cpu"), batch_size=2
    )

    assert len(operations) == len(bindings) == 2
    assert set(operations).issubset({"ADD", "SUBTRACT", "COMPARE"})
    assert set(bindings).issubset({0, 1})


def test_score_bottleneck_predictions_keeps_operation_binding_and_joint_separate():
    examples = (
        _example("Calculate 345 minus 456."),
        _example("Subtract 456 from 345."),
    )
    scores = score_bottleneck_predictions(
        examples,
        operation_predictions=("SUBTRACT", "ADD"),
        binding_predictions=(0, 1),
    )

    assert scores == {
        "operation_accuracy": 0.5,
        "binding_accuracy": 1.0,
        "joint_accuracy": 0.5,
    }


def test_analyze_bottleneck_screen_applies_gate_and_prefers_useful_binding_head():
    tracks = (
        "held_out_templates",
        "lexical_shift",
        "syntax_order_reversal",
        "discourse_distractor",
        "minimal_contrast",
    )

    def run(seed: int, operation: float, joint: float = 0.0) -> dict:
        return {
            "seed": seed,
            "worst_robust_accuracy": operation,
            "pressure_groups": {track: operation for track in tracks},
            "pressure_group_details": {
                track: {
                    "operation_accuracy": operation,
                    "binding_accuracy": 1.0,
                    "joint_accuracy": joint,
                }
                for track in tracks
            },
        }

    report = {
        "baseline_arm": "matched_generative_baseline",
        "arms": {
            "matched_generative_baseline": [run(17, 0.20), run(29, 0.20)],
            "discriminative_operation": [run(17, 0.35, 0.10), run(29, 0.35, 0.10)],
            "discriminative_operation_binding": [run(17, 0.33, 0.30), run(29, 0.33, 0.30)],
        },
        "gate": {"minimum_worst_group_gain": 0.10, "maximum_track_regression": 0.05},
    }

    analysis = analyze_bottleneck_screen(report)

    assert analysis["arm_effects"]["discriminative_operation"]["gate_passed"] is True
    assert analysis["arm_effects"]["discriminative_operation_binding"]["gate_passed"] is True
    assert analysis["binding_head_comparison"]["operation_noninferior"] is True
    assert analysis["binding_head_comparison"]["joint_macro_delta"]["mean"] == 0.2
    assert analysis["selected_arm"] == "discriminative_operation_binding"


def test_analyze_bottleneck_case_records_exposes_regressions_and_binding_holes():
    records = [
        {
            "seed": 17,
            "track": "discourse_distractor",
            "utterance": "u1",
            "expected_operation": "ADD",
            "expected_binding": 0,
            "baseline_operation": "ADD",
            "operation_only_operation": "ADD",
            "operation_only_binding": 1,
            "multitask_operation": "SUBTRACT",
            "multitask_binding": 0,
            "multitask_frozen_probe_binding": 0,
        },
        {
            "seed": 17,
            "track": "minimal_contrast",
            "utterance": "u2",
            "expected_operation": "COMPARE",
            "expected_binding": 0,
            "baseline_operation": "COMPARE",
            "operation_only_operation": "COMPARE",
            "operation_only_binding": 0,
            "multitask_operation": "COMPARE",
            "multitask_binding": 1,
            "multitask_frozen_probe_binding": 0,
        },
    ]

    analysis = analyze_bottleneck_case_records(records)

    assert analysis["tracks"]["discourse_distractor"]["multitask_operation_accuracy"] == 0.0
    assert analysis["tracks"]["minimal_contrast"]["multitask_binding_accuracy"] == 0.0
    assert analysis["tracks"]["minimal_contrast"]["multitask_frozen_probe_binding_accuracy"] == 1.0
    assert analysis["multitask_operation_confusion"]["ADD"]["SUBTRACT"] == 1
    assert analysis["baseline_correct_multitask_wrong"][0]["utterance"] == "u1"


def test_run_bottleneck_condition_persists_layer_specific_metrics(tmp_path):
    splits = build_condition_splits(
        1,
        train_pairs=((11, 7), (13, 8)),
        eval_pairs=((17, 9),),
        seed=42,
    )
    pressure = build_fixed_pressure_test(((101, 44),))
    config = {
        "tokenizer": {"mode": "fixed_byte", "vocab_size": 512},
        "model": {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "max_position_embeddings": 128,
            "tie_word_embeddings": True,
        },
        "training": {
            "learning_rate": 1e-3,
            "min_learning_rate": 1e-4,
            "weight_decay": 0.01,
            "batch_size": 4,
            "max_steps": 1,
            "warmup_steps": 1,
            "device": "cpu",
        },
        "evaluation": {"batch_size": 16},
    }

    metrics = run_bottleneck_condition(
        splits,
        tmp_path / "run",
        config,
        seed=17,
        pressure_groups=pressure,
        binding_loss_weight=1.0,
        arm_name="discriminative_operation_binding",
    )

    assert metrics["steps"] == 1
    assert metrics["arm"] == "discriminative_operation_binding"
    assert metrics["binding_loss_weight"] == 1.0
    assert set(metrics["pressure_groups"]) == set(pressure)
    assert set(metrics["pressure_group_details"]["lexical_shift"]) == {
        "operation_accuracy",
        "binding_accuracy",
        "joint_accuracy",
    }
    assert (tmp_path / "run" / "metrics.json").exists()
    assert (tmp_path / "run" / "bottleneck_final.pt").exists()
    assert (tmp_path / "run" / "step_metrics.csv").exists()


def test_run_invariance_condition_preserves_sequence_budget_and_metrics(tmp_path):
    base = build_condition_splits(
        8,
        train_pairs=((11, 7), (13, 8)),
        eval_pairs=((17, 9),),
        seed=42,
    )
    ordinary = build_binding_counterbalanced_examples(base.train)
    scaffolded = build_scaffold_counterbalanced_examples(
        ordinary, replacement_fraction=0.25, seed=42
    )
    splits = type(base)(
        train=scaffolded,
        seen_form=base.seen_form,
        same_meaning_unseen_form=base.same_meaning_unseen_form,
        unseen_operands_seen_form=base.unseen_operands_seen_form,
        minimal_contrasts=base.minimal_contrasts,
    )
    pairs = build_stagec_invariance_pairs(scaffolded, seed=42)
    pressure = build_fixed_pressure_test(((101, 44),))
    config = {
        "tokenizer": {"mode": "fixed_byte", "vocab_size": 512},
        "model": {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "max_position_embeddings": 128,
            "tie_word_embeddings": True,
        },
        "training": {
            "learning_rate": 1e-3,
            "min_learning_rate": 1e-4,
            "weight_decay": 0.01,
            "batch_size": 4,
            "max_steps": 1,
            "warmup_steps": 1,
            "device": "cpu",
        },
        "evaluation": {"batch_size": 16},
    }

    metrics = run_invariance_condition(
        splits,
        pairs,
        tmp_path / "run",
        config,
        seed=17,
        pressure_groups=pressure,
        binding_loss_weight=0.25,
        consistency_weight=1.0,
        pair_batch_size=2,
        arm_name="consistency_1",
    )

    assert metrics["steps"] == 1
    assert metrics["pair_batch_size"] == 2
    assert metrics["sequences_per_step"] == 4
    assert metrics["consistency_weight"] == 1.0
    assert set(metrics["pressure_group_details"]) == set(pressure)
    assert (tmp_path / "run" / "bottleneck_final.pt").exists()


def test_prepare_bottleneck_screen_validates_reused_source_without_training(tmp_path):
    train_pairs = ((11, 7), (13, 8))
    eval_pairs = ((17, 9),)
    pressure_pairs = ((101, 44),)
    splits = build_condition_splits(8, train_pairs, eval_pairs, seed=42)
    pressure = build_fixed_pressure_test(pressure_pairs)
    from eval.form_variation import build_clean_split_manifest

    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps(
            {
                "seeds": [17, 29],
                "split_manifests": {"8": build_clean_split_manifest(splits.train, pressure)},
                "runs": {
                    "8": {
                        "R1": [
                            {
                                "seed": seed,
                                "steps": 12,
                                "train_examples": 48,
                                "tokenizer_mode": "fixed_byte",
                                "pressure_groups": {track: 0.2 for track in pressure},
                                "worst_robust_accuracy": 0.2,
                            }
                            for seed in (17, 29)
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "phase4"
    config = {
        "name": "phase4_unit",
        "seed": 42,
        "phase4_bottleneck": {
            "output_dir": str(output_dir),
            "source_results": str(source_path),
            "source_variants": 8,
            "source_exposure": "R1",
            "variants_per_operation": 8,
            "exposure": 1.0,
            "binding_counterbalance": True,
            "binding_loss_weight": 1.0,
            "seeds": [17, 29],
            "train_pairs": [list(pair) for pair in train_pairs],
            "eval_pairs": [list(pair) for pair in eval_pairs],
            "pressure_pairs": [list(pair) for pair in pressure_pairs],
            "gate": {"minimum_worst_group_gain": 0.10, "maximum_track_regression": 0.05},
        },
        "tokenizer": {"mode": "fixed_byte", "vocab_size": 512},
        "model": {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "max_position_embeddings": 128,
            "tie_word_embeddings": True,
        },
        "training": {
            "learning_rate": 1e-3,
            "min_learning_rate": 1e-4,
            "weight_decay": 0.01,
            "batch_size": 4,
            "max_steps": 12,
            "warmup_steps": 1,
            "device": "cpu",
        },
        "evaluation": {"batch_size": 16},
    }
    config_path = tmp_path / "phase4.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    preparation = prepare_bottleneck_screen(str(config_path))

    assert preparation["registration"]["historical_reference_validation"] == "PASS"
    assert preparation["registration"]["schedule"]["max_steps"] == 12
    assert preparation["registration"]["binding_label_counts"] == {"A_FIRST": 24, "B_FIRST": 24}
    assert "matched_generative_baseline" in preparation["registration"]["arm_fingerprints"]
    assert preparation["registration"]["arm_fingerprints"]["discriminative_operation"] != preparation[
        "registration"
    ]["arm_fingerprints"]["discriminative_operation_binding"]
    assert (output_dir / "registration.json").exists()
    assert (output_dir / "split_manifest.json").exists()
    assert not (output_dir / "discriminative_operation").exists()
