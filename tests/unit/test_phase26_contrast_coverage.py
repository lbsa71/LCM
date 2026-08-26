"""Phase 2.6 specifications for targeted minimal × lexical contrast coverage."""

from __future__ import annotations

import pytest

from eval.form_variation import (
    OPERATIONS,
    ConditionSplits,
    build_condition_splits,
    build_fixed_pressure_test,
    build_phase25_stage_b_arms,
)


def _splits() -> ConditionSplits:
    return build_condition_splits(
        variants_per_operation=4,
        train_pairs=[(11, 7), (13, 8), (17, 9)],
        eval_pairs=[(59, 31)],
        seed=42,
    )


def test_lexical_contrasts_are_deterministic_balanced_and_budget_preserving():
    from eval.form_variation import build_lexical_contrast_examples

    splits = _splits()
    first = build_lexical_contrast_examples(splits.train, replacement_fraction=0.25, seed=29)
    second = build_lexical_contrast_examples(splits.train, replacement_fraction=0.25, seed=29)

    assert first == second
    assert len(first) == len(splits.train)
    assert {(example.left, example.right) for example in first} == {
        (example.left, example.right) for example in splits.train
    }
    for operation in OPERATIONS:
        assert sum(example.operation == operation for example in first) == sum(
            example.operation == operation for example in splits.train
        )
    assert sum(2_000 <= example.template_id < 3_000 for example in first) == 9


def test_lexical_training_language_is_disjoint_from_the_development_suite():
    from eval.form_variation import build_lexical_contrast_examples, build_phase25_split_manifest

    examples = build_lexical_contrast_examples(_splits().train, replacement_fraction=0.25, seed=29)
    pressure = build_fixed_pressure_test([(101, 44), (207, 86)])

    manifest = build_phase25_split_manifest(examples, pressure)

    assert manifest["validation"] == "PASS"
    assert manifest["train_development_exact_overlap_count"] == 0


def test_phase26_factorial_reuses_prior_arms_and_registers_ten_new_runs_only():
    from eval.form_variation import build_phase26_contrast_arms

    splits = _splits()
    phase25 = build_phase25_stage_b_arms(splits, replacement_fraction=0.25, contrast_seed=17)
    arms = build_phase26_contrast_arms(
        splits,
        replacement_fraction=0.25,
        minimal_seed=17,
        lexical_seed=29,
    )

    assert set(arms) == {
        "baseline",
        "minimal_contrast",
        "lexical_contrast",
        "minimal_lexical_contrast",
    }
    assert arms["baseline"]["requires_training"] is False
    assert arms["minimal_contrast"]["requires_training"] is False
    assert arms["lexical_contrast"]["requires_training"] is True
    assert arms["minimal_lexical_contrast"]["requires_training"] is True
    assert sum(bool(arm["requires_training"]) for arm in arms.values()) == 2
    assert {len(arm["splits"].train) for arm in arms.values()} == {len(splits.train)}
    assert arms["minimal_contrast"]["splits"].train == phase25["minimal_contrast"]["splits"].train
    assert {arm["target_mode"] for arm in arms.values()} == {"operation"}


def test_combined_arm_contains_one_disjoint_replacement_per_factor_and_cell():
    from eval.form_variation import build_phase26_contrast_arms

    arms = build_phase26_contrast_arms(
        _splits(),
        replacement_fraction=0.25,
        minimal_seed=17,
        lexical_seed=29,
    )
    combined = arms["minimal_lexical_contrast"]["splits"].train

    for left, right in {(example.left, example.right) for example in combined}:
        for operation in OPERATIONS:
            cell = [
                example
                for example in combined
                if example.left == left and example.right == right and example.operation == operation
            ]
            assert len(cell) == 4
            assert sum(1_000 <= example.template_id < 2_000 for example in cell) == 1
            assert sum(2_000 <= example.template_id < 3_000 for example in cell) == 1


def test_phase26_source_reuse_requires_paired_seeds_steps_and_fingerprints():
    from eval.form_variation import validate_phase26_reused_metrics

    def run(seed: int, fingerprint: str) -> dict:
        return {"seed": seed, "steps": 800, "phase25_training_fingerprint": fingerprint}

    source = {
        "arms": {
            "baseline": [run(17, "baseline-hash"), run(29, "baseline-hash")],
            "minimal_contrast": [run(17, "minimal-hash"), run(29, "minimal-hash")],
        }
    }
    expected = {"baseline": "baseline-hash", "minimal_contrast": "minimal-hash"}

    reused = validate_phase26_reused_metrics(source, expected, seeds=(17, 29), max_steps=800)

    assert [run["seed"] for run in reused["baseline"]] == [17, 29]
    assert [run["seed"] for run in reused["minimal_contrast"]] == [17, 29]

    source["arms"]["minimal_contrast"][1]["phase25_training_fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_phase26_reused_metrics(source, expected, seeds=(17, 29), max_steps=800)


def test_phase26_analysis_reports_paired_main_and_interaction_effects():
    from eval.form_variation import analyze_phase26_screen

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

    seeds = (1, 2, 3, 4, 5)
    report = {
        "arms": {
            "baseline": [run(seed, 0.2) for seed in seeds],
            "minimal_contrast": [run(seed, 0.3) for seed in seeds],
            "lexical_contrast": [run(seed, 0.4) for seed in seeds],
            "minimal_lexical_contrast": [run(seed, 0.7) for seed in seeds],
        }
    }

    analysis = analyze_phase26_screen(report)
    effects = analysis["factorial_effects"]["worst_robust_accuracy"]

    assert effects["minimal_without_lexical"]["mean"] == 0.1
    assert effects["lexical_without_minimal"]["mean"] == 0.2
    assert effects["minimal_with_lexical"]["mean"] == 0.3
    assert effects["lexical_with_minimal"]["mean"] == 0.4
    assert effects["interaction"]["mean"] == 0.2
    assert analysis["selected_arm"] == "minimal_lexical_contrast"


def test_phase26_confirmation_retrains_a_fresh_baseline_and_selected_arm():
    from eval.form_variation import build_phase26_confirmation_arms

    arms = build_phase26_confirmation_arms(
        _splits(), replacement_fraction=0.25, minimal_seed=17, lexical_seed=29
    )

    assert set(arms) == {"baseline", "minimal_lexical_contrast"}
    assert all(arm["requires_training"] for arm in arms.values())
    assert len(arms["baseline"]["splits"].train) == len(arms["minimal_lexical_contrast"]["splits"].train)
    assert any(example.template_id >= 1_000 for example in arms["minimal_lexical_contrast"]["splits"].train)
    assert any(example.template_id >= 2_000 for example in arms["minimal_lexical_contrast"]["splits"].train)


def test_phase26_sealed_suite_is_disjoint_from_training_and_development_text():
    from eval.form_variation import (
        build_phase26_confirmation_arms,
        build_phase26_sealed_pressure_test,
        build_phase25_split_manifest,
    )

    splits = _splits()
    arms = build_phase26_confirmation_arms(
        splits, replacement_fraction=0.25, minimal_seed=17, lexical_seed=29
    )
    development = build_fixed_pressure_test([(101, 44), (207, 86)])
    sealed = build_phase26_sealed_pressure_test([(809, 431), (907, 488)])
    manifest = build_phase25_split_manifest(
        arms["minimal_lexical_contrast"]["splits"].train, development, sealed_groups=sealed
    )

    assert manifest["validation"] == "PASS"
    assert manifest["sealed_example_count"] > 0


def test_phase26_confirmation_requires_the_registered_gate_at_each_breadth():
    from eval.form_variation import analyze_phase26_confirmation

    tracks = (
        "held_out_templates",
        "lexical_shift",
        "syntax_order_reversal",
        "discourse_distractor",
        "minimal_contrast",
    )

    def run(seed: int, score: float) -> dict:
        return {"seed": seed, "worst_robust_accuracy": score, "pressure_groups": {track: score for track in tracks}}

    seeds = (71, 73, 79, 83, 89)
    report = {
        "runs": {
            "4": {"baseline": [run(seed, 0.2) for seed in seeds], "minimal_lexical_contrast": [run(seed, 0.4) for seed in seeds]},
            "8": {"baseline": [run(seed, 0.2) for seed in seeds], "minimal_lexical_contrast": [run(seed, 0.35) for seed in seeds]},
        }
    }

    analysis = analyze_phase26_confirmation(report)

    assert analysis["by_variants"]["4"]["passes_confirmation_gate"] is True
    assert analysis["by_variants"]["8"]["passes_confirmation_gate"] is True
    assert analysis["confirmation_passed"] is True


def test_phase26_confirmation_schedule_matches_exposure_at_each_breadth():
    from eval.form_variation import _phase26_confirmation_schedule

    schedule = _phase26_confirmation_schedule(dataset_size=288, batch_size=16, exposure=88.888889)

    assert schedule == {"batch_size": 16, "max_steps": 1600, "warmup_steps": 160}


def test_operation_confusion_matrix_keeps_exact_case_counts():
    from eval.form_variation import FormVariationExample, operation_confusion_matrix

    examples = (
        FormVariationExample("a", "OP=ADD", "ADD", 1, 1, 2),
        FormVariationExample("b", "OP=SUBTRACT", "SUBTRACT", 2, 3, 1),
    )

    matrix = operation_confusion_matrix(examples, ("OP=SUBTRACT", "OP=SUBTRACT"))

    assert matrix["ADD"]["SUBTRACT"] == 1
    assert matrix["SUBTRACT"]["SUBTRACT"] == 1
