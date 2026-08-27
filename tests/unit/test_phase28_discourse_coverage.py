"""Phase 2.8 specifications for counterfactual discourse coverage."""

from __future__ import annotations

from collections import Counter

from eval.form_variation import OPERATIONS, build_condition_splits, build_fixed_pressure_test


def _splits():
    return build_condition_splits(
        variants_per_operation=8,
        train_pairs=[(11, 7), (13, 8), (17, 9), (19, 12)],
        eval_pairs=[(59, 31)],
        seed=42,
    )


def test_counterfactual_discourse_replacement_is_deterministic_and_budget_preserving():
    from eval.form_variation import build_counterfactual_discourse_examples

    splits = _splits()
    first = build_counterfactual_discourse_examples(splits.train, replacement_fraction=0.25, seed=37)
    second = build_counterfactual_discourse_examples(splits.train, replacement_fraction=0.25, seed=37)

    assert first == second
    assert len(first) == len(splits.train)
    assert {(x.left, x.right) for x in first} == {(x.left, x.right) for x in splits.train}
    for operation in OPERATIONS:
        assert sum(x.operation == operation for x in first) == sum(x.operation == operation for x in splits.train)
    assert sum(3_000 <= x.template_id < 4_000 for x in first) == 24


def test_discourse_cues_are_equally_represented_for_every_operation():
    from eval.form_variation import build_counterfactual_discourse_examples, discourse_cue_id

    examples = build_counterfactual_discourse_examples(_splits().train, replacement_fraction=0.25, seed=37)
    counts = {
        operation: Counter(
            discourse_cue_id(example.template_id)
            for example in examples
            if example.operation == operation and 3_000 <= example.template_id < 4_000
        )
        for operation in OPERATIONS
    }

    assert counts["ADD"] == counts["SUBTRACT"] == counts["COMPARE"]
    assert len(counts["ADD"]) >= 4


def test_phase28_arms_train_fresh_baseline_and_discourse_condition_only():
    from eval.form_variation import build_phase28_discourse_arms

    arms = build_phase28_discourse_arms(_splits(), replacement_fraction=0.25, discourse_seed=37)

    assert set(arms) == {"baseline", "counterfactual_discourse"}
    assert all(arm["requires_training"] for arm in arms.values())
    assert {arm["target_mode"] for arm in arms.values()} == {"operation"}
    assert {len(arm["splits"].train) for arm in arms.values()} == {len(_splits().train)}


def test_phase28_training_text_is_disjoint_from_development_pressure_suite():
    from eval.form_variation import build_phase25_split_manifest, build_phase28_discourse_arms

    arms = build_phase28_discourse_arms(_splits(), replacement_fraction=0.25, discourse_seed=37)
    development = build_fixed_pressure_test([(101, 44), (207, 86)])
    manifest = build_phase25_split_manifest(arms["counterfactual_discourse"]["splits"].train, development)

    assert manifest["validation"] == "PASS"
    assert manifest["train_development_exact_overlap_count"] == 0


def test_phase28_gate_selects_only_a_passing_discourse_arm():
    from eval.form_variation import analyze_phase28_screen

    tracks = (
        "held_out_templates",
        "lexical_shift",
        "syntax_order_reversal",
        "discourse_distractor",
        "minimal_contrast",
    )

    def run(seed: int, score: float):
        return {"seed": seed, "worst_robust_accuracy": score, "pressure_groups": {t: score for t in tracks}}

    seeds = (1, 2, 3, 4, 5)
    report = {
        "arms": {
            "baseline": [run(seed, 0.2) for seed in seeds],
            "counterfactual_discourse": [run(seed, 0.5) for seed in seeds],
        }
    }

    analysis = analyze_phase28_screen(report)

    assert analysis["selected_arm"] == "counterfactual_discourse"
    assert analysis["arms"]["counterfactual_discourse"]["passes_continuation_gate"] is True
