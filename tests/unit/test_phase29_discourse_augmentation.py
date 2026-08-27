"""Phase 2.9 specifications for replacement-versus-augmentation screening."""

from __future__ import annotations

from eval.form_variation import OPERATIONS, build_condition_splits


def _splits():
    return build_condition_splits(
        variants_per_operation=8,
        train_pairs=[(11, 7), (13, 8), (17, 9), (19, 12)],
        eval_pairs=[(59, 31)],
        seed=42,
    )


def test_discourse_augmentation_retains_every_standard_example_and_adds_25_percent():
    from eval.form_variation import build_counterfactual_discourse_augmentation

    splits = _splits()
    augmented = build_counterfactual_discourse_augmentation(
        splits.train, augmentation_fraction=0.25, seed=37
    )

    assert set(splits.train).issubset(set(augmented))
    assert len(augmented) == 120
    assert sum(3_000 <= x.template_id < 4_000 for x in augmented) == 24
    for operation in OPERATIONS:
        assert sum(x.operation == operation for x in augmented) == 40


def test_phase29_reuses_two_cells_and_registers_ten_new_runs():
    from eval.form_variation import build_phase29_augmentation_arms

    arms = build_phase29_augmentation_arms(_splits(), augmentation_fraction=0.25, discourse_seed=37)

    assert set(arms) == {
        "baseline",
        "counterfactual_replacement",
        "augmentation_fixed_updates",
        "augmentation_matched_exposure",
    }
    assert arms["baseline"]["requires_training"] is False
    assert arms["counterfactual_replacement"]["requires_training"] is False
    assert arms["augmentation_fixed_updates"]["requires_training"] is True
    assert arms["augmentation_matched_exposure"]["requires_training"] is True


def test_phase29_analysis_reports_compute_effect_and_selects_only_new_cells():
    from eval.form_variation import analyze_phase29_screen

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
    report = {"arms": {
        "baseline": [run(seed, 0.2) for seed in seeds],
        "counterfactual_replacement": [run(seed, 0.3) for seed in seeds],
        "augmentation_fixed_updates": [run(seed, 0.5) for seed in seeds],
        "augmentation_matched_exposure": [run(seed, 0.7) for seed in seeds],
    }}

    analysis = analyze_phase29_screen(report)

    assert analysis["selected_arm"] == "augmentation_matched_exposure"
    assert analysis["compute_effect"]["worst_robust_accuracy"]["mean"] == 0.2
