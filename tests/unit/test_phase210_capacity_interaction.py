"""Phase 2.10 specifications for the width-by-augmentation interaction."""

from __future__ import annotations


def _run(seed: int, score: float):
    tracks = (
        "held_out_templates",
        "lexical_shift",
        "syntax_order_reversal",
        "discourse_distractor",
        "minimal_contrast",
    )
    return {"seed": seed, "worst_robust_accuracy": score, "pressure_groups": {t: score for t in tracks}}


def test_phase210_analysis_reports_capacity_interaction_and_wide_gate():
    from eval.form_variation import analyze_phase210_screen

    seeds = (1, 2, 3, 4, 5)
    report = {"arms": {
        "narrow_baseline": [_run(seed, 0.2) for seed in seeds],
        "narrow_augmentation": [_run(seed, 0.3) for seed in seeds],
        "wide_baseline": [_run(seed, 0.4) for seed in seeds],
        "wide_augmentation": [_run(seed, 0.7) for seed in seeds],
    }}

    analysis = analyze_phase210_screen(report)

    assert analysis["selected_arm"] == "wide_augmentation"
    assert analysis["capacity_effects"]["worst_robust_accuracy"]["interaction"]["mean"] == 0.2
    assert analysis["wide_gate"]["arms"]["wide_augmentation"]["passes_continuation_gate"] is True


def test_phase210_model_override_changes_only_registered_capacity_fields():
    from eval.form_variation import phase210_wide_model_config

    model = {
        "hidden_size": 96,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "intermediate_size": 384,
        "max_position_embeddings": 128,
    }

    wide = phase210_wide_model_config(model, hidden_size=144, intermediate_size=576)

    assert wide["hidden_size"] == 144
    assert wide["intermediate_size"] == 576
    assert wide["num_hidden_layers"] == model["num_hidden_layers"]
    assert wide["num_attention_heads"] == model["num_attention_heads"]
    assert model["hidden_size"] == 96
