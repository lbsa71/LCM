import pytest

from eval.form_variation import analyze_scaling_reports


_TRACKS = (
    "held_out_templates",
    "lexical_shift",
    "syntax_order_reversal",
    "discourse_distractor",
    "minimal_contrast",
)


def _run(seed: int, score: float) -> dict:
    return {
        "seed": seed,
        "worst_robust_accuracy": score,
        "pressure_groups": {track: score for track in _TRACKS},
    }


def _report(name: str, exposure: str, value: float, scores: tuple[float, float]) -> dict:
    return {
        "experiment": name,
        "tokenizer_policy": "fixed_byte_vocabulary",
        "conditions": [1, 8],
        "exposures": {exposure: value},
        "seeds": [17, 29],
        "robust_pressure_tracks": list(_TRACKS),
        "split_manifests": {
            "1": {"training_corpus_sha256": "k1", "validation": "PASS"},
            "8": {"training_corpus_sha256": "k8", "validation": "PASS"},
        },
        "runs": {
            "1": {exposure: [_run(17, scores[0]), _run(29, scores[0])]},
            "8": {exposure: [_run(17, scores[1]), _run(29, scores[1])]},
        },
    }


def test_analyze_scaling_reports_merges_stages_and_normalizes_slopes():
    baseline = _report("baseline", "R178", 177.777778, (0.2, 0.3))
    stage_a = _report("stage_a", "R356", 355.555556, (0.3, 0.5))
    stage_b = _report("stage_b", "R889", 888.888889, (0.4, 0.6))

    analysis = analyze_scaling_reports((baseline, stage_a, stage_b))

    assert analysis["sources"] == ["baseline", "stage_a", "stage_b"]
    assert analysis["variants"] == ["1", "8"]
    assert analysis["exposures"] == ["R178", "R356", "R889"]
    assert analysis["reinforcement_slopes"]["macro_robust_accuracy"]["8"]["R178_to_R356"]["mean"] == 0.2
    assert analysis["reinforcement_slopes"]["macro_robust_accuracy"]["8"]["R356_to_R889"]["mean"] == pytest.approx(
        0.1 / 1.321928, abs=1e-4
    )


def test_analyze_scaling_reports_rejects_incompatible_or_duplicate_stages():
    baseline = _report("baseline", "R178", 177.777778, (0.2, 0.3))
    wrong_seeds = _report("wrong_seeds", "R356", 355.555556, (0.3, 0.5))
    wrong_seeds["seeds"] = [17, 41]
    with pytest.raises(ValueError, match="seeds"):
        analyze_scaling_reports((baseline, wrong_seeds))

    duplicate = _report("duplicate", "R178", 177.777778, (0.4, 0.6))
    with pytest.raises(ValueError, match="Duplicate exposure"):
        analyze_scaling_reports((baseline, duplicate))


def test_analyze_scaling_reports_rejects_split_manifest_drift():
    baseline = _report("baseline", "R178", 177.777778, (0.2, 0.3))
    drifted = _report("drifted", "R356", 355.555556, (0.3, 0.5))
    drifted["split_manifests"]["8"]["training_corpus_sha256"] = "different"

    with pytest.raises(ValueError, match="split manifest"):
        analyze_scaling_reports((baseline, drifted))
