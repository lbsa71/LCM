"""Unit tests for corpus linter."""

from synth.lint import CorpusLinter
from synth.generate import lintable_test_questions, raise_for_lint_errors


def test_forbidden_entity_detection():
    linter = CorpusLinter()
    # Test forbidden detection
    hits = linter.check_forbidden_terms("The capital of France is Paris.")
    assert "france" in hits or "paris" in hits

    clean_hits = linter.check_forbidden_terms("The settlement veska is inside torin.")
    assert len(clean_hits) == 0


def test_lint_dataset_clean():
    linter = CorpusLinter()
    train_texts = ["The settlement noru has population 400.", "The device veska is active."]
    val_texts = ["The settlement jora has population 500."]
    test_texts = ["What is the population of noru?"]
    train_seeds = {101, 102}
    test_seeds = {201, 202}

    res = linter.lint_dataset(train_texts, val_texts, test_texts, train_seeds, test_seeds)
    assert res["status"] == "PASS"
    assert len(res["errors"]) == 0


def test_lintable_test_questions_excludes_intentional_real_world_probes():
    """Counterfactual and closed-book probes are eval-only, not corpus contamination."""
    tasks = [
        {"suite": "suite_c_single_hop", "question": "What is the population of veska?"},
        {"suite": "suite_i_counterfactual_inversion", "question": "What is the capital of France?"},
        {"suite": "anti_memorization_closed_book", "question": "Who wrote Hamlet?"},
    ]

    assert lintable_test_questions(tasks) == ["What is the population of veska?"]


def test_lint_failure_blocks_training_pipeline():
    report = {"status": "FAIL", "errors": ["Forbidden term ['france'] found in train sample"]}

    try:
        raise_for_lint_errors(report)
    except ValueError as exc:
        assert "Corpus lint failed" in str(exc)
    else:
        raise AssertionError("A failed corpus lint must block downstream training")
