"""Unit tests for corpus linter."""

from synth.lint import CorpusLinter


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
