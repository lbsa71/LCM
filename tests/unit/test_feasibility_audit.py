"""Scientific summaries distinguish memorized prompts and partial evidence."""

import json

import pytest

from eval.feasibility_audit import audit_corpus, paired_comparison, strict_grounded_success


def test_corpus_audit_reports_unfinished_training_and_seen_questions(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [
        {"task_id": "task_a_train_1", "world_id": "train", "turns": [
            {"role": "user", "content": "Seen question"},
            {"role": "final", "content": 'EMIT "true" EVIDENCE []'}]},
        {"task_id": "task_f_train_1", "world_id": "train", "turns": [
            {"role": "user", "content": "Missing?"},
            {"role": "action", "content": 'SEARCH "missing" LIMIT 3'}]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    tasks = [{"task_id": "t", "world_id": "test", "question": "Seen question",
              "suite": "language", "metadata": {}, "required_evidence": {}}]
    report = audit_corpus(path, tasks, {"test": {"documents": {}}})
    assert report["training_trajectories"] == 2
    assert report["nonterminal_trajectories"] == 1
    assert report["train_eval_world_overlap"] == []
    assert report["suites"]["language"]["question_seen_in_training"] == 1
    assert report["trajectory_families"]["task_f_"]["nonterminal"] == 1


def test_strict_grounding_requires_all_proof_lines_and_observed_evidence():
    task = {"gold_answer": "3", "is_retrieval_required": True,
            "required_evidence": {"D": [1, 2]}}
    episode = {"model_answer": "3", "cited_evidence": [{"document_id": "D", "lines": [1, 1]}],
               "trace_steps": [{"observation": {"status": "success", "document_id": "D",
                                  "text": "OBS READ D LINES 1-2\nD:L1 Value 1\nD:L2 Value 2"}}]}
    assert strict_grounded_success(task, episode) is False
    episode["cited_evidence"][0]["lines"] = [1, 2]
    assert strict_grounded_success(task, episode) is True
    episode["trace_steps"] = []
    assert strict_grounded_success(task, episode) is False


def test_paired_comparison_validates_alignment_and_returns_cluster_uncertainty():
    before = [{"task_id": str(i), "world_id": str(i // 2), "success": False} for i in range(8)]
    after = [{**item, "success": True} for item in before]
    report = paired_comparison(before, after)
    assert report["gain_pp"] == 100.0
    assert report["world_cluster_ci95_pp"] == [100.0, 100.0]
    assert report["independent_training_replicates"] == 1
    with pytest.raises(ValueError, match="aligned"):
        paired_comparison(before, list(reversed(after)))
