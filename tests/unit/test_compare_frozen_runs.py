"""Architecture comparisons require complete matched task ledgers."""

import json

import pytest

from eval.compare_frozen_runs import compare_runs


def test_comparison_rejects_incomplete_ledger_and_reports_complete_pairs(tmp_path):
    tasks = [{"task_id": "one", "world_id": "w", "suite": "unit", "question": "True?",
              "gold_answer": "true", "is_retrieval_required": False}]
    for name, answer in [("before", "false"), ("after", "true")]:
        root = tmp_path / name
        (root / "step_3000").mkdir(parents=True)
        (root / "selected_tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
        (root / "step_3000/manifest.json").write_text(json.dumps({
            "tasks_sha256": "same", "input_hashes": {"worlds": "same"}, "runtime": {"profile": "legacy"},
        }), encoding="utf-8")
        (root / "step_3000/case_predictions.jsonl").write_text(json.dumps({"episode": {
            "task_id": "one", "world_id": "w", "model_answer": answer,
            "cited_evidence": [], "trace_steps": [], "elapsed_seconds": 0.5,
        }}), encoding="utf-8")
    report = compare_runs(tmp_path / "before", tmp_path / "after", 3000)
    assert report["legacy_grounding"]["gain_pp"] == 100.0
    assert report["strict_grounding"]["gain_pp"] == 100.0
    (tmp_path / "after/step_3000/case_predictions.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        compare_runs(tmp_path / "before", tmp_path / "after", 3000)
