"""Value swaps identify evidence dependence rather than memorized counterfacts."""

from eval.grounding_value_swap import build_probe
import pytest


def test_value_swap_changes_only_relevant_world_truth_and_keeps_question_identical():
    task = {"task_id": "t", "world_id": "w", "suite": "suite_i_counterfactual_inversion",
            "question": "What is the capital of France?", "gold_answer": "Lyon",
            "metadata": {"prior_answer": "Paris"}, "required_evidence": {"D": [1]}}
    world = {"world_id": "w", "seed": 1, "facts": {"F": {"value": "Lyon"}},
             "documents": {"D": {"id": "D", "lines": [
                 {"line_number": 1, "text": "The capital of France is Lyon.", "fact_ids": ["F"]},
                 {"line_number": 2, "text": "Unrelated information.", "fact_ids": []}]}}}
    tasks, worlds = build_probe([task], {"w": world})
    assert len(tasks) == len(worlds) == 2
    assert tasks[0]["question"] == tasks[1]["question"] == task["question"]
    assert len({task["gold_answer"], tasks[0]["gold_answer"], tasks[1]["gold_answer"]}) == 3
    for variant in tasks:
        value = variant["gold_answer"]
        changed = worlds[variant["world_id"]]
        assert value in changed["documents"]["D"]["lines"][0]["text"]
        assert changed["facts"]["F"]["value"] == value
        assert changed["documents"]["D"]["lines"][1] == world["documents"]["D"]["lines"][1]
        assert variant["metadata"]["source_task_id"] == "t"
        assert variant["metadata"]["training_counterfactual_answer"] == "Lyon"
    assert world["facts"]["F"]["value"] == "Lyon"
    assert build_probe([task], {"w": world}) == (tasks, worlds)


def test_probe_primary_score_requires_both_factual_variants_and_complete_ledger():
    from eval.grounding_value_swap import score_probe

    tasks, records = [], []
    for variant, answer in enumerate(["Velora", "Nareth"]):
        tasks.append({"task_id": str(variant), "world_id": str(variant), "gold_answer": answer,
                      "required_evidence": {"D": [1]}, "metadata": {
                          "source_task_id": "original", "variant": variant,
                          "training_counterfactual_answer": "Lyon", "prior_answer": "Paris"}})
        records.append({"episode": {"task_id": str(variant), "world_id": str(variant),
                                     "model_answer": answer,
                                     "cited_evidence": [{"document_id": "D", "lines": [1]}],
                                     "trace_steps": [{"observation": {"status": "success",
                                                       "document_id": "D", "text": f"D:L1 {answer}"}}]}})
    assert score_probe(tasks, records)["both_variants_strictly_correct_rate"] == 1.0
    records[1]["episode"]["model_answer"] = "Lyon"
    result = score_probe(tasks, records)
    assert result["strict_case_success_rate"] == 0.5
    assert result["both_variants_strictly_correct_rate"] == 0.0
    assert result["training_counterfact_reuse_rate"] == 0.5
    with pytest.raises(ValueError, match="complete"):
        score_probe(tasks, records[:1])
