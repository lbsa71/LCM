"""Unit tests for oracle baseline and epistemic scoring enforcement."""

from synth.world import WorldGenerator
from synth.documents.generator import DocumentGenerator
from synth.tasks.generator import TaskGenerator
from eval.oracle import OracleSolver
from eval.metrics import evaluate_episode_outcome


def test_oracle_100_percent():
    world_gen = WorldGenerator(base_seed=42)
    world = world_gen.generate_world("w1", seed=100)
    doc_gen = DocumentGenerator()
    doc_gen.generate_documents(world)
    task_gen = TaskGenerator()
    tasks = task_gen.generate_all_tasks(world, rng=world_gen.lexicon.rng)

    oracle = OracleSolver()
    for task in tasks:
        ep = oracle.solve(world, task)
        task_dict = {
            "task_id": task.task_id,
            "suite": task.suite,
            "gold_answer": task.gold_answer,
            "is_retrieval_required": task.is_retrieval_required,
            "is_insufficient_evidence": task.is_insufficient_evidence,
            "required_evidence": task.proof_graph.required_document_lines
        }
        out = evaluate_episode_outcome(ep, task_dict)
        assert out["grounded_success"] is True


def test_epistemic_enforcement_ungrounded_guess_fails():
    task_dict = {
        "task_id": "test_1",
        "suite": "suite_c_single_hop",
        "gold_answer": "482",
        "is_retrieval_required": True,
        "is_insufficient_evidence": False,
        "required_evidence": {"D01": [2]}
    }

    # Episode with lucky guess but NO cited evidence
    ep_no_evidence = {
        "model_answer": "482",
        "cited_evidence": [],
        "read_count": 0,
        "turns_used": 1
    }
    out = evaluate_episode_outcome(ep_no_evidence, task_dict)
    assert out["raw_match"] is True
    assert out["evidence_valid"] is False
    assert out["grounded_success"] is False
    assert out["failure_category"] == "UNSUPPORTED_CLAIM"
