"""Deterministic Oracle Solver (PRD Section 33)."""

from typing import Any, Dict
from synth.ontology import World, Task


class OracleSolver:
    """Deterministic oracle agent achieving 100% grounded accuracy."""

    def solve(self, world: World, task: Task) -> Dict[str, Any]:
        """Solves the task deterministically with exact ground truth proof graph evidence."""
        evidence_list = []
        req_lines = {}
        if hasattr(task, "proof_graph") and task.proof_graph:
            req_lines = task.proof_graph.required_document_lines
        elif hasattr(task, "required_evidence"):
            req_lines = task.required_evidence

        for doc_id, lines in req_lines.items():
            evidence_list.append({
                "document_id": doc_id,
                "lines": lines
            })

        return {
            "task_id": task.task_id,
            "world_id": world.world_id,
            "gold_answer": task.gold_answer,
            "model_answer": task.gold_answer,
            "cited_evidence": evidence_list,
            "turns_used": 3 if task.is_retrieval_required else 1,
            "search_count": 1 if task.is_retrieval_required else 0,
            "read_count": len(evidence_list),
            "exec_count": 1 if task.suite == "suite_e_retrieval_computation" else 0,
            "is_terminated": True,
            "termination_reason": "SUCCESS",
            "is_oracle": True
        }
