"""Majority class baseline (PRD Section 32)."""

from typing import Any, Dict
from synth.ontology import World, Task


class MajorityBaseline:
    """Predicts the most frequent global answer (e.g. 'yes' or 'active')."""

    def __init__(self, default_answer: str = "yes"):
        self.default_answer = default_answer

    def solve(self, world: World, task: Task) -> Dict[str, Any]:
        return {
            "task_id": task.task_id,
            "world_id": world.world_id,
            "gold_answer": task.gold_answer,
            "model_answer": self.default_answer,
            "cited_evidence": [],
            "turns_used": 1,
            "search_count": 0,
            "read_count": 0,
            "exec_count": 0,
            "is_terminated": True,
            "termination_reason": "MAJORITY_PREDICTION"
        }
