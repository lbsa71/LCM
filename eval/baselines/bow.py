"""Bag-of-words / text heuristic baseline without retrieval access."""

import re
from typing import Any, Dict
from synth.ontology import World, Task


class BagOfWordsBaseline:
    """Extracts numbers or entity names directly from question text without environment access."""

    def solve(self, world: World, task: Task) -> Dict[str, Any]:
        # If numbers are in prompt, take the first one
        numbers = re.findall(r"\b\d+\b", task.question)
        if numbers:
            ans = numbers[0]
        elif "yes" in task.question.lower():
            ans = "yes"
        elif "no" in task.question.lower():
            ans = "no"
        else:
            ans = "unknown"

        return {
            "task_id": task.task_id,
            "world_id": world.world_id,
            "gold_answer": task.gold_answer,
            "model_answer": ans,
            "cited_evidence": [],
            "turns_used": 1,
            "search_count": 0,
            "read_count": 0,
            "exec_count": 0,
            "is_terminated": True,
            "termination_reason": "BOW_HEURISTIC"
        }
