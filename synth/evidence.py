"""Task-specific evidence views shared by synthesis and benchmark execution."""

from dataclasses import replace

from synth.ontology import Task, World


def world_for_task(world: World, task: Task) -> World:
    """Hide external evidence for declared controls without mutating other tasks.

    The full evidence-disabled control removes the document collection, rather
    than leaving a label asking the model to ignore accessible facts. Ordinary
    missing-evidence tasks keep their distractors and use the live search tool.
    """
    if task.metadata.get("withhold_evidence") or task.task_type == "closed_book_leakage":
        return replace(world, documents={}, facts={})
    return world
