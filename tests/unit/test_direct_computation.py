"""Unit tests for Suite H Direct Computation and String Manipulation (Arithmetic & Strawberry-style queries)."""

import random
import pytest
from synth.ontology import World
from synth.tasks.generator import TaskGenerator
from synth.trajectories.generator import TrajectoryGenerator
from agent.tools.exec import RestrictedASTEvaluator


def test_suite_h_task_generation():
    world = World(world_id="W_TEST_H", seed=42)
    task_gen = TaskGenerator()
    rng = random.Random(42)

    tasks = task_gen.generate_suite_h_tasks(world, count=8, rng=rng)
    assert len(tasks) == 8

    # Verify task properties
    for t in tasks:
        assert t.suite == "suite_h_direct_computation"
        assert t.task_type == "direct_computation"
        assert not t.is_retrieval_required
        assert not t.is_contingent
        assert t.gold_answer is not None


def test_suite_h_trajectory_and_exec_consistency():
    world = World(world_id="W_TEST_H", seed=42)
    task_gen = TaskGenerator()
    traj_gen = TrajectoryGenerator(include_plan=True)
    evaluator = RestrictedASTEvaluator()
    rng = random.Random(42)

    tasks = task_gen.generate_suite_h_tasks(world, count=8, rng=rng)

    for task in tasks:
        traj = traj_gen.generate_trajectory_for_task(world, task, rng)
        turns = traj["turns"]

        # Turn 0: User
        assert turns[0]["role"] == "user"
        # Turn 1: Action MATH
        assert turns[1]["role"] == "action"
        assert turns[1]["content"].startswith("MATH ")

        math_code = turns[1]["content"].replace("MATH ", "").strip()
        exec_res = evaluator.evaluate(math_code)
        assert exec_res["status"] == "success", f"Execution failed for code: {math_code}"
        assert str(exec_res["result"]) == str(task.gold_answer)

        # Turn 2: Observation
        assert turns[2]["role"] == "observation"
        assert f"OBS MATH {task.gold_answer}" in turns[2]["content"]

        # Turn 3: Final EMIT
        assert turns[3]["role"] == "final"
        assert f'EMIT "{task.gold_answer}" EVIDENCE []' == turns[3]["content"]


def test_strawberry_direct_query_execution():
    evaluator = RestrictedASTEvaluator()
    query_code = '"Strawberry".lower().count("r")'
    res = evaluator.evaluate(query_code)
    assert res["status"] == "success"
    assert res["result"] == 3


def test_large_arithmetic_direct_query_execution():
    evaluator = RestrictedASTEvaluator()
    res = evaluator.evaluate('347 + 687')
    assert res["status"] == "success"
    assert res["result"] == 1034
