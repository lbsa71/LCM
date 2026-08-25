"""Unit tests for Suite I Active Counterfactual Inversion Benchmark."""

import random
import pytest

from synth.ontology import World, Task, Document, DocumentLine, Entity, Fact, ProofGraph
from synth.tasks.generator import TaskGenerator
from synth.trajectories.generator import TrajectoryGenerator
from agent.tools.exec import RestrictedASTEvaluator


def create_counterfactual_test_world() -> World:
    """Creates a world with explicit counterfactual document evidence."""
    world = World(world_id="W_CF_01", seed=42)
    
    # Entity: France
    e1 = Entity(id="E_FRANCE", name="France", entity_type="country")
    world.entities["E_FRANCE"] = e1
    
    # Fact: Capital of France is Lyon (Counterfactual)
    f1 = Fact(id="F_CF_01", subject_id="E_FRANCE", relation="capital", value="Lyon", is_contingent=True)
    world.facts["F_CF_01"] = f1

    # Document asserting counterfactual truth
    doc = Document(
        id="D01",
        title="Territorial Registry of Europe",
        doc_type="gazetteer",
        lines=[
            DocumentLine(line_number=1, text="The administrative capital of France is Lyon.", fact_ids=["F_CF_01"]),
            DocumentLine(line_number=2, text="France has a recorded population of 68 million.", fact_ids=[])
        ]
    )
    world.documents["D01"] = doc
    return world


def test_suite_i_task_generation():
    world = create_counterfactual_test_world()
    gen = TaskGenerator()
    rng = random.Random(42)
    
    tasks = gen.generate_suite_i_tasks(world, count=3, rng=rng)
    assert len(tasks) > 0
    
    task = tasks[0]
    assert task.suite == "suite_i_counterfactual_inversion"
    assert "France" in task.question or "capital" in task.question or "boiling" in task.question or "Python" in task.question
    assert task.gold_answer in ["Lyon", "42", "Ada Lovelace", "8", "14"]


def test_suite_i_trajectory_generation():
    world = create_counterfactual_test_world()
    t_gen = TaskGenerator()
    traj_gen = TrajectoryGenerator()
    rng = random.Random(42)
    
    # Generate counterfactual task
    task = Task(
        task_id="task_cf_test_1",
        task_type="counterfactual_inversion",
        suite="suite_i_counterfactual_inversion",
        question="What is the capital of France according to territorial records?",
        gold_answer="Lyon",
        proof_graph=ProofGraph(goal="counterfactual_retrieval", required_document_lines={"D01": [1]}),
        world_id="W_CF_01",
        is_retrieval_required=True,
        is_contingent=True,
        metadata={"prior_answer": "Paris"}
    )
    
    traj = traj_gen.generate_trajectory_for_task(world, task, rng)
    turns = traj["turns"]
    
    # Verify procedural steps
    roles = [t["role"] for t in turns]
    assert roles[0] == "user"
    assert roles[1] == "action"
    assert "SEARCH" in turns[1]["content"] and "France" in turns[1]["content"]
    assert roles[2] == "observation"
    assert "OBS SEARCH" in turns[2]["content"] and "D01" in turns[2]["content"]
    assert roles[3] == "action"
    assert "READ D01" in turns[3]["content"]
    assert roles[4] == "observation"
    assert "OBS READ D01" in turns[4]["content"]
    assert roles[5] == "final"
    assert 'EMIT "Lyon" EVIDENCE [D01:1]' in turns[5]["content"]
