"""Unit tests for procedural induction trajectories and atomic role transitions."""

import json
import random
import pytest

from synth.ontology import World, Task, Document, DocumentLine, Fact, Entity
from synth.trajectories.generator import TrajectoryGenerator, extract_entity_name_from_question


def create_test_world_and_task():
    world = World(world_id="W_TEST_01", seed=42)
    e1 = Entity(id="E01", name="Corvath", entity_type="settlement")
    world.entities["E01"] = e1
    
    doc = Document(
        id="D01",
        title="Settlement Record",
        doc_type="census",
        lines=[
            DocumentLine(line_number=1, text="Corvath is a settlement.", fact_ids=["F01"]),
            DocumentLine(line_number=2, text="The recorded population of Corvath is 742.", fact_ids=["F02"])
        ]
    )
    world.documents["D01"] = doc
    
    task = Task(
        task_id="T_TEST_01",
        task_type="single_hop_retrieval",
        suite="suite_c_single_hop",
        question="What is the recorded population of Corvath?",
        gold_answer="742",
        proof_graph=None,
        world_id="W_TEST_01",
        is_retrieval_required=True,
        is_contingent=True
    )
    return world, task


def test_entity_extraction():
    world, task = create_test_world_and_task()
    ent = extract_entity_name_from_question("What is the recorded population of Corvath?", world)
    assert ent == "Corvath"


def test_atomic_trajectory_generation():
    world, task = create_test_world_and_task()
    gen = TrajectoryGenerator()
    rng = random.Random(42)
    traj = gen.generate_trajectory_for_task(world, task, rng)
    
    turns = traj["turns"]
    roles = [t["role"] for t in turns]
    
    # In atomic protocol: user -> action (search) -> observation -> action (read) -> observation -> final
    # NO redundant 'plan' role that causes turn-streaming collisions
    assert "plan" not in roles
    assert roles[0] == "user"
    assert roles[1] == "action"
    assert roles[2] == "observation"
    
    # Check that search action strictly extracted the entity in RDL syntax
    assert turns[1]["content"] == 'SEARCH "Corvath" LIMIT 3'
    assert "OBS SEARCH" in turns[2]["content"] and "D01" in turns[2]["content"]
    assert "READ D01" in turns[3]["content"]
    assert "OBS READ D01" in turns[4]["content"]
    assert "EMIT" in turns[5]["content"]
    assert "EVIDENCE" in turns[5]["content"]
