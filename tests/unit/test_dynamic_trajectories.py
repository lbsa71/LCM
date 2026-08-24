"""Unit tests for closed-loop dynamic trajectory generation with live environment execution."""

import random
import pytest

from synth.ontology import World, Task, Document, DocumentLine, Entity, ProofGraph
from synth.trajectories.generator import TrajectoryGenerator, extract_entity_name_from_question
from agent.adapters.document import DocumentEvidenceProvider


def create_dynamic_test_world_and_tasks():
    world = World(world_id="W_DYNAMIC_01", seed=101)
    
    # Entities
    e1 = Entity(id="E01", name="Ardentis", entity_type="settlement")
    e2 = Entity(id="E02", name="Belgard", entity_type="settlement")
    e3 = Entity(id="R01", name="SouthernReach", entity_type="region")
    world.entities["E01"] = e1
    world.entities["E02"] = e2
    world.entities["R01"] = e3
    
    # Documents
    d1 = Document(
        id="D01",
        title="Settlement Gazetteer Alpha",
        doc_type="gazetteer",
        lines=[
            DocumentLine(line_number=1, text="The primary settlement Ardentis is located in the region SouthernReach.", fact_ids=["F01"]),
            DocumentLine(line_number=2, text="The census population of Ardentis is 850.", fact_ids=["F02"]),
        ]
    )
    d2 = Document(
        id="D02",
        title="Settlement Gazetteer Beta",
        doc_type="gazetteer",
        lines=[
            DocumentLine(line_number=1, text="The settlement Belgard is positioned inside SouthernReach.", fact_ids=["F03"]),
            DocumentLine(line_number=2, text="The census population of Belgard is 420.", fact_ids=["F04"]),
        ]
    )
    d3 = Document(
        id="D03",
        title="Census Multi-Settlement Summary",
        doc_type="census",
        lines=[
            DocumentLine(line_number=1, text="The recorded headcount for Ardentis is currently 850.", fact_ids=["F05"]),
            DocumentLine(line_number=2, text="The recorded headcount for Belgard is currently 420.", fact_ids=["F06"]),
        ]
    )
    world.documents["D01"] = d1
    world.documents["D02"] = d2
    world.documents["D03"] = d3

    # Task 1: Single Hop
    pg_c = ProofGraph(goal="Find population of Ardentis", required_document_lines={"D01": [2]})
    task_c = Task(
        task_id="T_DYN_C",
        task_type="single_hop_retrieval",
        suite="suite_c_single_hop",
        question="What is the census population of Ardentis?",
        gold_answer="850",
        proof_graph=pg_c,
        world_id="W_DYNAMIC_01",
        is_retrieval_required=True,
        is_contingent=True
    )

    # Task 2: Multi-Hop (Find largest settlement in SouthernReach)
    pg_d = ProofGraph(goal="multi_hop_largest_in_region", required_document_lines={"D01": [1], "D02": [1], "D03": [1, 2]})
    task_d = Task(
        task_id="T_DYN_D",
        task_type="multi_hop_comparison",
        suite="suite_d_multi_hop",
        question="Which settlement located inside the region SouthernReach has the largest population?",
        gold_answer="Ardentis",
        proof_graph=pg_d,
        world_id="W_DYNAMIC_01",
        is_retrieval_required=True,
        is_contingent=True
    )

    # Task 3: Retrieval + Computation (Separate Docs)
    pg_e_diff = ProofGraph(goal="Compute combined population", required_document_lines={"D01": [2], "D02": [2]})
    task_e_diff = Task(
        task_id="T_DYN_E_DIFF",
        task_type="retrieval_computation",
        suite="suite_e_retrieval_computation",
        question="What is the combined total population of Ardentis and Belgard?",
        gold_answer="1270",
        proof_graph=pg_e_diff,
        world_id="W_DYNAMIC_01",
        is_retrieval_required=True,
        is_contingent=True
    )

    # Task 4: Retrieval + Computation (Same Doc D03)
    pg_e_same = ProofGraph(goal="Compute combined population same doc", required_document_lines={"D03": [1, 2]})
    task_e_same = Task(
        task_id="T_DYN_E_SAME",
        task_type="retrieval_computation",
        suite="suite_e_retrieval_computation",
        question="What is the combined total population of Ardentis and Belgard?",
        gold_answer="1270",
        proof_graph=pg_e_same,
        world_id="W_DYNAMIC_01",
        is_retrieval_required=True,
        is_contingent=True
    )

    return world, task_c, task_d, task_e_diff, task_e_same


def test_dynamic_single_hop_trajectory_with_plan():
    world, task_c, _, _, _ = create_dynamic_test_world_and_tasks()
    gen = TrajectoryGenerator(include_plan=True)
    rng = random.Random(42)

    traj = gen.generate_trajectory_for_task(world, task_c, rng)
    turns = traj["turns"]
    
    assert turns[0]["role"] == "user"
    assert turns[1]["role"] == "plan"
    assert "Ardentis" in turns[1]["content"]

    assert turns[2]["role"] == "action"
    assert 'SEARCH "Ardentis"' in turns[2]["content"]

    assert turns[3]["role"] == "observation"
    assert "OBS SEARCH" in turns[3]["content"]

    assert turns[4]["role"] == "plan"
    assert "D01" in turns[4]["content"]

    assert turns[5]["role"] == "action"
    assert "READ D01" in turns[5]["content"]

    assert turns[6]["role"] == "observation"
    assert "OBS READ D01" in turns[6]["content"]
    assert "850" in turns[6]["content"]

    assert turns[7]["role"] == "plan"
    assert "850" in turns[7]["content"]

    final_turn = turns[8]
    assert final_turn["role"] == "final"
    assert 'EMIT "850"' in final_turn["content"]
    assert "D01:2" in final_turn["content"]


def test_dynamic_retrieval_computation_separate_docs():
    world, _, _, task_e_diff, _ = create_dynamic_test_world_and_tasks()
    gen = TrajectoryGenerator(include_plan=True)
    rng = random.Random(42)

    traj = gen.generate_trajectory_for_task(world, task_e_diff, rng)
    turns = traj["turns"]

    math_turns = [t for t in turns if t["role"] == "action" and "MATH" in t["content"]]
    assert len(math_turns) == 1
    assert "850" in math_turns[0]["content"] and "420" in math_turns[0]["content"]

    final_turn = turns[-1]
    assert 'EMIT "1270"' in final_turn["content"]


def test_dynamic_retrieval_computation_same_doc():
    world, _, _, _, task_e_same = create_dynamic_test_world_and_tasks()
    gen = TrajectoryGenerator(include_plan=True)
    rng = random.Random(42)

    traj = gen.generate_trajectory_for_task(world, task_e_same, rng)
    turns = traj["turns"]

    # In same doc, val1 and val2 must be distinct (850 and 420)
    math_turns = [t for t in turns if t["role"] == "action" and "MATH" in t["content"]]
    assert len(math_turns) == 1
    assert "850" in math_turns[0]["content"] and "420" in math_turns[0]["content"]

    # MATH expression must evaluate exactly to the gold answer
    expr = math_turns[0]["content"].replace("MATH", "").strip()
    assert eval(expr) == 1270

    final_turn = turns[-1]
    assert 'EMIT "1270"' in final_turn["content"]


def test_dynamic_multi_hop_comparison():
    world, _, task_d, _, _ = create_dynamic_test_world_and_tasks()
    gen = TrajectoryGenerator(include_plan=True)
    rng = random.Random(42)

    traj = gen.generate_trajectory_for_task(world, task_d, rng)
    turns = traj["turns"]

    # Ensure search was for the region SouthernReach
    search_turns = [t for t in turns if t["role"] == "action" and "SEARCH" in t["content"]]
    assert len(search_turns) >= 1
    assert "SouthernReach" in search_turns[0]["content"]

    # Final emission
    final_turn = turns[-1]
    assert 'EMIT "Ardentis"' in final_turn["content"]
