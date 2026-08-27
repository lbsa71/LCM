"""Regression tests for defects that invalidate architecture-level comparisons."""

import json
import random
import sys

import pytest
import yaml

from agent.adapters.document import DocumentEvidenceProvider
from agent.protocol import format_search_hop
from synth import generate
from synth.ontology import Document, DocumentLine, Entity, Fact, ProofGraph, Task, World
from synth.tasks.generator import TaskGenerator
from synth.trajectories.generator import TrajectoryGenerator


def test_generation_serializes_documents_added_by_evaluation_tasks(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "name": "validity_test", "seed": 42,
        "corpus": {"train_worlds": 1, "val_worlds": 1, "test_worlds": 1},
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["generate", "--config", str(config_path),
                                     "--output_dir", str(tmp_path / "run")])
    generate.main()
    data = tmp_path / "run/data"
    worlds = json.loads((data / "eval_worlds.json").read_text(encoding="utf-8"))
    tasks = json.loads((data / "eval_tasks.json").read_text(encoding="utf-8"))
    counterfactuals = [task for task in tasks if task["suite"] == "suite_i_counterfactual_inversion"]
    assert counterfactuals
    for task in tasks:
        for doc, lines in task["required_evidence"].items():
            assert doc in worlds[task["world_id"]]["documents"]
            actual = {line["line_number"] for line in worlds[task["world_id"]]["documents"][doc]["lines"]}
            assert set(lines) <= actual


@pytest.mark.parametrize("include_plan", [False, True])
def test_missing_evidence_demonstration_ends_with_live_observation_and_abstention(include_plan):
    world = World(world_id="missing", seed=1)
    task = TaskGenerator().generate_suite_f_tasks(world, 1, random.Random(1))[0]
    turns = TrajectoryGenerator(include_plan=include_plan).generate_trajectory_for_task(
        world, task, random.Random(1))["turns"]
    assert turns[-1] == {"role": "final", "content": 'EMIT "insufficient_evidence" EVIDENCE []', "train": True}
    assert turns[-2] == {"role": "observation", "content": format_search_hop([]), "train": False}


def test_disabled_evidence_is_withheld_without_mutating_shared_world():
    from synth.evidence import world_for_task

    world = World(world_id="world", seed=1, documents={
        "D": Document(id="D", title="Record", doc_type="record", lines=[
            DocumentLine(line_number=1, text="Zorp population is 34.")])})
    task = Task(task_id="disabled", world_id="world", task_type="evidence_disabled",
                suite="anti_memorization_evidence_disabled", question="Zorp population?",
                gold_answer="insufficient_evidence", proof_graph=ProofGraph(goal="abstain"),
                is_insufficient_evidence=True, metadata={"withhold_evidence": True})
    view = world_for_task(world, task)
    assert DocumentEvidenceProvider(view).search("Zorp") == []
    assert DocumentEvidenceProvider(view).read("D") is None
    assert "D" in world.documents
    turns = TrajectoryGenerator().generate_trajectory_for_task(world, task, random.Random(1))["turns"]
    assert turns[-2]["content"] == format_search_hop([])
    assert turns[-1]["role"] == "final"


def _comparison_world(tied=False):
    world = World(world_id="compare", seed=1)
    world.entities = {
        "R": Entity(id="R", name="Region", entity_type="region"),
        "A": Entity(id="A", name="Zorp", entity_type="settlement"),
        "B": Entity(id="B", name="Bink", entity_type="settlement"),
    }
    for index, (subject, population) in enumerate([("A", 40), ("B", 40 if tied else 20)]):
        inside = Fact(id=f"inside{index}", subject_id=subject, relation="inside", value="R")
        pop = Fact(id=f"pop{index}", subject_id=subject, relation="population", value=population)
        world.facts.update({inside.id: inside, pop.id: pop})
        world.documents[subject] = Document(id=subject, title=subject, doc_type="record", lines=[
            DocumentLine(line_number=1, text=f"{subject} inside Region", fact_ids=[inside.id]),
            DocumentLine(line_number=2, text=f"{subject} population {population}", fact_ids=[pop.id]),
        ])
    return world


def test_multihop_uses_canonical_inside_facts_and_proves_both_memberships():
    tasks = TaskGenerator().generate_suite_d_tasks(_comparison_world(), 2, random.Random(1))
    assert len(tasks) == 1
    assert tasks[0].gold_answer == "Zorp"
    assert tasks[0].proof_graph.required_document_lines == {"A": [1, 2], "B": [1, 2]}


def test_multihop_does_not_label_a_tied_pair_as_higher():
    assert TaskGenerator().generate_suite_d_tasks(_comparison_world(tied=True), 2, random.Random(1)) == []


@pytest.mark.parametrize("suite", ["c", "d", "g", "h"])
def test_oracle_label_never_enters_intermediate_plans_or_tool_observations(suite):
    world = _comparison_world()
    task = getattr(TaskGenerator(), f"generate_suite_{suite}_tasks")(world, 1, random.Random(1))[0]
    task.gold_answer = "ORACLE_ONLY_CANARY"
    turns = TrajectoryGenerator(include_plan=True).generate_trajectory_for_task(world, task, random.Random(1))["turns"]
    assert all("ORACLE_ONLY_CANARY" not in turn["content"] for turn in turns if turn["role"] != "final")


def test_direct_computation_refuses_missing_expression_instead_of_using_gold():
    world = _comparison_world()
    task = TaskGenerator().generate_suite_h_tasks(world, 1, random.Random(1))[0]
    task.proof_graph.goal = ""
    with pytest.raises(ValueError, match="expression"):
        TrajectoryGenerator().generate_trajectory_for_task(world, task, random.Random(1))
