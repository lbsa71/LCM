"""Replays must preserve task identity, runtime provenance, and partial evidence."""

import json

import pytest
import torch

from eval.frozen_checkpoint_audit import (
    audit_manifest,
    deserialize_task,
    evaluate_tasks,
    runtime_settings,
)
from synth.ontology import World
from synth.ontology import Document, DocumentLine


def _task(task_id="one"):
    return {
        "task_id": task_id, "world_id": "world", "suite": "unit",
        "task_type": "logic", "question": "A premise. Is it true?",
        "gold_answer": "true", "required_evidence": {"doc": [1, 2]},
        "is_retrieval_required": False, "is_contingent": False,
        "context_text": "A premise.", "metadata": {"prior_answer": "false"},
    }


class FakeShell:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def run_episode(self, world, task):
        self.calls.append(task.task_id)
        if task.task_id == self.fail_on:
            raise RuntimeError("interrupted")
        return {
            "task_id": task.task_id, "world_id": world.world_id,
            "model_answer": "true", "elapsed_seconds": 0.25,
            "trace_steps": [], "cited_evidence": [],
        }


def test_task_deserialization_preserves_context_metadata_and_proof():
    task = deserialize_task(_task())
    assert task.context_text == "A premise."
    assert task.metadata == {"prior_answer": "false"}
    assert task.proof_graph.required_document_lines == {"doc": [1, 2]}


def test_runtime_explicitly_distinguishes_configured_and_historical_limits():
    config = {"agent_runtime": {"max_turns": 14, "max_tokens_per_turn": 256,
                                "temperature": 0.0, "greedy": True}}
    assert runtime_settings(config, "configured")["max_turns"] == 14
    assert runtime_settings(config, "legacy")["max_turns"] == 12
    assert runtime_settings(config, "legacy")["max_tokens_per_turn"] == 128
    with pytest.raises(ValueError, match="deterministic"):
        runtime_settings({"agent_runtime": {"temperature": 0.5}}, "configured")


def test_manifest_fingerprints_input_runtime_and_checkpoint(tmp_path):
    inputs = tmp_path / "input.json"
    inputs.write_text("original", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"weights")
    arguments = dict(tasks=[_task()], files={"data": inputs},
                     checkpoint=checkpoint, runtime={"max_turns": 12},
                     source_files=[inputs])
    first = audit_manifest(**arguments)
    assert first == audit_manifest(**arguments)
    weights.write_bytes(b"changed")
    assert first["fingerprint"] != audit_manifest(**arguments)["fingerprint"]
    weights.write_bytes(b"weights")
    inputs.write_text("changed input", encoding="utf-8")
    assert first["fingerprint"] != audit_manifest(**arguments)["fingerprint"]
    arguments["runtime"] = {"max_turns": 14}
    assert first["fingerprint"] != audit_manifest(**arguments)["fingerprint"]


def test_journal_survives_interruption_and_resumes_only_missing_tasks(tmp_path):
    tasks = [_task("one"), _task("two")]
    worlds = {"world": World(world_id="world", seed=1)}
    manifest = {"fingerprint": "frozen", "task_ids": ["one", "two"]}
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate_tasks(FakeShell(fail_on="two"), worlds, tasks, tmp_path, manifest)
    journal = tmp_path / "case_predictions.jsonl"
    first_record = json.loads(journal.read_text(encoding="utf-8"))
    assert first_record["outcome"]["grounded_success"] is True
    assert first_record["episode"]["elapsed_seconds"] == 0.25
    shell = FakeShell()
    metrics = evaluate_tasks(shell, worlds, tasks, tmp_path, manifest)
    assert shell.calls == ["two"]
    assert metrics["total_eval_tasks"] == 2
    assert metrics["total_episode_seconds"] == 0.5
    assert metrics["ms_per_task"] == 250.0
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 2
    with pytest.raises(ValueError, match="manifest"):
        evaluate_tasks(shell, worlds, tasks, tmp_path,
                       {**manifest, "fingerprint": "different"})


def test_resume_rejects_wrong_task_order_and_duplicate_ids(tmp_path):
    tasks = [_task("one"), _task("two")]
    worlds = {"world": World(world_id="world", seed=1)}
    manifest = {"fingerprint": "frozen", "task_ids": ["one", "two"]}
    evaluate_tasks(FakeShell(), worlds, tasks, tmp_path, manifest)
    with pytest.raises(ValueError, match="task"):
        evaluate_tasks(FakeShell(), worlds, list(reversed(tasks)), tmp_path, manifest)
    with pytest.raises(ValueError, match="unique"):
        evaluate_tasks(FakeShell(), worlds, [tasks[0], tasks[0]], tmp_path, manifest)


def test_corrected_replay_withholds_evidence_without_changing_shared_world(tmp_path):
    task = {**_task(), "metadata": {"withhold_evidence": True}}
    world = World(world_id="world", seed=1, documents={
        "doc": Document(id="doc", title="Record", doc_type="record", lines=[
            DocumentLine(line_number=1, text="hidden")])})

    class EvidenceShell(FakeShell):
        def run_episode(self, current_world, current_task):
            assert current_world.documents == {}
            return super().run_episode(current_world, current_task)

    metrics = evaluate_tasks(EvidenceShell(), {"world": world}, [task], tmp_path,
                             {"fingerprint": "controlled", "task_ids": ["one"]},
                             evidence_controls=True)
    assert "doc" in world.documents
    assert metrics["strict_grounded_success_rate"] == 1.0


def test_corrected_replay_refuses_missing_evidence_and_silent_suite_omission():
    from eval.frozen_checkpoint_audit import validate_evaluation_inputs

    task = {**_task(), "is_retrieval_required": True}
    worlds = {"world": World(world_id="world", seed=1)}
    with pytest.raises(ValueError, match="evidence"):
        validate_evaluation_inputs([task], worlds, ["unit"])
    with pytest.raises(ValueError, match="suite"):
        validate_evaluation_inputs([_task()], worlds, ["missing_suite"])


def test_runtime_model_metadata_is_measured_not_assumed_float32():
    from eval.frozen_checkpoint_audit import model_runtime_metadata

    model = torch.nn.Linear(2, 3).to(dtype=torch.float64)
    metadata = model_runtime_metadata(model)
    assert metadata == {"weight_dtype": "torch.float64", "parameters": 9}
