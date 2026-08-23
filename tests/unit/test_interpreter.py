"""Unit tests and latency benchmarks for the native Python RDL Interpreter (Milestone 2)."""

import time
import pytest
from synth.ontology import World, Document, DocumentLine
from agent.interpreter import RDLInterpreter
from agent.adapters.document import DocumentEvidenceProvider
from agent.adapters.registry import AdapterRegistry


def create_sample_world():
    world = World(world_id="W_TEST_INTERP", seed=42)
    world.documents["D01"] = Document(
        id="D01",
        title="Census Fort Valerius",
        doc_type="census",
        lines=[
            DocumentLine(line_number=1, text="Census report for Valerius Fort.", fact_ids=[]),
            DocumentLine(line_number=2, text="Fort garrison count is 140.", fact_ids=[]),
            DocumentLine(line_number=3, text="Stationed in Northern valley.", fact_ids=[]),
        ]
    )
    world.documents["D04"] = Document(
        id="D04",
        title="Census Fort Albia",
        doc_type="census",
        lines=[
            DocumentLine(line_number=1, text="Fort Albia standing garrison report.", fact_ids=[]),
            DocumentLine(line_number=2, text="Active infantry stationed: 210.", fact_ids=[]),
        ]
    )
    return world


def test_interpreter_search_quote_stripping_and_hop():
    """Verify interpreter executes SEARCH with quotes stripped and returns valid HOP output."""
    world = create_sample_world()
    registry = AdapterRegistry()
    registry.register("docs", DocumentEvidenceProvider(world))
    interpreter = RDLInterpreter(registry=registry, default_doc_adapter="docs")

    step = interpreter.execute('SEARCH "Valerius Fort" LIMIT 2')
    assert step.is_action is True
    assert step.is_final is False
    assert "OBS SEARCH [D01" in step.hop_observation
    assert step.action_data["tool"] == "search"
    assert step.action_data["arguments"]["query"] == "Valerius Fort"


def test_interpreter_read_line_alignment_and_hop():
    """Verify interpreter executes READ with 1-indexed line ranges mapped to 0-indexed arrays."""
    world = create_sample_world()
    registry = AdapterRegistry()
    registry.register("docs", DocumentEvidenceProvider(world))
    interpreter = RDLInterpreter(registry=registry, default_doc_adapter="docs")

    step = interpreter.execute("READ D01 LINES 1-2")
    assert step.is_action is True
    assert "OBS READ D01 LINES 1-2" in step.hop_observation
    assert "D01:L1 Census report for Valerius Fort." in step.hop_observation
    assert "D01:L2 Fort garrison count is 140." in step.hop_observation
    assert "D01:L3" not in step.hop_observation


def test_interpreter_math_execution():
    """Verify interpreter executes MATH safely and returns HOP numeric observation."""
    interpreter = RDLInterpreter()

    step = interpreter.execute("MATH (140 + 210) * 2")
    assert step.is_action is True
    assert step.hop_observation == "OBS MATH 700"

    # Math error handling
    step_err = interpreter.execute("MATH 10 / 0")
    assert "OBS MATH ERROR DIVISION_BY_ZERO" in step_err.hop_observation


def test_interpreter_emit_and_abstain():
    """Verify interpreter handles EMIT with evidence and ABSTAIN with reasons."""
    interpreter = RDLInterpreter()

    step_emit = interpreter.execute('EMIT "350" EVIDENCE [D01:2, D04:2]')
    assert step_emit.is_final is True
    assert step_emit.final_answer == "350"
    assert len(step_emit.cited_evidence) == 2
    assert step_emit.cited_evidence[0]["document_id"] == "D01"
    assert step_emit.cited_evidence[0]["lines"] == [2]

    step_abstain = interpreter.execute("ABSTAIN REASON insufficient_evidence")
    assert step_abstain.is_final is True
    assert step_abstain.final_answer == "insufficient_evidence"
    assert len(step_abstain.cited_evidence) == 0


def test_interpreter_submillisecond_latency():
    """Verify that RDL interpreter execution achieves <0.5ms step latency."""
    world = create_sample_world()
    registry = AdapterRegistry()
    registry.register("docs", DocumentEvidenceProvider(world))
    interpreter = RDLInterpreter(registry=registry, default_doc_adapter="docs")

    # Warmup
    interpreter.execute('SEARCH "Valerius Fort" LIMIT 2')

    n_runs = 500
    start = time.perf_counter()
    for _ in range(n_runs):
        interpreter.execute('SEARCH "Valerius Fort" LIMIT 2')
        interpreter.execute("READ D01 LINES 1-2")
        interpreter.execute("MATH 140 + 210")
    elapsed = time.perf_counter() - start
    
    total_steps = n_runs * 3
    avg_latency_ms = (elapsed / total_steps) * 1000.0
    print(f"\n[+] Interpreter Average Step Latency: {avg_latency_ms:.3f} ms/step")
    assert avg_latency_ms < 0.5, f"Expected <0.5ms per step, got {avg_latency_ms:.3f}ms"
