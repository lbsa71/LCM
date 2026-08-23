"""Unit tests for native Retrieval Domain Language (RDL) & Host Observation Protocol (HOP)."""

import pytest
from tokenizers import Tokenizer
from agent.protocol import (
    parse_and_validate_message,
    ToolCallMessage,
    FinalMessage,
    ProtocolError,
    format_search_hop,
    format_read_hop,
    format_math_hop,
    format_error_hop
)
from agent.shell import DeterministicShell
from synth.ontology import World, Document, DocumentLine, Task, ProofGraph
from eval.metrics import evaluate_episode_outcome


def test_rdl_action_parsing():
    """Test parsing of all core RDL action statements."""
    # 1. SEARCH with quotes and limit
    search_raw = 'SEARCH "Fort Valerius" LIMIT 3'
    msg1 = parse_and_validate_message(search_raw)
    assert isinstance(msg1, ToolCallMessage)
    assert msg1.tool == "search"
    assert msg1.arguments == {"query": "Fort Valerius", "limit": 3}

    # 2. SEARCH default limit
    search_raw2 = 'SEARCH "Northern Sector"'
    msg2 = parse_and_validate_message(search_raw2)
    assert isinstance(msg2, ToolCallMessage)
    assert msg2.tool == "search"
    assert msg2.arguments == {"query": "Northern Sector", "limit": 5}

    # 3. READ with line range
    read_raw = "READ D04 LINES 2-6"
    msg3 = parse_and_validate_message(read_raw)
    assert isinstance(msg3, ToolCallMessage)
    assert msg3.tool == "read"
    assert msg3.arguments == {"document_id": "D04", "lines": [2, 3, 4, 5, 6]}

    # 4. READ full document
    read_raw2 = "READ D01"
    msg4 = parse_and_validate_message(read_raw2)
    assert isinstance(msg4, ToolCallMessage)
    assert msg4.tool == "read"
    assert msg4.arguments == {"document_id": "D01"}

    # 5. MATH pure arithmetic
    math_raw = "MATH 140 + 260"
    msg5 = parse_and_validate_message(math_raw)
    assert isinstance(msg5, ToolCallMessage)
    assert msg5.tool == "exec"
    assert msg5.arguments == {"code": "140 + 260"}

    # 6. EMIT with evidence citations
    emit_raw = 'EMIT "400" EVIDENCE [D01:2, D03:5]'
    msg6 = parse_and_validate_message(emit_raw)
    assert isinstance(msg6, FinalMessage)
    assert msg6.answer == "400"
    assert len(msg6.evidence) == 2
    assert msg6.evidence[0].document_id == "D01"
    assert msg6.evidence[0].lines == [2]
    assert msg6.evidence[1].document_id == "D03"
    assert msg6.evidence[1].lines == [5]

    # 7. ABSTAIN with reason
    abstain_raw = "ABSTAIN REASON insufficient_evidence"
    msg7 = parse_and_validate_message(abstain_raw)
    assert isinstance(msg7, FinalMessage)
    assert msg7.answer == "insufficient_evidence"
    assert msg7.evidence == []


def test_hop_formatting():
    """Test Host Observation Protocol (HOP) formatting functions."""
    # Search HOP with hits
    hits = [{"document_id": "D01", "score": 8.4}, {"document_id": "D04", "score": 5.2}]
    obs_search = format_search_hop(hits)
    assert obs_search == "OBS SEARCH [D01 (8.4), D04 (5.2)]"

    # Search HOP empty
    obs_empty = format_search_hop([])
    assert obs_empty == "OBS SEARCH EMPTY"

    # Read HOP with line content
    doc = Document(
        id="D01",
        title="Registry",
        doc_type="census",
        lines=[
            DocumentLine(line_number=1, text="Registry start."),
            DocumentLine(line_number=2, text="Valerius Fort holds 140."),
            DocumentLine(line_number=3, text="Northern post.")
        ]
    )
    obs_read = format_read_hop(doc, lines=[1, 2], doc_id="D01")
    assert "OBS READ D01 LINES 1-2" in obs_read
    assert "D01:L1 Registry start." in obs_read
    assert "D01:L2 Valerius Fort holds 140." in obs_read

    # Math HOP
    obs_math = format_math_hop(400)
    assert obs_math == "OBS MATH 400"

    # Error HOP
    obs_err = format_error_hop("DIVISION_BY_ZERO")
    assert obs_err == "OBS ERROR DIVISION_BY_ZERO"


def test_rdl_deterministic_shell_execution():
    """End-to-end smoke test executing a scripted policy emitting RDL statements."""
    world = World(world_id="w_smoke", seed=42)
    doc1 = Document(
        id="D01",
        title="Census Record",
        doc_type="census",
        lines=[
            DocumentLine(line_number=1, text="Census registry."),
            DocumentLine(line_number=2, text="Fort Valerius has 140 infantry.", fact_ids=["f1"]),
            DocumentLine(line_number=3, text="Western perimeter.")
        ]
    )
    doc2 = Document(
        id="D02",
        title="Territorial Report",
        doc_type="report",
        lines=[
            DocumentLine(line_number=1, text="Outpost registry."),
            DocumentLine(line_number=2, text="Fort Albia has 210 garrison troops.", fact_ids=["f2"]),
        ]
    )
    world.documents["D01"] = doc1
    world.documents["D02"] = doc2

    task = Task(
        task_id="t_smoke_01",
        task_type="retrieval_computation",
        suite="suite_e_retrieval_computation",
        question="What is the combined troops in Fort Valerius and Fort Albia?",
        gold_answer="350",
        world_id="w_smoke",
        is_retrieval_required=True,
        is_contingent=True,
        proof_graph=ProofGraph(
            goal="troop_sum",
            required_fact_ids={"f1", "f2"},
            required_document_lines={"D01": [2], "D02": [2]}
        )
    )

    class MockScriptedModel:
        def __init__(self, script):
            self.script = list(script)
            self.step = 0

        def generate(self, input_tensor, **kwargs):
            import torch
            ans = self.script[self.step]
            self.step += 1
            return torch.tensor([[0] * input_tensor.shape[1] + list(ans.encode("utf-8"))])

    class MockTokenizer:
        def encode(self, text):
            class Out:
                ids = list(text.encode("utf-8"))
            return Out()

        def decode(self, token_ids):
            return bytes(token_ids).decode("utf-8", errors="ignore")

        def token_to_id(self, tag):
            return 1

    scripted_turns = [
        'SEARCH "Fort Valerius" LIMIT 2',
        'READ D01 LINES 1-2',
        'SEARCH "Fort Albia" LIMIT 2',
        'READ D02 LINES 1-2',
        'MATH 140 + 210',
        'EMIT "350" EVIDENCE [D01:2, D02:2]'
    ]

    model = MockScriptedModel(scripted_turns)
    tok = MockTokenizer()
    shell = DeterministicShell(model=model, tokenizer=tok, device="cpu")

    episode = shell.run_episode(world, task)

    assert episode["is_terminated"] is True
    assert episode["model_answer"] == "350"
    assert len(episode["trace_steps"]) == 6

    # Verify 100% grounded accuracy
    outcome = evaluate_episode_outcome(episode, task)
    assert outcome["raw_match"] is True
    assert outcome["grounded_success"] is True
    assert outcome["failure_category"] is None
