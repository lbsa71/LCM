"""Unit tests for atomic action protocol without plan-turn collisions."""

import json
import pytest
from pydantic import ValidationError

from agent.protocol import ToolCallMessage, FinalMessage, ProtocolError, parse_and_validate_message


def test_atomic_tool_call_parsing():
    """Verify tool call is cleanly parsed without needing intermediate plan tokens."""
    raw_tool = json.dumps({
        "type": "tool_call",
        "tool": "search",
        "arguments": {"query": "barel", "limit": 3}
    })
    parsed = parse_and_validate_message(raw_tool)
    assert isinstance(parsed, ToolCallMessage)
    assert parsed.tool == "search"
    assert parsed.arguments["query"] == "barel"


def test_atomic_final_parsing_with_citations():
    """Verify final message with citations is parsed deterministically."""
    raw_final = json.dumps({
        "type": "final",
        "answer": "613",
        "evidence": [{"document_id": "D01", "lines": [2]}]
    })
    parsed = parse_and_validate_message(raw_final)
    assert isinstance(parsed, FinalMessage)
    assert parsed.answer == "613"
    assert len(parsed.evidence) == 1
    assert parsed.evidence[0].document_id == "D01"
    assert parsed.evidence[0].lines == [2]


def test_atomic_final_abstention():
    """Verify clean insufficient evidence abstention."""
    raw_abstain = json.dumps({
        "type": "final",
        "answer": "insufficient_evidence",
        "evidence": []
    })
    parsed = parse_and_validate_message(raw_abstain)
    assert isinstance(parsed, FinalMessage)
    assert parsed.answer == "insufficient_evidence"
    assert len(parsed.evidence) == 0


def test_atomic_rdl_and_json_interoperability():
    """Verify both flat RDL and JSON envelopes are parsed interoperably."""
    rdl_search = parse_and_validate_message("SEARCH Bareldan LIMIT 3")
    json_search = parse_and_validate_message(json.dumps({"type": "tool_call", "tool": "search", "arguments": {"query": "Bareldan", "limit": 3}}))
    assert isinstance(rdl_search, ToolCallMessage)
    assert isinstance(json_search, ToolCallMessage)
    assert rdl_search.tool == json_search.tool
    assert rdl_search.arguments == json_search.arguments

    rdl_final = parse_and_validate_message("EMIT 613 EVIDENCE [D01:2]")
    json_final = parse_and_validate_message(json.dumps({"type": "final", "answer": "613", "evidence": [{"document_id": "D01", "lines": [2]}]}))
    assert isinstance(rdl_final, FinalMessage)
    assert isinstance(json_final, FinalMessage)
    assert rdl_final.answer == json_final.answer
    assert rdl_final.evidence == json_final.evidence

