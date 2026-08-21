"""Unit tests for protocol parser and shell state machine."""

from agent.protocol import parse_and_validate_message, PlanMessage, ToolCallMessage, FinalMessage, ProtocolError
from agent.state import AgentState


def test_protocol_parsing():
    # Valid Plan
    plan_raw = '{"type": "plan", "goal": "find_x", "needs": ["a"], "next_action": "search"}'
    msg = parse_and_validate_message(plan_raw)
    assert isinstance(msg, PlanMessage)
    assert msg.next_action == "search"

    # Valid Tool Call
    action_raw = '{"type": "tool_call", "tool": "search", "arguments": {"query": "noru"}}'
    msg2 = parse_and_validate_message(action_raw)
    assert isinstance(msg2, ToolCallMessage)
    assert msg2.tool == "search"

    # Valid Final
    final_raw = '{"type": "final", "answer": "veska", "evidence": [{"document_id": "D01", "lines": [1]}]}'
    msg3 = parse_and_validate_message(final_raw)
    assert isinstance(msg3, FinalMessage)
    assert msg3.answer == "veska"

    # Compound JSON (Plan followed by Final in single output stream)
    compound_raw = '{"type": "plan", "goal": "find_x", "needs": ["a"], "next_action": "final"}{"type": "final", "answer": "veska", "evidence": [{"document_id": "D01", "lines": [1]}]}'
    msg4 = parse_and_validate_message(compound_raw)
    assert isinstance(msg4, FinalMessage)
    assert msg4.answer == "veska"

    # Compound JSON (Plan followed by Tool Call)
    compound_action_raw = '{"type": "plan", "goal": "find_x", "needs": ["a"], "next_action": "search"}{"type": "tool_call", "tool": "search", "arguments": {"query": "noru"}}'
    msg5 = parse_and_validate_message(compound_action_raw)
    assert isinstance(msg5, ToolCallMessage)
    assert msg5.tool == "search"

    # Malformed JSON
    bad_raw = '{"type": "plan", goal: broken}'
    err = parse_and_validate_message(bad_raw)
    assert isinstance(err, ProtocolError)


def test_agent_state_limits():
    state = AgentState(max_turns=3, max_search_calls=2)
    
    assert state.increment_turn() is None
    assert state.increment_turn() is None
    assert state.increment_turn() is None
    assert state.increment_turn() == "MAX_TURNS_EXCEEDED"
    assert state.is_terminated is True

    state2 = AgentState(max_search_calls=1)
    assert state2.record_tool_call("search") is None
    assert state2.record_tool_call("search") == "MAX_SEARCH_CALLS_EXCEEDED"
