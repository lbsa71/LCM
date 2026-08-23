"""Unit tests for the OpenAI-compatible LCM API server."""

import json
import pytest
from fastapi.testclient import TestClient
from synth.ontology import World, Document, DocumentLine, Task
from agent.shell import DeterministicShell
from agent.server import create_app


@pytest.fixture
def mock_world():
    doc = Document(
        id="D01",
        title="Alpha Registry",
        doc_type="registry",
        lines=[
            DocumentLine(line_number=1, text="Alpha Corp is located in Sector 7."),
            DocumentLine(line_number=2, text="Annual revenue for Alpha Corp is 45000."),
            DocumentLine(line_number=3, text="Status is active.")
        ]
    )
    world = World(
        world_id="test_world",
        seed=42,
        documents={"D01": doc}
    )
    return world


@pytest.fixture
def test_client(mock_world):
    # DeterministicShell with fallback / dummy behavior (no neural weights loaded)
    shell = DeterministicShell(model=None, tokenizer=None, max_turns=6)
    app = create_app(shell=shell, world=mock_world, model_id="lcm-smoke")
    return TestClient(app)


def test_health_endpoint(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_id" in data
    assert "device" in data


def test_list_models_endpoint(test_client):
    response = test_client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1
    assert data["data"][0]["id"] == "lcm-smoke"
    assert data["data"][0]["object"] == "model"
    assert data["data"][0]["owned_by"] == "lcm"


def test_chat_completions_sync(test_client):
    req_payload = {
        "model": "lcm-smoke",
        "messages": [
            {"role": "system", "content": "You are a factual LCM agent."},
            {"role": "user", "content": "What is the revenue for Alpha Corp?"}
        ],
        "temperature": 0.0,
        "stream": False
    }
    response = test_client.post("/v1/chat/completions", json=req_payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["object"] == "chat.completion"
    assert data["model"] == "lcm-smoke"
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["finish_reason"] in ["stop", "length"]
    assert "content" in choice["message"]
    assert "usage" in data
    assert data["usage"]["total_tokens"] > 0


def test_chat_completions_streaming(test_client):
    req_payload = {
        "model": "lcm-smoke",
        "messages": [
            {"role": "user", "content": "Find location of Alpha Corp"}
        ],
        "stream": True
    }
    response = test_client.post("/v1/chat/completions", json=req_payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    
    lines = [line.strip() for line in response.text.split("\n") if line.strip()]
    assert len(lines) >= 2
    
    # Verify SSE data line formats
    data_events = []
    for line in lines:
        if line.startswith("data: "):
            payload_str = line[len("data: "):]
            if payload_str == "[DONE]":
                break
            data_events.append(json.loads(payload_str))
    
    assert len(data_events) >= 1
    assert data_events[0]["object"] == "chat.completion.chunk"
    assert "delta" in data_events[0]["choices"][0]


def test_chat_completions_validation_errors(test_client):
    # Missing user message
    res1 = test_client.post("/v1/chat/completions", json={"messages": []})
    assert res1.status_code == 400
    
    # System message only
    res2 = test_client.post("/v1/chat/completions", json={"messages": [{"role": "system", "content": "hi"}]})
    assert res2.status_code == 400

    # Whitespace only user message
    res3 = test_client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "   "}]})
    assert res3.status_code == 400


def test_cors_headers(test_client):
    response = test_client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"
        }
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in ["*", "http://localhost:3000"]


def test_chat_completions_with_trace_steps(mock_world):
    # Create a custom mock shell that simulates intermediate plan and action steps
    class MockTraceShell(DeterministicShell):
        def run_episode(self, world, task):
            return {
                "task_id": task.task_id,
                "world_id": world.world_id,
                "gold_answer": "45000",
                "model_answer": "45000",
                "cited_evidence": [{"document_id": "D01", "lines": [2]}],
                "turns_used": 2,
                "is_terminated": True,
                "trace_steps": [
                    {
                        "turn": 1,
                        "raw_output": '{"type": "plan", "goal": "Find revenue", "needs": ["revenue"], "next_action": "search"}',
                        "parsed_type": "plan"
                    },
                    {
                        "turn": 2,
                        "raw_output": '{"type": "tool_call", "tool": "search", "arguments": {"query": "Alpha Corp"}}',
                        "parsed_type": "action",
                        "observation": {"status": "success", "hop": "OBS SEARCH [D01: Score 0.95]"}
                    }
                ]
            }

    shell = MockTraceShell(model=None, tokenizer=None)
    app = create_app(shell=shell, world=mock_world, model_id="lcm-test")
    client = TestClient(app)

    # Test sync format with evidence
    res_sync = client.post("/v1/chat/completions", json={
        "model": "lcm-test",
        "messages": [{"role": "user", "content": "What is revenue?"}],
        "stream": False
    })
    assert res_sync.status_code == 200
    content = res_sync.json()["choices"][0]["message"]["content"]
    assert "45000" in content
    assert "`D01:L2`" in content

    # Test stream format with trace deltas
    res_stream = client.post("/v1/chat/completions", json={
        "model": "lcm-test",
        "messages": [{"role": "user", "content": "What is revenue?"}],
        "stream": True
    })
    assert res_stream.status_code == 200
    assert "Plan:" in res_stream.text
    assert "OBS SEARCH" in res_stream.text
    assert "45000" in res_stream.text

