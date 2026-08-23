"""End-to-end integration test for LCM OpenAI-compatible API Server.

Spawns a real live Uvicorn HTTP server in a background thread, executes non-trivial
multi-hop, computation, and abstention queries via HTTP API, and verifies expected
procedural outputs and epistemic evidence citations.
"""

import json
import socket
import threading
import time
from typing import Any, Dict, List
import httpx
import pytest
import uvicorn

from agent.protocol import FinalMessage, PlanMessage, ToolCallMessage
from agent.server import create_app
from agent.shell import DeterministicShell
from synth.ontology import Document, DocumentLine, World, Task


def find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ScriptedMultiTurnAgent(DeterministicShell):
    """Deterministic policy simulating procedural multi-hop, math, and abstention steps."""

    def run_episode(self, world: World, task: Task) -> Dict[str, Any]:
        q = task.question.lower()

        # Case 1: Multi-Hop Fact Retrieval
        if "quantumtech" in q and "energy output" in q:
            trace_steps = [
                {
                    "turn": 1,
                    "raw_output": '{"type": "plan", "goal": "Find QuantumTech sector then lookup energy output", "needs": ["headquarters", "energy"], "next_action": "search"}',
                    "parsed_type": "plan"
                },
                {
                    "turn": 2,
                    "raw_output": '{"type": "tool_call", "tool": "search", "arguments": {"query": "QuantumTech"}}',
                    "parsed_type": "action",
                    "observation": {"status": "success", "hop": "OBS SEARCH [D01: Score 0.98]"}
                },
                {
                    "turn": 3,
                    "raw_output": '{"type": "tool_call", "tool": "read", "arguments": {"document_id": "D01", "lines": [1]}}',
                    "parsed_type": "action",
                    "observation": {"status": "success", "hop": "OBS READ D01 LINES 1-1\nD01:L1 QuantumTech headquarters is located in Sector 9."}
                },
                {
                    "turn": 4,
                    "raw_output": '{"type": "tool_call", "tool": "search", "arguments": {"query": "Sector 9"}}',
                    "parsed_type": "action",
                    "observation": {"status": "success", "hop": "OBS SEARCH [D02: Score 0.94]"}
                },
                {
                    "turn": 5,
                    "raw_output": '{"type": "tool_call", "tool": "read", "arguments": {"document_id": "D02", "lines": [1]}}',
                    "parsed_type": "action",
                    "observation": {"status": "success", "hop": "OBS READ D02 LINES 1-1\nD02:L1 Sector 9 base energy output is 14200 MW."}
                }
            ]
            return {
                "task_id": task.task_id,
                "world_id": world.world_id,
                "gold_answer": "14200 MW",
                "model_answer": "14200 MW",
                "cited_evidence": [
                    {"document_id": "D01", "lines": [1]},
                    {"document_id": "D02", "lines": [1]}
                ],
                "turns_used": 5,
                "search_count": 2,
                "read_count": 2,
                "exec_count": 0,
                "is_terminated": True,
                "termination_reason": "FINAL_ANSWER",
                "elapsed_seconds": 0.0035,
                "trace_steps": trace_steps
            }

        # Case 2: Retrieval + Math Computation
        elif "solarishub" in q and "net profit" in q:
            trace_steps = [
                {
                    "turn": 1,
                    "raw_output": '{"type": "plan", "goal": "Retrieve SolarisHub revenue and cost, then calculate profit", "needs": ["revenue", "cost"], "next_action": "search"}',
                    "parsed_type": "plan"
                },
                {
                    "turn": 2,
                    "raw_output": '{"type": "tool_call", "tool": "search", "arguments": {"query": "SolarisHub"}}',
                    "parsed_type": "action",
                    "observation": {"status": "success", "hop": "OBS SEARCH [D03: Score 0.99]"}
                },
                {
                    "turn": 3,
                    "raw_output": '{"type": "tool_call", "tool": "read", "arguments": {"document_id": "D03", "lines": [1, 2]}}',
                    "parsed_type": "action",
                    "observation": {"status": "success", "hop": "OBS READ D03 LINES 1-2\nD03:L1 SolarisHub annual revenue is 85000.\nD03:L2 SolarisHub operating cost is 32000."}
                },
                {
                    "turn": 4,
                    "raw_output": '{"type": "tool_call", "tool": "exec", "arguments": {"code": "85000 - 32000"}}',
                    "parsed_type": "action",
                    "observation": {"status": "success", "hop": "OBS MATH 53000"}
                }
            ]
            return {
                "task_id": task.task_id,
                "world_id": world.world_id,
                "gold_answer": "53000",
                "model_answer": "53000",
                "cited_evidence": [
                    {"document_id": "D03", "lines": [1, 2]}
                ],
                "turns_used": 4,
                "search_count": 1,
                "read_count": 1,
                "exec_count": 1,
                "is_terminated": True,
                "termination_reason": "FINAL_ANSWER",
                "elapsed_seconds": 0.0028,
                "trace_steps": trace_steps
            }

        # Case 3: Epistemic Abstention / Missing Evidence
        elif "nebulaunknown" in q:
            trace_steps = [
                {
                    "turn": 1,
                    "raw_output": '{"type": "plan", "goal": "Search for NebulaUnknown", "needs": ["population"], "next_action": "search"}',
                    "parsed_type": "plan"
                },
                {
                    "turn": 2,
                    "raw_output": '{"type": "tool_call", "tool": "search", "arguments": {"query": "NebulaUnknown"}}',
                    "parsed_type": "action",
                    "observation": {"status": "success", "hop": "OBS SEARCH []"}
                }
            ]
            return {
                "task_id": task.task_id,
                "world_id": world.world_id,
                "gold_answer": "insufficient_evidence",
                "model_answer": "insufficient_evidence",
                "cited_evidence": [],
                "turns_used": 2,
                "search_count": 1,
                "read_count": 0,
                "exec_count": 0,
                "is_terminated": True,
                "termination_reason": "INSUFFICIENT_EVIDENCE",
                "elapsed_seconds": 0.0019,
                "trace_steps": trace_steps
            }

        # Fallback default
        return super().run_episode(world, task)


@pytest.fixture(scope="module")
def e2e_server():
    """Spawns an actual Uvicorn HTTP server on a free port in a background thread."""
    # Build procedural world
    d1 = Document(
        id="D01",
        title="QuantumTech Registry",
        doc_type="registry",
        lines=[
            DocumentLine(line_number=1, text="QuantumTech headquarters is located in Sector 9."),
            DocumentLine(line_number=2, text="QuantumTech status is active.")
        ]
    )
    d2 = Document(
        id="D02",
        title="Sector 9 Infrastructure",
        doc_type="report",
        lines=[
            DocumentLine(line_number=1, text="Sector 9 base energy output is 14200 MW."),
            DocumentLine(line_number=2, text="Sector 9 grid tax is 15%.")
        ]
    )
    d3 = Document(
        id="D03",
        title="SolarisHub Financials",
        doc_type="table",
        lines=[
            DocumentLine(line_number=1, text="SolarisHub annual revenue is 85000."),
            DocumentLine(line_number=2, text="SolarisHub operating cost is 32000.")
        ]
    )
    world = World(
        world_id="e2e_world",
        seed=101,
        documents={"D01": d1, "D02": d2, "D03": d3}
    )

    shell = ScriptedMultiTurnAgent(model=None, tokenizer=None, max_turns=8)
    app = create_app(shell=shell, world=world, model_id="lcm-e2e")

    port = find_free_port()
    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config=config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"

    # Wait for server to become responsive
    max_wait = 10.0
    start = time.time()
    while time.time() - start < max_wait:
        try:
            with httpx.Client(timeout=1.0) as client:
                res = client.get(f"{base_url}/health")
                if res.status_code == 200:
                    break
        except Exception:
            time.sleep(0.05)
    else:
        pytest.fail(f"Server failed to start within {max_wait} seconds on port {port}")

    yield base_url

    server.should_exit = True
    thread.join(timeout=2.0)


def test_e2e_health_and_models(e2e_server):
    with httpx.Client(base_url=e2e_server, timeout=5.0) as client:
        # Check Health
        res_health = client.get("/health")
        assert res_health.status_code == 200
        health_data = res_health.json()
        assert health_data["status"] == "ok"
        assert health_data["model_id"] == "lcm-e2e"
        assert health_data["documents_loaded"] == 3

        # Check Models
        res_models = client.get("/v1/models")
        assert res_models.status_code == 200
        models_data = res_models.json()
        model_ids = [m["id"] for m in models_data["data"]]
        assert "lcm-e2e" in model_ids


def test_e2e_multi_hop_query(e2e_server):
    """Exercise prompt 1: Multi-Hop Fact Retrieval (QuantumTech -> Sector 9 -> Energy Output)."""
    with httpx.Client(base_url=e2e_server, timeout=5.0) as client:
        payload = {
            "model": "lcm-e2e",
            "messages": [
                {
                    "role": "user",
                    "content": "Where is the headquarters of QuantumTech, and what is the base energy output of that sector?"
                }
            ],
            "temperature": 0.0,
            "stream": False
        }
        res = client.post("/v1/chat/completions", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert data["object"] == "chat.completion"
        assert data["model"] == "lcm-e2e"
        choice = data["choices"][0]
        content = choice["message"]["content"]

        # Verify ground truth answer and multi-hop citations
        assert "14200 MW" in content
        assert "`D01:L1`" in content
        assert "`D02:L1`" in content
        assert choice["finish_reason"] == "stop"


def test_e2e_retrieval_and_computation_query(e2e_server):
    """Exercise prompt 2: Retrieval + Math Computation (SolarisHub Revenue - Cost)."""
    with httpx.Client(base_url=e2e_server, timeout=5.0) as client:
        payload = {
            "model": "lcm-e2e",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the net profit of SolarisHub?"
                }
            ],
            "temperature": 0.0,
            "stream": False
        }
        res = client.post("/v1/chat/completions", json=payload)
        assert res.status_code == 200
        data = res.json()

        choice = data["choices"][0]
        content = choice["message"]["content"]

        # Verify calculated answer 85000 - 32000 = 53000 and citations
        assert "53000" in content
        assert "`D03:L1,2`" in content or ("`D03:L1`" in content and "`D03:L2`" in content)
        assert choice["finish_reason"] == "stop"


def test_e2e_epistemic_abstention_query(e2e_server):
    """Exercise prompt 3: Epistemic Abstention (Unknown entity without guessing)."""
    with httpx.Client(base_url=e2e_server, timeout=5.0) as client:
        payload = {
            "model": "lcm-e2e",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the population of NebulaUnknown?"
                }
            ],
            "temperature": 0.0,
            "stream": False
        }
        res = client.post("/v1/chat/completions", json=payload)
        assert res.status_code == 200
        data = res.json()

        choice = data["choices"][0]
        content = choice["message"]["content"]

        # Model must abstain when evidence is not found
        assert "insufficient_evidence" in content
        assert "Evidence cited:" not in content


def test_e2e_streaming_multi_turn_trace(e2e_server):
    """Exercise SSE streaming across multi-hop reasoning steps."""
    with httpx.Client(base_url=e2e_server, timeout=5.0) as client:
        payload = {
            "model": "lcm-e2e",
            "messages": [
                {
                    "role": "user",
                    "content": "Where is the headquarters of QuantumTech, and what is the base energy output of that sector?"
                }
            ],
            "stream": True
        }
        res = client.post("/v1/chat/completions", json=payload)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]

        body_text = res.text
        # Verify SSE stream contains reasoning steps and final answer
        assert "data: " in body_text
        assert "data: [DONE]" in body_text
        assert "Find QuantumTech sector" in body_text
        assert "OBS SEARCH [D01" in body_text
        assert "OBS READ D01" in body_text
        assert "14200 MW" in body_text
        assert "`D01:L1`" in body_text
