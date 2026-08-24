"""OpenAI-compatible HTTP API server for Language & Computation Model (LCM).

Supports standard /v1/models and /v1/chat/completions endpoints for seamless
integration with Open WebUI, LibreChat, Promptfoo, DeepEval, and LiteLLM.
"""

import argparse
import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional
import yaml
import torch
from tokenizers import Tokenizer
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.shell import DeterministicShell
from synth.ontology import World, Task, ProofGraph
from synth.world import WorldGenerator
from training.model import SyntheticTransformer


# ---------------------------------------------------------
# OpenAI Compatible Pydantic Schemas
# ---------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "lcm-primary"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 128
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None


class ChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "lcm"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelCard]


# ---------------------------------------------------------
# Application Factory
# ---------------------------------------------------------

def format_answer_with_citations(result: Dict[str, Any]) -> str:
    """Formats the episode result with epistemic evidence citations."""
    answer = result.get("model_answer") or "insufficient_evidence"
    evidence = result.get("cited_evidence", [])
    
    if evidence:
        citations = []
        for e in evidence:
            doc_id = e.get("document_id", "D0")
            lines = e.get("lines", [])
            if lines:
                citations.append(f"`{doc_id}:L{','.join(map(str, lines))}`")
            else:
                citations.append(f"`{doc_id}`")
        citation_str = ", ".join(citations)
        return f"{answer}\n\n*Evidence cited:* {citation_str}"
    return answer


def create_app(
    shell: DeterministicShell,
    world: World,
    model_id: str = "lcm-primary"
) -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    app = FastAPI(
        title="LCM OpenAI-Compatible Server",
        description="Deterministic ReAct Procedural Execution Host for LCM",
        version="0.1.0"
    )

    # Enable CORS for web frontends (Open WebUI, LibreChat)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        device_name = str(shell.device) if hasattr(shell, "device") else "cpu"
        return {
            "status": "ok",
            "model_id": model_id,
            "device": device_name,
            "world_id": world.world_id if world else "none",
            "documents_loaded": len(world.documents) if world and hasattr(world, "documents") else 0
        }

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models():
        models = [
            ModelCard(id=model_id),
            ModelCard(id="lcm-smoke"),
            ModelCard(id="lcm-small"),
            ModelCard(id="lcm-primary")
        ]
        # Deduplicate while preserving order
        seen = set()
        unique_models = []
        for m in models:
            if m.id not in seen:
                seen.add(m.id)
                unique_models.append(m)
        return ModelListResponse(data=unique_models)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        # 1. Extract and validate user prompt
        user_messages = [m for m in request.messages if m.role == "user" and m.content.strip()]
        if not user_messages:
            raise HTTPException(status_code=400, detail="Request must contain at least one non-empty user message.")
        
        user_query = user_messages[-1].content.strip()
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        task = Task(
            task_id=task_id,
            task_type="interactive",
            suite="interactive",
            question=user_query,
            gold_answer="",
            proof_graph=ProofGraph(goal=user_query),
            world_id=world.world_id if world else "none"
        )

        # 2. Synchronous (Non-Streaming) Mode
        if not request.stream:
            result = shell.run_episode(world, task)
            formatted_answer = format_answer_with_citations(result)
            
            prompt_tokens = max(1, len(user_query.split()))
            completion_tokens = max(1, len(formatted_answer.split()))
            
            return ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionMessage(role="assistant", content=formatted_answer),
                        finish_reason="stop" if result.get("is_terminated") else "length"
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )

        # 3. Streaming (Server-Sent Events) Mode
        async def sse_generator() -> AsyncGenerator[str, None]:
            req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created_ts = int(time.time())

            # Initial role chunk
            initial_chunk = ChatCompletionChunk(
                id=req_id,
                created=created_ts,
                model=request.model,
                choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(role="assistant"))]
            )
            yield f"data: {initial_chunk.model_dump_json()}\n\n"
            await asyncio.sleep(0.001)

            # Run episode execution
            result = shell.run_episode(world, task)
            
            # Stream intermediate reasoning / tool trace steps
            for step in result.get("trace_steps", []):
                step_type = step.get("parsed_type", "")
                raw = step.get("raw_output", "")
                
                if step_type == "plan":
                    delta_text = f"> **Plan:** `{raw}`\n\n"
                elif step_type == "action":
                    obs = step.get("observation", {})
                    obs_hop = obs.get("hop", "") if isinstance(obs, dict) else str(obs)
                    delta_text = f"> **Action:** `{raw}`\n> **Obs:** `{obs_hop}`\n\n"
                else:
                    continue

                chunk = ChatCompletionChunk(
                    id=req_id,
                    created=created_ts,
                    model=request.model,
                    choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=delta_text))]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0.001)

            # Stream final formatted answer
            formatted_answer = format_answer_with_citations(result)
            final_chunk = ChatCompletionChunk(
                id=req_id,
                created=created_ts,
                model=request.model,
                choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=formatted_answer))]
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            await asyncio.sleep(0.001)

            # End of stream chunk
            stop_chunk = ChatCompletionChunk(
                id=req_id,
                created=created_ts,
                model=request.model,
                choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(), finish_reason="stop")]
            )
            yield f"data: {stop_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    return app


# ---------------------------------------------------------
# CLI Server Runner Entrypoint
# ---------------------------------------------------------

def main():
    """CLI launcher for the LCM API server."""
    parser = argparse.ArgumentParser(description="Launch OpenAI-compatible LCM API server.")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml", help="Path to experiment config YAML.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--model-id", type=str, default="lcm-primary", help="Exposed model identifier.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu).")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[*] Initializing LCM Server on device: {device}")

    # Generate synthetic world environment
    world_gen = WorldGenerator(base_seed=config.get("seed", 42))
    world = world_gen.generate_world(
        world_id=config.get("world", {}).get("world_id", "world0"),
        seed=config.get("seed", 42),
        num_entities=config.get("world", {}).get("num_entities", 20),
        num_facts=config.get("world", {}).get("num_facts", 25),
        held_out_lexicon=config.get("world", {}).get("held_out_lexicon", False)
    )
    print(f"[*] Loaded World ID: {world.world_id} with {len(world.documents)} documents.")

    # Initialize DeterministicShell
    shell = DeterministicShell(
        model=None,
        tokenizer=None,
        device=device,
        max_turns=config.get("agent", {}).get("max_turns", 12)
    )

    app = create_app(shell=shell, world=world, model_id=args.model_id)

    import uvicorn
    print(f"[*] Starting LCM API server at http://{args.host}:{args.port}")
    print(f"[*] Compatible with OpenAI endpoint: http://{args.host}:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
