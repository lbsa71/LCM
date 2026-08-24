"""Deterministic ReAct execution shell coordinating neural policy and tools."""

import json
import time
from typing import Any, Dict, List, Optional
import torch
from tokenizers import Tokenizer

from synth.ontology import World, Task
from agent.protocol import (
    parse_and_validate_message,
    PlanMessage,
    ToolCallMessage,
    FinalMessage,
    ProtocolError,
    format_search_hop,
    format_read_hop,
    format_math_hop,
    format_filter_hop,
    format_error_hop,
)
from agent.tools.search import DeterministicBM25Search
from agent.tools.read import DocumentReader
from agent.tools.exec import RestrictedASTEvaluator
from agent.adapters.document import DocumentEvidenceProvider, strip_query_quotes
from agent.adapters.registry import AdapterRegistry
from agent.interpreter import RDLInterpreter
from agent.state import AgentState
from training.model import SyntheticTransformer


class DeterministicShell:
    """ReAct execution runtime managing deterministic tools, state, and protocol validation."""

    def __init__(
        self,
        model: Optional[SyntheticTransformer] = None,
        tokenizer: Optional[Tokenizer] = None,
        device: torch.device = torch.device("cpu"),
        max_turns: int = 12,
        max_search_calls: int = 6,
        max_read_calls: int = 8,
        max_exec_calls: int = 4,
        max_filter_calls: int = 4,
        max_tokens_per_turn: int = 128
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_turns = max_turns
        self.max_search_calls = max_search_calls
        self.max_read_calls = max_read_calls
        self.max_exec_calls = max_exec_calls
        self.max_filter_calls = max_filter_calls
        self.max_tokens_per_turn = max_tokens_per_turn
        self.exec_evaluator = RestrictedASTEvaluator()

    def run_episode(self, world: World, task: Task) -> Dict[str, Any]:
        """Executes a full interactive episode for a task in a world."""
        state = AgentState(
            max_turns=self.max_turns,
            max_search_calls=self.max_search_calls,
            max_read_calls=self.max_read_calls,
            max_exec_calls=self.max_exec_calls,
            max_filter_calls=self.max_filter_calls
        )

        doc_adapter = DocumentEvidenceProvider(world)
        registry = AdapterRegistry()
        registry.register("docs", doc_adapter)
        interpreter = RDLInterpreter(registry=registry, default_doc_adapter="docs", math_evaluator=self.exec_evaluator)

        trace_steps = []
        start_time = time.time()

        # Initial User Message
        prompt_text = task.question
        state.history.append({"role": "user", "content": prompt_text})

        eos_id = self.tokenizer.token_to_id("<EOS>") if self.tokenizer else 2
        stop_ids = [eos_id] if eos_id is not None else []
        if self.tokenizer:
            for tag_name in ["<OBSERVATION>", "<USER>"]:
                tid = self.tokenizer.token_to_id(tag_name)
                if tid is not None and tid not in stop_ids:
                    stop_ids.append(tid)


        while not state.is_terminated:
            turn_err = state.increment_turn()
            if turn_err:
                break

            # 1. Format context for model
            raw_output = ""
            if self.model and self.tokenizer:
                bos_id = self.tokenizer.token_to_id("<BOS>")
                eos_id = self.tokenizer.token_to_id("<EOS>")
                input_ids = [bos_id] if bos_id is not None else []
                for h in state.history:
                    role_tag = f"<{h['role'].upper()}>"
                    tag_id = self.tokenizer.token_to_id(role_tag)
                    if tag_id is None:
                        tag_id = self.tokenizer.token_to_id("<UNK>")
                    input_ids.append(tag_id)
                    input_ids.extend(self.tokenizer.encode(h["content"]).ids)
                    if eos_id is not None:
                        input_ids.append(eos_id)


                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
                
                # Predict next action / plan / final
                generated = self.model.generate(
                    input_tensor,
                    max_new_tokens=self.max_tokens_per_turn,
                    stop_token_ids=stop_ids,
                    temperature=0.0
                )
                new_tokens = generated[0, input_tensor.shape[1]:].tolist()
                raw_output = self.tokenizer.decode(new_tokens)
            else:
                # Fallback / dummy if model not provided
                raw_output = json.dumps({"type": "final", "answer": "insufficient_evidence", "evidence": []})

            # Clean raw output
            cleaned_output = raw_output.replace("<EOS>", "").replace("<PAD>", "").strip()
            
            # 2. Parse and validate protocol message
            parsed = parse_and_validate_message(cleaned_output)

            step_record = {
                "turn": state.turn_count,
                "raw_output": cleaned_output,
                "parsed_type": parsed.type if hasattr(parsed, "type") else "protocol_error",
                "parsed_data": parsed.model_dump() if hasattr(parsed, "model_dump") else str(parsed)
            }

            if isinstance(parsed, ProtocolError):
                state.history.append({"role": "action", "content": cleaned_output})
                err_obs = format_error_hop("INVALID_OPCODE", parsed.message)
                state.history.append({"role": "observation", "content": err_obs})
                step_record["observation"] = {"status": "error", "error_type": "INVALID_OPCODE", "message": parsed.message, "hop": err_obs}
                trace_steps.append(step_record)
                continue

            elif isinstance(parsed, PlanMessage):
                state.history.append({"role": "plan", "content": cleaned_output})
                trace_steps.append(step_record)
                continue

            elif isinstance(parsed, ToolCallMessage):
                state.history.append({"role": "action", "content": cleaned_output})
                
                limit_err = state.record_tool_call(parsed.tool)
                if limit_err:
                    obs_str = format_error_hop(limit_err)
                    obs_data = {"status": "error", "error_type": limit_err, "message": f"Resource limit exceeded: {limit_err}"}
                else:
                    # Execute tool
                    if parsed.tool == "search":
                        q = parsed.arguments.get("query", "")
                        lim = parsed.arguments.get("limit", 5)
                        hits = doc_adapter.search(q, limit=lim)
                        obs_str = format_search_hop(hits)
                        obs_data = {"status": "success", "results": hits}
                    elif parsed.tool == "read":
                        d_id = parsed.arguments.get("document_id", "")
                        lines = parsed.arguments.get("lines")
                        line_tuple = (min(lines), max(lines)) if lines else None
                        slice_data = doc_adapter.read(d_id, lines=line_tuple)
                        if slice_data is None:
                            obs_str = f"OBS READ {d_id} NOT_FOUND"
                            obs_data = {"status": "error", "error_type": "DOCUMENT_NOT_FOUND"}
                        else:
                            start_l = slice_data["start_line"]
                            end_l = slice_data["end_line"]
                            entries = [f"{d_id}:L{l['line_number']} {l['text']}" for l in slice_data["lines"]]
                            header = f"OBS READ {d_id} LINES {start_l}-{end_l}"
                            obs_str = f"{header}\n" + "\n".join(entries) if entries else header
                            obs_data = {"status": "success", "document_id": d_id, "text": obs_str}
                    elif parsed.tool == "exec":
                        code = parsed.arguments.get("code", "")
                        inps = parsed.arguments.get("inputs")
                        if inps:
                            obs_data = self.exec_evaluator.evaluate(code, inps)
                        else:
                            obs_data = self.exec_evaluator.evaluate_pure_math(code)
                            if obs_data.get("status") == "error":
                                obs_data = self.exec_evaluator.evaluate(code)
                        if obs_data.get("status") == "success":
                            obs_str = format_math_hop(obs_data.get("result"))
                        else:
                            obs_str = format_math_hop(None, error=obs_data.get("error_type", "ERROR"))
                    elif parsed.tool == "filter":
                        f_field = parsed.arguments.get("field", "")
                        f_op = parsed.arguments.get("op", "EQ")
                        f_val = parsed.arguments.get("value")
                        tab_adapter = registry.get("table")
                        if tab_adapter:
                            records = tab_adapter.filter(f_field, f_op, f_val)
                            obs_str = format_filter_hop(records)
                            obs_data = {"status": "success", "results": records}
                        else:
                            obs_str = format_filter_hop([])
                            obs_data = {"status": "success", "results": []}
                    else:
                        obs_str = format_error_hop("UNKNOWN_TOOL")
                        obs_data = {"status": "error", "message": "Unknown tool"}

                state.history.append({"role": "observation", "content": obs_str})
                step_record["observation"] = obs_data
                trace_steps.append(step_record)
                continue

            elif isinstance(parsed, FinalMessage):
                state.is_terminated = True
                state.final_answer = parsed.answer
                state.cited_evidence = [e.model_dump() for e in parsed.evidence]
                state.history.append({"role": "final", "content": cleaned_output})
                trace_steps.append(step_record)
                break

        elapsed = time.time() - start_time
        return {
            "task_id": task.task_id,
            "world_id": world.world_id,
            "gold_answer": task.gold_answer,
            "model_answer": state.final_answer,
            "cited_evidence": state.cited_evidence,
            "turns_used": state.turn_count,
            "search_count": state.search_count,
            "read_count": state.read_count,
            "exec_count": state.exec_count,
            "is_terminated": state.is_terminated,
            "termination_reason": state.termination_reason,
            "elapsed_seconds": round(elapsed, 4),
            "trace_steps": trace_steps
        }
