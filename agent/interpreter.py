"""Native Python RDL Host Interpreter with sub-millisecond dispatch and HOP formatting (Milestone 2)."""

import time
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from agent.protocol import (
    parse_and_validate_message,
    ToolCallMessage,
    FinalMessage,
    ProtocolError,
    PlanMessage,
    format_search_hop,
    format_read_hop,
    format_math_hop,
    format_error_hop,
)
from agent.adapters.registry import AdapterRegistry
from agent.adapters.document import strip_query_quotes
from agent.tools.exec import RestrictedASTEvaluator


class RDLStepResult(BaseModel):
    """Result of executing a single RDL turn."""
    raw_statement: str
    is_action: bool = False
    is_final: bool = False
    is_error: bool = False
    action_data: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    cited_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    hop_observation: str = ""
    latency_ms: float = 0.0


class RDLInterpreter:
    """Fast, in-process execution engine for RDL actions and HOP observation generation."""

    def __init__(
        self,
        registry: Optional[AdapterRegistry] = None,
        default_doc_adapter: str = "docs",
        math_evaluator: Optional[RestrictedASTEvaluator] = None,
    ):
        self.registry = registry or AdapterRegistry()
        self.default_doc_adapter = default_doc_adapter
        self.math_evaluator = math_evaluator or RestrictedASTEvaluator()

    def execute(self, statement: str) -> RDLStepResult:
        """Executes a single RDL statement in <0.5ms and returns structured result + HOP observation."""
        t0 = time.perf_counter()
        parsed = parse_and_validate_message(statement)

        if isinstance(parsed, ProtocolError):
            latency = (time.perf_counter() - t0) * 1000.0
            hop_err = format_error_hop("INVALID_OPCODE", parsed.message)
            return RDLStepResult(
                raw_statement=statement,
                is_error=True,
                hop_observation=hop_err,
                latency_ms=round(latency, 4)
            )

        if isinstance(parsed, FinalMessage):
            latency = (time.perf_counter() - t0) * 1000.0
            return RDLStepResult(
                raw_statement=statement,
                is_final=True,
                final_answer=parsed.answer,
                cited_evidence=[e.model_dump() for e in parsed.evidence],
                hop_observation="",
                latency_ms=round(latency, 4)
            )

        if isinstance(parsed, ToolCallMessage):
            tool = parsed.tool
            args = parsed.arguments

            if tool == "search":
                raw_q = args.get("query", "")
                clean_q = strip_query_quotes(raw_q)
                limit = args.get("limit", 5)

                doc_adapter = self.registry.get(self.default_doc_adapter)
                if doc_adapter:
                    hits = doc_adapter.search(clean_q, limit=limit)
                else:
                    hits = []

                hop_obs = format_search_hop(hits)
                latency = (time.perf_counter() - t0) * 1000.0
                return RDLStepResult(
                    raw_statement=statement,
                    is_action=True,
                    action_data={"tool": "search", "arguments": {"query": clean_q, "limit": limit}, "results": hits},
                    hop_observation=hop_obs,
                    latency_ms=round(latency, 4)
                )

            elif tool == "read":
                doc_id = args.get("document_id", "")
                line_list = args.get("lines")
                line_tuple = (min(line_list), max(line_list)) if line_list else None

                doc_adapter = self.registry.get(self.default_doc_adapter)
                if doc_adapter:
                    slice_data = doc_adapter.read(doc_id, lines=line_tuple)
                else:
                    slice_data = None

                if slice_data is None:
                    hop_obs = f"OBS READ {doc_id} NOT_FOUND"
                else:
                    start_l = slice_data["start_line"]
                    end_l = slice_data["end_line"]
                    entries = [f"{doc_id}:L{l['line_number']} {l['text']}" for l in slice_data["lines"]]
                    header = f"OBS READ {doc_id} LINES {start_l}-{end_l}"
                    hop_obs = f"{header}\n" + "\n".join(entries) if entries else header

                latency = (time.perf_counter() - t0) * 1000.0
                return RDLStepResult(
                    raw_statement=statement,
                    is_action=True,
                    action_data={"tool": "read", "arguments": args, "slice": slice_data},
                    hop_observation=hop_obs,
                    latency_ms=round(latency, 4)
                )

            elif tool == "exec":
                code = args.get("code", "")
                inps = args.get("inputs")
                if inps:
                    res = self.math_evaluator.evaluate(code, inps)
                else:
                    res = self.math_evaluator.evaluate_pure_math(code)

                if res.get("status") == "success":
                    hop_obs = format_math_hop(res.get("result"))
                else:
                    hop_obs = format_math_hop(None, error=res.get("error_type", "ERROR"))

                latency = (time.perf_counter() - t0) * 1000.0
                return RDLStepResult(
                    raw_statement=statement,
                    is_action=True,
                    action_data={"tool": "exec", "arguments": args, "result": res},
                    hop_observation=hop_obs,
                    latency_ms=round(latency, 4)
                )

            elif tool == "filter":
                field = args.get("field", "")
                op = args.get("op", "EQ")
                val = args.get("value")
                tab_adapter = self.registry.get(args.get("table", "default"))
                records = tab_adapter.filter(field, op, val) if tab_adapter else []
                hop_obs = f"OBS FILTER [{len(records)} records]"
                latency = (time.perf_counter() - t0) * 1000.0
                return RDLStepResult(
                    raw_statement=statement,
                    is_action=True,
                    action_data={"tool": "filter", "arguments": args, "results": records},
                    hop_observation=hop_obs,
                    latency_ms=round(latency, 4)
                )

        # Fallback / PlanMessage
        latency = (time.perf_counter() - t0) * 1000.0
        return RDLStepResult(
            raw_statement=statement,
            is_action=False,
            hop_observation="",
            latency_ms=round(latency, 4)
        )
