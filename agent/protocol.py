"""Deterministic JSON protocol definitions and schema validation (PRD Section 24)."""

import json
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, ValidationError


class PlanMessage(BaseModel):
    """Structured plan step."""
    type: Literal["plan"] = "plan"
    goal: str = Field(..., description="High-level target goal")
    needs: List[str] = Field(default_factory=list, description="Required information pieces")
    next_action: Literal["search", "read", "exec", "final"] = Field(..., description="Next chosen step")


class SearchArgs(BaseModel):
    query: str
    limit: Optional[int] = 5


class ReadArgs(BaseModel):
    document_id: str


class ExecArgs(BaseModel):
    code: str
    inputs: Optional[Dict[str, Any]] = None


class ToolCallMessage(BaseModel):
    """Action / Tool invocation step."""
    type: Literal["tool_call"] = "tool_call"
    tool: Literal["search", "read", "exec"]
    arguments: Dict[str, Any]


class EvidenceRef(BaseModel):
    """Cited evidence provenance."""
    document_id: str
    lines: List[int] = Field(default_factory=list)


class FinalMessage(BaseModel):
    """Final answer step with evidence provenance."""
    type: Literal["final"] = "final"
    answer: str
    evidence: List[EvidenceRef] = Field(default_factory=list)


class ProtocolError(BaseModel):
    """Deterministic protocol error returned to model upon malformed output."""
    status: Literal["error"] = "error"
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None


def _validate_single_dict(data: dict) -> Union[PlanMessage, ToolCallMessage, FinalMessage, ProtocolError]:
    """Validates a single parsed dictionary against the protocol schemas."""
    msg_type = data.get("type")
    try:
        if msg_type == "plan":
            return PlanMessage(**data)
        elif msg_type == "tool_call":
            tool_name = data.get("tool")
            args = data.get("arguments", {})
            if tool_name == "search":
                SearchArgs(**args)
            elif tool_name == "read":
                ReadArgs(**args)
            elif tool_name == "exec":
                ExecArgs(**args)
            else:
                return ProtocolError(
                    error_type="UNKNOWN_TOOL",
                    message=f"Unknown tool '{tool_name}'. Allowed tools: search, read, exec."
                )
            return ToolCallMessage(**data)
        elif msg_type == "final":
            return FinalMessage(**data)
        else:
            return ProtocolError(
                error_type="UNKNOWN_MESSAGE_TYPE",
                message=f"Message type '{msg_type}' is unrecognized. Allowed: plan, tool_call, final."
            )
    except ValidationError as ve:
        return ProtocolError(
            error_type="SCHEMA_VALIDATION_ERROR",
            message=f"Protocol schema validation failed: {str(ve)}",
            details={"errors": ve.errors()}
        )


def extract_json_objects(text: str) -> List[dict]:
    """Extracts all top-level JSON objects found in a string."""
    decoder = json.JSONDecoder()
    results = []
    idx = 0
    text_len = len(text)
    while idx < text_len:
        start = text.find("{", idx)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, idx=start)
            if isinstance(obj, dict):
                results.append(obj)
            idx = max(end, start + 1)
        except Exception:
            idx = start + 1
    return results


def parse_and_validate_message(raw_text: str) -> Union[PlanMessage, ToolCallMessage, FinalMessage, ProtocolError]:
    """Parses raw text into one of the allowed protocol messages or returns a ProtocolError."""
    cleaned = raw_text.strip()
    for tag in ["<PLAN>", "</PLAN>", "<ACTION>", "</ACTION>", "<FINAL>", "</FINAL>", "<TOOL>", "</TOOL>", "<OBSERVATION>", "</OBSERVATION>", "```json", "```"]:
        cleaned = cleaned.replace(tag, "").strip()

    # 1. Attempt exact parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return _validate_single_dict(data)
    except Exception:
        pass

    # 2. Resilient multi-object / substring extraction
    extracted_objects = extract_json_objects(cleaned)
    if not extracted_objects:
        return ProtocolError(
            error_type="JSON_DECODE_ERROR",
            message="Model output contains no valid JSON objects.",
            details={"raw_text": raw_text[:200]}
        )

    # If multiple JSON objects are present, prioritize executable messages (final > tool_call > plan)
    valid_messages = []
    last_error = None
    for obj in extracted_objects:
        res = _validate_single_dict(obj)
        if isinstance(res, (FinalMessage, ToolCallMessage, PlanMessage)):
            valid_messages.append(res)
        else:
            last_error = res

    if valid_messages:
        # Prefer FinalMessage > ToolCallMessage > PlanMessage
        for vm in valid_messages:
            if isinstance(vm, FinalMessage):
                return vm
        for vm in valid_messages:
            if isinstance(vm, ToolCallMessage):
                return vm
        return valid_messages[0]

    return last_error or ProtocolError(
        error_type="INVALID_SCHEMA",
        message="Top-level protocol message must be a valid JSON object.",
        details={"raw_text": raw_text[:200]}
    )

