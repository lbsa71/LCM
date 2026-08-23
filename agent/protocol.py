"""Deterministic JSON, RDL, and Host Observation Protocol (HOP) definitions and schema validation (PRD Section 24, specs/RDL_SPEC.md)."""

import json
import re
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
    lines: Optional[List[int]] = None


class ExecArgs(BaseModel):
    code: str
    inputs: Optional[Dict[str, Any]] = None


class ToolCallMessage(BaseModel):
    """Action / Tool invocation step."""
    type: Literal["tool_call"] = "tool_call"
    tool: Literal["search", "read", "exec", "filter"]
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
            elif tool_name == "filter":
                pass
            else:
                return ProtocolError(
                    error_type="UNKNOWN_TOOL",
                    message=f"Unknown tool '{tool_name}'. Allowed tools: search, read, exec, filter."
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


def _parse_citations(cite_str: str) -> List[EvidenceRef]:
    """Extracts document and line citations from a bracketed citation string (e.g. 'D01:2, D04:12')."""
    if not cite_str or not cite_str.strip():
        return []
    doc_lines_map: Dict[str, List[int]] = {}
    for match in re.finditer(r'([A-Za-z0-9_]+)\s*:\s*(\d+)', cite_str):
        doc_id = match.group(1).strip()
        line_no = int(match.group(2))
        if doc_id not in doc_lines_map:
            doc_lines_map[doc_id] = []
        if line_no not in doc_lines_map[doc_id]:
            doc_lines_map[doc_id].append(line_no)
    return [EvidenceRef(document_id=d_id, lines=lines) for d_id, lines in doc_lines_map.items()]


def parse_rdl_message(text: str) -> Optional[Union[ToolCallMessage, FinalMessage, ProtocolError]]:
    """Parses a raw text string into a structured RDL ToolCallMessage or FinalMessage if matching RDL grammar."""
    stripped = text.strip()
    if not stripped:
        return None

    # 1. SEARCH "<query>" [LIMIT <k>]
    search_match = re.match(r"^SEARCH\s+(.+?)(?:\s+LIMIT\s+(\d+))?$", stripped, re.IGNORECASE | re.DOTALL)
    if search_match:
        query = search_match.group(1).strip()
        if (query.startswith('"') and query.endswith('"')) or (query.startswith("'") and query.endswith("'")):
            query = query[1:-1].strip()
        limit = int(search_match.group(2)) if search_match.group(2) else 5
        return ToolCallMessage(
            tool="search",
            arguments={"query": query, "limit": limit}
        )

    # 2. READ <doc_id> [LINES <start>-<end>]
    read_match = re.match(r"^READ\s+([A-Za-z0-9_\"]+)(?:\s+LINES\s+(\d+)\s*(?:-|(?:\.\.))\s*(\d+))?$", stripped, re.IGNORECASE)
    if read_match:
        doc_id = read_match.group(1).strip().strip('"\'')
        args: Dict[str, Any] = {"document_id": doc_id}
        if read_match.group(2) and read_match.group(3):
            start = int(read_match.group(2))
            end = int(read_match.group(3))
            args["lines"] = list(range(start, end + 1))
        return ToolCallMessage(
            tool="read",
            arguments=args
        )

    # 3. MATH <infix_expr>
    math_match = re.match(r"^MATH\s+(.+)$", stripped, re.IGNORECASE | re.DOTALL)
    if math_match:
        code = math_match.group(1).strip()
        return ToolCallMessage(
            tool="exec",
            arguments={"code": code}
        )

    # 4. EMIT <answer> EVIDENCE [<citations>]
    emit_match = re.match(r"^EMIT\s+(.+?)(?:\s+EVIDENCE\s*\[(.*?)\])?$", stripped, re.IGNORECASE | re.DOTALL)
    if emit_match:
        answer_str = emit_match.group(1).strip()
        if (answer_str.startswith('"') and answer_str.endswith('"')) or (answer_str.startswith("'") and answer_str.endswith("'")):
            answer_str = answer_str[1:-1].strip()
        cites_raw = emit_match.group(2) if emit_match.group(2) is not None else ""
        evidence_refs = _parse_citations(cites_raw)
        return FinalMessage(
            answer=answer_str,
            evidence=evidence_refs
        )

    # 5. ABSTAIN [REASON <reason>]
    abstain_match = re.match(r"^ABSTAIN(?:\s+(?:\[?\s*REASON\s+)?([A-Za-z0-9_]+)\]?)?$", stripped, re.IGNORECASE)
    if abstain_match:
        reason = abstain_match.group(1)
        reason_str = reason.strip() if reason else "insufficient_evidence"
        return FinalMessage(
            answer=reason_str,
            evidence=[]
        )

    return None


def format_search_hop(results: List[Dict[str, Any]]) -> str:
    """Formats search results according to the Host Observation Protocol (HOP)."""
    if not results:
        return "OBS SEARCH EMPTY"
    hits = [f"{r['document_id']} ({r.get('score', 0.0)})" for r in results]
    return f"OBS SEARCH [{', '.join(hits)}]"


def format_read_hop(doc: Optional[Any], lines: Optional[List[int]] = None, doc_id: str = "") -> str:
    """Formats document read observations according to the Host Observation Protocol (HOP)."""
    if doc is None:
        return f"OBS READ {doc_id} NOT_FOUND"
    
    actual_doc_id = getattr(doc, "id", doc_id)
    doc_lines = getattr(doc, "lines", [])
    
    if lines:
        start_line = min(lines)
        end_line = max(lines)
        selected_lines = [l for l in doc_lines if getattr(l, "line_number", 0) in lines]
    else:
        start_line = 1
        end_line = len(doc_lines) if doc_lines else 1
        selected_lines = doc_lines

    line_entries = "\n".join(
        [f"{actual_doc_id}:L{getattr(l, 'line_number', idx + 1)} {getattr(l, 'text', str(l))}" for idx, l in enumerate(selected_lines)]
    )
    header = f"OBS READ {actual_doc_id} LINES {start_line}-{end_line}"
    return f"{header}\n{line_entries}" if line_entries else header


def format_math_hop(result: Any, error: Optional[str] = None) -> str:
    """Formats math computation observations according to the Host Observation Protocol (HOP)."""
    if error:
        return f"OBS MATH ERROR {error}"
    return f"OBS MATH {result}"


def format_error_hop(error_code: str, message: Optional[str] = None) -> str:
    """Formats protocol/resource errors according to the Host Observation Protocol (HOP)."""
    if message:
        return f'OBS ERROR {error_code} "{message}"'
    return f"OBS ERROR {error_code}"


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
    """Parses raw text into one of the allowed protocol messages (RDL or JSON) or returns a ProtocolError."""
    cleaned = raw_text.strip()
    for tag in ["<PLAN>", "</PLAN>", "<ACTION>", "</ACTION>", "<FINAL>", "</FINAL>", "<TOOL>", "</TOOL>", "<OBSERVATION>", "</OBSERVATION>", "```json", "```rdl", "```"]:
        cleaned = cleaned.replace(tag, "").strip()

    # 1. Attempt RDL parse
    rdl_parsed = parse_rdl_message(cleaned)
    if rdl_parsed is not None:
        return rdl_parsed

    # 2. Attempt exact JSON parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return _validate_single_dict(data)
    except Exception:
        pass

    # 3. Resilient multi-object / substring extraction
    extracted_objects = extract_json_objects(cleaned)
    if not extracted_objects:
        return ProtocolError(
            error_type="PARSE_ERROR",
            message="Model output contains neither valid RDL statements nor JSON protocol objects.",
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
        message="Top-level protocol message must be a valid RDL statement or JSON object.",
        details={"raw_text": raw_text[:200]}
    )
