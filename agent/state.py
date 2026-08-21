"""Agent state management and turn / resource limits (PRD Section 26)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    """Tracks execution state, tool budgets, history, and evidence."""
    max_turns: int = 12
    max_search_calls: int = 6
    max_read_calls: int = 8
    max_exec_calls: int = 4

    turn_count: int = 0
    search_count: int = 0
    read_count: int = 0
    exec_count: int = 0

    history: List[Dict[str, Any]] = field(default_factory=list)
    cited_evidence: List[Dict[str, Any]] = field(default_factory=list)
    
    is_terminated: bool = False
    termination_reason: Optional[str] = None
    final_answer: Optional[str] = None

    def increment_turn(self) -> Optional[str]:
        self.turn_count += 1
        if self.turn_count > self.max_turns:
            self.is_terminated = True
            self.termination_reason = "MAX_TURNS_EXCEEDED"
            return self.termination_reason
        return None

    def record_tool_call(self, tool_name: str) -> Optional[str]:
        if tool_name == "search":
            self.search_count += 1
            if self.search_count > self.max_search_calls:
                return "MAX_SEARCH_CALLS_EXCEEDED"
        elif tool_name == "read":
            self.read_count += 1
            if self.read_count > self.max_read_calls:
                return "MAX_READ_CALLS_EXCEEDED"
        elif tool_name == "exec":
            self.exec_count += 1
            if self.exec_count > self.max_exec_calls:
                return "MAX_EXEC_CALLS_EXCEEDED"
        return None
