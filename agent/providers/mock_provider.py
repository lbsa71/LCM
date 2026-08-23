"""Mock in-memory EvidenceProvider for isolated unit testing and AST validation."""

from typing import Any, Dict, List, Optional, Tuple
from agent.adapters.base import EvidenceProvider


class MockEvidenceProvider(EvidenceProvider):
    """Deterministic in-memory mock provider with pre-seeded documents and tables."""

    def __init__(
        self,
        docs: Optional[Dict[str, List[str]]] = None,
        records: Optional[List[Dict[str, Any]]] = None
    ):
        self.docs: Dict[str, List[str]] = docs or {
            "D01": [
                "Fort Valerius registry record.",
                "Active infantry stationed: 140.",
                "Stationed in Northern valley."
            ],
            "D02": [
                "Fort Albia garrison report.",
                "Total standing troops: 210.",
                "Southern boundary watch."
            ]
        }
        self.records: List[Dict[str, Any]] = records or [
            {"id": 1, "name": "Fort Valerius", "garrison": 140, "status": "active"},
            {"id": 2, "name": "Fort Albia", "garrison": 210, "status": "active"},
            {"id": 3, "name": "Outpost Corvath", "garrison": 45, "status": "dormant"}
        ]

    def introspect(self) -> Dict[str, Any]:
        return {
            "type": "mock",
            "capabilities": ["search", "read", "filter"],
            "total_documents": len(self.docs),
            "doc_ids": sorted(list(self.docs.keys())),
            "total_records": len(self.records)
        }

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        clean_q = query.strip().strip("\"'").lower()
        results = []
        for doc_id, lines in sorted(self.docs.items()):
            full_text = " ".join(lines).lower()
            if clean_q in full_text:
                score = 5.0 + full_text.count(clean_q)
                results.append({
                    "document_id": doc_id,
                    "score": round(score, 2),
                    "snippet": lines[0] if lines else ""
                })
        results.sort(key=lambda x: -x["score"])
        return results[:limit]

    def read(self, doc_id: str, lines: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
        clean_id = doc_id.strip().strip("\"'")
        if clean_id not in self.docs:
            return None
        doc_lines = self.docs[clean_id]
        if lines is not None:
            start_l, end_l = lines
            s_idx = max(0, start_l - 1)
            e_idx = min(len(doc_lines), end_l)
            sel_lines = doc_lines[s_idx:e_idx]
        else:
            start_l = 1
            end_l = len(doc_lines)
            sel_lines = doc_lines

        formatted = [
            {"line_number": start_l + idx, "text": text}
            for idx, text in enumerate(sel_lines)
        ]
        return {
            "document_id": clean_id,
            "title": f"Mock Document {clean_id}",
            "start_line": start_l,
            "end_line": end_l,
            "lines": formatted
        }

    def filter(self, field: str, op: str, value: Any) -> List[Dict[str, Any]]:
        op_upper = op.upper()
        results = []
        for row in self.records:
            if field not in row:
                continue
            r_val = row[field]
            if op_upper == "EQ" and r_val == value:
                results.append(row)
            elif op_upper == "GT" and r_val > value:
                results.append(row)
            elif op_upper == "LT" and r_val < value:
                results.append(row)
            elif op_upper == "CONTAINS" and str(value).lower() in str(r_val).lower():
                results.append(row)
        return results
