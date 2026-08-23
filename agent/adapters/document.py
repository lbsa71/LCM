"""Document EvidenceProvider adapter wrapping lexical BM25 and line-addressed reading."""

from typing import Any, Dict, List, Optional, Tuple
from synth.ontology import World, Document
from agent.adapters.base import EvidenceProvider
from agent.tools.search import DeterministicBM25Search


def strip_query_quotes(query: str) -> str:
    """Strips enclosing quotes from search query strings."""
    stripped = query.strip()
    if (stripped.startswith('"') and stripped.endswith('"')) or (stripped.startswith("'") and stripped.endswith("'")):
        return stripped[1:-1].strip()
    return stripped


class DocumentEvidenceProvider(EvidenceProvider):
    """EvidenceProvider for unstructured text documents with BM25 indexing and line addressing."""

    def __init__(self, world: World):
        self.world = world
        self.bm25 = DeterministicBM25Search(world)

    def introspect(self) -> Dict[str, Any]:
        return {
            "type": "document",
            "capabilities": ["search", "read"],
            "total_documents": len(self.world.documents),
            "doc_ids": sorted(list(self.world.documents.keys())),
        }

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Executes BM25 search after ensuring query quotes are cleanly stripped."""
        clean_query = strip_query_quotes(query)
        res = self.bm25.search(clean_query, limit=limit)
        return res.get("results", [])

    def read(self, doc_id: str, lines: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
        """Reads document content with strict 1-indexed to 0-indexed line alignment."""
        doc = self.world.documents.get(doc_id)
        if doc is None:
            return None

        doc_lines = doc.lines
        if lines is not None:
            start_line, end_line = lines
            # 1-indexed to 0-indexed array slicing
            start_idx = max(0, start_line - 1)
            end_idx = min(len(doc_lines), end_line)
            selected_lines = doc_lines[start_idx:end_idx]
        else:
            start_line = 1
            end_line = len(doc_lines) if doc_lines else 1
            selected_lines = doc_lines

        formatted_lines = [
            {
                "line_number": getattr(l, "line_number", idx + start_line),
                "text": getattr(l, "text", str(l)),
            }
            for idx, l in enumerate(selected_lines)
        ]

        return {
            "document_id": doc.id,
            "title": doc.title,
            "start_line": start_line,
            "end_line": end_line,
            "lines": formatted_lines,
        }
