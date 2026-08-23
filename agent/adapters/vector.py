"""Vector EvidenceProvider adapter for dense semantic embeddings."""

import math
from typing import Any, Dict, List, Optional
from agent.adapters.base import EvidenceProvider


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorEvidenceProvider(EvidenceProvider):
    """EvidenceProvider for dense vector collections with cosine similarity search."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.entries: List[Dict[str, Any]] = []

    def introspect(self) -> Dict[str, Any]:
        return {
            "type": "vector",
            "dimension": self.dimension,
            "capabilities": ["vector_search"],
            "total_vectors": len(self.entries),
        }

    def insert(self, entry_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None):
        """Inserts a vector entry."""
        self.entries.append({
            "id": entry_id,
            "vector": vector,
            "metadata": metadata or {}
        })

    def vector_search(self, query_vec: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Finds top-k nearest neighbors via cosine similarity."""
        scored = []
        for entry in self.entries:
            sim = _cosine_similarity(query_vec, entry["vector"])
            scored.append({
                "id": entry["id"],
                "score": round(sim, 4),
                "metadata": entry["metadata"]
            })
        scored.sort(key=lambda x: -x["score"])
        return scored[:k]
