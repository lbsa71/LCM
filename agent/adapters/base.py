"""Abstract base class for all multi-modal EvidenceProvider adapters."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class EvidenceProvider(ABC):
    """Universal interface for external verifiable knowledge sources."""

    @abstractmethod
    def introspect(self) -> Dict[str, Any]:
        """Returns self-describing metadata, type, and capabilities (__adapter_meta)."""
        pass

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Executes lexical or text-based search. Default raises NotImplementedError."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support text search.")

    def read(self, doc_id: str, lines: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
        """Reads document or item content with optional line slicing. Default raises NotImplementedError."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support line-addressed reading.")

    def filter(self, field: str, op: str, value: Any) -> List[Dict[str, Any]]:
        """Executes predicate filtering on structured records. Default raises NotImplementedError."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support relational filtering.")

    def vector_search(self, query_vec: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Executes dense vector similarity search. Default raises NotImplementedError."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support vector search.")
