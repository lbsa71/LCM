"""Multi-modal adapter registry coordinating all EvidenceProviders with __adapter_meta introspection."""

from typing import Any, Dict, Optional
from agent.adapters.base import EvidenceProvider


class AdapterRegistry:
    """Central registry and router for multi-modal EvidenceProviders."""

    def __init__(self):
        self._adapters: Dict[str, EvidenceProvider] = {}

    def register(self, name: str, adapter: EvidenceProvider):
        """Registers a named adapter."""
        self._adapters[name] = adapter

    def get(self, name: str) -> Optional[EvidenceProvider]:
        """Retrieves a named adapter."""
        return self._adapters.get(name)

    def introspect(self) -> Dict[str, Any]:
        """Dynamically builds self-describing __adapter_meta registry table."""
        meta: Dict[str, Any] = {
            "version": "1.0",
            "adapters": {}
        }
        for name, adapter in self._adapters.items():
            meta["adapters"][name] = adapter.introspect()
        return meta

    @property
    def adapters(self) -> Dict[str, EvidenceProvider]:
        return self._adapters
