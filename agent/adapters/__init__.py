"""Multi-modal EvidenceProvider adapter framework (Milestone 2)."""

from agent.adapters.base import EvidenceProvider
from agent.adapters.document import DocumentEvidenceProvider
from agent.adapters.tabular import TabularEvidenceProvider
from agent.adapters.vector import VectorEvidenceProvider
from agent.adapters.registry import AdapterRegistry

__all__ = [
    "EvidenceProvider",
    "DocumentEvidenceProvider",
    "TabularEvidenceProvider",
    "VectorEvidenceProvider",
    "AdapterRegistry",
]
