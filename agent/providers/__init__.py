"""EvidenceProvider concrete and mock providers."""

from agent.adapters.base import EvidenceProvider
from agent.adapters.document import DocumentEvidenceProvider
from agent.adapters.tabular import TabularEvidenceProvider
from agent.adapters.vector import VectorEvidenceProvider
from agent.providers.mock_provider import MockEvidenceProvider
from agent.providers.concrete_provider import ConcreteCorpusProvider, get_provider

__all__ = [
    "EvidenceProvider",
    "DocumentEvidenceProvider",
    "TabularEvidenceProvider",
    "VectorEvidenceProvider",
    "MockEvidenceProvider",
    "ConcreteCorpusProvider",
    "get_provider",
]
