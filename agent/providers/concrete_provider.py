"""Concrete EvidenceProvider reading directly from procedural corpus world or dataset on disk."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from synth.ontology import World, Document, DocumentLine
from agent.adapters.base import EvidenceProvider
from agent.adapters.document import DocumentEvidenceProvider
from agent.adapters.tabular import TabularEvidenceProvider
from agent.providers.mock_provider import MockEvidenceProvider


class ConcreteCorpusProvider(EvidenceProvider):
    """Concrete provider backed by an actual synthetic World or disk corpus JSON."""

    def __init__(self, world_or_path: Union[World, str]):
        if isinstance(world_or_path, str):
            with open(world_or_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.world = World(world_id=data["world_id"], seed=data["seed"])
            for d_id, d_data in data.get("documents", {}).items():
                lines = [
                    DocumentLine(line_number=l["line_number"], text=l["text"], fact_ids=l.get("fact_ids", []))
                    for l in d_data.get("lines", [])
                ]
                doc = Document(id=d_data["id"], title=d_data["title"], doc_type=d_data["doc_type"], lines=lines)
                self.world.documents[d_id] = doc
        else:
            self.world = world_or_path

        self._doc_adapter = DocumentEvidenceProvider(self.world)

    def introspect(self) -> Dict[str, Any]:
        return {
            "type": "concrete_corpus",
            "world_id": self.world.world_id,
            "capabilities": ["search", "read", "filter"],
            "total_documents": len(self.world.documents),
            "doc_ids": sorted(list(self.world.documents.keys()))
        }

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        return self._doc_adapter.search(query, limit=limit)

    def read(self, doc_id: str, lines: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
        return self._doc_adapter.read(doc_id, lines=lines)

    def filter(self, field: str, op: str, value: Any) -> List[Dict[str, Any]]:
        # Filter across all entity properties in the concrete world
        results = []
        op_upper = op.upper()
        for e in self.world.entities.values():
            e_dict = {"id": e.id, "name": e.name, "type": e.entity_type, **e.properties}
            if field in e_dict:
                val = e_dict[field]
                if op_upper == "EQ" and val == value:
                    results.append(e_dict)
                elif op_upper == "GT" and val > value:
                    results.append(e_dict)
                elif op_upper == "LT" and val < value:
                    results.append(e_dict)
                elif op_upper == "CONTAINS" and str(value).lower() in str(val).lower():
                    results.append(e_dict)
        return results


def get_provider(mode: str = "mock", world: Optional[World] = None) -> EvidenceProvider:
    """Factory retrieving mock or concrete EvidenceProvider based on runtime mode."""
    if mode == "mock":
        return MockEvidenceProvider()
    elif mode == "concrete" or mode == "real":
        if world is None:
            raise ValueError("Concrete provider requires an active World instance.")
        return ConcreteCorpusProvider(world)
    else:
        raise ValueError(f"Unknown provider mode '{mode}'. Allowed: mock, concrete, real.")
