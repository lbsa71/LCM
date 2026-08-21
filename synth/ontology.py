"""Ontology and core data models for procedural worlds, documents, facts, and proof graphs."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union


@dataclass
class Entity:
    """An entity within a procedural world."""
    id: str
    name: str
    entity_type: str  # settlement, region, artifact, device, event, record, station
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Fact:
    """A canonical fact within a world."""
    id: str
    subject_id: str
    relation: str  # population, inside, contains, connected_to, distance_to, measured_value, timestamp, status
    value: Union[str, int, float, bool, List[str]]
    is_contingent: bool = True  # True if randomized between worlds (Tier D), False if formal invariant (Tier B)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentLine:
    """A single line in a rendered document."""
    line_number: int
    text: str
    fact_ids: List[str] = field(default_factory=list)


@dataclass
class Document:
    """A rendered natural language document with line identifiers."""
    id: str  # e.g., "D01"
    title: str
    doc_type: str  # report, directory, registry, log, table
    lines: List[DocumentLine] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def formatted_text(self) -> str:
        """Returns the document formatted with deterministic line identifiers."""
        result = [f"# Document {self.id}: {self.title}"]
        for line in self.lines:
            result.append(f"{self.id}:L{line.line_number} {line.text}")
        return "\n".join(result)


@dataclass
class ProofNode:
    """A node in a proof graph supporting a ground-truth conclusion."""
    node_id: str
    description: str
    fact_id: Optional[str] = None
    document_id: Optional[str] = None
    line_numbers: List[int] = field(default_factory=list)
    children: List[ProofNode] = field(default_factory=list)


@dataclass
class ProofGraph:
    """Hidden ground truth proof graph mapping answer back to canonical facts & document lines."""
    goal: str
    root_nodes: List[ProofNode] = field(default_factory=list)
    required_document_lines: Dict[str, List[int]] = field(default_factory=dict)  # doc_id -> list of line numbers
    required_fact_ids: Set[str] = field(default_factory=set)

    def add_evidence(self, doc_id: str, line_no: int, fact_id: Optional[str] = None):
        if doc_id not in self.required_document_lines:
            self.required_document_lines[doc_id] = []
        if line_no not in self.required_document_lines[doc_id]:
            self.required_document_lines[doc_id].append(line_no)
        if fact_id:
            self.required_fact_ids.add(fact_id)


@dataclass
class World:
    """A canonical procedural world."""
    world_id: str
    seed: int
    entities: Dict[str, Entity] = field(default_factory=dict)
    facts: Dict[str, Fact] = field(default_factory=dict)
    documents: Dict[str, Document] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        for entity in self.entities.values():
            if entity.name.lower() == name.lower():
                return entity
        return None

    def get_facts_for_subject(self, subject_id: str) -> List[Fact]:
        return [f for f in self.facts.values() if f.subject_id == subject_id]

    def get_facts_by_relation(self, relation: str) -> List[Fact]:
        return [f for f in self.facts.values() if f.relation == relation]


@dataclass
class Task:
    """A formal evaluation or training task."""
    task_id: str
    task_type: str
    suite: str  # suite_a_language, suite_b_invariants, suite_c_single_hop, etc.
    question: str
    gold_answer: str
    proof_graph: ProofGraph
    world_id: str
    is_retrieval_required: bool = True
    is_contingent: bool = True
    is_insufficient_evidence: bool = False
    context_text: Optional[str] = None  # For non-retrieval context / language / invariant tasks
    oracle_trajectory: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
