"""Renders world facts into line-addressed documents with ground-truth provenance."""

from __future__ import annotations
import random
from typing import Dict, List, Tuple, Optional
from synth.ontology import World, Document, DocumentLine, Fact
from synth.language.grammar import GrammarRenderer


class DocumentGenerator:
    """Partitions world facts into multiple line-numbered documents with distractors."""

    def __init__(self, template_set: str = "all"):
        self.grammar = GrammarRenderer(template_set=template_set)

    def generate_documents(
        self,
        world: World,
        docs_per_world: int = 10,
        distractors_per_world: int = 3,
        rng: Optional[random.Random] = None
    ) -> Dict[str, Document]:
        """Generates line-addressed documents and updates world.documents."""
        if rng is None:
            rng = random.Random(world.seed)

        documents: Dict[str, Document] = {}
        all_facts = list(world.facts.values())
        rng.shuffle(all_facts)

        # Categorize facts by theme
        pop_facts = [f for f in all_facts if f.relation == "population"]
        loc_facts = [f for f in all_facts if f.relation in ("inside", "distance_to")]
        telemetry_facts = [f for f in all_facts if f.relation in ("measured_value", "status")]
        event_facts = [f for f in all_facts if f.relation == "timestamp"]

        doc_counter = 1

        def create_doc(title: str, doc_type: str, facts: List[Fact]) -> Document:
            nonlocal doc_counter
            doc_id = f"D{doc_counter:02d}"
            doc_counter += 1

            lines = []
            for line_idx, fact in enumerate(facts, start=1):
                text = self.grammar.render_fact(fact, world.entities, rng)
                lines.append(DocumentLine(
                    line_number=line_idx,
                    text=text,
                    fact_ids=[fact.id]
                ))

            doc = Document(
                id=doc_id,
                title=title,
                doc_type=doc_type,
                lines=lines,
                metadata={"num_facts": len(facts)}
            )
            return doc

        # 1. Demographic Registry (chunked)
        for i in range(0, len(pop_facts), 3):
            chunk = pop_facts[i:i+3]
            if chunk:
                d = create_doc(f"Demographic Survey Record Part {i//3 + 1}", "registry", chunk)
                documents[d.id] = d

        # 2. Territorial Atlas
        for i in range(0, len(loc_facts), 3):
            chunk = loc_facts[i:i+3]
            if chunk:
                d = create_doc(f"Regional Territorial Atlas Chapter {i//3 + 1}", "atlas", chunk)
                documents[d.id] = d

        # 3. Telemetry Log
        for i in range(0, len(telemetry_facts), 3):
            chunk = telemetry_facts[i:i+3]
            if chunk:
                d = create_doc(f"Telemetry Status Log Batch {i//3 + 1}", "log", chunk)
                documents[d.id] = d

        # 4. Chronological Events
        for i in range(0, len(event_facts), 3):
            chunk = event_facts[i:i+3]
            if chunk:
                d = create_doc(f"Historical Chronicle Volume {i//3 + 1}", "chronicle", chunk)
                documents[d.id] = d

        # 5. Distractor Documents (Irrelevant synthetic records with overlapping keywords)
        for dist_idx in range(1, distractors_per_world + 1):
            doc_id = f"D{doc_counter:02d}"
            doc_counter += 1
            lines = [
                DocumentLine(
                    line_number=1,
                    text=f"Archive header index {rng.randint(100, 999)} for auxiliary maintenance.",
                    fact_ids=[]
                ),
                DocumentLine(
                    line_number=2,
                    text=f"Routine monitoring cycle {rng.randint(1000, 9999)} completed without parameter changes.",
                    fact_ids=[]
                ),
                DocumentLine(
                    line_number=3,
                    text=f"Auxiliary sector review reports standard ambient levels across all quadrants.",
                    fact_ids=[]
                )
            ]
            doc = Document(
                id=doc_id,
                title=f"Auxiliary Archival Digest {dist_idx}",
                doc_type="digest",
                lines=lines,
                metadata={"is_distractor": True}
            )
            documents[doc.id] = doc

        world.documents = documents
        return documents
