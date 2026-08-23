"""Unit tests for Multi-Modal EvidenceProvider adapters and AdapterRegistry (Milestone 2)."""

import pytest
from synth.ontology import World, Document, DocumentLine
from agent.adapters.base import EvidenceProvider
from agent.adapters.document import DocumentEvidenceProvider
from agent.adapters.tabular import TabularEvidenceProvider
from agent.adapters.vector import VectorEvidenceProvider
from agent.adapters.registry import AdapterRegistry


def create_sample_world():
    world = World(world_id="W_TEST_ADAPTERS", seed=42)
    world.documents["D01"] = Document(
        id="D01",
        title="Census Fort Valerius",
        doc_type="census",
        lines=[
            DocumentLine(line_number=1, text="Valerius Fort garrison registry.", fact_ids=[]),
            DocumentLine(line_number=2, text="Active infantry stationed: 140.", fact_ids=[]),
            DocumentLine(line_number=3, text="Location: Northern valley.", fact_ids=[]),
        ]
    )
    return world


def test_document_adapter_search_quote_stripping():
    """Verify DocumentEvidenceProvider strips enclosing quotes on search query before BM25."""
    world = create_sample_world()
    adapter = DocumentEvidenceProvider(world)

    # Both quoted and unquoted search should yield valid results
    res_quoted = adapter.search('"Valerius Fort"', limit=2)
    res_unquoted = adapter.search("Valerius Fort", limit=2)

    assert len(res_quoted) > 0
    assert len(res_unquoted) > 0
    assert res_quoted[0]["document_id"] == "D01"
    assert res_unquoted[0]["document_id"] == "D01"


def test_document_adapter_line_indexing():
    """Verify DocumentEvidenceProvider handles 1-indexed line extraction strictly."""
    world = create_sample_world()
    adapter = DocumentEvidenceProvider(world)

    # 1-indexed LINES 1-2
    slice_1_2 = adapter.read("D01", lines=(1, 2))
    assert slice_1_2 is not None
    assert len(slice_1_2["lines"]) == 2
    assert slice_1_2["lines"][0]["line_number"] == 1
    assert slice_1_2["lines"][1]["line_number"] == 2
    assert "Valerius Fort garrison registry." in slice_1_2["lines"][0]["text"]

    # 1-indexed LINES 2-3
    slice_2_3 = adapter.read("D01", lines=(2, 3))
    assert slice_2_3 is not None
    assert len(slice_2_3["lines"]) == 2
    assert slice_2_3["lines"][0]["line_number"] == 2
    assert slice_2_3["lines"][1]["line_number"] == 3


def test_tabular_adapter_filtering():
    """Verify TabularEvidenceProvider structured filtering."""
    schema = {"id": "int", "name": "str", "garrison": "int", "status": "str"}
    records = [
        {"id": 1, "name": "Fort Valerius", "garrison": 140, "status": "active"},
        {"id": 2, "name": "Fort Albia", "garrison": 210, "status": "active"},
        {"id": 3, "name": "Outpost Corvath", "garrison": 45, "status": "dormant"},
    ]
    tab_adapter = TabularEvidenceProvider("garrisons", schema, records)

    # Filter GT
    gt_res = tab_adapter.filter("garrison", "GT", 100)
    assert len(gt_res) == 2

    # Filter EQ
    eq_res = tab_adapter.filter("status", "EQ", "active")
    assert len(eq_res) == 2

    # Filter CONTAINS
    cont_res = tab_adapter.filter("name", "CONTAINS", "Fort")
    assert len(cont_res) == 2


def test_vector_adapter_semantic_search():
    """Verify VectorEvidenceProvider cosine similarity ranking."""
    # 3-dim mock vectors
    vec_adapter = VectorEvidenceProvider(dimension=3)
    vec_adapter.insert("doc_alpha", [1.0, 0.0, 0.0], {"title": "Alpha Document"})
    vec_adapter.insert("doc_beta", [0.0, 1.0, 0.0], {"title": "Beta Document"})
    vec_adapter.insert("doc_gamma", [0.7, 0.7, 0.0], {"title": "Gamma Document"})

    # Query close to [1.0, 0.1, 0.0]
    hits = vec_adapter.vector_search([1.0, 0.1, 0.0], k=2)
    assert len(hits) == 2
    assert hits[0]["id"] == "doc_alpha"


def test_adapter_registry_introspection():
    """Verify AdapterRegistry dynamically builds self-describing __adapter_meta table."""
    world = create_sample_world()
    doc_adapter = DocumentEvidenceProvider(world)
    tab_adapter = TabularEvidenceProvider("troops", {"id": "int", "count": "int"}, [{"id": 1, "count": 140}])
    
    registry = AdapterRegistry()
    registry.register("docs", doc_adapter)
    registry.register("troops", tab_adapter)

    meta = registry.introspect()
    assert "adapters" in meta
    assert "docs" in meta["adapters"]
    assert "troops" in meta["adapters"]
    assert meta["adapters"]["docs"]["type"] == "document"
    assert meta["adapters"]["troops"]["type"] == "tabular"
