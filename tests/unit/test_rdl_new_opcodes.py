"""Unit tests for RDL opcodes (FILTER, DOCSLICE, MATH, EMIT, ABSTAIN) and providers."""

import pytest
from agent.protocol import (
    parse_and_validate_message,
    ToolCallMessage,
    FinalMessage,
    ProtocolError,
    format_filter_hop,
    format_search_hop,
    format_read_hop,
    format_math_hop,
)
from agent.providers import MockEvidenceProvider, ConcreteCorpusProvider, get_provider
from synth.ontology import World, Document, DocumentLine


def test_parse_rdl_filter_statement():
    """Verify FILTER opcode parsing with various relational operators."""
    # 1. FILTER EQ string
    msg1 = parse_and_validate_message('FILTER status EQ "active"')
    assert isinstance(msg1, ToolCallMessage)
    assert msg1.tool == "filter"
    assert msg1.arguments == {"field": "status", "op": "EQ", "value": "active"}

    # 2. FILTER GT numeric
    msg2 = parse_and_validate_message('FILTER garrison GT 100')
    assert isinstance(msg2, ToolCallMessage)
    assert msg2.tool == "filter"
    assert msg2.arguments == {"field": "garrison", "op": "GT", "value": 100}

    # 3. FILTER CONTAINS
    msg3 = parse_and_validate_message('FILTER name CONTAINS "Fort"')
    assert isinstance(msg3, ToolCallMessage)
    assert msg3.tool == "filter"
    assert msg3.arguments == {"field": "name", "op": "CONTAINS", "value": "Fort"}


def test_format_filter_hop():
    """Verify Host Observation Protocol formatting for FILTER."""
    records = [{"id": 1, "name": "Fort Valerius"}, {"id": 2, "name": "Fort Albia"}]
    obs = format_filter_hop(records)
    assert obs == "OBS FILTER [2 records]"

    obs_empty = format_filter_hop([])
    assert obs_empty == "OBS FILTER EMPTY"


def test_mock_evidence_provider_all_methods():
    """Verify MockEvidenceProvider search, read, filter, introspect."""
    provider = get_provider("mock")
    meta = provider.introspect()
    assert meta["type"] == "mock"
    assert "filter" in meta["capabilities"]

    # Search
    hits = provider.search("Valerius", limit=2)
    assert len(hits) > 0
    assert hits[0]["document_id"] == "D01"

    # Read
    slice_data = provider.read("D01", lines=(1, 2))
    assert slice_data is not None
    assert len(slice_data["lines"]) == 2
    assert slice_data["lines"][0]["line_number"] == 1

    # Filter
    filtered = provider.filter("garrison", "GT", 100)
    assert len(filtered) == 2


def test_concrete_corpus_provider():
    """Verify ConcreteCorpusProvider with synthetic World."""
    world = World(world_id="w_test_concrete", seed=42)
    world.documents["D01"] = Document(
        id="D01",
        title="Survey Record",
        doc_type="survey",
        lines=[
            DocumentLine(line_number=1, text="Sensor reading for Alpha is 300.", fact_ids=["f1"]),
            DocumentLine(line_number=2, text="Operating status: nominal.", fact_ids=["f2"])
        ]
    )
    provider = get_provider("concrete", world=world)
    assert provider.introspect()["type"] == "concrete_corpus"

    # Search
    hits = provider.search("Sensor", limit=2)
    assert len(hits) > 0
    assert hits[0]["document_id"] == "D01"

    # Read
    slice_data = provider.read("D01", lines=(1, 1))
    assert slice_data is not None
    assert len(slice_data["lines"]) == 1
    assert "Sensor reading" in slice_data["lines"][0]["text"]
