"""Unit tests for deterministic tools (Search, Read, Exec)."""

import pytest
from synth.world import WorldGenerator
from synth.documents.generator import DocumentGenerator
from agent.tools.search import DeterministicBM25Search
from agent.tools.read import DocumentReader
from agent.tools.exec import RestrictedASTEvaluator


def test_bm25_search_and_tie_breaking():
    gen = WorldGenerator(base_seed=42)
    world = gen.generate_world("w1", seed=100)
    doc_gen = DocumentGenerator()
    doc_gen.generate_documents(world)

    searcher = DeterministicBM25Search(world)
    res = searcher.search("population", limit=3)
    assert res["status"] == "success"
    assert len(res["results"]) > 0

    # Determinism check
    res2 = searcher.search("population", limit=3)
    assert res["results"] == res2["results"]


def test_document_reader():
    gen = WorldGenerator(base_seed=42)
    world = gen.generate_world("w1", seed=100)
    doc_gen = DocumentGenerator()
    doc_gen.generate_documents(world)

    reader = DocumentReader(world)
    res = reader.read("D01")
    assert res["status"] == "success"
    assert "D01:L1" in res["text"]

    err_res = reader.read("D999")
    assert err_res["status"] == "error"
    assert err_res["error_type"] == "DOCUMENT_NOT_FOUND"


def test_restricted_ast_exec_safe():
    evaluator = RestrictedASTEvaluator()
    # Math & builtins
    res = evaluator.evaluate("sum([10, 20, 30]) + min(5, 8)")
    assert res["status"] == "success"
    assert res["result"] == 65

    # List comprehension
    res_comp = evaluator.evaluate("[x * 2 for x in [1, 2, 3] if x > 1]")
    assert res_comp["status"] == "success"
    assert res_comp["result"] == [4, 6]


def test_restricted_ast_exec_security_rejections():
    evaluator = RestrictedASTEvaluator()
    
    # Imports
    res_imp = evaluator.evaluate("__import__('os').system('ls')")
    assert res_imp["status"] == "error"

    # Dunder access
    res_dunder = evaluator.evaluate("().__class__.__bases__[0].__subclasses__()")
    assert res_dunder["status"] == "error"

    # Unauthorized statements
    res_stmt = evaluator.evaluate("import math")
    assert res_stmt["status"] == "error"
