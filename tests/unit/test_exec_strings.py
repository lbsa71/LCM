"""Unit tests for safe string execution and collection method support in RestrictedASTEvaluator."""

import pytest
from agent.tools.exec import RestrictedASTEvaluator


def test_string_letter_counting():
    evaluator = RestrictedASTEvaluator()
    res = evaluator.evaluate('"Strawberry".lower().count("r")')
    assert res["status"] == "success"
    assert res["result"] == 3


def test_string_slice_reversal():
    evaluator = RestrictedASTEvaluator()
    res = evaluator.evaluate('"almanac"[::-1]')
    assert res["status"] == "success"
    assert res["result"] == "canamla"


def test_string_case_and_replace():
    evaluator = RestrictedASTEvaluator()
    res = evaluator.evaluate('"hello world".upper().replace("WORLD", "LCM")')
    assert res["status"] == "success"
    assert res["result"] == "HELLO LCM"


def test_string_comprehension_counting():
    evaluator = RestrictedASTEvaluator()
    res = evaluator.evaluate('len([c for c in "Mississippi".lower() if c == "s"])')
    assert res["status"] == "success"
    assert res["result"] == 4


def test_prohibited_attribute_fails():
    evaluator = RestrictedASTEvaluator()
    res = evaluator.evaluate('"test".__class__')
    assert res["status"] == "error"
    assert res["error_type"] == "SECURITY_VIOLATION"
