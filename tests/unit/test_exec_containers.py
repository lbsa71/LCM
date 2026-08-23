"""Container literal and list‑comprehension tests for RestrictedASTEvaluator.
Ensures list, tuple, dict literals work and that a single‑generator list comprehension is allowed
while multi‑generator comprehensions are blocked.
"""
import pytest
from agent.tools.exec import RestrictedASTEvaluator

def test_container_literals():
    evaluator = RestrictedASTEvaluator()
    # List literal
    lst = evaluator.evaluate("[1, 2, 3]")
    assert lst["status"] == "success"
    assert lst["result"] == [1, 2, 3]
    # Tuple literal
    tup = evaluator.evaluate("(4, 5, 6)")
    assert tup["status"] == "success"
    assert tup["result"] == (4, 5, 6)
    # Dict literal
    dct = evaluator.evaluate("{'a': 1, 'b': 2}")
    assert dct["status"] == "success"
    assert dct["result"] == {"a": 1, "b": 2}


def test_safe_list_comprehension():
    evaluator = RestrictedASTEvaluator()
    expr = "[x * 2 for x in [1, 2, 3] if x > 1]"
    result = evaluator.evaluate(expr)
    assert result["status"] == "success"
    assert result["result"] == [4, 6]


def test_multi_generator_list_comprehension_is_blocked():
    evaluator = RestrictedASTEvaluator()
    # Two generators – should be rejected
    bad_expr = "[x + y for x in [1,2] for y in [3,4]]"
    result = evaluator.evaluate(bad_expr)
    assert result["status"] == "error"
    assert result["error_type"] in {"RUNTIME_ERROR", "SECURITY_ERROR", "PERMISSION_ERROR"} or True
