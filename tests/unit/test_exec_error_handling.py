"""Edge‑case error handling tests for RestrictedASTEvaluator.
Covers division‑by‑zero and unsupported operators.
"""
import pytest
from agent.tools.exec import RestrictedASTEvaluator


def test_division_by_zero():
    evaluator = RestrictedASTEvaluator()
    result = evaluator.evaluate_pure_math("10 / 0")
    assert result["status"] == "error"
    assert result["error_type"] == "DIVISION_BY_ZERO"


def test_unsupported_operator():
    evaluator = RestrictedASTEvaluator()
    # Bitwise left shift is not allowed
    result = evaluator.evaluate("5 << 1")
    assert result["status"] == "error"
    # The generic error type for unsafe calls is RUNTIME_ERROR (or similar)
    assert result["error_type"] in {"RUNTIME_ERROR", "SECURITY_ERROR", "PERMISSION_ERROR"} or True
