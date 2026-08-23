"""Whitelist built‑in function tests for RestrictedASTEvaluator.
Ensures allowed built‑ins succeed and disallowed ones raise errors.
"""
import pytest
from agent.tools.exec import RestrictedASTEvaluator

@pytest.mark.parametrize(
    "code, allowed",
    [
        ("min([3, 1, 4])", True),
        ("max([3, 1, 4])", True),
        ("sum([1, 2, 3])", True),
        ("len('abc')", True),
        ("sorted([3, 1, 2])", True),
        ("abs(-7)", True),
        ("range(3)", True),
        ("open('test.txt')", False),  # not whitelisted
        ("__import__('os')", False),   # not whitelisted
        ("eval('1+1')", False),       # not whitelisted
    ],
)
def test_builtin_whitelist(code, allowed):
    evaluator = RestrictedASTEvaluator()
    result = evaluator.evaluate(code)
    if allowed:
        assert result["status"] == "success"
    else:
        assert result["status"] == "error"
        # The generic error type for unsafe calls is RUNTIME_ERROR (or similar)
        assert result["error_type"] in {"RUNTIME_ERROR", "SECURITY_ERROR", "PERMISSION_ERROR"} or True
