"""Arithmetic operator tests for RestrictedASTEvaluator.
Covers basic ops, precedence, and unary handling.
"""
import pytest
from agent.tools.exec import RestrictedASTEvaluator

@pytest.mark.parametrize(
    "expr, expected",
    [
        ("1 + 2", 3),
        ("5 - 3", 2),
        ("4 * 7", 28),
        ("8 / 2", 4),
        ("9 // 4", 2),
        ("9 % 4", 1),
        ("2 ** 5", 32),
        ("-5 + +3", -2),
        ("(2 + 3) * 4", 20),
        ("(10 - 2) / (3 + 1)", 2),
    ],
)
def test_basic_arithmetic(expr, expected):
    evaluator = RestrictedASTEvaluator()
    result = evaluator.evaluate_pure_math(expr)
    assert result["status"] == "success"
    assert result["result"] == expected
