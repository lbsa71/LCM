"""Tests for architecture-neutral semantic-routing benchmark scoring."""

from eval.architecture_benchmark import (
    BenchmarkCase,
    build_pressure_test_cases,
    score_agent_output,
    summarize_predictions,
)


def _case(operation: str = "ADD", shared: bool = True) -> BenchmarkCase:
    return BenchmarkCase(
        case_id="unit_case",
        track="unit",
        utterance="Calculate 345 plus 456.",
        operation=operation,
        left=345,
        right=456,
        shared=shared,
    )


def test_pressure_test_has_shared_and_explicit_extension_tracks():
    cases = build_pressure_test_cases()

    assert {case.operation for case in cases if case.shared} == {"ADD", "SUBTRACT"}
    assert {case.operation for case in cases if not case.shared} == {"COMPARE"}
    assert len({case.case_id for case in cases}) == len(cases)
    assert {case.track for case in cases} >= {
        "seen_style",
        "held_out_style",
        "lexical_shift",
        "discourse_distractor",
    }


def test_agent_scoring_separates_routing_from_execution_readiness():
    correct = score_agent_output("MATH 345 + 456", _case())
    wrong_operands = score_agent_output("MATH 3 + 4", _case())
    wrong_operator = score_agent_output("MATH 345 - 456", _case())

    assert correct.protocol_valid is True
    assert correct.operation_correct is True
    assert correct.execution_correct is True
    assert wrong_operands.operation_correct is True
    assert wrong_operands.execution_correct is False
    assert wrong_operator.operation_correct is False


def test_summary_reports_only_applicable_shared_metrics():
    predictions = [
        score_agent_output("MATH 345 + 456", _case("ADD")),
        score_agent_output("MATH 345 - 456", _case("ADD")),
        score_agent_output("MATH 345 > 456", _case("COMPARE", shared=False)),
    ]

    summary = summarize_predictions(predictions)

    assert summary["shared"]["count"] == 2
    assert summary["shared"]["operation_accuracy"] == 0.5
    assert summary["extension"]["count"] == 1
