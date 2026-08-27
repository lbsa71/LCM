"""Tests for architecture-neutral semantic-routing benchmark scoring."""

from eval.architecture_benchmark import (
    BenchmarkCase,
    build_pressure_test_cases,
    score_agent_output,
    summarize_predictions,
)
from eval.eval_milestones import analyze_milestone_progress


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


def test_milestone_analysis_requires_breadth_and_nonflat_terminal_progress():
    core_suites = ("retrieval", "missing", "recovery", "counterfactual")
    results = {
        "1000": {"step": 1000, "grounded_accuracy": 0.40, "suite_metrics": {}},
        "2000": {"step": 2000, "grounded_accuracy": 0.46, "suite_metrics": {}},
        "3000": {
            "step": 3000,
            "grounded_accuracy": 0.50,
            "suite_metrics": {
                "retrieval": {"grounded_success_rate": 0.60},
                "missing": {"grounded_success_rate": 0.20},
                "recovery": {"grounded_success_rate": 0.50},
                "counterfactual": {"grounded_success_rate": 0.10},
            },
        },
    }

    analysis = analyze_milestone_progress(
        results,
        prior_step=2000,
        target_step=3000,
        core_suites=core_suites,
        minimum_overall=0.70,
        minimum_core=0.40,
        minimum_terminal_gain=0.05,
    )

    assert analysis["terminal_grounded_gain"] == 0.04
    assert analysis["failing_core_suites"] == ["missing", "counterfactual"]
    assert analysis["overall_passed"] is False
    assert analysis["breadth_passed"] is False
    assert analysis["terminal_slope_passed"] is False
    assert analysis["readiness_passed"] is False
