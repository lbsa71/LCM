"""Architecture-neutral semantic-routing benchmark for LCM experiments.

This is an experimental scorecard, deliberately separate from ``tests/``.
It normalizes parser labels and ReAct agents' first tool action into a common
operation score, while retaining protocol and executable-action metrics that
would otherwise be hidden by a single end-to-end accuracy number.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import torch
import yaml
from tokenizers import Tokenizer

from agent.protocol import ToolCallMessage, parse_and_validate_message
from agent.tools.exec import RestrictedASTEvaluator
from eval.form_variation import OPERATIONS, _candidate_loss
from training.model import SyntheticTransformer, TransformerConfig
from training.model_loader import load_model_and_tokenizer


@dataclass(frozen=True)
class BenchmarkCase:
    """One fact-free utterance with a canonical operation and executable gold value."""

    case_id: str
    track: str
    utterance: str
    operation: str
    left: int
    right: int
    shared: bool

    @property
    def gold_value(self) -> int | bool:
        if self.operation == "ADD":
            return self.left + self.right
        if self.operation == "SUBTRACT":
            return self.left - self.right
        if self.operation == "COMPARE":
            return self.left > self.right
        raise ValueError(f"Unsupported operation: {self.operation}")


@dataclass(frozen=True)
class BenchmarkPrediction:
    """Normalized result from one architecture on one benchmark case."""

    case: BenchmarkCase
    raw_output: str
    predicted_operation: Optional[str]
    protocol_valid: Optional[bool]
    operation_correct: bool
    execution_correct: Optional[bool]
    latency_ms: float


def _case_rows() -> dict[str, tuple[tuple[str, str, str], ...]]:
    """Return templates per pressure-test track, indexed by canonical operation."""
    return {
        "seen_style": (
            ("ADD", "Calculate {left} plus {right}.", "shared"),
            ("SUBTRACT", "Subtract {right} from {left}.", "shared"),
            ("COMPARE", "Are {left} greater than {right}?", "extension"),
        ),
        "held_out_style": (
            ("ADD", "Find the total when {left} is joined with {right}.", "shared"),
            ("SUBTRACT", "Reduce {left} by {right}.", "shared"),
            ("COMPARE", "Check whether the count of {left} outranks {right}.", "extension"),
        ),
        "lexical_shift": (
            ("ADD", "How much do {left} and {right} make in all?", "shared"),
            ("SUBTRACT", "By what amount does {left} outrun {right}?", "shared"),
            ("COMPARE", "Is {left} above {right}?", "extension"),
        ),
        "discourse_distractor": (
            (
                "ADD",
                "Although {left} exceeds {right}, what is the combined count of {left} zols and {right} binks?",
                "shared",
            ),
            (
                "SUBTRACT",
                "The quantities belong together, but remove {right} binks from {left} zols. How many remain?",
                "shared",
            ),
            (
                "COMPARE",
                "A total may be useful later. First decide whether {left} zols outnumber {right} binks.",
                "extension",
            ),
        ),
    }


def build_pressure_test_cases() -> tuple[BenchmarkCase, ...]:
    """Build a balanced fact-free suite with declared shared/extension tracks.

    Addition and subtraction form the shared track because all three existing
    approaches were trained to handle direct arithmetic. Comparison remains an
    extension track: the semantic parser was explicitly trained for it, whereas
    the historic ReAct training corpus did not guarantee that capability.
    """
    pairs = ((345, 456), (901, 278), (613, 149))
    cases: list[BenchmarkCase] = []
    for track, templates in _case_rows().items():
        for operation, template, scope in templates:
            for index, (left, right) in enumerate(pairs, start=1):
                cases.append(
                    BenchmarkCase(
                        case_id=f"{track}_{operation.lower()}_{index}",
                        track=track,
                        utterance=template.format(left=left, right=right),
                        operation=operation,
                        left=left,
                        right=right,
                        shared=scope == "shared",
                    )
                )
    return tuple(cases)


def _operation_from_expression(expression: str) -> Optional[str]:
    """Map a strict first-level Python expression to the benchmark taxonomy."""
    try:
        node = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        return None
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            return "ADD"
        if isinstance(node.op, ast.Sub):
            return "SUBTRACT"
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Gt):
        return "COMPARE"
    return None


def score_agent_output(raw_output: str, case: BenchmarkCase, latency_ms: float = 0.0) -> BenchmarkPrediction:
    """Score a ReAct agent's first action without granting partial protocol credit."""
    cleaned = raw_output.replace("<EOS>", "").replace("<PAD>", "").strip()
    parsed = parse_and_validate_message(cleaned)
    if not isinstance(parsed, ToolCallMessage) or parsed.tool != "exec":
        return BenchmarkPrediction(
            case=case,
            raw_output=cleaned,
            predicted_operation=None,
            protocol_valid=False,
            operation_correct=False,
            execution_correct=False,
            latency_ms=round(latency_ms, 3),
        )

    expression = str(parsed.arguments.get("code", ""))
    predicted_operation = _operation_from_expression(expression)
    evaluation = RestrictedASTEvaluator().evaluate(expression)
    execution_correct = evaluation.get("status") == "success" and evaluation.get("result") == case.gold_value
    return BenchmarkPrediction(
        case=case,
        raw_output=cleaned,
        predicted_operation=predicted_operation,
        protocol_valid=True,
        operation_correct=predicted_operation == case.operation,
        execution_correct=execution_correct,
        latency_ms=round(latency_ms, 3),
    )


def score_parser_prediction(
    case: BenchmarkCase,
    predicted_operation: str,
    latency_ms: float = 0.0,
) -> BenchmarkPrediction:
    """Normalize a parser's canonical label to the common result record."""
    return BenchmarkPrediction(
        case=case,
        raw_output=f"OP={predicted_operation}",
        predicted_operation=predicted_operation,
        protocol_valid=None,
        operation_correct=predicted_operation == case.operation,
        execution_correct=None,
        latency_ms=round(latency_ms, 3),
    )


def _mean(values: Iterable[float]) -> Optional[float]:
    items = tuple(values)
    return round(sum(items) / len(items), 4) if items else None


def _summarize_slice(predictions: Sequence[BenchmarkPrediction]) -> dict[str, Any]:
    protocol = [float(p.protocol_valid) for p in predictions if p.protocol_valid is not None]
    execution = [float(p.execution_correct) for p in predictions if p.execution_correct is not None]
    return {
        "count": len(predictions),
        "operation_accuracy": _mean(float(p.operation_correct) for p in predictions),
        "protocol_valid_rate": _mean(protocol),
        "execution_ready_accuracy": _mean(execution),
        "mean_latency_ms": _mean(p.latency_ms for p in predictions),
    }


def summarize_predictions(predictions: Sequence[BenchmarkPrediction]) -> dict[str, Any]:
    """Aggregate overall, shared, extension, and per-track benchmark scores."""
    tracks = sorted({p.case.track for p in predictions})
    return {
        "overall": _summarize_slice(predictions),
        "shared": _summarize_slice([p for p in predictions if p.case.shared]),
        "extension": _summarize_slice([p for p in predictions if not p.case.shared]),
        "by_track": {
            track: _summarize_slice([p for p in predictions if p.case.track == track]) for track in tracks
        },
    }


def _device_from_name(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _parser_from_config(config: dict[str, Any], tokenizer: Tokenizer) -> SyntheticTransformer:
    model_config = config["model"]
    parser_config = TransformerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=int(model_config["hidden_size"]),
        num_hidden_layers=int(model_config["num_hidden_layers"]),
        num_attention_heads=int(model_config["num_attention_heads"]),
        intermediate_size=int(model_config["intermediate_size"]),
        max_position_embeddings=int(model_config["max_position_embeddings"]),
        rms_norm_eps=float(model_config.get("rms_norm_eps", 1e-5)),
        rope_theta=float(model_config.get("rope_theta", 10000.0)),
        tie_word_embeddings=bool(model_config.get("tie_word_embeddings", True)),
        pad_token_id=tokenizer.token_to_id("<PAD>") or 0,
        bos_token_id=tokenizer.token_to_id("<BOS>") or 1,
        eos_token_id=tokenizer.token_to_id("<EOS>") or 2,
    )
    return SyntheticTransformer(parser_config)


@torch.inference_mode()
def _predict_parser_operation(
    model: SyntheticTransformer,
    tokenizer: Tokenizer,
    utterance: str,
    device: torch.device,
) -> str:
    candidates = tuple(f"OP={operation}" for operation in OPERATIONS)
    selected = min(candidates, key=lambda candidate: _candidate_loss(model, tokenizer, utterance, candidate, device))
    return selected.removeprefix("OP=")


@torch.inference_mode()
def _generate_agent_first_turn(model: Any, tokenizer: Any, utterance: str, device: torch.device, max_new_tokens: int) -> str:
    """Generate the first action with the exact role serialization used by the shell."""
    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    user_id = tokenizer.token_to_id("<USER>")
    input_ids = [bos_id] if bos_id is not None else []
    if user_id is not None:
        input_ids.append(user_id)
    input_ids.extend(tokenizer.encode(utterance).ids)
    if eos_id is not None:
        input_ids.append(eos_id)
    inputs = torch.tensor([input_ids], dtype=torch.long, device=device)
    if hasattr(model, "can_generate"):
        hf_eos = getattr(getattr(tokenizer, "hf_tokenizer", None), "eos_token_id", None)
        stop_ids = [token_id for token_id in (eos_id, hf_eos) if token_id is not None]
        generated = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=list(set(stop_ids)) if len(set(stop_ids)) > 1 else stop_ids[0] if stop_ids else None,
            pad_token_id=tokenizer.token_to_id("<PAD>") or eos_id,
        )
    else:
        generated = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            stop_token_ids=[eos_id] if eos_id is not None else [],
            temperature=0.0,
        )
    return tokenizer.decode(generated[0, inputs.shape[1] :].tolist())


def _evaluate_parser(entry: dict[str, Any], cases: Sequence[BenchmarkCase], device: torch.device) -> tuple[dict, list[BenchmarkPrediction]]:
    with Path(entry["config"]).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    tokenizer = Tokenizer.from_file(entry["tokenizer_path"])
    model = _parser_from_config(config, tokenizer).to(device)
    model.load_state_dict(torch.load(entry["checkpoint_path"], map_location=device, weights_only=True))
    model.eval()
    predictions: list[BenchmarkPrediction] = []
    for case in cases:
        started = time.perf_counter()
        operation = _predict_parser_operation(model, tokenizer, case.utterance, device)
        predictions.append(score_parser_prediction(case, operation, (time.perf_counter() - started) * 1000))
    return {"id": entry["id"], "family": "semantic_parser", "capability": "canonical_operation"}, predictions


def _evaluate_agent(entry: dict[str, Any], cases: Sequence[BenchmarkCase], device: torch.device, max_new_tokens: int) -> tuple[dict, list[BenchmarkPrediction]]:
    with Path(entry["config"]).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("backend", "custom") == "custom":
        config["tokenizer_path"] = entry["tokenizer_path"]
    model, tokenizer = load_model_and_tokenizer(config, device=str(device), checkpoint_path=entry["checkpoint_path"])
    model.eval()
    predictions: list[BenchmarkPrediction] = []
    for case in cases:
        started = time.perf_counter()
        raw_output = _generate_agent_first_turn(model, tokenizer, case.utterance, device, max_new_tokens)
        predictions.append(score_agent_output(raw_output, case, (time.perf_counter() - started) * 1000))
    return {"id": entry["id"], "family": "react_agent", "capability": "first_math_action"}, predictions


def run_benchmark(config_path: str) -> dict[str, Any]:
    """Run all configured architectures on the fixed pressure-test suite."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    benchmark = config["benchmark"]
    device = _device_from_name(str(benchmark.get("device", "auto")))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    cases = build_pressure_test_cases()
    results = []
    for entry in benchmark["architectures"]:
        kind = entry["kind"]
        if kind == "parser":
            descriptor, predictions = _evaluate_parser(entry, cases, device)
        elif kind == "agent":
            descriptor, predictions = _evaluate_agent(
                entry, cases, device, int(benchmark.get("max_new_tokens", 64))
            )
        else:
            raise ValueError(f"Unknown benchmark architecture kind: {kind}")
        results.append({**descriptor, "summary": summarize_predictions(predictions), "predictions": [asdict(p) for p in predictions]})
        del predictions
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    report = {
        "benchmark": "architecture_pressure_test_v1",
        "scope": {
            "shared_track": "ADD/SUBTRACT semantic routing and execution-ready first action",
            "extension_track": "COMPARE; reported separately because only the parser was trained for it",
            "legacy_scratch_status": "diagnostic_only: its historical training corpus failed current contamination validation",
        },
        "case_count": len(cases),
        "cases": [asdict(case) for case in cases],
        "results": results,
    }
    output_dir = Path(benchmark["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LCM architecture pressure-test benchmark")
    parser.add_argument("--config", default="configs/architecture_benchmark.yaml")
    args = parser.parse_args()
    report = run_benchmark(args.config)
    for result in report["results"]:
        shared = result["summary"]["shared"]
        print(
            f"{result['id']}: shared operation={shared['operation_accuracy']:.1%}; "
            f"protocol={shared['protocol_valid_rate']}; execution={shared['execution_ready_accuracy']}"
        )


if __name__ == "__main__":
    main()
