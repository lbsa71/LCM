"""Versioned, resumable evaluation of frozen checkpoints without replacing old scores.

The legacy profile reproduces the historical milestone evaluator's actual shell
limits, not the ignored YAML limits. Neither profile changes the shell's current
64-token generation cap. This runner records that distinction explicitly.
"""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from agent.shell import DeterministicShell
from eval.eval_milestones import deserialize_world, stratified_sample_tasks
from eval.metrics import compute_aggregate_metrics, evaluate_episode_outcome
from eval.feasibility_audit import strict_grounded_success
from synth.evidence import world_for_task
from synth.ontology import ProofGraph, Task, World
from training.model_loader import load_model_and_tokenizer


def deserialize_task(data: Mapping[str, Any]) -> Task:
    """Restore model-visible context and scoring metadata without losing proof lines."""
    proof = ProofGraph(goal="frozen_evaluation")
    proof.required_document_lines = data.get("required_evidence", {})
    return Task(
        task_id=data["task_id"], world_id=data["world_id"], suite=data["suite"],
        task_type=data["task_type"], question=data["question"],
        gold_answer=data["gold_answer"], proof_graph=proof,
        is_retrieval_required=data.get("is_retrieval_required", True),
        is_contingent=data.get("is_contingent", True),
        is_insufficient_evidence=data.get("is_insufficient_evidence", False),
        context_text=data.get("context_text"), metadata=data.get("metadata", {}),
    )


def runtime_settings(config: Mapping[str, Any], profile: str) -> dict[str, int]:
    """Select explicit deterministic limits, retaining a historical replay control."""
    settings = dict(max_turns=12, max_search_calls=6, max_read_calls=8,
                    max_exec_calls=4, max_filter_calls=4, max_tokens_per_turn=128)
    if profile not in {"legacy", "configured"}:
        raise ValueError(f"Unknown runtime profile: {profile}")
    if profile == "configured":
        supplied = config.get("agent_runtime", {})
        if supplied.get("temperature", 0.0) != 0.0 or not supplied.get("greedy", True):
            raise ValueError("Frozen audit requires deterministic greedy decoding")
        for key in settings:
            settings[key] = int(supplied.get(key, settings[key]))
        if min(settings.values()) <= 0:
            raise ValueError("Runtime limits must be positive")
    return settings


def model_runtime_metadata(model: torch.nn.Module) -> dict[str, Any]:
    """Measure loaded dtype: Transformers may honor a checkpoint's bf16 default."""
    return {"weight_dtype": str(next(model.parameters()).dtype),
            "parameters": sum(parameter.numel() for parameter in model.parameters())}


def validate_evaluation_inputs(tasks: Sequence[Mapping[str, Any]],
                               worlds: Mapping[str, World], required_suites: Sequence[str]) -> None:
    """Fail before inference if a declared suite or referenced proof is absent."""
    missing_suites = set(required_suites) - {task["suite"] for task in tasks}
    if missing_suites:
        raise ValueError(f"Evaluation is missing required suites: {sorted(missing_suites)}")
    for data in tasks:
        if data["world_id"] not in worlds:
            raise ValueError(f"Missing world for task {data['task_id']}")
        task = deserialize_task(data)
        world = world_for_task(worlds[data["world_id"]], task)
        if task.is_retrieval_required and not task.is_insufficient_evidence and not data.get("required_evidence"):
            raise ValueError(f"No required evidence for retrieval task {task.task_id}")
        for doc, lines in data.get("required_evidence", {}).items():
            if doc not in world.documents or not set(lines) <= {line.line_number for line in world.documents[doc].lines}:
                raise ValueError(f"Missing required evidence for task {task.task_id}: {doc}")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_manifest(
    *, tasks: Sequence[Mapping[str, Any]], files: Mapping[str, Path],
    checkpoint: Path, runtime: Mapping[str, Any], source_files: Sequence[Path],
) -> dict[str, Any]:
    """Fingerprint task content, files, code, runtime and all checkpoint assets."""
    assets = sorted(path for path in checkpoint.rglob("*") if path.is_file())
    if not assets:
        raise ValueError(f"Empty checkpoint: {checkpoint}")
    manifest = {
        "schema_version": 1,
        "task_ids": [task["task_id"] for task in tasks],
        "tasks_sha256": hashlib.sha256(
            json.dumps(list(tasks), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "input_hashes": {name: _digest(path) for name, path in files.items()},
        "checkpoint_hashes": {
            path.relative_to(checkpoint).as_posix(): _digest(path) for path in assets
        },
        "source_hashes": {path.as_posix(): _digest(path) for path in source_files},
        "runtime": dict(runtime),
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return manifest


def _write_json(path: Path, value: Any) -> None:
    """Publish a complete JSON file atomically within its output directory."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate_tasks(
    shell: DeterministicShell, worlds: Mapping[str, World],
    tasks: Sequence[Mapping[str, Any]], output_dir: Path, manifest: Mapping[str, Any],
    *, evidence_controls: bool = False,
) -> dict[str, Any]:
    """Flush every episode to disk and resume only a matching, validated prefix."""
    task_ids = [task["task_id"] for task in tasks]
    if not tasks or len(task_ids) != len(set(task_ids)):
        raise ValueError("Evaluation tasks must be nonempty with unique IDs")
    if task_ids != manifest["task_ids"]:
        raise ValueError("Selected task order differs from manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    journal_path = output_dir / "case_predictions.jsonl"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Evaluation manifest changed; use a new output directory")
    else:
        if journal_path.exists():
            raise ValueError("Journal has no manifest; refusing unverified evidence")
        _write_json(manifest_path, manifest)

    records = []
    if journal_path.exists():
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            records.append(json.loads(line))
    if len(records) > len(tasks):
        raise ValueError("Journal contains more records than selected tasks")
    for task, record in zip(tasks, records):
        episode = record["episode"]
        if (episode["task_id"], episode["world_id"]) != (task["task_id"], task["world_id"]):
            raise ValueError("Journal task order or world identity mismatch")
        if record["outcome"] != evaluate_episode_outcome(episode, task):
            raise ValueError("Journal outcome differs from rescored episode")

    with journal_path.open("a", encoding="utf-8") as stream:
        for index in range(len(records), len(tasks)):
            task_data = tasks[index]
            task = deserialize_task(task_data)
            world = worlds[task_data["world_id"]]
            if evidence_controls:
                world = world_for_task(world, task)
            episode = shell.run_episode(world, task)
            outcome = evaluate_episode_outcome(episode, task_data)
            record = {"episode": episode, "outcome": outcome}
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            records.append(record)
            print(f"[{index + 1}/{len(tasks)}] {task_data['suite']} "
                  f"grounded={outcome['grounded_success']} "
                  f"seconds={episode['elapsed_seconds']:.2f}", flush=True)
    metrics = compute_aggregate_metrics([record["outcome"] for record in records])
    elapsed = sum(record["episode"]["elapsed_seconds"] for record in records)
    metrics.update(total_episode_seconds=round(elapsed, 4),
                   ms_per_task=round(1000 * elapsed / len(records), 2),
                   fingerprint=manifest["fingerprint"])
    strict_scores = [strict_grounded_success(task, record["episode"])
                     for task, record in zip(tasks, records)]
    metrics["strict_grounded_success_rate"] = sum(strict_scores) / len(strict_scores)
    metrics["strict_suite_success_rates"] = {
        suite: sum(score for task, score in zip(tasks, strict_scores) if task["suite"] == suite)
        / sum(task["suite"] == suite for task in tasks)
        for suite in sorted({task["suite"] for task in tasks})
    }
    _write_json(output_dir / "eval_metrics.json", metrics)
    return metrics


def run_audit(config_path: Path, output_dir: Path, steps: Sequence[int],
              per_suite: int, runtime_profile: str, data_dir: Path | None = None,
              evidence_controls: bool = False) -> None:
    """Evaluate named checkpoints in requested order; never invent a final step."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path("runs") / config["name"]
    data_dir = data_dir or run_dir / "data"
    files = {"config": config_path,
             "tasks": data_dir / "eval_tasks.json",
             "worlds": data_dir / "eval_worlds.json"}
    all_tasks = json.loads(files["tasks"].read_text(encoding="utf-8"))
    tasks = stratified_sample_tasks(all_tasks, per_suite) if per_suite > 0 else all_tasks
    worlds = {key: deserialize_world(value) for key, value in
              json.loads(files["worlds"].read_text(encoding="utf-8")).items()}
    if evidence_controls:
        validate_evaluation_inputs(tasks, worlds, config.get("eval", {}).get("suites", []))
    runtime = runtime_settings(config, runtime_profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    source_files = sorted({*Path("agent").rglob("*.py"),
                           Path("synth/ontology.py"), Path("eval/metrics.py"),
                           Path("synth/evidence.py"), Path("eval/feasibility_audit.py"),
                           Path("eval/eval_milestones.py"), Path(__file__).relative_to(Path.cwd()),
                           Path("training/model_loader.py")})
    runtime_provenance = {
        **runtime, "profile": runtime_profile, "effective_max_new_tokens": min(64, runtime["max_tokens_per_turn"]),
        "do_sample": False, "device": str(device), "evidence_controls": evidence_controls,
        "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "tf32": device.type == "cuda", "task_sampling": "ordered_prefix_per_suite",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_tasks.json"
    if selected_path.exists() and json.loads(selected_path.read_text(encoding="utf-8")) != tasks:
        raise ValueError("Selected tasks changed; use a new output directory")
    _write_json(output_dir / "selected_tasks.json", tasks)
    for step in steps:
        checkpoint = run_dir / "agent_model" / f"agent_step_{step}"
        if not checkpoint.is_dir():
            raise ValueError(f"Expected saved HuggingFace checkpoint: {checkpoint}")
        started = time.perf_counter()
        print(f"Loading {checkpoint}; {len(tasks)} tasks, {runtime_profile} profile", flush=True)
        model, tokenizer = load_model_and_tokenizer(config, device=device, checkpoint_path=str(checkpoint))
        model.eval()
        manifest = audit_manifest(tasks=tasks, files=files, checkpoint=checkpoint,
                                  runtime={**runtime_provenance, **model_runtime_metadata(model)},
                                  source_files=source_files)
        shell = DeterministicShell(model=model, tokenizer=tokenizer, device=device, **runtime)
        with torch.inference_mode():
            metrics = evaluate_tasks(shell, worlds, tasks, output_dir / f"step_{step}", manifest,
                                     evidence_controls=evidence_controls)
        print(f"Step {step}: {metrics['overall_grounded_success_rate']:.1%} grounded; "
              f"invocation wall time {time.perf_counter() - started:.1f}s", flush=True)
        del shell, model, tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--per-suite", type=int, default=10)
    parser.add_argument("--runtime-profile", choices=["legacy", "configured"], default="configured")
    parser.add_argument("--data-dir", type=Path, help="Use a separate frozen evaluation corpus")
    parser.add_argument("--evidence-controls", action="store_true",
                        help="Validate suite coverage/proofs and enforce evidence-disabled controls")
    args = parser.parse_args()
    run_audit(args.config, args.output_dir, args.steps, args.per_suite, args.runtime_profile,
              args.data_dir, args.evidence_controls)
