"""Corpus-validity checks and paired diagnostics, separate from historical scoring."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from eval.eval_milestones import stratified_sample_tasks
from eval.metrics import evaluate_episode_outcome


def audit_corpus(train_path: Path, tasks: Sequence[Mapping[str, Any]],
                 worlds: Mapping[str, Any]) -> dict[str, Any]:
    """Count observable corpus defects; shared questions alone do not prove leakage."""
    train_questions: set[str] = set()
    train_worlds: set[str] = set()
    families: dict[str, Counter] = defaultdict(Counter)
    total = nonterminal = 0
    with train_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            turns = row["turns"]
            total += 1
            terminal_missing = not turns or turns[-1]["role"] != "final"
            nonterminal += int(terminal_missing)
            family = row["task_id"].partition(row["world_id"])[0]
            families[family]["count"] += 1
            families[family]["nonterminal"] += int(terminal_missing)
            train_worlds.add(row["world_id"])
            train_questions.update(turn["content"] for turn in turns if turn["role"] == "user")
    suites: dict[str, Counter] = defaultdict(Counter)
    questions: dict[str, set[str]] = defaultdict(set)
    test_worlds: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        suite = task["suite"]
        suites[suite]["count"] += 1
        suites[suite]["question_seen_in_training"] += int(task["question"] in train_questions)
        questions[suite].add(task["question"])
        test_worlds[suite].add(task["world_id"])
        docs = worlds[task["world_id"]]["documents"]
        missing = sum(
            1 for doc, lines in task.get("required_evidence", {}).items() for line in lines
            if doc not in docs or not any(item["line_number"] == line for item in docs[doc]["lines"])
        )
        suites[suite]["missing_required_document_lines"] += missing
        if task.get("metadata", {}).get("withhold_evidence") and docs:
            suites[suite]["withhold_flag_but_nonempty_world"] += 1
        if task.get("task_type") == "closed_book_leakage" and docs:
            suites[suite]["closed_book_but_nonempty_world"] += 1
    return {
        "training_source": str(train_path),
        "training_trajectories": total,
        "nonterminal_trajectories": nonterminal,
        "training_unique_questions": len(train_questions),
        "train_eval_world_overlap": sorted(train_worlds & {task["world_id"] for task in tasks}),
        "trajectory_families": dict(families),
        "suites": {suite: {**counts, "unique_questions": len(questions[suite]),
                           "worlds": len(test_worlds[suite])} for suite, counts in suites.items()},
        "interpretation": "Question overlap is not full-context overlap for retrieval tasks. "
                          "Nonterminal demonstrations and ineffective evidence withholding need separate controls.",
    }


def strict_grounded_success(task: Mapping[str, Any], episode: Mapping[str, Any]) -> bool:
    """Require all proof lines to be cited AND observed; retain legacy raw matching."""
    legacy = evaluate_episode_outcome(dict(episode), dict(task))
    if not legacy["grounded_success"]:
        return False
    if not task.get("is_retrieval_required", True) or task.get("is_insufficient_evidence", False):
        return True
    required = {(doc, line) for doc, lines in task.get("required_evidence", {}).items() for line in lines}
    cited = {(cite["document_id"], line) for cite in episode.get("cited_evidence", [])
             for line in cite.get("lines", [])}
    observed = set()
    for step in episode.get("trace_steps", []):
        observation = step.get("observation", {})
        doc = observation.get("document_id")
        if doc and observation.get("status") == "success":
            for line in re.findall(rf"(?m)^{re.escape(doc)}:L(\d+) ", observation.get("text", "")):
                observed.add((doc, int(line)))
    return bool(required) and required <= cited and required <= observed


def paired_comparison(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Bootstrap paired task changes by world, NOT fictitious training replicates."""
    ids_before = [(row["task_id"], row["world_id"]) for row in before]
    ids_after = [(row["task_id"], row["world_id"]) for row in after]
    if not before or ids_before != ids_after or len(set(ids_before)) != len(ids_before):
        raise ValueError("Paired observations must be nonempty, unique and aligned")
    groups: dict[str, list[float]] = defaultdict(list)
    for left, right in zip(before, after):
        groups[left["world_id"]].append(float(right["success"]) - float(left["success"]))
    totals = np.asarray([sum(values) for values in groups.values()])
    sizes = np.asarray([len(values) for values in groups.values()])
    rng = np.random.default_rng(20260827)
    indices = rng.integers(0, len(groups), size=(10000, len(groups)))
    gains = totals[indices].sum(axis=1) / sizes[indices].sum(axis=1)
    return {
        "tasks": len(before), "world_clusters": len(groups),
        "before_success_rate": sum(row["success"] for row in before) / len(before),
        "after_success_rate": sum(row["success"] for row in after) / len(after),
        "gain_pp": round(100 * totals.sum() / len(before), 4),
        "world_cluster_ci95_pp": [round(float(x) * 100, 4) for x in np.quantile(gains, [0.025, 0.975])],
        "independent_training_replicates": 1,
        "uncertainty_scope": "Conditional on these frozen runs and sampled worlds; not seed uncertainty or a population guarantee.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-suite", type=int, default=10)
    args = parser.parse_args()
    data = args.run_dir / "data"
    tasks = json.loads((data / "eval_tasks.json").read_text(encoding="utf-8"))
    selected = stratified_sample_tasks(tasks, args.per_suite) if args.per_suite > 0 else tasks
    worlds = json.loads((data / "eval_worlds.json").read_text(encoding="utf-8"))
    report = audit_corpus(data / "agent_sft_train.jsonl", selected, worlds)
    report["data_hashes"] = {}
    for name in ("agent_sft_train.jsonl", "eval_tasks.json", "eval_worlds.json"):
        with (data / name).open("rb") as stream:
            report["data_hashes"][name] = hashlib.file_digest(stream, "sha256").hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
