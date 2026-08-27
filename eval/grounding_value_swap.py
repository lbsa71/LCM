"""Prepare an evaluation-only paired probe of dependence on document values."""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from eval.eval_milestones import stratified_sample_tasks
from eval.feasibility_audit import strict_grounded_success


SUITE = "grounding_value_swap"


def build_probe(tasks: Sequence[Mapping[str, Any]], worlds: Mapping[str, Any]) -> tuple[list[dict], dict]:
    """Keep questions fixed while independently swapping the referenced world fact.

    Answers are generated before inference. No model output informs the choice.
    A fixed answer to a question cannot get both members of a pair correct.
    """
    probe_tasks, probe_worlds = [], {}
    syllables = ("vel", "nar", "tor", "kas", "mir", "pel", "zor", "dan", "lor", "fin")
    for index, source in enumerate(tasks):
        old_value = str(source["gold_answer"])
        if not source.get("required_evidence"):
            raise ValueError("Value-swap probe requires referenced evidence")
        for variant in (0, 1):
            task = copy.deepcopy(source)
            world = copy.deepcopy(worlds[source["world_id"]])
            if re.fullmatch(r"[-+]?\d+(\.\d+)?", old_value):
                new_value = str(1200 + index * 37 + variant * 211)
            else:
                new_value = (syllables[index % len(syllables)] + ("ora" if variant == 0 else "eth")).capitalize()
            if new_value in (old_value, str(source.get("metadata", {}).get("prior_answer", ""))):
                raise ValueError("Generated value collides with a known answer")
            replacements = 0
            for doc_id, line_numbers in source["required_evidence"].items():
                for line in world["documents"][doc_id]["lines"]:
                    if line["line_number"] not in line_numbers:
                        continue
                    line["text"], count = re.subn(rf"\b{re.escape(old_value)}\b", new_value, line["text"])
                    replacements += count
                    if count:
                        for fact_id in line.get("fact_ids", []):
                            if fact_id in world.get("facts", {}):
                                world["facts"][fact_id]["value"] = new_value
            if replacements == 0:
                raise ValueError(f"No document value replaced for {source['task_id']}")
            task["task_id"] = f"{source['task_id']}_swap_{variant}"
            task["world_id"] = f"{source['task_id']}_world_swap_{variant}"
            task["gold_answer"] = new_value
            task["suite"] = SUITE
            task["metadata"] = {**task.get("metadata", {}), "source_task_id": source["task_id"],
                                "variant": variant, "training_counterfactual_answer": old_value}
            world["world_id"] = task["world_id"]
            probe_worlds[task["world_id"]] = world
            probe_tasks.append(task)
    return probe_tasks, probe_worlds


def score_probe(tasks: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score complete question-matched pairs; one successful variant is not enough."""
    if not tasks or len(tasks) != len(records):
        raise ValueError("A complete probe ledger is required")
    pairs: dict[str, dict[int, dict[str, Any]]] = {}
    correct = reused = prior = 0
    for task, record in zip(tasks, records):
        episode = record["episode"]
        if (task["task_id"], task["world_id"]) != (episode["task_id"], episode["world_id"]):
            raise ValueError("Probe task ledger is not aligned")
        metadata = task["metadata"]
        pair = pairs.setdefault(metadata["source_task_id"], {})
        variant = metadata["variant"]
        if variant in pair:
            raise ValueError("Duplicate probe variant")
        success = strict_grounded_success(task, episode)
        answer = str(episode.get("model_answer", "")).strip().lower()
        correct += int(success)
        reused += int(answer == str(metadata["training_counterfactual_answer"]).lower())
        prior += int(answer == str(metadata.get("prior_answer", "")).lower())
        pair[variant] = {"strict_success": success, "model_answer": answer,
                         "gold_answer": task["gold_answer"], "task_id": task["task_id"]}
    if any(set(pair) != {0, 1} for pair in pairs.values()):
        raise ValueError("Every probe pair must be complete")
    both = sum(all(item["strict_success"] for item in pair.values()) for pair in pairs.values())
    return {
        "cases": len(tasks), "pairs": len(pairs),
        "strict_case_success_rate": correct / len(tasks),
        "both_variants_strictly_correct": both,
        "both_variants_strictly_correct_rate": both / len(pairs),
        "training_counterfact_reuse_rate": reused / len(tasks),
        "real_world_prior_reuse_rate": prior / len(tasks),
        "pair_outcomes": pairs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--configs", type=Path, nargs="+")
    parser.add_argument("--score-dir", type=Path, help="Score an already completed step-3000 probe replay")
    args = parser.parse_args()
    if args.score_dir:
        tasks = json.loads((args.score_dir / "selected_tasks.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (args.score_dir / "step_3000/case_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        result = score_probe(tasks, records)
        (args.score_dir / "paired_probe_analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "pair_outcomes"}, indent=2))
        raise SystemExit(0)
    if args.source_data is None or args.output_dir is None or not args.configs:
        parser.error("Preparation requires --source-data, --output-dir and --configs")
    if args.output_dir.exists():
        raise ValueError("Probe output exists; preserve the registered artifact")
    tasks_path, worlds_path = args.source_data / "eval_tasks.json", args.source_data / "eval_worlds.json"
    selected = stratified_sample_tasks(json.loads(tasks_path.read_text(encoding="utf-8")), 10)
    source_tasks = [task for task in selected if task["suite"] == "suite_i_counterfactual_inversion"]
    if len(source_tasks) != 10:
        raise ValueError("Probe registration requires ten source cases / twenty paired variants")
    tasks, worlds = build_probe(source_tasks, json.loads(worlds_path.read_text(encoding="utf-8")))
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True)
    for name, value in [("eval_tasks.json", tasks), ("eval_worlds.json", worlds)]:
        (data_dir / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    for config_path in args.configs:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["eval"] = {"suites": [SUITE]}
        config["evaluation_only"] = True
        config.pop("feasibility_gate", None)
        (args.output_dir / config_path.name).write_text(yaml.safe_dump(config), encoding="utf-8")
    registration = {
        "probe": SUITE, "cases": len(tasks), "pairs": len(source_tasks),
        "source_tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "source_worlds_sha256": hashlib.sha256(worlds_path.read_bytes()).hexdigest(),
        "tasks_sha256": hashlib.sha256((data_dir / "eval_tasks.json").read_bytes()).hexdigest(),
        "primary": "Proportion of source pairs with both variants strictly grounded and correct",
        "secondary": "Single-case accuracy and reuse of the training counterfactual answer",
        "scope": "Frozen-checkpoint development diagnostic; no training or architecture-wide inference",
    }
    (args.output_dir / "registration.json").write_text(json.dumps(registration, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(registration, indent=2))
