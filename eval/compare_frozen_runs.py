"""Compare complete, task-matched frozen-run ledgers with explicit uncertainty."""

import argparse
import json
from pathlib import Path
from typing import Any

from eval.feasibility_audit import paired_comparison, strict_grounded_success
from eval.metrics import evaluate_episode_outcome


def compare_runs(baseline: Path, candidate: Path, step: int) -> dict[str, Any]:
    """Keep historical and full-proof success separate; reject mismatched inputs."""
    tasks = json.loads((baseline / "selected_tasks.json").read_text(encoding="utf-8"))
    if tasks != json.loads((candidate / "selected_tasks.json").read_text(encoding="utf-8")):
        raise ValueError("Task sets differ between compared runs")
    manifests = [json.loads((root / f"step_{step}/manifest.json").read_text(encoding="utf-8"))
                 for root in (baseline, candidate)]
    for key in ("tasks_sha256", "source_hashes"):
        if manifests[0].get(key) != manifests[1].get(key):
            raise ValueError(f"Comparison mismatch in {key}")
    if manifests[0]["input_hashes"]["worlds"] != manifests[1]["input_hashes"]["worlds"]:
        raise ValueError("World corpora differ")
    runtime = [{key: value for key, value in manifest["runtime"].items() if key != "parameters"}
               for manifest in manifests]
    if runtime[0] != runtime[1]:
        raise ValueError("Runtime settings differ")
    observations = []
    for root in (baseline, candidate):
        records = [json.loads(line) for line in
                   (root / f"step_{step}/case_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(records) != len(tasks):
            raise ValueError(f"Run has an incomplete task ledger: {root}")
        rows = []
        for task, record in zip(tasks, records):
            episode = record["episode"]
            if (episode["task_id"], episode["world_id"]) != (task["task_id"], task["world_id"]):
                raise ValueError("Task ledger is not aligned")
            rows.append({"task_id": task["task_id"], "world_id": task["world_id"], "suite": task["suite"],
                         "legacy": evaluate_episode_outcome(episode, task)["grounded_success"],
                         "strict": strict_grounded_success(task, episode)})
        observations.append(rows)
    report: dict[str, Any] = {"baseline": str(baseline), "candidate": str(candidate), "step": step}
    for score in ("legacy", "strict"):
        paired = [[{**row, "success": row[score]} for row in rows] for rows in observations]
        report[f"{score}_grounding"] = paired_comparison(*paired)
        report[f"{score}_suites"] = {
            suite: paired_comparison(*[[row for row in rows if row["suite"] == suite] for rows in paired])
            for suite in sorted({task["suite"] for task in tasks})
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--step", type=int, default=3000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(args.baseline, args.candidate, args.step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("_suites")}, indent=2))
