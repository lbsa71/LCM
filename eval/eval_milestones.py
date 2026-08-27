"""Evaluates model checkpoints across all benchmark suites and logs milestone scaling progression."""

import argparse
import json
import os
import time
import yaml
from collections import defaultdict
from typing import Mapping, Sequence
import torch

from synth.ontology import World, Task, Document, DocumentLine
from agent.shell import DeterministicShell
from eval.metrics import evaluate_episode_outcome, compute_aggregate_metrics
from training.model_loader import load_model_and_tokenizer
from training.agent_sft import summarize_sft_exposure


def deserialize_world(data: dict) -> World:
    w = World(world_id=data["world_id"], seed=data["seed"])
    for d_id, d_data in data.get("documents", {}).items():
        lines = [
            DocumentLine(line_number=l["line_number"], text=l["text"], fact_ids=l.get("fact_ids", []))
            for l in d_data.get("lines", [])
        ]
        doc = Document(id=d_data["id"], title=d_data["title"], doc_type=d_data["doc_type"], lines=lines)
        w.documents[d_id] = doc
    return w


SUITE_ORDER = [
    "suite_a_language",
    "suite_b_invariants",
    "suite_c_single_hop",
    "suite_d_multi_hop",
    "suite_e_retrieval_computation",
    "suite_f_missing_evidence",
    "suite_g_tool_recovery",
    "suite_h_direct_computation",
    "suite_i_counterfactual_inversion",
    "anti_memorization_permutation",
    "anti_memorization_prior_reversal",
    "anti_memorization_evidence_disabled",
    "anti_memorization_closed_book"
]


def stratified_sample_tasks(tasks, per_suite=30):
    by_suite = defaultdict(list)
    for t in tasks:
        by_suite[t["suite"]].append(t)
    sampled = []
    
    seen_suites = set()
    for s_name in SUITE_ORDER:
        if s_name in by_suite:
            sampled.extend(by_suite[s_name][:per_suite])
            seen_suites.add(s_name)
            
    for suite, suite_tasks in sorted(by_suite.items()):
        if suite not in seen_suites:
            sampled.extend(suite_tasks[:per_suite])
    return sampled


def analyze_milestone_progress(
    results: Mapping[str, Mapping[str, object]],
    *,
    prior_step: int,
    target_step: int,
    core_suites: Sequence[str],
    minimum_overall: float,
    minimum_core: float,
    minimum_terminal_gain: float,
) -> dict[str, object]:
    """Apply a preregistered breadth and terminal-slope readiness gate."""
    indexed = {int(result["step"]): result for result in results.values()}
    if prior_step not in indexed or target_step not in indexed:
        raise ValueError("Milestone readiness analysis requires both registered steps")
    prior = indexed[prior_step]
    target = indexed[target_step]
    suite_metrics = target.get("suite_metrics")
    if not isinstance(suite_metrics, Mapping):
        raise ValueError("Target milestone is missing suite metrics")
    core_scores: dict[str, float] = {}
    for suite in core_suites:
        suite_result = suite_metrics.get(suite)
        if not isinstance(suite_result, Mapping) or "grounded_success_rate" not in suite_result:
            raise ValueError(f"Target milestone is missing core suite {suite}")
        core_scores[suite] = float(suite_result["grounded_success_rate"])
    prior_accuracy = float(prior["grounded_accuracy"])
    target_accuracy = float(target["grounded_accuracy"])
    terminal_gain = round(target_accuracy - prior_accuracy, 4)
    failing_core_suites = [
        suite for suite in core_suites if core_scores[suite] < minimum_core
    ]
    overall_passed = target_accuracy >= minimum_overall
    breadth_passed = not failing_core_suites
    terminal_slope_passed = terminal_gain >= minimum_terminal_gain
    return {
        "prior_step": prior_step,
        "target_step": target_step,
        "prior_grounded_accuracy": prior_accuracy,
        "target_grounded_accuracy": target_accuracy,
        "terminal_grounded_gain": terminal_gain,
        "core_suite_grounded_accuracy": core_scores,
        "thresholds": {
            "minimum_overall": minimum_overall,
            "minimum_core": minimum_core,
            "minimum_terminal_gain": minimum_terminal_gain,
        },
        "failing_core_suites": failing_core_suites,
        "overall_passed": overall_passed,
        "breadth_passed": breadth_passed,
        "terminal_slope_passed": terminal_slope_passed,
        "readiness_passed": overall_passed and breadth_passed and terminal_slope_passed,
    }


def evaluate_checkpoint(model, tokenizer, device, worlds, tasks, checkpoint_name):
    shell = DeterministicShell(model=model, tokenizer=tokenizer, device=device)
    episodes = []
    outcomes = []
    t0 = time.time()

    class StructTask:
        def __init__(self, d):
            self.task_id = d["task_id"]
            self.suite = d["suite"]
            self.task_type = d["task_type"]
            self.question = d["question"]
            self.gold_answer = d["gold_answer"]
            self.is_retrieval_required = d["is_retrieval_required"]
            self.is_contingent = d["is_contingent"]
            self.is_insufficient_evidence = d["is_insufficient_evidence"]
            self.required_evidence = d.get("required_evidence", {})
            self.proof_graph = None

    print(f"\n[*] Evaluating checkpoint: {checkpoint_name} on {len(tasks)} tasks...")
    for idx, task_data in enumerate(tasks):
        world_id = task_data["world_id"]
        world = worlds[world_id]
        t_obj = StructTask(task_data)
        
        ep = shell.run_episode(world, t_obj)
        episodes.append(ep)
        out = evaluate_episode_outcome(ep, task_data)
        outcomes.append(out)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(tasks):
            elapsed = time.time() - t0
            current_acc = sum(1 for o in outcomes if o["grounded_success"]) / len(outcomes)
            print(f"    [{idx+1}/{len(tasks)}] Grounded Acc: {current_acc*100:.1f}% (elapsed: {elapsed:.1f}s)")

    metrics = compute_aggregate_metrics(outcomes)
    return metrics, outcomes, episodes


def run_milestone_evaluation(config_path: str, per_suite: int = 20, final_only: bool = False):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    preset_name = config.get("name", "smoke")
    backend = config.get("backend", "custom")
    run_dir = f"runs/{preset_name}"
    data_dir = os.path.join(run_dir, "data")
    tok_dir = os.path.join(run_dir, "tokenizer")
    base_dir = os.path.join(run_dir, "base_model")
    agent_dir = os.path.join(run_dir, "agent_model")
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Load Worlds & Tasks
    eval_worlds_path = os.path.join(data_dir, "eval_worlds.json")
    eval_tasks_path = os.path.join(data_dir, "eval_tasks.json")

    with open(eval_worlds_path, "r", encoding="utf-8") as f:
        worlds_raw = json.load(f)
    worlds = {w_id: deserialize_world(w_data) for w_id, w_data in worlds_raw.items()}

    with open(eval_tasks_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    if per_suite > 0:
        tasks = stratified_sample_tasks(all_tasks, per_suite=per_suite)
    else:
        tasks = all_tasks

    print(f"[*] Total evaluation tasks: {len(tasks)} across {len(worlds)} held-out worlds.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Discover Milestones
    milestones = []
    if backend == "huggingface":
        step_tuples = []
        if os.path.exists(agent_dir):
            for fname in os.listdir(agent_dir):
                if fname.startswith("agent_step_") and os.path.isdir(os.path.join(agent_dir, fname)):
                    step_str = fname.replace("agent_step_", "")
                    try:
                        step_num = int(step_str)
                        step_tuples.append((step_num, os.path.join(agent_dir, fname)))
                    except ValueError:
                        pass
        step_tuples.sort(key=lambda x: x[0])
        for step_num, fpath in step_tuples:
            milestones.append((f"Step {step_num}", fpath, step_num))

        if os.path.exists(os.path.join(agent_dir, "config.json")) or os.path.exists(os.path.join(agent_dir, "model.safetensors")):
            final_step = step_tuples[-1][0] + 1 if step_tuples else 9999
            milestones.append(("Final Step", agent_dir, final_step))
    else:
        step_tuples = []
        if os.path.exists(agent_dir):
            for fname in os.listdir(agent_dir):
                if fname.startswith("agent_step_") and fname.endswith(".pt"):
                    step_str = fname.replace("agent_step_", "").replace(".pt", "")
                    try:
                        step_num = int(step_str)
                        step_tuples.append((step_num, os.path.join(agent_dir, fname)))
                    except ValueError:
                        pass
        step_tuples.sort(key=lambda x: x[0])
        for step_num, fpath in step_tuples:
            milestones.append((f"Step {step_num}", fpath, step_num))

        final_path = os.path.join(agent_dir, "agent_final.pt")
        if os.path.exists(final_path):
            final_step = step_tuples[-1][0] + 1 if step_tuples else 9999
            milestones.append(("Final Step", final_path, final_step))

    summary_path = os.path.join(eval_dir, "milestone_scaling_results.json")
    all_milestone_results = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                all_milestone_results = json.load(f)
        except Exception:
            all_milestone_results = {}

    if final_only:
        milestones = [m for m in milestones if m[0] == "Final Step" or "3000" in m[0]]

    for label, weights_path, step in milestones:
        step_key = str(step)
        if step_key in all_milestone_results and "grounded_accuracy" in all_milestone_results[step_key]:
            print(f"[*] Skipping already evaluated {label}: Grounded Acc = {all_milestone_results[step_key]['grounded_accuracy']*100:.1f}%")
            continue

        if not os.path.exists(weights_path):
            print(f"[!] Warning: weights {weights_path} not found. Skipping.")
            continue

        # Load model checkpoint
        if backend == "huggingface":
            model, tokenizer = load_model_and_tokenizer(config, device=device, checkpoint_path=weights_path)
        else:
            tok_path = os.path.join(tok_dir, "tokenizer.json")
            config["tokenizer_path"] = tok_path
            model, tokenizer = load_model_and_tokenizer(config, device=device, checkpoint_path=weights_path)

        model.eval()

        eval_t0 = time.time()
        metrics, outcomes, episodes = evaluate_checkpoint(model, tokenizer, device, worlds, tasks, label)
        eval_elapsed = time.time() - eval_t0
        
        grounded_acc = metrics["overall_grounded_success_rate"]
        error_rate = 1.0 - grounded_acc
        task_success = metrics["overall_task_success_rate"]
        unsupported_rate = metrics["unsupported_claim_rate"]
        sft_config = config.get("agent_sft", {})
        exposure = summarize_sft_exposure(
            microsteps=step,
            batch_size=int(sft_config.get("batch_size", 4)),
            gradient_accumulation_steps=int(
                sft_config.get("gradient_accumulation_steps", 8)
            ),
            sequence_length=1,
        )
        approx_samples = exposure["sequences_processed"]

        all_milestone_results[step_key] = {
            "label": label,
            "step": step,
            "approx_samples": approx_samples,
            "wall_clock_seconds": round(eval_elapsed, 2),
            "ms_per_task": round((eval_elapsed / max(1, len(tasks))) * 1000, 2),
            "grounded_accuracy": round(grounded_acc, 4),
            "error_rate": round(error_rate, 4),
            "raw_task_accuracy": round(task_success, 4),
            "unsupported_claim_rate": round(unsupported_rate, 4),
            "prior_contamination_rate": round(metrics.get("overall_prior_contamination_rate", 0.0), 4),
            "suite_metrics": metrics.get("suite_metrics", {}),
            "failure_taxonomy_distribution": metrics.get("failure_taxonomy_distribution", {})
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_milestone_results, f, indent=2)
        print(f"[+] Saved checkpoint metrics to {summary_path}")

    feasibility_gate = config.get("feasibility_gate")
    if isinstance(feasibility_gate, Mapping):
        analysis = analyze_milestone_progress(
            all_milestone_results,
            prior_step=int(feasibility_gate["prior_step"]),
            target_step=int(feasibility_gate["target_step"]),
            core_suites=tuple(str(suite) for suite in feasibility_gate["core_suites"]),
            minimum_overall=float(feasibility_gate["minimum_overall"]),
            minimum_core=float(feasibility_gate["minimum_core"]),
            minimum_terminal_gain=float(feasibility_gate["minimum_terminal_gain"]),
        )
        analysis_path = os.path.join(eval_dir, "milestone_feasibility_analysis.json")
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        print(f"[+] Saved feasibility analysis to {analysis_path}")

    # Final Summary Table
    print("\n" + "=" * 90)
    print("  LCM SAMPLE SIZE SCALING & ERROR RATE PROGRESSION (SmolLM2 Adapter)")
    print("=" * 90)
    print(f"{'Milestone':<25} | {'Grounded Acc':>13} | {'Error Rate':>11} | {'Raw Match':>10} | {'Unsupported':>12} | {'Prior Contam':>13}")
    print("-" * 90)
    
    sorted_steps = sorted(all_milestone_results.keys(), key=lambda k: all_milestone_results[k]["step"])
    for s_k in sorted_steps:
        res = all_milestone_results[s_k]
        lbl = res["label"]
        g_acc = f"{res['grounded_accuracy']*100:.1f}%"
        e_rate = f"{res['error_rate']*100:.1f}%"
        r_acc = f"{res['raw_task_accuracy']*100:.1f}%"
        u_rate = f"{res['unsupported_claim_rate']*100:.1f}%"
        p_rate = f"{res.get('prior_contamination_rate', 0.0)*100:.1f}%"
        print(f"{lbl:<25} | {g_acc:>13} | {e_rate:>11} | {r_acc:>10} | {u_rate:>12} | {p_rate:>13}")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smollm2_135m_agent.yaml")
    parser.add_argument("--per-suite", type=int, default=20)
    parser.add_argument("--final-only", action="store_true", help="Only evaluate final/converged checkpoints")
    args = parser.parse_args()
    run_milestone_evaluation(args.config, per_suite=args.per_suite, final_only=args.final_only)
