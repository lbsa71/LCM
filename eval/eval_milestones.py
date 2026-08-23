"""Milestone checkpoint evaluator tracking error rate and performance scaling across sample sizes."""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
import torch
import yaml
from tokenizers import Tokenizer

from synth.ontology import World, Task, Document, DocumentLine
from training.model import SyntheticTransformer, TransformerConfig
from agent.shell import DeterministicShell
from eval.metrics import evaluate_episode_outcome, compute_aggregate_metrics
from eval.oracle import OracleSolver
from eval.baselines.no_tool import NoToolBaseline


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
    "suite_h_adversarial_distractors",
    "suite_i_relational_filter",
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
    
    # Process suites in canonical SUITE_ORDER first
    seen_suites = set()
    for s_name in SUITE_ORDER:
        if s_name in by_suite:
            sampled.extend(by_suite[s_name][:per_suite])
            seen_suites.add(s_name)
            
    # Add any remaining suites not in canonical order
    for suite, suite_tasks in sorted(by_suite.items()):
        if suite not in seen_suites:
            sampled.extend(suite_tasks[:per_suite])
    return sampled


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
            curr_acc = sum(1 for o in outcomes if o.get("grounded_success", False)) / (idx + 1)
            print(f"    [{idx + 1}/{len(tasks)}] Grounded Acc: {curr_acc * 100:.1f}% (elapsed: {time.time() - t0:.1f}s)", flush=True)

    metrics = compute_aggregate_metrics(outcomes)
    return metrics, outcomes, episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/small.yaml")
    parser.add_argument("--per-suite", type=int, default=30, help="Number of tasks per suite for stratified sample (0 for all)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    preset_name = config.get("name", "small")
    run_dir = f"runs/{preset_name}"
    data_dir = os.path.join(run_dir, "data")
    tok_dir = os.path.join(run_dir, "tokenizer")
    agent_dir = os.path.join(run_dir, "agent_model")
    base_dir = os.path.join(run_dir, "base_model")
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Load Tokenizer
    tok_path = os.path.join(tok_dir, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)

    # Load Worlds & Tasks
    eval_worlds_path = os.path.join(data_dir, "eval_worlds.json")
    eval_tasks_path = os.path.join(data_dir, "eval_tasks.json")

    with open(eval_worlds_path, "r", encoding="utf-8") as f:
        worlds_raw = json.load(f)
    worlds = {w_id: deserialize_world(w_data) for w_id, w_data in worlds_raw.items()}

    with open(eval_tasks_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    if args.per_suite > 0:
        tasks = stratified_sample_tasks(all_tasks, per_suite=args.per_suite)
    else:
        tasks = all_tasks

    print(f"[*] Total evaluation tasks: {len(tasks)} across {len(worlds)} held-out worlds.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Setup Base Model
    cfg_path = os.path.join(base_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        m_dict = json.load(f)
    trans_cfg = TransformerConfig(**m_dict)
    model = SyntheticTransformer(trans_cfg).to(device)

    # Milestones to evaluate (dynamically discovered from agent_dir)
    step_tuples = []
    for fname in os.listdir(agent_dir):
        if fname.startswith("agent_step_") and fname.endswith(".pt"):
            step_str = fname.replace("agent_step_", "").replace(".pt", "")
            try:
                step_num = int(step_str)
                step_tuples.append((step_num, fname))
            except ValueError:
                pass
    step_tuples.sort(key=lambda x: x[0])

    milestones = []
    for step_num, fname in step_tuples:
        milestones.append((f"Step {step_num}", os.path.join(agent_dir, fname), step_num))

    final_path = os.path.join(agent_dir, "agent_final.pt")
    if os.path.exists(final_path):
        final_step = step_tuples[-1][0] + 1 if step_tuples else 9999
        milestones.append(("Final Step", final_path, final_step))

    # Check if existing results exist to resume
    summary_path = os.path.join(eval_dir, "milestone_scaling_results.json")
    all_milestone_results = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                all_milestone_results = json.load(f)
        except Exception:
            all_milestone_results = {}

    for label, weights_path, step in milestones:
        step_key = str(step)
        if step_key in all_milestone_results and "grounded_accuracy" in all_milestone_results[step_key]:
            print(f"[*] Skipping already evaluated {label}: Grounded Acc = {all_milestone_results[step_key]['grounded_accuracy']*100:.1f}%")
            continue

        if not os.path.exists(weights_path):
            print(f"[!] Warning: weights {weights_path} not found. Skipping.")
            continue
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        eval_t0 = time.time()
        metrics, outcomes, episodes = evaluate_checkpoint(model, tokenizer, device, worlds, tasks, label)
        eval_elapsed = time.time() - eval_t0
        
        # Extract error rate
        grounded_acc = metrics["overall_grounded_success_rate"]
        error_rate = 1.0 - grounded_acc
        task_success = metrics["overall_task_success_rate"]
        unsupported_rate = metrics["unsupported_claim_rate"]
        fail_dist = metrics["failure_taxonomy_distribution"]

        all_milestone_results[step_key] = {
            "label": label,
            "step": step,
            "approx_samples": step * 10 if step < 1000 else 10155,
            "wall_clock_seconds": round(eval_elapsed, 2),
            "ms_per_task": round((eval_elapsed / max(1, len(tasks))) * 1000.0, 2),
            "grounded_accuracy": grounded_acc,
            "error_rate": error_rate,
            "raw_task_accuracy": task_success,
            "unsupported_claim_rate": unsupported_rate,
            "suite_metrics": metrics["suite_metrics"],
            "failure_taxonomy_distribution": fail_dist
        }

        # Save incremental milestone summary
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_milestone_results, f, indent=2)
        print(f"[+] Saved checkpoint metrics to {summary_path}")

    print("\n" + "=" * 80)
    print("  LCM SAMPLE SIZE SCALING & ERROR RATE PROGRESSION")
    print("=" * 80)
    print(f"{'Milestone':<30} | {'Grounded Acc':<14} | {'Error Rate':<12} | {'Raw Match':<10} | {'Unsupported':<12}")
    print("-" * 80)
    for step, res in all_milestone_results.items():
        print(f"{res['label']:<30} | {res['grounded_accuracy']*100:>11.1f}% | {res['error_rate']*100:>9.1f}% | {res['raw_task_accuracy']*100:>7.1f}% | {res['unsupported_claim_rate']*100:>9.1f}%")
    print("=" * 80)
    print(f"[+] Milestone results saved to {summary_path}")


if __name__ == "__main__":
    main()
