"""Evaluation runner executing test suites across unseen worlds with HTML report generation."""

import argparse
import csv
import json
import os
import yaml
import torch
from tokenizers import Tokenizer

from synth.ontology import World, Task, Document, DocumentLine
from training.model import SyntheticTransformer, TransformerConfig
from agent.shell import DeterministicShell
from eval.oracle import OracleSolver
from eval.baselines.majority import MajorityBaseline
from eval.baselines.bow import BagOfWordsBaseline
from eval.baselines.no_tool import NoToolBaseline
from eval.baselines.rule_based import RuleBasedAgent
from eval.metrics import evaluate_episode_outcome, compute_aggregate_metrics


def deserialize_world(data: dict) -> World:
    """Reconstructs World from serialized dict."""
    w = World(world_id=data["world_id"], seed=data["seed"])
    for d_id, d_data in data.get("documents", {}).items():
        lines = [
            DocumentLine(line_number=l["line_number"], text=l["text"], fact_ids=l.get("fact_ids", []))
            for l in d_data.get("lines", [])
        ]
        doc = Document(id=d_data["id"], title=d_data["title"], doc_type=d_data["doc_type"], lines=lines)
        w.documents[d_id] = doc
    return w


def generate_html_report(results: dict, output_path: str):
    """Generates a self-contained HTML evaluation report."""
    agent_summary = results.get("agent_model", {})
    oracle_summary = results.get("oracle", {})
    no_tool_summary = results.get("no_tool", {})
    rule_summary = results.get("rule_based", {})

    suite_rows = ""
    for s_name, s_data in agent_summary.get("suite_metrics", {}).items():
        suite_rows += f"""
        <tr>
            <td style="font-weight:600;">{s_name}</td>
            <td>{s_data.get('total_tasks', 0)}</td>
            <td>{s_data.get('task_success_rate', 0.0) * 100:.1f}%</td>
            <td>{s_data.get('grounded_success_rate', 0.0) * 100:.1f}%</td>
        </tr>
        """

    failure_rows = ""
    for f_cat, f_cnt in agent_summary.get("failure_taxonomy_distribution", {}).items():
        failure_rows += f"<tr><td>{f_cat}</td><td>{f_cnt}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LCM Benchmark Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 2rem; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        h1, h2, h3 {{ color: #38bdf8; }}
        .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
        .card {{ background: #1e293b; padding: 1.5rem; border-radius: 8px; border: 1px solid #334155; }}
        .card h4 {{ margin: 0 0 0.5rem 0; color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; }}
        .card .value {{ font-size: 2rem; font-weight: bold; color: #38bdf8; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; background: #1e293b; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #0f172a; color: #94a3b8; }}
        .badge {{ display: inline-block; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }}
        .badge-pass {{ background: #065f46; color: #34d399; }}
        .badge-fail {{ background: #881337; color: #f87171; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Synthetic-Only Agentic Language Model (LCM) — Benchmark Report</h1>
        <p>Proof-of-concept evaluation demonstrating separation of procedural competence from contingent memorization.</p>

        <div class="card-grid">
            <div class="card">
                <h4>Agent Grounded Success</h4>
                <div class="value">{agent_summary.get('overall_grounded_success_rate', 0.0) * 100:.1f}%</div>
            </div>
            <div class="card">
                <h4>Oracle Success</h4>
                <div class="value">{oracle_summary.get('overall_grounded_success_rate', 0.0) * 100:.1f}%</div>
            </div>
            <div class="card">
                <h4>No-Tool Leakage Rate</h4>
                <div class="value">{no_tool_summary.get('overall_task_success_rate', 0.0) * 100:.1f}%</div>
            </div>
            <div class="card">
                <h4>Unsupported Claim Rate</h4>
                <div class="value">{agent_summary.get('unsupported_claim_rate', 0.0) * 100:.1f}%</div>
            </div>
        </div>

        <h2>Performance by Task Suite</h2>
        <table>
            <thead>
                <tr>
                    <th>Task Suite</th>
                    <th>Tasks</th>
                    <th>Task Success</th>
                    <th>Grounded Success</th>
                </tr>
            </thead>
            <tbody>
                {suite_rows}
            </tbody>
        </table>

        <h2>Failure Taxonomy Analysis</h2>
        <table>
            <thead>
                <tr>
                    <th>Failure Category</th>
                    <th>Count</th>
                </tr>
            </thead>
            <tbody>
                {failure_rows if failure_rows else '<tr><td colspan="2">No failures observed.</td></tr>'}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="LCM Benchmark Evaluation Runner")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml", help="Path to config yaml")
    parser.add_argument("--weights", type=str, default=None, help="Path to specific model weights checkpoint")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output report and results")
    parser.add_argument("--max-tasks", type=int, default=None, help="Maximum number of tasks to evaluate")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    preset_name = config.get("name", "smoke")
    run_dir = f"runs/{preset_name}"
    data_dir = os.path.join(run_dir, "data")
    tok_dir = os.path.join(run_dir, "tokenizer")
    agent_dir = os.path.join(run_dir, "agent_model")
    base_dir = os.path.join(run_dir, "base_model")
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Load Tokenizer
    tok_path = os.path.join(tok_dir, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path) if os.path.exists(tok_path) else None

    # Load Model & Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"[*] Evaluation running on CUDA GPU: {torch.cuda.get_device_name(device)} (TF32 enabled)")
    else:
        print(f"[*] Evaluation running on device: {device}")
    cfg_path = os.path.join(base_dir, "config.json")
    model = None
    if os.path.exists(cfg_path) and tokenizer:
        with open(cfg_path, "r", encoding="utf-8") as f:
            m_dict = json.load(f)
        trans_cfg = TransformerConfig(**m_dict)
        model = SyntheticTransformer(trans_cfg).to(device)
        
        agent_weights = args.weights if args.weights else os.path.join(agent_dir, "agent_final.pt")
        if os.path.exists(agent_weights):
            model.load_state_dict(torch.load(agent_weights, map_location=device))
            model.eval()
            print(f"[*] Loaded trained agent model from {agent_weights}")


    # Load Test Worlds and Tasks
    eval_worlds_path = os.path.join(data_dir, "eval_worlds.json")
    eval_tasks_path = os.path.join(data_dir, "eval_tasks.json")

    with open(eval_worlds_path, "r", encoding="utf-8") as f:
        worlds_raw = json.load(f)
    worlds = {w_id: deserialize_world(w_data) for w_id, w_data in worlds_raw.items()}

    with open(eval_tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if args.max_tasks and args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]

    print(f"[*] Starting benchmark evaluation over {len(tasks)} tasks across {len(worlds)} held-out worlds...")

    # Agents & Baselines
    shell = DeterministicShell(model=model, tokenizer=tokenizer, device=device)
    oracle = OracleSolver()
    no_tool = NoToolBaseline(model=model, tokenizer=tokenizer, device=device)
    rule_agent = RuleBasedAgent()

    agent_episodes = []
    agent_outcomes = []
    oracle_outcomes = []
    no_tool_outcomes = []
    rule_outcomes = []

    for idx, task_data in enumerate(tasks):
        world_id = task_data["world_id"]
        world = worlds[world_id]

        # 1. Evaluate Agent Model
        # Construct lightweight mock task object for shell
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

        t_obj = StructTask(task_data)
        
        ep = shell.run_episode(world, t_obj)
        agent_episodes.append(ep)
        out = evaluate_episode_outcome(ep, task_data)
        agent_outcomes.append(out)

        # 2. Evaluate Oracle Baseline
        orc_ep = oracle.solve(world, t_obj)
        oracle_outcomes.append(evaluate_episode_outcome(orc_ep, task_data))

        # 3. Evaluate No-Tool Baseline
        nt_ep = no_tool.solve(world, t_obj)
        no_tool_outcomes.append(evaluate_episode_outcome(nt_ep, task_data))

        # 4. Evaluate Rule-based Agent
        rb_ep = rule_agent.solve(world, t_obj)
        rule_outcomes.append(evaluate_episode_outcome(rb_ep, task_data))

        if (idx + 1) % 50 == 0 or (idx + 1) == len(tasks):
            curr_acc = sum(1 for o in agent_outcomes if o.get("grounded_success", False)) / (idx + 1)
            print(f"    [{idx + 1}/{len(tasks)}] Agent grounded success so far: {curr_acc * 100:.1f}%")

    # Aggregate summaries
    results = {
        "agent_model": compute_aggregate_metrics(agent_outcomes),
        "oracle": compute_aggregate_metrics(oracle_outcomes),
        "no_tool": compute_aggregate_metrics(no_tool_outcomes),
        "rule_based": compute_aggregate_metrics(rule_outcomes)
    }

    sfx = f"_{args.suffix}" if args.suffix else ""

    # Save results.json
    res_path = os.path.join(eval_dir, f"results{sfx}.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save traces.jsonl
    traces_path = os.path.join(eval_dir, f"traces{sfx}.jsonl")
    with open(traces_path, "w", encoding="utf-8") as f:
        for ep in agent_episodes:
            f.write(json.dumps(ep) + "\n")

    # Save results.csv
    csv_path = os.path.join(eval_dir, f"results{sfx}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "suite", "raw_match", "evidence_valid", "grounded_success", "failure_category"])
        for o in agent_outcomes:
            writer.writerow([o["task_id"], o["suite"], o["raw_match"], o["evidence_valid"], o["grounded_success"], o["failure_category"]])

    # Save HTML report
    html_path = os.path.join(eval_dir, f"report{sfx}.html")
    generate_html_report(results, html_path)


    print(f"[+] Evaluation finished.")
    print(f"    - Agent Grounded Success: {results['agent_model']['overall_grounded_success_rate'] * 100:.1f}%")
    print(f"    - Oracle Grounded Success: {results['oracle']['overall_grounded_success_rate'] * 100:.1f}%")
    print(f"    - No-Tool Leakage Rate: {results['no_tool']['overall_task_success_rate'] * 100:.1f}%")
    print(f"    - Full Report: {html_path}")


if __name__ == "__main__":
    main()
