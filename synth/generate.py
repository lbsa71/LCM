"""Top-level CLI for generating synthetic worlds, pretraining corpora, trajectories, and tasks."""

import argparse
import json
import os
import random
import yaml
from typing import Any, Dict, List

from synth.ontology import World, Task
from synth.world import WorldGenerator
from synth.documents.generator import DocumentGenerator
from synth.tasks.generator import TaskGenerator
from synth.trajectories.generator import TrajectoryGenerator
from synth.lint import CorpusLinter
from synth.manifest import generate_manifest


def serialize_world(world: World) -> Dict[str, Any]:
    """Serializes a World object into a JSON-compatible dict."""
    return {
        "world_id": world.world_id,
        "seed": world.seed,
        "entities": {e_id: {"id": e.id, "name": e.name, "type": e.entity_type, "properties": e.properties} for e_id, e in world.entities.items()},
        "facts": {f_id: {"id": f.id, "subject_id": f.subject_id, "relation": f.relation, "value": f.value, "is_contingent": f.is_contingent} for f_id, f in world.facts.items()},
        "documents": {
            d_id: {
                "id": d.id,
                "title": d.title,
                "doc_type": d.doc_type,
                "lines": [{"line_number": l.line_number, "text": l.text, "fact_ids": l.fact_ids} for l in d.lines]
            }
            for d_id, d in world.documents.items()
        }
    }


def serialize_task(task: Task) -> Dict[str, Any]:
    """Serializes a Task object into a JSON-compatible dict."""
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "suite": task.suite,
        "question": task.question,
        "gold_answer": task.gold_answer,
        "world_id": task.world_id,
        "is_retrieval_required": task.is_retrieval_required,
        "is_contingent": task.is_contingent,
        "is_insufficient_evidence": task.is_insufficient_evidence,
        "context_text": task.context_text,
        "required_evidence": task.proof_graph.required_document_lines,
        "metadata": task.metadata
    }


def main():
    parser = argparse.ArgumentParser(description="Synthetic Corpus & World Generator")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml", help="Path to config yaml")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory override")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    preset_name = config.get("name", "smoke")
    base_seed = config.get("seed", 42)
    output_dir = args.output_dir or f"runs/{preset_name}"
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    corpus_cfg = config.get("corpus", {})
    train_worlds_cnt = corpus_cfg.get("train_worlds", 20)
    val_worlds_cnt = corpus_cfg.get("val_worlds", 5)
    test_worlds_cnt = corpus_cfg.get("test_worlds", 10)

    world_gen = WorldGenerator(base_seed=base_seed)
    train_doc_gen = DocumentGenerator(template_set="train")
    eval_doc_gen = DocumentGenerator(template_set="eval")
    train_task_gen = TaskGenerator(template_set="train")
    eval_task_gen = TaskGenerator(template_set="eval")
    traj_gen = TrajectoryGenerator()

    # Disjoint seed sets
    train_seeds = set(range(base_seed + 1, base_seed + 1 + train_worlds_cnt))
    val_seeds = set(range(base_seed + 100000, base_seed + 100000 + val_worlds_cnt))
    test_seeds = set(range(base_seed + 200000, base_seed + 200000 + test_worlds_cnt))

    pretrain_train_lines = []
    pretrain_val_lines = []
    agent_sft_trajectories = []
    eval_worlds_dict = {}
    eval_tasks_list = []

    print(f"[*] Generating {train_worlds_cnt} train worlds...")
    for seed in train_seeds:
        world_id = f"w_tr_{seed}"
        w = world_gen.generate_world(world_id, seed)
        train_doc_gen.generate_documents(w, docs_per_world=corpus_cfg.get("docs_per_world", 10))
        
        # Pretraining text from documents
        for doc in w.documents.values():
            for line in doc.lines:
                pretrain_train_lines.append(line.text)

        # Generate tasks and SFT trajectories
        rng = random.Random(seed)
        tasks = train_task_gen.generate_all_tasks(w, rng)
        for task in tasks:
            # Trajectory for SFT
            traj = traj_gen.generate_trajectory_for_task(w, task, rng)
            agent_sft_trajectories.append(traj)

        # Procedural in-context induction sequences for pretraining
        for ent in w.entities.values():
            pretrain_train_lines.append(f"Query regarding entity {ent.name}: search(query='{ent.name}') yields document records for {ent.name}.")
            pretrain_train_lines.append(f"Extract subject from question: 'What is the recorded status of {ent.name}?' -> query='{ent.name}'.")

        for i in range(10):
            n1 = rng.randint(100, 999)
            n2 = rng.randint(100, 999)
            pretrain_train_lines.append(f"In-context arithmetic: items in Record A = {n1}, items in Record B = {n2}. Expression: exec(code='{n1} + {n2}') -> {n1 + n2}.")

    print(f"[*] Generating {val_worlds_cnt} validation worlds...")
    for seed in val_seeds:
        world_id = f"w_val_{seed}"
        w = world_gen.generate_world(world_id, seed)
        train_doc_gen.generate_documents(w)
        for doc in w.documents.values():
            for line in doc.lines:
                pretrain_val_lines.append(line.text)

    print(f"[*] Generating {test_worlds_cnt} evaluation test worlds...")
    for seed in test_seeds:
        world_id = f"w_te_{seed}"
        w = world_gen.generate_world(world_id, seed, held_out_lexicon=True)
        eval_doc_gen.generate_documents(w, docs_per_world=corpus_cfg.get("docs_per_world", 10))
        eval_worlds_dict[world_id] = serialize_world(w)

        rng = random.Random(seed)
        tasks = eval_task_gen.generate_all_tasks(w, rng)
        for task in tasks:
            eval_tasks_list.append(serialize_task(task))

    # Write files
    pretrain_tr_path = os.path.join(data_dir, "pretrain_train.txt")
    with open(pretrain_tr_path, "w", encoding="utf-8") as f:
        f.write("\n".join(pretrain_train_lines) + "\n")

    pretrain_val_path = os.path.join(data_dir, "pretrain_val.txt")
    with open(pretrain_val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(pretrain_val_lines) + "\n")

    sft_path = os.path.join(data_dir, "agent_sft_train.jsonl")
    with open(sft_path, "w", encoding="utf-8") as f:
        for traj in agent_sft_trajectories:
            f.write(json.dumps(traj) + "\n")

    eval_worlds_path = os.path.join(data_dir, "eval_worlds.json")
    with open(eval_worlds_path, "w", encoding="utf-8") as f:
        json.dump(eval_worlds_dict, f, indent=2)

    eval_tasks_path = os.path.join(data_dir, "eval_tasks.json")
    with open(eval_tasks_path, "w", encoding="utf-8") as f:
        json.dump(eval_tasks_list, f, indent=2)

    # Linting
    linter = CorpusLinter()
    # Exclude intentional closed-book memorization probe questions from corpus leakage checks
    test_texts = [
        t["question"] for t in eval_tasks_list
        if t.get("suite") != "anti_memorization_closed_book"
    ]
    lint_report = linter.lint_dataset(
        train_texts=pretrain_train_lines,
        val_texts=pretrain_val_lines,
        test_texts=test_texts,
        train_seeds=train_seeds,
        test_seeds=test_seeds
    )
    print(f"[*] Lint status: {lint_report['status']}")
    if lint_report["errors"]:
        print(f"[!] Lint errors: {lint_report['errors']}")

    # Manifest
    stats = {
        "pretrain_train_lines": len(pretrain_train_lines),
        "pretrain_val_lines": len(pretrain_val_lines),
        "sft_trajectories": len(agent_sft_trajectories),
        "eval_worlds": len(eval_worlds_dict),
        "eval_tasks": len(eval_tasks_list),
        "lint_report": lint_report
    }
    generate_manifest(output_dir, config, stats, lint_report["status"])
    print(f"[+] Generation complete. Data saved to {data_dir}")


if __name__ == "__main__":
    main()
