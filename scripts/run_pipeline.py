# scripts/run_pipeline.py
"""High‑level orchestrator for the LCM experimental pipeline with fail-safe checkpointing.

Sequentially runs the five major stages (world generation, tokenizer training,
pretraining, agent SFT, evaluation) and automatically retries failed stages
using exponential back‑off. Progress and retry state are persisted in
`pipeline_state.json` so that the script can be re‑run after any interruption
and resume cleanly where it left off.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.retry import run_with_retry


def _run_stage(cmd: list[str]) -> int:
    """Execute *cmd* via subprocess and stream output live to stdout."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)
        process.stdout.close()
    return process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="LCM Autonomous Pipeline Orchestrator")
    parser.add_argument("--config", type=str, default="configs/scaled_cot_300m.yaml", help="Path to config yaml")
    parser.add_argument("--per-suite", type=int, default=20, help="Number of tasks per suite for eval")
    parser.add_argument("--force", action="store_true", help="Force re-running all stages even if recorded as complete")
    parser.add_argument("--state-file", type=str, default="pipeline_state.json", help="Path to state tracking JSON")
    args = parser.parse_args()

    config_path = args.config
    state_path = Path(args.state_file)

    python_exe = sys.executable

    stages = [
        {
            "name": "world_generation",
            "cmd": [
                python_exe,
                "-m",
                "synth.generate",
                "--config",
                config_path,
            ],
        },
        {
            "name": "tokenizer_training",
            "cmd": [
                python_exe,
                "-m",
                "training.tokenizer",
                "--config",
                config_path,
            ],
        },
        {
            "name": "pretraining",
            "cmd": [
                python_exe,
                "-m",
                "training.pretrain",
                "--config",
                config_path,
            ],
        },
        {
            "name": "agent_sft",
            "cmd": [
                python_exe,
                "-m",
                "training.agent_sft",
                "--config",
                config_path,
            ],
        },
        {
            "name": "evaluation",
            "cmd": [
                python_exe,
                "-m",
                "eval.eval_milestones",
                "--config",
                config_path,
                "--per-suite",
                str(args.per_suite),
            ],
        },
    ]

    print(f"============================================================")
    print(f"  LCM Autonomous Pipeline Runner")
    print(f"  Config: {config_path}")
    print(f"  State:  {state_path}")
    print(f"============================================================")

    for stage in stages:
        print(f"\n[>>>] Starting Stage: {stage['name']}")
        try:
            run_with_retry(
                func=lambda cmd=stage["cmd"]: _run_stage(cmd),
                stage_name=stage["name"],
                state_path=state_path,
                max_retries=5,
                base_delay=5.0,
                backoff_factor=2.0,
                skip_if_complete=not args.force,
            )
            print(f"[+] Stage '{stage['name']}' successfully completed.")
        except RuntimeError as e:
            print(f"[!] FATAL: Stage '{stage['name']}' failed after maximum retries: {e}")
            sys.exit(1)

    print("\n============================================================")
    print("  [+] Complete LCM Pipeline Successfully Finished!")
    print("============================================================")


if __name__ == "__main__":
    main()
