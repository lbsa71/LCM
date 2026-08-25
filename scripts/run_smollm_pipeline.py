"""Autonomous pipeline runner for SmolLM2 base model SFT and milestone evaluation."""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_stage(cmd: list[str], description: str) -> None:
    print(f"\n============================================================")
    print(f"  [>>>] {description}")
    print(f"============================================================")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    if proc.stdout:
        for line in iter(proc.stdout.readline, ""):
            print(line, end="", flush=True)
        proc.stdout.close()
    ret = proc.wait()
    if ret != 0:
        print(f"[!] ERROR: Command {cmd} failed with exit code {ret}")
        sys.exit(ret)
    print(f"[+] Successfully finished: {description}")


def main():
    config_path = "configs/smollm2_135m_agent.yaml"
    python_exe = sys.executable

    # 1. World & Task Generation
    run_stage(
        [python_exe, "-m", "synth.generate", "--config", config_path],
        "Stage 1: Multi-Domain World & Task Generation (Suites A-H + Anti-Memorization)"
    )

    # 2. Agent SFT on SmolLM2-135M Base
    run_stage(
        [python_exe, "-m", "training.agent_sft", "--config", config_path],
        "Stage 2: SmolLM2-135M Base Agent SFT (3,000 steps)"
    )

    # 3. Milestone Benchmark Evaluation
    run_stage(
        [python_exe, "-m", "eval.eval_milestones", "--config", config_path, "--per-suite", "10", "--final-only"],
        "Stage 3: Milestone Benchmark Evaluation (All Suites A-H)"
    )

    print("\n============================================================")
    print("  [+] SmolLM2 Pipeline Complete!")
    print("============================================================")


if __name__ == "__main__":
    main()
