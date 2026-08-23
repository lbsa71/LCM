# AGENTS.md — Development Guidelines & Engineering Principles

This document defines the mandatory engineering standards, development workflows, and architectural rules for all AI agents and contributors working on the **Language & Computation Model (LCM)** codebase.

---

## 1. Core Principles: TDD & The Boy Scout Rule

### A. Strict Test-Driven Development (TDD)
All feature development, bug fixes, protocol modifications, and architectural enhancements must strictly follow the **Red-Green-Refactor** TDD cycle:

1. **Red (Test First)**:
   - Before writing or modifying any implementation code, write a focused unit or integration test in `tests/` that clearly defines the expected behavior, interfaces, edge cases, and protocol invariants.
   - Execute the test suite via `pytest tests/unit/` to verify that the new test fails for the expected reason.

2. **Green (Minimal Implementation)**:
   - Implement the minimal, cleanest solution necessary to make the failing test pass.
   - Verify that all unit and integration tests pass without regression.

3. **Refactor (Clean & Optimize)**:
   - Refactor the code for maximum clarity, modularity, strict typing, and computational efficiency while keeping all tests 100% green.

### B. The Boy Scout Rule
> *"Always leave the campground cleaner than you found it."*

- Every time you modify or inspect a file:
  - Eliminate code smells, dead code, unused imports, and silent exceptions.
  - Improve type annotations, function docstrings, and comments explaining non-obvious design rationales.
  - Expand test coverage for any uncovered edge cases encountered in adjacent code.
  - Never introduce temporary hacks, hardcoded heuristics, or shortcuts without full encapsulation and test verification.

---

## 2. Code Quality & Architectural Standards

1. **Deterministic Protocol Serialization**:
   - All agent communication protocols (`agent/protocol.py`, `agent/shell.py`) must use strict Pydantic schemas with unambiguous type definitions and deterministic JSON validation.
   - Avoid ambiguous token streaming or multi-message turn collisions.

2. **Procedural Grounding vs. Memorization**:
   - The core thesis of LCM is the strict separation of procedural execution from contingent world facts.
   - Training trajectories (`synth/trajectories/`) must model pure procedural steps (in-context entity extraction, AST calculation, pointer dereferencing) and never leak oracle target answers into intermediate reasoning steps.

3. **Performance & Profiling Discipline**:
   - All pretraining, SFT, and evaluation runners must maintain automated wall-clock timers, logging step latency (`ms/step`), token throughput (`tokens/sec`), and per-task inference latency to `training_metrics.json` and `eval_metrics.json`.

4. **Continuous Verification**:
   - Before submitting changes or initiating new training experiments, ensure the entire test suite passes:
     ```powershell
     .\.venv\Scripts\python.exe -m pytest tests/unit/
     ```

---

## 3. Directory Layout & Ownership

- `synth/`: Synthetic world ontology, grammar rendering, task generation, and procedural trajectory synthesis.
- `training/`: Tokenizer, SyntheticTransformer architecture (RoPE, RMSNorm, SwiGLU), causal pretraining, and agent SFT pipelines.
- `agent/`: ReAct execution shell, environment tools (`search`, `read`, `exec`), and protocol validation.
- `eval/`: Benchmark runner, held-out suite evaluation, oracle baselines, failure taxonomy, and milestone scaling analytics.
- `configs/`: Reproducible YAML presets (`smoke.yaml`, `small.yaml`, `small_v2.yaml`, `medium.yaml`).
- `tests/`: Unit and integration test suites enforcing protocol, linter, oracle, and tool invariants.
