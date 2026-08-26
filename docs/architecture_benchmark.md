# Architecture Pressure-Test Benchmark

`eval/architecture_benchmark.py` is a reusable experimental benchmark, not a
unit-test suite. It normalizes different architecture boundaries into a common
scorecard and persists every case-level output in
`runs/architecture_benchmark/results.json`.

The active experimental order and stopping rules live in the
[robust-generalization slope roadmap](research_roadmap.md). This document
retains the benchmark contract and completed observations.

## Benchmark contract

The initial suite has 36 deliberately fact-free arithmetic utterances across
four equally sized linguistic stress tracks:

| Track | What it tests |
| --- | --- |
| `seen_style` | The original direct wording family |
| `held_out_style` | Original generator templates held out from the K=8 parser train split |
| `lexical_shift` | New wording such as “make in all” and “outrun” |
| `discourse_distractor` | Nearby words that point to a different operation |

The **shared track** consists of 24 addition/subtraction cases. All three
existing approaches were intended to support direct arithmetic, so it is the
only head-to-head metric. The 12 comparison cases are an **extension track**:
the parser was trained for `COMPARE`, whereas the historic ReAct training data
did not establish that capability. They are reported but not treated as a
fair architecture ranking.

Each architecture receives the following scores:

- `operation_accuracy`: canonical semantic operation selected.
- `protocol_valid_rate`: for ReAct agents only, whether the first response is a
  valid `MATH` tool action.
- `execution_ready_accuracy`: for ReAct agents only, whether the action safely
  evaluates to the requested answer. This is deliberately stricter than
  selecting the correct operation.
- `mean_latency_ms`: local checkpoint inference time per utterance.

The semantic parser emits `OP=ADD`, `OP=SUBTRACT`, or `OP=COMPARE`; it is not
credited or penalized for an executor action. This keeps representation quality
separate from the controller and deterministic execution layers.

## First run — 2026-08-25

| Architecture | Shared operation | Shared protocol | Shared execution-ready | Comparison extension | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Semantic parser K=8, seed 17 | 62.5% | N/A | N/A | 50.0% | 31.2 ms |
| Semantic parser K=8, seed 29 | 58.3% | N/A | N/A | 50.0% | 25.3 ms |
| SmolLM2-135M ReAct | 66.7% | 91.7% | 54.2% | 0.0% | 2143.3 ms |
| Scratch-300M ReAct (legacy) | 0.0% | 0.0% | 0.0% | 0.0% | 1165.2 ms |

The parser average is **60.4%** shared operation accuracy. It is perfect on
the original style but falls to 50.0% on held-out templates and discourse
distractors (and 41.7% on lexical shift across the two seeds). This is a real
generalization gap, not a numerical/operand generalization problem.

SmolLM2 is robust to the new lexical wording (100% shared operation and
execution-ready), but fails all shared discourse-distractor cases. It also
often chooses subtraction correctly while copying an operand incorrectly, so
its 66.7% routing score overstates usable execution (54.2%).

The legacy scratch checkpoint emitted `EMIT` responses rather than a first
`MATH` action. Its zero here is therefore an interface-and-policy failure, not
evidence that a scratch model cannot learn arithmetic. More importantly, its
historical corpus failed current contamination validation and omitted Suites
H/I, so it must remain **diagnostic-only** until rerun using the hardened
corpus gate.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m eval.architecture_benchmark --config configs/architecture_benchmark.yaml
```

The YAML declares checkpoints and makes new architectures additive. Future
tracks should preserve the same separation: routing, protocol, tool-choice,
then full grounded episode success. That lets us compare the two-layer parser
architecture fairly with monolithic ReAct models without pretending their
internal outputs are identical.
